
import logging
import subprocess
import sys
from pathlib import Path
from state import TestResults, TestFailure, update_usage_from_crewai, get_global_state
from prompts import TESTER_PROMPT_TEMPLATE
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class TestGenerationResponse(BaseModel):
    test_code: str
    explanation: str


class TesterAgent:
    def __init__(self, state, agent) -> None:
        self.state = state
        self.agent = agent
        # Set base_dir to the crewai directory, not project root
        self.base_dir = Path(__file__).parent.parent

    def write_tests(self) -> None:
        """Generate test code using Agent and create test_main.py file"""
    
        code_to_test = ""
        if hasattr(self.state, 'code_gen_state') and self.state.code_gen_state and self.state.code_gen_state.code_result and self.state.code_gen_state.code_result.code:
            code_to_test = self.state.code_gen_state.code_result.code
        else:
            main_file = self.base_dir / self.state.code_gen_state.package_name / "main.py"
            if main_file.exists():
                code_to_test = main_file.read_text(encoding='utf-8')
        
        if not code_to_test:
            logger.warning("No code found to test")
            return "No code available for testing"
        
        prompt = TESTER_PROMPT_TEMPLATE.format(
            code_to_test=code_to_test,
            max_tests=self.state.code_gen_state.max_tests
        )
        
        test_code = ""
        explanation = ""
        max_retries = 3
        
        for attempt in range(1, max_retries + 1):
            if attempt == 1:
                logger.info("Generating tests...")
            else:
                logger.info(f"Retry attempt {attempt}/{max_retries}...")
            
            try:
                result = self.agent.kickoff(prompt, response_format=TestGenerationResponse)
                
                usage_data = None
                if result and hasattr(result, 'usage_metrics'):
                    usage_data = result.usage_metrics
                
                if usage_data:
                    global_state = get_global_state()
                    if global_state:
                        logger.info(f"Updating tester usage with: {usage_data}")
                        update_usage_from_crewai(global_state, usage_data)
                
                if result and hasattr(result, 'pydantic'):
                    response = result.pydantic
                    test_code = response.test_code
                    explanation = response.explanation or "Tests generated successfully"
                    
                    if test_code.strip():  
                        logger.info("Tests generated successfully")
                        break
                    
            except Exception as e:
                logger.error(f"Test generation failed: {e}")
        
        if not test_code.strip():
            error_msg = f"Failed to generate tests after {max_retries} attempts"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Write test file
        self.compile_tests(test_code)

        self.state.code_gen_state.test_results = TestResults(
            test_explanation=explanation
        )
        return


    def compile_tests(self, test_code: str) -> None:
        """Create or update test_main.py in the app folder with the generated test code"""
        
        app_dir = self.base_dir / self.state.code_gen_state.package_name
        app_dir.mkdir(parents=True, exist_ok=True)
        test_file = app_dir / "test_main.py"
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_code)

    def run_tests(self):
        """Run the tests and return results"""
        
        test_file = self.base_dir / self.state.code_gen_state.package_name / "test_main.py"
        
        if not test_file.exists():
            logger.warning("test_main.py not found")
            return TestResults(passed=0, failed=1, failures=[
                TestFailure(name="test_file_missing", trace="test_main.py file not found")
            ])
        
        try:
            logger.info("Running tests...")
            app_dir = self.base_dir / self.state.code_gen_state.package_name
            logger.info(f"Test directory: {app_dir}")
            logger.info(f"Running command in: {app_dir}")
            
            result = subprocess.run([
                sys.executable, '-m', 'unittest', 'test_main'
            ], capture_output=True, text=True, cwd=str(app_dir), timeout=60)  # Increased timeout to 60s
            
            existing_explanation = ""
            if self.state.code_gen_state.test_results and self.state.code_gen_state.test_results.test_explanation:
                existing_explanation = self.state.code_gen_state.test_results.test_explanation
            
            test_results = self._parse_test_output(result.stdout, result.stderr, result.returncode)
            test_results.test_explanation = existing_explanation
            self.state.code_gen_state.test_results = test_results
            return 'tests_processed'
            
        except subprocess.TimeoutExpired as e:
            logger.warning(f"Tests timed out after 60 seconds in directory: {app_dir}")
            logger.warning(f"Command that timed out: {e.cmd}")
            # Try to log any partial output
            if hasattr(e, 'stdout') and e.stdout:
                logger.warning(f"Partial stdout: {e.stdout[:500]}")
            if hasattr(e, 'stderr') and e.stderr:
                logger.warning(f"Partial stderr: {e.stderr[:500]}")
            self.state.code_gen_state.test_results = TestResults(passed=0, failed=1, failures=[
                TestFailure(name="test_timeout", trace=f"Tests timed out after 60 seconds. Command: {e.cmd}")
            ])
        except Exception as e:
            logger.error(f"Error running tests: {e}")
            self.state.code_gen_state.test_results = TestResults(passed=0, failed=1, failures=[
                TestFailure(name="test_execution_error", trace=str(e))
            ])


    def _parse_test_output(self, stdout: str, stderr: str, returncode: int) -> TestResults:
        """Parse unittest output to extract test results"""
        
        failures = []
        passed = 0
        failed = 0
        
        output = stdout + stderr
        
        if returncode == 0:
            lines = output.split('\n')
            for line in lines:
                if 'Ran' in line and 'test' in line:
                    try:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part == 'Ran':
                                passed = int(parts[i + 1])
                                break
                    except (ValueError, IndexError):
                        passed = 1  
        else:
            lines = output.split('\n')
            current_failure = None
            collecting_trace = False
            trace_lines = []
            
            for line in lines:
                if line.startswith('FAIL:') or line.startswith('ERROR:'):
                    if current_failure:
                        failures.append(TestFailure(
                            name=current_failure,
                            trace='\n'.join(trace_lines)
                        ))
                    
                    current_failure = line.split(':', 1)[1].strip()
                    trace_lines = []
                    collecting_trace = True
                    failed += 1
                    
                elif collecting_trace and line.startswith('------'):
                    collecting_trace = False
                    
                elif collecting_trace:
                    trace_lines.append(line)
                
                elif 'Ran' in line and 'test' in line:
                    try:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part == 'Ran':
                                total_tests = int(parts[i + 1])
                                passed = total_tests - failed
                                break
                    except (ValueError, IndexError):
                        pass
            
            if current_failure:
                failures.append(TestFailure(
                    name=current_failure,
                    trace='\n'.join(trace_lines)
                ))
        
        if passed == 0 and failed == 0:
            if returncode == 0:
                passed = 1 
            else:
                failed = 1  
                if not failures:
                    failures.append(TestFailure(
                        name="unknown_test_failure",
                        trace=output
                    ))
        
        return TestResults(passed=passed, failed=failed, failures=failures)
