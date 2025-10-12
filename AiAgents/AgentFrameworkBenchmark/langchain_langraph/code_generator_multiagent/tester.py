import json
import logging
import subprocess
import sys
from pathlib import Path
from state import TestResults, TestFailure
from prompts import TESTER_PROMPT_TEMPLATE
from langchain_core.prompts import PromptTemplate
from langchain_community.callbacks import get_openai_callback

logger = logging.getLogger(__name__)


class TesterAgent:
    def __init__(self, package_name: str = "app", llm_client=None, max_test: int = 15) -> None:
        self.package_name = package_name
        self.llm_client = llm_client
        self.max_test = max_test
        self.base_dir = Path.cwd()
        self.prompt_template = PromptTemplate.from_template(TESTER_PROMPT_TEMPLATE)

    def write_tests(self, state) -> dict:
        """Generate test code using LLM and create test_main.py file
        
        Returns:
            dict: Token usage information from the LLM calls
        """
    
        code_to_test = ""
        if hasattr(state, 'code_gen_state') and state.code_gen_state and state.code_gen_state.code_result and state.code_gen_state.code_result.code:
            code_to_test = state.code_gen_state.code_result.code
        else:
            main_file = self.base_dir / self.package_name / "main.py"
            if main_file.exists():
                code_to_test = main_file.read_text(encoding='utf-8')
        
        if not code_to_test:
            logger.warning("No code found to test")
            return {"requests": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        
        # Format the prompt using PromptTemplate
        formatted_prompt = self.prompt_template.format(
            code_to_test=code_to_test,
            max_test=self.max_test
        )
        
        test_code = ""
        explanation = ""
        max_retries = 3
        total_usage = {"requests": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        
        with get_openai_callback() as cb:
            for attempt in range(1, max_retries + 1):
                if attempt == 1:
                    logger.info("Generating tests...")
                else:
                    logger.info(f"Retry attempt {attempt}/{max_retries}...")
                
                try:
                    response = self.llm_client.invoke(formatted_prompt)
                    llm_response_str = response.content if hasattr(response, 'content') else str(response)
                    
                    if llm_response_str:
                        try:
                            response_json = json.loads(llm_response_str)
                            test_code = response_json.get("test_code", "")
                            explanation = response_json.get("explanation", "Tests generated successfully")
                            
                            if test_code.strip():  # Check if we got valid test code
                                logger.info("Tests generated successfully")
                                break
                                
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON parsing failed: {e}")
                        
                except Exception as e:
                    logger.error(f"Test generation failed: {e}")
            
            # Extract token usage from callback handler
            total_usage = {
                "requests": cb.successful_requests,
                "input_tokens": cb.prompt_tokens,
                "output_tokens": cb.completion_tokens,
                "total_tokens": cb.total_tokens
            }
            
            # Log token usage for this tester interaction
            if cb.successful_requests > 0:
                logger.info(
                    f"🔢 TesterAgent LLM calls - Input: {cb.prompt_tokens}, "
                    f"Output: {cb.completion_tokens}, Total: {cb.total_tokens} tokens "
                    f"({cb.successful_requests} requests) - Cost: ${cb.total_cost:.4f}"
                )
        
        if not test_code.strip():
            error_msg = f"Failed to generate tests after {max_retries} attempts"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Write test file
        self.compile_tests(test_code)

        state.code_gen_state.test_results = TestResults(
            test_explanation=explanation
        )
        return total_usage


    def compile_tests(self, test_code: str) -> None:
        """Create or update test_main.py in the app folder with the generated test code"""
        
        app_dir = self.base_dir / self.package_name
        app_dir.mkdir(parents=True, exist_ok=True)
        test_file = app_dir / "test_main.py"
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_code)

    def run_tests(self, state):
        """Run the tests and return results"""
        
        test_file = self.base_dir / self.package_name / "test_main.py"
        
        if not test_file.exists():
            logger.warning("test_main.py not found")
            return TestResults(passed=0, failed=1, failures=[
                TestFailure(name="test_file_missing", trace="test_main.py file not found")
            ])
        
        try:
            logger.info("Running tests...")
            app_dir = self.base_dir / self.package_name
            result = subprocess.run([
                sys.executable, '-m', 'unittest', 'test_main'
            ], capture_output=True, text=True, cwd=str(app_dir), timeout=30)
            
            existing_explanation = ""
            if state.codeee_gen_state.test_results and state.code_gen_state.test_results.test_explanation:
                existing_explanation = state.code_gen_state.test_results.test_explanation
            
            test_results = self._parse_test_output(result.stdout, result.stderr, result.returncode)
            test_results.test_explanation = existing_explanation
            state.code_gen_state.test_results = test_results
            
        except subprocess.TimeoutExpired:
            logger.warning("Tests timed out after 30 seconds")
            state.code_gen_state.test_results = TestResults(passed=0, failed=1, failures=[
                TestFailure(name="test_timeout", trace="Tests timed out after 30 seconds")
            ])
        except Exception as e:
            logger.error(f"Error running tests: {e}")
            state.code_gen_state.test_results = TestResults(passed=0, failed=1, failures=[
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
