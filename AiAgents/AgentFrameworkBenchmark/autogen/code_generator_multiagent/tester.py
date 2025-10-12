import json
import logging
import subprocess
import sys
from pathlib import Path
from state import TestResults, TestFailure, track_usage_from_result, State
from session_manager import get_global_session_manager
from prompts import TESTER_PROMPT_TEMPLATE
import json
import logging
from autogen_core import (
    MessageContext,
    RoutedAgent,
    TopicId,
    message_handler,
    type_subscription,
    CancellationToken,
)
from autogen_core.models import AssistantMessage
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_ext.tools.code_execution import PythonCodeExecutionTool
from openai import OpenAI

logger = logging.getLogger(__name__)



@type_subscription(topic_type="TestingRoom")
class TesterAgent(RoutedAgent):
    def __init__(self, model_client: OpenAI) -> None:
        super().__init__("A testing agent.")
        self._model_client = model_client
        self.base_dir = Path.cwd()

    @message_handler
    async def write_tests(self, message: State, ctx: MessageContext) -> None:
        """Generate test code using LLM and create test_main.py file"""
    
        code_to_test = ""
        if (hasattr(message, 'code_gen_state') and 
            message.code_gen_state and 
            message.code_gen_state.code_result and 
            message.code_gen_state.code_result.code):
            code_to_test = message.code_gen_state.code_result.code
        else:
            main_file = self.base_dir / message.code_gen_state.package_name / "main.py"
            if main_file.exists():
                code_to_test = main_file.read_text(encoding='utf-8')
        
        if not code_to_test:
            logger.warning("No code found to test")
            return "No code available for testing"
        
        prompt = TESTER_PROMPT_TEMPLATE.format(
            code_to_test=code_to_test,
            max_tests=message.code_gen_state.max_tests
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
                params = {
                    "model": message.model_name,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ]
                }
                response = self._model_client.chat.completions.create(**params)
                content = response.choices[0].message.content
                
                if message:
                    track_usage_from_result(response, message)
                
                if content and content.strip():
                    content = content.strip()
                    logger.info(f"Received test response: {content[:100]}..." if len(content) > 100 else f"Received test response: {content}")
                    
                    json_content = self._extract_json_from_markdown(content)
                    
                    try:
                        # Try parsing as JSON first
                        response_json = json.loads(json_content)
                        test_code = response_json.get("test_code", "")
                        explanation = response_json.get("explanation", "Tests generated successfully")
                        
                        if test_code.strip():  # Check if we got valid test code
                            logger.info("Successfully parsed JSON test response")
                            break
                            
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON parsing failed: {e}")
                        logger.info(f"Raw content causing JSON error: '{json_content[:200]}'")
                        # If JSON parsing fails, treat the entire response as test code
                        logger.info("Treating response as raw test code since JSON parsing failed")
                        test_code = content
                        explanation = "Tests generated (raw format)"
                        
                        if test_code.strip():
                            logger.info("Raw test code accepted")
                            break
                else:
                    logger.warning(f"Empty or None content from response: {content}")
                    continue  # Try again
                    
            except Exception as e:
                logger.error(f"Test generation failed: {e}")
        
        if not test_code.strip():
            error_msg = f"Failed to generate tests after {max_retries} attempts"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        self.compile_tests(test_code)

        message.code_gen_state.test_results = TestResults(
            test_explanation=explanation
        )
        await self.run_tests(message)

    def _extract_json_from_markdown(self, content: str) -> str:
        """Extract JSON from markdown code blocks if present"""
        if content.startswith("```json") and content.endswith("```"):
            lines = content.split('\n')
            if len(lines) >= 3:
                json_lines = lines[1:-1]
                return '\n'.join(json_lines)
        return content


    def compile_tests(self, test_code: str) -> None:
        """Create or update test_main.py in the app folder with the generated test code"""
        
        app_dir = self.base_dir / "app"
        app_dir.mkdir(parents=True, exist_ok=True)
        test_file = app_dir / "test_main.py"
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_code)

    async def _run_tests_with_docker(self, app_dir: Path) -> tuple[str, str, int]:
        """Run tests using Docker-based code execution"""
        
        # Read the test file content
        test_file = app_dir / "test_main.py"
        if not test_file.exists():
            raise FileNotFoundError("test_main.py not found")
            
        test_code = test_file.read_text(encoding='utf-8')
        
        # Prepare executor and tool
        executor = DockerCommandLineCodeExecutor(
            image="python:3.12-slim",
            timeout=30,
            work_dir=str(app_dir)
        )
        await executor.start()
        
        try:
            code_tool = PythonCodeExecutionTool(executor)
            cancellation_token = CancellationToken()
            
            # Execute the actual test code from the file
            tool_result = await code_tool.run_json({"code": test_code}, cancellation_token)
            
            # Convert result to string or structured form
            output = code_tool.return_value_as_string(tool_result)
            
            # Get exit code from the result object properly
            exit_code = 1  # Default to failure
            if hasattr(tool_result, 'exit_code'):
                exit_code = tool_result.exit_code
            elif hasattr(tool_result, 'returncode'):
                exit_code = tool_result.returncode
            elif isinstance(tool_result, dict):
                exit_code = tool_result.get("exit_code", tool_result.get("returncode", 1))
            
            # For this implementation, we'll treat stdout and stderr as combined output
            return output, "", exit_code
            
        finally:
            # Cleanup
            await executor.stop()

    async def run_tests(self, message: State):
        """Run the tests and return results"""
        
        test_file = self.base_dir / "app" / "test_main.py"
        
        if not test_file.exists():
            logger.warning("test_main.py not found")
            message.code_gen_state.test_results = TestResults(passed=0, failed=1, failures=[
                TestFailure(name="test_file_missing", trace="test_main.py file not found")
            ])

        else:
            try:
                logger.info("Running tests...")
                app_dir = self.base_dir / "app"
                
                # Use Docker-based execution instead of subprocess
                stdout, stderr, returncode = await self._run_tests_with_docker(app_dir)
                
                existing_explanation = ""
                if message.code_gen_state.test_results and message.code_gen_state.test_results.test_explanation:
                    existing_explanation = message.code_gen_state.test_results.test_explanation
                
                test_results = self._parse_test_output(stdout, stderr, returncode)
                test_results.test_explanation = existing_explanation
                message.code_gen_state.test_results = test_results
                
            except Exception as e:
                error_msg = str(e)
                if "timeout" in error_msg.lower():
                    logger.warning("Tests timed out after 30 seconds")
                    message.code_gen_state.test_results = TestResults(passed=0, failed=1, failures=[
                        TestFailure(name="test_timeout", trace="Tests timed out after 30 seconds")
                    ])
                else:
                    logger.error(f"Error running tests: {e}")
                    message.code_gen_state.test_results = TestResults(passed=0, failed=1, failures=[
                        TestFailure(name="test_execution_error", trace=str(e))
                    ])

        session_manager = get_global_session_manager(message.session_id) if message.session_id else None
        if session_manager and message.code_gen_state and message.code_gen_state.test_results:
            test_results = message.code_gen_state.test_results
            assistant_message = AssistantMessage(
                content=f"TesterAgent (Iteration {message.code_gen_state.iteration}): Test results - {test_results.passed} passed, {test_results.failed} failed. {test_results.test_explanation}",
                source="TesterAgent"
            )
            session_manager.add_session_message(assistant_message)

        await self.publish_message(message, topic_id=TopicId("ReviewingRoom", source=self.id.key))


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
