"""
Tester Agent using openai-agents library pattern
"""
import logging
import subprocess
import sys
from pathlib import Path
from pydantic import BaseModel
from agents import Agent, RunContextWrapper
from context import MainContext
from .types import TestResults, TestFailure
from prompts import TESTER_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


class TestCodeOutput(BaseModel):
    """Output structure for the Tester agent when generating test code"""
    test_code: str
    explanation: str
    test_count: int


def get_tester_instructions(
    wrapper: RunContextWrapper[MainContext], 
    agent: Agent[MainContext]
) -> str:
    """
    Dynamic instructions for the Tester agent based on current context.
    Now works with MainContext and accesses code_gen_context.
    """
    main_context = wrapper.context
    
    # Access the code generation context
    if not main_context.code_gen_context:
        logger.error("No code_gen_context found in MainContext!")
        return "Error: No code generation context available."
    
    context = main_context.code_gen_context
    
    # Get the current code to test
    code_to_test = ""
    if context.current_code:
        code_to_test = context.current_code.code
    elif context.previous_code:
        code_to_test = context.previous_code
    else:
        main_file = Path.cwd() / context.package_name / "main.py"
        if main_file.exists():
            code_to_test = main_file.read_text(encoding='utf-8')
    
    formatted_prompt = TESTER_PROMPT_TEMPLATE.format(
        code_to_test=code_to_test,
        max_test=context.max_tests
    )
    
    return formatted_prompt


def compile_tests(test_code: str, package_name: str) -> None:
    """Save the generated test code to test_main.py"""
    base_dir = Path.cwd()
    app_dir = base_dir / package_name
    
    # Ensure app directory exists
    app_dir.mkdir(parents=True, exist_ok=True)
    
    # Write the test code
    test_file = app_dir / "test_main.py"
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(test_code)
    
    logger.info(f"Tests compiled to: {test_file}")


def run_tests(package_name: str) -> TestResults:
    """Execute the tests and parse results"""
    base_dir = Path.cwd()
    test_file = base_dir / package_name / "test_main.py"
    
    if not test_file.exists():
        logger.warning("test_main.py not found")
        return TestResults(
            passed=0,
            failed=1,
            failures=[TestFailure(
                name="test_file_missing",
                trace="test_main.py file not found"
            )]
        )
    
    try:
        logger.info("Running tests...")
        app_dir = base_dir / package_name
        
        result = subprocess.run(
            [sys.executable, '-m', 'unittest', 'test_main'],
            capture_output=True,
            text=True,
            cwd=str(app_dir),
            timeout=30
        )
        
        return parse_test_output(result.stdout, result.stderr, result.returncode)
        
    except subprocess.TimeoutExpired:
        logger.warning("Tests timed out after 30 seconds")
        return TestResults(
            passed=0,
            failed=1,
            failures=[TestFailure(
                name="test_timeout",
                trace="Tests timed out after 30 seconds"
            )]
        )
    except Exception as e:
        logger.error(f"Error running tests: {e}")
        return TestResults(
            passed=0,
            failed=1,
            failures=[TestFailure(
                name="test_execution_error",
                trace=str(e)
            )]
        )


def parse_test_output(stdout: str, stderr: str, returncode: int) -> TestResults:
    """Parse unittest output to extract test results"""
    failures = []
    passed = 0
    failed = 0
    
    output = stdout + stderr
    
    if returncode == 0:
        # All tests passed
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
        # Parse failures
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
                
                current_failure = line.split(':', 1)[1].strip() if ':' in line else line
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
    
    # Handle edge cases
    if passed == 0 and failed == 0:
        if returncode == 0:
            passed = 1
        else:
            failed = 1
            if not failures:
                failures.append(TestFailure(
                    name="unknown_test_failure",
                    trace=output[:500]
                ))
    
    return TestResults(passed=passed, failed=failed, failures=failures)


def create_tester(context: MainContext) -> Agent[MainContext]:
    """
    Create and configure the Tester agent.
    
    This agent is responsible for generating comprehensive test suites
    for the code produced by the Coder agent.
    Now configured to work with MainContext and uses configurable model.
    """
    agent = Agent[MainContext](
        name="Tester Agent",
        instructions=get_tester_instructions,  # Dynamic instructions
        model=context.model_name,
        output_type=TestCodeOutput,  # Structured output for test code
        tools=[],  # No tools needed for now
    )
    
    return agent


def post_process_test_output(
    main_context: MainContext,
    test_output: TestCodeOutput
) -> TestResults:
    """
    Post-process the test output from the agent.
    This saves the test code, runs the tests, and updates the context.
    Now works with MainContext.
    """
    if not main_context.code_gen_context:
        logger.error("No code_gen_context found in MainContext!")
        return TestResults(passed=0, failed=1, failures=[])
    
    context = main_context.code_gen_context
    
    # Compile the test code to file
    compile_tests(test_output.test_code, context.package_name)
    
    # Run the tests and get results
    test_results = run_tests(context.package_name)
    
    # Preserve the explanation from the agent
    test_results.test_explanation = test_output.explanation
    
    # Update context with test results
    context.test_results = test_results
    
    logger.info(f"Tests generated: {test_output.test_count} tests")
    logger.info(f"Test results: {test_results.passed} passed, {test_results.failed} failed")
    
    return test_results