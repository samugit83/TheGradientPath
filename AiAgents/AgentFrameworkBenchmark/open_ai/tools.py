"""
Tools module for the chat system using openai-agents function_tool pattern
"""

import logging
import shutil
from pathlib import Path
from typing import Any, List, Optional
from agents import function_tool, RunContextWrapper
from context import MainContext, CodeGenContext

logger = logging.getLogger(__name__)


# ============================================================================
# Mathematical Tools
# ============================================================================

@function_tool
async def sum_numbers(wrapper: RunContextWrapper, a: float, b: float) -> str:
    """
    Adds two numbers together and returns the result.
    
    Args:
        a: First number to add
        b: Second number to add
    """
    try:
        result = a + b
        return f"The sum of {a} and {b} is {result}"
    except Exception as e:
        logger.error(f"Error in sum calculation: {e}")
        return f"Error calculating sum: {str(e)}"


@function_tool
async def multiply_numbers(wrapper: RunContextWrapper, a: float, b: float) -> str:
    """
    Multiplies two numbers and returns the product.
    
    Args:
        a: First number to multiply
        b: Second number to multiply
    """
    try:
        result = a * b
        return f"The product of {a} and {b} is {result}"
    except Exception as e:
        logger.error(f"Error in multiplication: {e}")
        return f"Error calculating product: {str(e)}"


# ============================================================================
# Code Generation Tool
# ============================================================================

@function_tool
async def generate_code(wrapper: RunContextWrapper, prompt: str) -> str:
    """
    Generates complete code implementations with tests using a multi-agent system.
    This tool creates production-ready Python code with proper structure and comprehensive test coverage.
    
    Args:
        prompt: The main requirements or description of what code to generate
    """
    try:
        from code_generator_agents.code_orchestrator_agent import run_code_orchestrator

        package_name = "app"
        
        # Empty the app folder per user preference [[memory:5986796]]
        app_dir = Path.cwd() / package_name
        if app_dir.exists():
            logger.info("Emptying app folder before code generation.")
            shutil.rmtree(app_dir)
        
        # Get context from wrapper
        wrapper_context = getattr(wrapper, 'context', None)
        # Run the multi-agent code generation
        logger.info("Starting multi-agent code generation...")

        constraints = {
            "language": "Python 3.8+",
            "style": "PEP 8 compliant",
            "documentation": "Add docstrings",
            "allowed_packages": "stdlib"
        }
        
        # Set the user prompt on the code generation context
        if not wrapper_context.code_gen_context:
            wrapper_context.code_gen_context = CodeGenContext(
                user_prompt_for_app=prompt,
                constraints=constraints,
                package_name=package_name,
                max_iterations=3,
                max_tests=8
            )
        else:
            wrapper_context.code_gen_context.user_prompt_for_app = prompt
        
        # Extract session from context to pass to orchestrator
        session = getattr(wrapper_context, 'session', None)
        
        # The orchestrator will update the context with usage info
        final_context = await run_code_orchestrator(
            main_context=wrapper_context,
            max_iterations=3,
            package_name=package_name,
            max_tests=8,
            constraints=constraints,
            session=session
        )
        
        wrapper_context.update_usage({
            "requests": final_context.requests,
            "input_tokens": final_context.input_tokens,
            "output_tokens": final_context.output_tokens,
            "total_tokens": final_context.total_tokens
        })
        
        if final_context.code_gen_context and final_context.code_gen_context.process_completed:
            test_info = ""
            if final_context.code_gen_context.test_results:
                test_results = final_context.code_gen_context.test_results
                test_info = f" Tests: {test_results.passed} passed, {test_results.failed} failed."
            
            return (
                f"✅ Code generated successfully in {final_context.code_gen_context.iteration} iteration(s)!{test_info}\n"
                f"Check the '{package_name}' folder for the generated code.\n"
                f"Files created: main.py and test_main.py"
            )
        else:
            iterations = final_context.code_gen_context.iteration if final_context.code_gen_context else "unknown"
            
            return (
                f"⚠️ Code generation incomplete after {iterations} iteration(s).\n"
                f"Some tests may be failing. Check the '{package_name}' folder for partial results."
            )
            
    except Exception as e:
        logger.error(f"Code generation error: {e}", exc_info=True)
        return f"❌ Code generation failed: {str(e)}"


async def get_all_tools(context: Optional[MainContext] = None) -> List[Any]:
    """
    Get all available tools for the agent.
    Returns a list of function tools that can access the context.
    
    Args:
        context: Optional MainContext to inject into tools
    """
    # The tools will receive context through the wrapper when called
    tools = [
        sum_numbers,
        multiply_numbers,
        generate_code
    ]
    
    return tools