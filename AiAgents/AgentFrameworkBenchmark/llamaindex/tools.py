"""
Tools module for the chat system
"""

import logging
import shutil
import os
from pathlib import Path
from typing import Any, List, Annotated
from state import CodeGenState
from llama_index.core.workflow import Context
from llama_index.tools.mcp import BasicMCPClient, aget_tools_from_mcp_url


logger = logging.getLogger(__name__)



async def sum_numbers(
    ctx: Context, 
    a: Annotated[float, "First number to add"],
    b: Annotated[float, "Second number to add"]
) -> str:
    """
    Adds two numbers together and returns the result.
    
    Args:
        ctx: LlamaIndex state for accessing state
        a: First number to add
        b: Second number to add
    """
    try:
        # Access state state
        async with ctx.store.edit_state() as ctx_state:
            # Update usage in main_state
            main_state = ctx_state["state"].get("main_state")
            if main_state:
                main_state.requests += 1
                ctx_state["state"]["main_state"] = main_state
        
        result = a + b
        return f"The sum of {a} and {b} is {result}"
    except Exception as e:
        logger.error(f"Error in sum calculation: {e}")
        return f"Error calculating sum: {str(e)}"


async def multiply_numbers(
    ctx: Context, 
    a: Annotated[float, "First number to multiply"],
    b: Annotated[float, "Second number to multiply"]
) -> str:
    """
    Multiplies two numbers and returns the product.
    
    Args:
        ctx: LlamaIndex state for accessing state
        a: First number to multiply
        b: Second number to multiply
    """
    try:
        # Access state state
        async with ctx.store.edit_state() as ctx_state:
            # Update usage in main_state
            main_state = ctx_state["state"].get("main_state")
            if main_state:
                main_state.requests += 1
                ctx_state["state"]["main_state"] = main_state
        
        result = a * b
        return f"The product of {a} and {b} is {result}"
    except Exception as e:
        logger.error(f"Error in multiplication: {e}")
        return f"Error calculating product: {str(e)}"


async def get_mcp_tools() -> List[Any]:
    """
    Get tools from MCP servers.
    Returns a list of FunctionTools from MCP servers.
    """
    mcp_tools = []
    
    try:
        # Weather MCP Server
        weather_api_key = os.environ.get("ACCUWEATHER_API_KEY")
        if weather_api_key:
            client = BasicMCPClient(
                "npx", 
                args=["-y", "@timlukahorstmann/mcp-weather"],
                env={
                    **os.environ,
                    "ACCUWEATHER_API_KEY": weather_api_key
                }
            )
            
            # Get tools from weather MCP server
            weather_tools = await aget_tools_from_mcp_url(
                "",  # Empty URL since we're using a custom stdio client
                client=client
            )
            mcp_tools.extend(weather_tools)
            logger.info(f"Loaded {len(weather_tools)} tools from Weather MCP server")
        else:
            logger.warning("ACCUWEATHER_API_KEY not set. Weather MCP tools will not be available.")
            
    except Exception as e:
        logger.error(f"Error loading MCP tools: {e}")
    
    return mcp_tools


# ============================================================================
# Code Generation Tool
# ============================================================================

# IMPORTANT: this a Subagent as tool: https://docs.llamaindex.ai/en/stable/understanding/agent/multi_agent/


async def generate_code(
    ctx: Context, 
    prompt: Annotated[str, "Description of the code to generate"]
) -> str:
    """
    Generates complete code implementations with tests using a multi-agent system.
    This tool creates production-ready Python code with proper structure and comprehensive test coverage.
    
    Args:
        ctx: LlamaIndex state for accessing state
        prompt: Description of the code to generate
    """
    try:
        if not prompt:
            return "❌ Error: 'prompt' parameter is required and cannot be empty"
        
        from code_generator_agents.code_orchestrator_agent import run_code_orchestrator

        async with ctx.store.edit_state() as ctx_state:
            main_state = ctx_state["state"].get("main_state")
            
            if not main_state:
                return "❌ Error: MainState not found in context"
            
            # Initialize code_gen_state if it doesn't exist
            if not main_state.code_gen_state:
                main_state.code_gen_state = CodeGenState()
            
            # Set the user prompt in the code generation state
            main_state.code_gen_state.user_prompt_for_app = prompt
            package_name = main_state.code_gen_state.package_name
        
        # Empty the app folder per user preference
        app_dir = Path.cwd() / package_name
        if app_dir.exists():
            logger.info("Emptying app folder before code generation.")
            shutil.rmtree(app_dir)
        
        # Run the multi-agent code generation
        logger.info("Starting multi-agent code generation...")
        
        # The orchestrator will update the state with usage info
        await run_code_orchestrator(ctx=ctx)
        
        # Update state state with usage information
        async with ctx.store.edit_state() as ctx_state:
            main_state = ctx_state["state"].get("main_state")
            if not main_state:
                return "❌ Error: MainState not found in context"
            if main_state.code_gen_state and main_state.code_gen_state.process_completed:
                test_info = ""
                if main_state.code_gen_state.test_results:
                    test_results = main_state.code_gen_state.test_results
                    test_info = f" Tests: {test_results.passed} passed, {test_results.failed} failed."
                
                return (
                    f"✅ Code generated successfully in {main_state.code_gen_state.iteration} iteration(s)!{test_info}\n"
                    f"Check the '{package_name}' folder for the generated code.\n"
                    f"Files created: main.py and test_main.py"
                )
            else:
                iterations = main_state.code_gen_state.iteration if main_state.code_gen_state else "unknown"
                return (
                    f"⚠️ Code generation incomplete after {iterations} iteration(s).\n"
                    f"Some tests may be failing. Check the '{package_name}' folder for partial results."
                )
            
    except Exception as e:
        logger.error(f"Code generation error: {e}", exc_info=True)
        return f"❌ Code generation failed: {str(e)}"


async def get_all_tools() -> List[Any]:
    """
    Get all available tools for the agent.
    Returns a list of function tools that can be used with LlamaIndex agents.
    """
    tools = [
        sum_numbers,
        multiply_numbers,
        generate_code
    ]
    
    # Add MCP tools
    mcp_tools = await get_mcp_tools()
    tools.extend(mcp_tools)
    
    return tools