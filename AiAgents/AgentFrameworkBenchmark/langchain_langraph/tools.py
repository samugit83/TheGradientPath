"""
Tools module for the chat system using LangChain tool decorators
"""

import asyncio
import logging
import os
from typing import Any, List
from pathlib import Path
import shutil
from langchain_core.tools import tool
from state import MainState, update_usage

# MCP Integration imports
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.session import ClientSession

logger = logging.getLogger(__name__)


def add_numbers(a: float, b: float, state: MainState = None) -> str:
    """Add two numbers together and return the result.
    
    Args:
        a: First number to add
        b: Second number to add
        state: The current agent state (automatically injected)
        
    Returns:
        String describing the result
    """
    try:
        result = a + b
        return f"The sum of {a} and {b} is {result}"
    except Exception as e:
        logger.error(f"Error in sum calculation: {e}")
        return f"Error calculating sum: {str(e)}"


def multiply_numbers(a: float, b: float, state: MainState = None) -> str:
    """Multiply two numbers together and return the product.
    
    Args:
        a: First number to multiply
        b: Second number to multiply
        state: The current agent state (automatically injected)
        
    Returns:
        String describing the result
    """
    try:
        result = a * b
        return f"The product of {a} and {b} is {result}"
    except Exception as e:
        logger.error(f"Error in multiplication: {e}")
        return f"Error calculating product: {str(e)}"



def generate_code(prompt: str, state: MainState) -> str:
    """Generate complete code implementations with tests using a multi-agent system.
    
    This tool creates production-ready Python code with proper structure and comprehensive test coverage.
    It iteratively improves the code until all tests pass.
    
    Args:
        prompt: Description of the code to generate
        state: The current agent state (automatically injected)
        
    Returns:
        String describing the generation result
    """
    try:
        if not prompt:
            return "❌ Error: 'prompt' parameter is required and cannot be empty"
        
        # Clear the app folder before code generation (user preference)
        app_folder = Path("app")
        if app_folder.exists():
            try:
                shutil.rmtree(app_folder)
                logger.info("Cleared existing app folder before code generation")
            except Exception as e:
                logger.warning(f"Could not clear app folder: {e}")
        
        from code_generator_multiagent import CodeOrchestratorAgent
        from state import CodeGenState
        
        # Ensure code_gen_state is initialized (since __post_init__ doesn't work with TypedDict)
        if state.code_gen_state is None:
            state.code_gen_state = CodeGenState()
        
        # Set the prompt in the code generation state
        state.code_gen_state.user_prompt_for_app = prompt
        
        # Create and run orchestrator with state (use the main state's LLM if available)
        from langchain.chat_models import init_chat_model
        llm_client = init_chat_model(state.model_name, model_provider="openai")
        orchestrator = CodeOrchestratorAgent(llm_client=llm_client)
        final_state = orchestrator.run(state)
        
        # Extract token usage from code generation and update main state
        if hasattr(final_state, '_code_gen_usage') and final_state._code_gen_usage:
            update_usage(state, final_state._code_gen_usage)
            logger.info(f"🔢 Code generation consumed {final_state._code_gen_usage['total_tokens']} tokens "
                       f"across {final_state._code_gen_usage['requests']} LLM requests")
        
        package_name = final_state.code_gen_state.package_name
        
        if final_state.code_gen_state and final_state.code_gen_state.process_completed:
            test_info = ""
            if final_state.code_gen_state.test_results:
                test_results = final_state.code_gen_state.test_results
                test_info = f" Tests: {test_results.passed} passed, {test_results.failed} failed."
            
            return (
                f"✅ Code generated successfully in {final_state.code_gen_state.iteration} iteration(s)!{test_info}\n"
                f"Check the '{package_name}' folder for the generated code.\n"
                f"Files created: main.py and test_main.py"
            )
        else:
            iterations = final_state.code_gen_state.iteration if final_state.code_gen_state else "unknown"
            return (
                f"⚠️ Code generation incomplete after {iterations} iteration(s).\n"
                f"Some tests may be failing. Check the '{package_name}' folder for partial results."
            )
            
    except Exception as e:
        logger.error(f"Code generation error: {e}", exc_info=True)
        return f"❌ Code generation failed: {str(e)}"


async def load_mcp_weather_tools() -> List[Any]:
    """Load MCP weather tools using langchain_mcp_adapters following the provided example.
    
    Returns:
        List of MCP weather tools wrapped as LangChain tools
    """
    try:
        # Check if API key is set
        if not os.environ.get("ACCUWEATHER_API_KEY"):
            logger.warning("ACCUWEATHER_API_KEY not set in environment variables. Weather tools may not work properly.")
        
        # Configure the MCP server parameters for weather service with environment
        server_params = StdioServerParameters(
            command="npx",
            args=["-y", "@timlukahorstmann/mcp-weather"],
            env=os.environ.copy()  # Pass current environment including API key
        )
        
        logger.info("Connecting to MCP weather server...")
        
        # Connect and load tools using the langchain MCP adapter (following the example)
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                mcp_tools = await load_mcp_tools(session)
                
                logger.info(f"Successfully loaded {len(mcp_tools)} MCP weather tools")
                return mcp_tools
                
    except Exception as e:
        logger.error(f"Failed to load MCP weather tools: {e}")
        return []


def _run_async_in_thread(async_func, *args, **kwargs):
    """Helper to run async function in new thread with new event loop."""
    import concurrent.futures
    
    def run_in_loop():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(async_func(*args, **kwargs))
        finally:
            loop.close()
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        return executor.submit(run_in_loop).result(timeout=30)


async def _call_mcp_tool(tool_name: str, kwargs: dict) -> str:
    """Call MCP tool with fresh connection."""
    server_params = StdioServerParameters(
        command="npx", 
        args=["-y", "@timlukahorstmann/mcp-weather"],
        env=os.environ.copy()
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            
            for tool in tools:
                if tool.name == tool_name:
                    return await tool.ainvoke(kwargs)
            
            return f"Tool {tool_name} not found"


def create_sync_tool_wrapper(async_tool):
    """Create a synchronous wrapper for an async MCP tool."""
    from langchain_core.tools import BaseTool
    from typing import Type, Any, Optional
    
    class SyncMCPTool(BaseTool):
        name: str = async_tool.name
        description: str = async_tool.description
        args_schema: Optional[Type[Any]] = async_tool.args_schema
        
        def _run(self, **kwargs) -> str:
            if not os.environ.get("ACCUWEATHER_API_KEY"):
                return "❌ ACCUWEATHER_API_KEY not set in environment variables."
            
            try:
                return _run_async_in_thread(_call_mcp_tool, async_tool.name, kwargs)
            except Exception as e:
                logger.error(f"Error calling MCP tool {self.name}: {e}")
                return f"❌ Weather service error: {str(e)}"
    
    return SyncMCPTool()


def get_mcp_weather_tools() -> List[Any]:
    """Get MCP weather tools as synchronous wrappers."""
    try:
        async_tools = _run_async_in_thread(load_mcp_weather_tools)
        sync_tools = [create_sync_tool_wrapper(tool) for tool in async_tools]
        
        for tool in sync_tools:
            logger.info(f"Created sync wrapper for: {tool.name}")
        
        return sync_tools
        
    except Exception as e:
        logger.error(f"Error loading MCP weather tools: {e}")
        return []


def get_all_tools(state: MainState = None) -> List[Any]:
    """Get all available tools for the agent.
    
    Args:
        state: Optional MainState to bind to tools that need it
    
    Returns:
        List of function tools that can be used with LangGraph agents
    """
    
    @tool
    def add_numbers_bound(a: float, b: float) -> str:
        """Add two numbers together and return the result.
        
        Args:
            a: First number to add
            b: Second number to add
            
        Returns:
            String describing the result
        """
        return add_numbers(a, b, state)
    
    @tool
    def multiply_numbers_bound(a: float, b: float) -> str:
        """Multiply two numbers together and return the product.
        
        Args:
            a: First number to multiply
            b: Second number to multiply
            
        Returns:
            String describing the result
        """
        return multiply_numbers(a, b, state)
    
    @tool
    def generate_code_bound(prompt: str) -> str:
        """Generate complete code implementations with tests using a multi-agent system.
        
        This tool creates production-ready Python code with proper structure and comprehensive test coverage.
        It iteratively improves the code until all tests pass.
        
        Args:
            prompt: Description of the code to generate
            
        Returns:
            String describing the generation result
        """
        return generate_code(prompt, state)
    
    # Start with core tools
    tools = [add_numbers_bound, multiply_numbers_bound, generate_code_bound]
    
    # Add MCP weather tools
    try:
        mcp_weather_tools = get_mcp_weather_tools()
        if mcp_weather_tools:
            tools.extend(mcp_weather_tools)
            logger.info(f"Added {len(mcp_weather_tools)} MCP weather tools to available tools")
        else:
            logger.info("No MCP weather tools available")
    except Exception as e:
        logger.error(f"Failed to add MCP weather tools: {e}")
    
    return tools

