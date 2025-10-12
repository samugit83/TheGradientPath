"""
Tools module for the chat system using CrewAI structure
"""

import logging
import os
from typing import Type, Optional, List
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Global MCP tools storage
_mcp_tools_cache = None
_mcp_server_adapter = None

class SumNumbersInput(BaseModel):
    """Input schema for SumNumbersTool."""
    a: float = Field(..., description="The first number to add.")
    b: float = Field(..., description="The second number to add.")

class SumNumbersTool(BaseTool):
    name: str = "Sum Numbers"
    description: str = "Adds two numbers together and returns the result. Use this tool when you need to perform addition calculations."
    args_schema: Type[BaseModel] = SumNumbersInput

    def _run(self, a: float, b: float) -> str:
        """Adds two numbers together and returns the result."""
        result = a + b
        logger.info(f"SumNumbersTool: {a} + {b} = {result}")
        return f"The sum of {a} and {b} is {result}"

class MultiplyNumbersInput(BaseModel):
    """Input schema for MultiplyNumbersTool."""
    a: float = Field(..., description="The first number to multiply.")
    b: float = Field(..., description="The second number to multiply.")

class MultiplyNumbersTool(BaseTool):
    name: str = "Multiply Numbers"
    description: str = "Multiplies two numbers together and returns the result. Use this tool when you need to perform multiplication calculations."
    args_schema: Type[BaseModel] = MultiplyNumbersInput

    def _run(self, a: float, b: float) -> str:
        """Multiplies two numbers together and returns the result."""
        result = a * b
        logger.info(f"MultiplyNumbersTool: {a} * {b} = {result}")
        return f"The product of {a} and {b} is {result}"


class CodeGeneratorMultiagentTool(BaseTool):
    name: str = "Code Generator"
    description: str = "Generates and tests code based on user requirements. Use this tool when the user wants to create, develop, or build an application."

    def _run(self, user_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        This tool is used to generate code for the application.
        """
        from code_generator_multiagent import CodeOrchestratorAgent

        orchestrator = CodeOrchestratorAgent()
        
        final_state = orchestrator.run()

        if final_state.code_gen_state.process_completed:
            return {
                "success": True,
                "result": "Code generated successfully",
                "explanation": f"✅ Code generated in {final_state.code_gen_state.iteration} iteration(s). Check the folder.",
                "iterations": final_state.code_gen_state.iteration,
                "test_results": {
                    "passed": final_state.code_gen_state.test_results.passed if final_state.code_gen_state.test_results else 0,
                    "failed": final_state.code_gen_state.test_results.failed if final_state.code_gen_state.test_results else 0
                }
            }
        else:
            return {
                "success": False,
                "result": "Generation incomplete",
                "explanation": f"⚠️ Generation incomplete after {final_state.code_gen_state.iteration} iteration(s)",
                "iterations": final_state.code_gen_state.iteration,
                "test_results": {
                    "passed": final_state.code_gen_state.test_results.passed if final_state.code_gen_state.test_results else 0,
                    "failed": final_state.code_gen_state.test_results.failed if final_state.code_gen_state.test_results else 0
                }
            }
            

            

def initialize_mcp_server():
    """Initialize MCP server and cache the tools."""
    global _mcp_tools_cache, _mcp_server_adapter
    
    if _mcp_tools_cache is not None:
        return _mcp_tools_cache
    
    try:
        # Import MCP dependencies
        from crewai_tools import MCPServerAdapter
        from mcp import StdioServerParameters
        
        # Set up MCP server parameters
        weather_api_key = os.getenv("ACCUWEATHER_API_KEY", "")
        server_params = StdioServerParameters(
            command="npx",
            args=["-y", "@timlukahorstmann/mcp-weather"],
            env={
                **os.environ,
                "ACCUWEATHER_API_KEY": weather_api_key,
                "UV_PYTHON": "3.12"
            }
        )
        
        # Initialize MCP server adapter without context manager
        _mcp_server_adapter = MCPServerAdapter(server_params, connect_timeout=60)
        
        # Get tools from the adapter - tools property returns a ToolCollection
        tools_collection = _mcp_server_adapter.tools
        mcp_tools = list(tools_collection)  # Convert ToolCollection to list
        
        _mcp_tools_cache = mcp_tools
        logger.info(f"Available MCP tools: {[tool.name for tool in mcp_tools]}")
        return mcp_tools
        
    except Exception as e:
        logger.warning(f"Failed to initialize MCP server: {e}")
        _mcp_tools_cache = []
        return []


def cleanup_mcp_server():
    """Cleanup MCP server connection."""
    global _mcp_server_adapter
    
    if _mcp_server_adapter is not None:
        try:
            _mcp_server_adapter.stop()
            logger.info("MCP server connection cleaned up")
        except Exception as e:
            logger.warning(f"Error during MCP server cleanup: {e}")
        finally:
            _mcp_server_adapter = None


def get_mcp_tools():
    """Get MCP tools if available."""
    return _mcp_tools_cache or []


def get_tools(include_mcp: bool = False):
    """
    Get all available tools for the CrewAI agents.
    
    Args:
        include_mcp: Whether to include MCP tools (default True)
    
    Returns:
        list: A list of instantiated tool objects ready to be used by agents
    """
    # Base tools
    tools = [
        SumNumbersTool(),
        MultiplyNumbersTool(),
        CodeGeneratorMultiagentTool()
    ]
    
    # Add MCP tools if requested
    if include_mcp:
        mcp_tools = initialize_mcp_server()
        tools.extend(mcp_tools)
    
    return tools
