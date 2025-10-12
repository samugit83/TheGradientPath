"""
Tools module for the chat system
Each tool is a callable class with a run method and metadata
"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import shutil
import json
import subprocess
import atexit
import os

logger = logging.getLogger(__name__)


class BaseTool:
    """Base class for all tools"""
    
    def __init__(self):
        self.name = ""
        self.description = ""
        self.parameter_schema = {}
        self.examples = []
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get tool metadata for LLM"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameter_schema,
            "examples": self.examples
        }
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool - to be implemented by subclasses"""
        raise NotImplementedError


class SumTool(BaseTool):
    """Tool for adding two numbers"""
    
    def __init__(self):
        super().__init__()
        self.name = "sum"
        self.description = "Adds two numbers together and returns the result"
        self.parameter_schema = {
            "a": {
                "type": "number",
                "description": "First number to add",
                "required": True
            },
            "b": {
                "type": "number", 
                "description": "Second number to add",
                "required": True
            }
        }
        self.examples = [
            {
                "input": "What is 5 plus 3?",
                "parameters": {"a": 5, "b": 3}
            },
            {
                "input": "Add 10 and 25",
                "parameters": {"a": 10, "b": 25}
            }
        ]
    
    def run(self, a: float, b: float) -> Dict[str, Any]:
        """
        Sum two numbers
        
        Args:
            a: First number
            b: Second number
            
        Returns:
            Dict with result and explanation
        """
        try:
            result = a + b
            return {
                "success": True,
                "result": result,
                "explanation": f"The sum of {a} and {b} is {result}",
                "tool": self.name
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "tool": self.name
            }


class MultiplicationTool(BaseTool):
    """Tool for multiplying two numbers"""
    
    def __init__(self):
        super().__init__()
        self.name = "multiplication"
        self.description = "Multiplies two numbers and returns the product"
        self.parameter_schema = {
            "a": {
                "type": "number",
                "description": "First number to multiply",
                "required": True
            },
            "b": {
                "type": "number",
                "description": "Second number to multiply", 
                "required": True
            }
        }
        self.examples = [
            {
                "input": "Calculate 7 times 9",
                "parameters": {"a": 7, "b": 9}
            },
            {
                "input": "What is 8 multiplied by 6?",
                "parameters": {"a": 8, "b": 6}
            }
        ]
    
    def run(self, a: float, b: float) -> Dict[str, Any]:
        """
        Multiply two numbers
        
        Args:
            a: First number
            b: Second number
            
        Returns:
            Dict with result and explanation
        """
        try:
            result = a * b
            return {
                "success": True,
                "result": result,
                "explanation": f"The product of {a} and {b} is {result}",
                "tool": self.name
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "tool": self.name
            }


class CodeGeneratorMultiagentTool(BaseTool):
    """Tool for generating code using CodeOrchestratorAgent"""
    
    def __init__(self, package_name: str = "app"):
        super().__init__()
        self.name = "code_generator"
        self.description = (
            "Generates complete code implementations with tests using a multi-agent system. "
            "This tool creates production-ready python code with proper structure, "
            "and comprehensive test coverage. It iteratively improves the code until all tests pass."
        )
        self.package_name = package_name
        self.parameter_schema = {
            "prompt": {
                "type": "string",
                "description": "The main requirements or description of what code to generate",
                "required": True
            }
        }
        self.examples = []
    
    def run(self, prompt: str) -> Dict[str, Any]:
        """
        Generate code using the orchestrator
        
        Args:
            prompt: User requirements for code generation
            constraints: Code generation constraints
            
        Returns:
            Dict with generation results
        """
        try:
            from code_generator_multiagent import CodeOrchestratorAgent
            from session_manager import get_session_manager, add_agent_message
            
            # Clear the app folder before generating new code
            app_dir = Path.cwd() / self.package_name
            if app_dir.exists():
                logger.info("Emptying app folder before code generation")
                shutil.rmtree(app_dir)
            
            # Log start of code generation to session if available
            session_manager = get_session_manager()
            if session_manager:
                add_agent_message(
                    agent_name="CodeOrchestratorAgent",
                    agent_type="orchestrator",
                    content=f"Starting code generation for: {prompt}"
                )
            

            constraints = {
                "language": "Python 3.8+",
                "style": "PEP 8 compliant",
                "documentation": "Add docstrings",
                "allowed_packages": "stdlib"
            }
        
            # Create and run orchestrator
            orchestrator = CodeOrchestratorAgent(
                user_prompt=prompt,
                constraints=constraints,
                max_iters=3,
                package_name=self.package_name,
                max_test=8
            )
            
            final_state = orchestrator.run()
            
            # Extract token usage from the final state
            usage = {
                "requests": final_state.requests,
                "input_tokens": final_state.input_tokens,
                "output_tokens": final_state.output_tokens,
                "total_tokens": final_state.total_tokens
            }
            
            if final_state.code_gen_state.process_completed:
                if session_manager:
                    add_agent_message(
                        agent_name="CodeOrchestratorAgent",
                        agent_type="orchestrator",
                        content=f"Code generation completed successfully in {final_state.code_gen_state.iteration} iteration(s)"
                    )
                
                return {
                    "success": True,
                    "result": "Code generated successfully",
                    "explanation": f"✅ Code generated in {final_state.code_gen_state.iteration} iteration(s). Check the '{self.package_name}' folder.",
                    "iterations": final_state.code_gen_state.iteration,
                    "test_results": {
                        "passed": final_state.code_gen_state.test_results.passed if final_state.code_gen_state.test_results else 0,
                        "failed": final_state.code_gen_state.test_results.failed if final_state.code_gen_state.test_results else 0
                    },
                    "tool": self.name,
                    "_usage": usage  # Include usage information
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
                    },
                    "tool": self.name,
                    "_usage": usage  # Include usage information
                }
                
        except Exception as e:
            logger.error(f"Code generation error: {e}")
            return {
                "success": False,
                "error": str(e),
                "tool": self.name
            }


class MCPToolAdapter(BaseTool):
    """Adapter that wraps an MCP tool to work with our BaseTool interface"""
    
    def __init__(self, tool_info: Dict[str, Any], mcp_connection):
        """
        Initialize adapter with MCP tool metadata and connection
        
        Args:
            tool_info: Tool metadata from MCP server
            mcp_connection: MCPWeatherIntegration instance for RPC calls
        """
        super().__init__()
        self.mcp_connection = mcp_connection
        self.mcp_tool_name = tool_info["name"]
        
        # Map MCP tool info to our format
        self.name = tool_info["name"].replace("-", "_")  # Convert to Python-friendly name
        self.description = tool_info.get("description", "")
        
        # Convert MCP inputSchema to our parameter_schema format
        self._convert_schema(tool_info.get("inputSchema", {}))
        
        # Generate examples based on the tool
        self._generate_examples()
    
    def _convert_schema(self, input_schema: Dict[str, Any]):
        """Convert MCP inputSchema to our parameter_schema format"""
        self.parameter_schema = {}
        
        if "properties" in input_schema:
            for param_name, param_info in input_schema["properties"].items():
                self.parameter_schema[param_name] = {
                    "type": param_info.get("type", "string"),
                    "description": param_info.get("description", ""),
                    "required": param_name in input_schema.get("required", [])
                }
                
                # Add enum values if present
                if "enum" in param_info:
                    self.parameter_schema[param_name]["enum"] = param_info["enum"]
    
    def _generate_examples(self):
        """Generate usage examples based on the tool type"""
        if "hourly" in self.name:
            self.examples = [
                {
                    "input": "Get hourly weather for Rome",
                    "parameters": {"location": "Rome, Italy", "units": "metric"}
                },
                {
                    "input": "Show next 12 hours weather in New York in Fahrenheit",
                    "parameters": {"location": "New York, USA", "units": "imperial"}
                }
            ]
        elif "daily" in self.name:
            self.examples = [
                {
                    "input": "Get 5-day weather forecast for London",
                    "parameters": {"location": "London, UK", "days": 5, "units": "metric"}
                },
                {
                    "input": "Show 10-day forecast for Tokyo",
                    "parameters": {"location": "Tokyo, Japan", "days": 10, "units": "metric"}
                }
            ]
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the MCP tool via RPC
        
        Args:
            **kwargs: Tool arguments
            
        Returns:
            Dict with execution results
        """
        try:
            # Call the MCP tool via RPC
            result = self.mcp_connection.call_tool(self.mcp_tool_name, kwargs)
            logger.info(f"MCP tool result: {result}")
            
            if result.get("result"):
                # Parse the content if it's a list with text content
                content = result["result"].get("content", [])
                if content and isinstance(content, list) and len(content) > 0:
                    text_content = content[0].get("text", "")
                    return {
                        "success": True,
                        "result": text_content,
                        "explanation": f"Weather data retrieved successfully",
                        "tool": self.name
                    }
                else:
                    return {
                        "success": True,
                        "result": result["result"],
                        "explanation": f"Tool executed successfully",
                        "tool": self.name
                    }
            elif result.get("error"):
                return {
                    "success": False,
                    "error": result["error"].get("message", "Unknown error"),
                    "tool": self.name
                }
            else:
                return {
                    "success": False,
                    "error": "Unexpected response format from MCP server",
                    "tool": self.name
                }
                
        except Exception as e:
            logger.error(f"MCP tool execution error: {e}")
            return {
                "success": False,
                "error": str(e),
                "tool": self.name
            }


class MCPWeatherIntegration:
    """Integration for MCP Weather server that dynamically loads available tools"""
    
    def __init__(self, mcp_command: List[str] = None):
        """
        Initialize MCP Weather integration
        
        Args:
            mcp_command: Command to start MCP server (defaults to weather MCP)
        """
        self.mcp_command = mcp_command or ["npx", "-y", "@timlukahorstmann/mcp-weather"]
        self.process = None
        self.rpc_id = 0
        self.tools = []
        
        # Start the server before trying to load tools
        self._start_server()
        self._load_tools()
    
    def _start_server(self):
        """Start the MCP server process"""
        try:
            # Check if API key is set
            if not os.environ.get("ACCUWEATHER_API_KEY"):
                logger.warning("ACCUWEATHER_API_KEY not set in environment variables")
            
            logger.info(f"Starting MCP server with command: {' '.join(self.mcp_command)}")
            self.process = subprocess.Popen(
                self.mcp_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            # Register cleanup on exit
            atexit.register(self.cleanup)
            
            logger.info("MCP server started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start MCP server: {e}")
            raise
    
    def _restart_server(self):
        """Restart the MCP server process after a crash"""
        try:
            # Clean up old process if it exists
            if self.process:
                try:
                    self.process.terminate()
                    self.process.wait(timeout=2)
                except:
                    pass
            
            # Start fresh server
            self._start_server()
            
            # Reload tools from the new server
            self.tools = []  # Clear old tools
            self._load_tools()
            
            logger.info("MCP server restarted successfully")
            
        except Exception as e:
            logger.error(f"Failed to restart MCP server: {e}")
            raise
    
    def _rpc(self, method: str, params: Optional[Dict] = None, retry: bool = True) -> Dict[str, Any]:
        """
        Send JSON-RPC request to MCP server with auto-recovery
        
        Args:
            method: RPC method name
            params: Optional parameters
            retry: Whether to retry on failure (internal use)
            
        Returns:
            RPC response
        """
        if not self.process or self.process.poll() is not None:
            if retry:
                logger.warning("MCP server not running, attempting to restart...")
                self._restart_server()
            else:
                raise RuntimeError("MCP server not running")
        
        self.rpc_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self.rpc_id,
            "method": method
        }
        
        if params is not None:
            request["params"] = params
        
        try:
            # Send request
            request_str = json.dumps(request) + "\n"
            self.process.stdin.write(request_str)
            self.process.stdin.flush()
            
            # Read response
            response_str = self.process.stdout.readline()
            if not response_str:
                raise RuntimeError("No response from MCP server")
            
            response = json.loads(response_str)
            return response
            
        except (BrokenPipeError, OSError) as e:
            logger.error(f"RPC communication error: {e}")
            if retry:
                logger.info("Attempting to restart MCP server and retry request...")
                self._restart_server()
                return self._rpc(method, params, retry=False)  # Retry once
            else:
                raise
        except Exception as e:
            logger.error(f"RPC communication error: {e}")
            raise
    
    def _load_tools(self):
        """Load available tools from MCP server"""
        try:
            response = self._rpc("tools/list")
            
            if "result" in response and "tools" in response["result"]:
                mcp_tools = response["result"]["tools"]
                logger.info(f"Loaded {len(mcp_tools)} tools from MCP server")
                
                # Create adapters for each tool
                for tool_info in mcp_tools:
                    adapter = MCPToolAdapter(tool_info, self)
                    self.tools.append(adapter)
                    logger.info(f"Created adapter for MCP tool: {tool_info['name']}")
            else:
                logger.warning("No tools found in MCP server response")
                
        except Exception as e:
            logger.error(f"Failed to load MCP tools: {e}")
            raise
    
    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call an MCP tool via RPC
        
        Args:
            tool_name: Name of the MCP tool
            arguments: Tool arguments
            
        Returns:
            Tool execution result
        """
        return self._rpc("tools/call", {
            "name": tool_name,
            "arguments": arguments
        })
    
    def get_adapted_tools(self) -> List[BaseTool]:
        """Get all adapted tools ready for registration"""
        return self.tools
    
    def cleanup(self):
        """Clean up MCP server process"""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            except Exception as e:
                logger.error(f"Error during MCP server cleanup: {e}")
            finally:
                self.process = None
                logger.info("MCP server terminated")


class ToolRegistry:
    """Registry for managing available tools"""
    
    def __init__(self):
        self.tools = {}
        self.mcp_integration = None
        self._register_default_tools()
        self._register_mcp_weather_tools()
    
    def _register_default_tools(self):
        """Register the default tools"""
        self.register(SumTool())
        self.register(MultiplicationTool())
        self.register(CodeGeneratorMultiagentTool())
    
    def _register_mcp_weather_tools(self):
        """Register MCP Weather tools dynamically"""
        try:
            logger.info("Initializing MCP Weather integration...")
            self.mcp_integration = MCPWeatherIntegration()
            
            # Register all adapted tools from MCP
            self._refresh_mcp_tools()
            
            logger.info(f"Successfully registered {len(self.mcp_integration.get_adapted_tools())} MCP Weather tools")
            
        except Exception as e:
            logger.error(f"Failed to register MCP Weather tools: {e}")
    
    def _refresh_mcp_tools(self):
        """Refresh MCP tool registrations (useful after server restart)"""
        if not self.mcp_integration:
            return
        
        # Remove old MCP tools
        old_mcp_tools = [name for name, tool in self.tools.items() if isinstance(tool, MCPToolAdapter)]
        for tool_name in old_mcp_tools:
            del self.tools[tool_name]
        
        # Re-register current MCP tools
        for tool in self.mcp_integration.get_adapted_tools():
            self.register(tool)
        
        logger.info(f"Refreshed {len(self.mcp_integration.get_adapted_tools())} MCP tools")
    
    def register(self, tool: BaseTool):
        """Register a new tool"""
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name"""
        return self.tools.get(name)
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools with basic info"""
        return [
            {
                "name": tool.name,
                "description": tool.description
            }
            for tool in self.tools.values()
        ]
    
    def get_tools_metadata(self) -> List[Dict[str, Any]]:
        """Get detailed metadata for all tools"""
        return [tool.get_metadata() for tool in self.tools.values()]
    
    def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute a tool by name with given arguments"""
        tool = self.get_tool(tool_name)
        if not tool:
            return {
                "success": False,
                "error": f"Tool '{tool_name}' not found",
                "available_tools": [t["name"] for t in self.list_tools()]
            }
        
        try:
            return tool.run(**kwargs)
        except TypeError as e:
            return {
                "success": False,
                "error": f"Invalid arguments for tool '{tool_name}': {str(e)}",
                "tool": tool_name
            }
    
    def validate_parameters(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate if all required parameters are provided for a tool"""
        tool = self.get_tool(tool_name)
        if not tool:
            return {
                "valid": False,
                "error": f"Tool '{tool_name}' not found"
            }
        
        missing_params = []
        for param_name, param_info in tool.parameter_schema.items():
            if param_info.get("required", False) and param_name not in parameters:
                missing_params.append(param_name)
        
        if missing_params:
            return {
                "valid": False,
                "missing_parameters": missing_params,
                "parameter_schema": tool.parameter_schema
            }
        
        return {"valid": True}
    
    def cleanup(self):
        """Clean up resources, especially MCP server processes"""
        if self.mcp_integration:
            self.mcp_integration.cleanup()
            logger.info("Cleaned up MCP integration")
    
    def get_mcp_tools(self) -> List[str]:
        """Get list of MCP tool names"""
        mcp_tools = []
        for name, tool in self.tools.items():
            if isinstance(tool, MCPToolAdapter):
                mcp_tools.append(name)
        return mcp_tools