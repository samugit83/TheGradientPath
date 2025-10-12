"""
MCP (Model Context Protocol) Server Configurations
This module provides MCP servers that extend the capabilities of AI agents
by connecting them to various data sources and tools.
"""

import os
import logging
from typing import List, Tuple
from agents.mcp import MCPServerStdio

logger = logging.getLogger(__name__)


async def get_mcp_servers() -> List[Tuple[str, MCPServerStdio]]:
    """
    Initialize and return all configured MCP servers with their names.
    
    Returns:
        List of tuples containing (server_name, server_instance)
    """
    servers = []
    

    weather_api_key = os.environ.get("ACCUWEATHER_API_KEY")
    if weather_api_key:
        try:
            weather_server = MCPServerStdio(
                params={
                    "command": "npx",
                    "args": ["-y", "@timlukahorstmann/mcp-weather"],
                    "env": {
                        **os.environ,
                        "ACCUWEATHER_API_KEY": weather_api_key
                    }
                },
                # Optional: Add tool filtering if needed
                # tool_filter=create_static_tool_filter(
                #     allowed_tool_names=["get_weather", "get_forecast"]
                # )
            )
            # Connect the server before adding it to the list
            await weather_server.connect()
            servers.append(("Weather API", weather_server))
            logger.info("Weather MCP server initialized and connected successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Weather MCP server: {e}")
    else:
        logger.warning("ACCUWEATHER_API_KEY not set. Weather MCP server will not be available.")
    
    # Filesystem MCP Server (Example - commented out)
    # Uncomment to enable filesystem access
    # try:
    #     filesystem_server = MCPServerStdio(
    #         params={
    #             "command": "npx",
    #             "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/documents"],
    #         },
    #         # Restrict filesystem operations for safety
    #         tool_filter=create_static_tool_filter(
    #             allowed_tool_names=["read_file", "list_directory"],
    #             blocked_tool_names=["delete_file", "write_file"]
    #         )
    #     )
    #     await filesystem_server.connect()
    #     servers.append(("Filesystem", filesystem_server))
    #     logger.info("Filesystem MCP server initialized successfully")
    # except Exception as e:
    #     logger.error(f"Failed to initialize Filesystem MCP server: {e}")
    
    # GitHub MCP Server (Example - commented out)
    # Uncomment to enable GitHub integration
    # github_token = os.environ.get("GITHUB_TOKEN")
    # if github_token:
    #     try:
    #         github_server = MCPServerStdio(
    #             params={
    #                 "command": "npx",
    #                 "args": ["-y", "@modelcontextprotocol/server-github"],
    #                 "env": {
    #                     **os.environ,
    #                     "GITHUB_TOKEN": github_token
    #                 }
    #             }
    #         )
    #         await github_server.connect()
    #         servers.append(("GitHub", github_server))
    #         logger.info("GitHub MCP server initialized successfully")
    #     except Exception as e:
    #         logger.error(f"Failed to initialize GitHub MCP server: {e}")
    
    # Database MCP Server (Example - commented out)
    # Uncomment to enable database access
    # db_connection_string = os.environ.get("DATABASE_URL")
    # if db_connection_string:
    #     try:
    #         db_server = MCPServerStdio(
    #             params={
    #                 "command": "npx",
    #                 "args": ["-y", "@modelcontextprotocol/server-postgres"],
    #                 "env": {
    #                     **os.environ,
    #                     "DATABASE_URL": db_connection_string
    #                 }
    #             },
    #             # Only allow read operations
    #             tool_filter=create_static_tool_filter(
    #                 allowed_tool_names=["query", "list_tables"],
    #                 blocked_tool_names=["execute", "drop_table", "create_table"]
    #             )
    #         )
    #         await db_server.connect()
    #         servers.append(("Database", db_server))
    #         logger.info("Database MCP server initialized successfully")
    #     except Exception as e:
    #         logger.error(f"Failed to initialize Database MCP server: {e}")
    
    return servers


async def cleanup_mcp_servers(servers: List):
    """
    Properly cleanup MCP servers
    
    Args:
        servers: List of MCP server instances to cleanup
    """
    for server in servers:
        try:
            # Try different cleanup methods based on what's available
            if hasattr(server, 'close'):
                await server.close()
            elif hasattr(server, '__aexit__'):
                await server.__aexit__(None, None, None)
            logger.debug("MCP server cleaned up")
        except Exception as e:
            # Log as debug since cleanup errors are often expected during shutdown
            logger.debug(f"Error during MCP server cleanup (can be ignored): {e}")