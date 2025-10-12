#!/usr/bin/env python3
"""
LLM-Powered Interactive Chat System with Tool Selection using openai-agents
"""

import os
import sys
import asyncio
import logging
from logging_config import setup_logging, get_logger, cleanup_old_logs
from agents import SQLiteSession
from tools import get_all_tools
from mcp_servers import get_mcp_servers, cleanup_mcp_servers
from context import MainContext
from helpers import run_chat_loop_with_usage
from handoff_agents import setup_handoff_agents


log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
log_levels = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}
log_level = log_levels.get(log_level_str, logging.INFO)

setup_logging(
    log_dir="logs",
    log_level=log_level,
    max_bytes=10 * 1024 * 1024, 
    backup_count=5, 
    console_output=True 
)

cleanup_old_logs(log_dir="logs", days_to_keep=30)
logger = get_logger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


async def main() -> None:
    """Entry point for the chat"""
    print("\n" + "=" * 60)
    print("  🤖 Welcome to AI-Powered Chat with Tool Selection! 🚀")
    print("=" * 60)
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("\n⚠️  Warning: OPENAI_API_KEY not set!")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    

    main_context = MainContext()
    tools = await get_all_tools(main_context)
    
    print("\n📦 Available tools:")
    for tool in tools:
        if hasattr(tool, 'name') and hasattr(tool, 'description'):
            desc = tool.description
            if len(desc) > 80:
                desc = desc[:77] + "..."
            print(f"  • {tool.name}: {desc}")
    

    mcp_servers_with_names = await get_mcp_servers()
    
    if mcp_servers_with_names:
        print("\n🔌 Connected MCP servers:")
        for name, server in mcp_servers_with_names:
            print(f"  • {name}")
        mcp_servers = [server for _, server in mcp_servers_with_names]
    else:
        print("\n⚠️  No MCP servers configured.")
        mcp_servers = []
        
    print("\n💬 Chat Instructions:")
    print("  • Just type naturally and I'll understand what you need!")
    print("  • Type 'quit' or 'exit' to leave the chat")
    print("  • I'll intelligently route your request to the right specialist:")
    print("    - ⚖️ Legal Expert: For law-related questions and legal topics")
    print("    - 🔧 General Agent: For everything else (programming, tools, general knowledge)")
    print("\n🛡️  Content Safety:")
    print("  • The routing agent has content safety guardrails enabled")
    print("  • Requests for illegal, harmful, or inappropriate content will be blocked")
    print("  • The system prioritizes safe and helpful interactions\n")

    session = None
    if main_context.use_persistent_memory:
        print("🧠 Persistent memory enabled - conversations will be saved to database")
        session = SQLiteSession("user_main", "conversations.db")
    else:
        print("🧠 In-memory mode - conversations will not be saved")
    
    main_context.session = session

    # Use the new routing agent with handoffs to specialized agents
    agent = setup_handoff_agents(
        tools=tools,
        mcp_servers=mcp_servers,
        model_name=main_context.model_name,
        enable_content_safety=main_context.enable_content_safety
    )
    
    try:
        await run_chat_loop_with_usage(agent, main_context, session)
    finally:
        await cleanup_mcp_servers(mcp_servers)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye! Thanks for using the AI-powered chat.")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)