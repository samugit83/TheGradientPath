#!/usr/bin/env python3
"""
LLM-Powered Interactive Chat System with Agent Routing using CrewAI

This module provides the main entry point for a conversational AI system that uses
CrewAI to route user queries between specialized agents (Legal Expert and Operational Agent)
through an intelligent Conversation Manager.
"""

import os
import sys
import logging
import atexit
from logging_config import setup_logging, get_logger, cleanup_old_logs
from run_chat import run_chat_loop
from chat_crew import ChatCrew
from state import initialize_global_state
from tools import cleanup_mcp_server


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
    console_output=True,
    setup_semantic_kernel_logging=True
)

cleanup_old_logs(log_dir="logs", days_to_keep=30)
logger = get_logger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def main() -> None:
    """Main entry point for the CrewAI-powered chat system.
    
    Initializes the chat crew and starts the interactive chat loop.
    Handles environment variable validation and graceful shutdown.
    """
    print("\n" + "=" * 60)
    print("  🤖 Welcome to AI-Powered Chat with Tool Selection! 🚀")
    print("=" * 60)
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("\n⚠️  Warning: OPENAI_API_KEY not set!")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)

    # Register cleanup function
    atexit.register(cleanup_mcp_server)
    
    session_id = "my_session"
    initialize_global_state(session_id=session_id)
    crew = ChatCrew().crew(session_id=session_id)    
    run_chat_loop(crew, session_id)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye! Thanks for using the AI-powered chat.")
        cleanup_mcp_server()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        cleanup_mcp_server()
        sys.exit(1)