#!/usr/bin/env python3
"""
LLM-Powered Interactive Chat System with Tool Selection using openai-agents
"""

import os
import sys
import asyncio
import logging
from logging_config import setup_logging, get_logger, cleanup_old_logs
from run_chat import run_chat_loop
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

    session_id = "my_session"

    handoff_orchestration = setup_handoff_agents()
    await run_chat_loop(handoff_orchestration, session_id)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye! Thanks for using the AI-powered chat.")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)