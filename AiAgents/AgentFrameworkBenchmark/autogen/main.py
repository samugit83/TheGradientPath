#!/usr/bin/env python3
"""
LLM-Powered Interactive Chat System with Tool Selection
"""

import os
import logging
from logging_config import setup_autogen_logging, get_logger, cleanup_old_logs

log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
log_levels = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}
log_level = log_levels.get(log_level_str, logging.INFO)


setup_autogen_logging(
    log_dir="logs",
    log_level=log_level,
    max_bytes=10 * 1024 * 1024, 
    backup_count=5, 
    console_output=True,
    reduce_autogen_noise=True 
)

cleanup_old_logs(log_dir="logs", days_to_keep=30)
logger = get_logger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import asyncio
from autogen_runtime import create_autogen_chat_system


def main():
    """Entry point for the AutoGen-powered interactive chat"""
    
    session_id = "autogen_session"
    
    async def run_autogen_chat():
        """Async wrapper to run the AutoGen chat system"""
        try:
            chat_system = await create_autogen_chat_system(session_id=session_id)
            await chat_system.run_interactive()
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Thanks for using the AutoGen-powered chat.")
        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)
            print(f"\n❌ Fatal error: {e}")
    
    try:
        asyncio.run(run_autogen_chat())
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye! Thanks for using the AutoGen-powered chat.")


if __name__ == "__main__":
    main()