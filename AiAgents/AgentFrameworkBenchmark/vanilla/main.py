#!/usr/bin/env python3
"""
LLM-Powered Interactive Chat System with Tool Selection
"""

import os
import logging
from logging_config import setup_logging, get_logger, cleanup_old_logs

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

from helpers import ChatLoop


def main():
    """Entry point for the AI-powered interactive chat"""

    session_id = "my_session"
    chat = ChatLoop(session_id=session_id)
    
    try:
        chat.run()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye! Thanks for using the AI-powered chat.")
        if hasattr(chat, 'state') and chat.state.requests > 0:
            print(f"\n{chat.state.get_usage_summary()}")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\n❌ Fatal error: {e}")


if __name__ == "__main__":
    main()