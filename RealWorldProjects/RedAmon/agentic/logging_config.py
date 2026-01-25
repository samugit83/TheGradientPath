"""
RedAmon Agent Logging Configuration

Configures logging with file rotation, console output, and proper formatting.
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from params import LOG_MAX_MB, LOG_BACKUP_COUNT

# =============================================================================
# LOGGING SETTINGS
# =============================================================================

# Log directory (relative to this file)
LOG_DIR = Path(__file__).parent / "logs"

# Log file settings
LOG_FILE_NAME = "agent.log"
LOG_MAX_BYTES = LOG_MAX_MB * 1024 * 1024  # Convert MB to bytes

# Log levels
FILE_LOG_LEVEL = logging.DEBUG
CONSOLE_LOG_LEVEL = logging.INFO

# Log format
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Detailed format for file (includes more context)
FILE_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-25s | %(funcName)-20s | %(message)s"


def setup_logging(
    log_level: int = logging.INFO,
    log_to_console: bool = True,
    log_to_file: bool = True,
) -> None:
    """
    Configure logging for the RedAmon agent.

    Args:
        log_level: Minimum log level for console output
        log_to_console: Whether to output logs to console
        log_to_file: Whether to output logs to file with rotation
    """
    # Ensure log directory exists
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Get root logger for agentic module
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture all levels, handlers will filter

    # Clear existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

    # File handler with rotation
    if log_to_file:
        log_file_path = LOG_DIR / LOG_FILE_NAME
        file_handler = RotatingFileHandler(
            filename=str(log_file_path),
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(FILE_LOG_LEVEL)
        file_formatter = logging.Formatter(FILE_LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.INFO)
    logging.getLogger("langgraph").setLevel(logging.INFO)
    logging.getLogger("neo4j").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    # MCP client logs are very verbose - suppress them
    logging.getLogger("mcp").setLevel(logging.WARNING)
    logging.getLogger("mcp.client").setLevel(logging.WARNING)
    logging.getLogger("mcp.client.sse").setLevel(logging.WARNING)

    # Log startup message
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured - File: {LOG_DIR / LOG_FILE_NAME}")
    logger.info(f"Max file size: {LOG_MAX_BYTES / 1024 / 1024:.1f} MB, Backup count: {LOG_BACKUP_COUNT}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)
