"""
Centralized logging configuration with file rotation for the multi-agent system.
"""

import logging
import logging.handlers
import os
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logging(
    log_dir: str = "logs",
    log_level: int = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB per file
    backup_count: int = 5,  # Keep 5 backup files
    console_output: bool = True,
    log_format: Optional[str] = None,
    setup_semantic_kernel_logging: bool = True
) -> None:
    """
    Configure centralized logging with file rotation for all loggers.
    
    Args:
        log_dir: Directory to store log files
        log_level: Logging level (default: INFO)
        max_bytes: Maximum size of each log file in bytes (default: 10MB)
        backup_count: Number of backup files to keep (default: 5)
        console_output: Whether to also output logs to console (default: True)
        log_format: Custom log format string (optional)
    """
    
    # Create logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Default format if not provided
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    
    # Create formatters
    formatter = logging.Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')
    
    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Create rotating file handler for general logs
    general_log_file = log_path / f"app_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.handlers.RotatingFileHandler(
        filename=general_log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Create rotating file handler for error logs
    error_log_file = log_path / f"errors_{datetime.now().strftime('%Y%m%d')}.log"
    error_handler = logging.handlers.RotatingFileHandler(
        filename=error_log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    root_logger.addHandler(error_handler)
    
    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        # Use a simpler format for console output
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # Log initial message
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized - logs directory: {log_path.absolute()}")
    logger.info(f"Log rotation configured: max {max_bytes / 1024 / 1024:.1f}MB per file, keeping {backup_count} backups")
    
    # Set up semantic kernel logging if requested
    if setup_semantic_kernel_logging:
        setup_semantic_kernel_logging_config()


def setup_semantic_kernel_logging_config() -> None:
    """
    Configure logging levels for semantic kernel components to reduce verbosity.
    This function sets various semantic kernel loggers to WARNING level to minimize
    noise while preserving important warnings and errors.
    """
    # List of semantic kernel loggers to set to WARNING level
    sk_loggers = [
        "kernel",
        "semantic_kernel", 
        "in_process_runtime",
        "in_process_runtime.events",
        "semantic_kernel.connectors.ai.chat_completion_client_base",
        "semantic_kernel.functions.kernel_function",
        "httpx"
    ]
    
    for logger_name in sk_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info("Semantic kernel logging configured - verbosity reduced")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


class LoggerContext:
    """Context manager for temporary log level changes."""
    
    def __init__(self, logger: logging.Logger, level: int):
        self.logger = logger
        self.new_level = level
        self.original_level = None
    
    def __enter__(self):
        self.original_level = self.logger.level
        self.logger.setLevel(self.new_level)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.original_level)


def cleanup_old_logs(log_dir: str = "logs", days_to_keep: int = 30) -> None:
    """
    Clean up log files older than specified days.
    
    Args:
        log_dir: Directory containing log files
        days_to_keep: Number of days to keep logs (default: 30)
    """
    import time
    
    log_path = Path(log_dir)
    if not log_path.exists():
        return
    
    current_time = time.time()
    cutoff_time = current_time - (days_to_keep * 24 * 60 * 60)
    
    logger = logging.getLogger(__name__)
    
    for log_file in log_path.glob("*.log*"):
        if log_file.is_file():
            file_modified_time = os.path.getmtime(log_file)
            if file_modified_time < cutoff_time:
                try:
                    log_file.unlink()
                    logger.info(f"Deleted old log file: {log_file.name}")
                except Exception as e:
                    logger.error(f"Failed to delete old log file {log_file.name}: {e}")
