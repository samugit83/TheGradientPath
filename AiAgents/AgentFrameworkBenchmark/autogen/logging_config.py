"""
Enhanced logging configuration for AutoGen integration
"""

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional


def setup_autogen_logging(
    log_dir: str = "logs",
    log_level: int = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    console_output: bool = True,
    reduce_autogen_noise: bool = True
) -> None:
    """
    Setup logging configuration with AutoGen-specific adjustments.
    
    Args:
        log_dir: Directory for log files
        log_level: Logging level for application logs
        max_bytes: Maximum size per log file
        backup_count: Number of backup files to keep
        console_output: Whether to output to console
        reduce_autogen_noise: Whether to reduce AutoGen internal logging noise
    """
    # Create logs directory if it doesn't exist
    Path(log_dir).mkdir(exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S',
        force=True
    )
    
    # If reducing AutoGen noise, set specific loggers to higher levels
    if reduce_autogen_noise:
        # Completely disable AutoGen event system noise (including LLMCall logs)
        logging.getLogger("autogen_core.events").setLevel(logging.CRITICAL)
        logging.getLogger("autogen_core").setLevel(logging.WARNING)
        
        # Reduce HTTP request logs 
        logging.getLogger("httpx").setLevel(logging.WARNING)
        
        # Disable other noisy AutoGen internals
        logging.getLogger("autogen_agentchat").setLevel(logging.WARNING)
        logging.getLogger("autogen_ext").setLevel(logging.WARNING)
        
        # Keep our application logs at the desired level
        logging.getLogger("autogen_agents").setLevel(log_level)
        logging.getLogger("autogen_runtime").setLevel(log_level)
        logging.getLogger("tools").setLevel(log_level)
        
        print("🔇 AutoGen detailed logging disabled - clean output enabled")
    
    # Add file handler for persistent logging
    try:
        log_file = Path(log_dir) / f"autogen_{os.getpid()}.log"
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        )
        
        # Add file handler to root logger
        logging.getLogger().addHandler(file_handler)
        
    except Exception as e:
        logging.warning(f"Could not setup file logging: {e}")


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with consistent naming"""
    return logging.getLogger(name)


def cleanup_old_logs(log_dir: str = "logs", days_to_keep: int = 30) -> None:
    """Clean up old log files"""
    try:
        import time
        from datetime import datetime, timedelta
        
        cutoff = datetime.now() - timedelta(days=days_to_keep)
        log_path = Path(log_dir)
        
        if not log_path.exists():
            return
            
        cleaned = 0
        for log_file in log_path.glob("*.log*"):
            try:
                file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                if file_time < cutoff:
                    log_file.unlink()
                    cleaned += 1
            except Exception:
                continue  # Skip files we can't process
                
        if cleaned > 0:
            logging.info(f"Cleaned up {cleaned} old log files")
            
    except Exception as e:
        logging.warning(f"Error cleaning up logs: {e}")


# For backward compatibility with existing imports
def setup_logging(*args, **kwargs):
    """Backward compatible logging setup"""
    return setup_autogen_logging(*args, **kwargs)