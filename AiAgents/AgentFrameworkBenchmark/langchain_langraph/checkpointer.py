"""
Checkpointer Module for LangGraph Persistent Storage

This module provides a Checkpointer class that initializes and manages
persistent storage using SQLite through LangGraph's checkpointer system.
"""

import os
import sqlite3
from pathlib import Path
from langgraph.checkpoint.sqlite import SqliteSaver
import logging

logger = logging.getLogger(__name__)


class Checkpointer:
    """
    Checkpointer class for managing persistent storage with SQLite.
    
    This class provides a simple interface to initialize and manage
    persistent memory storage for LangGraph applications using SQLite checkpointer.
    """
    
    @staticmethod
    def initialize(db_path: str = None):
        """
        Initialize Checkpointer with SQLite checkpointer
        
        Args:
            db_path (str, optional): Path to SQLite database file. 
                                   If None, uses './memory/langgraph_checkpoints.db'
            
        Returns:
            SqliteSaver: Configured SQLite checkpointer instance
        """
        try:
            # Set default database path if not provided
            if db_path is None:
                # Create a memory directory if it doesn't exist
                memory_dir = Path("./memory")
                memory_dir.mkdir(exist_ok=True)
                db_path = str(memory_dir / "langgraph_checkpoints.db")
            
            # Ensure the directory exists
            db_dir = Path(db_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Initializing SQLite checkpointer at: {db_path}")
            
            # Create SQLite connection with thread safety enabled
            conn = sqlite3.connect(
                db_path,
                check_same_thread=False,  # Allow connection to be used across threads
                timeout=30.0  # Add timeout to prevent deadlocks
            )
            
            # Enable WAL mode for better concurrent access
            conn.execute("PRAGMA journal_mode=WAL")
            conn.commit()
            
            # Create the SQLite checkpointer with the connection
            checkpointer = SqliteSaver(conn)
            
            logger.info("SQLite checkpointer initialized successfully")
            return checkpointer
            
        except Exception as e:
            logger.error(f"Failed to initialize SQLite checkpointer: {e}")
            raise RuntimeError(f"Database initialization failed: {e}")
