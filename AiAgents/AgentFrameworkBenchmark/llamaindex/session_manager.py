"""
Persistent session management for LlamaIndex workflows.
Handles memory persistence, chat store management, and context serialization.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple
from state import MainState
from llama_index.core.workflow import Context, JsonPickleSerializer
from llama_index.core.memory import Memory
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.memory import ChatMemoryBuffer

logger = logging.getLogger(__name__)


def get_session_directory(session_id: str) -> Path:
    """Get the directory for storing session data."""
    sessions_dir = Path("sessions")
    sessions_dir.mkdir(exist_ok=True)
    session_dir = sessions_dir / session_id
    session_dir.mkdir(exist_ok=True)
    return session_dir


def get_chat_store_path(session_id: str) -> str:
    """Get the file path for chat store persistence."""
    session_dir = get_session_directory(session_id)
    return str(session_dir / "chat_store.json")


def get_context_path(session_id: str) -> str:
    """Get the file path for context persistence."""
    session_dir = get_session_directory(session_id)
    return str(session_dir / "context.json")


def setup_persistent_memory(session_id: str, main_state: MainState) -> Tuple[Optional[ChatMemoryBuffer], Optional[Context]]:
    """
    Set up persistent memory and context for a session.
    Returns (chat_memory, context) if persistent memory is enabled, (None, None) otherwise.
    """
    if not main_state.use_persistent_memory:
        logger.info("Persistent memory disabled - using cache-based memory")
        return None, None
    
    logger.info(f"Setting up persistent memory for session: {session_id}")
    
    # Try to load existing chat store or create new one
    chat_store_path = get_chat_store_path(session_id)
    try:
        if os.path.exists(chat_store_path):
            logger.info(f"Loading existing chat store from: {chat_store_path}")
            chat_store = SimpleChatStore.from_persist_path(chat_store_path)
        else:
            logger.info("Creating new chat store")
            chat_store = SimpleChatStore()
    except Exception as e:
        logger.warning(f"Failed to load chat store, creating new one: {e}")
        chat_store = SimpleChatStore()
    
    chat_memory = ChatMemoryBuffer.from_defaults(
        token_limit=3000,
        chat_store=chat_store,
        chat_store_key=session_id
    )
    
    # Try to load existing context or create new one
    context_path = get_context_path(session_id)
    context = None
    try:
        if os.path.exists(context_path):
            logger.info(f"Loading existing context from: {context_path}")
            import json
            with open(context_path, 'r') as f:
                ctx_dict = json.load(f)
            # Note: We'll need to pass the workflow to restore context
            # For now, we'll create a new context and let it be restored later
            context = None  # Will be restored in run_chat_loop
        else:
            logger.info("Creating new context")
            context = None  # Will be created in run_chat_loop
    except Exception as e:
        logger.warning(f"Failed to load context, will create new one: {e}")
        context = None
    
    return chat_memory, context


def save_persistent_memory(session_id: str, chat_memory: ChatMemoryBuffer, context: Context) -> None:
    """Save persistent memory and context to disk."""
    # Save chat store
    try:
        chat_store_path = get_chat_store_path(session_id)
        chat_memory.chat_store.persist(chat_store_path)
        logger.debug(f"Chat store saved to: {chat_store_path}")
    except Exception as e:
        logger.error(f"Failed to save chat store: {e}")
    
    # Save context
    context_path = get_context_path(session_id)
    
    try:
        # Serialize context to dictionary
        ctx_dict = context.to_dict(serializer=JsonPickleSerializer())
        
        # Write context file
        import json
        with open(context_path, 'w') as f:
            json.dump(ctx_dict, f, indent=2)
        
        logger.debug(f"Context saved to: {context_path}")
        
    except Exception as e:
        logger.error(f"Failed to save context: {type(e).__name__}: {e}")


def restore_context_from_file(workflow, session_id: str) -> Optional[Context]:
    """
    Restore context from saved file.
    Returns None if restoration fails or no saved context exists.
    """
    context_path = get_context_path(session_id)
    
    try:
        if not os.path.exists(context_path):
            logger.info("No saved context found")
            return None
            
        logger.info(f"Attempting to restore context from: {context_path}")
        
        # Try to load and validate the JSON first
        import json
        with open(context_path, 'r') as f:
            ctx_dict = json.load(f)
        
        if not ctx_dict or not isinstance(ctx_dict, dict):
            logger.warning("Context file contains invalid data, will create new context")
            return None
        
        # Attempt context restoration with JsonPickleSerializer
        ctx = Context.from_dict(workflow, ctx_dict, serializer=JsonPickleSerializer())
        logger.info("✅ Successfully restored context from persistent storage")
        return ctx
            
    except Exception as e:
        logger.warning(f"Context restoration failed: {type(e).__name__}: {e}")
        return None


def setup_session_memory(workflow, session_id: str, main_state: MainState) -> Tuple[Optional[ChatMemoryBuffer], Context]:
    """
    Complete setup for session memory including context restoration.
    Returns (chat_memory, context) where context is always valid.
    """
    # Set up persistent memory if enabled
    chat_memory, _ = setup_persistent_memory(session_id, main_state)
    
    # Create or restore context
    ctx = None
    if main_state.use_persistent_memory:
        # Try to restore context from saved state
        ctx = restore_context_from_file(workflow, session_id)
    
    # Create new context if restoration failed or persistent memory is disabled
    if ctx is None:
        ctx = Context(workflow=workflow)
        if main_state.use_persistent_memory:
            logger.info("🆕 Created new context for persistent session")
        else:
            logger.info("🆕 Created new context for cache-based session")
    
    return chat_memory, ctx
