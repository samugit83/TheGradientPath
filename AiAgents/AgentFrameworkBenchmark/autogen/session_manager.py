#!/usr/bin/env python3
"""
Unified Session Manager for AI-powered chat system.
Provides both SQLite-based persistent and simple list-based in-memory session management for AutoGen Core.
Handles both persistent storage and thread-safe in-memory storage with configurable buffer size.
"""

import sqlite3
import json
import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import threading
import asyncio

from autogen_core.models import (
    LLMMessage,
    SystemMessage,
    UserMessage as CoreUserMessage,
    AssistantMessage,
    FunctionExecutionResultMessage,
)
from autogen_core.model_context import BufferedChatCompletionContext

logger = logging.getLogger(__name__)


class MessageRole(Enum):
    """Enum for message roles in conversation"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"
    ERROR = "error"


class ToolStatus(Enum):
    """Status of tool execution"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


@dataclass
class ToolCall:
    """Structure for tool call information"""
    tool_name: str
    tool_id: str
    arguments: Dict[str, Any]
    status: ToolStatus
    result: Optional[str] = None
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    duration_ms: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "tool_name": self.tool_name,
            "tool_id": self.tool_id,
            "arguments": self.arguments,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms
        }


@dataclass
class Message:
    """
    Comprehensive message structure for session history.
    Compatible with AutoGen Core LLMMessage schema while maintaining extensibility.
    """
    role: MessageRole
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    message_id: Optional[str] = None
    
    # Tool-related fields
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_call_id: Optional[str] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Parent message for threading
    parent_message_id: Optional[str] = None
    
    # Source field for AutoGen compatibility
    source: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "message_id": self.message_id,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "tool_call_id": self.tool_call_id,
            "metadata": self.metadata,
            "parent_message_id": self.parent_message_id,
            "source": self.source
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create Message from dictionary"""
        # Handle role
        role = MessageRole(data["role"])
        
        # Handle tool calls
        tool_calls = []
        if data.get("tool_calls"):
            for tc_data in data["tool_calls"]:
                tool_calls.append(ToolCall(
                    tool_name=tc_data["tool_name"],
                    tool_id=tc_data["tool_id"],
                    arguments=tc_data["arguments"],
                    status=ToolStatus(tc_data["status"]),
                    result=tc_data.get("result"),
                    error=tc_data.get("error"),
                    timestamp=tc_data["timestamp"],
                    duration_ms=tc_data.get("duration_ms")
                ))
        
        return cls(
            role=role,
            content=data["content"],
            timestamp=data["timestamp"],
            message_id=data.get("message_id"),
            tool_calls=tool_calls,
            tool_call_id=data.get("tool_call_id"),
            metadata=data.get("metadata", {}),
            parent_message_id=data.get("parent_message_id"),
            source=data.get("source")
        )
    
    @classmethod
    def from_llm_message(cls, llm_message: LLMMessage) -> 'Message':
        """Create Message from AutoGen LLMMessage"""
        # Map AutoGen message types to our roles
        role_mapping = {
            SystemMessage: MessageRole.SYSTEM,
            CoreUserMessage: MessageRole.USER,
            AssistantMessage: MessageRole.ASSISTANT,
            FunctionExecutionResultMessage: MessageRole.TOOL,
        }
        
        role = role_mapping.get(type(llm_message), MessageRole.ASSISTANT)
        source = getattr(llm_message, 'source', None)
        
        return cls(
            role=role,
            content=str(llm_message.content),
            source=source
        )
    
    def to_llm_message(self) -> LLMMessage:
        """Convert to AutoGen LLMMessage format"""
        if self.role == MessageRole.USER:
            return CoreUserMessage(content=self.content, source=self.source or "user")
        elif self.role == MessageRole.SYSTEM:
            return SystemMessage(content=self.content)
        elif self.role == MessageRole.ASSISTANT:
            return AssistantMessage(content=self.content, source=self.source or "assistant")
        elif self.role == MessageRole.TOOL:
            # For tool messages, we might need to handle FunctionExecutionResult
            return AssistantMessage(content=self.content, source=self.source or "tool")
        else:
            # Default to assistant message
            return AssistantMessage(content=self.content, source=self.source or "assistant")


class PersistentSession:
    """
    SQLite-based session storage with thread-safe operations.
    Manages conversation history, tool calls, and agent interactions.
    """
    
    def __init__(self, db_path: str, session_id: str):
        """
        Initialize database connection and create tables if needed.
        
        Args:
            db_path: Path to SQLite database file
            session_id: Unique identifier for this session
        """
        self.db_path = db_path
        self.session_id = session_id
        self.lock = threading.Lock()
        
        # Create database and tables
        self._init_database()
        
        logger.info(f"Initialized PersistentSession: {db_path} (session: {session_id})")
    
    def _init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata TEXT
                )
            """)
            
            # Messages table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    message_id TEXT UNIQUE,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    tool_calls TEXT,
                    tool_call_id TEXT,
                    metadata TEXT,
                    parent_message_id TEXT,
                    sequence_number INTEGER,
                    source TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)
            
            # Create indexes for performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_session 
                ON messages(session_id, sequence_number)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_timestamp 
                ON messages(session_id, timestamp)
            """)
            
            # Initialize session if not exists
            now = datetime.now().isoformat()
            cursor.execute("""
                INSERT OR IGNORE INTO sessions (session_id, created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?)
            """, (self.session_id, now, now, json.dumps({})))
            
            conn.commit()
    
    def add_message(self, message: Message) -> int:
        """
        Add a message to the session history.
        
        Args:
            message: Message object to store
            
        Returns:
            Database row ID of the inserted message
        """
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get next sequence number
                cursor.execute("""
                    SELECT MAX(sequence_number) FROM messages WHERE session_id = ?
                """, (self.session_id,))
                result = cursor.fetchone()
                seq_num = (result[0] + 1) if result[0] is not None else 1
                
                # Generate message_id if not provided
                if not message.message_id:
                    message.message_id = f"{self.session_id}_{seq_num}_{datetime.now().timestamp()}"
                
                # Insert message
                cursor.execute("""
                    INSERT INTO messages (
                        session_id, message_id, role, content, timestamp,
                        tool_calls, tool_call_id,
                        metadata, parent_message_id, sequence_number, source
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    self.session_id,
                    message.message_id,
                    message.role.value,
                    message.content,
                    message.timestamp,
                    json.dumps([tc.to_dict() for tc in message.tool_calls]) if message.tool_calls else None,
                    message.tool_call_id,
                    json.dumps(message.metadata) if message.metadata else None,
                    message.parent_message_id,
                    seq_num,
                    message.source
                ))
                
                # Update session updated_at
                cursor.execute("""
                    UPDATE sessions SET updated_at = ? WHERE session_id = ?
                """, (datetime.now().isoformat(), self.session_id))
                
                conn.commit()
                
                logger.debug(f"Added message: {message.role.value} (seq: {seq_num})")
                return cursor.lastrowid
    
    def get_messages(self, limit: Optional[int] = None, offset: int = 0) -> List[Message]:
        """
        Retrieve messages from session history.
        
        Args:
            limit: Maximum number of messages to retrieve (None for all)
            offset: Number of messages to skip
            
        Returns:
            List of Message objects
        """
        try:
            with self.lock:
                with sqlite3.connect(self.db_path, timeout=5.0) as conn:  # Add 5 second timeout
                    cursor = conn.cursor()
                
                query = """
                    SELECT message_id, role, content, timestamp, tool_calls,
                           tool_call_id, metadata,
                           parent_message_id, source
                    FROM messages
                    WHERE session_id = ?
                    ORDER BY sequence_number ASC
                """
                
                if limit is not None:
                    query += f" LIMIT {limit} OFFSET {offset}"
                
                cursor.execute(query, (self.session_id,))
                rows = cursor.fetchall()
                
                messages = []
                for row in rows:
                    data = {
                        "message_id": row[0],
                        "role": row[1],
                        "content": row[2],
                        "timestamp": row[3],
                        "tool_calls": self._safe_json_loads(row[4], []),
                        "tool_call_id": row[5],
                        "metadata": self._safe_json_loads(row[6], {}),
                        "parent_message_id": row[7],
                        "source": row[8]
                    }
                    messages.append(Message.from_dict(data))
                
                return messages
        except Exception as e:
            logger.error(f"Error getting messages: {e}")
            return []  # Return empty list on error
    
    def get_recent_messages(self, count: int = 10) -> List[Message]:
        """
        Get the most recent messages.
        
        Args:
            count: Number of recent messages to retrieve
            
        Returns:
            List of Message objects
        """
        try:
            with self.lock:
                with sqlite3.connect(self.db_path, timeout=5.0) as conn:  # Add 5 second timeout
                    cursor = conn.cursor()
                    
                    # Get total count
                    cursor.execute("""
                        SELECT COUNT(*) FROM messages WHERE session_id = ?
                    """, (self.session_id,))
                    total = cursor.fetchone()[0]
                    
                    # Calculate offset for last N messages
                    offset = max(0, total - count)
                    
                    # Get messages directly here to avoid recursive lock
                    query = """
                        SELECT message_id, role, content, timestamp, tool_calls,
                               tool_call_id, metadata,
                               parent_message_id, source
                        FROM messages
                        WHERE session_id = ?
                        ORDER BY sequence_number ASC
                        LIMIT ? OFFSET ?
                    """
                    
                    cursor.execute(query, (self.session_id, count, offset))
                    rows = cursor.fetchall()
                    
                    messages = []
                    for row in rows:
                        data = {
                            "message_id": row[0],
                            "role": row[1],
                            "content": row[2],
                            "timestamp": row[3],
                            "tool_calls": self._safe_json_loads(row[4], []),
                            "tool_call_id": row[5],
                            "metadata": self._safe_json_loads(row[6], {}),
                            "parent_message_id": row[7],
                            "source": row[8]
                        }
                        messages.append(Message.from_dict(data))
                    
                    return messages
        except Exception as e:
            logger.error(f"Error getting recent messages: {e}")
            return []  # Return empty list on error
    
    def clear_session(self):
        """Clear all messages in the current session"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    DELETE FROM messages WHERE session_id = ?
                """, (self.session_id,))
                conn.commit()
                logger.info(f"Cleared session: {self.session_id}")
    
    def _safe_json_loads(self, json_str, default):
        """Safely load JSON with fallback to default value"""
        if not json_str:
            return default
        try:
            return json.loads(json_str)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Failed to parse JSON from database: {e}")
            return default
    
    def get_session_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current session.
        
        Returns:
            Dictionary with session statistics
        """
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get message count by role
                cursor.execute("""
                    SELECT role, COUNT(*) FROM messages 
                    WHERE session_id = ?
                    GROUP BY role
                """, (self.session_id,))
                role_counts = dict(cursor.fetchall())
                
                # Get tool call count
                cursor.execute("""
                    SELECT COUNT(*) FROM messages
                    WHERE session_id = ? AND tool_calls IS NOT NULL
                """, (self.session_id,))
                tool_call_count = cursor.fetchone()[0]
                
                return {
                    "session_id": self.session_id,
                    "message_counts": role_counts,
                    "total_messages": sum(role_counts.values()),
                    "tool_calls": tool_call_count
                }
    




class InMemorySession:
    """
    Simple in-memory session using AutoGen's BufferedChatCompletionContext directly.
    This is completely separate from PersistentSession - no conversion, just direct usage.
    """
    
    def __init__(self, session_id: str, buffer_size: int = 10):
        """Initialize with AutoGen's BufferedChatCompletionContext"""
        self.session_id = session_id
        self.buffer_size = buffer_size
        self.context = BufferedChatCompletionContext(buffer_size=buffer_size)
        
        logger.info(f"Initialized InMemorySession with BufferedChatCompletionContext: {session_id} (buffer_size: {buffer_size})")
    
    # Direct methods that work with LLMMessage - no conversion
    async def add_session_message(self, llm_message: LLMMessage) -> None:
        """Add an LLMMessage directly to the context"""
        await self.context.add_message(llm_message)
        logger.debug(f"Added LLM message to buffer: {type(llm_message).__name__}")
    
    async def get_llm_messages(self) -> List[LLMMessage]:
        """Get LLM messages directly from context"""
        return await self.context.get_messages()
    
    async def clear(self) -> None:
        """Clear the context"""
        await self.context.clear()
        logger.info(f"Cleared in-memory session: {self.session_id}")
    
    # Simple stats without conversion
    async def get_stats(self) -> Dict[str, Any]:
        """Get simple stats"""
        messages = await self.context.get_messages()
        return {
            "session_id": self.session_id,
            "total_messages": len(messages),
            "buffer_size": self.buffer_size
        }
    




class SessionManager:
    """
    Unified session manager that handles both persistent and in-memory storage.
    Persistent: Uses our Message format with database storage
    In-memory: Uses LLMMessage format with AutoGen's BufferedChatCompletionContext
    """
    
    def __init__(self, session_id: str, use_persistent_memory: bool = True, db_path: str = "conversations.db", buffer_size: int = 10):
        """Initialize session manager with different storage backends"""
        self.session_id = session_id
        self.use_persistent_memory = use_persistent_memory
        
        if use_persistent_memory:
            self.persistent_session = PersistentSession(db_path, session_id)
            self.memory_session = None
            logger.info(f"SessionManager initialized with persistent storage - Session: {session_id}")
        else:
            self.persistent_session = None
            self.memory_session = InMemorySession(session_id, buffer_size)
            logger.info(f"SessionManager initialized with in-memory storage - Session: {session_id}")
    
    def _run_async(self, coro):
        """Helper method to run async coroutines for in-memory session"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an event loop, we need to handle this differently
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, coro)
                    return future.result()
            else:
                return loop.run_until_complete(coro)
        except RuntimeError:
            return asyncio.run(coro)
    
    # Main method for working with LLMMessage directly (simplest approach)
    def add_session_message(self, llm_message: LLMMessage) -> int:
        """Add an AutoGen LLMMessage to the session"""
        if self.use_persistent_memory:
            # Convert to our Message format for persistent storage
            message = Message.from_llm_message(llm_message)
            return self.persistent_session.add_message(message)
        else:
            # Use directly with BufferedChatCompletionContext
            self._run_async(self.memory_session.add_session_message(llm_message))
            return 1
    
    def get_recent_session_messages(self, count: int = 10) -> List[LLMMessage]:
        """Get recent LLM messages - works differently for each backend"""
        if self.use_persistent_memory:
            # Convert from our Message format
            messages = self.persistent_session.get_recent_messages(count)
            return [msg.to_llm_message() for msg in messages]
        else:
            # Get directly from BufferedChatCompletionContext
            all_messages = self._run_async(self.memory_session.get_llm_messages())
            return all_messages[-count:] if len(all_messages) > count else all_messages
    
    def clear_session(self):
        """Clear all messages in the current session"""
        if self.use_persistent_memory:
            self.persistent_session.clear_session()
        else:
            self._run_async(self.memory_session.clear())
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics about the current session"""
        if self.use_persistent_memory:
            return self.persistent_session.get_session_stats()
        else:
            return self._run_async(self.memory_session.get_stats())
    
    @property
    def context(self) -> Optional[BufferedChatCompletionContext]:
        """Get direct access to BufferedChatCompletionContext for in-memory sessions"""
        if not self.use_persistent_memory:
            return self.memory_session.context
        return None


def init_session_manager(session_id: str, use_persistent_memory: bool = True, db_path: str = "conversations.db", buffer_size: int = 10) -> SessionManager:
    """
    Initialize session manager.
    
    Args:
        session_id: Session identifier
        use_persistent_memory: Whether to use persistent storage or in-memory
        db_path: Path to database file (for persistent mode)
        buffer_size: Buffer size for in-memory mode
        
    Returns:
        SessionManager instance
    """
    return SessionManager(session_id, use_persistent_memory, db_path, buffer_size)


# Backward compatibility function
def init_persistent_session_manager(session_id: str, db_path: str = "conversations.db") -> SessionManager:
    """Initialize persistent session manager (backward compatibility)"""
    return init_session_manager(session_id, use_persistent_memory=True, db_path=db_path)


# Global context storage
_contexts = {}

def set_global_context(session_id: str, session_manager=None, runtime=None):
    """Set global context for session_manager and runtime"""
    _contexts[session_id] = {
        'session_manager': session_manager,
        'runtime': runtime
    }

def get_global_session_manager(session_id: str):
    """Get session manager from global context"""
    return _contexts.get(session_id, {}).get('session_manager')

def get_global_runtime(session_id: str):
    """Get runtime from global context"""
    return _contexts.get(session_id, {}).get('runtime')

def cleanup_global_context(session_id: str):
    """Remove context for session"""
    if session_id in _contexts:
        del _contexts[session_id]
