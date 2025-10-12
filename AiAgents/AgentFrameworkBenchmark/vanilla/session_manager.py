#!/usr/bin/env python3
"""
Unified Session Manager for AI-powered chat system.
Provides both in-memory and persistent session management with a unified interface.
"""

import sqlite3
import json
import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from enum import Enum
import threading

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
    Captures all aspects of conversation including tool calls and agent interactions.
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
            "parent_message_id": self.parent_message_id
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
            parent_message_id=data.get("parent_message_id")
        )


class InMemorySession:
    """Simple in-memory session storage for conversation history"""
    
    def __init__(self, session_id: str, max_history: int = 40):
        self.session_id = session_id
        self.history: List[Dict] = []
        self.max_history = max_history  # Max messages to keep in history
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
        
        logger.info(f"Initialized InMemorySession: {session_id}")
        
    def add_message(self, content: str, subject: str, task_type: str = "chat"):
        """Add a single message to history
        
        Args:
            content: The message content
            subject: Either "user" or "assistant"
            task_type: Type of interaction (chat, tool_execution, etc.)
        """
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "subject": subject,
            "content": content,
            "type": task_type
        })
        
        # Keep only the last max_history messages
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        
        self.updated_at = datetime.now().isoformat()
    
    def get_history(self, last_n: int = None) -> List[Dict]:
        """Get conversation history"""
        if last_n:
            return self.history[-last_n:]
        return self.history
    
    def get_recent_messages(self, count: int = 10) -> List[Dict]:
        """Get recent messages in a format compatible with persistent session"""
        recent = self.history[-count:] if count < len(self.history) else self.history
        # Convert to format expected by the rest of the system
        messages = []
        for item in recent:
            # Map subject to role
            role_mapping = {
                "user": MessageRole.USER,
                "assistant": MessageRole.ASSISTANT
            }
            role = role_mapping.get(item.get('subject', 'assistant'), MessageRole.ASSISTANT)
            
            message = Message(
                role=role,
                content=item.get('content', ''),
                timestamp=item.get('timestamp', datetime.now().isoformat())
            )
            messages.append(message)
        return messages
    
    def clear(self):
        """Clear all history"""
        self.history = []
        self.updated_at = datetime.now().isoformat()
        logger.info(f"Memory cleared for session: {self.session_id}")
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        user_messages = sum(1 for msg in self.history if msg.get('subject') == 'user')
        assistant_messages = sum(1 for msg in self.history if msg.get('subject') == 'assistant')
        
        return {
            "session_id": self.session_id,
            "message_counts": {
                "user": user_messages,
                "assistant": assistant_messages
            },
            "total_messages": len(self.history),
            "tool_calls": 0
        }


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
                        metadata, parent_message_id, sequence_number
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    seq_num
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
                           parent_message_id
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
                        "tool_calls": json.loads(row[4]) if row[4] else [],
                        "tool_call_id": row[5],
                        "metadata": json.loads(row[6]) if row[6] else {},
                        "parent_message_id": row[7]
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
                               parent_message_id
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
                            "tool_calls": json.loads(row[4]) if row[4] else [],
                            "tool_call_id": row[5],
                            "metadata": json.loads(row[6]) if row[6] else {},
                            "parent_message_id": row[7]
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
    
    def export_session(self, format: str = "json") -> Union[str, List[Dict[str, Any]]]:
        """
        Export session data.
        
        Args:
            format: Export format ('json' or 'list')
            
        Returns:
            Session data in requested format
        """
        messages = self.get_messages()
        message_dicts = [msg.to_dict() for msg in messages]
        
        if format == "json":
            return json.dumps({
                "session_id": self.session_id,
                "messages": message_dicts,
                "stats": self.get_session_stats()
            }, indent=2)
        else:
            return message_dicts
    
    def close(self):
        """Close database connection (for cleanup)"""
        logger.info(f"Closed PersistentSession: {self.session_id}")


class SessionManager:
    """
    Unified session manager that can handle both in-memory and persistent storage.
    Provides a single interface for session management regardless of storage mode.
    """
    
    def __init__(self, session_id: str, use_persistent: bool = False, 
                 db_path: str = "conversations.db", max_history: int = 40):
        """
        Initialize session manager.
        
        Args:
            session_id: Unique identifier for this session
            use_persistent: Whether to use persistent storage (SQLite) or in-memory
            db_path: Path to database file (only used if use_persistent=True)
            max_history: Max messages for in-memory storage (only used if use_persistent=False)
        """
        self.session_id = session_id
        self.use_persistent = use_persistent
        
        if use_persistent:
            self.session = PersistentSession(db_path, session_id)
        else:
            self.session = InMemorySession(session_id, max_history)
        
        logger.info(f"SessionManager initialized - Session: {session_id}, "
                   f"Mode: {'Persistent' if use_persistent else 'In-Memory'}")
    
    def add_user_message(self, content: str, **kwargs) -> int:
        """Add a user message to the session"""
        if self.use_persistent:
            # Extract task_type from kwargs and put it in metadata
            task_type = kwargs.pop("task_type", "chat")
            metadata = kwargs.pop("metadata", {})
            metadata["task_type"] = task_type
            
            message = Message(
                role=MessageRole.USER,
                content=content,
                metadata=metadata,
                **kwargs
            )
            return self.session.add_message(message)
        else:
            self.session.add_message(content, "user", kwargs.get("task_type", "chat"))
            return len(self.session.history)  # Return current length as ID
    
    def add_assistant_message(self, content: str, **kwargs) -> int:
        """Add an assistant message to the session"""
        if self.use_persistent:
            # Extract task_type from kwargs and put it in metadata
            task_type = kwargs.pop("task_type", "chat")
            metadata = kwargs.pop("metadata", {})
            metadata["task_type"] = task_type
            
            message = Message(
                role=MessageRole.ASSISTANT,
                content=content,
                metadata=metadata,
                **kwargs
            )
            return self.session.add_message(message)
        else:
            self.session.add_message(content, "assistant", kwargs.get("task_type", "chat"))
            return len(self.session.history)  # Return current length as ID
    
    def add_tool_message(self, tool_name: str, tool_id: str, arguments: Dict[str, Any], 
                        result: Optional[str] = None, error: Optional[str] = None, **kwargs) -> int:
        """Add a tool call message to the session"""
        if self.use_persistent:
            # Extract task_type from kwargs and put it in metadata
            task_type = kwargs.pop("task_type", "tool_execution")
            metadata = kwargs.pop("metadata", {})
            metadata["task_type"] = task_type
            
            tool_call = ToolCall(
                tool_name=tool_name,
                tool_id=tool_id,
                arguments=arguments,
                status=ToolStatus.SUCCESS if result else ToolStatus.FAILED,
                result=result,
                error=error
            )
            
            message = Message(
                role=MessageRole.TOOL,
                content=result or error or f"Tool {tool_name} executed",
                tool_calls=[tool_call],
                tool_call_id=tool_id,
                metadata=metadata,
                **kwargs
            )
            return self.session.add_message(message)
        else:
            # For in-memory, just add a simple message
            content = result or error or f"Tool {tool_name} executed"
            self.session.add_message(content, "assistant", "tool_execution")
            return len(self.session.history)
    
    def add_agent_message(self, agent_name: str, agent_type: str, content: str, 
                         iteration: Optional[int] = None, **kwargs) -> int:
        """Add an agent message to the session"""
        if self.use_persistent:
            # Extract task_type from kwargs and put it in metadata
            task_type = kwargs.pop("task_type", f"agent_{agent_type}")
            metadata = kwargs.pop("metadata", {})
            metadata["task_type"] = task_type
            
            # Store agent information in metadata
            metadata["agent_name"] = agent_name
            metadata["agent_type"] = agent_type
            if iteration is not None:
                metadata["agent_iteration"] = iteration
            
            message = Message(
                role=MessageRole.ASSISTANT,  # Use ASSISTANT instead of AGENT
                content=content,
                metadata=metadata,
                **kwargs
            )
            return self.session.add_message(message)
        else:
            # For in-memory, just add a simple message with agent prefix
            prefixed_content = f"[{agent_name}] {content}"
            self.session.add_message(prefixed_content, "assistant", f"agent_{agent_type}")
            return len(self.session.history)
    
    def get_conversation_history(self, last_n: int = 20) -> List[Dict]:
        """
        Get conversation history in a format compatible with existing code.
        
        Args:
            last_n: Number of recent messages to retrieve
            
        Returns:
            List of conversation history items with 'subject' and 'content' keys
        """
        if self.use_persistent:
            # Get from persistent storage
            recent_messages = self.session.get_recent_messages(count=last_n)
            history = []
            for msg in recent_messages:
                # Map database roles to expected format (user/assistant)
                role_mapping = {
                    MessageRole.USER: "user",
                    MessageRole.ASSISTANT: "assistant",
                    MessageRole.SYSTEM: "assistant",
                    MessageRole.TOOL: "assistant",
                    MessageRole.ERROR: "assistant"
                }
                subject = role_mapping.get(msg.role, "assistant")
                history.append({
                    "subject": subject,
                    "content": msg.content
                })
            return history
        else:
            # Get from in-memory storage
            return self.session.get_history(last_n=last_n)
    
    def clear_session(self):
        """Clear all messages in the current session"""
        self.session.clear_session() if self.use_persistent else self.session.clear()
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics about the current session"""
        return self.session.get_session_stats()


# Global session instance
_session_manager: Optional[SessionManager] = None


def init_session_manager(session_id: str, use_persistent: bool = False, 
                        db_path: str = "conversations.db", max_history: int = 40) -> SessionManager:
    """
    Initialize global session manager.
    
    Args:
        session_id: Session identifier
        use_persistent: Whether to use persistent storage
        db_path: Path to database file
        max_history: Max messages for in-memory storage
        
    Returns:
        SessionManager instance
    """
    global _session_manager
    _session_manager = SessionManager(session_id, use_persistent, db_path, max_history)
    return _session_manager


def get_session_manager() -> Optional[SessionManager]:
    """Get the global session manager instance"""
    return _session_manager


# Convenience functions for backward compatibility
def init_session(db_path: str = "conversations.db", session_id: str = "default"):
    """Initialize persistent session (backward compatibility)"""
    return init_session_manager(session_id, use_persistent=True, db_path=db_path).session


def get_session():
    """Get the current session (backward compatibility)"""
    if _session_manager and _session_manager.use_persistent:
        return _session_manager.session
    return None


def add_user_message(content: str, **kwargs) -> int:
    """Convenience function to add user message"""
    if _session_manager:
        return _session_manager.add_user_message(content, **kwargs)
    return -1


def add_assistant_message(content: str, **kwargs) -> int:
    """Convenience function to add assistant message"""
    if _session_manager:
        return _session_manager.add_assistant_message(content, **kwargs)
    return -1


def add_tool_message(tool_name: str, tool_id: str, arguments: Dict[str, Any], 
                     result: Optional[str] = None, error: Optional[str] = None, **kwargs) -> int:
    """Convenience function to add tool call message"""
    if _session_manager:
        return _session_manager.add_tool_message(tool_name, tool_id, arguments, result, error, **kwargs)
    return -1


def add_agent_message(agent_name: str, agent_type: str, content: str, 
                     iteration: Optional[int] = None, **kwargs) -> int:
    """Convenience function to add agent message"""
    if _session_manager:
        return _session_manager.add_agent_message(agent_name, agent_type, content, iteration, **kwargs)
    return -1
