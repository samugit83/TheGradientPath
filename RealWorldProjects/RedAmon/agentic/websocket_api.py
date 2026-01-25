"""
WebSocket API for RedAmon Agent

Provides WebSocket endpoint for real-time bidirectional communication with the agent.
Supports streaming of LLM thoughts, tool executions, and interactive approval/question flows.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Optional, Any, Callable
from enum import Enum

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ValidationError


def serialize_for_json(obj):
    """Convert objects to JSON-serializable format, handling datetime objects."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_for_json(item) for item in obj]
    return obj

logger = logging.getLogger(__name__)


# =============================================================================
# MESSAGE TYPE DEFINITIONS
# =============================================================================

class MessageType(str, Enum):
    """WebSocket message types"""
    # Client → Server
    INIT = "init"
    QUERY = "query"
    APPROVAL = "approval"
    ANSWER = "answer"
    PING = "ping"

    # Server → Client
    CONNECTED = "connected"
    THINKING = "thinking"
    THINKING_CHUNK = "thinking_chunk"
    TOOL_START = "tool_start"
    TOOL_OUTPUT_CHUNK = "tool_output_chunk"
    TOOL_COMPLETE = "tool_complete"
    PHASE_UPDATE = "phase_update"
    TODO_UPDATE = "todo_update"
    APPROVAL_REQUEST = "approval_request"
    QUESTION_REQUEST = "question_request"
    RESPONSE = "response"
    EXECUTION_STEP = "execution_step"
    ERROR = "error"
    PONG = "pong"
    TASK_COMPLETE = "task_complete"


# =============================================================================
# CLIENT MESSAGE MODELS
# =============================================================================

class InitMessage(BaseModel):
    """Initialize WebSocket session"""
    user_id: str
    project_id: str
    session_id: str


class QueryMessage(BaseModel):
    """Send query to agent"""
    question: str


class ApprovalMessage(BaseModel):
    """Respond to phase transition approval request"""
    decision: str  # 'approve' | 'modify' | 'abort'
    modification: Optional[str] = None


class AnswerMessage(BaseModel):
    """Answer agent's question"""
    answer: str


# =============================================================================
# WEBSOCKET CONNECTION MANAGER
# =============================================================================

class WebSocketConnection:
    """Manages individual WebSocket connection state"""

    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.user_id: Optional[str] = None
        self.project_id: Optional[str] = None
        self.session_id: Optional[str] = None
        self.authenticated = False
        self.connected_at = datetime.utcnow()
        self.last_ping = datetime.utcnow()

    async def send_message(self, message_type: MessageType, payload: Any):
        """Send JSON message to client"""
        try:
            # Serialize payload to handle datetime objects
            serialized_payload = serialize_for_json(payload)
            message = {
                "type": message_type.value,
                "payload": serialized_payload,
                "timestamp": datetime.utcnow().isoformat()
            }
            await self.websocket.send_json(message)
            logger.debug(f"Sent {message_type.value} message to {self.session_id}")
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            raise

    def get_key(self) -> Optional[str]:
        """Get unique key for this connection"""
        if self.authenticated:
            return f"{self.user_id}:{self.project_id}:{self.session_id}"
        return None


class WebSocketManager:
    """Manages active WebSocket connections"""

    def __init__(self):
        # Map of session_key → WebSocketConnection
        self.active_connections: Dict[str, WebSocketConnection] = {}
        self.lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> WebSocketConnection:
        """Accept new WebSocket connection"""
        await websocket.accept()
        connection = WebSocketConnection(websocket)
        logger.info(f"WebSocket connection accepted from {websocket.client}")
        return connection

    async def authenticate(
        self,
        connection: WebSocketConnection,
        user_id: str,
        project_id: str,
        session_id: str
    ):
        """Authenticate and register connection"""
        async with self.lock:
            connection.user_id = user_id
            connection.project_id = project_id
            connection.session_id = session_id
            connection.authenticated = True

            session_key = connection.get_key()

            # Close existing connection for this session if any
            if session_key in self.active_connections:
                old_conn = self.active_connections[session_key]
                try:
                    await old_conn.websocket.close(code=1000, reason="New connection established")
                except Exception as e:
                    logger.warning(f"Error closing old connection: {e}")

            self.active_connections[session_key] = connection
            logger.info(f"Authenticated session: {session_key}")

    async def disconnect(self, connection: WebSocketConnection):
        """Remove connection from active connections"""
        async with self.lock:
            session_key = connection.get_key()
            if session_key and session_key in self.active_connections:
                del self.active_connections[session_key]
                logger.info(f"Disconnected session: {session_key}")

    def get_connection(self, user_id: str, project_id: str, session_id: str) -> Optional[WebSocketConnection]:
        """Get active connection by session identifiers"""
        session_key = f"{user_id}:{project_id}:{session_id}"
        return self.active_connections.get(session_key)

    def get_connection_count(self) -> int:
        """Get number of active connections"""
        return len(self.active_connections)


# =============================================================================
# STREAMING CALLBACK INTERFACE
# =============================================================================

class StreamingCallback:
    """Callback interface for streaming events from orchestrator"""

    def __init__(self, connection: WebSocketConnection):
        self.connection = connection
        # Response and task_complete are truly once-per-session events
        self._task_complete_sent = False
        self._response_sent = False

    async def on_thinking(self, iteration: int, phase: str, thought: str, reasoning: str):
        """Called when agent starts thinking"""
        await self.connection.send_message(MessageType.THINKING, {
            "iteration": iteration,
            "phase": phase,
            "thought": thought,
            "reasoning": reasoning
        })

    async def on_thinking_chunk(self, chunk: str):
        """Called during LLM generation for streaming thoughts"""
        await self.connection.send_message(MessageType.THINKING_CHUNK, {
            "chunk": chunk
        })

    async def on_tool_start(self, tool_name: str, tool_args: dict):
        """Called when tool execution starts"""
        await self.connection.send_message(MessageType.TOOL_START, {
            "tool_name": tool_name,
            "tool_args": tool_args
        })

    async def on_tool_output_chunk(self, tool_name: str, chunk: str, is_final: bool = False):
        """Called when tool outputs data chunk"""
        await self.connection.send_message(MessageType.TOOL_OUTPUT_CHUNK, {
            "tool_name": tool_name,
            "chunk": chunk,
            "is_final": is_final
        })

    async def on_tool_complete(
        self,
        tool_name: str,
        success: bool,
        output_summary: str,
        actionable_findings: list = None,
        recommended_next_steps: list = None,
    ):
        """Called when tool execution completes"""
        await self.connection.send_message(MessageType.TOOL_COMPLETE, {
            "tool_name": tool_name,
            "success": success,
            "output_summary": output_summary,
            "actionable_findings": actionable_findings or [],
            "recommended_next_steps": recommended_next_steps or [],
        })

    async def on_phase_update(self, current_phase: str, iteration_count: int):
        """Called when phase changes"""
        await self.connection.send_message(MessageType.PHASE_UPDATE, {
            "current_phase": current_phase,
            "iteration_count": iteration_count
        })

    async def on_todo_update(self, todo_list: list):
        """Called when todo list is updated"""
        await self.connection.send_message(MessageType.TODO_UPDATE, {
            "todo_list": todo_list
        })

    async def on_approval_request(self, approval_request: dict):
        """Called when agent requests phase transition approval"""
        # Deduplication is handled by orchestrator using _emitted_approval marker in state
        await self.connection.send_message(MessageType.APPROVAL_REQUEST, approval_request)
        logger.info(f"Approval request sent to session {self.connection.session_id}")

    async def on_question_request(self, question_request: dict):
        """Called when agent asks user a question"""
        # Deduplication is handled by orchestrator using _emitted_question marker in state
        await self.connection.send_message(MessageType.QUESTION_REQUEST, question_request)
        logger.info(f"Question request sent to session {self.connection.session_id}")

    async def on_response(self, answer: str, iteration_count: int, phase: str, task_complete: bool):
        """Called when agent provides final response"""
        if not self._response_sent:
            await self.connection.send_message(MessageType.RESPONSE, {
                "answer": answer,
                "iteration_count": iteration_count,
                "phase": phase,
                "task_complete": task_complete
            })
            self._response_sent = True
            logger.info(f"Response sent to session {self.connection.session_id}")
        else:
            logger.debug(f"Duplicate response blocked for session {self.connection.session_id}")

    async def on_execution_step(self, step: dict):
        """Called after each execution step"""
        await self.connection.send_message(MessageType.EXECUTION_STEP, step)

    async def on_error(self, error_message: str, recoverable: bool = True):
        """Called when error occurs"""
        await self.connection.send_message(MessageType.ERROR, {
            "message": error_message,
            "recoverable": recoverable
        })

    async def on_task_complete(self, message: str, final_phase: str, total_iterations: int):
        """Called when task is complete"""
        if not self._task_complete_sent:
            await self.connection.send_message(MessageType.TASK_COMPLETE, {
                "message": message,
                "final_phase": final_phase,
                "total_iterations": total_iterations
            })
            self._task_complete_sent = True
            logger.info(f"Task complete sent to session {self.connection.session_id}")
        else:
            logger.debug(f"Duplicate task_complete blocked for session {self.connection.session_id}")


# =============================================================================
# MESSAGE HANDLERS
# =============================================================================

class WebSocketHandler:
    """Handles WebSocket messages and routes to orchestrator"""

    def __init__(self, orchestrator, ws_manager: WebSocketManager):
        self.orchestrator = orchestrator
        self.ws_manager = ws_manager

    async def handle_init(self, connection: WebSocketConnection, payload: dict):
        """Handle session initialization"""
        try:
            init_msg = InitMessage(**payload)

            # Authenticate connection
            await self.ws_manager.authenticate(
                connection,
                init_msg.user_id,
                init_msg.project_id,
                init_msg.session_id
            )

            # Send connected confirmation
            await connection.send_message(MessageType.CONNECTED, {
                "session_id": init_msg.session_id,
                "message": "WebSocket connection established",
                "timestamp": datetime.utcnow().isoformat()
            })

            logger.info(f"Session initialized: {init_msg.session_id}")

        except ValidationError as e:
            logger.error(f"Invalid init message: {e}")
            await connection.send_message(MessageType.ERROR, {
                "message": "Invalid initialization message",
                "recoverable": False
            })

    async def handle_query(self, connection: WebSocketConnection, payload: dict):
        """Handle user query"""
        try:
            query_msg = QueryMessage(**payload)

            if not connection.authenticated:
                await connection.send_message(MessageType.ERROR, {
                    "message": "Not authenticated. Send init message first.",
                    "recoverable": False
                })
                return

            # Create streaming callback
            callback = StreamingCallback(connection)

            # Execute query with streaming
            logger.info(f"Processing query for session {connection.session_id}: {query_msg.question[:50]}...")

            # Call orchestrator with streaming callback
            # This will be implemented in Phase 2
            result = await self.orchestrator.invoke_with_streaming(
                question=query_msg.question,
                user_id=connection.user_id,
                project_id=connection.project_id,
                session_id=connection.session_id,
                streaming_callback=callback
            )

            logger.info(f"Query completed for session {connection.session_id}")

        except ValidationError as e:
            logger.error(f"Invalid query message: {e}")
            await connection.send_message(MessageType.ERROR, {
                "message": "Invalid query format",
                "recoverable": True
            })
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            await connection.send_message(MessageType.ERROR, {
                "message": f"Error processing query: {str(e)}",
                "recoverable": True
            })

    async def handle_approval(self, connection: WebSocketConnection, payload: dict):
        """Handle approval response"""
        try:
            approval_msg = ApprovalMessage(**payload)

            if not connection.authenticated:
                await connection.send_message(MessageType.ERROR, {
                    "message": "Not authenticated",
                    "recoverable": False
                })
                return

            # Create streaming callback
            callback = StreamingCallback(connection)

            # Resume orchestrator after approval
            logger.info(f"Processing approval for session {connection.session_id}: {approval_msg.decision}")

            result = await self.orchestrator.resume_after_approval_with_streaming(
                session_id=connection.session_id,
                user_id=connection.user_id,
                project_id=connection.project_id,
                decision=approval_msg.decision,
                modification=approval_msg.modification,
                streaming_callback=callback
            )

            logger.info(f"Approval processed for session {connection.session_id}")

        except ValidationError as e:
            logger.error(f"Invalid approval message: {e}")
            await connection.send_message(MessageType.ERROR, {
                "message": "Invalid approval format",
                "recoverable": True
            })
        except Exception as e:
            logger.error(f"Error processing approval: {e}")
            await connection.send_message(MessageType.ERROR, {
                "message": f"Error processing approval: {str(e)}",
                "recoverable": True
            })

    async def handle_answer(self, connection: WebSocketConnection, payload: dict):
        """Handle answer to agent question"""
        try:
            answer_msg = AnswerMessage(**payload)

            if not connection.authenticated:
                await connection.send_message(MessageType.ERROR, {
                    "message": "Not authenticated",
                    "recoverable": False
                })
                return

            # Create streaming callback
            callback = StreamingCallback(connection)

            # Resume orchestrator after answer
            logger.info(f"Processing answer for session {connection.session_id}")

            result = await self.orchestrator.resume_after_answer_with_streaming(
                session_id=connection.session_id,
                user_id=connection.user_id,
                project_id=connection.project_id,
                answer=answer_msg.answer,
                streaming_callback=callback
            )

            logger.info(f"Answer processed for session {connection.session_id}")

        except ValidationError as e:
            logger.error(f"Invalid answer message: {e}")
            await connection.send_message(MessageType.ERROR, {
                "message": "Invalid answer format",
                "recoverable": True
            })
        except Exception as e:
            logger.error(f"Error processing answer: {e}")
            await connection.send_message(MessageType.ERROR, {
                "message": f"Error processing answer: {str(e)}",
                "recoverable": True
            })

    async def handle_ping(self, connection: WebSocketConnection, payload: dict):
        """Handle ping for keep-alive"""
        connection.last_ping = datetime.utcnow()
        await connection.send_message(MessageType.PONG, {})
        logger.debug(f"Pong sent to session {connection.session_id}")

    async def handle_message(self, connection: WebSocketConnection, raw_message: str):
        """Route incoming message to appropriate handler"""
        try:
            message = json.loads(raw_message)
            msg_type = message.get("type")
            payload = message.get("payload", {})

            if msg_type == MessageType.INIT:
                await self.handle_init(connection, payload)
            elif msg_type == MessageType.QUERY:
                await self.handle_query(connection, payload)
            elif msg_type == MessageType.APPROVAL:
                await self.handle_approval(connection, payload)
            elif msg_type == MessageType.ANSWER:
                await self.handle_answer(connection, payload)
            elif msg_type == MessageType.PING:
                await self.handle_ping(connection, payload)
            else:
                logger.warning(f"Unknown message type: {msg_type}")
                await connection.send_message(MessageType.ERROR, {
                    "message": f"Unknown message type: {msg_type}",
                    "recoverable": True
                })

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON message: {e}")
            await connection.send_message(MessageType.ERROR, {
                "message": "Invalid JSON format",
                "recoverable": True
            })
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            await connection.send_message(MessageType.ERROR, {
                "message": f"Internal error: {str(e)}",
                "recoverable": True
            })


# =============================================================================
# WEBSOCKET ENDPOINT
# =============================================================================

async def websocket_endpoint(
    websocket: WebSocket,
    orchestrator,
    ws_manager: WebSocketManager
):
    """
    Main WebSocket endpoint for agent communication.

    Handles connection lifecycle, message routing, and error handling.
    """
    connection = await ws_manager.connect(websocket)
    handler = WebSocketHandler(orchestrator, ws_manager)

    try:
        while True:
            # Receive message from client
            message_data = await websocket.receive()

            # Check for disconnect event
            if message_data.get("type") == "websocket.disconnect":
                logger.info(f"WebSocket disconnect received: {connection.get_key() or 'unauthenticated'}")
                break

            # Handle different message types
            if "text" in message_data:
                raw_message = message_data["text"]
            elif "bytes" in message_data:
                # Convert bytes to string if sent as binary
                raw_message = message_data["bytes"].decode("utf-8")
            else:
                logger.warning(f"Received unexpected message type: {message_data}")
                continue

            # Handle message
            await handler.handle_message(connection, raw_message)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {connection.get_key() or 'unauthenticated'}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await connection.send_message(MessageType.ERROR, {
                "message": f"Fatal error: {str(e)}",
                "recoverable": False
            })
        except:
            pass
    finally:
        await ws_manager.disconnect(connection)
