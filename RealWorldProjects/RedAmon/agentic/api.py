"""
RedAmon Agent WebSocket API

FastAPI application providing WebSocket endpoint for real-time agent communication.
Supports session-based conversation continuity and phase-based approval flow.

Endpoints:
    WS /ws/agent - WebSocket endpoint for real-time bidirectional streaming
    GET /health - Health check
"""
 
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from logging_config import setup_logging
from orchestrator import AgentOrchestrator
from utils import get_session_count
from websocket_api import WebSocketManager, websocket_endpoint

# Initialize logging with file rotation
setup_logging(log_level=logging.INFO, log_to_console=True, log_to_file=True)
logger = logging.getLogger(__name__)

orchestrator: Optional[AgentOrchestrator] = None
ws_manager: Optional[WebSocketManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Initializes the orchestrator and WebSocket manager on startup and cleans up on shutdown.
    """
    global orchestrator, ws_manager

    logger.info("Starting RedAmon Agent API...")

    # Initialize orchestrator
    orchestrator = AgentOrchestrator()
    await orchestrator.initialize()

    # Initialize WebSocket manager
    ws_manager = WebSocketManager()

    logger.info("RedAmon Agent API ready (WebSocket)")

    yield

    logger.info("Shutting down RedAmon Agent API...")
    if orchestrator:
        await orchestrator.close()


app = FastAPI(
    title="RedAmon Agent API",
    description="WebSocket API for real-time agent communication with phase tracking, MCP tools, and Neo4j integration",
    version="3.0.0",
    lifespan=lifespan
)

# Add CORS middleware for webapp (allow all origins for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # Must be False when allow_origins is ["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# RESPONSE MODELS (for /health endpoint only)
# =============================================================================

class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    version: str
    tools_loaded: int
    active_sessions: int


# =============================================================================
# ENDPOINTS
# =============================================================================


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """
    Health check endpoint.

    Returns the API status, version, number of loaded tools, and active sessions.
    """
    tools_count = 0
    if orchestrator and orchestrator.tool_executor:
        tools_count = len(orchestrator.tool_executor.get_all_tools())

    sessions_count = get_session_count()

    return HealthResponse(
        status="ok" if orchestrator and orchestrator._initialized else "initializing",
        version="3.0.0",
        tools_loaded=tools_count,
        active_sessions=sessions_count
    )


@app.websocket("/ws/agent")
async def agent_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time agent communication.

    Provides bidirectional streaming of:
    - LLM thinking process
    - Tool executions and outputs
    - Phase transitions
    - Approval requests
    - Agent questions
    - Todo list updates

    The client must send an 'init' message first to authenticate the session.
    """
    if not orchestrator:
        await websocket.close(code=1011, reason="Orchestrator not initialized")
        return

    if not ws_manager:
        await websocket.close(code=1011, reason="WebSocket manager not initialized")
        return

    await websocket_endpoint(websocket, orchestrator, ws_manager)
