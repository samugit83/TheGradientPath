# protocol_types.py  ·  Python ≥ 3.11 · Pydantic v2
# Client-side subset of protocol types, focusing on WebSocket communication.
from __future__ import annotations
from typing import Any, List, Union, Optional, Literal, Annotated
from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# 1.  RPCs for tool-calling (via WebSocket)                                   #
# --------------------------------------------------------------------------- #
class ToolCallRequestParams(BaseModel):
    name: str
    arguments: dict[str, Any]


class ToolCallRequest(BaseModel):
    method: Literal["tools/call"]                # ← discriminator to dont create confusion on type recognitions
    params: ToolCallRequestParams


class ToolCallResult(BaseModel):
    type: Literal["tool"] = "tool"      # ← discriminator to dont create confusion on type recognitions
    result: Any

class ProcessUpdate(BaseModel):
    """A generic message sent from the server to the client about the process update."""
    message: str


# --------------------------------------------------------------------------- #
# 2.  Capabilities handshake & Initialization (WebSocket)                     #
# --------------------------------------------------------------------------- #
class ToolCapability(BaseModel):
    name: str


# Added PromptCapability for completeness as it's in ServerCapabilities
class PromptCapability(BaseModel):
    name: str


class ServerCapabilities(BaseModel):
    tools: List[ToolCapability] = []
    prompts: List[PromptCapability] = []
    resources: List[ResourceCapability] = []


class InitializationOptions(BaseModel):
    """Server-side options, included here as it's part of SessionMessage."""
    server_name: str
    server_version: str
    capabilities: ServerCapabilities
    instructions: str


# --------------------------------------------------------------------------- #
# 3-A.  Generic RPCs (via WebSocket)                                          #
# --------------------------------------------------------------------------- #
# HealthRequest/Result and InfoRequest/Result removed as they are REST-only.

class ToolsListRequest(BaseModel):
    method: Literal["tools/list"]
    params: Optional[dict] = None


class _BareToolSpec(BaseModel):
    name: str
    description: str
    input_schema: dict
    annotations: Optional[dict] = None


class ToolsListResult(BaseModel):
    tools: List[_BareToolSpec]


class PromptCallRequestParams(BaseModel):
    name: str
    arguments: dict[str, Any]

class PromptCallRequest(BaseModel):
    method: Literal["prompt/call"]
    params: PromptCallRequestParams

class PromptCallResult(BaseModel):
    type: Literal["prompt"] = "prompt"      # ← discriminator to dont create confusion on type recognitions
    result: Any

class _BarePromptSpec(BaseModel):
    name: str
    description: str
    input_schema: dict
    annotations: Optional[dict] = None

class PromptsListRequest(BaseModel):
    method: Literal["prompts/list"]
    params: Optional[dict] = None

class PromptsListResult(BaseModel):
    prompts: List[_BarePromptSpec]


# +++ Resource Types Start (Client) +++
class ResourceCapability(BaseModel):
    name: str # URI Pattern, e.g., "users://{user_id}/profile"


class _BareResourceSpec(BaseModel):
    name: str # URI pattern
    description: str
    path_param_schema: Optional[dict] = None # Schema for path parameters
    annotations: Optional[dict] = None


class ResourcesListRequest(BaseModel):
    method: Literal["resources/list"]
    params: Optional[dict] = None


class ResourcesListResult(BaseModel):
    resources: List[_BareResourceSpec]


class ResourceCallRequestParams(BaseModel):
    uri: str # Concrete URI, e.g., "users://123/profile"


class ResourceCallRequest(BaseModel):
    method: Literal["resources/call"]
    params: ResourceCallRequestParams


class ResourceCallResult(BaseModel):
    type: Literal["resource"] = "resource"
    result: Any
# +++ Resource Types End (Client) +++


# --------------------------------------------------------------------------- #
# 3-B.  Wrappers (discriminated on `method`) for WebSocket Client Requests    #
# --------------------------------------------------------------------------- #
ClientRequest = Annotated[
    Union[
        ToolCallRequest,
        ToolsListRequest,
        PromptCallRequest,
        PromptsListRequest,
        ResourcesListRequest,
        ResourceCallRequest,
    ],
    Field(discriminator="method"),
]

# Results don't need a discriminator—each result model is unique
ServerResult = Union[
    ToolCallResult,
    ToolsListResult,
    PromptCallResult,
    PromptsListResult,
    ResourcesListResult,
    ResourceCallResult,
]


# --------------------------------------------------------------------------- #
# 4.  WebSocket framing                                                       #
# --------------------------------------------------------------------------- #
class ServerError(BaseModel):
    code: int
    message: str


class InitializationRequest(BaseModel):
    """Sent by the client immediately after opening the socket."""
    pass


class InitializationResponse(BaseModel):
    server_name: str
    server_version: str
    capabilities: ServerCapabilities
    instructions: str

class CloseSession(BaseModel):
    """Sent by the server to indicate it's closing the session."""
    reason: Optional[str] = None 

class SessionMessage(BaseModel):
    """
    Outer envelope used on the wire for WebSocket communication.
    """
    root: Union[
        ClientRequest,              # Client -> Server (RPC calls)
        ServerResult,               # Server -> Client (RPC results)
        ServerError,                # Server -> Client (Errors)
        InitializationRequest,      # Client -> Server (Handshake start)
        InitializationResponse,     # Server -> Client (Handshake response)
        InitializationOptions,      # Included for completeness, though server-internal
        CloseSession,               # Server -> Client (Session end)
        ProcessUpdate,              # Server -> Client (Process update)
    ]
