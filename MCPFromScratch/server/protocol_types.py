# protocol_types.py  ·  Python ≥ 3.11 · Pydantic v2
from __future__ import annotations
from typing import Any, List, Union, Optional, Literal, Annotated, Dict, Callable, Awaitable, TypedDict
from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# 1.  RPCs for tool-calling                                                   #
# --------------------------------------------------------------------------- #
class ToolCallRequestParams(BaseModel):
    name: str
    arguments: dict[str, Any]


class ToolCallRequest(BaseModel):
    method: Literal["tools/call"]        
    params: ToolCallRequestParams


class ToolCallResult(BaseModel):
    type: Literal["tool"] = "tool"      # ← discriminator to dont create confusion on type recognitions
    result: Any


class ProcessUpdate(BaseModel):
    """A generic message sent from the server to the client about the process update."""
    message: str



# --------------------------------------------------------------------------- #
# 2.  Capabilities handshake                                                  #
# --------------------------------------------------------------------------- #
class ToolCapability(BaseModel):
    name: str


class PromptCapability(BaseModel):
    name: str


class ServerCapabilities(BaseModel):
    tools: List[ToolCapability] = []
    prompts: List[PromptCapability] = []
    resources: List[ResourceCapability] = []


class InitializationOptions(BaseModel):
    server_name: str
    server_version: str
    capabilities: ServerCapabilities
    instructions: str


# --------------------------------------------------------------------------- #
# 3-A.  Generic REST-mirror RPCs                                              #
# --------------------------------------------------------------------------- #
class HealthRequest(BaseModel):
    method: Literal["health"]
    params: Optional[dict] = None


class HealthResult(BaseModel):
    ok: bool


class InfoRequest(BaseModel):
    method: Literal["info"]
    params: Optional[dict] = None


class InfoResult(BaseModel):
    name: str
    version: str
    capabilities: ServerCapabilities


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


class CallInput(BaseModel):
    name: str
    arguments: Dict[str, Any]


class ToolSpec(TypedDict):
    name: str
    description: str
    input_schema: Dict[str, Any]
    annotations: Optional[Dict[str, Any]]
    handler: Callable[..., Awaitable[Any] | Any]
    expects_ctx: bool


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


class PromptSpec(TypedDict):
    name: str
    description: str
    input_schema: Dict[str, Any]
    annotations: Optional[Dict[str, Any]]
    handler: Callable[..., Awaitable[Any] | Any]
    expects_ctx: bool


class PromptsListRequest(BaseModel):
    method: Literal["prompts/list"]
    params: Optional[dict] = None


class PromptsListResult(BaseModel):
    prompts: List[_BarePromptSpec]


class ResourceSpec(TypedDict):
    uri_pattern: str
    description: str
    path_param_schema: Dict[str, Any] # Schema for extracted path parameters
    handler: Callable[..., Awaitable[Any] | Any]
    expects_ctx: bool
    path_param_names: List[str] # Ordered names of path parameters


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





# --------------------------------------------------------------------------- #
# 3-B.  Wrappers (discriminated on `method`)                                  #
# --------------------------------------------------------------------------- #
# ClientRequest represents any one of the possible request types a client can send.
# The `Union` indicates it can be *one of* the listed types (ToolCallRequest OR HealthRequest OR ...).
# Pydantic uses the `method` field (specified by the discriminator)
# to determine which specific type it is when parsing incoming data.
ClientRequest = Annotated[
    Union[
        ToolCallRequest,
        HealthRequest,
        InfoRequest,
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
    HealthResult,
    InfoResult,
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
    The top-level structure for all messages sent over the WebSocket.
    It acts as an envelope, wrapping the actual payload (request, result,
    error, initialization info, etc.) in the `root` field. This allows
    the receiver to parse the outer structure first and then determine
    the specific type of message contained within.
    """
    root: Union[
        ClientRequest,
        ServerResult,
        ServerError,
        InitializationRequest,
        InitializationResponse,
        InitializationOptions,      # (server passes this into session ctor)
        CloseSession,
        ProcessUpdate
    ]


class ResourceCapability(BaseModel):
    name: str # URI Pattern, e.g., "users://{user_id}/profile"

class _BareResourceSpec(BaseModel):
    name: str # URI pattern
    description: str
    path_param_schema: Optional[dict] = None # Schema for path parameters
    annotations: Optional[dict] = None



