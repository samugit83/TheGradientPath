from typing import Any, AsyncIterator, Generic, TypeVar, Optional, TYPE_CHECKING
import anyio
import protocol_types as types
from anyio.abc import ObjectReceiveStream, ObjectSendStream

if TYPE_CHECKING:
    from utilities.auth import InMemoryKeyStore


TReq = TypeVar("TReq")
TRes = TypeVar("TRes")

class RequestResponder(Generic[TReq, TRes]):
    """
    Wraps an incoming client request and provides convenient methods to
    send back a response or an error. Use as:

        with responder:
            await responder.respond(result)
        # or:
        with responder:
            await responder.respond_error(code, message)
    """
    request: types.SessionMessage
    _responded: bool = False
    _stream: anyio.abc.ObjectSendStream

    def __init__(self, write_stream: anyio.abc.ObjectSendStream, session_message: types.SessionMessage):
        self.request = session_message
        self._stream = write_stream

    def __enter__(self) -> "RequestResponder[TReq, TRes]":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        # No automatic error reporting here; handlers catch and send errors explicitly
        return False

    async def respond(self, result: types.ServerResult) -> None:
        """
        Send a normal (successful) response back to the client.
        """
        msg = types.SessionMessage(root=result)
        await self._stream.send(msg)
        self._responded = True

    async def respond_error(self, code: int, message: str) -> None:
        """
        Send an error response back to the client.
        """
        error = types.ServerError(code=code, message=message)
        msg = types.SessionMessage(root=error)
        await self._stream.send(msg)
        self._responded = True


class ServerSession:
    """
    Base session for WebSocket/MCP communication. It:
      1. Sends an InitializationResponse upon start
      2. Reads incoming `SessionMessage` objects
      3. Wraps any `ClientRequest` in a `RequestResponder` and exposes them
         on `self.incoming_messages`, for handlers to consume.

    Usage:
        session = ServerSession(read_stream, write_stream, init_options)
        async with anyio.create_task_group() as tg:
            tg.start_soon(session._receive_loop)
            tg.start_soon(<your dispatch loop using session.incoming_messages>)
    """
    def __init__(
        self,
        read_stream: ObjectReceiveStream[types.SessionMessage],
        write_stream: ObjectSendStream[types.SessionMessage],
        init_options: types.InitializationOptions,
        api_key: Optional[str] = None,
        key_store: Optional["InMemoryKeyStore"] = None,
        app_state: Optional[Any] = None,
    ) -> None:
        self._read_stream = read_stream
        self._write_stream = write_stream
        self.init_options = init_options
        self.api_key = api_key
        self.key_store = key_store
        self.app_state = app_state
        send_chan, recv_chan = anyio.create_memory_object_stream(0)
        self._msg_send = send_chan
        self.incoming_messages: AsyncIterator[RequestResponder[Any, Any]] = recv_chan

    async def _receive_loop(self) -> None:
        """
        1. Send our InitializationResponse
        2. Read each incoming SessionMessage
        3. If auth is enabled, check quota and increment for each request message
        4. Wrap any valid ClientRequest in a RequestResponder and enqueue it
        """
        # 1) Send InitializationResponse
        init_resp = types.InitializationResponse(
            server_name=self.init_options.server_name,
            server_version=self.init_options.server_version,
            capabilities=self.init_options.capabilities,
            instructions=self.init_options.instructions,
        )
        await self._write_stream.send(types.SessionMessage(root=init_resp))

        # 2) Pump incoming messages
        async for msg in self._read_stream:
            root = msg.root
            if isinstance(root, types.InitializationRequest):
                continue

            # --- Start Quota Check ---
            # Check if it's a request type that should be rate-limited AND if auth is enabled
            is_rate_limited_request = isinstance(
                root,
                (
                    types.ToolsListRequest,
                    types.ToolCallRequest,
                    types.HealthRequest,
                    types.InfoRequest,
                    types.PromptsListRequest,
                    types.PromptCallRequest,
                    types.ResourcesListRequest,
                    types.ResourceCallRequest
                )
            )

            if self.api_key and self.key_store and is_rate_limited_request:
                # Check quota *before* processing
                # Use validate which checks and increments atomically if possible
                is_valid = await self.key_store.validate(self.api_key, increment=True)

                if not is_valid:
                    # Check again *without* incrementing to see if it was invalid key or quota
                    quota_info = await self.key_store.get(self.api_key)
                    if not quota_info:
                        code, message = 401, "Invalid API Key (session)"
                    else:
                         code, message = 429, "Quota exceeded"

                    # Quota exceeded or key invalid, send error and skip processing
                    error = types.ServerError(code=code, message=message)
                    error_msg = types.SessionMessage(root=error)
                    # Prevent trying to send on a closed stream
                    try:
                        await self._write_stream.send(error_msg)
                    except anyio.BrokenResourceError:
                        pass # Stream likely closed, can't send error
                    continue # Skip to next message
            # --- End Quota Check ---

             # 3) Wrap client requests (only runs if quota check passed or wasn't needed)
                
            if isinstance(
                root,
                (
                    types.ToolsListRequest,
                    types.ToolCallRequest,
                    types.HealthRequest,
                    types.InfoRequest,
                    types.PromptsListRequest,
                    types.PromptCallRequest,
                    types.ResourcesListRequest,
                    types.ResourceCallRequest
                )
            ):
                responder = RequestResponder(self._write_stream, msg)
                await self._msg_send.send(responder)
            else:
                # Optional: Log ignored messages for debugging
                # logger.debug(f"Ignoring message type: {type(root)}")
                continue

