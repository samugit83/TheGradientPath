from __future__ import annotations

import inspect, anyio, re
from functools import wraps
from typing import Any, Awaitable, Callable, Dict, Optional, List, Tuple
from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, Header, status 
from fastapi.responses import JSONResponse
from pydantic import ValidationError, create_model
import protocol_types as types
from server_session import WizServerSession
from context_models import Context, RequestContext, LifespanContext
import logging
from uvicorn.config import LOGGING_CONFIG
from dotenv import load_dotenv 
from utilities.auth import api_key_auth, InMemoryKeyStore

load_dotenv()

LOGGING_CONFIG["loggers"]["mcpwiz"] = {   
    "handlers": ["default"],
    "level": "DEBUG",                
}

logging.config.dictConfig(LOGGING_CONFIG)

# Helper to convert URI pattern to regex and extract param names
def _parse_uri_pattern(pattern: str) -> Tuple[re.Pattern, List[str]]:
    param_names = []
    regex_pattern = "^"
    last_pos = 0
    for match in re.finditer(r"{([a-zA-Z_][a-zA-Z0-9_]*)}", pattern):
        param_name = match.group(1)
        if param_name in param_names:
            raise ValueError(f"Duplicate path parameter '{param_name}' in URI pattern '{pattern}'")
        param_names.append(param_name)
        regex_pattern += re.escape(pattern[last_pos:match.start()])
        regex_pattern += r"([^/]+)" # Capture anything until next slash
        last_pos = match.end()
    regex_pattern += re.escape(pattern[last_pos:]) + "$"
    return re.compile(regex_pattern), param_names

class MCPWizServer:
    def __init__(
        self,
        title: str = "mcpwiz-server",
        version: str = "0.1.0",
        *,
        lifespan: Optional[Callable[..., Awaitable[Any]]] = None,
    ) -> None:
        self.app = FastAPI(title=title, version=version, lifespan=lifespan)
        self._tool_registry: Dict[str, types.ToolSpec] = {}
        self._prompt_registry: Dict[str, types.PromptSpec] = {}
        self._resource_registry: Dict[str, types.ResourceSpec] = {}
        self._setup_routes()

    def tool(
        self,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        input_schema: Optional[Dict[str, Any]] = None,
        annotations: Optional[Dict[str, Any]] = None,
    ):
        def decorator(fn: Callable[..., Any]):
            tool_name = name or fn.__name__
            if tool_name in self._tool_registry:
                raise ValueError(f"Tool '{tool_name}' already registered")

            tool_desc = description or (fn.__doc__ or "").strip()
            schema = input_schema or self._schema_from_signature(fn)

            sig = inspect.signature(fn)
            params = list(sig.parameters.values())
            expects_ctx = bool(params) and params[0].annotation is Context

            @wraps(fn)
            async def async_wrapper(**kwargs):
                result = fn(**kwargs)
                if inspect.isawaitable(result):
                    result = await result
                return result

            self._tool_registry[tool_name] = {
                "name": tool_name,
                "description": tool_desc,
                "input_schema": schema,
                "annotations": annotations,
                "handler": async_wrapper,
                "expects_ctx": expects_ctx,
            }
            return fn
        return decorator

    def prompt(
        self,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        input_schema: Optional[Dict[str, Any]] = None,
        annotations: Optional[Dict[str, Any]] = None,
    ):
        def decorator(fn: Callable[..., Any]):
            prompt_name = name or fn.__name__
            if prompt_name in self._prompt_registry:
                raise ValueError(f"Prompt '{prompt_name}' already registered")

            prompt_desc = description or (fn.__doc__ or "").strip()
            schema = input_schema or self._schema_from_signature(fn)

            sig = inspect.signature(fn)
            params = list(sig.parameters.values())
            expects_ctx = bool(params) and params[0].annotation is Context

            @wraps(fn)
            async def async_wrapper(**kwargs):
                result = fn(**kwargs)
                if inspect.isawaitable(result):
                    result = await result
                return result

            self._prompt_registry[prompt_name] = {
                "name": prompt_name,
                "description": prompt_desc,
                "input_schema": schema,
                "annotations": annotations,
                "handler": async_wrapper,
                "expects_ctx": expects_ctx,
            }
            return fn
        return decorator

    def resource(
        self,
        uri_pattern: str, # e.g., "users://{user_id}/profile"
        *,
        description: Optional[str] = None,
        annotations: Optional[Dict[str, Any]] = None, # Retained for future use, though not directly used for path params schema
    ):
        def decorator(fn: Callable[..., Any]):
            if uri_pattern in self._resource_registry:
                raise ValueError(f"Resource URI pattern '{uri_pattern}' already registered")

            resource_desc = description or (fn.__doc__ or "").strip()
            
            _regex_pattern, path_param_names = _parse_uri_pattern(uri_pattern)

            sig = inspect.signature(fn)
            fn_params = list(sig.parameters.values())
            
            expects_ctx = bool(fn_params) and fn_params[0].annotation is Context
            
            expected_fn_param_names = [p.name for p in fn_params[1 if expects_ctx else 0:]]

            # Validate that path param names from URI match function signature (excluding context)
            if set(path_param_names) != set(expected_fn_param_names):
                raise ValueError(
                    f"Mismatch between path parameters in URI pattern '{uri_pattern}' ({path_param_names}) "
                    f"and function signature '{fn.__name__}' ({expected_fn_param_names})."
                )

            # Create a simple schema for path parameters based on function signature
            path_param_schema: Dict[str, Any] = {"type": "object", "properties": {}, "required": []}
            
            param_start_index = 1 if expects_ctx else 0
            for p_name in path_param_names: # Iterate in order of URI definition for consistency
                fn_param = sig.parameters.get(p_name)
                if not fn_param: # Should be caught by the set comparison above, but as a safeguard
                    raise ValueError(f"Path parameter '{p_name}' not found in function signature of '{fn.__name__}'.")

                ann = fn_param.annotation
                param_type = "string" # Default
                if ann is str:
                    param_type = "string"
                elif ann is int:
                    param_type = "integer"
                elif ann is float:
                    param_type = "number"
                elif ann is bool:
                    param_type = "boolean"
                # Add more type mappings if needed, or use a more robust schema generation
                
                path_param_schema["properties"][p_name] = {"type": param_type, "description": f"Path parameter: {p_name}"}
                if fn_param.default is inspect.Parameter.empty:
                     path_param_schema["required"].append(p_name)


            @wraps(fn)
            async def async_wrapper(**kwargs): # kwargs will be populated by extracted path params
                # Path params are already validated and typed (to some extent) by FastAPI/Pydantic when routes are set up,
                # or manually in _invoke_resource_http for direct calls.
                # Here, we assume kwargs are correctly provided.
                result = fn(**kwargs)
                if inspect.isawaitable(result):
                    result = await result
                return result

            self._resource_registry[uri_pattern] = types.ResourceSpec({
                "uri_pattern": uri_pattern,
                "description": resource_desc,
                "path_param_schema": path_param_schema,
                "handler": async_wrapper,
                "expects_ctx": expects_ctx,
                "path_param_names": path_param_names,
                # Store the compiled regex pattern for efficient matching later
                "_regex_pattern": _regex_pattern 
            })
            return fn
        return decorator

    def _setup_routes(self) -> None:

        @self.app.get("/health")
        async def health() -> Dict[str, bool]:
            return {"ok": True}

        @self.app.get("/info", dependencies=[Depends(api_key_auth)])
        async def info() -> Dict[str, Any]:
            return {
                "name": self.app.title,
                "version": self.app.version,
                "capabilities": {
                    "tools": [
                        {"name": spec["name"], "description": spec["description"]}
                        for spec in self._tool_registry.values()
                    ],
                    "prompts": [
                        {"name": spec["name"], "description": spec["description"]}
                        for spec in self._prompt_registry.values()
                    ],
                    "resources": [
                        {"name": spec["uri_pattern"], "description": spec["description"]}
                        for spec in self._resource_registry.values()
                    ]
                },
            }

        @self.app.get("/tools/list", dependencies=[Depends(api_key_auth)])
        async def list_tools() -> Dict[str, Any]:
            return {
                "tools": [
                    {k: v for k, v in spec.items() if k not in ("handler", "expects_ctx")}
                    for spec in self._tool_registry.values()
                ]
            }

        @self.app.get("/prompts/list", dependencies=[Depends(api_key_auth)])
        async def list_prompts() -> Dict[str, Any]:
            return {
                "prompts": [
                    {k: v for k, v in spec.items() if k not in ("handler", "expects_ctx")}
                    for spec in self._prompt_registry.values()
                ]
            }

        @self.app.get("/resources/list", dependencies=[Depends(api_key_auth)])
        async def list_resources() -> Dict[str, Any]:
            return {
                "resources": [
                    {
                        "name": spec["uri_pattern"], 
                        "description": spec["description"],
                        "path_param_schema": spec["path_param_schema"],
                    }
                    for spec in self._resource_registry.values()
                ]
            }

        async def _resolve_ctx(request: Request) -> Context:
            lifespan_ctx = LifespanContext(request.app.state._state)
            return Context(
                request_context=RequestContext(request=request,
                                               lifespan_context=lifespan_ctx)
            )

        @self.app.post("/tools/call", dependencies=[Depends(api_key_auth)])
        async def call_tool(
            call: types.CallInput,
            ctx: Context = Depends(_resolve_ctx),
        ):
            return await self._invoke_tool_http(call.name, call.arguments, ctx)

        @self.app.post("/prompt/call", dependencies=[Depends(api_key_auth)])
        async def call_prompt(
            call: types.PromptCallRequest,
            ctx: Context = Depends(_resolve_ctx),
        ):
            return await self._invoke_prompt_http(call.params.name, call.params.arguments, ctx)

        @self.app.post("/resources/call", dependencies=[Depends(api_key_auth)])
        async def call_resource(
            call: types.ResourceCallRequest,
            ctx: Context = Depends(_resolve_ctx),
        ):
            return await self._invoke_resource_http(call.params.uri, ctx)

        # 2-c  NEW: WebSocket /mcp  --------------------------------------#
        @self.app.websocket("/mcp")
        async def mcp_endpoint(
            ws: WebSocket,
            x_api_key: str = Header(..., description="Your API key for authentication")
        ):
            key_store: InMemoryKeyStore = ws.app.state.key_store  
            # --- Start WebSocket Auth Check ---
            quota_info = await key_store.get(x_api_key)
            if not quota_info:
                await ws.close(code=status.WS_1008_POLICY_VIOLATION, reason="Invalid API Key")
                return
            quota, used = quota_info
            # Check quota *before* accepting connection.
            # Note: We don't increment here; increment happens per-message in the session.
            if used >= quota:
                 await ws.close(code=status.WS_1008_POLICY_VIOLATION, reason="Quota exceeded")
                 return
            # --- End WebSocket Auth Check ---

            await ws.accept()

            # Create communication channels
            c2s_send, c2s_recv = anyio.create_memory_object_stream(0)
            s2c_send, s2c_recv = anyio.create_memory_object_stream(0)

            # WebSocket reader task
            async def ws_reader():
                try:
                    async for text in ws.iter_text():
                        try:
                            parsed_message = types.SessionMessage.model_validate_json(text)
                            await c2s_send.send(parsed_message)
                        except ValidationError as e:
                            logging.warning(f"WS validation error: {e}")
                            # Optionally send an error message back to client
                            error = types.ServerError(code=400, message=f"Invalid message format: {e}")
                            error_msg = types.SessionMessage(root=error)
                            json_text = error_msg.model_dump_json()
                            await ws.send_text(json_text)

                except WebSocketDisconnect:
                    logging.info(f"WebSocket client disconnected: {x_api_key[:4]}...") # Log disconnect
                except Exception as e:
                    logging.error(f"WS Reader error: {e}", exc_info=True)
                finally:
                    await c2s_send.aclose()

            # WebSocket writer task
            async def ws_writer():
                try:
                    async for msg in s2c_recv:
                        json_text = msg.model_dump_json()
                        await ws.send_text(json_text)
                except Exception as e:
                     logging.error(f"WS Writer error: {e}", exc_info=True)
                finally:
                    # Ensure writer closes if reader closes stream
                    pass


            # Initialise MCP session
            init_opts = types.InitializationOptions(
                server_name=self.app.title,
                server_version=self.app.version,
                capabilities=types.ServerCapabilities(
                    tools=[
                        types.ToolCapability(name=spec["name"])
                        for spec in self._tool_registry.values()
                    ],
                    prompts=[
                        types.PromptCapability(name=spec["name"])
                        for spec in self._prompt_registry.values()
                    ],
                    resources=[
                        types.ResourceCapability(name=spec["uri_pattern"])
                        for spec in self._resource_registry.values()
                    ]
                ),
                instructions="MCPWiz is ready to interact with you!",
            )
            session = WizServerSession(
                registry=self._tool_registry,
                prompt_registry=self._prompt_registry,
                resource_registry=self._resource_registry,
                read_stream=c2s_recv,
                write_stream=s2c_send,
                init_options=init_opts,
                api_key=x_api_key,      # Pass validated key
                key_store=key_store,  # Pass key store instance
                app_state=ws.app.state # Pass the app_state
            )

            # Run tasks
            try:
                async with anyio.create_task_group() as tg:
                    tg.start_soon(ws_reader)
                    tg.start_soon(ws_writer)
                    tg.start_soon(session._receive_loop)
                    tg.start_soon(session.start_dispatching)

            except Exception as e:
                 logging.error(f"MCP Session error: {e}", exc_info=True)
            finally:
                 try:
                     close_session = types.CloseSession()
                     close_session_msg = types.SessionMessage(root=close_session)
                     await s2c_send.send(close_session_msg)
                 except Exception as e:
                     logging.error(f"Error sending CloseSession: {e}", exc_info=True)

                 await s2c_send.aclose()
                 await c2s_send.aclose()

                 try:
                     await ws.close()
                 except RuntimeError: # Already closed
                     pass
                 logging.info(f"MCP Session ended for key: {x_api_key[:4]}...")


    async def _invoke_tool_http(
        self,
        name: str,
        arguments: dict[str, Any],
        ctx: Context,
    ):
        spec = self._tool_registry.get(name)
        if spec is None:
            raise HTTPException(404, f"Unknown tool '{name}'")

        ParamsModel = create_model(
            "ParamsModel", **{p: (Any, ...) for p in spec["input_schema"]["properties"]}
        )
        try:
            validated = ParamsModel(**arguments)
        except ValidationError as e:
            raise HTTPException(422, detail=e.errors())

        kwargs = validated.dict()
        if spec["expects_ctx"]:
            kwargs = {"ctx": ctx, **kwargs}

        result = await spec["handler"](**kwargs)    # type: ignore
        return JSONResponse(content={"content": [{"type": "text", "text": str(result)}]})

    async def _invoke_prompt_http(
        self,
        name: str,
        arguments: dict[str, Any],
        ctx: Context,
    ):
        spec = self._prompt_registry.get(name)
        if spec is None:
            raise HTTPException(404, f"Unknown prompt '{name}'")

        ParamsModel = create_model(
            "ParamsModel", **{p: (Any, ...) for p in spec["input_schema"]["properties"]}
        )
        try:
            validated = ParamsModel(**arguments)
        except ValidationError as e:
            raise HTTPException(422, detail=e.errors())

        kwargs = validated.dict()
        if spec["expects_ctx"]:
            kwargs = {"ctx": ctx, **kwargs}

        result = await spec["handler"](**kwargs)    # type: ignore
        return JSONResponse(content={"result": result})

    async def _invoke_resource_http(
        self,
        uri: str,
        ctx: Context,
    ):
        matched_spec = None
        extracted_args = {}

        for pattern, spec_candidate in self._resource_registry.items():
            regex_pattern = spec_candidate["_regex_pattern"]
            match = regex_pattern.match(uri)
            if match:
                matched_spec = spec_candidate
                path_values = match.groups()
                param_names = matched_spec["path_param_names"]
                if len(path_values) == len(param_names):
                    extracted_args = dict(zip(param_names, path_values))
                break
        
        if matched_spec is None:
            raise HTTPException(404, f"No resource found matching URI '{uri}'")

        final_kwargs = {}
        param_schema_props = matched_spec["path_param_schema"].get("properties", {})

        for name, value_str in extracted_args.items():
            prop_details = param_schema_props.get(name)
            if not prop_details:
                raise HTTPException(500, f"Schema definition missing for path parameter '{name}'")

            try:
                target_type = prop_details.get("type")
                if target_type == "integer":
                    final_kwargs[name] = int(value_str)
                elif target_type == "number":
                    final_kwargs[name] = float(value_str)
                elif target_type == "boolean":
                    if value_str.lower() == 'true':
                        final_kwargs[name] = True
                    elif value_str.lower() == 'false':
                        final_kwargs[name] = False
                    else:
                        raise ValueError(f"Invalid boolean value for {name}: {value_str}")
                else:
                    final_kwargs[name] = str(value_str)
            except ValueError as e:
                raise HTTPException(422, f"Invalid value for path parameter '{name}': {value_str}. Expected {target_type}. Error: {e}")

        if matched_spec["expects_ctx"]:
            final_kwargs = {"ctx": ctx, **final_kwargs}
        
        result = await matched_spec["handler"](**final_kwargs)
        return JSONResponse(content={"content": [{"type": "text", "text": str(result)}]})

    def run(self, host: str = "0.0.0.0", port: int = 8000):
        import uvicorn
        uvicorn.run(self.app, host=host, port=port)

    # ---------------------- JSON-schema helper ---------------------------#
    @staticmethod
    def _schema_from_signature(fn: Callable[..., Any]) -> Dict[str, Any]:
        fields: Dict[str, tuple] = {}
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())
        start_index = 1 if params and params[0].annotation is Context else 0

        for p in params[start_index:]:
            ann = p.annotation if p.annotation is not inspect._empty else Any
            default = p.default if p.default is not inspect._empty else ...
            fields[p.name] = (ann, default)

        if not fields:
            return {"type": "object", "properties": {}, "required": []}

        ParamModel = create_model(f"{fn.__name__.title()}Params", **fields)
        return ParamModel.model_json_schema()
