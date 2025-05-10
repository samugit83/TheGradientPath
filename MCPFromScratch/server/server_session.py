from __future__ import annotations

import inspect, re
from typing import Any, Dict, TYPE_CHECKING # Added TYPE_CHECKING
import anyio
import protocol_types as types
from context_models import Context, RequestContext, LifespanContext # Added RequestContext, LifespanContext
from base_session import ServerSession, RequestResponder
import logging

if TYPE_CHECKING:
    from utilities.auth import InMemoryKeyStore, ToolSpec

logger = logging.getLogger("mcpwiz.server_session")


class WizServerSession(ServerSession):
    """
    Lets every REST-style RPC (health, info, tools/list, tools/call)
    be reached over the MCP websocket.
    """
    def __init__(
        self,
        registry: Dict[str, Any],
        prompt_registry: Dict[str, Any],
        resource_registry: Dict[str, types.ResourceSpec],
        read_stream,
        write_stream,
        init_options: types.InitializationOptions,
        api_key: str,
        key_store: "InMemoryKeyStore",
        app_state: Any | None = None,
    ) -> None:
        super().__init__(
            read_stream,
            write_stream,
            init_options,
            api_key=api_key,
            key_store=key_store,
            app_state=app_state,
        )
        self._tool_registry = registry
        self._prompt_registry = prompt_registry
        self._resource_registry = resource_registry
        self._bare_tools = [
            {k: v for k, v in spec.items() if k not in ("handler", "expects_ctx")}
            for spec in registry.values()
        ]
        self._bare_prompts = [
            {k: v for k, v in spec.items() if k not in ("handler", "expects_ctx")}
            for spec in prompt_registry.values()
        ]
        self._bare_resources = [
            {
                "name": spec["uri_pattern"],
                "description": spec["description"],
                "path_param_schema": spec["path_param_schema"],
            }
            for spec in resource_registry.values()
        ]


    async def start_dispatching(self) -> None:
        async for responder in self.incoming_messages:
            await self._handle_request(responder)


    async def _handle_request(
        self,
        responder: RequestResponder[Any, Any],
    ) -> None:
        inner = responder.request.root      
       
        match inner:
            case types.ToolCallRequest(
                params=types.ToolCallRequestParams(name=tool, arguments=args)
            ):
                await self._exec_tool(tool, args, responder)

            case types.PromptCallRequest(
                params=types.PromptCallRequestParams(name=prompt, arguments=args)
            ):
                await self._exec_prompt(prompt, args, responder)

            case types.ResourceCallRequest(
                params=types.ResourceCallRequestParams(uri=uri)
            ):
                await self._exec_resource(uri, responder)

            case types.HealthRequest():
                with responder:
                    await responder.respond(types.HealthResult(ok=True))

            case types.InfoRequest():
                with responder:
                    await responder.respond(
                        types.InfoResult(
                            name=self.init_options.server_name,
                            version=self.init_options.server_version,
                            capabilities=self.init_options.capabilities,
                        )
                    )

            case types.ToolsListRequest():
                with responder:
                    await anyio.sleep(3)
                    update_msg = "Preparing the list of available tools..."
             
                    await self._write_stream.send(
                        types.SessionMessage(root=types.ProcessUpdate(message=update_msg))
                    )
                    await anyio.sleep(3)
                    update_msg = "Done preparing the list of available tools."
            
                    await self._write_stream.send(
                        types.SessionMessage(root=types.ProcessUpdate(message=update_msg))
                    )

                    await responder.respond(
                        types.ToolsListResult(tools=self._bare_tools)
                    )

            case types.PromptsListRequest():
                with responder:
                    await responder.respond(
                        types.PromptsListResult(prompts=self._bare_prompts)
                    )

            case types.ResourcesListRequest():
                with responder:
                    await responder.respond(
                        types.ResourcesListResult(resources=self._bare_resources)
                    )

            case _:
                with responder:
                    await responder.respond_error(501, "method not implemented")



    async def _exec_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        responder: RequestResponder[Any, Any],
    ) -> None:
        spec = self._tool_registry.get(tool_name)
        if spec is None:
            with responder:
                await responder.respond_error(404, f"unknown tool '{tool_name}'")
            return

        try:
            # ---- 3-a. validate params -------------------------------- #
            from pydantic import create_model

            props = spec["input_schema"]["properties"]
            ParamsModel = create_model("ParamsModel", **{p: (Any, ...) for p in props})
            validated = ParamsModel(**arguments)

            # ---- 3-b. call handler ----------------------------------- #
            kwargs = validated.dict()
            if spec["expects_ctx"]:
                lifespan_ctx_instance = None
                if self.app_state:
                    lifespan_ctx_instance = LifespanContext(self.app_state)
                
                request_ctx_instance = RequestContext(
                    request=None, # No direct HTTP request in WebSocket flow
                    lifespan_context=lifespan_ctx_instance
                )
                ctx_obj = Context(request_context=request_ctx_instance)
                kwargs = {"ctx": ctx_obj, **kwargs}

            maybe_awaitable = spec["handler"](**kwargs)
            result = (
                await maybe_awaitable
                if inspect.isawaitable(maybe_awaitable)
                else maybe_awaitable
            )
            # ---- 3-c. reply ------------------------------------------ #
            with responder:
                await responder.respond(types.ToolCallResult(result=result, type="tool"))

        except Exception as exc:
            with responder:
                await responder.respond_error(500, str(exc))
                

    async def _exec_prompt(
        self,
        prompt_name: str,
        arguments: dict[str, Any],
        responder: RequestResponder[Any, Any],
    ) -> None:
        spec = self._prompt_registry.get(prompt_name)
        if spec is None:
            with responder:
                await responder.respond_error(404, f"unknown prompt '{prompt_name}'")
            return

        try:
            # Validate params
            from pydantic import create_model
            props = spec["input_schema"]["properties"]
            ParamsModel = create_model("ParamsModel", **{p: (Any, ...) for p in props})
            validated = ParamsModel(**arguments)

            # Call handler
            kwargs = validated.dict()
            if spec["expects_ctx"]:
                lifespan_ctx_instance = None
                if self.app_state:
                    lifespan_ctx_instance = LifespanContext(self.app_state)

                request_ctx_instance = RequestContext(
                    request=None, # No direct HTTP request in WebSocket flow
                    lifespan_context=lifespan_ctx_instance
                )
                ctx_obj = Context(request_context=request_ctx_instance)
                kwargs = {"ctx": ctx_obj, **kwargs}

            maybe_awaitable = spec["handler"](**kwargs)
            result = (
                await maybe_awaitable
                if inspect.isawaitable(maybe_awaitable)
                else maybe_awaitable
            )

            # Reply
            with responder:
                await responder.respond(types.PromptCallResult(result=result, type="prompt"))

        except Exception as exc:
            with responder:
                await responder.respond_error(500, str(exc))

    async def _exec_resource(
        self,
        uri: str,
        responder: RequestResponder[Any, Any],
    ) -> None:
        matched_spec = None
        extracted_args_from_uri = {}

        for pattern, spec_candidate in self._resource_registry.items():
            regex_pattern = spec_candidate.get("_regex_pattern") 
            if not regex_pattern or not isinstance(regex_pattern, re.Pattern):
                logger.error(f"Regex pattern missing or invalid for resource '{pattern}'.")
                continue
            
            match = regex_pattern.match(uri)
            if match:
                matched_spec = spec_candidate
                path_values = match.groups()
                param_names = matched_spec["path_param_names"]
                if len(path_values) == len(param_names):
                    extracted_args_from_uri = dict(zip(param_names, path_values))
                break
        
        if matched_spec is None:
            with responder:
                await responder.respond_error(404, f"No resource found matching URI '{uri}'")
            return

        try:
            final_kwargs = {}
            param_schema_props = matched_spec["path_param_schema"].get("properties", {})

            for name, value_str in extracted_args_from_uri.items():
                prop_details = param_schema_props.get(name)
                if not prop_details:
                    with responder:
                        await responder.respond_error(500, f"Schema definition missing for path parameter '{name}'")
                    return
                
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
                            raise ValueError("Invalid boolean")
                    else:
                        final_kwargs[name] = str(value_str)
                except ValueError:
                    with responder:
                        await responder.respond_error(422, f"Invalid value for path parameter '{name}': '{value_str}'. Expected {target_type}.")
                    return

            if matched_spec["expects_ctx"]:
                lifespan_ctx_instance = None
                if self.app_state:
                    lifespan_ctx_instance = LifespanContext(self.app_state)
                
                request_ctx_instance = RequestContext(
                    request=None, # No direct HTTP request in WebSocket flow
                    lifespan_context=lifespan_ctx_instance
                )
                ctx_obj = Context(request_context=request_ctx_instance)
                kwargs = {"ctx": ctx_obj, **final_kwargs}
            else:
                kwargs = final_kwargs

            maybe_awaitable = matched_spec["handler"](**kwargs)
            result = (
                await maybe_awaitable
                if inspect.isawaitable(maybe_awaitable)
                else maybe_awaitable
            )
            with responder:
                await responder.respond(types.ResourceCallResult(result=result, type="resource"))

        except Exception as exc:
            logger.error(f"Error executing resource '{uri}': {exc}", exc_info=True)
            with responder:
                await responder.respond_error(500, str(exc))
