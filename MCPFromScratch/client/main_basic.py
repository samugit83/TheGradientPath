# client.py  –  ask for tools/list over Web-Socket
from __future__ import annotations
import asyncio
import websockets
import protocol_types as types
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("MCP_CLIENT_API_KEY") 
MCP_URI = "ws://localhost:8000/mcp" 

if not API_KEY:
    print("Error: MCP_CLIENT_API_KEY environment variable not set.")
    exit(1)

async def main() -> None:
    print(f"🔌 Attempting to connect to {MCP_URI} using X-API-Key header")
    headers = {"X-API-Key": API_KEY}
    try:
        async with websockets.connect(MCP_URI, additional_headers=headers) as ws:
            await ws.send(types.SessionMessage(
                root=types.InitializationRequest()).model_dump_json())

            init_resp_text = await ws.recv()
            init = types.SessionMessage.model_validate_json(init_resp_text)

            if isinstance(init.root, types.InitializationResponse):
                print("🔗  Connected:")
                print(f"  Server Name: {init.root.server_name}")
                print(f"  Server Version: {init.root.server_version}")
                print(f"  Instructions: {init.root.instructions}")
                print(f"  Capabilities:")
                if init.root.capabilities.tools:
                    print(f"    Tools: {[t.name for t in init.root.capabilities.tools]}")
                if init.root.capabilities.prompts:
                    print(f"    Prompts: {[p.name for p in init.root.capabilities.prompts]}")
                if init.root.capabilities.resources:
                    print(f"    Resources: {[r.name for r in init.root.capabilities.resources]}")
                print("\n")
            else:
                print(f"❗ Unexpected response during initialization: {init.root}")
                return 



            print("📜 Requesting list of tools...")
            await ws.send(types.SessionMessage(
                root=types.ToolsListRequest(method="tools/list")
            ).model_dump_json())

            print("📜 Requesting list of tools...")
            await ws.send(types.SessionMessage(
                root=types.ToolCallRequest(
                    method="tools/call",
                    params=types.ToolCallRequestParams(name="add", arguments={"a": 5, "b": 3, "c": 2})
                )
            ).model_dump_json())

            print("📜 Requesting list of prompts...")
            await ws.send(types.SessionMessage(
                root=types.PromptsListRequest(method="prompts/list")
            ).model_dump_json())

            print("📜 Requesting list of prompts...")
            await ws.send(types.SessionMessage(
                root=types.PromptCallRequest(
                    method="prompt/call",
                    params=types.PromptCallRequestParams(
                        name="summarize_text",
                        arguments={"text_to_summarize": "The quick brown fox jumps over the lazy dog", "max_sentences": 1, "style": "concise"}
                    )
                )
            ).model_dump_json())

            print("📜 Requesting list of resources...")
            await ws.send(types.SessionMessage(
                root=types.ResourcesListRequest(method="resources/list")
            ).model_dump_json())

            print("📞 Calling resource 'config://app'...")
            await ws.send(types.SessionMessage(
                root=types.ResourceCallRequest(
                    method="resources/call",
                    params=types.ResourceCallRequestParams(uri="config://app")
                )
            ).model_dump_json())

            user_id_to_call = "testuser789"
            print(f"📞 Calling resource 'users://{user_id_to_call}/profile'...")
            await ws.send(types.SessionMessage(
                root=types.ResourceCallRequest(
                    method="resources/call",
                    params=types.ResourceCallRequestParams(uri=f"users://{user_id_to_call}/profile")
                )
            ).model_dump_json())

            while True:
                try:
                    msg_text = await ws.recv()
                    msg = types.SessionMessage.model_validate_json(msg_text)

                    if isinstance(msg.root, types.ProcessUpdate):
                        print(f"🔔 Process update: {msg.root.message}")
                        continue

                    if isinstance(msg.root, types.ToolsListResult):
                        print("🛠️  Registered tools:")
                        for spec in msg.root.tools:
                            print(f"  - Name: {spec.name}")
                            print(f"    Description: {spec.description}")
                            print(f"    Input Schema: {spec.input_schema}")
                            if spec.annotations:
                                print(f"    Annotations: {spec.annotations}")
                            print()

                    elif isinstance(msg.root, types.PromptsListResult):
                        print("💭 Registered prompts:")
                        for spec in msg.root.prompts:
                            print(f"  - Name: {spec.name}")
                            print(f"    Description: {spec.description}")
                            print(f"    Input Schema: {spec.input_schema}")
                            if spec.annotations:
                                print(f"    Annotations: {spec.annotations}")
                            print()

                    elif isinstance(msg.root, types.ResourcesListResult):
                        print("📦 Registered resources:")
                        if not msg.root.resources:
                            print("  No resources available.")
                        for spec in msg.root.resources:
                            print(f"  - URI Pattern: {spec.name}")
                            print(f"    Description: {spec.description}")
                            if spec.path_param_schema and spec.path_param_schema.get("properties"):
                                print(f"    Path Params Schema: {spec.path_param_schema}")
                            else:
                                print("    Path Params Schema: None")
                            if spec.annotations:
                                print(f"    Annotations: {spec.annotations}")
                            print()

                    elif isinstance(msg.root, types.PromptCallResult):
                        print("💬 Prompt response:")
                        print(msg.root.result)

                    elif isinstance(msg.root, types.ToolCallResult):
                        print("🛠️ Tool response:")
                        print(msg.root.result)

                    elif isinstance(msg.root, types.ResourceCallResult):
                        print("📄 Resource response:")
                        print(f"  Result: {msg.root.result}")
                        print()

                    elif isinstance(msg.root, types.ServerError):
                        print(f"❗ Server error received: {msg.root.code} - {msg.root.message}")
                        break 

                    elif isinstance(msg.root, types.CloseSession):
                        reason = f"Reason: {msg.root.reason}" if msg.root.reason else "No reason provided."
                        print(f"🚪 Server closed the session. {reason}")
                        break 

                    else:
                        print(f"ℹ️ Received unexpected message type: {type(msg.root)}")
                        continue

                except websockets.exceptions.ConnectionClosedOK:
                    print("🚪 Connection closed normally by server.")
                    break
                except websockets.exceptions.ConnectionClosedError as e:
                    print(f"❗ Connection closed unexpectedly: Code={e.code}, Reason='{e.reason}'")
                    break


    except websockets.exceptions.InvalidStatusCode as e:
        reason = e.headers.get('X-WebSocket-Reject-Reason', e.headers.get('sec-websocket-protocol', '')) 
        print(f"❗ Connection failed with status {e.status_code}. Reason: {reason}")
        if e.status_code == 401 or e.status_code == 403:
             print("   -> Check if the API key in X-API-Key header is correct and valid.")
        elif e.status_code == 429:
             print("   -> API key quota may be exceeded.")
        else:
             print(f"   -> Server rejected connection.")

    except ConnectionRefusedError:
        print(f"❗ Connection refused. Is the server running at {MCP_URI.replace('ws://', 'http://')} ?")
    except websockets.exceptions.ConnectionClosedError as e:
         print(f"❗ Connection closed unexpectedly: Code={e.code}, Reason='{e.reason}'")
    except Exception as e:
        print(f"❗ An unexpected error occurred: {type(e).__name__} - {e}")


if __name__ == "__main__":
    asyncio.run(main())
