import asyncio
import json
import os
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

import websockets
from openai import AsyncOpenAI
import protocol_types as types

# Load environment variables from .env file
load_dotenv()

class ChatAssistant:
    """
    Chat AI assistant that uses an external LLM to pick and invoke only WebSocket tools:
    - tools/list to discover available tools
    - tools/call to execute the chosen tool

    Flow:
    1) Receive user query
    2) Call tools/list
    3) Ask LLM to pick a tool (e.g., text_to_sql) and assemble its arguments
    4) Call tools/call with chosen name and arguments
    5) Return the tool result to the user
    """
    def __init__(
        self,
        uri: str,
        mcp_api_key: str,
        openai_api_key: str,
        timeout: int = 10
    ):
        self._uri = uri
        self._mcp_api_key = mcp_api_key
        self._timeout = timeout
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._openai_client = AsyncOpenAI(api_key=openai_api_key)

    async def connect(self) -> None:
        headers = {"X-API-Key": self._mcp_api_key}
        self._ws = await websockets.connect(self._uri, additional_headers=headers)
        await self._initialize_session()

    async def _initialize_session(self) -> None:
        assert self._ws, "WebSocket is not connected"
        await self._ws.send(
            types.SessionMessage(root=types.InitializationRequest()).model_dump_json()
        )
        resp = await self._recv_session_message()
        if not isinstance(resp.root, types.InitializationResponse):
            raise RuntimeError(f"Initialization failed: {resp.root}")

    async def _recv_session_message(self) -> types.SessionMessage:
        assert self._ws, "WebSocket is not connected"
        text = await self._ws.recv()
        return types.SessionMessage.model_validate_json(text)

    async def _call_tools_list(self) -> List[types._BareToolSpec]:
        """Calls tools/list and returns the list of ToolSpec."""
        req = types.ToolsListRequest(method="tools/list")
        await self._ws.send(types.SessionMessage(root=req).model_dump_json())
        
        while True:
            resp = await self._recv_session_message()
            if isinstance(resp.root, types.ProcessUpdate):
                print(f"Server update: {resp.root.message}")
                continue
            elif isinstance(resp.root, types.ToolsListResult):
                print(f"Tools list: {resp.root.tools}")
                return resp.root.tools
            else:
                raise RuntimeError(f"Unexpected response to tools/list: {resp.root}")

    async def _call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Calls tools/call with the given name and arguments."""
        req = types.ToolCallRequest(
            method="tools/call",
            params=types.ToolCallRequestParams(name=name, arguments=arguments)
        )
        await self._ws.send(types.SessionMessage(root=req).model_dump_json())
        
        while True:
            resp = await self._recv_session_message()
            if isinstance(resp.root, types.ProcessUpdate):
                print(f"Server update: {resp.root.message}")
                continue
            elif isinstance(resp.root, types.ToolCallResult):
                return resp.root.result
            elif isinstance(resp.root, types.ServerError):
                raise RuntimeError(f"Tool error: {resp.root.code} - {resp.root.message}")
            else:
                raise RuntimeError(f"Unexpected response to tools/call: {resp.root}")

    async def handle_user_query(self, query: str) -> Any:
        """
        Main entry point: routes a natural language query to the appropriate tool by:
        1) listing tools
        2) asking the LLM which tool to use and what args
        3) executing the tool
        4) asking the LLM to formulate a conversational answer from the tool's result
        5) returning its result
        """
        tools = await self._call_tools_list()
        tool_name, tool_args = await self._select_tool_with_llm(query, tools)
        print(f"Selected tool: {tool_name}")
        print(f"Tool arguments: {tool_args}")
        tool_result = await self._call_tool(tool_name, tool_args)
        conversational_answer = await self._generate_conversational_answer_with_llm(query, tool_result)
        return conversational_answer

    async def _select_tool_with_llm(
        self, query: str, tools: List[types._BareToolSpec]
    ) -> tuple[str, Dict[str, Any]]:
        system_prompt = (
            "You are an assistant that routes a user query to a specific tool.\n"
            "Available tools:\n"
        )
        
        for t in tools:
            system_prompt += f"\nTool: {t.name}\n"
            system_prompt += f"Description: {t.description}\n"
            system_prompt += f"Input Schema: {json.dumps(t.input_schema, indent=2)}\n"
            if t.annotations:
                system_prompt += f"Annotations: {json.dumps(t.annotations, indent=2)}\n"
        
        system_prompt += (
            "\nGiven the user query, choose the best tool and produce a JSON with keys 'tool' and 'args'.\n"
            "'tool' must be one of the names above. 'args' must match the input schema of the chosen tool.\n"
            "Respond with JSON only, no extra text."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User query: {query}"}
        ]

        resp = await self._openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0
        )

        try:
            content = resp.choices[0].message.content
            # Remove markdown code block formatting if present
            if content.startswith("```json"):
                content = content[7:]  # Remove ```json
            if content.endswith("```"):
                content = content[:-3]  # Remove ```
            content = content.strip()
            
            chooser = json.loads(content)
            tool = chooser["tool"]
            args = chooser.get("args", {})
            
            if tool not in [t.name for t in tools]:
                raise ValueError(f"LLM selected unknown tool: {tool}")
            
            return tool, args
        except Exception as e:
            raise RuntimeError(f"Failed to select tool: {e}\nResponse was: {resp.choices[0].message.content}")

    async def _generate_conversational_answer_with_llm(self, original_query: str, tool_result: Any) -> str:
        """
        Uses the LLM to generate a conversational answer from the tool's output.
        """
        system_prompt = (
            "You are an assistant that helps convert raw tool output into a friendly, conversational answer.\n"
            "The user asked the following query:\n"
            f"'{original_query}'\n\n"
            "The tool executed and produced the following result (in JSON format):\n"
            f"{json.dumps(tool_result, indent=2)}\n\n"
            "Please formulate a natural language response to the user based on this information. "
            "Be helpful and conversational. Do not just repeat the JSON."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Please provide a conversational answer."} 
        ]

        try:
            resp = await self._openai_client.chat.completions.create(
                model="gpt-4o", 
                messages=messages,
                temperature=0.7 
            )
            
            content = resp.choices[0].message.content
            return content.strip()
        except Exception as e:
            # Fallback or error handling
            print(f"Error generating conversational answer: {e}")
            # Fallback to returning the raw tool result as a string
            return f"I found this information: {json.dumps(tool_result)}"

    async def close(self) -> None:
        if self._ws:
            await self._ws.close()


async def main():
    assistant = ChatAssistant(
        uri="ws://localhost:8000/mcp",
        mcp_api_key=os.getenv("MCP_CLIENT_API_KEY", ""),
        openai_api_key=os.getenv("OPENAI_API_KEY", "")
    )
    await assistant.connect()
    answer = await assistant.handle_user_query(
        "Find all people who are older than 30 years old and provide their first name, last name, and email address."
    )
    print("Answer:", answer)
    await assistant.close()

if __name__ == "__main__":
    asyncio.run(main())
