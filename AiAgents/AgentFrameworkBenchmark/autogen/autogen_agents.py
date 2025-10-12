"""
AutoGen-based agents that wrap the existing functionality
"""
import json
import logging
from typing import List
import asyncio

from autogen_core import (
    MessageContext, 
    RoutedAgent, 
    message_handler,
    CancellationToken,
    FunctionCall,
)
from autogen_core.models import (
    LLMMessage,
    SystemMessage,
    UserMessage,
    AssistantMessage,
    FunctionExecutionResult,
    FunctionExecutionResultMessage,
)
from autogen_core.tools import FunctionTool, Tool, ToolResult, Workbench
from autogen_ext.models.openai import OpenAIChatCompletionClient

from tools import get_all_tool_functions
from state import track_usage_from_result
from session_manager import get_global_session_manager
from prompts import (
    LEGAL_EXPERT_SYSTEM_PROMPT,
    GENERAL_AGENT_SYSTEM_PROMPT,
    ROUTER_AGENT_PROMPT_TEMPLATE,
    CONTENT_SAFETY_PROMPT
)

logger = logging.getLogger(__name__)

class LegalExpertAgent(RoutedAgent):
    """AutoGen agent for legal expertise using core AutoGen"""
    
    def __init__(
        self, 
        name: str = "legal_expert", 
        state = None
    ) -> None:
        super().__init__(name)

        self.state = state
        self.model_client = OpenAIChatCompletionClient(model="gpt-4o")
        self._system_messages: List[LLMMessage] = [
            SystemMessage(content=LEGAL_EXPERT_SYSTEM_PROMPT)
        ]
    
    @message_handler
    async def handle_user_message(self, message: UserMessage, ctx: MessageContext) -> str:
        """Handle user messages related to legal topics"""
        logger.info(f"⚖️ Legal Expert processing: {message.content[:50]}...")
        
        try:
            # Handle message context using global session manager
            session_manager = get_global_session_manager(self.state.session_id) if self.state and self.state.session_id else None
            if session_manager:
                session_manager.add_session_message(message)
                context_messages = session_manager.get_recent_session_messages(count=10)
                # Filter out system messages to avoid duplication
                context_messages = [msg for msg in context_messages if not isinstance(msg, SystemMessage)]
                all_messages = self._system_messages + context_messages
            else:
                # Fallback to just the current message if no session manager
                all_messages = self._system_messages + [message]
            
            # Get response from model client
            create_result = await self.model_client.create(
                messages=all_messages,
                cancellation_token=ctx.cancellation_token,
            )
            
            # Track usage after LLM call (following vanilla pattern)
            if self.state:
                track_usage_from_result(create_result, self.state)
            
            assert isinstance(create_result.content, str)
            
            # Create assistant message and handle context storage
            assistant_message = AssistantMessage(content=create_result.content, source="legal_expert")
            
            if session_manager:
                # Log assistant response to persistent storage
                session_manager.add_session_message(assistant_message)
            
            logger.info("⚖️ Legal Expert response generated")
            return create_result.content
            
        except Exception as e:
            logger.error(f"Legal Expert error: {e}")
            return f"❌ Sorry, I encountered an error while processing your legal question: {str(e)}"


class GeneralToolAgent(RoutedAgent):
    """AutoGen agent that uses tools for general tasks"""
    
    def __init__(
        self, 
        name: str = "general_agent", 
        state = None
    ) -> None:
        super().__init__(name)

        self.state = state
        self.model_client = OpenAIChatCompletionClient(model="gpt-4o")
        self.autogen_tools: List[Tool] = self._create_autogen_tools()
        self._system_messages: List[LLMMessage] = [
            SystemMessage(content=GENERAL_AGENT_SYSTEM_PROMPT)
        ]
    
    def _create_autogen_tools(self) -> List[Tool]:
        """Create AutoGen FunctionTools with state injection"""
        import inspect
        tools = []
        tool_functions = get_all_tool_functions()
        
        # Helper function to create a wrapper with captured state
        def make_wrapper(original_func, state):
            """Create a wrapper function that dynamically injects state"""
            sig = inspect.signature(original_func)
        
            clean_params = [
                (name, param) for name, param in sig.parameters.items() 
                if name != 'state'
            ]
            
            async def wrapper(**kwargs):
                kwargs['state'] = state
                return await original_func(**kwargs)
            
            new_params = [
                inspect.Parameter(name, param.kind, default=param.default, annotation=param.annotation)
                for name, param in clean_params
            ]
            wrapper.__signature__ = inspect.Signature(new_params, return_annotation=sig.return_annotation)
            wrapper.__name__ = original_func.__name__
            wrapper.__doc__ = original_func.__doc__
            wrapper.__annotations__ = {
                name: param.annotation for name, param in clean_params 
                if param.annotation != inspect.Parameter.empty
            }
            # Add return annotation
            if sig.return_annotation != inspect.Parameter.empty:
                wrapper.__annotations__['return'] = sig.return_annotation
                
            return wrapper
        
        # Create tools with wrappers
        for func_name, original_func in tool_functions.items():
            try:
                wrapper_func = make_wrapper(original_func, self.state)
                function_tool = FunctionTool(wrapper_func, description=original_func.__doc__ or f"Execute {func_name}")
                tools.append(function_tool)
                
            except Exception as e:
                logger.error(f"❌ Failed to register AutoGen tool {func_name}: {e}")
        
        return tools
    
    
    @message_handler
    async def handle_user_message(self, message: UserMessage, ctx: MessageContext) -> str:
        """Handle user messages with tool capabilities"""
        logger.info(f"🔧 General Agent processing: {message.content[:50]}...")
        
        try:
            # Handle message context using global session manager
            session_manager = get_global_session_manager(self.state.session_id) if self.state and self.state.session_id else None
            if session_manager:
                session_manager.add_session_message(message)
                context_messages = session_manager.get_recent_session_messages(count=10)
                # Filter out system messages to avoid duplication
                context_messages = [msg for msg in context_messages if not isinstance(msg, SystemMessage)]
                current_messages = self._system_messages + context_messages
            else:
                # Fallback to just the current message if no session manager
                current_messages = self._system_messages + [message]

            # First, try to get a response with tools
            create_result = await self.model_client.create(
                messages=current_messages,
                tools=self.autogen_tools,
                cancellation_token=ctx.cancellation_token,
            )
            
            # Track usage after first LLM call (following vanilla pattern)
            if self.state:
                track_usage_from_result(create_result, self.state)
            
            # If no tool calls, return the direct response
            if isinstance(create_result.content, str):
                # Create assistant message and handle context storage
                assistant_message = AssistantMessage(content=create_result.content, source="general_agent")
                
                if session_manager:
                    # Log assistant response to persistent storage
                    session_manager.add_session_message(assistant_message)
                
                return create_result.content
            
            # Handle tool calls
            assert isinstance(create_result.content, list) and all(
                isinstance(call, FunctionCall) for call in create_result.content
            )
            
            # Store the assistant message with tool calls in persistent session
            if session_manager:
                # Create a summary of what tools are being called
                tool_calls_summary = []
                for call in create_result.content:
                    try:
                        args_str = json.dumps(json.loads(call.arguments), indent=2) if call.arguments else "{}"
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.warning(f"Failed to parse tool arguments as JSON: {e}")
                        args_str = str(call.arguments) if call.arguments else "{}"
                    tool_calls_summary.append(f"Calling {call.name} with arguments: {args_str}")
                
                assistant_with_tools_msg = AssistantMessage(
                    content="I'll help you with that. " + "; ".join(tool_calls_summary),
                    source="general_agent"
                )
                session_manager.add_session_message(assistant_with_tools_msg)
            
            # Execute tool calls
            results = await asyncio.gather(
                *[self._execute_tool_call(call, ctx.cancellation_token) for call in create_result.content]
            )
            
            function_result_message = FunctionExecutionResultMessage(content=results)
            
            if session_manager:
                context_messages = session_manager.get_recent_session_messages(count=10)
                # Filter out system and tool messages
                context_messages = [
                    msg for msg in context_messages 
                    if not isinstance(msg, (SystemMessage, FunctionExecutionResultMessage))
                ]
                
                # Create a summary message of tool results instead of using FunctionExecutionResultMessage
                tool_results_summary = []
                for result in results:
                    tool_name = result.name
                    if result.is_error:
                        tool_results_summary.append(f"Tool {tool_name} failed: {result.content}")
                    else:
                        tool_results_summary.append(f"Tool {tool_name} result: {result.content}")
                
                # Add tool results as a simulated user message for context
                tool_summary_message = UserMessage(
                    content=f"Tool execution results:\n" + "\n".join(tool_results_summary),
                    source="tool_system"
                )
                
                final_messages = self._system_messages + context_messages + [tool_summary_message]
            else:
                # Fallback: include proper sequence: context + assistant_with_tools + tool_results
                assistant_with_tools = AssistantMessage(content=create_result.content, source="general_agent")
                final_messages = current_messages + [assistant_with_tools, function_result_message]
            
            # Get final response reflecting on tool results
            final_result = await self.model_client.create(
                messages=final_messages,
                cancellation_token=ctx.cancellation_token,
            )
            
            # Track usage after final LLM call (following vanilla pattern)
            if self.state:
                track_usage_from_result(final_result, self.state)
            
            assert isinstance(final_result.content, str)
            
            # Create assistant message and handle context storage
            assistant_message = AssistantMessage(content=final_result.content, source="general_agent")
            
            if session_manager:
                # Log assistant response to persistent storage
                session_manager.add_session_message(assistant_message)
                
                # Also log tool execution if there were tool calls
                for call in create_result.content:
                    if isinstance(call, FunctionCall):
                        # Find corresponding result
                        result_obj = next((r for r in results if r.call_id == call.id), None)
                        if result_obj:
                            # Create a FunctionExecutionResultMessage for tool results
                            # FunctionExecutionResultMessage expects content to be a list of FunctionExecutionResult objects
                            execution_result = FunctionExecutionResult(
                                call_id=call.id,
                                name=call.name,
                                content=result_obj.content,
                                is_error=result_obj.is_error if hasattr(result_obj, 'is_error') else False
                            )
                            
                            tool_result_message = FunctionExecutionResultMessage(
                                content=[execution_result],
                                source=f"Tool-{call.name}"
                            )
                            session_manager.add_session_message(tool_result_message)
            
            logger.info("🔧 General Agent response with tools completed")
            return final_result.content
            
        except Exception as e:
            logger.error(f"General Agent error: {e}")
            return f"❌ Sorry, I encountered an error: {str(e)}"
    
    async def _execute_tool_call(
        self, call: FunctionCall, cancellation_token: CancellationToken
    ) -> FunctionExecutionResult:
        """Execute a tool call and return the result"""
        # Find the tool by name
        tool = next((tool for tool in self.autogen_tools if tool.name == call.name), None)
        if tool is None:
            return FunctionExecutionResult(
                call_id=call.id, 
                content=f"Tool {call.name} not found", 
                is_error=True, 
                name=call.name
            )
        
        try:
            # Parse arguments and execute tool
            try:
                arguments = json.loads(call.arguments) if call.arguments else {}
            except (json.JSONDecodeError, TypeError) as e:
                logger.error(f"Failed to parse tool arguments as JSON: {e}")
                return FunctionExecutionResult(
                    call_id=call.id,
                    content=f"Invalid JSON arguments: {str(e)}",
                    is_error=True,
                    name=call.name
                )
            result = await tool.run_json(arguments, cancellation_token)
            
            return FunctionExecutionResult(
                call_id=call.id,
                content=tool.return_value_as_string(result),
                is_error=False,
                name=tool.name
            )
            
        except Exception as e:
            logger.error(f"Tool execution error for {call.name}: {e}")
            return FunctionExecutionResult(
                call_id=call.id,
                content=str(e),
                is_error=True,
                name=tool.name
            )


class RouterAgent(RoutedAgent):
    """AutoGen agent that routes messages to appropriate specialist agents"""
    
    def __init__(self, name: str = "router", state = None, workbench_tools = None) -> None:
        super().__init__(name)
        self.model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")  # Use fast model for routing
        self.state = state
        self.guardrail_client = OpenAIChatCompletionClient(model="gpt-4o-mini")  # Fast model for safety checks
        self.workbench_tools = workbench_tools or []
        
    def _get_workbench_tools_description(self) -> str:
        """Generate a dynamic description of available workbench tools"""
        if not self.workbench_tools:
            return "- No advanced tools currently available (do not route to workbench_agent)"
        
        tool_descriptions = []
        for tool in self.workbench_tools:
            # Get tool name and description
            tool_name = getattr(tool, 'name', 'Unknown tool')
            tool_desc = getattr(tool, 'description', 'No description available')
            
            # Format the tool information
            tool_descriptions.append(f"- {tool_name}: {tool_desc}")
        
        return "\n".join(tool_descriptions)
    
    async def _check_content_safety(self, message: UserMessage, ctx: MessageContext) -> str:
        """Check content safety using integrated guardrail functionality"""
        logger.info(f"🛡️ Guardrail analyzing: {message.content[:50]}...")
        
        try:
            # Create safety check prompt
            safety_prompt = f"{CONTENT_SAFETY_PROMPT}\n\nUser message to analyze: {message.content}"
            
            # Get safety assessment
            messages = [UserMessage(content=safety_prompt, source="user")]
                
            create_result = await self.guardrail_client.create(
                messages=messages,
                cancellation_token=ctx.cancellation_token
            )
            
            # Track usage after LLM call (following vanilla pattern)
            if self.state:
                track_usage_from_result(create_result, self.state)
            
            assert isinstance(create_result.content, str)
            safety_result = create_result.content.strip()
            
            # Try to parse JSON response
            try:
                safety_data = json.loads(safety_result)
                
                if safety_data.get("is_inappropriate", False):
                    logger.warning(f"🛡️ Content flagged: {safety_data.get('category', 'unknown')}")
                    return "guardrail_alert"
                else:
                    logger.info("🛡️ Content approved by guardrail")
                    return "content_approved"
                    
            except json.JSONDecodeError:
                logger.error(f"Failed to parse guardrail JSON response: {safety_result}")
                # Default to approved on parsing error to avoid blocking legitimate content
                return "content_approved"
            
        except Exception as e:
            logger.error(f"Guardrail error: {e}")
            # Default to approved on error to avoid blocking legitimate content
            return "content_approved"
    
    @message_handler
    async def handle_user_message(self, message: UserMessage, ctx: MessageContext) -> str:
        """Route user messages to the appropriate agent"""
        logger.info(f"🔀 Router analyzing: {message.content[:50]}...")
        
        try:
            # First, check content safety
            safety_result = await self._check_content_safety(message, ctx)
            
            if safety_result == "guardrail_alert":
                return "guardrail_alert"
            
            # If content is safe, proceed with routing
            # Create routing prompt using template with dynamic workbench tools
            workbench_tools_desc = self._get_workbench_tools_description()
            routing_prompt = ROUTER_AGENT_PROMPT_TEMPLATE.format(
                user_message=message.content,
                workbench_tools_description=workbench_tools_desc
            )

            # Get routing decision
            messages = [UserMessage(content=routing_prompt, source="user")]
                
            create_result = await self.model_client.create(
                messages=messages,
                cancellation_token=ctx.cancellation_token
            )
            
            # Track usage after LLM call (following vanilla pattern)
            if self.state:
                track_usage_from_result(create_result, self.state)
            
            assert isinstance(create_result.content, str)
            agent_choice = create_result.content.strip().lower()
            
            # Validate and determine target agent
            if "legal_expert" in agent_choice:
                target_agent = "legal_expert"
                logger.info("⚖️ Routing to Legal Expert")
            elif "workbench_agent" in agent_choice and self.workbench_tools:
                target_agent = "workbench_agent"
                logger.info("🛠️ Routing to Workbench Agent")
            else:
                target_agent = "general_agent"
                logger.info("🔧 Routing to General Agent")
            
            return f"ROUTE_TO:{target_agent}"  # Special response format for routing
            
        except Exception as e:
            logger.error(f"Router error: {e}")
            # Default to general agent on error
            return "ROUTE_TO:general_agent"


class GuardrailAgent(RoutedAgent):
    """AutoGen agent for content safety filtering"""
    
    def __init__(self, name: str = "guardrail", state = None) -> None:
        super().__init__(name)
        self.model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")  # Use fast model for safety checks
        self.state = state
        
    @message_handler
    async def handle_user_message(self, message: UserMessage, ctx: MessageContext) -> str:
        """Analyze user message for content safety"""
        logger.info(f"🛡️ Guardrail analyzing: {message.content[:50]}...")
        
        try:
            # Create safety check prompt
            safety_prompt = f"{CONTENT_SAFETY_PROMPT}\n\nUser message to analyze: {message.content}"
            
            # Get safety assessment
            messages = [UserMessage(content=safety_prompt, source="user")]
                
            create_result = await self.model_client.create(
                messages=messages,
                cancellation_token=ctx.cancellation_token
            )
            
            # Track usage after LLM call (following vanilla pattern)
            if self.state:
                track_usage_from_result(create_result, self.state)
            
            assert isinstance(create_result.content, str)
            safety_result = create_result.content.strip()
            
            # Try to parse JSON response
            try:
                import json
                safety_data = json.loads(safety_result)
                
                if safety_data.get("is_inappropriate", False):
                    logger.warning(f"🛡️ Content flagged: {safety_data.get('category', 'unknown')}")
                    return "guardrail_alert"
                else:
                    logger.info("🛡️ Content approved by guardrail")
                    return "content_approved"
                    
            except json.JSONDecodeError:
                logger.error(f"Failed to parse guardrail JSON response: {safety_result}")
                # Default to approved on parsing error to avoid blocking legitimate content
                return "content_approved"
            
        except Exception as e:
            logger.error(f"Guardrail error: {e}")
            # Default to approved on error to avoid blocking legitimate content
            return "content_approved"




class WorkbenchAgent(RoutedAgent):
    """AutoGen agent with workbench tools using session manager pattern"""
    
    def __init__(
        self, 
        model_client,
        workbench: Workbench,
        state = None,
        name: str = "workbench_agent"
    ) -> None:
        super().__init__(name)
        self._system_messages: List[LLMMessage] = [SystemMessage(content="You are a helpful AI assistant with access to various tools through the workbench.")]
        self.model_client = model_client
        self._workbench = workbench
        self.state = state

    @message_handler
    async def handle_user_message(self, message: UserMessage, ctx: MessageContext) -> str:
        """Handle user message with workbench tools using session manager pattern"""
        logger.info(f"🔧 WorkbenchAgent processing message: {message.content[:100]}...")
        
        try:
            # Handle message context using global session manager
            session_manager = get_global_session_manager(self.state.session_id) if self.state and self.state.session_id else None
            if session_manager:
                session_manager.add_session_message(message)
                context_messages = session_manager.get_recent_session_messages(count=10)
                # Filter out system messages to avoid duplication
                context_messages = [msg for msg in context_messages if not isinstance(msg, SystemMessage)]
                all_messages = self._system_messages + context_messages
            else:
                # In-memory mode - just use system messages + current message
                user_msg = UserMessage(content=message.content, source="user")
                all_messages = self._system_messages + [user_msg]

            # Run the chat completion with the tools.
            create_result = await self.model_client.create(
                messages=all_messages,
                tools=(await self._workbench.list_tools()),
                cancellation_token=ctx.cancellation_token,
            )
            
            # Track usage
            if self.state:
                track_usage_from_result(create_result, self.state)

            # Run tool call loop.
            while isinstance(create_result.content, list) and all(
                isinstance(call, FunctionCall) for call in create_result.content
            ):
                logger.info("---------MCP Function Calls-----------")
                for call in create_result.content:
                    print(call)

                # Store the assistant message with tool calls in persistent session
                if session_manager:
                    tool_calls_summary = []
                    for call in create_result.content:
                        try:
                            args = json.loads(call.arguments)
                            args_str = ", ".join([f"{k}={v}" for k, v in args.items()])
                            tool_calls_summary.append(f"calling {call.name}({args_str})")
                        except:
                            tool_calls_summary.append(f"calling {call.name}")
                    
                    assistant_with_tools_msg = AssistantMessage(
                        content="I'll help you with that. " + "; ".join(tool_calls_summary),
                        source="workbench_agent"
                    )
                    session_manager.add_session_message(assistant_with_tools_msg)

                # Call the tools using the workbench.
                logger.info("---------MCP Function Call Results-----------")
                results: List[ToolResult] = []
                for call in create_result.content:
                    result = await self._workbench.call_tool(
                        call.name, arguments=json.loads(call.arguments), cancellation_token=ctx.cancellation_token
                    )
                    results.append(result)
                    print(result)

                # Create function execution result message
                function_result_message = FunctionExecutionResultMessage(
                    content=[
                        FunctionExecutionResult(
                            call_id=call.id,
                            content=result.to_text(),
                            is_error=result.is_error,
                            name=result.name,
                        )
                        for call, result in zip(create_result.content, results, strict=False)
                    ]
                )
                
                if session_manager:
                    # For session manager mode, we need to construct proper message sequence:
                    # [system] + [conversation history] + [assistant_with_tools] + [tool_results]
                    context_messages = session_manager.get_recent_session_messages(count=10)
                    # Filter out only system messages to avoid duplication
                    context_messages = [msg for msg in context_messages if not isinstance(msg, SystemMessage)]
                    
                    # Create assistant message with tool calls for proper OpenAI sequence
                    assistant_with_tools = AssistantMessage(content=create_result.content, source="workbench_agent")
                    
                    # Prepare messages for next round with proper sequence
                    messages_for_completion = (
                        self._system_messages + 
                        context_messages + 
                        [assistant_with_tools, function_result_message]
                    )
                else:
                    # In-memory mode - build message history manually with proper sequence
                    assistant_with_tools = AssistantMessage(content=create_result.content, source="workbench_agent") 
                    messages_for_completion = all_messages + [assistant_with_tools, function_result_message]

                # Run the chat completion again to reflect on the history and function execution results.
                create_result = await self.model_client.create(
                    messages=messages_for_completion,
                    tools=(await self._workbench.list_tools()),
                    cancellation_token=ctx.cancellation_token,
                )
                
                # Track usage for the follow-up call
                if self.state:
                    track_usage_from_result(create_result, self.state)

            # Now we have a single message as the result.
            assert isinstance(create_result.content, str)

            # Create assistant message and handle context storage
            assistant_message = AssistantMessage(content=create_result.content, source="workbench_agent")
            
            if session_manager:
                # Log assistant response to persistent storage
                session_manager.add_session_message(assistant_message)
            
            logger.info("🔧 WorkbenchAgent response with tools completed")
            return create_result.content
            
        except Exception as e:
            logger.error(f"WorkbenchAgent error: {e}", exc_info=True)
            return f"I encountered an error: {str(e)}"
