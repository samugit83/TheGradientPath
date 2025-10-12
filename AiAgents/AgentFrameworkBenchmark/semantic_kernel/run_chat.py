"""
Simple Chat Loop for Semantic Kernel Handoff Orchestration
"""
import logging
import os
import uuid
from semantic_kernel.agents.runtime import InProcessRuntime
from semantic_kernel.contents import ChatMessageContent, AuthorRole
from state import State, calculate_usage_from_content, update_usage, calculate_cost, get_usage_summary
from session_manager import init_session_manager

logger = logging.getLogger(__name__)


class RuntimeChat:
    """
    A chat runtime that manages the Semantic Kernel handoff orchestration
    """
    
    def __init__(self, handoff_orchestration, session_id: str = None):
        """
        Initialize the chat runtime
        
        Args:
            handoff_orchestration: The configured handoff orchestration from handoff_agents
            session_id: Optional session ID, will generate one if not provided
        """
        self.handoff_orchestration = handoff_orchestration
        self.runtime = None
        self.is_running = False
        
        # Initialize State to track usage across the session (following vanilla pattern)
        self.state = State()
        
        # Use provided session_id or generate one
        if session_id:
            self.state.session_id = session_id
        else:
            self.state.session_id = f"session_{uuid.uuid4().hex[:8]}"
        
        # Initialize unified session manager (following vanilla pattern)
        self.session_manager = init_session_manager(
            session_id=self.state.session_id,
            use_persistent=self.state.use_persistent_memory,
            db_path="conversations.db",
            max_history=40
        )
        logger.info(f"Initialized session manager: {self.state.session_id} "
                   f"(mode: {'persistent' if self.state.use_persistent_memory else 'in-memory'})")
        
        # Set reference to this RuntimeChat instance on the handoff_orchestration callback
        # so it can access state and session_manager
        if hasattr(self.handoff_orchestration, 'agent_response_callback'):
            self.handoff_orchestration.agent_response_callback._runtime_chat = self
    
    def _convert_history_to_chat_messages(self, history: list) -> list:
        """
        Convert session history to Semantic Kernel ChatMessageContent format
        
        Args:
            history: List of history items from session manager
            
        Returns:
            List of ChatMessageContent objects for Semantic Kernel
        """
        chat_messages = []
        
        for item in history:
            try:
                # Map subject to AuthorRole
                if item.get('subject') == 'user':
                    role = AuthorRole.USER
                else:
                    role = AuthorRole.ASSISTANT
                
                # Create ChatMessageContent
                chat_message = ChatMessageContent(
                    role=role,
                    content=item.get('content', ''),
                    name=item.get('agent_name') if role == AuthorRole.ASSISTANT else None
                )
                chat_messages.append(chat_message)
                
            except Exception as e:
                logger.warning(f"Failed to convert history item to ChatMessageContent: {e}")
                continue
        
        return chat_messages
    
    def _format_conversation_context(self, last_n: int = 10) -> str:
        """
        Format conversation history as a well-formatted string for task context
        
        Args:
            last_n: Number of recent messages to retrieve
            
        Returns:
            Formatted conversation history string
        """
        try:
            # Get history from session manager
            history = self.session_manager.get_conversation_history(last_n=last_n)
            
            if not history:
                return ""
            
            # Format each message
            formatted_messages = []
            for item in history:
                subject = item.get('subject', 'unknown')
                content = item.get('content', '')
                agent_name = item.get('agent_name')
                
                if subject == 'user':
                    formatted_messages.append(f"User: {content}")
                else:
                    if agent_name:
                        formatted_messages.append(f"{agent_name}: {content}")
                    else:
                        formatted_messages.append(f"Assistant: {content}")
            
            context = "\n".join(formatted_messages)
            return context
            
        except Exception as e:
            logger.error(f"Failed to format conversation context: {e}")
            return ""
    
    async def start(self):
        """Start the chat runtime"""
        if self.is_running:
            logger.warning("Chat runtime is already running")
            return
        
        self._display_welcome_message()
        
        # Start the runtime
        self.runtime = InProcessRuntime()
        self.runtime.start()
        self.is_running = True
        
        logger.info("Chat runtime started successfully")
    
    async def stop(self):
        """Stop the chat runtime"""
        if not self.is_running:
            logger.warning("Chat runtime is not running")
            return
        
        try:
            # Stop the runtime
            await self.runtime.stop_when_idle()
            self.is_running = False
            logger.info("Runtime stopped and cleaned up")
        except Exception as e:
            logger.error(f"Error stopping runtime: {e}")
    
    async def run(self):
        """Run the interactive chat loop"""
        if not self.is_running:
            await self.start()
        
        try:
            # Main chat loop
            while True:
                try:
                    print("\n" + "-" * 40)
                    user_input = input("💬 You > ").strip()
                    
                    if not user_input:
                        continue
                    
                    # Handle special commands (following vanilla pattern)
                    if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
                        print("\n👋 Goodbye! Thanks for using the AI-powered chat.")
                        # Display final usage stats (following vanilla pattern)
                        if self.state.requests > 0:
                            print(f"\n{get_usage_summary(self.state)}")
                        break
                    
                    if user_input.lower() in ['clear', 'clear history']:
                        self.session_manager.clear_session()
                        print("✨ Conversation history cleared!")
                        continue
                
                    # Track the state BEFORE this turn (following vanilla pattern)
                    prev_requests = self.state.requests
                    prev_input = self.state.input_tokens
                    prev_output = self.state.output_tokens
                    prev_total = self.state.total_tokens
                    
                    # Log user message to session (following vanilla pattern)
                    self.session_manager.add_user_message(user_input)
                    
                    # Get conversation history and format it as a string for the task
                    conversation_context = self._format_conversation_context(last_n=10)
                    
                    # Create task with conversation context and current user input
                    if conversation_context:
                        formatted_task = f"Conversation history:\n{conversation_context}\n\nCurrent user message: {user_input}"
                    else:
                        formatted_task = user_input
                    
                    logger.info("🔀 Processing message through handoff orchestration...")
                    # Invoke orchestration with formatted task including conversation context
                    orchestration_result = await self.handoff_orchestration.invoke(
                        task=formatted_task,
                        runtime=self.runtime
                    )
                    
                    result = await orchestration_result.get()
                    
                    # Calculate token usage manually since Semantic Kernel streaming doesn't provide usage metadata
                    try:
                        # Get the agent response content
                        output_content = ""
                        if hasattr(result, 'content') and result.content:
                            output_content = result.content
                        
                        # Calculate usage using tiktoken (manual counting for streaming responses)
                        usage = calculate_usage_from_content(
                            input_text=formatted_task,
                            output_text=output_content,
                            model=self.state.model_name
                        )
                        
                        if usage["requests"] > 0:
                            update_usage(self.state, usage)
                            logger.info(f"Calculated token usage - Input: {usage['input_tokens']}, "
                                      f"Output: {usage['output_tokens']}, Total: {usage['total_tokens']}")
                        
                    except Exception as e:
                        logger.warning(f"Could not calculate token usage: {e}")
                        # Fallback: just count the request
                        update_usage(self.state, {"requests": 1, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                    
                    # Calculate and display turn usage if there was LLM activity (following vanilla pattern)
                    turn_requests = self.state.requests - prev_requests
                    turn_input = self.state.input_tokens - prev_input
                    turn_output = self.state.output_tokens - prev_output
                    turn_total = self.state.total_tokens - prev_total
                    
                    if turn_requests > 0:
                        # Calculate turn cost
                        turn_input_cost = (turn_input / 1_000_000) * self.state.input_token_price_per_million
                        turn_output_cost = (turn_output / 1_000_000) * self.state.output_token_price_per_million
                        turn_cost = turn_input_cost + turn_output_cost
                        
                        # Calculate cumulative cost
                        total_cost = calculate_cost(self.state)
                        logger.info(f"\n📊 This turn: {turn_input:,} in, {turn_output:,} out, {turn_total:,} total tokens ({turn_requests} requests) - Cost: ${turn_cost:.4f}")
                        logger.info(f"\n📊 Cumulative: {self.state.input_tokens:,} in, {self.state.output_tokens:,} out, {self.state.total_tokens:,} total tokens ({self.state.requests} requests) - Total cost: ${total_cost:.4f}")
                    
                    # The agent responses are already printed by the agent_response_callback
                    # We don't need to print them again here

                except KeyboardInterrupt:
                    print("\n\n⚠️  Interrupted! Type 'exit' to quit or continue chatting.")
                    # Display final usage stats on interrupt (following vanilla pattern)
                    if self.state.requests > 0:
                        print(f"\n{get_usage_summary(self.state)}")
                except Exception as e:
                    print(f"\n❌ Error: {e}")
                    logger.error(f"REPL error: {e}", exc_info=True)
                    
        finally:
            await self.stop()
    

    
    def _display_welcome_message(self):
        """Display the welcome message and instructions"""
        print("\n" + "=" * 60)
        print("  🤖 Welcome to AI-Powered Chat with Handoff Orchestration! 🚀")
        print("=" * 60)
        
        print("\n💬 Chat Instructions:")
        print("  • Just type naturally and I'll understand what you need!")
        print("  • Type 'quit' or 'exit' to leave the chat")
        print("  • I'll intelligently route your request to the right specialist:")
        print("    - ⚖️ Legal Expert: For law-related questions and legal topics")
        print("    - 🔧 General Agent: For everything else (programming, tools, general knowledge)")
        print("\n🛡️  Content Safety:")
        print("  • The routing agent has content safety guardrails enabled")
        print("  • Requests for illegal, harmful, or inappropriate content will be blocked")
        print("  • The system prioritizes safe and helpful interactions\n")

async def run_chat_loop(handoff_orchestration, session_id: str = None):
    """
    Convenience function to run the chat loop
    
    Args:
        handoff_orchestration: The configured handoff orchestration from handoff_agents
        session_id: Optional session ID to use
    """
    chat_runtime = RuntimeChat(handoff_orchestration, session_id)
    await chat_runtime.run()
