"""
AutoGen Runtime Integration for the Chat System
"""
import logging
import uuid
import os
from typing import Optional
from autogen_core import (
    SingleThreadedAgentRuntime,
    AgentId,
)
from autogen_core.models import UserMessage
from autogen_agents import (
    LegalExpertAgent,
    GeneralToolAgent,
    RouterAgent,
    WorkbenchAgent
)

from code_generator_multiagent.coder import CoderAgent
from code_generator_multiagent.tester import TesterAgent
from code_generator_multiagent.reviewer import ReviewerAgent
from state import State, get_state_usage_summary, calculate_state_cost
from session_manager import init_session_manager, set_global_context, get_global_session_manager, cleanup_global_context
from autogen_ext.models.openai import OpenAIChatCompletionClient
from openai import OpenAI
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams, mcp_server_tools

logger = logging.getLogger(__name__)


async def get_mcp_servers():
    """
    Initialize and return all configured MCP servers adapted for AutoGen.
    
    Returns:
        List of tools from all MCP servers
    """
    all_tools = []
    
    weather_api_key = os.environ.get("ACCUWEATHER_API_KEY")
    if weather_api_key:
        try:
            server_params = StdioServerParams(
                command="npx",
                args=["-y", "@timlukahorstmann/mcp-weather"],
                env={
                    **os.environ,
                    "ACCUWEATHER_API_KEY": weather_api_key
                }
            )
            # Get tools from the MCP server
            weather_tools = await mcp_server_tools(server_params)
            all_tools.extend(weather_tools)
            logger.info("Weather MCP server initialized and connected successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Weather MCP server: {e}")
    else:
        logger.warning("ACCUWEATHER_API_KEY not set. Weather MCP server will not be available.")
    
    return all_tools


class AutoGenChatSystem:
    """
    AutoGen-based chat system that replaces the existing ChatLoop
    """
    
    def __init__(self, session_id: str = None):
        """Initialize the AutoGen chat system"""
        # Initialize State to track usage across the session
        self.state = State()
        
        if session_id:
            self.state.session_id = session_id
        else:
            self.state.session_id = f"autogen_session_{uuid.uuid4().hex[:8]}"
        
        session_manager = init_session_manager(
            session_id=self.state.session_id,
            use_persistent_memory=self.state.use_persistent_memory,
            db_path="conversations.db",
            buffer_size=10
        )
        
        # Set session_manager in global context
        set_global_context(self.state.session_id, session_manager=session_manager)
        
        if self.state.use_persistent_memory:
            logger.info(f"Initialized persistent session: {self.state.session_id}")
        else:
            logger.info(f"Initialized in-memory session: {self.state.session_id}")
        
        # Initialize AutoGen runtime and agents
        self.runtime = None
        self.runtime_started = False
        self.workbench = None  # Store workbench reference for cleanup
        self.router_agent_id = AgentId("router", "default")
        self.legal_agent_id = AgentId("legal_expert", "default") 
        self.general_agent_id = AgentId("general_agent", "default")
        self.workbench_agent_id = AgentId("workbench_agent", "default")
        self._model_client = OpenAIChatCompletionClient(model=self.state.model_name)
        self.openai_client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY")
        )
        
        # Runtime will be initialized lazily when needed
    
    async def _initialize_runtime(self):
        """Initialize AutoGen runtime and register agents"""
        try:
            logger.info("Initializing AutoGen runtime...")

            # Get MCP tools first
            mcp_tools = await get_mcp_servers()
            
            # Create workbench with MCP server parameters
            mcp_server_params = None
            weather_api_key = os.environ.get("ACCUWEATHER_API_KEY")
            if weather_api_key:
                mcp_server_params = StdioServerParams(
                    command="npx",
                    args=["-y", "@timlukahorstmann/mcp-weather"],
                    env={
                        **os.environ,
                        "ACCUWEATHER_API_KEY": weather_api_key
                    }
                )
            
            self.runtime = SingleThreadedAgentRuntime()
                
            # Register agents with the runtime, passing state which contains session_manager
            await RouterAgent.register(
                    self.runtime,
                    "router",
                    lambda: RouterAgent("router", state=self.state, workbench_tools=mcp_tools)
                )
            await LegalExpertAgent.register(
                    self.runtime, 
                    "legal_expert",
                    lambda: LegalExpertAgent(
                        "legal_expert", 
                        state=self.state
                    )
                )
                
            await GeneralToolAgent.register(
                    self.runtime,
                    "general_agent", 
                    lambda: GeneralToolAgent(
                        "general_agent", 
                        state=self.state
                    )
                )

            # Register WorkbenchAgent with MCP tools if available
            if mcp_server_params:
                try:
                    self.workbench = McpWorkbench(mcp_server_params)
                    await self.workbench.start()  # Properly initialize the workbench
                    
                    await WorkbenchAgent.register(
                        self.runtime,
                        "workbench_agent",
                        lambda: WorkbenchAgent(
                            model_client=self._model_client,
                            workbench=self.workbench,
                            state=self.state,
                            name="workbench_agent"
                        )
                    )
                    logger.info("✅ WorkbenchAgent registered with MCP tools")
                except Exception as e:
                    logger.warning(f"Failed to register WorkbenchAgent: {e}")
            else:
                logger.info("No MCP server parameters available, skipping WorkbenchAgent registration")


            await CoderAgent.register(
                    self.runtime, 
                    type="CodingRoom", 
                    factory=lambda: CoderAgent(
                        model_client=self.openai_client
                    )
                )

            await TesterAgent.register(
                    self.runtime, 
                    type="TestingRoom", 
                    factory=lambda: TesterAgent(
                        model_client=self.openai_client
                    )
                )

            await ReviewerAgent.register(
                    self.runtime, 
                    type="ReviewingRoom", 
                    factory=lambda: ReviewerAgent(
                        model_client=self.openai_client
                    )
                )


            # Update runtime in global context (preserve existing session_manager)
            existing_session_manager = get_global_session_manager(self.state.session_id)
            set_global_context(self.state.session_id, session_manager=existing_session_manager, runtime=self.runtime)
            
            logger.info("✅ AutoGen runtime initialized with all agents")
                
        except Exception as e:
            logger.error(f"Failed to initialize AutoGen runtime: {e}")
            raise
        
    def _handle_special_commands(self, user_input: str) -> Optional[str]:
        """Handle special commands like exit, clear, etc."""
        if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
            return "SYSTEM_EXIT"
        
        if user_input.lower() in ['clear', 'clear history']:
            session_manager = get_global_session_manager(self.state.session_id)
            if session_manager:
                session_manager.clear_session()
                return "✨ Conversation history cleared!"
            else:
                # For in-memory mode, no persistent session manager available
                return "✨ In-memory mode - no persistent history to clear!"
        
        return None
    
    async def process_message(self, user_input: str) -> str:
        """
        Process user message using AutoGen agents
        
        Args:
            user_input: The user's input message
            
        Returns:
            Response string from the appropriate agent
        """
        try:
            # Handle special commands first
            special_response = self._handle_special_commands(user_input)
            if special_response:
                return special_response
            
            # Ensure runtime is initialized and started
            if not self.runtime:
                await self._initialize_runtime()
            
            if not self.runtime_started:
                self.runtime.start()
                self.runtime_started = True
            
            try:
                message = UserMessage(content=user_input, source="user")
                self.state.user_prompt = user_input
                
                logger.info("🔀 Routing message...")
                router_response = await self.runtime.send_message(
                    message, 
                    self.router_agent_id
                )
                
                # Parse routing response
                if isinstance(router_response, str) and router_response == "guardrail_alert":
                    logger.warning("🛡️ Content blocked by guardrail")
                    return "🛡️ I'm sorry, but I cannot process this request as it appears to contain inappropriate content. Please rephrase your request or ask something else."
                
                elif isinstance(router_response, str) and router_response.startswith("ROUTE_TO:"):
                    target_agent_type = router_response.replace("ROUTE_TO:", "")
                    logger.info(f"📍 Routed to: {target_agent_type}")
                    
                    # Create new message for the final agent
                    final_message = UserMessage(content=user_input, source="user")
                    
                    # Send message to the appropriate agent
                    if target_agent_type == "legal_expert":
                        final_response = await self.runtime.send_message(
                            final_message, 
                            self.legal_agent_id
                        )
                    elif target_agent_type == "workbench_agent":
                        final_response = await self.runtime.send_message(
                            final_message, 
                            self.workbench_agent_id
                        )
                    else:  # default to general_agent
                        final_response = await self.runtime.send_message(
                            final_message, 
                            self.general_agent_id
                        )
                    
                    response_content = str(final_response)
                    
                    return response_content
                    
                else:
                    logger.warning("Routing failed, using general agent")
                    final_response = await self.runtime.send_message(
                        message,
                        self.general_agent_id
                    )
                    
                    response_content = str(final_response)
                    
                    return response_content
                    
            finally:
                pass
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return f"❌ Sorry, I encountered an error: {str(e)}"
    
    async def run_interactive(self):
        """Run the interactive chat loop"""
        print("\n" + "=" * 70)
        print("  🤖 Welcome to AutoGen-Powered AI Chat System! 🚀")
        print("=" * 70)
        
        # Show session information based on actual configuration
        if self.state.use_persistent_memory:
            print(f"\n🧠 Persistent memory enabled - Session ID: {self.state.session_id}")
            print("   Conversations will be saved to: conversations.db")
            session_manager = get_global_session_manager(self.state.session_id)
            if session_manager:
                stats = session_manager.get_session_stats()
                if stats.get('total_messages', 0) > 0:
                    print(f"   Resuming session with {stats['total_messages']} previous messages")
        else:
            print(f"\n🧠 In-memory mode enabled - Session ID: {self.state.session_id}")
            print("   Conversations will not be persisted between sessions")
        
        print("\n🤖 AutoGen Agents:")
        print("  • 🔀 Router Agent: Intelligently routes messages to specialists")
        print("  • ⚖️ Legal Expert: Specialized in law and legal matters") 
        print("  • 🔧 General Agent: Handles calculations, code generation, weather, etc.")
        print("  • 🛠️ Workbench Agent: Uses MCP tools for advanced functionality")
        
        print("\n💬 Smart Routing & Context:")
        print("  • I'll intelligently route your request to the right specialist")
        print("  • Conversation context is managed automatically by each agent")
        print("  • Just type naturally and I'll understand what you need!")
        print("\nType 'help' for more info or 'exit' to quit.\n")
        
        # Check API key
        if not os.environ.get("OPENAI_API_KEY"):
            print("⚠️  Warning: OPENAI_API_KEY not set!")
            print("Set it with: export OPENAI_API_KEY='your-key-here'")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                return
        
        # Main chat loop
        try:
            while True:
                try:
                    print("\n" + "-" * 40)
                    user_input = input("💬 You > ").strip()
                    
                    if not user_input:
                        continue
                    
                    # Track the state BEFORE this turn
                    prev_requests = self.state.requests
                    prev_input = self.state.input_tokens
                    prev_output = self.state.output_tokens
                    prev_total = self.state.total_tokens
                    
                    response = await self.process_message(user_input)
                    
                    # Calculate and display turn usage if there was LLM activity
                    turn_requests = self.state.requests - prev_requests
                    turn_input = self.state.input_tokens - prev_input
                    turn_output = self.state.output_tokens - prev_output
                    turn_total = self.state.total_tokens - prev_total
                    
                    if response == "SYSTEM_EXIT":
                        print("\n👋 Goodbye! Thanks for using the AutoGen-powered chat.")
                        if self.state.requests > 0:
                            print(f"\n{get_state_usage_summary(self.state)}")
                        break
                    else:
                        print(f"\n💬 Assistant > {response}")
                        
                        # Display turn usage if there was LLM activity (same as vanilla)
                        if turn_requests > 0:
                            # Calculate turn cost
                            turn_input_cost = (turn_input / 1_000_000) * self.state.input_token_price_per_million
                            turn_output_cost = (turn_output / 1_000_000) * self.state.output_token_price_per_million
                            turn_cost = turn_input_cost + turn_output_cost
                            
                            # Calculate cumulative cost
                            total_cost = calculate_state_cost(self.state)
                            logger.info(f"\n📊 This turn: {turn_input:,} in, {turn_output:,} out, {turn_total:,} total tokens ({turn_requests} requests) - Cost: ${turn_cost:.4f}")
                            logger.info(f"\n📊 Cumulative: {self.state.input_tokens:,} in, {self.state.output_tokens:,} out, {self.state.total_tokens:,} total tokens ({self.state.requests} requests) - Total cost: ${total_cost:.4f}")
                    
                except KeyboardInterrupt:
                    print("\n\n⚠️  Interrupted! Type 'exit' to quit or continue chatting.")
                    # Display final usage stats on interrupt
                    if self.state.requests > 0:
                        print(f"\n{get_state_usage_summary(self.state)}")
                except Exception as e:
                    print(f"\n❌ Error: {e}")
                    logger.error(f"REPL error: {e}", exc_info=True)
                
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """Clean up resources"""
        try:
            # Stop workbench first if it exists
            if self.workbench:
                try:
                    await self.workbench.stop()
                    logger.info("McpWorkbench stopped and cleaned up")
                except Exception as e:
                    logger.error(f"Error stopping workbench: {e}")
            
            if self.runtime and self.runtime_started:
                await self.runtime.stop()
                await self.runtime.close()
                self.runtime_started = False
                logger.info("AutoGen runtime stopped and closed")
            
            # Cleanup global context
            cleanup_global_context(self.state.session_id)
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


async def create_autogen_chat_system(session_id: str = None) -> AutoGenChatSystem:
    """
    Factory function to create and initialize an AutoGen chat system
    
    Args:
        session_id: Optional session ID
        use_persistent_memory: Whether to use persistent storage or in-memory context
        
    Returns:
        Initialized AutoGenChatSystem instance
    """
    system = AutoGenChatSystem(session_id)
    await system._initialize_runtime()
    return system
