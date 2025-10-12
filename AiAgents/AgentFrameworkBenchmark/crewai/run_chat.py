"""
Interactive Chat Loop for CrewAI Agent Routing System

This module provides the chat runtime that manages user interactions
with the CrewAI crew, handling input/output and maintaining conversation flow.
"""
import logging
import uuid
from state import initialize_global_state, update_user_prompt_for_app, clear_global_state, get_global_state, update_usage_from_crewai, calculate_cost, get_usage_summary

logger = logging.getLogger(__name__)


class RuntimeChat:
    """
    A chat runtime that manages the CrewAI agent routing system.
    
    This class handles the interactive chat loop, user input processing,
    and coordination with the CrewAI crew for intelligent response generation.
    """
    
    def __init__(self, crew, session_id: str = None):
        """
        Initialize the chat runtime.
        
        Args:
            crew: The configured CrewAI crew for handling chat interactions
            session_id: Optional session ID, will generate one if not provided
        """
        self.crew = crew
        self.session_id = session_id or f"session_{uuid.uuid4().hex[:8]}"
        # Initialize global state

        self._display_welcome_message()

    def run(self):
        """Run the interactive chat loop.
        
        Manages the main conversation flow, handling user input,
        processing queries through the CrewAI crew, and displaying responses.
        Includes special commands for session management and graceful exit.
        """
        try:
            # Main chat loop
            while True:
                try:
                    print("\n" + "-" * 40)
                    user_input = input("💬 You > ").strip()

                    if not user_input:
                        continue
                    
                    if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
                        print("\n👋 Goodbye! Thanks for using the AI-powered chat.")
                        break
                    
                    if user_input.lower() in ['clear', 'clear history']:
                        print("✨ Conversation history cleared!")
                        clear_global_state(session_id=self.session_id)
                        continue
                    
                    update_user_prompt_for_app(user_input)
                    
                    state = get_global_state()
                    prev_requests = state.requests if state else 0
                    prev_input = state.input_tokens if state else 0
                    prev_output = state.output_tokens if state else 0
                    prev_total = state.total_tokens if state else 0
                    
                    result = self.crew.kickoff(inputs={"query": user_input})
                    print(f"\n🤖 {result}")
                    
                    if state and hasattr(result, 'token_usage') and result.token_usage:
                        update_usage_from_crewai(state, result.token_usage)
                        
                        turn_requests = state.requests - prev_requests
                        turn_input = state.input_tokens - prev_input
                        turn_output = state.output_tokens - prev_output
                        turn_total = state.total_tokens - prev_total
                        
                        if turn_requests > 0:
                            turn_input_cost = (turn_input / 1_000_000) * state.input_token_price_per_million
                            turn_output_cost = (turn_output / 1_000_000) * state.output_token_price_per_million
                            turn_cost = turn_input_cost + turn_output_cost
                            
                            total_cost = calculate_cost(state)
                            logger.info(f"📊 This turn: {turn_input:,} in, {turn_output:,} out, {turn_total:,} total tokens ({turn_requests} requests) - Cost: ${turn_cost:.4f}")
                            logger.info(f"📊 Cumulative: {state.input_tokens:,} in, {state.output_tokens:,} out, {state.total_tokens:,} total tokens ({state.requests} requests) - Total cost: ${total_cost:.4f}")

                except KeyboardInterrupt:
                    print("\n\n⚠️  Interrupted! Type 'exit' to quit or continue chatting.")
                except Exception as e:
                    print(f"\n❌ Error: {e}")
                    logger.error(f"REPL error: {e}", exc_info=True)
        
        except Exception as e:
            logger.error(f"Chat loop error: {e}", exc_info=True)
            print(f"\n❌ Fatal error: {e}")
        
        # Display final usage summary
        state = get_global_state()
        if state and state.requests > 0:
            print(f"\n{get_usage_summary(state)}")
    
    
    def _display_welcome_message(self):
        """Display the welcome message and usage instructions.
        
        Shows the user how to interact with the system and explains
        the available agents and their capabilities.
        """
        print("\n" + "=" * 60)
        print("  🤖 Welcome to AI-Powered Chat with CrewAI Agent Routing! 🚀")
        print("=" * 60)
        
        print("\n💬 Chat Instructions:")
        print("  • Just type naturally and I'll understand what you need!")
        print("  • Type 'quit' or 'exit' to leave the chat")
        print("  • I'll intelligently route your request to the right specialist:")
        print("    - ⚖️ Legal Expert: For law-related questions and legal topics")
        print("    - 🔧 Operational Agent: For everything else (programming, tools, general knowledge)")
        print("\n🎯 How it works:")
        print("  • The Conversation Manager analyzes your query")
        print("  • Routes it to the most appropriate specialist agent")
        print("  • Provides expert responses tailored to your needs\n")

def run_chat_loop(crew, session_id: str = None):
    """
    Convenience function to run the chat loop.
    
    Args:
        crew: The configured CrewAI crew for handling chat interactions
        session_id: Optional session ID to use for conversation tracking
    """
    chat_runtime = RuntimeChat(crew, session_id)
    chat_runtime.run()
