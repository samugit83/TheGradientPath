
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage
from langchain_community.callbacks import get_openai_callback
from state import MainState, update_usage, calculate_cost, get_usage_summary
from agents import RoutingAgent, LegalExpertAgent, GeneralAgent, ContentSafetyAgent, SafetyTerminationAgent
from checkpointer import Checkpointer
import logging
import os
import sys

logger = logging.getLogger(__name__)


class OrchestratorAgent(): 
    def __init__(self, session_id: str = "default_session"):

        self.session_id = session_id
        self.state = MainState()
        self.main_graph = StateGraph(MainState)
        self.llm = init_chat_model(self.state.model_name, model_provider="openai")
        self.logger = logger
        
        # Initialize persistent checkpointer conditionally
        if self.state.use_persistent_memory:
            self.checkpointer = Checkpointer.initialize()
            self.logger.info("Persistent memory enabled - checkpointer initialized")
        else:
            self.checkpointer = None
            self.logger.info("Persistent memory disabled - using session-only memory")
        
        self.content_safety_agent = ContentSafetyAgent(self)
        self.routing_agent = RoutingAgent(self)
        self.legal_expert_agent = LegalExpertAgent(self)
        self.general_agent = GeneralAgent(self)
        self.safety_termination_agent = SafetyTerminationAgent(self)
        memory_status = "with persistent memory" if self.state.use_persistent_memory else "with session-only memory"
        self.logger.info(f"OrchestratorAgent initialized {memory_status}.")

        # All agents will be reinitialized at each node function call.
        def _content_safety_agent_node(state: MainState, config: RunnableConfig):
            return self.content_safety_agent.run(state, config)

        def _routing_agent_node(state: MainState, config: RunnableConfig):
            return self.routing_agent.run(state, config)

        def _legal_expert_agent_node(state: MainState, config: RunnableConfig):
            return self.legal_expert_agent.run(state, config)

        def _general_agent_node(state: MainState, config: RunnableConfig):
            return self.general_agent.run(state, config)

        def _safety_termination_agent_node(state: MainState, config: RunnableConfig):
            return self.safety_termination_agent.run(state, config)

        self.main_graph.add_node("content_safety", _content_safety_agent_node)
        self.main_graph.add_node("routing_agent", _routing_agent_node)
        self.main_graph.add_node("legal_expert", _legal_expert_agent_node) 
        self.main_graph.add_node("general_agent", _general_agent_node)
        self.main_graph.add_node("safety_termination", _safety_termination_agent_node)

        def safety_check_router(state: MainState):
            """Route based on content safety analysis"""
            content_safety_result = state.get("content_safety_result") if isinstance(state, dict) else state.content_safety_result
            if content_safety_result and not content_safety_result.is_appropriate:
                return "safety_termination"
            else:
                return "routing_agent"

        def route_agent(state: MainState):
            """Route to appropriate agent based on routing decision"""
            route_name = state.get("routeName") if isinstance(state, dict) else state.routeName
            if route_name == "legal_expert":
                return "legal_expert"
            else:
                return "general_agent"

        # Define graph edges with content safety first
        self.main_graph.add_edge(START, "content_safety")
        
        self.main_graph.add_conditional_edges(
            "content_safety",
            safety_check_router,
            {
                "safety_termination": "safety_termination",
                "routing_agent": "routing_agent"
            }
        )

        self.main_graph.add_conditional_edges(
            "routing_agent",
            route_agent,
            {
                "legal_expert": "legal_expert",
                "general_agent": "general_agent"
            }
        )

        # All paths lead to END
        self.main_graph.add_edge("legal_expert", END)
        self.main_graph.add_edge("general_agent", END)
        self.main_graph.add_edge("safety_termination", END)

        # Compile graph with or without checkpointer based on configuration
        if self.checkpointer is not None:
            self.main_graph = self.main_graph.compile(checkpointer=self.checkpointer)
        else:
            self.main_graph = self.main_graph.compile()


    def main_graph_invoke(self, user_input: str, session_id: str = None):
        """
        Sends the user input into the LangGraph state machine and streams back the final answer.
        Checkpointer automatically handles conversation persistence via MainState.messages.
        Tracks token usage and cost for each interaction.
        """
        # Use provided session_id or default to instance session_id
        thread_id = session_id or self.session_id
        

        current_user_message = HumanMessage(content=user_input)
        

        input_data = {
            "messages": [current_user_message]
        }

        config = {"configurable": {"thread_id": thread_id}}

        # Initialize callback handler for token tracking
        with get_openai_callback() as cb:
            responses = self.main_graph.invoke(input_data, config)
            
            # Extract token usage from callback handler
            usage_dict = {
                "requests": cb.successful_requests,
                "input_tokens": cb.prompt_tokens,
                "output_tokens": cb.completion_tokens,
                "total_tokens": cb.total_tokens
            }
            
            # Update state with token usage
            update_usage(self.state, usage_dict)
            
            # Log token usage for this interaction
            if cb.successful_requests > 0:
                logger.info(
                    f"🔢 LangGraph interaction - Input: {cb.prompt_tokens}, "
                    f"Output: {cb.completion_tokens}, Total: {cb.total_tokens} tokens "
                    f"({cb.successful_requests} requests) - Cost: ${cb.total_cost:.4f}"
                )
        
        # Extract the AI response content for return
        response_content = ""
        if responses.get("messages"):
            # Get the last AI message from the response
            for message in reversed(responses["messages"]):
                if hasattr(message, 'content') and message.content and message.__class__.__name__ == "AIMessage":
                    response_content = message.content
                    break
        
        # Return both response and usage for the caller to use
        return response_content, usage_dict


    def run(self):
        """
        Main chat loop for the orchestrator agent.
        """
        if not os.environ.get("OPENAI_API_KEY"):
            print("⚠️  Warning: OPENAI_API_KEY not set!")
            print("Set it with: export OPENAI_API_KEY='your-key-here'")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                sys.exit(1)
        
        print("\n🤖 LangGraph Multi-Agent System Started!")
        print("Memory commands: 'history', 'clear memory', 'memory summary'")
        print("Type 'exit', 'quit', or press Ctrl+C to quit\n")
        
        while True:
            try:
                print("\n" + "-" * 40)
                user_input = input("💬 You > ").strip()
                
                if not user_input:
                    continue
                
                # Check for exit commands
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print("\n👋 Goodbye! Thanks for using the AI-powered chat.")
                    break

                # Track the state BEFORE this turn for usage calculation
                prev_requests = self.state.requests
                prev_input = self.state.input_tokens
                prev_output = self.state.output_tokens
                prev_total = self.state.total_tokens
                
                # Process the user input through the graph
                try:
                    response, usage_dict = self.main_graph_invoke(user_input, self.session_id)
                    print(f"\n🤖 Assistant > {response}")
                    
                    # Calculate and display turn usage if there was LLM activity
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
                    
                except Exception as graph_error:
                    print(f"\n❌ Error processing request: {graph_error}")
                    self.logger.error(f"Graph processing error: {graph_error}", exc_info=True)
                    
            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupted! Type 'exit' to quit or continue chatting.")
                # Display final usage stats on interrupt
                if self.state.requests > 0:
                    print(f"\n{get_usage_summary(self.state)}")
            except Exception as e:
                print(f"\n❌ Error: {e}")
                self.logger.error(f"REPL error: {e}", exc_info=True)
        
        # Display final usage stats on exit
        if self.state.requests > 0:
            print(f"\n{get_usage_summary(self.state)}")

