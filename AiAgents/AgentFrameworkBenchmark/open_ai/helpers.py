"""
Helper functions for the AI chat application
"""

import logging
from typing import Any
from context import MainContext, extract_usage_from_result

logger = logging.getLogger(__name__)


async def run_chat_loop_with_usage(agent: Any, main_context: MainContext, session: Any = None) -> None:
    """
    Run an interactive chat loop with token usage tracking.
    
    This custom loop:
    - Passes context to tools via RunContextWrapper
    - Tracks token usage for each turn
    - Shows both per-turn and cumulative usage statistics
    - Session for persistent memory is passed to Runner.run()
    - Handles content safety guardrail violations
    
    Args:
        agent: The configured Agent instance
        main_context: The MainContext for tracking usage across the session
        session: Optional SQLiteSession for persistent memory
    """
    from agents import Runner, InputGuardrailTripwireTriggered
    
    print("\nType 'quit' or 'exit' to leave the chat.\n")
    
    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Bye!")
            break
            
        if user_input.lower() in {"quit", "exit"}:
            print("👋 Bye!")
            break
        if not user_input:
            continue
            
        # Track the context state BEFORE this turn
        prev_requests = main_context.requests
        prev_input = main_context.input_tokens
        prev_output = main_context.output_tokens
        prev_total = main_context.total_tokens
        
        try:
            # Run the agent with context - this passes context to tools via wrapper
            # Tools may update main_context during execution
            kwargs = {
                "starting_agent": agent, 
                "input": user_input, 
                "context": main_context
            }
            
            # Add session if provided for persistent memory
            if session is not None:
                kwargs["session"] = session
                
            result = await Runner.run(**kwargs)
            
            if hasattr(result, 'final_output') and result.final_output:
                print(result.final_output)
                
        except InputGuardrailTripwireTriggered as e:
            # Handle content safety violations
            logger.warning(f"Content safety violation detected: {e}")
            print("\n⚠️  Your request was blocked for safety reasons.")
            print("The AI assistant cannot help with:")
            print("  • Illegal activities or harmful content")
            print("  • Violence, self-harm, or threats")
            print("  • Hate speech or discrimination")
            print("  • Inappropriate sexual content")
            print("  • Privacy violations or doxxing")
            print("  • Creating misinformation")
            print("  • Academic dishonesty")
            print("\nPlease rephrase your request or ask something else.")
            continue
        
        usage = extract_usage_from_result(result)
        if usage['requests'] > 0:
            main_context.update_usage(usage)
        
        turn_requests = main_context.requests - prev_requests
        turn_input = main_context.input_tokens - prev_input
        turn_output = main_context.output_tokens - prev_output
        turn_total = main_context.total_tokens - prev_total
        
        if turn_requests > 0:
            # Calculate turn cost
            turn_input_cost = (turn_input / 1_000_000) * main_context.input_token_price_per_million
            turn_output_cost = (turn_output / 1_000_000) * main_context.output_token_price_per_million
            turn_cost = turn_input_cost + turn_output_cost
            
            # Calculate cumulative cost
            total_cost = main_context.calculate_cost()
            logger.info(f"\n📊 This turn: {turn_input:,} in, {turn_output:,} out, {turn_total:,} total tokens ({turn_requests} requests) - Cost: ${turn_cost:.4f}")
            logger.info(f"\n📊 Cumulative: {main_context.input_tokens:,} in, {main_context.output_tokens:,} out, {main_context.total_tokens:,} total tokens ({main_context.requests} requests) - Total cost: ${total_cost:.4f}")
    
    # Display final usage stats
    print(f"\n{main_context.get_usage_summary()}")
