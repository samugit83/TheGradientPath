"""
Helper functions for the AI chat application
"""

import logging
from typing import Any
from state import extract_usage_from_result, update_main_usage, calculate_main_cost, get_main_usage_summary
from session_manager import setup_session_memory, save_persistent_memory


logger = logging.getLogger(__name__)


async def run_chat_loop(agent_workflow: Any, main_state: Any) -> None:
    """
    Run an interactive chat loop with comprehensive token usage tracking.
    
    This custom loop provides:
    - Interactive user input handling with graceful exit options
    - Detailed token usage tracking for each conversation turn
    - Real-time cost calculation and display (both per-turn and cumulative)
    - Comprehensive logging of usage statistics
    - Graceful handling of keyboard interrupts and EOF
    - Optional persistent memory for conversation history across sessions
    
    The function tracks:
    - Input/output/total tokens per turn
    - Number of API requests per turn
    - Cost calculations based on token prices
    - Cumulative session statistics
    
    Args:
        agent_workflow: The configured AgentWorkflow instance to process user input
        main_state: The MainState instance for tracking usage across the session
    """
    
    print("\nType 'quit' or 'exit' to leave the chat.\n")
    
    session_id = main_state.session_id or "default_session"
    chat_memory, ctx = setup_session_memory(agent_workflow, session_id, main_state)
    
    while True:
        try:
            user_input = input("> You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Bye!")
            break
            
        if user_input.lower() in {"quit", "exit"}:
            print("👋 Bye!")
            break
        if not user_input:
            continue
            
        prev_requests = main_state.requests
        prev_input = main_state.input_tokens
        prev_output = main_state.output_tokens
        prev_total = main_state.total_tokens

        if main_state.use_persistent_memory and chat_memory:
            result = await agent_workflow.run(user_msg=user_input, ctx=ctx, memory=chat_memory)
        else:
            result = await agent_workflow.run(user_msg=user_input, ctx=ctx)
        
        if hasattr(result, 'response') and result.response:
            if hasattr(result, 'current_agent_name') and result.current_agent_name == 'GuardrailAgent':
                import json
                try:
                    response_text = str(result.response)
                    start = response_text.find('{')
                    end = response_text.rfind('}') + 1
                    json_text = response_text[start:end]
                    parsed = json.loads(json_text)
                    
                    print(f"⚠️  Content Safety Alert:")
                    print(f"   Reason: {parsed.get('reason', 'Unknown')}")
                    print(f"   Category: {parsed.get('category', 'Unknown')}")
                except:
                    print(f"🤖: {result.response}")
                    logger.info("GuardrailAgent response - could not parse structured data")
            else:
                print(f"🤖: {result.response}")
        
        usage = extract_usage_from_result(result, token_counter=main_state.token_counter)
        
        if hasattr(result, 'current_agent_name'):
            logger.info(f"Current agent: {result.current_agent_name}")


        if usage['requests'] > 0:
            update_main_usage(main_state, usage)
        else:
            logger.info("Received response but couldn't extract usage information")
        
        turn_requests = main_state.requests - prev_requests
        turn_input = main_state.input_tokens - prev_input
        turn_output = main_state.output_tokens - prev_output
        turn_total = main_state.total_tokens - prev_total
        
        if turn_requests > 0:
            turn_input_cost = (turn_input / 1_000_000) * main_state.input_token_price_per_million
            turn_output_cost = (turn_output / 1_000_000) * main_state.output_token_price_per_million
            turn_cost = turn_input_cost + turn_output_cost
            
            total_cost = calculate_main_cost(main_state)
            logger.info(f"\n📊 This turn: {turn_input:,} in, {turn_output:,} out, {turn_total:,} total tokens ({turn_requests} requests) - Cost: ${turn_cost:.4f}")
            logger.info(f"\n📊 Cumulative: {main_state.input_tokens:,} in, {main_state.output_tokens:,} out, {main_state.total_tokens:,} total tokens ({main_state.requests} requests) - Total cost: ${total_cost:.4f}")
        
        # Save persistent memory after each turn if enabled
        if main_state.use_persistent_memory and chat_memory:
            try:
                save_persistent_memory(session_id, chat_memory, ctx)
                logger.debug("Persistent memory saved successfully")
            except Exception as e:
                logger.error(f"Failed to save persistent memory: {e}")
    
    print(f"\n{get_main_usage_summary(main_state)}")
    
    # Final save of persistent memory before exit
    if main_state.use_persistent_memory and chat_memory:
        try:
            save_persistent_memory(session_id, chat_memory, ctx)
            logger.info("Final persistent memory save completed")
        except Exception as e:
            logger.error(f"Failed to perform final memory save: {e}")
