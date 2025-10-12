"""
Simplified tools module with AutoGen-compatible async functions
"""

import logging
from autogen_core import (TopicId)
from autogen_core.models import AssistantMessage
from state import State
from session_manager import get_global_runtime, get_global_session_manager
logger = logging.getLogger(__name__)


async def sum_numbers(a: float, b: float, state = None) -> float:
    """Add two numbers together and return the result"""
    try:
        result = a + b
        logger.info(f"Sum calculated: {a} + {b} = {result}")
        return result
    except Exception as e:
        logger.error(f"Sum calculation error: {e}")
        raise


async def multiply_numbers(a: float, b: float, state = None) -> float:
    """Multiply two numbers and return the product"""
    try:
        result = a * b
        logger.info(f"Multiplication calculated: {a} * {b} = {result}")
        return result
    except Exception as e:
        logger.error(f"Multiplication calculation error: {e}")
        raise



async def generate_code(state = None) -> str:
    """Generate complete code implementations with tests using multi-agent system"""
    try:
        if state is None:
            raise ValueError("State is required for code generation")
        
        # Get runtime from global context
        runtime = get_global_runtime(state.session_id) if state.session_id else None
        if runtime is None:
            raise ValueError("Runtime is required for code generation")

        # Initialize code_gen_state if not present
        if not hasattr(state, 'code_gen_state') or state.code_gen_state is None:
            from state import CodeGenState
            state.code_gen_state = CodeGenState()

        # Publish message to CodingRoom to trigger the multi-agent workflow
        await runtime.publish_message(
            state,  # Pass the state object directly
            topic_id=TopicId("CodingRoom", source="default"),
        )

        logger.info("Code generation started, waiting for completion...")
        
        # Wait for the multi-agent workflow to complete
        import asyncio
        max_wait_time = 300  # 5 minutes timeout
        poll_interval = 1.0  # Check every second
        elapsed_time = 0
        
        while elapsed_time < max_wait_time:
            if state.code_gen_state and state.code_gen_state.process_completed:
                logger.info("Code generation workflow completed!")
                break
                
            await asyncio.sleep(poll_interval)
            elapsed_time += poll_interval
            
            # Log progress every 10 seconds
            if elapsed_time % 10 == 0:
                iteration = state.code_gen_state.iteration if state.code_gen_state else 0
                logger.info(f"Still processing... (iteration {iteration}, elapsed: {elapsed_time}s)")
        
        # Check final status
        if state.code_gen_state and state.code_gen_state.process_completed:
            # Add session message if we have session manager
            session_manager = get_global_session_manager(state.session_id) if state.session_id else None
            if session_manager:
                assistant_message = AssistantMessage(
                    content=f"Code generation completed successfully in {state.code_gen_state.iteration} iteration(s)",
                    source="CodeOrchestratorAgent"
                )
                session_manager.add_session_message(assistant_message)
            return f"✅ Code generated successfully in {state.code_gen_state.iteration} iteration(s). Check the 'app' folder."
        else:
            # Timeout or failure
            iterations = state.code_gen_state.iteration if state.code_gen_state else 0
            if elapsed_time >= max_wait_time:
                return f"⏰ Code generation timed out after {max_wait_time}s (completed {iterations} iteration(s))"
            else:
                return f"⚠️ Generation incomplete after {iterations} iteration(s)"
            
    except Exception as e:
        logger.error(f"Code generation error: {e}")
        raise




def get_all_tool_functions():
    """Get all AutoGen-compatible async tool functions"""
    return {
        "sum_numbers": sum_numbers,
        "multiply_numbers": multiply_numbers,
        "generate_code": generate_code
    }