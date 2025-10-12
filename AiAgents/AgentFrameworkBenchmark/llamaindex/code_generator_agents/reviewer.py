"""
Reviewer Agent using LlamaIndex FunctionAgent pattern
"""
import logging
from typing import Any
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from state import MainState, CodeGenState, codegen_mark_complete
from .types import ReviewOutput
from prompts import REVIEWER_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


def get_reviewer_instructions(main_state: MainState) -> str:
    """
    Dynamic instructions for the Reviewer agent based on current state.
    Now works with MainState and accesses code_gen_state.
    """
    # Access the code generation state
    if not main_state.code_gen_state:
        logger.error("No code_gen_state found in MainState!")
        return "Error: No code generation state available."
    
    state = main_state.code_gen_state
    
    # Get the code to review
    code = ""
    if state.current_code:
        code = state.current_code.code
    
    # Prepare test result parameters
    passed = 0
    failed = 0
    test_explanation = ""
    failure_details = ""
    
    if state.test_results:
        passed = state.test_results.passed
        failed = state.test_results.failed
        test_explanation = state.test_results.test_explanation or ""
        
        if state.test_results.failures:
            failure_details_list = []
            for i, failure in enumerate(state.test_results.failures[:5], 1):
                failure_details_list.append(f"{i}. {failure.name}:\n   {failure.trace[:300]}...")
            failure_details = "\n".join(failure_details_list)
    
    # Format the prompt using the template from prompts.py
    formatted_prompt = REVIEWER_PROMPT_TEMPLATE.format(
        code=code,
        passed=passed,
        failed=failed,
        test_explanation=test_explanation,
        failure_details=failure_details
    )
    
    return formatted_prompt


def create_reviewer(main_state: MainState) -> FunctionAgent:
    """
    Create and configure the Reviewer agent.
    
    This agent is responsible for reviewing code and test results,
    providing feedback, and deciding whether to continue iterations.
    Now configured to work with MainState and uses configurable model with token tracking.
    """
    # Use the same callback manager as main agents for consistent token tracking
    if hasattr(main_state, 'token_counter') and main_state.token_counter:
        from llama_index.core.callbacks import CallbackManager
        cb_manager = CallbackManager([main_state.token_counter])
        llm = OpenAI(model=main_state.model_name, callback_manager=cb_manager)
    else:
        llm = OpenAI(model=main_state.model_name)
    
    # Get dynamic instructions based on current state
    instructions = get_reviewer_instructions(main_state)
    
    agent = FunctionAgent(
        name="Reviewer Agent",
        description="An expert code reviewer and quality assurance specialist that analyzes code and test results",
        system_prompt=instructions,
        llm=llm,
        tools=[],
    )
    
    return agent


def post_process_review_output(
    main_state: MainState,
    review_output: ReviewOutput,
    ctx: Any = None,
) -> None:
    """
    Post-process the review output from the agent.
    Updates the state based on review decisions.
    Now works with MainState.
    """
    if not main_state.code_gen_state:
        logger.error("No code_gen_state found in MainState!")
        return
    
    if ctx is not None:
        # Update via ctx state
        async def _update_via_ctx():
            async with ctx.store.edit_state() as ctx_state:
                mc: MainState = ctx_state["state"]["main_state"]
                cgx = mc.code_gen_state
                cgx.review_notes = review_output.review_notes
                if not review_output.should_continue:
                    codegen_mark_complete(cgx)
                    logger.info("Review complete - All tests passed! Process completed successfully.")
                else:
                    logger.info(f"Review complete - {len(review_output.review_notes)} issues to address")
                    logger.info(f"Recommendation: {review_output.recommendation}")
                ctx_state["state"]["main_state"] = mc
        # Run the coroutine if provided a ctx
        import asyncio
        asyncio.get_event_loop().run_until_complete(_update_via_ctx())
    else:
        state = main_state.code_gen_state
        state.review_notes = review_output.review_notes
        if not review_output.should_continue:
            codegen_mark_complete(state)
            logger.info("Review complete - All tests passed! Process completed successfully.")
        else:
            logger.info(f"Review complete - {len(review_output.review_notes)} issues to address")
            logger.info(f"Recommendation: {review_output.recommendation}")
    
    # Log the analysis
    logger.debug(f"Review analysis: {review_output.analysis}")


def get_fallback_review(state: CodeGenState) -> ReviewOutput:
    """
    Provide a fallback review when the LLM-based review fails.
    This ensures the process can continue even if the reviewer agent fails.
    Still works directly with CodeGenState for fallback purposes.
    """
    review_notes = []
    should_continue = True
    recommendation = "revise"
    
    if state.test_results:
        if state.test_results.failed == 0:
            # All tests passed
            review_notes = ["All tests passed successfully. Implementation is complete."]
            should_continue = False
            recommendation = "approve"
            analysis = "The code successfully passes all test cases."
        else:
            # Tests failed
            review_notes.append(f"Fix {state.test_results.failed} failing tests")
            
            # Add specific failure notes
            if state.test_results.failures:
                for failure in state.test_results.failures[:3]:
                    review_notes.append(f"Fix test: {failure.name}")
            
            review_notes.extend([
                "Check input validation and edge cases",
                "Verify implementation logic matches requirements",
                "Ensure proper error handling for invalid inputs"
            ])
            
            analysis = f"Code has {context.test_results.failed} test failures that need to be addressed."
    else:
        review_notes.append("No test results available - ensure tests are running correctly")
        analysis = "Unable to assess code without test results."
    
    return ReviewOutput(
        review_notes=review_notes,
        analysis=analysis,
        recommendation=recommendation,
        should_continue=should_continue
    )