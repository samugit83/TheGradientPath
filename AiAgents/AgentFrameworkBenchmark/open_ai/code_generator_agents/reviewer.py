"""
Reviewer Agent using openai-agents library pattern
"""
import logging
from agents import Agent, RunContextWrapper
from context import MainContext, CodeGenContext
from .types import ReviewOutput
from prompts import REVIEWER_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


def get_reviewer_instructions(
    wrapper: RunContextWrapper[MainContext], 
    agent: Agent[MainContext]
) -> str:
    """
    Dynamic instructions for the Reviewer agent based on current context.
    Now works with MainContext and accesses code_gen_context.
    """
    main_context = wrapper.context
    
    # Access the code generation context
    if not main_context.code_gen_context:
        logger.error("No code_gen_context found in MainContext!")
        return "Error: No code generation context available."
    
    context = main_context.code_gen_context
    
    # Get the code to review
    code = ""
    if context.current_code:
        code = context.current_code.code
    
    # Prepare test result parameters
    passed = 0
    failed = 0
    test_explanation = ""
    failure_details = ""
    
    if context.test_results:
        passed = context.test_results.passed
        failed = context.test_results.failed
        test_explanation = context.test_results.test_explanation or ""
        
        if context.test_results.failures:
            failure_details_list = []
            for i, failure in enumerate(context.test_results.failures[:5], 1):
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


def create_reviewer(context: MainContext) -> Agent[MainContext]:
    """
    Create and configure the Reviewer agent.
    
    This agent is responsible for reviewing code and test results,
    providing feedback, and deciding whether to continue iterations.
    Now configured to work with MainContext and uses configurable model.
    """
    agent = Agent[MainContext](
        name="Reviewer Agent",
        instructions=get_reviewer_instructions,  # Dynamic instructions
        model=context.model_name,
        output_type=ReviewOutput,  # Structured output
        tools=[],  # No tools needed
    )
    
    return agent


def post_process_review_output(
    main_context: MainContext,
    review_output: ReviewOutput
) -> None:
    """
    Post-process the review output from the agent.
    Updates the context based on review decisions.
    Now works with MainContext.
    """
    if not main_context.code_gen_context:
        logger.error("No code_gen_context found in MainContext!")
        return
    
    context = main_context.code_gen_context
    
    # Update context with review notes
    context.review_notes = review_output.review_notes
    
    # Check if process should complete
    if not review_output.should_continue:
        context.mark_complete()
        logger.info("Review complete - All tests passed! Process completed successfully.")
    else:
        logger.info(f"Review complete - {len(review_output.review_notes)} issues to address")
        logger.info(f"Recommendation: {review_output.recommendation}")
    
    # Log the analysis
    logger.debug(f"Review analysis: {review_output.analysis}")


def get_fallback_review(context: CodeGenContext) -> ReviewOutput:
    """
    Provide a fallback review when the LLM-based review fails.
    This ensures the process can continue even if the reviewer agent fails.
    Still works directly with CodeGenContext for fallback purposes.
    """
    review_notes = []
    should_continue = True
    recommendation = "revise"
    
    if context.test_results:
        if context.test_results.failed == 0:
            # All tests passed
            review_notes = ["All tests passed successfully. Implementation is complete."]
            should_continue = False
            recommendation = "approve"
            analysis = "The code successfully passes all test cases."
        else:
            # Tests failed
            review_notes.append(f"Fix {context.test_results.failed} failing tests")
            
            # Add specific failure notes
            if context.test_results.failures:
                for failure in context.test_results.failures[:3]:
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