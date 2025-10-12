"""
Coder Agent using openai-agents library pattern
"""
import logging
from pathlib import Path
from agents import Agent, RunContextWrapper
from context import MainContext
from .types import CodeOutput
from prompts import CODER_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


def get_coder_instructions(
    wrapper: RunContextWrapper[MainContext], 
    agent: Agent[MainContext]
) -> str:
    """
    Dynamic instructions for the Coder agent based on current context.
    Uses the CODER_PROMPT_TEMPLATE from prompts.py.
    Now works with MainContext and accesses code_gen_context.
    """
    main_context = wrapper.context
    
    # Access the code generation context
    if not main_context.code_gen_context:
        logger.error("No code_gen_context found in MainContext!")
        return "Error: No code generation context available."
    
    context = main_context.code_gen_context
    
    previous_work = ""
    if context.iteration > 0:
        if context.previous_code:
            previous_work = "## Previous Implementation:\n"
            previous_work += "The previous iteration generated code that needs improvement.\n"
            previous_work += "Here is the code that failed:\n\n```python\n"
            previous_work += context.previous_code
            previous_work += "\n```\n\n"
            previous_work += "Please fix the issues identified in the review feedback below.\n\n"
    
    constraints_section = ""
    if context.constraints:
        constraints_section = "## Additional Constraints:\n"
        for key, value in context.constraints.items():
            constraints_section += f"- **{key}**: {value}\n"
        constraints_section += "\n"
    
    review_feedback = ""
    if context.review_notes:
        review_feedback = "## Review Feedback:\n"
        review_feedback += "The following issues were identified in the previous iteration:\n\n"
        for note in context.review_notes:
            review_feedback += f"- {note}\n"
        review_feedback += "\n"
        
        # Add test failure details if available
        if context.test_results and context.test_results.failed > 0:
            review_feedback += "## Test Failures:\n"
            review_feedback += f"Tests Failed: {context.test_results.failed}\n\n"
            
            for i, failure in enumerate(context.test_results.failures[:5], 1):
                review_feedback += f"{i}. **{failure.name}**:\n"
                review_feedback += f"   {failure.trace[:300]}...\n\n"
    
    # Format the prompt using the template
    formatted_prompt = CODER_PROMPT_TEMPLATE.format(
        user_requirements=context.user_prompt_for_app,
        iteration=context.iteration,
        previous_work=previous_work,
        constraints_section=constraints_section,
        review_feedback=review_feedback
    )

    logger.debug(f"Coder instructions length: {len(formatted_prompt)}")
    
    return formatted_prompt


def compile_code(code: str, package_name: str) -> None:
    """Save the generated code to the main.py file"""
    base_dir = Path.cwd()
    app_dir = base_dir / package_name
    
    # Ensure app directory exists
    app_dir.mkdir(parents=True, exist_ok=True)
    
    # Write the code to main.py
    main_file = app_dir / "main.py"
    with open(main_file, 'w', encoding='utf-8') as f:
        f.write(code)
    
    logger.info(f"Code compiled to: {main_file}")


def create_coder(context: MainContext) -> Agent[MainContext]:
    """
    Create and configure the Coder agent.
    
    This agent is responsible for generating Python code based on requirements
    and iteratively improving it based on test results and review feedback.
    Now configured to work with MainContext and uses configurable model.
    """
    agent = Agent[MainContext](
        name="Coder Agent",
        instructions=get_coder_instructions,  # Dynamic instructions based on context
        model=context.model_name, 
        output_type=CodeOutput,  # Structured output
        tools=[],
    )
    
    return agent
