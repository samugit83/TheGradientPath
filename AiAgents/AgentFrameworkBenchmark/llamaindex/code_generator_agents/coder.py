"""
Coder Agent using LlamaIndex FunctionAgent pattern
"""
import logging
from pathlib import Path
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from state import MainState, codegen_update_code
from .types import CodeOutput
from prompts import CODER_PROMPT_TEMPLATE
from llama_index.core.workflow import Context

logger = logging.getLogger(__name__)


def get_coder_instructions(main_state: MainState) -> str:
    """
    Dynamic instructions for the Coder agent based on current state.
    Uses the CODER_PROMPT_TEMPLATE from prompts.py.
    Now works with MainState and accesses code_gen_state.
    """
    # Access the code generation state
    if not main_state.code_gen_state:
        logger.error("No code_gen_state found in MainState!")
        return "Error: No code generation state available."
    
    state = main_state.code_gen_state
    
    previous_work = ""
    if state.iteration > 0:
        if state.previous_code:
            previous_work = "## Previous Implementation:\n"
            previous_work += "The previous iteration generated code that needs improvement.\n"
            previous_work += "Here is the code that failed:\n\n```python\n"
            previous_work += state.previous_code
            previous_work += "\n```\n\n"
            previous_work += "Please fix the issues identified in the review feedback below.\n\n"
    
    constraints_section = ""
    if state.constraints:
        constraints_section = "## Additional Constraints:\n"
        for key, value in state.constraints.items():
            constraints_section += f"- **{key}**: {value}\n"
        constraints_section += "\n"
    
    review_feedback = ""
    if state.review_notes:
        review_feedback = "## Review Feedback:\n"
        review_feedback += "The following issues were identified in the previous iteration:\n\n"
        for note in state.review_notes:
            review_feedback += f"- {note}\n"
        review_feedback += "\n"
        
        # Add test failure details if available
        if state.test_results and state.test_results.failed > 0:
            review_feedback += "## Test Failures:\n"
            review_feedback += f"Tests Failed: {state.test_results.failed}\n\n"
            
            for i, failure in enumerate(state.test_results.failures[:5], 1):
                review_feedback += f"{i}. **{failure.name}**:\n"
                review_feedback += f"   {failure.trace[:300]}...\n\n"
    
    # Format the prompt using the template
    formatted_prompt = CODER_PROMPT_TEMPLATE.format(
        user_requirements=state.user_prompt_for_app,
        iteration=state.iteration,
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


def create_coder(main_state: MainState) -> FunctionAgent:
    """
    Create and configure the Coder agent.
    
    This agent is responsible for generating Python code based on requirements
    and iteratively improving it based on test results and review feedback.
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
    instructions = get_coder_instructions(main_state)
    
    agent = FunctionAgent(
        name="Coder Agent",
        description="An expert Python developer that generates complete Python applications based on requirements",
        system_prompt=instructions,
        llm=llm,
        tools=[],
    )
    
    return agent

