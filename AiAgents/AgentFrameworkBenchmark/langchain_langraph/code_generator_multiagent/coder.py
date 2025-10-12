import json
import logging
from pathlib import Path
from prompts import CODER_PROMPT_TEMPLATE  
from state import CodeResult
from langchain_core.prompts import PromptTemplate
from langchain_community.callbacks import get_openai_callback

logger = logging.getLogger(__name__)


class CoderAgent:
    def __init__(self, package_name: str = "app", llm_client=None) -> None:
        self.package_name = package_name
        self.llm_client = llm_client
        self.base_dir = Path.cwd()
        self.prompt_template = PromptTemplate.from_template(CODER_PROMPT_TEMPLATE)
    
    def compile_main(self, code: str) -> None:
        """Create or update main.py in the app folder with the generated code"""

        app_dir = self.base_dir / self.package_name
        
        if app_dir.exists():
            import shutil
            shutil.rmtree(app_dir)
        
        app_dir.mkdir(parents=True, exist_ok=True)
        
        main_file = app_dir / "main.py"
        with open(main_file, 'w', encoding='utf-8') as f:
            f.write(code)
        
        logger.info(f"Compiled code to: {main_file}")
    
    
    def produce(self, state) -> dict:
        """Generate code using LLM with retry mechanism (max 3 attempts)
        
        Returns:
            dict: Token usage information from the LLM calls
        """
        
        # Prepare prompt variables
        previous_work = ""
        if state.code_gen_state.iteration > 0 and state.code_gen_state.previous_code:
            previous_work = "## Previous Implementation:\n"
            previous_work += "The previous iteration generated code that needs improvement.\n"
            previous_work += "Here is the code that failed:\n\n```python\n"
            previous_work += state.code_gen_state.previous_code
            previous_work += "\n```\n\n"
            previous_work += "Please fix the issues identified in the review feedback below.\n\n"
        
        constraints_section = ""
        if state.code_gen_state.constraints:
            constraints_section = "## Additional Constraints:\n"
            for key, value in state.code_gen_state.constraints.items():
                constraints_section += f"- **{key}**: {value}\n"
            constraints_section += "\n"
        
        review_feedback = ""
        if state.code_gen_state.review_notes:
            review_feedback = "## Review Feedback from Previous Iterations:\n"
            for note in state.code_gen_state.review_notes:
                review_feedback += f"- {note}\n"
            review_feedback += "\n"
        
        if state.code_gen_state.test_results and state.code_gen_state.test_results.failed > 0:
            review_feedback += "## Test Failures to Address:\n"
            for failure in state.code_gen_state.test_results.failures:
                review_feedback += f"- **{failure.name}**: {failure.trace}\n"
            review_feedback += "\nPlease fix these test failures in the new implementation.\n\n"
        
        # Format the prompt using PromptTemplate
        formatted_prompt = self.prompt_template.format(
            user_requirements=state.code_gen_state.user_prompt_for_app,
            iteration=state.code_gen_state.iteration,
            previous_work=previous_work,
            constraints_section=constraints_section,
            review_feedback=review_feedback
        )
        
        code = ""
        explanation = ""
        max_retries = 3
        total_usage = {"requests": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        
        with get_openai_callback() as cb:
            for attempt in range(1, max_retries + 1):
                if attempt == 1:
                    logger.info("Calling LLM for code generation...")
                else:
                    logger.info(f"Retry attempt {attempt}/{max_retries}...")
                
                try:
                    response = self.llm_client.invoke(formatted_prompt)
                    llm_response_str = response.content if hasattr(response, 'content') else str(response)
                    
                    if llm_response_str:
                        try:
                            response_json = json.loads(llm_response_str)
                            code = response_json.get("code", "")
                            explanation = response_json.get("explanation", "Code generated successfully")
                            
                            if code.strip():  
                                logger.info("Code generated successfully")
                                break
                                
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON parsing failed: {e}")
                        
                except Exception as e:
                    logger.error(f"LLM call failed: {e}")
            
            # Extract token usage from callback handler
            total_usage = {
                "requests": cb.successful_requests,
                "input_tokens": cb.prompt_tokens,
                "output_tokens": cb.completion_tokens,
                "total_tokens": cb.total_tokens
            }
            
            # Log token usage for this coder interaction
            if cb.successful_requests > 0:
                logger.info(
                    f"🔢 CoderAgent LLM calls - Input: {cb.prompt_tokens}, "
                    f"Output: {cb.completion_tokens}, Total: {cb.total_tokens} tokens "
                    f"({cb.successful_requests} requests) - Cost: ${cb.total_cost:.4f}"
                )
        
        if not code.strip():
            error_msg = f"Failed to generate code after {max_retries} attempts"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        self.compile_main(code)

        state.code_gen_state.code_result = CodeResult(
            code=code,
            code_explanation=explanation
        )

        return total_usage

