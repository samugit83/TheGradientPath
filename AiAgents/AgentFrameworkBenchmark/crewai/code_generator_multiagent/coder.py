
import logging
from pathlib import Path
from prompts import CODER_PROMPT_TEMPLATE
from state import CodeResult, update_usage_from_crewai, get_global_state
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class CodeGenerationResponse(BaseModel):
    code: str
    explanation: str


class CoderAgent:
    def __init__(self, state, agent) -> None:
        self.state = state
        # Set base_dir to the crewai directory, not project root
        self.base_dir = Path(__file__).parent.parent
        self.agent = agent
    
    def compile_main(self, code: str) -> None:
        """Create or update main.py in the app folder with the generated code"""

        app_dir = self.base_dir / self.state.code_gen_state.package_name
        
        if app_dir.exists():
            import shutil
            shutil.rmtree(app_dir)
        
        app_dir.mkdir(parents=True, exist_ok=True)
        
        main_file = app_dir / "main.py"
        with open(main_file, 'w', encoding='utf-8') as f:
            f.write(code)
        
        logger.info(f"Compiled code to: {main_file}")
    
    def _format_prompt(self) -> str:
        """Format the prompt using the current state information"""
        
        previous_work = ""
        if self.state.code_gen_state.iteration > 0 and self.state.code_gen_state.previous_code:
            previous_work = "## Previous Implementation:\n"
            previous_work += "The previous iteration generated code that needs improvement.\n"
            previous_work += "Here is the code that failed:\n\n```python\n"
            previous_work += self.state.code_gen_state.previous_code
            previous_work += "\n```\n\n"
            previous_work += "Please fix the issues identified in the review feedback below.\n\n"
        
        constraints_section = ""
        if self.state.code_gen_state.constraints:
            constraints_section = "## Additional Constraints:\n"
            for key, value in self.state.code_gen_state.constraints.items():
                constraints_section += f"- **{key}**: {value}\n"
            constraints_section += "\n"
        
        review_feedback = ""
        if self.state.code_gen_state.review_notes:
            review_feedback = "## Review Feedback from Previous Iterations:\n"
            for note in self.state.code_gen_state.review_notes:
                review_feedback += f"- {note}\n"
            review_feedback += "\n"
        
        if self.state.code_gen_state.test_results and self.state.code_gen_state.test_results.failed > 0:
            review_feedback += "## Test Failures to Address:\n"
            for failure in self.state.code_gen_state.test_results.failures:
                review_feedback += f"- **{failure.name}**: {failure.trace}\n"
            review_feedback += "\nPlease fix these test failures in the new implementation.\n\n"
        
        formatted_prompt = CODER_PROMPT_TEMPLATE.format(
            user_requirements=self.state.code_gen_state.user_prompt_for_app,
            iteration=self.state.code_gen_state.iteration,
            previous_work=previous_work,
            constraints_section=constraints_section,
            review_feedback=review_feedback
        )
        
        return formatted_prompt
    
    def produce(self) -> None:
        """Generate code using LLM with retry mechanism (max 3 attempts)"""
        
        prompt = self._format_prompt()
        code = ""
        explanation = ""
        max_retries = 3
        
        for attempt in range(1, max_retries + 1):
            if attempt == 1:
                logger.info("Calling LLM for code generation...")
            else:
                logger.info(f"Retry attempt {attempt}/{max_retries}...")
            
            try:
                result = self.agent.kickoff(prompt, response_format=CodeGenerationResponse)
                
                usage_data = None
                if result and hasattr(result, 'usage_metrics'):
                    usage_data = result.usage_metrics
                
                if usage_data:
                    global_state = get_global_state()
                    if global_state:
                        logger.info(f"Updating reviewer usage with: {usage_data}")
                        update_usage_from_crewai(global_state, usage_data)
                
                if result and hasattr(result, 'pydantic'):
                    response = result.pydantic
                    code = response.code
                    explanation = response.explanation or "Code generated successfully"
                    
                    if code.strip():  
                        logger.info("Code generated successfully")
                        break
                    
            except Exception as e:
                logger.error(f"Agent call failed: {e}")
        
        if not code.strip():
            error_msg = f"Failed to generate code after {max_retries} attempts"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        self.compile_main(code)

        self.state.code_gen_state.code_result = CodeResult(
            code=code,
            code_explanation=explanation
        )

        return 'code_generated'

