import json
import logging
from prompts import CODER_PROMPT_TEMPLATE
from state import CodeResult, track_usage_from_result, State
from session_manager import get_global_session_manager
from autogen_core import (
    MessageContext,
    RoutedAgent,
    TopicId,
    message_handler,
    type_subscription,
)
from autogen_core.models import AssistantMessage
from pathlib import Path
import os
from openai import OpenAI

logger = logging.getLogger(__name__)



@type_subscription(topic_type="CodingRoom")
class CoderAgent(RoutedAgent):
    def __init__(self, model_client: OpenAI) -> None:
        super().__init__("A coding agent.")
        self._model_client = model_client
        self.base_dir = Path.cwd()
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY")
        )
        
    @message_handler
    async def produce(self, message: State, ctx: MessageContext) -> None:
        """Generate code using LLM with retry mechanism (max 3 attempts)"""

        message.code_gen_state.iteration += 1
        logger.info(f"Starting coding iteration {message.code_gen_state.iteration}")

        code = ""
        explanation = ""
        max_retries = 3
        
        for attempt in range(1, max_retries + 1):
            if attempt == 1:
                logger.info("Calling LLM for code generation...")
            else:
                logger.info(f"Retry attempt {attempt}/{max_retries}...")
            
            try:    

                prompt = self._format_prompt(message)
                logger.info(f"Prompt code generation: {prompt}")

                params = {
                    "model": message.model_name,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ]
                }
                
                response = self.client.chat.completions.create(**params)
                content = response.choices[0].message.content
                
                if message:
                    track_usage_from_result(response, message)
                
                if content and content.strip():
                    content = content.strip()
                    logger.info(f"Received code response: {content[:100]}..." if len(content) > 100 else f"Received code response: {content}")
                    
                    json_content = self._extract_json_from_markdown(content)
                    
                    try:
                        response_json = json.loads(json_content)
                        code = response_json.get("code", "")
                        explanation = response_json.get("explanation", "Code generated successfully")
                        
                        if code.strip():  
                            logger.info("Successfully parsed JSON code response")
                            break
                            
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON parsing failed: {e}")
                        logger.info(f"Raw content causing JSON error: '{json_content[:200]}'")
                        logger.info("Treating response as raw code since JSON parsing failed")
                        code = content
                        explanation = "Code generated (raw format)"
                        
                        if code.strip():
                            logger.info("Raw code accepted")
                            break
                else:
                    logger.warning(f"Empty or None content from create_result: {content}")
                    continue  # Try again
                    
            except Exception as e:
                logger.error(f"LLM call failed: {e}")
        
        if not code.strip():
            error_msg = f"Failed to generate code after {max_retries} attempts"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        self.compile_main(code, message.code_gen_state.package_name)

        message.code_gen_state.code_result = CodeResult(
            code=code,
            code_explanation=explanation
        )

        session_manager = get_global_session_manager(message.session_id) if message.session_id else None
        if session_manager and message.code_gen_state and message.code_gen_state.code_result:
            assistant_message = AssistantMessage(
                content=f"CoderAgent (Iteration {message.code_gen_state.iteration}): Generated code - {message.code_gen_state.code_result.code_explanation}",
                source="CoderAgent"
            )
            session_manager.add_session_message(assistant_message)


        await self.publish_message(message, topic_id=TopicId("TestingRoom", source=self.id.key))



    def _format_prompt(self, message: State) -> str:
        """Format the prompt using the current state information"""
        
        previous_work = ""
        if message.code_gen_state.iteration > 0 and message.code_gen_state.previous_code:
            previous_work = "## Previous Implementation:\n"
            previous_work += "The previous iteration generated code that needs improvement.\n"
            previous_work += "Here is the code that failed:\n\n```python\n"
            previous_work += message.code_gen_state.previous_code
            previous_work += "\n```\n\n"
            previous_work += "Please fix the issues identified in the review feedback below.\n\n"
        
        constraints_section = ""
        if message.code_gen_state.constraints:
            constraints_section = "## Additional Constraints:\n"
            for key, value in message.code_gen_state.constraints.items():
                constraints_section += f"- **{key}**: {value}\n"
            constraints_section += "\n"
        
        review_feedback = ""
        if message.code_gen_state.review_notes:
            review_feedback = "## Review Feedback from Previous Iterations:\n"
            for note in message.code_gen_state.review_notes:
                review_feedback += f"- {note}\n"
            review_feedback += "\n"
        
        if message.code_gen_state.test_results and message.code_gen_state.test_results.failed > 0:
            review_feedback += "## Test Failures to Address:\n"
            for failure in message.code_gen_state.test_results.failures:
                review_feedback += f"- **{failure.name}**: {failure.trace}\n"
            review_feedback += "\nPlease fix these test failures in the new implementation.\n\n"
        
        formatted_prompt = CODER_PROMPT_TEMPLATE.format(
            user_requirements=message.user_prompt,
            iteration=message.code_gen_state.iteration,
            previous_work=previous_work,
            constraints_section=constraints_section,
            review_feedback=review_feedback
        )
        
        return formatted_prompt

    def compile_main(self, code: str, package_name: str) -> None:
        """Create or update main.py in the app folder with the generated code"""

        app_dir = self.base_dir / "app"
        
        if app_dir.exists():
            import shutil
            import os
            import stat
            
            # More robust directory cleanup that handles .pyc files gracefully
            def safe_remove_tree(directory):
                """Safely remove directory tree, handling permission issues"""
                try:
                    # First, try to clean __pycache__ directories specifically
                    for root, dirs, files in os.walk(directory, topdown=False):
                        # Remove files first
                        for file in files:
                            file_path = os.path.join(root, file)
                            try:
                                if file.endswith('.pyc'):
                                    # For .pyc files, try to make writable first
                                    try:
                                        os.chmod(file_path, stat.S_IWRITE | stat.S_IREAD)
                                    except (OSError, PermissionError):
                                        pass  # If chmod fails, try removal anyway
                                os.unlink(file_path)
                            except (OSError, PermissionError):
                                logger.warning(f"Could not remove file: {file_path}")
                                continue
                        
                        # Then remove directories
                        for dir_name in dirs:
                            dir_path = os.path.join(root, dir_name)
                            try:
                                os.rmdir(dir_path)
                            except (OSError, PermissionError):
                                logger.warning(f"Could not remove directory: {dir_path}")
                                continue
                    
                    # Finally try to remove the main directory
                    try:
                        os.rmdir(directory)
                    except (OSError, PermissionError):
                        logger.warning(f"Could not remove main directory: {directory}")
                        
                except Exception as e:
                    logger.warning(f"Error during safe directory removal: {e}")
                    # As last resort, try regular rmtree with ignore_errors
                    try:
                        shutil.rmtree(directory, ignore_errors=True)
                    except Exception:
                        logger.warning(f"Could not fully clean directory: {directory}")
            
            safe_remove_tree(str(app_dir))
        
        app_dir.mkdir(parents=True, exist_ok=True)
        
        main_file = app_dir / "main.py"
        with open(main_file, 'w', encoding='utf-8') as f:
            f.write(code)
        
        logger.info(f"Compiled code to: {main_file}")
    
    def _extract_json_from_markdown(self, content: str) -> str:
        """Extract JSON from markdown code blocks if present"""
        if content.startswith("```json") and content.endswith("```"):
            lines = content.split('\n')
            if len(lines) >= 3:
                json_lines = lines[1:-1]
                return '\n'.join(json_lines)
        
        return content



