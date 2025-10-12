import json
import logging
from prompts import REVIEWER_PROMPT_TEMPLATE
from state import track_usage_from_result, State, ReviewResult
from session_manager import get_global_session_manager
from pathlib import Path
import json
import logging
from autogen_core import (
    MessageContext,
    RoutedAgent,
    TopicId,
    message_handler,
    type_subscription,
)
from autogen_core.models import AssistantMessage
from openai import OpenAI

logger = logging.getLogger(__name__)


@type_subscription(topic_type="ReviewingRoom")
class ReviewerAgent(RoutedAgent):
    def __init__(self, model_client: OpenAI) -> None:
        super().__init__("A testing agent.")
        self._model_client = model_client
        self.base_dir = Path.cwd()

    @message_handler
    async def assess(self, message: State, ctx: MessageContext) -> None:
        """Review the code and test results to provide feedback"""
        
        if message.code_gen_state.test_results and message.code_gen_state.test_results.failed == 0:
            review_result = ReviewResult(
                review_notes=["All tests passed successfully. Implementation is complete."],
                analysis="The code successfully passes all test cases.",
                recommendation="approve",
                should_continue=False
            )
            
            message.code_gen_state.review_notes = review_result.review_notes
            message.code_gen_state.process_completed = True
            logger.info("All tests passed - process completed successfully!")
            
            session_manager = get_global_session_manager(message.session_id) if message.session_id else None
            if session_manager:
                review_content = f"ReviewerAgent (Iteration {message.code_gen_state.iteration}): {', '.join(review_result.review_notes) if review_result.review_notes else 'All tests passed'}"
                assistant_message = AssistantMessage(
                    content=review_content,
                    source="ReviewerAgent"
                )
                session_manager.add_session_message(assistant_message)
            
            return  # Message handlers should return None

        review_result = None
        
        if self._model_client and message.code_gen_state.code_result and message.code_gen_state.test_results:
            review_result = await self._get_llm_feedback(message)
        else:
            review_result = self._get_fallback_feedback(message)

        message.code_gen_state.review_notes = review_result.review_notes

        session_manager = get_global_session_manager(message.session_id) if message.session_id else None
        if session_manager:
            review_content = f"ReviewerAgent (Iteration {message.code_gen_state.iteration}): {', '.join(review_result.review_notes) if review_result.review_notes else 'All tests passed'}"
            assistant_message = AssistantMessage(
                content=review_content,
                source="ReviewerAgent"
            )
            session_manager.add_session_message(assistant_message)

        
        if not review_result.should_continue:
            message.code_gen_state.process_completed = True
            logger.info("All tests passed - process completed successfully!")

        elif message.code_gen_state.iteration < message.code_gen_state.max_iterations:
            if message.code_gen_state.code_result:
                message.code_gen_state.previous_code = message.code_gen_state.code_result.code
            logger.info(f"Review complete - {len(review_result.review_notes)} issues to address. Will start a new coding iteration for improvements.")
            await self.publish_message(message, topic_id=TopicId("CodingRoom", source=self.id.key))
        else:
            message.code_gen_state.process_completed = True
            logger.warning(f"Max iterations ({message.code_gen_state.max_iterations}) reached without completing all tests")



    def _extract_json_from_markdown(self, content: str) -> str:
        """Extract JSON from markdown code blocks if present"""
        if content.startswith("```json") and content.endswith("```"):
            lines = content.split('\n')
            if len(lines) >= 3:
                json_lines = lines[1:-1]
                return '\n'.join(json_lines)
        return content
    
    async def _get_llm_feedback(self, message: State) -> ReviewResult:
        """Get detailed feedback from LLM"""
        
        failure_details = ""
        if message.code_gen_state.test_results and message.code_gen_state.test_results.failures:
            for failure in message.code_gen_state.test_results.failures[:5]:  # Limit to first 5 failures
                failure_details += f"- {failure.name}: {failure.trace[:500]}\n"
        
        prompt = REVIEWER_PROMPT_TEMPLATE.format(
            code=message.code_gen_state.code_result.code if message.code_gen_state.code_result else "No code available",
            passed=message.code_gen_state.test_results.passed if message.code_gen_state.test_results else 0,
            failed=message.code_gen_state.test_results.failed if message.code_gen_state.test_results else 0,
            test_explanation=message.code_gen_state.test_results.test_explanation if message.code_gen_state.test_results else "",
            failure_details=failure_details or "No specific failure details available"
        )
        
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            if attempt == 1:
                logger.info("Requesting code review from LLM...")
            else:
                logger.info(f"Retry attempt {attempt}/{max_retries}...")
            
            try:
                params = {
                    "model": message.model_name,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ]
                }
                response = self._model_client.chat.completions.create(**params)
                content = response.choices[0].message.content
                
                # Track usage from the result
                if message:
                    track_usage_from_result(response, message)
                
                if content and content.strip():
                    content = content.strip()
                    logger.info(f"Received review response: {content[:100]}..." if len(content) > 100 else f"Received review response: {content}")
                    
                    # Extract JSON from markdown code blocks if present
                    json_content = self._extract_json_from_markdown(content)
                    
                    try:
                        # Try parsing as JSON first
                        response_json = json.loads(json_content)
                        review_notes = response_json.get("review_notes", [])
                        analysis = response_json.get("analysis", "")
                        recommendation = response_json.get("recommendation", "revise")
                        
                        logger.info("Successfully parsed JSON review response")
                        return ReviewResult(
                            review_notes=review_notes,
                            analysis=analysis,
                            recommendation=recommendation,
                            should_continue=(recommendation == "revise")
                        )
                    
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON parsing failed: {e}")
                        logger.info(f"Raw content causing JSON error: '{json_content[:200]}'")
                        # If JSON parsing fails, extract basic review info from text
                        logger.info("Parsing non-JSON review response")
                        
                        # Try to extract key information from text
                        review_notes = [content[:200] + "..." if len(content) > 200 else content]
                        analysis = "Review analysis from raw response"
                        
                        # Simple heuristic: if "approve" or "pass" in response, don't continue
                        recommendation = "approve" if any(word in content.lower() for word in ["approve", "pass", "complete", "success"]) else "revise"
                        
                        return ReviewResult(
                            review_notes=review_notes,
                            analysis=analysis,
                            recommendation=recommendation,
                            should_continue=(recommendation == "revise")
                        )
                else:
                    logger.warning(f"Empty or None content from response: {content}")

                    return ReviewResult(
                        review_notes=["No response received from reviewer"],
                        analysis="Empty response received",
                        recommendation="revise",
                        should_continue=True
                    )
                    
            except Exception as e:
                logger.error(f"Review request failed: {e}")
        
        # If LLM fails, use fallback
        logger.warning("LLM review failed, using fallback")
        return self._get_fallback_feedback(message)
    
    def _get_fallback_feedback(self, message: State) -> ReviewResult:
        """Provide basic feedback without LLM"""
        
        review_notes = []
        
        if message.code_gen_state.test_results:
            if message.code_gen_state.test_results.failed > 0:
                review_notes.append(f"Fix {message.code_gen_state.test_results.failed} failing tests")
                
                # Add specific failure notes
                if message.code_gen_state.test_results.failures:
                    for failure in message.code_gen_state.test_results.failures[:3]:
                        review_notes.append(f"Fix test: {failure.name}")
                
                review_notes.append("Check input validation and edge cases")
                review_notes.append("Verify Unicode normalization is working correctly")
                review_notes.append("Ensure proper error handling for invalid inputs")
            
            analysis = f"Code has {message.code_gen_state.test_results.failed} test failures that need to be addressed."
            recommendation = "revise"
            should_continue = True
        else:
            review_notes.append("No test results available - ensure tests are running")
            analysis = "Unable to assess code without test results."
            recommendation = "revise"
            should_continue = True
        
        return ReviewResult(
            review_notes=review_notes,
            analysis=analysis,
            recommendation=recommendation,
            should_continue=should_continue
        )