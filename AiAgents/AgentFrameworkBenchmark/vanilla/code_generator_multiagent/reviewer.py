import json
import logging
from typing import List
from dataclasses import dataclass, field
from prompts import REVIEWER_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

@dataclass
class ReviewResult:
    """Result from the reviewer agent"""
    review_notes: List[str] = field(default_factory=list)
    analysis: str = ""
    recommendation: str = ""
    should_continue: bool = True

class ReviewerAgent:
    def __init__(self, llm_client=None) -> None:
        self.llm_client = llm_client

    def assess(self, state) -> ReviewResult:
        """Review the code and test results to provide feedback"""
        
        # Check if all tests passed
        if state.code_gen_state.test_results and state.code_gen_state.test_results.failed == 0:
            logger.info("All tests passed - recommending approval")
            return ReviewResult(
                review_notes=["All tests passed successfully. Implementation is complete."],
                analysis="The code successfully passes all test cases.",
                recommendation="approve",
                should_continue=False
            )
        
        # If we have an LLM client, get detailed feedback
        if self.llm_client and state.code_gen_state.code_result and state.code_gen_state.test_results:
            return self._get_llm_feedback(state)
        
        # Fallback review for failed tests without LLM
        return self._get_fallback_feedback(state)
    
    def _get_llm_feedback(self, state) -> ReviewResult:
        """Get detailed feedback from LLM"""
        
        failure_details = ""
        if state.code_gen_state.test_results and state.code_gen_state.test_results.failures:
            for failure in state.code_gen_state.test_results.failures[:5]:  # Limit to first 5 failures
                failure_details += f"- {failure.name}: {failure.trace[:500]}\n"
        
        prompt = REVIEWER_PROMPT_TEMPLATE.format(
            code=state.code_gen_state.code_result.code if state.code_gen_state.code_result else "No code available",
            passed=state.code_gen_state.test_results.passed if state.code_gen_state.test_results else 0,
            failed=state.code_gen_state.test_results.failed if state.code_gen_state.test_results else 0,
            test_explanation=state.code_gen_state.test_results.test_explanation if state.code_gen_state.test_results else "",
            failure_details=failure_details or "No specific failure details available"
        )
        
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            if attempt == 1:
                logger.info("Requesting code review from LLM...")
            else:
                logger.info(f"Retry attempt {attempt}/{max_retries}...")
            
            try:
                llm_response_str = self.llm_client.api_call(prompt)
                
                if llm_response_str:
                    try:
                        response_json = json.loads(llm_response_str)
                        review_notes = response_json.get("review_notes", [])
                        analysis = response_json.get("analysis", "")
                        recommendation = response_json.get("recommendation", "revise")
                        
                        if review_notes:
                            logger.info("Review completed successfully")
                            return ReviewResult(
                                review_notes=review_notes,
                                analysis=analysis,
                                recommendation=recommendation,
                                should_continue=(recommendation == "revise")
                            )
                    
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON parsing failed: {e}")
                    
            except Exception as e:
                logger.error(f"Review request failed: {e}")
        
        # If LLM fails, use fallback
        logger.warning("LLM review failed, using fallback")
        return self._get_fallback_feedback(state)
    
    def _get_fallback_feedback(self, state) -> ReviewResult:
        """Provide basic feedback without LLM"""
        
        review_notes = []
        
        if state.code_gen_state.test_results:
            if state.code_gen_state.test_results.failed > 0:
                review_notes.append(f"Fix {state.code_gen_state.test_results.failed} failing tests")
                
                # Add specific failure notes
                if state.code_gen_state.test_results.failures:
                    for failure in state.code_gen_state.test_results.failures[:3]:
                        review_notes.append(f"Fix test: {failure.name}")
                
                review_notes.append("Check input validation and edge cases")
                review_notes.append("Verify Unicode normalization is working correctly")
                review_notes.append("Ensure proper error handling for invalid inputs")
            
            analysis = f"Code has {state.code_gen_state.test_results.failed} test failures that need to be addressed."
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