
import logging
from typing import List
from dataclasses import dataclass, field
from prompts import REVIEWER_PROMPT_TEMPLATE
from state import update_usage_from_crewai, get_global_state
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class ReviewResponse(BaseModel):
    review_notes: List[str]
    analysis: str
    recommendation: str

@dataclass
class ReviewResult:
    """Result from the reviewer agent"""
    review_notes: List[str] = field(default_factory=list)
    analysis: str = ""
    recommendation: str = ""
    should_continue: bool = True

class ReviewerAgent:
    def __init__(self, state, agent) -> None:
        self.state = state
        self.agent = agent

    def assess(self) -> ReviewResult:
        """Review the code and test results to provide feedback"""
        
        # Check if all tests passed
        if self.state.code_gen_state.test_results and self.state.code_gen_state.test_results.failed == 0:
            logger.info("All tests passed - recommending approval")
            return ReviewResult(
                review_notes=["All tests passed successfully. Implementation is complete."],
                analysis="The code successfully passes all test cases.",
                recommendation="approve",
                should_continue=False
            )
        
        # If we have an agent, get detailed feedback
        if self.agent and self.state.code_gen_state.code_result and self.state.code_gen_state.test_results:
            return self._get_agent_feedback()
        
        # Fallback review for failed tests without LLM
        return self._get_fallback_feedback(self.state)
    
    def _get_agent_feedback(self) -> ReviewResult:
        """Get detailed feedback from Agent"""
        
        failure_details = ""
        if self.state.code_gen_state.test_results and self.state.code_gen_state.test_results.failures:
            for failure in self.state.code_gen_state.test_results.failures[:5]:  # Limit to first 5 failures
                failure_details += f"- {failure.name}: {failure.trace[:500]}\n"
        
        prompt = REVIEWER_PROMPT_TEMPLATE.format(
            code=self.state.code_gen_state.code_result.code if self.state.code_gen_state.code_result else "No code available",
            passed=self.state.code_gen_state.test_results.passed if self.state.code_gen_state.test_results else 0,
            failed=self.state.code_gen_state.test_results.failed if self.state.code_gen_state.test_results else 0,
            test_explanation=self.state.code_gen_state.test_results.test_explanation if self.state.code_gen_state.test_results else "",
            failure_details=failure_details or "No specific failure details available"
        )
        
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            if attempt == 1:
                logger.info("Requesting code review from Agent...")
            else:
                logger.info(f"Retry attempt {attempt}/{max_retries}...")
                
            try:
                result = self.agent.kickoff(prompt, response_format=ReviewResponse)

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
                    review_notes = response.review_notes or []
                    analysis = response.analysis or ""
                    recommendation = response.recommendation or "revise"
                    
                    if review_notes:
                        logger.info("Review completed successfully")
                        return ReviewResult(
                            review_notes=review_notes,
                            analysis=analysis,
                            recommendation=recommendation,
                            should_continue=(recommendation == "revise")
                        )
                    
            except Exception as e:
                logger.error(f"Review request failed: {e}")
        
        # If Agent fails, use fallback
        logger.warning("Agent review failed, using fallback")
        return self._get_fallback_feedback()
    
    def _get_fallback_feedback(self) -> ReviewResult:
        """Provide basic feedback without LLM"""
        
        review_notes = []
        
        if self.state.code_gen_state.test_results:
            if self.state.code_gen_state.test_results.failed > 0:
                review_notes.append(f"Fix {self.state.code_gen_state.test_results.failed} failing tests")
                
                # Add specific failure notes
                if self.state.code_gen_state.test_results.failures:
                    for failure in self.state.code_gen_state.test_results.failures[:3]:
                        review_notes.append(f"Fix test: {failure.name}")
                
                review_notes.append("Check input validation and edge cases")
                review_notes.append("Verify Unicode normalization is working correctly")
                review_notes.append("Ensure proper error handling for invalid inputs")
            
            analysis = f"Code has {self.state.code_gen_state.test_results.failed} test failures that need to be addressed."
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