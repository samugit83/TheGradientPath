import logging
from typing import Optional
from dataclasses import replace
from state import State, CodeGenState
from .coder import CoderAgent
from .tester import TesterAgent
from .reviewer import ReviewerAgent
from models import LLMClient
import json
from session_manager import get_session_manager, add_agent_message

logger = logging.getLogger(__name__)

class CodeOrchestratorAgent:
    def __init__(
        self,
        user_prompt: str,
        constraints: Optional[dict] = None,
        max_iters: int = 3,
        package_name: str = "app",
        llm_client: Optional[LLMClient] = None,
        max_test: int = 15
    ) -> None:

        self.llm_client = llm_client or LLMClient()
        
        self.coder = CoderAgent(package_name=package_name, llm_client=self.llm_client)
        self.tester = TesterAgent(package_name=package_name, llm_client=self.llm_client, max_test=max_test)
        self.reviewer = ReviewerAgent(llm_client=self.llm_client)
        
        self.max_iters = max_iters
        self.state = State(user_prompt=user_prompt)
        # Initialize code_gen_state with constraints
        self.state.code_gen_state = CodeGenState(constraints=constraints or {})

    def step_code(self):
        """Generate code"""
        try:
            self.coder.produce(self.state)
            
            # Log to persistent session
            session_manager = get_session_manager()
            if session_manager and self.state.code_gen_state and self.state.code_gen_state.code_result:
                add_agent_message(
                    agent_name="CoderAgent",
                    agent_type="coder",
                    content=f"Generated code:\n{self.state.code_gen_state.code_result.code_explanation}",
                    iteration=self.state.code_gen_state.iteration
                )
            
            # Log usage after code generation
            if self.llm_client:
                usage = self.llm_client.get_total_usage()
                logger.info(f"Cumulative usage after code generation: {usage['requests']} requests")
        except RuntimeError as e:
            logger.error(f"Code generation failed: {e}")

    def step_tests(self):
        """Write tests and run them"""
        try:
            self.tester.write_tests(self.state)
            self.tester.run_tests(self.state)
            
            # Log to persistent session
            session_manager = get_session_manager()
            if session_manager and self.state.code_gen_state and self.state.code_gen_state.test_results:
                test_results = self.state.code_gen_state.test_results
                add_agent_message(
                    agent_name="TesterAgent",
                    agent_type="tester",
                    content=f"Test results: {test_results.passed} passed, {test_results.failed} failed\n{test_results.test_explanation}",
                    iteration=self.state.code_gen_state.iteration
                )
            
            # Log usage after test generation
            if self.llm_client:
                usage = self.llm_client.get_total_usage()
                logger.info(f"Cumulative usage after test generation: {usage['requests']} requests")
        except RuntimeError as e:
            logger.error(f"Test generation failed: {e}")

    def step_review(self):
        """Review the code and test results, decide if process is complete"""
        review_result = self.reviewer.assess(self.state)
        self.state.code_gen_state.review_notes = review_result.review_notes
        
        # Log to persistent session
        session_manager = get_session_manager()
        if session_manager:
            add_agent_message(
                agent_name="ReviewerAgent",
                agent_type="reviewer",
                content=f"Review: {', '.join(review_result.review_notes) if review_result.review_notes else 'All tests passed'}",
                iteration=self.state.code_gen_state.iteration
            )
        
        # Log usage after review
        if self.llm_client:
            usage = self.llm_client.get_total_usage()
            logger.info(f"Cumulative usage after review: {usage['requests']} requests")
        
        if not review_result.should_continue:
            self.state.code_gen_state.process_completed = True
            logger.info("All tests passed - process completed successfully!")
        else:
            if self.state.code_gen_state.code_result:
                self.state.code_gen_state.previous_code = self.state.code_gen_state.code_result.code
            logger.info(f"Review complete - {len(review_result.review_notes)} issues to address")
    
    def run(self) -> State:
        """Run the complete orchestration loop"""
        
        for _ in range(self.max_iters):
            self.state.code_gen_state.iteration += 1
            logger.info(f"Starting iteration {self.state.code_gen_state.iteration}")
            
            self.step_code()
            logger.info(f"State after code generation - {json.dumps(self.state.__dict__, indent=2, default=str)}")
            
            self.step_tests()
            logger.info(f"State after tests - {json.dumps(self.state.__dict__, indent=2, default=str)}")
            
            self.step_review()
            logger.info(f"State after review - {json.dumps(self.state.__dict__, indent=2, default=str)}")
            
            if self.state.code_gen_state.process_completed:
                logger.info(f"Process completed successfully in iteration {self.state.code_gen_state.iteration}!")
                break
            
            logger.info(f"Iteration {self.state.code_gen_state.iteration} complete - continuing with improvements")
        
        if not self.state.code_gen_state.process_completed:
            logger.warning(f"Max iterations ({self.max_iters}) reached without completing all tests")
        
        # Update state with token usage from the LLM client
        if self.llm_client:
            usage = self.llm_client.get_total_usage()
            self.state.update_usage(usage)
            logger.info(f"Code generation complete - Total LLM calls: {usage['requests']}")
            logger.info(f"Total token usage - Input: {usage['input_tokens']:,}, Output: {usage['output_tokens']:,}, Total: {usage['total_tokens']:,}")
        
        return replace(self.state)

