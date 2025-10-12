import logging
from typing import Optional
from dataclasses import replace
from state import MainState, CodeGenState
from .coder import CoderAgent
from .tester import TesterAgent
from .reviewer import ReviewerAgent
from langchain.chat_models import init_chat_model
import json


logger = logging.getLogger(__name__)

class CodeOrchestratorAgent:
    def __init__(
        self,
        llm_client = None
    ) -> None:

        self.llm_client = llm_client
        # Initialize token tracking
        self.total_usage = {"requests": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    
    def _aggregate_usage(self, usage_dict: dict):
        """Aggregate token usage from individual agents"""
        self.total_usage["requests"] += usage_dict.get("requests", 0)
        self.total_usage["input_tokens"] += usage_dict.get("input_tokens", 0)
        self.total_usage["output_tokens"] += usage_dict.get("output_tokens", 0)
        self.total_usage["total_tokens"] += usage_dict.get("total_tokens", 0) 

    def step_code(self, state: MainState):
        """Generate code"""
        try:
            usage = self.coder.produce(state)
            self._aggregate_usage(usage)
        except RuntimeError as e:
            logger.error(f"Code generation failed: {e}")

    def step_tests(self, state: MainState):
        """Write tests and run them"""
        try:
            usage = self.tester.write_tests(state)
            self._aggregate_usage(usage)
            self.tester.run_tests(state)  # run_tests doesn't use LLM, so no token usage
        except RuntimeError as e:
            logger.error(f"Test generation failed: {e}")

    def step_review(self, state: MainState):
        """Review the code and test results, decide if process is complete"""
        review_result = self.reviewer.assess(state)
        state.code_gen_state.review_notes = review_result.review_notes
        
        # Aggregate token usage from reviewer
        self._aggregate_usage(review_result.token_usage)
        
        if not review_result.should_continue:
            state.code_gen_state.process_completed = True
            logger.info("All tests passed - process completed successfully!")
        else:
            if state.code_gen_state.current_code:
                state.code_gen_state.previous_code = state.code_gen_state.current_code.code
            logger.info(f"Review complete - {len(review_result.review_notes)} issues to address")
    
    def run(self, state: MainState) -> MainState:
        """Run the complete orchestration loop"""
        
        # Initialize agents with state parameters
        package_name = state.code_gen_state.package_name
        max_test = state.code_gen_state.max_tests
        max_iters = state.code_gen_state.max_iterations
        
        self.coder = CoderAgent(package_name=package_name, llm_client=self.llm_client)
        self.tester = TesterAgent(package_name=package_name, llm_client=self.llm_client, max_test=max_test)
        self.reviewer = ReviewerAgent(llm_client=self.llm_client)
        
        for _ in range(max_iters):
            state.code_gen_state.iteration += 1
            logger.info(f"Starting iteration {state.code_gen_state.iteration}")
            
            self.step_code(state)
            logger.info(f"State after code generation - iteration {state.code_gen_state.iteration}")
            
            self.step_tests(state)
            logger.info(f"State after tests - iteration {state.code_gen_state.iteration}")
            
            self.step_review(state)
            logger.info(f"State after review - iteration {state.code_gen_state.iteration}")
            
            if state.code_gen_state.process_completed:
                logger.info(f"Process completed successfully in iteration {state.code_gen_state.iteration}!")
                break
            
            logger.info(f"Iteration {state.code_gen_state.iteration} complete - continuing with improvements")
        
        if not state.code_gen_state.process_completed:
            logger.warning(f"Max iterations ({max_iters}) reached without completing all tests")
        
        # Log final token usage summary for code generation
        if self.total_usage["requests"] > 0:
            logger.info(f"🔢 Code generation complete - Total LLM calls: {self.total_usage['requests']}")
            logger.info(f"🔢 Total token usage - Input: {self.total_usage['input_tokens']:,}, "
                       f"Output: {self.total_usage['output_tokens']:,}, "
                       f"Total: {self.total_usage['total_tokens']:,}")
        
        # Add token usage to state for external access
        state._code_gen_usage = self.total_usage
        
        return state

