import logging
from dataclasses import replace
from pathlib import Path
import shutil
from state import State
from .coder import CoderAgent
from .tester import TesterAgent
from .reviewer import ReviewerAgent
import json
from state import get_global_state
from crewai.flow.flow import Flow, listen, start
from crewai import Agent
from crewai.llm import LLM
from prompts import (
    CODER_AGENT_GOAL, 
    CODER_AGENT_BACKSTORY,
    TESTER_AGENT_GOAL,
    TESTER_AGENT_BACKSTORY,
    REVIEWER_AGENT_GOAL,
    REVIEWER_AGENT_BACKSTORY
)


logger = logging.getLogger(__name__)

class CodeOrchestratorAgent:
    def __init__(
        self
    ) -> None:

        self.llm = LLM(model="gpt-4-turbo")
        self.state = get_global_state()
        
        # Create CrewAI agents
        self.coder_agent = Agent(
            role="Python Coder",
            goal=CODER_AGENT_GOAL,
            backstory=CODER_AGENT_BACKSTORY,
            verbose=True,
            llm=self.llm
        )
        
        self.tester_agent = Agent(
            role="Python Tester",
            goal=TESTER_AGENT_GOAL,
            backstory=TESTER_AGENT_BACKSTORY,
            verbose=True,
            llm=self.llm
        )
        
        self.reviewer_agent = Agent(
            role="Code Reviewer",
            goal=REVIEWER_AGENT_GOAL,
            backstory=REVIEWER_AGENT_BACKSTORY,
            verbose=True,
            llm=self.llm
        )
        
        self.coder = CoderAgent(self.state, self.coder_agent)
        self.tester = TesterAgent(self.state, self.tester_agent)
        self.reviewer = ReviewerAgent(self.state, self.reviewer_agent)
        self.flow = self.StructuredStateFlow(self, self.coder, self.tester, self.reviewer)
    
    class StructuredStateFlow(Flow):
        def __init__(self, parent, coder, tester, reviewer):
            super().__init__()
            self.parent = parent  # Reference to CodeOrchestratorAgent instance
            self.coder = coder
            self.tester = tester
            self.reviewer = reviewer
        
        @start()
        def step_code(self):
            self.coder.produce()
            logger.info(f"State after code generation - {json.dumps(self.parent.state.__dict__, indent=2, default=str)}")
            return "Code generated successfully"

        @listen(step_code)
        def step_tests(self, previous_result):
            """Write tests and run them"""
            try:    
                logger.info(f"Return from Coder: {previous_result}")
                self.tester.write_tests()
                self.tester.run_tests()
                logger.info(f"State after tests - {json.dumps(self.parent.state.__dict__, indent=2, default=str)}")
                return "Tests completed"
            except RuntimeError as e:
                logger.error(f"Test generation failed: {e}")
                return f"Test failed: {e}"

        @listen(step_tests)
        def step_review(self, previous_result):
            """Review the code and test results, decide if process is complete"""
            logger.info(f"Return from Tests: {previous_result}")
            review_result = self.reviewer.assess()
            self.parent.state.code_gen_state.review_notes = review_result.review_notes
            logger.info(f"State after review - {json.dumps(self.parent.state.__dict__, indent=2, default=str)}")
            
            if not review_result.should_continue:
                self.parent.state.code_gen_state.process_completed = True
                logger.info("All tests passed - process completed successfully!")
                return "Process completed successfully"
            else:
                if self.parent.state.code_gen_state.code_result:
                    self.parent.state.code_gen_state.previous_code = self.parent.state.code_gen_state.code_result.code
                logger.info(f"Review complete - {len(review_result.review_notes)} issues to address")
                return f"Review complete - {len(review_result.review_notes)} issues to address"

        
    def run(self) -> State:
        """Run the complete orchestration loop"""
        
        for _ in range(self.state.code_gen_state.max_iterations):
            self.state.code_gen_state.iteration += 1
            logger.info(f"Starting iteration {self.state.code_gen_state.iteration}")

            app_dir = Path(__file__).parent.parent / self.state.code_gen_state.package_name
            if app_dir.exists():
                logger.info("Emptying app folder before code generation")
                shutil.rmtree(app_dir)
            
            self.flow.kickoff()
            
            if self.state.code_gen_state.process_completed:
                logger.info(f"Process completed successfully in iteration {self.state.code_gen_state.iteration}!")
                break
            
            logger.info(f"Iteration {self.state.code_gen_state.iteration} complete - continuing with improvements")
        
        if not self.state.code_gen_state.process_completed:
            logger.warning(f"Max iterations ({self.state.code_gen_state.max_iterations}) reached without completing all tests")
        
        return replace(self.state)


