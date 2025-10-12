"""
CrewAI agent crew implementation for routing chat requests between specialized agents 
"""

import logging
import os
from crewai import Agent, Task, Crew, Process
from crewai.llm import LLM
from tools import get_tools
from prompts import (
    CONVERSATION_MANAGER_GOAL,
    CONVERSATION_MANAGER_BACKSTORY,
    ROUTING_TASK_DESCRIPTION,
    ROUTING_TASK_EXPECTED_OUTPUT,
    LEGAL_EXPERT_GOAL, 
    LEGAL_EXPERT_BACKSTORY,
    GENERAL_AGENT_GOAL,
    GENERAL_AGENT_BACKSTORY,
    CONTENT_SAFETY_AGENT_GOAL,
    CONTENT_SAFETY_AGENT_BACKSTORY
)

from crewai.memory.long_term.long_term_memory import LongTermMemory
from crewai.memory.short_term.short_term_memory import ShortTermMemory
from crewai.memory.entity.entity_memory import EntityMemory
from state import get_global_state
logger = logging.getLogger(__name__)

class ChatCrew:
    """CrewAI crew configuration for intelligent chat routing.
    
    This class defines and configures all agents and tasks needed for the
    conversational AI system, including a conversation manager that routes
    requests to specialized agents based on the query content.
    """
    
    def __init__(self):
        """Initialize the ChatCrew with GPT-4o model configuration."""
        self.llm = LLM(model="gpt-4.1")


    
    def chat_manager(self) -> Agent:
        """Create the Conversation Manager agent.
        
        This agent acts as the intelligent router that analyzes user queries
        and delegates them to the appropriate specialist agent.
        
        Returns:
            Agent: The configured Conversation Manager agent
        """
        return Agent(
            role="Conversation Manager",
            goal=CONVERSATION_MANAGER_GOAL,
            backstory=CONVERSATION_MANAGER_BACKSTORY,
            allow_delegation=True,
            allowed_agents=["Legal Expert", "General Agent", "Content Safety Agent"],
            verbose=True,
            llm=self.llm
        )

    def content_safety_agent(self) -> Agent:
        """Create the Content Safety Agent.
        
        This agent is responsible for checking the content of the user's request and determining if it is appropriate.
        """
        return Agent(
            role="Content Safety Agent",
            goal=CONTENT_SAFETY_AGENT_GOAL,
            backstory=CONTENT_SAFETY_AGENT_BACKSTORY,
            verbose=True,
            llm=self.llm
        )

    def legal_agent(self) -> Agent:
        """Create the Legal Expert agent.
        
        This specialist agent handles all legal-related queries, providing
        expertise on law, regulations, legal procedures, and compliance matters.
        
        Returns:
            Agent: The configured Legal Expert agent
        """
        return Agent(
            role="Legal Expert",
            goal=LEGAL_EXPERT_GOAL,
            backstory=LEGAL_EXPERT_BACKSTORY,
            allow_delegation=False,
            verbose=True,
            llm=self.llm
        )

    def general_agent(self) -> Agent:
        """Create the General Agent.
        
        This general-purpose agent handles non-legal queries including
        programming, tools, general knowledge, conversational questions,
        educational content, creative tasks, and technical operations.
            
        Returns:
            Agent: The configured General Agent
        """            
        return Agent(
            role="General Agent",
            goal=GENERAL_AGENT_GOAL,
            backstory=GENERAL_AGENT_BACKSTORY,
            allow_delegation=False,
            verbose=True,
            llm=self.llm,
            tools=get_tools()
        )

    def route_task(self) -> Task:
        """Create the routing task for the conversation manager.
        
        This task defines how the conversation manager should analyze
        and route user queries to the appropriate specialist agents.
        
        Returns:
            Task: The configured routing task
        """
        return Task(
            description=ROUTING_TASK_DESCRIPTION,
            expected_output=ROUTING_TASK_EXPECTED_OUTPUT,
            agent=self.chat_manager()
        )


    def crew(self, session_id: str) -> Crew:
        """Create and configure the complete CrewAI crew.
        
        Assembles all agents and tasks into a cohesive crew that can
        handle intelligent conversation routing and specialized responses.
        
        Returns:
            Crew: The fully configured CrewAI crew
        """

        state = get_global_state()
        short_term_memory = ShortTermMemory()
        entity_memory = EntityMemory()
        
        if state and state.use_persistent_memory:
            session_id = state.session_id or "default_session"
            db_path = f"./crewai_memory_{session_id}.db"
            long_term_memory = LongTermMemory(path=db_path)
        else:
            long_term_memory = None

        return Crew(
            agents=[
                self.content_safety_agent(),
                self.chat_manager(),
                self.legal_agent(),
                self.general_agent()
            ],
            tasks=[self.route_task()],
            process=Process.sequential,
            verbose=True,
            short_term_memory=short_term_memory,
            entity_memory=entity_memory,
            long_term_memory=long_term_memory,
            llm=self.llm)