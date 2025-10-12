"""
Handoff agents implementation for routing between specialized agents
"""

import logging
import tiktoken
from prompts import ROUTING_AGENT_PROMPT, LEGAL_EXPERT_PROMPT, GENERAL_AGENT_PROMPT, CONTENT_SAFETY_PROMPT
# Agent import removed - using AgentWorkflow and FunctionAgent directly
from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core import Settings
from state import MainState
from tools import get_all_tools
from pydantic import BaseModel, Field

# Set up logger with the same configuration as main
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ContentSafetyResult(BaseModel):
    reason: str = Field(description="the reason for the content being inappropriate")
    category: str = Field(description="the category of the content being inappropriate")


async def setup_handoff_agents(session_id: str = None) -> tuple[AgentWorkflow, MainState]:
    """
    Set up the complete handoff system with routing agent and specialized agents.
    
    This function creates a multi-agent workflow system consisting of:
    - RouterAgent: Routes user requests to appropriate specialized agents
    - GeneralAgent: Handles general questions and tool usage with MCP servers
    - LegalAgent: Specialized in legal questions and expertise
    - GuardrailAgent: Content safety checking
    
    The system provides intelligent routing between agents based on the nature
    of user requests, allowing for specialized handling of different query types.
    
    Uses TokenCountingHandler for REAL token usage tracking instead of estimates.
    
    Args:
        session_id (str, optional): Unique identifier for the session. 
                                  Used for tracking and state management.
    
    Returns:
        tuple[AgentWorkflow, MainState]: A tuple containing:
            - agent_workflow: The configured AgentWorkflow instance with all agents
            - main_state: The MainState instance for session tracking and usage
    """

    main_state = MainState()
    main_state.session_id = session_id

    # Token counter + callback manager for REAL usage tracking
    token_counter = TokenCountingHandler(
        tokenizer=tiktoken.encoding_for_model(main_state.model_name).encode
    )
    cb_manager = CallbackManager([token_counter])
    
    # Store token counter on main_state so the chat loop can read it
    main_state.token_counter = token_counter

    # Use the same callback manager on the LLM used by all agents
    llm = OpenAI(model=main_state.model_name, callback_manager=cb_manager)
    tools = await get_all_tools()
    

    router_agent = FunctionAgent(
        name="RouterAgent",
        description="An agent that routes the user's request to the appropriate agent.",
        system_prompt=ROUTING_AGENT_PROMPT,
        llm=llm,
        tools=[],
        can_handoff_to=["GuardrailAgent", "GeneralAgent", "LegalAgent"],
    )

    general_agent = FunctionAgent(
        name="GeneralAgent",
        description="An agent that can answer general questions and use tools and MCP servers to help the user. Use this agent when the request doesnt match any of the other agents.",
        system_prompt=GENERAL_AGENT_PROMPT,
        llm=llm,
        tools=tools
    )

    legal_agent = FunctionAgent(
        name="LegalAgent",
        description="An agent specialized in legal questions.",
        system_prompt=LEGAL_EXPERT_PROMPT,
        llm=llm,
        tools=[]
    )

    if main_state.enable_content_safety:
        guardrail_agent = FunctionAgent(
            name="GuardrailAgent",
            description="An agent that checks the content of the user's request for safety.",
            system_prompt=CONTENT_SAFETY_PROMPT,
            llm=llm,
            output_cls=ContentSafetyResult,
            tools=[]
        )

    initial_state = {
        "main_state": main_state
    }

    agents = [router_agent, general_agent, legal_agent]
    if main_state.enable_content_safety:
        agents.append(guardrail_agent)
    
    agent_workflow = AgentWorkflow(
        agents=agents,
        root_agent=router_agent.name,
        initial_state=initial_state
    )
    
    return agent_workflow, main_state