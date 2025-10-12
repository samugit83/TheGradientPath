"""
Handoff agents implementation for routing between specialized agents
"""

from agents import Agent, handoff, RunContextWrapper
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from agents.extensions import handoff_filters
from typing import List, Any
from pydantic import BaseModel
import logging
from prompts import ROUTING_AGENT_PROMPT, LEGAL_EXPERT_PROMPT, GENERAL_AGENT_PROMPT

# Set up logger with the same configuration as main
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RouteDecision(BaseModel):
    """Output model for the routing agent's decision"""
    route_to_legal: bool
    reason: str




def on_legal_handoff(ctx: RunContextWrapper[None]):
    """Callback when handing off to the Legal Expert"""
    logger.info("HANDOFF: Routing to Legal Expert agent for law-related question")
    print("⚖️ Routing to Legal Expert...")
    return None


def on_general_handoff(ctx: RunContextWrapper[None]):
    """Callback when handing off to General Agent"""  
    logger.info("HANDOFF: Routing to General Agent for non-legal task")
    print("🔧 Routing to General Agent...")
    return None


def setup_handoff_agents(
    tools: List[Any],
    mcp_servers: List[Any],
    model_name: str = "gpt-4o",
    enable_content_safety: bool = True
) -> Agent:
    """
    Set up the complete handoff system with routing agent and specialized agents.
    
    Args:
        tools: List of tools for the General Agent
        mcp_servers: List of MCP servers for the General Agent
        model_name: Model to use for the agents
        enable_content_safety: Whether to enable content safety guardrails (only on router)
    
    Returns:
        The configured routing agent with handoffs
    """
    # Create the specialized agents (without content safety)
    legal_expert = Agent(
        name="Legal Expert",
        instructions=f"{RECOMMENDED_PROMPT_PREFIX}\n\n{LEGAL_EXPERT_PROMPT}",
        model=model_name,
        tools=[],  # Legal expert doesn't need tools
        mcp_servers=[]
    )
    
    general_agent = Agent(
        name="General Agent",
        instructions=f"{RECOMMENDED_PROMPT_PREFIX}\n\n{GENERAL_AGENT_PROMPT}",
        model=model_name,
        tools=tools,
        mcp_servers=mcp_servers or []
    )

    legal_handoff = handoff(
        agent=legal_expert,
        on_handoff=on_legal_handoff,
        tool_name_override="transfer_to_legal_expert",
        tool_description_override="Transfer to the Legal Expert for law-related questions",
        input_filter=handoff_filters.remove_all_tools  # Remove tool calls from history for legal expert
    )
    
    general_handoff = handoff(
        agent=general_agent,
        on_handoff=on_general_handoff,
        tool_name_override="transfer_to_general_agent",
        tool_description_override="Transfer to the General Agent for all non-legal tasks"
    )
    
    from guardrails import content_safety_guardrail
    
    if enable_content_safety:
        routing_agent = Agent(
            name="Router",
            instructions=f"{RECOMMENDED_PROMPT_PREFIX}\n\n{ROUTING_AGENT_PROMPT}",
            model="gpt-4o-mini",  # Use lighter model for routing
            tools=[],  # NO tools for router - forces handoff usage
            handoffs=[legal_handoff, general_handoff]  # Pass handoffs during creation
        )
        routing_agent.input_guardrails = [content_safety_guardrail]
    else:
        routing_agent = Agent(
            name="Router",
            instructions=f"{RECOMMENDED_PROMPT_PREFIX}\n\n{ROUTING_AGENT_PROMPT}",
            model="gpt-4o-mini",
            tools=[],  # NO tools for router - forces handoff usage
            handoffs=[legal_handoff, general_handoff]  # Pass handoffs during creation
        )
    
    return routing_agent