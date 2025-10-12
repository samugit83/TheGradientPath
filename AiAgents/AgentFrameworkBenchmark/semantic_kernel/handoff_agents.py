"""
Handoff agents implementation for routing between specialized agents
"""

import logging

from prompts import (
    ROUTING_AGENT_PROMPT,
    LEGAL_EXPERT_PROMPT,
    GENERAL_AGENT_PROMPT
)
from semantic_kernel.agents import (
    ChatCompletionAgent,
    OrchestrationHandoffs,
    HandoffOrchestration
)
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.contents import ChatMessageContent
from tools import get_all_plugin_instances

logger = logging.getLogger(__name__)


def setup_handoff_agents() -> any:

    openai_service = OpenAIChatCompletion(ai_model_id="gpt-4.1")

    router_agent = ChatCompletionAgent(
        name="RouterAgent",
        description="An agent that routes the user's request to the appropriate agent.",
        instructions=ROUTING_AGENT_PROMPT,
        service=openai_service
    )

    general_agent = ChatCompletionAgent(
        name="GeneralAgent",
        description="An agent that can answer general questions and use tools and MCP servers to help the user. Use this agent when the request doesnt match any of the other agents.",
        instructions=GENERAL_AGENT_PROMPT,
        service=openai_service,
        plugins=get_all_plugin_instances()
    )

    legal_agent = ChatCompletionAgent(
        name="LegalAgent",
        description="An agent specialized in legal questions.",
        instructions=LEGAL_EXPERT_PROMPT,
        service=openai_service
    )

    handoffs = (
        OrchestrationHandoffs()
        .add_many(  
            source_agent=router_agent.name,
            target_agents={
                general_agent.name: "Transfer to this agent if the issue is general questions or requires tools or MCP server tools.",
                legal_agent.name: "Transfer to this agent if the issue is legal related",
            },
        )
    )

    def agent_response_callback(message: ChatMessageContent) -> None:
        if message.content:
            print(f"💬 Assistant {message.name}: {message.content}")
            
            # Log assistant messages to session manager (only for GeneralAgent and LegalAgent)
            # The RuntimeChat instance will be set on the handoff_orchestration object
            if (hasattr(agent_response_callback, '_runtime_chat') and 
                agent_response_callback._runtime_chat and 
                message.name in ["GeneralAgent", "LegalAgent"]):
                try:
                    runtime_chat = agent_response_callback._runtime_chat
                    # Add agent message with agent metadata
                    runtime_chat.session_manager.add_agent_message(
                        agent_name=message.name,
                        agent_type="handoff_agent",
                        content=message.content,
                        task_type="chat"
                    )
                    logger.debug(f"Logged {message.name} response to session")
                except Exception as e:
                    logger.warning(f"Failed to log {message.name} response to session: {e}")


    handoff_orchestration = HandoffOrchestration(
        members=[
            router_agent,
            general_agent,
            legal_agent,
        ],
        handoffs=handoffs,
        agent_response_callback=agent_response_callback
    )
    
    return handoff_orchestration