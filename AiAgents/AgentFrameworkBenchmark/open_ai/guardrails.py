"""
Content safety guardrails for the AI Assistant
"""

from pydantic import BaseModel
from agents import (
    Agent,
    GuardrailFunctionOutput,
    RunContextWrapper,
    Runner,
    TResponseInputItem,
    input_guardrail,
)
import logging
from prompts import CONTENT_SAFETY_PROMPT

logger = logging.getLogger(__name__)


class ContentSafetyOutput(BaseModel):
    """Output model for content safety guardrail checks"""
    is_inappropriate: bool
    reason: str
    category: str 


content_safety_agent = Agent(
    name="Content Safety Guardrail",
    instructions=CONTENT_SAFETY_PROMPT,
    output_type=ContentSafetyOutput,
    model="gpt-4o-mini"  # Using a faster, cheaper model for guardrails
)


@input_guardrail
async def content_safety_guardrail(
    ctx: RunContextWrapper[None], 
    agent: Agent, 
    input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    """
    Input guardrail that checks for inappropriate content in user requests.
    
    This guardrail runs before the main agent processes the request and blocks
    any content that violates safety guidelines.
    """
    try:

        if isinstance(input, list):
            input_text = " ".join(str(item) for item in input)
        else:
            input_text = str(input)
        
        result = await Runner.run(content_safety_agent, input_text, context=ctx.context)

        if result.final_output.is_inappropriate:
            logger.warning(f"Content safety guardrail triggered: {result.final_output.category} - {result.final_output.reason}")
        
        return GuardrailFunctionOutput(
            output_info=result.final_output,
            tripwire_triggered=result.final_output.is_inappropriate,
        )
        
    except Exception as e:
        logger.error(f"Error in content safety guardrail: {e}", exc_info=True)
        return GuardrailFunctionOutput(
            output_info=ContentSafetyOutput(
                is_inappropriate=False,
                reason="Guardrail check failed, allowing request with caution",
                category="error"
            ),
            tripwire_triggered=False,
        )

