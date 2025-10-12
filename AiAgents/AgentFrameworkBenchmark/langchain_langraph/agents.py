from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel
from prompts import AGENT_ROUTING_PROMPT, LEGAL_EXPERT_PROMPT, GENERAL_AGENT_PROMPT, CONTENT_SAFETY_PROMPT
import logging
import json
from state import MainState, ContentSafetyResult
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage    
from tools import get_all_tools
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
        

class ResponseFormatter(BaseModel):
    routing_decision: str

class ContentSafetyAgent:
    def __init__(self, orchestrator_agent):
        self.orchestrator_agent = orchestrator_agent
        self.llm = self.orchestrator_agent.llm
        self.logger = logging.getLogger(__name__)
        self.logger.info("ContentSafetyAgent initialized.")

    def run(self, state: MainState, config: RunnableConfig):
        """
        Analyze user input for content safety violations.
        Returns updated state with safety analysis results.
        """
        # Extract the latest user message
        user_input = ""
        messages = state.get("messages", []) if isinstance(state, dict) else state.messages
        if messages:
            for message in reversed(messages):
                if isinstance(message, HumanMessage):
                    user_input = message.content
                    break

        # Skip safety check if content safety is disabled
        enable_content_safety = state.get("enable_content_safety", True) if isinstance(state, dict) else state.enable_content_safety
        if not enable_content_safety:
            self.logger.info("Content safety check disabled, proceeding without analysis.")
            return {"content_safety_result": ContentSafetyResult()}

        try:
            # Create safety analysis prompt
            safety_prompt = f"{CONTENT_SAFETY_PROMPT}\n\nUser message to analyze: {user_input}"
            
            # Get safety assessment from LLM
            response = self.llm.invoke(safety_prompt)
            response_content = response.content if hasattr(response, 'content') else str(response)
            
            # Parse JSON response
            safety_data = json.loads(response_content.strip())
            
            # Create safety result
            safety_result = ContentSafetyResult(
                is_appropriate=not safety_data.get("is_inappropriate", False),
                reason=safety_data.get("reason", "Content analysis completed"),
                category=safety_data.get("category", "safe")
            )
            
            # Log safety analysis results
            if not safety_result.is_appropriate:
                self.logger.warning(
                    f"🛡️ CONTENT SAFETY VIOLATION DETECTED - "
                    f"reason: '{safety_result.reason}', "
                    f"category: '{safety_result.category}'"
                )
                self.logger.warning(f"🛡️ Flagged content: '{user_input[:100]}...' (truncated)")
            else:
                self.logger.info("🛡️ Content safety check passed - content is appropriate")
            
            return {"content_safety_result": safety_result}
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse content safety JSON response: {e}")
            # Default to safe on parsing error to avoid blocking legitimate content
            return {"content_safety_result": ContentSafetyResult()}
            
        except Exception as e:
            self.logger.error(f"Content safety analysis error: {e}")
            # Default to safe on error to avoid blocking legitimate content
            return {"content_safety_result": ContentSafetyResult()}

class RoutingAgent:
    def __init__(self, orchestrator_agent):
        self.orchestrator_agent = orchestrator_agent
        self.structured_llm = self.orchestrator_agent.llm.with_structured_output(ResponseFormatter) 
        self.logger = logging.getLogger(__name__)
        self.logger.info("RoutingAgent initialized.")

    def run(self, state: MainState, config: RunnableConfig):
        messages = state.get("messages", []) if isinstance(state, dict) else state.messages
        
        # Format messages for the prompt
        messages_text = ""
        for message in messages:
            if isinstance(message, HumanMessage):
                messages_text += f"User: {message.content}\n"
            elif isinstance(message, AIMessage):
                messages_text += f"Assistant: {message.content}\n"
        
        prompt_template = PromptTemplate(
            input_variables=["messages"],
            template=AGENT_ROUTING_PROMPT
        )

        formatted_prompt = prompt_template.format(messages=messages_text)
        response = self.structured_llm.invoke(formatted_prompt)
        response_dict = response.dict() 
        routeName = response_dict.get("routing_decision", "")

        return {"routeName": routeName}


class LegalExpertAgent:
    def __init__(self, orchestrator_agent):
        self.orchestrator_agent = orchestrator_agent
        self.llm = self.orchestrator_agent.llm
        self.logger = logging.getLogger(__name__)
        self.logger.info("LegalExpertAgent initialized.")

    def run(self, state: MainState, config: RunnableConfig):
        messages = state.get("messages", []) if isinstance(state, dict) else state.messages
        
        messages_text = ""
        for message in messages:
            if isinstance(message, HumanMessage):
                messages_text += f"User: {message.content}\n"
            elif isinstance(message, AIMessage):
                messages_text += f"Assistant: {message.content}\n"

        prompt_template = PromptTemplate(
            input_variables=["messages"],
            template=LEGAL_EXPERT_PROMPT
        )

        formatted_prompt = prompt_template.format(messages=messages_text)
        response = self.llm.invoke(formatted_prompt)
        
        response_content = response.content if hasattr(response, 'content') else str(response)
        
        ai_message = AIMessage(content=response_content)
        
        return {"messages": [ai_message]}


class GeneralAgent:
    def __init__(self, orchestrator_agent):
        self.orchestrator_agent = orchestrator_agent
        self.llm = self.orchestrator_agent.llm
        self.logger = logging.getLogger(__name__)
        
        # Create a prompt that includes agent scratchpad for tool calling
        self.tool_prompt = ChatPromptTemplate.from_messages([
            ("system", GENERAL_AGENT_PROMPT.replace("Conversation history:\n{messages}\n\nResponse:", "")),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        self.logger.info("GeneralAgent initialized - tools will be created per run with state context.")

    def run(self, state: MainState, config: RunnableConfig):
        messages = state.get("messages", []) if isinstance(state, dict) else state.messages
        
        # Format messages for context and get the latest user input
        messages_text = ""
        user_input = ""
        for message in messages:
            if isinstance(message, HumanMessage):
                messages_text += f"User: {message.content}\n"
                user_input = message.content  # Keep the latest user input for the agent
            elif isinstance(message, AIMessage):
                messages_text += f"Assistant: {message.content}\n"

        try:
            # Get tools with state access for this specific run
            tools = get_all_tools(state)
            
            # Create the tool-calling agent with current state
            agent = create_tool_calling_agent(
                tools=tools,
                llm=self.llm,
                prompt=self.tool_prompt
            )
            
            # Create the agent executor with current tools
            agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=True,
                handle_parsing_errors=True
            )

            # Include conversation context in the input
            full_input = f"Conversation history:\n{messages_text}\nCurrent request: {user_input}"

            response = agent_executor.invoke({
                "input": full_input
            })

            response_content = response.get("output", "I couldn't process your request.")

            ai_message = AIMessage(content=response_content)
            
            return {"messages": [ai_message]}
            
        except Exception as e:
            self.logger.error(f"Error in GeneralAgent: {e}")
            error_message = f"I encountered an error while processing your request: {str(e)}"
            ai_message = AIMessage(content=error_message)
            
            return {"messages": [ai_message]}


class SafetyTerminationAgent:
    def __init__(self, orchestrator_agent):
        self.orchestrator_agent = orchestrator_agent
        self.logger = logging.getLogger(__name__)
        self.logger.info("SafetyTerminationAgent initialized.")

    def run(self, state: MainState, config: RunnableConfig):
        """
        Generate appropriate response for unsafe content and terminate the conversation.
        """
        safety_result = state.get("content_safety_result") if isinstance(state, dict) else state.content_safety_result
        
        if safety_result and not safety_result.is_appropriate:
            # Create concise response with safety details
            safety_response = (
                f"I can't assist with that request due to safety guidelines.\n\n"
                f"Reason: {safety_result.reason}\n"
                f"Category: {safety_result.category}\n\n"
                f"I'm happy to help with programming, general knowledge, legal education, "
                f"calculations, or technical analysis instead!"
            )
        else:
            # Fallback response (shouldn't normally reach here)
            safety_response = (
                "I'm unable to process your request at this time. "
                "Please try rephrasing your question or ask about something else I can help with."
            )
        
        # Create AI message with safety response
        ai_message = AIMessage(content=safety_response)
        
        # Note: LangGraph checkpointer automatically handles message persistence
        
        return {"messages": [ai_message]}