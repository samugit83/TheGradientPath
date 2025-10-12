"""
Helper classes for the AI-powered chat system
"""

import os
import sys
import json
import re
import logging
from typing import Dict, List, Any, Optional
from tools import ToolRegistry
from models import LLMClient
from state import State
from prompts import TOOL_SELECTOR_PROMPT_TEMPLATE, TOOL_RESULT_INTERPRETER_PROMPT_TEMPLATE, CONTENT_SAFETY_PROMPT, AGENT_ROUTING_PROMPT, LEGAL_EXPERT_PROMPT
from session_manager import ( init_session_manager )

logger = logging.getLogger(__name__)



class LLMToolSelector:
    """LLM-based tool selector that analyzes messages and selects appropriate tools"""
    
    def __init__(self, llm_client, tool_registry=None):
        self.llm_client = llm_client
        self.tool_registry = tool_registry or ToolRegistry()
        
    def create_selection_prompt(self, message: str, history_text: str) -> str:
        """Create a prompt for the LLM to select the appropriate tool"""
        
        # Get detailed tool metadata
        tools_metadata = self.tool_registry.get_tools_metadata()
        
        # Build tools text with descriptions and parameter schemas
        tools_text = ""
        for tool in tools_metadata:
            tools_text += f"\n### Tool: {tool['name']}\n"
            tools_text += f"**Description:** {tool['description']}\n"
            
            # Add parameter schema
            if tool['parameters']:
                tools_text += "**Parameters:**\n"
                for param_name, param_info in tool['parameters'].items():
                    required = "required" if param_info.get('required', False) else "optional"
                    tools_text += f"  - `{param_name}` ({param_info['type']}, {required}): {param_info['description']}\n"
            else:
                tools_text += "**Parameters:** None\n"
        
        # Build examples text from tool examples
        examples_text = ""
        example_num = 1
        
        # Add a conversational example
        examples_text += f"\n{example_num}. User: \"Thanks for your help!\"\n"
        params_json = json.dumps({
            "tool": "none",
            "reasoning": "User is thanking, no tool needed",
            "conversational_response": "You're welcome! I'm here to help with calculations or code generation whenever you need."
        }, indent=3)
        examples_text += f"   Response: {params_json}\n"
        example_num += 1
        
        # Add an example with missing parameters
        examples_text += f"\n{example_num}. User: \"I need to add 5 to something\"\n"
        params_json = json.dumps({
            "tool": "sum",
            "parameters": {"a": 5},
            "reasoning": "User wants to add but only provided one number",
            "ask_parameter_message": "I have 5 as the first number. What's the second number you'd like to add to it?"
        }, indent=3)
        examples_text += f"   Response: {params_json}\n"
        example_num += 1
        
        # Add an example with type validation error
        examples_text += f"\n{example_num}. User: \"Add five plus ten\"\n"
        params_json = json.dumps({
            "tool": "sum",
            "parameters": {},
            "reasoning": "User wants to add but provided text instead of numbers",
            "ask_parameter_message": "I'd be happy to add those for you! Could you provide the numeric values? What numbers would you like to add?"
        }, indent=3)
        examples_text += f"   Response: {params_json}\n"
        example_num += 1
        
        # Add tool execution examples
        for tool in tools_metadata:
            if tool.get('examples'):
                example = tool['examples'][0] if tool['examples'] else None
                if example:
                    examples_text += f"\n{example_num}. User: \"{example['input']}\"\n"
                    params_json = json.dumps({
                        "tool": tool['name'],
                        "parameters": example['parameters'],
                        "reasoning": f"User wants to {tool['description'].lower()[:50]}..."
                    }, indent=3)
                    examples_text += f"   Response: {params_json}\n"
                    example_num += 1
        
        # Use the template from prompts.py
        prompt = TOOL_SELECTOR_PROMPT_TEMPLATE.format(
            history_text=history_text,
            message=message,
            tools_text=tools_text,
            examples_text=examples_text if examples_text else "No examples available. Use the tool descriptions and parameter schemas to guide your selection."
        )
        
        return prompt
    
    def select_tool(self, message: str, history_text: str) -> Dict[str, Any]:
        """Use LLM to select the appropriate tool for the message"""
        
        prompt = self.create_selection_prompt(message, history_text)
        
        try:
            # Call LLM for tool selection
            logger.info("Requesting tool selection from LLM...")
            llm_response, usage = self.llm_client.api_call_with_usage(prompt)
            
            # Parse the JSON response
            if llm_response:
                # Try to extract JSON from the response
                json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
                if json_match:
                    selection = json.loads(json_match.group())
                    logger.info(f"Selected tool: {selection.get('tool', 'unknown')}")
                    # Include usage in the response
                    selection['_usage'] = usage
                    return selection
                else:
                    logger.error("No valid JSON found in LLM response")
                    return {"error": "Invalid response format from LLM", "_usage": usage}
            else:
                return {"error": "No response from LLM", "_usage": usage}
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            return {"error": f"JSON parsing error: {str(e)}"}
        except Exception as e:
            logger.error(f"Tool selection failed: {e}")
            return {"error": f"Tool selection error: {str(e)}"}


class ChatLoop:
    """Interactive chat system with LLM-based tool selection"""
    
    def __init__(self, session_id: str = None):
        self.llm_client = None
        self.tool_selector = None
        self.tool_registry = ToolRegistry()
        # Initialize State to track usage across the session
        self.state = State(user_prompt="")
        
        # Use provided session_id or generate one
        if session_id:
            self.state.session_id = session_id
        else:
            import uuid
            self.state.session_id = f"session_{uuid.uuid4().hex[:8]}"
        
        # Initialize unified session manager
        self.session_manager = init_session_manager(
            session_id=self.state.session_id,
            use_persistent=self.state.use_persistent_memory,
            db_path="conversations.db",
            max_history=40
        )
        logger.info(f"Initialized session manager: {self.state.session_id} "
                   f"(mode: {'persistent' if self.state.use_persistent_memory else 'in-memory'})")
        
        # Initialize content safety guardrail if enabled
        self.content_safety_enabled = getattr(self.state, 'enable_content_safety', True)
        self.safety_model = "gpt-4o-mini"  # Use fast, cheap model for safety checks
        
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the LLM client and tool selector"""
        try:
            self.llm_client = LLMClient()
            self.tool_selector = LLMToolSelector(self.llm_client, self.tool_registry)  # Pass shared registry
            logger.info("LLM client and tool selector initialized")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            logger.warning(f"⚠️ Warning: LLM initialization failed: {e}")
    
    def _format_conversation_history(self, max_messages: int = 15, max_content_length: int = 300) -> str:
        """
        Reconstruct conversation history from session manager as a formatted string.
        
        Args:
            max_messages: Maximum number of messages to include
            max_content_length: Maximum length of content per message
            
        Returns:
            Formatted conversation history string
        """
        logger.debug("Reconstructing conversation history...")
        history = self.session_manager.get_conversation_history(last_n=20)
        logger.debug(f"Retrieved {len(history)} messages for history reconstruction")
        
        if not history:
            return ""
        
        history_text = "\n## Recent Conversation History:\n"
        for item in history[-max_messages:]:  # Use last N messages
            subject = item.get('subject', '')
            content = item.get('content', '')[:max_content_length]  # Limit content length
            if subject == 'user':
                history_text += f"User: {content}...\n"
            elif subject == 'assistant':
                history_text += f"Assistant: {content}...\n"
        
        return history_text
    
    def generate_natural_response(self, tool_name: str, tool_result: Dict[str, Any], user_input: str) -> str:
        """
        Generate a natural language response from tool execution results
        
        Args:
            tool_name: Name of the executed tool
            tool_result: The result from tool execution
            user_input: The original user request
            
        Returns:
            Natural language interpretation of the tool result
        """
        try:
            # Get tool description
            tool = self.tool_registry.get_tool(tool_name)
            tool_description = tool.description if tool else "Tool for processing requests"
            
            # Format the tool result for the prompt
            if tool_result.get("success"):
                result_text = tool_result.get("result", tool_result.get("explanation", "Operation completed"))
            else:
                result_text = f"Error: {tool_result.get('error', 'Unknown error')}"
            
            # Create the prompt for natural language generation
            prompt = TOOL_RESULT_INTERPRETER_PROMPT_TEMPLATE.format(
                tool_name=tool_name,
                tool_description=tool_description,
                user_request=user_input,
                tool_result=result_text
            )
            
            # Get natural language response from LLM
            logger.info("Generating natural language response from tool result...")
            natural_response, usage = self.llm_client.api_call_with_usage(prompt)
            self.state.update_usage(usage)
            
            if natural_response:
                return natural_response
            else:
                # Fallback to the original explanation if LLM fails
                return tool_result.get("explanation", result_text)
                
        except Exception as e:
            logger.error(f"Failed to generate natural response: {e}")
            # Fallback to the original tool result
            if tool_result.get("success"):
                return tool_result.get("explanation", tool_result.get("result", "Operation completed"))
            else:
                return f"Error: {tool_result.get('error', 'Unknown error')}"
    
    
    
    def _handle_special_commands(self, user_input: str) -> Optional[str]:
        """Handle special commands like exit, clear, etc.
        
        Returns:
            Response string if command was handled, None otherwise
        """
        if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
            logger.info("\n👋 Goodbye! Thanks for using the AI-powered chat.")
            # Display final usage stats
            if self.state.requests > 0:
                print(f"\n{self.state.get_usage_summary()}")
            self.tool_registry.cleanup()  # Clean up MCP servers
            sys.exit(0)
        
        if user_input.lower() in ['clear', 'clear history']:
            self.session_manager.clear_session()
            return "✨ Conversation history cleared!"
        
        return None
    
    
    def _perform_tool_selection(self, user_input: str) -> Dict[str, Any]:
        """Perform tool selection and handle parameter requests
        
        Args:
            user_input: The user's input message
            
        Returns:
            Dictionary with tool selection results or response to return
        """
        logger.info("\n🤔 Analyzing your request...")
        
        # Get formatted conversation history
        history_text = self._format_conversation_history(max_messages=6, max_content_length=100)
        
        selection = self.tool_selector.select_tool(user_input, history_text)
        
        # Extract and track usage from tool selection
        usage = selection.pop('_usage', None)
        if usage:
            self.state.update_usage(usage)
        
        if "error" in selection:
            logger.error(f"Tool selection error: {selection['error']}")
            return {
                "type": "error",
                "response": f"❌ Sorry, I couldn't understand your request: {selection['error']}"
            }
        
        tool_name = selection.get("tool", "").strip()
        parameters = selection.get("parameters", {})
        reasoning = selection.get("reasoning", "")
        ask_parameter_message = selection.get("ask_parameter_message", "")
        conversational_response = selection.get("conversational_response", "")
        
        logger.info(f"Selected tool: {tool_name}")
        logger.info(f"Tool parameters: {parameters}")

        if reasoning:
            logger.info(f"💭 {reasoning}")
        
        # Handle conversational response (no tool needed)
        if not tool_name or tool_name.lower() in ["none", "null", "", "n/a", "na"]:
            response = conversational_response or "I'm here to help! You can ask me to perform calculations or generate code."
            logger.info(f"\n💬 Assistant > {response}")
            
            self.session_manager.add_assistant_message(response, task_type="conversation")
            return {
                "type": "conversational",
                "response": response
            }
        
        logger.info(f"🔧 Selected tool: {tool_name}")
        
        # Check if we need to ask for parameters
        if ask_parameter_message:
            logger.info(f"\n💬 Assistant > {ask_parameter_message}")
            
            self.session_manager.add_assistant_message(ask_parameter_message, task_type="parameter_request")
            return {
                "type": "parameter_request",
                "response": ask_parameter_message
            }
        
        # Return tool execution details
        return {
            "type": "tool_execution",
            "tool_name": tool_name,
            "parameters": parameters
        }
    
    
    def _execute_tool(self, tool_name: str, parameters: Dict[str, Any], user_input: str) -> str:
        """Execute a tool and generate appropriate response
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Tool parameters
            user_input: Original user input
            
        Returns:
            Response string from tool execution
        """
        logger.info(f"🚀 Executing {tool_name}...")
        
        # Generate tool call ID for tracking
        import uuid
        tool_id = f"tool_{uuid.uuid4().hex[:8]}"
        
        result = self.tool_registry.execute_tool(tool_name, **parameters)
        
        # Log tool call to session
        self.session_manager.add_tool_message(
            tool_name=tool_name,
            tool_id=tool_id,
            arguments=parameters,
            result=str(result.get("result", "")) if result.get("success") else None,
            error=result.get("error") if not result.get("success") else None
        )
        
        # Extract and track usage from tool execution (e.g., from code_generator)
        tool_usage = result.pop('_usage', None)
        if tool_usage:
            self.state.update_usage(tool_usage)
            logger.info(f"Tool {tool_name} used {tool_usage['requests']} LLM requests")
        
        # Check if this is a tool that returns structured data (like weather tools)
        # and generate a natural language response
        tool_needs_interpretation = (
            "weather" in tool_name.lower() or 
            (result.get("success") and isinstance(result.get("result"), str) and 
             ("°C" in str(result.get("result", "")) or "°F" in str(result.get("result", ""))))
        )
        
        if result.get("success"):
            if tool_needs_interpretation:
                # Generate natural language response for structured data
                response = self.generate_natural_response(tool_name, result, user_input)
            else:
                # Use the original explanation for simple tools
                response = result.get("explanation", result.get("result", "Operation completed"))
            logger.info(f"\n💬 Assistant > {response}")
        else:
            response = f"Error: {result.get('error', 'Unknown error')}"
            logger.error(f"\n❌ {response}")
        
        # Store response (token usage is tracked in state.py)
        self.session_manager.add_assistant_message(response, task_type="tool_execution")
        
        return response
    
    def _check_content_safety(self, user_input: str) -> tuple[bool, str]:
        """
        Check if user input violates content safety policies.
        
        Returns:
            Tuple of (is_safe, violation_message)
            If is_safe is False, violation_message contains the error to show user
        """
        if not self.content_safety_enabled:
            return True, ""
        
        try:
            # Create a dedicated LLM client for safety checks
            safety_client = LLMClient(model=self.safety_model)
            
            # Build the safety check prompt
            safety_prompt = f"{CONTENT_SAFETY_PROMPT}\n\nUser Input to Analyze:\n{user_input}\n\nProvide your analysis as a JSON response:"
            
            # Get response and track usage
            response, usage = safety_client.api_call_with_usage(safety_prompt)
            
            # Update usage statistics
            if usage["requests"] > 0:
                self.state.update_usage(usage)
            
            try:
                cleaned_response = response.strip()
                if cleaned_response.startswith("```json"):
                    cleaned_response = cleaned_response[7:] 
                if cleaned_response.startswith("```"):
                    cleaned_response = cleaned_response[3:] 
                if cleaned_response.endswith("```"):
                    cleaned_response = cleaned_response[:-3]  
                cleaned_response = cleaned_response.strip()
                
                safety_result = json.loads(cleaned_response)
                
                is_inappropriate = safety_result.get("is_inappropriate", False)
                reason = safety_result.get("reason", "Unknown reason")
                category = safety_result.get("category", "unknown")
                
                if is_inappropriate:
                    logger.warning(f"Content safety violation: {category} - {reason}")
                    violation_message = (
                        "\n⚠️  Your request was blocked for safety reasons.\n"
                        f"Detected issue: {reason}\n\n"
                        "The AI assistant cannot help with:\n"
                        "  • Illegal activities or harmful content\n"
                        "  • Violence, self-harm, or threats\n"
                        "  • Hate speech or discrimination\n"
                        "  • Inappropriate sexual content\n"
                        "  • Privacy violations or doxxing\n"
                        "  • Creating misinformation\n"
                        "  • Academic dishonesty\n"
                        "\nPlease rephrase your request or ask something else."
                    )
                    return False, violation_message
                
                return True, ""
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Failed to parse safety check response: {e}")
                logger.debug(f"Raw response was: {response[:500]}")  # Log first 500 chars for debugging
                # On parse error, allow with caution but log the issue
                return True, ""
                
        except Exception as e:
            logger.error(f"Error in content safety check: {e}", exc_info=True)
            # On error, allow the request to proceed
            return True, ""
    
    def _route_to_agent(self, user_input: str) -> str:
        """
        Route user request to appropriate agent (legal_expert or general_agent).
        
        Args:
            user_input: The user's input message
            
        Returns:
            Either "legal_expert" or "general_agent" 
        """
        try:
            # Get formatted conversation history
            history_text = self._format_conversation_history(max_messages=10, max_content_length=200)
            
            # Create routing prompt using template
            routing_prompt = AGENT_ROUTING_PROMPT.format(
                history_text=history_text,
                user_input=user_input
            )
            
            # Get routing decision from LLM
            logger.info("🤔 Determining which agent should handle this request...")
            response, usage = self.llm_client.api_call_with_usage(routing_prompt)
            
            # Update usage statistics
            if usage["requests"] > 0:
                self.state.update_usage(usage)
            
            # Clean and validate the response
            agent_choice = response.strip().lower()
            if "legal_expert" in agent_choice:
                logger.info("⚖️ Routing to Legal Expert")
                return "legal_expert"
            elif "general_agent" in agent_choice:
                logger.info("🔧 Routing to General Agent") 
                return "general_agent"
            else:
                # Default to general_agent if unclear
                logger.warning(f"Unclear routing response: {response}, defaulting to general_agent")
                return "general_agent"
                
        except Exception as e:
            logger.error(f"Error in agent routing: {e}")
            # Default to general_agent on error
            return "general_agent"
    
    def _handle_legal_expert(self, user_input: str) -> str:
        """
        Handle requests that should be processed by the legal expert.
        
        Args:
            user_input: The user's legal question
            
        Returns:
            Legal expert response
        """
        try:
            # Get formatted conversation history
            history_text = self._format_conversation_history(max_messages=15, max_content_length=300)
            
            # Create legal expert prompt using template
            legal_prompt = LEGAL_EXPERT_PROMPT.format(
                history_text=history_text,
                user_input=user_input
            )
            
            # Get legal expert response
            logger.info("⚖️ Processing with Legal Expert...")
            response, usage = self.llm_client.api_call_with_usage(legal_prompt)
            
            # Update usage statistics
            if usage["requests"] > 0:
                self.state.update_usage(usage)
            
            logger.info(f"\n💬 Legal Expert > {response}")
            
            # Store the conversation
            self.session_manager.add_assistant_message(response, task_type="legal_expert")
            
            return response
            
        except Exception as e:
            logger.error(f"Error with legal expert: {e}")
            return f"❌ Sorry, I encountered an error while processing your legal question: {str(e)}"
    
    def _handle_general_agent(self, user_input: str) -> str:
        """
        Handle requests that should be processed by the general agent using tool selection.
        
        Args:
            user_input: The user's input message
            
        Returns:
            General agent response
        """

        selection_result = self._perform_tool_selection(user_input)

        if selection_result["type"] in ["error", "conversational", "parameter_request"]:
            return selection_result["response"]
        
        if selection_result["type"] == "tool_execution":
            return self._execute_tool(
                selection_result["tool_name"],
                selection_result["parameters"],
                user_input
            )
        
        return "I'm here to help! You can ask me to perform calculations or generate code."
    
    def process_message(self, user_input: str) -> str:
        """Process user message using LLM tool selection
        
        This is the main entry point for processing user messages.
        It orchestrates the flow through special commands, tool selection,
        and tool execution.
        """
        # Check content safety first (except for special commands)
        if user_input.lower() not in ['help', 'exit', 'quit', 'clear']:
            is_safe, violation_message = self._check_content_safety(user_input)
            if not is_safe:
                return violation_message
        
        # Log user message to persistent session
        self.session_manager.add_user_message(user_input)
        
        # Check if LLM client is initialized
        if not self.llm_client:
            return "❌ LLM client not initialized. Please check your API key."
        
        # Handle special commands
        special_response = self._handle_special_commands(user_input)
        if special_response:
            return special_response
        
        # Route to appropriate agent first
        agent_choice = self._route_to_agent(user_input)
        
        # If routed to legal expert, handle directly
        if agent_choice == "legal_expert":
            return self._handle_legal_expert(user_input)

        return self._handle_general_agent(user_input)
    

    def run(self):
        """Main REPL loop"""
        print("\n" + "=" * 60)
        print("  🤖 Welcome to AI-Powered Chat with Tool Selection! 🚀")
        print("=" * 60)
        
        # Show session status
        if self.state.use_persistent_memory:
            print(f"\n🧠 Persistent memory enabled - Session ID: {self.state.session_id}")
            print("   Conversations will be saved to: conversations.db")
            stats = self.session_manager.get_session_stats()
            if stats['total_messages'] > 0:
                print(f"   Resuming session with {stats['total_messages']} previous messages")
        else:
            print("\n🧠 In-memory mode - conversations will not be saved")
        
        # Show available tools
        tools = self.tool_registry.list_tools()
        if tools:
            print("\n📦 Available tools:")
            for tool in tools:
                print(f"  • {tool['name']}: {tool['description']}")
        
        # Show MCP tools specifically
        mcp_tools = self.tool_registry.get_mcp_tools()
        if mcp_tools:
            print(f"\n🌐 MCP Weather tools loaded: {', '.join(mcp_tools)}")
        
        # Show content safety status
        print("\n🛡️  Content Safety:")
        if self.content_safety_enabled:
            print("  • Content safety guardrails are ENABLED")
            print("  • Requests for illegal, harmful, or inappropriate content will be blocked")
        else:
            print("  • Content safety guardrails are DISABLED")
            print("  • ⚠️  Operating without content safety checks")
        
        print("\n💬 Smart Routing:")
        print("  • I'll intelligently route your request to the right specialist:")
        print("    - ⚖️ Legal Expert: For law-related questions and legal topics") 
        print("    - 🔧 General Agent: For everything else (programming, tools, general knowledge)")
        print("\nJust type naturally and I'll understand what you need!")
        print("Type 'help' for more info or 'exit' to quit.\n")
        
        if not os.environ.get("OPENAI_API_KEY"):
            print("⚠️  Warning: OPENAI_API_KEY not set!")
            print("Set it with: export OPENAI_API_KEY='your-key-here'")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                sys.exit(1)
        
        while True:
            try:
                print("\n" + "-" * 40)
                user_input = input("💬 You > ").strip()
                
                if not user_input:
                    continue
                
                # Track the state BEFORE this turn
                prev_requests = self.state.requests
                prev_input = self.state.input_tokens
                prev_output = self.state.output_tokens
                prev_total = self.state.total_tokens
                
                response = self.process_message(user_input)
                
                # Calculate and display turn usage if there was LLM activity
                turn_requests = self.state.requests - prev_requests
                turn_input = self.state.input_tokens - prev_input
                turn_output = self.state.output_tokens - prev_output
                turn_total = self.state.total_tokens - prev_total
                
                if turn_requests > 0:
                    # Calculate turn cost
                    turn_input_cost = (turn_input / 1_000_000) * self.state.input_token_price_per_million
                    turn_output_cost = (turn_output / 1_000_000) * self.state.output_token_price_per_million
                    turn_cost = turn_input_cost + turn_output_cost
                    
                    # Calculate cumulative cost
                    total_cost = self.state.calculate_cost()
                    logger.info(f"\n📊 This turn: {turn_input:,} in, {turn_output:,} out, {turn_total:,} total tokens ({turn_requests} requests) - Cost: ${turn_cost:.4f}")
                    logger.info(f"\n📊 Cumulative: {self.state.input_tokens:,} in, {self.state.output_tokens:,} out, {self.state.total_tokens:,} total tokens ({self.state.requests} requests) - Total cost: ${total_cost:.4f}")
                    
            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupted! Type 'exit' to quit or continue chatting.")
                # Display final usage stats on interrupt
                if self.state.requests > 0:
                    print(f"\n{self.state.get_usage_summary()}")
            except Exception as e:
                print(f"\n❌ Error: {e}")
                logger.error(f"REPL error: {e}", exc_info=True)
