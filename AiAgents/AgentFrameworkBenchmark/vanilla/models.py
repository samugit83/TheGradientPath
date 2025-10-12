import logging
import os
from typing import Optional, Dict, Any, Tuple
from openai import OpenAI
from state import extract_usage_from_response

logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4.1"):
        """Initialize OpenAI client"""
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY")
        )
        self.model = model
        # Track cumulative usage for this client instance
        self.total_usage = {"requests": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    
    def api_call(self, prompt: str) -> str:
        """Make an API call and return the raw response string"""
        content, usage = self.api_call_with_usage(prompt)
        # Log that a call was made even when using the simple api_call method
        if usage['requests'] > 0:
            logger.debug(f"API call made via api_call() - {usage['input_tokens']} in, {usage['output_tokens']} out")
        return content
    
    def api_call_with_usage(self, prompt: str) -> Tuple[str, Dict[str, int]]:
        """
        Make an API call and return both the response and usage information.
        
        Returns:
            Tuple of (response_content, usage_dict)
        """
        try:
            params = {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
            
            response = self.client.chat.completions.create(**params)
            
            # Extract usage information
            usage = extract_usage_from_response(response)
            
            # Update cumulative usage
            self.total_usage["requests"] += usage["requests"]
            self.total_usage["input_tokens"] += usage["input_tokens"]
            self.total_usage["output_tokens"] += usage["output_tokens"]
            self.total_usage["total_tokens"] += usage["total_tokens"]
            
            # Log usage for this call
            if usage["requests"] > 0:
                call_num = self.total_usage["requests"]  # This is the call number AFTER updating
                logger.info(
                    f"LLM API call #{call_num} - Input: {usage['input_tokens']}, "
                    f"Output: {usage['output_tokens']}, Total: {usage['total_tokens']} tokens"
                )
            
            # Return the raw message content and usage
            return response.choices[0].message.content, usage
            
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            # Return empty string and zero usage on error
            return "", {"requests": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    
    def get_total_usage(self) -> Dict[str, int]:
        """Get the cumulative usage for this client instance"""
        return self.total_usage.copy()