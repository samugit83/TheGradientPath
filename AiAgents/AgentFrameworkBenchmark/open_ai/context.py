"""
Context management for the application.
This provides the main context for dependency injection and state management.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, TYPE_CHECKING

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from code_generator_agents.types import TestResults, CodeOutput


@dataclass
class CodeGenContext:
    """
    Context object for code generation agents.
    This manages the state for the code generation process.
    """
    # User input for code generation
    user_prompt_for_app: Optional[str] = None
    
    # Constraints for code generation
    constraints: Dict[str, str]
    
    # Configuration
    package_name: str 
    max_iterations: int 
    max_tests: int
    
    # Runtime state
    iteration: int = 0
    process_completed: bool = False
    
    # Results from agents - using Any to avoid circular imports
    current_code: Optional[Any] = None  # Actually CodeOutput
    previous_code: Optional[str] = None
    test_results: Optional[Any] = None  # Actually TestResults
    review_notes: List[str] = field(default_factory=list)
    
    def increment_iteration(self):
        """Increment the iteration counter"""
        self.iteration += 1
    
    def update_code(self, code_output: Any):
        """Update the current code and save previous"""
        if self.current_code:
            self.previous_code = self.current_code.code
        self.current_code = code_output
    
    def mark_complete(self):
        """Mark the process as complete"""
        self.process_completed = True


@dataclass
class MainContext:
    """
    Main context object that provides a clean interface for the application.
    This wraps the code generation context and tracks usage metrics.
    """

    # Configuration
    use_persistent_memory: bool = True
    model_name: str = "gpt-4.1"
    enable_content_safety: bool = True
    
    #State variables
    code_gen_context: Optional[CodeGenContext] = None
    
    # Session for persistent memory (shared across all agents)
    session: Optional[Any] = None
    
    # Token usage tracking
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_token_price_per_million: float = 2.5
    output_token_price_per_million: float = 10
    
    
    def update_usage(self, usage_dict: Dict[str, int]):
        """Update token usage from a usage dictionary"""
        self.requests += usage_dict.get("requests", 0)
        self.input_tokens += usage_dict.get("input_tokens", 0)
        self.output_tokens += usage_dict.get("output_tokens", 0)
        self.total_tokens += usage_dict.get("total_tokens", 0)
    
    def calculate_cost(self) -> float:
        """Calculate the total cost based on token usage"""
        input_cost = (self.input_tokens / 1_000_000) * self.input_token_price_per_million
        output_cost = (self.output_tokens / 1_000_000) * self.output_token_price_per_million
        return input_cost + output_cost
    
    def get_usage_summary(self) -> str:
        """Get a formatted summary of token usage and cost"""
        total_cost = self.calculate_cost()
        return (
            f"📊 Token Usage:\n"
            f"  • Requests: {self.requests}\n"
            f"  • Input tokens: {self.input_tokens:,}\n"
            f"  • Output tokens: {self.output_tokens:,}\n"
            f"  • Total tokens: {self.total_tokens:,}\n"
            f"  • Total cost: ${total_cost:.4f}"
        )

def extract_usage_from_result(result: Any) -> Dict[str, int]:
    """
    Extract token usage from an agents.Runner result.
    Based on the pattern from helpers.py
    """
    usage = {"requests": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    
    raw = getattr(result, "raw_responses", None) or []
    for r in raw:
        r_usage = getattr(r, "usage", None)
        if r_usage is None:
            r_usage = getattr(getattr(r, "response", None), "usage", None)
        
        if not r_usage:
            continue
        
        # Extract tokens with fallback names
        in_toks = getattr(r_usage, "input_tokens", getattr(r_usage, "prompt_tokens", 0))
        out_toks = getattr(r_usage, "output_tokens", getattr(r_usage, "completion_tokens", 0))
        tot_toks = getattr(r_usage, "total_tokens", in_toks + out_toks)
        
        usage["requests"] += 1
        usage["input_tokens"] += in_toks
        usage["output_tokens"] += out_toks
        usage["total_tokens"] += tot_toks
    
    return usage