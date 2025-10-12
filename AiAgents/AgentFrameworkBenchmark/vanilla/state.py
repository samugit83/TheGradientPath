"""
State management for the vanilla code generation application.
This provides the main context for dependency injection and state management.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

# Data classes for code generation process
@dataclass
class TestFailure:
    """Represents a single test failure"""
    name: str
    trace: str

@dataclass
class TestResults:
    """Results from running tests"""
    passed: int = 0
    failed: int = 0
    failures: List[TestFailure] = field(default_factory=list)
    test_explanation: str = ""

@dataclass
class CodeResult:
    """Results from code generation"""
    code: str = ""
    code_explanation: str = ""


@dataclass
class CodeGenState:
    """
    State object for code generation agents.
    This manages the state for the code generation process.
    """
    # Constraints (no user_prompt here - it stays at root level)
    constraints: Dict[str, str] = field(default_factory=dict)
    
    # Configuration
    package_name: str = "app"
    max_iterations: int = 3
    max_tests: int = 8
    
    # Runtime state
    iteration: int = 0
    process_completed: bool = False
    
    # Results from agents
    code_result: Optional[CodeResult] = None
    previous_code: str = ""
    test_results: Optional[TestResults] = None
    review_notes: List[str] = field(default_factory=list)
    
    def increment_iteration(self):
        """Increment the iteration counter"""
        self.iteration += 1
    
    def update_code(self, code_result: CodeResult):
        """Update the current code and save previous"""
        if self.code_result:
            self.previous_code = self.code_result.code
        self.code_result = code_result
    
    def mark_complete(self):
        """Mark the process as complete"""
        self.process_completed = True


@dataclass
class State: 
    """
    Main state object that provides a clean interface for the application.
    This wraps the code generation state and tracks usage metrics.
    This is the main state, similar to MainContext in open_ai but named State for backward compatibility.
    """
    user_prompt: str

    code_gen_state: Optional[CodeGenState] = None

    # Configuration
    use_persistent_memory: bool = False
    model_name: str = "gpt-4.1"
    enable_content_safety: bool = True
    
    # Token usage tracking (new)
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_token_price_per_million: float = 2.5
    output_token_price_per_million: float = 10
    
    session_id: Optional[str] = None
    
    def __post_init__(self):
        """Initialize code_gen_state with default values if not provided"""
        if self.code_gen_state is None:
            self.code_gen_state = CodeGenState()
    
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


def extract_usage_from_response(response: Any) -> Dict[str, int]:
    """
    Extract token usage from an OpenAI API response.
    This works with the vanilla OpenAI library response format.
    """
    usage = {"requests": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    
    # Check if response has usage attribute
    if hasattr(response, 'usage') and response.usage:
        usage["requests"] = 1
        usage["input_tokens"] = getattr(response.usage, 'prompt_tokens', 0)
        usage["output_tokens"] = getattr(response.usage, 'completion_tokens', 0)
        usage["total_tokens"] = getattr(response.usage, 'total_tokens', 0)
    
    return usage
