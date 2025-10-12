"""
State management for the vanilla code generation application.
This provides the main context for dependency injection and state management.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)

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
class ReviewResult:
    """Result from the reviewer agent"""
    review_notes: List[str] = field(default_factory=list)
    analysis: str = ""
    recommendation: str = ""
    should_continue: bool = True

    
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


class State(BaseModel): 
    """
    Main state object that provides a clean interface for the application.
    This wraps the code generation state and tracks usage metrics.
    This is the main state, similar to MainContext in open_ai but named State for backward compatibility.
    """
    user_prompt: str = ""

    code_gen_state: Optional[CodeGenState] = None

    # Configuration
    use_persistent_memory: bool = True
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
    
    # Pydantic configuration
    model_config = {"arbitrary_types_allowed": True}
    
    def model_post_init(self, __context: Any) -> None:
        """Initialize code_gen_state with default values if not provided"""
        if self.code_gen_state is None:
            self.code_gen_state = CodeGenState()


# Utility functions moved outside the State class to fix AutoGen serialization
def update_state_usage(state: State, usage_dict: Dict[str, int]) -> None:
    """Update token usage from a usage dictionary"""
    state.requests += usage_dict.get("requests", 0)
    state.input_tokens += usage_dict.get("input_tokens", 0)
    state.output_tokens += usage_dict.get("output_tokens", 0)
    state.total_tokens += usage_dict.get("total_tokens", 0)

def calculate_state_cost(state: State) -> float:
    """Calculate the total cost based on token usage"""
    input_cost = (state.input_tokens / 1_000_000) * state.input_token_price_per_million
    output_cost = (state.output_tokens / 1_000_000) * state.output_token_price_per_million
    return input_cost + output_cost

def get_state_usage_summary(state: State) -> str:
    """Get a formatted summary of token usage and cost"""
    total_cost = calculate_state_cost(state)
    return (
        f"📊 Token Usage:\n"
        f"  • Requests: {state.requests}\n"
        f"  • Input tokens: {state.input_tokens:,}\n"
        f"  • Output tokens: {state.output_tokens:,}\n"
        f"  • Total tokens: {state.total_tokens:,}\n"
        f"  • Total cost: ${total_cost:.4f}"
    )


def extract_usage_from_autogen_result(create_result) -> Dict[str, int]:
    """
    Extract token usage from AutoGen's CreateResult.
    This works with the result from model_client.create().
    """
    usage = {"requests": 1, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    
    # Extract usage information from the result
    if hasattr(create_result, 'usage'):
        usage_info = create_result.usage
        if hasattr(usage_info, 'prompt_tokens'):
            usage["input_tokens"] = usage_info.prompt_tokens
        if hasattr(usage_info, 'completion_tokens'):
            usage["output_tokens"] = usage_info.completion_tokens  
        if hasattr(usage_info, 'total_tokens'):
            usage["total_tokens"] = usage_info.total_tokens
        else:
            usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    
    return usage


def track_usage_from_result(create_result, state):
    """
    Track usage from AutoGen's CreateResult and update state.
    This is similar to vanilla's pattern of extracting usage from each response.
    
    Args:
        create_result: Result from model_client.create()
        state: State object to update
    """
    usage = extract_usage_from_autogen_result(create_result)
    
    # Update state with usage if there was any
    if usage["requests"] > 0 and usage["total_tokens"] > 0:
        update_state_usage(state, usage)
