"""
State management for the vanilla code generation application.
This provides the main context for dependency injection and state management.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

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

    user_prompt_for_app: Optional[str] = None

    constraints: Dict[str, str] = field(default_factory=lambda: {
        "language": "Python 3.8+",
        "style": "PEP 8 compliant",
        "documentation": "Add docstrings",
        "allowed_packages": "stdlib"
    })
    
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


@dataclass
class State: 
    """
    Main state object that provides a clean interface for the application.
    This wraps the code generation state and tracks usage metrics.
    This is the main state, similar to MainContext in open_ai but named State for backward compatibility.
    """

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
    


# =============================================================================
# Token counting functions
# =============================================================================

def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Count tokens in text using tiktoken.
    
    Args:
        text: Text to count tokens for
        model: Model name for tokenizer (default: gpt-4)
        
    Returns:
        Number of tokens in the text
    """
    if not TIKTOKEN_AVAILABLE:
        # Fallback estimation if tiktoken is not available
        return max(1, len(text) // 4)
    
    try:
        encoder = tiktoken.encoding_for_model(model)
        tokens = encoder.encode(text)
        return len(tokens)
    except Exception as e:
        # Fallback estimation if tiktoken fails
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to count tokens with tiktoken: {e}")
        return max(1, len(text) // 4)


# =============================================================================
# Usage tracking functions (standalone)
# =============================================================================

def update_usage(state: State, usage_dict: Dict[str, int]) -> None:
    """Update token usage from a usage dictionary"""
    state.requests += usage_dict.get("requests", 0)
    state.input_tokens += usage_dict.get("input_tokens", 0)
    state.output_tokens += usage_dict.get("output_tokens", 0)
    state.total_tokens += usage_dict.get("total_tokens", 0)


def calculate_cost(state: State) -> float:
    """Calculate the total cost based on token usage"""
    input_cost = (state.input_tokens / 1_000_000) * state.input_token_price_per_million
    output_cost = (state.output_tokens / 1_000_000) * state.output_token_price_per_million
    return input_cost + output_cost


def get_usage_summary(state: State) -> str:
    """Get a formatted summary of token usage and cost"""
    total_cost = calculate_cost(state)
    return (
        f"📊 Token Usage:\n"
        f"  • Requests: {state.requests}\n"
        f"  • Input tokens: {state.input_tokens:,}\n"
        f"  • Output tokens: {state.output_tokens:,}\n"
        f"  • Total tokens: {state.total_tokens:,}\n"
        f"  • Total cost: ${total_cost:.4f}"
    )


def calculate_usage_from_content(input_text: str, output_text: str, model: str = "gpt-4") -> Dict[str, int]:
    """
    Calculate token usage by counting tokens in input and output text.
    This is used when Semantic Kernel doesn't provide usage metadata (e.g., with streaming).
    
    Args:
        input_text: The input text sent to the model
        output_text: The output text received from the model
        model: Model name for accurate token counting
        
    Returns:
        Usage dictionary with token counts
    """
    input_tokens = count_tokens(input_text, model)
    output_tokens = count_tokens(output_text, model)
    total_tokens = input_tokens + output_tokens
    
    return {
        "requests": 1,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens
    }
