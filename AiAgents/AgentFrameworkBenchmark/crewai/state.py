"""
State management for the vanilla code generation application.
This provides the main context for dependency injection and state management.
Includes global state management functionality.
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

    constraints: Dict[str, str] = field(default_factory=lambda: {
        "language": "Python 3.8+",
        "style": "PEP 8 compliant",
        "documentation": "Add docstrings",
        "allowed_packages": "stdlib"
    })

    user_prompt_for_app: Optional[str] = None
    
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
    
    def __post_init__(self):
        """Initialize code_gen_state with default values if not provided"""
        if self.code_gen_state is None:
            self.code_gen_state = CodeGenState()
    





# Global state management
_global_state: Optional[State] = None


def initialize_global_state(session_id: str) -> State:
    """
    Initialize the global state with a new State instance.
        
    Returns:
        The initialized State instance
    """
    global _global_state
    _global_state = State(session_id=session_id)
    return _global_state


def get_global_state() -> Optional[State]:
    """
    Get the current global state instance.
    
    Returns:
        The current global State instance, or None if not initialized
    """
    return _global_state


def update_user_prompt_for_app(user_prompt: str) -> None:
    """
    Update the user_prompt_for_app in the global state.
    
    Args:
        user_prompt: The user prompt to set for app generation
    """
    global _global_state
    if _global_state and _global_state.code_gen_state:
        _global_state.code_gen_state.user_prompt_for_app = user_prompt


def clear_global_state(session_id: str = None) -> State:
    """
    Clear the global state and reinitialize it.
        
    Args:
        session_id: Optional session ID to reinitialize with
        
    Returns:
        The new initialized State instance
    """
    global _global_state
    _global_state = State(session_id=session_id)
    return _global_state


def is_global_state_initialized() -> bool:
    """
    Check if the global state has been initialized.
    
    Returns:
        True if global state is initialized, False otherwise
    """
    return _global_state is not None


# Usage tracking utility functions
def update_usage_from_crewai(state: State, usage_metrics):
    """Update token usage from CrewAI UsageMetrics object or dictionary"""
    if usage_metrics:
        # Handle both dictionary and object formats
        if isinstance(usage_metrics, dict):
            state.requests += usage_metrics.get('successful_requests', 0)
            state.input_tokens += usage_metrics.get('prompt_tokens', 0)
            state.output_tokens += usage_metrics.get('completion_tokens', 0)
            state.total_tokens += usage_metrics.get('total_tokens', 0)
        else:
            # Object format
            state.requests += getattr(usage_metrics, 'successful_requests', 0)
            state.input_tokens += getattr(usage_metrics, 'prompt_tokens', 0)
            state.output_tokens += getattr(usage_metrics, 'completion_tokens', 0)
            state.total_tokens += getattr(usage_metrics, 'total_tokens', 0)

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


    
    