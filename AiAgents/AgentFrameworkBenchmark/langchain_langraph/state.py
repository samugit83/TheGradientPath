"""
State management for the langraph application.
This provides the main state for dependency injection and state management.
"""
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from langgraph.graph import add_messages
from typing_extensions import Annotated
from langgraph.graph.message import add_messages

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
    # User input for code generation
    user_prompt_for_app: Optional[str] = None
    
    # Constraints for code generation
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
    
    current_code: Optional[Any] = None  
    previous_code: Optional[str] = None
    test_results: Optional[Any] = None  
    review_notes: List[str] = field(default_factory=list)

    
    
def codegen_increment_iteration(code_gen_state: CodeGenState) -> None:
    """Increment the iteration counter for code generation."""
    code_gen_state.iteration += 1


def codegen_update_code(code_gen_state: CodeGenState, code_output: Any) -> None:
    """Update the current code and save previous code snapshot."""
    if code_gen_state.current_code:
        code_gen_state.previous_code = code_gen_state.current_code.code
    code_gen_state.current_code = code_output


def codegen_mark_complete(code_gen_state: CodeGenState) -> None:
    """Mark the code generation process as complete."""
    code_gen_state.process_completed = True



@dataclass
class ContentSafetyResult:
    """Results from content safety analysis"""
    is_appropriate: bool = True
    reason: str = "Content is appropriate"


    category: str = "safe"
@dataclass
class MainState(): 
    """
    Main state object that provides a clean interface for the application.
    This wraps the code generation state and tracks usage metrics.
    Extends AgentState to support LangGraph's InjectedState functionality.
    """

    messages: Annotated[list, add_messages] = field(default_factory=list)

    # Configuration
    use_persistent_memory: bool = True
    model_name: str = "gpt-4.1"
    enable_content_safety: bool = True
    
    routeName: Optional[str] = field(default=None)
    
    # Content safety state
    content_safety_result: Optional[ContentSafetyResult] = field(default=None)
    
    # Code generation state
    code_gen_state: Optional[CodeGenState] = field(default=None)
    
    # Token usage tracking (new)
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_token_price_per_million: float = 2.5
    output_token_price_per_million: float = 10

    def __post_init__(self):
        """Initialize code_gen_state with default values if not provided"""
        if self.code_gen_state is None:
            self.code_gen_state = CodeGenState()


# Standalone token usage tracking functions
def update_usage(state: MainState, usage_dict: Dict[str, int]):
    """Update token usage from a usage dictionary"""
    state.requests += usage_dict.get("requests", 0)
    state.input_tokens += usage_dict.get("input_tokens", 0)
    state.output_tokens += usage_dict.get("output_tokens", 0)
    state.total_tokens += usage_dict.get("total_tokens", 0)


def calculate_cost(state: MainState) -> float:
    """Calculate the total cost based on token usage"""
    input_cost = (state.input_tokens / 1_000_000) * state.input_token_price_per_million
    output_cost = (state.output_tokens / 1_000_000) * state.output_token_price_per_million
    return input_cost + output_cost


def get_usage_summary(state: MainState) -> str:
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
    
