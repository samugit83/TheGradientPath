"""
State management for the application.
This provides the main state for dependency injection and state management.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


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



@dataclass
class MainState:
    """
    Main state object that provides a clean interface for the application.
    This wraps the code generation state and tracks usage metrics.
    """

    # Configuration
    use_persistent_memory: bool = False
    model_name: str = "gpt-4.1"
    enable_content_safety: bool = True
    
    #State variables
    code_gen_state: Optional[CodeGenState] = None
    
    session_id: Optional[str] = None
    
    # Token usage tracking
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_token_price_per_million: float = 2.5
    output_token_price_per_million: float = 10
    
    # Real token counter for accurate usage tracking
    token_counter: Optional[Any] = None
    


def extract_usage_from_result(result: Any, token_counter: Any = None) -> Dict[str, int]:
    """
    Extract token usage from various result types:
    - Uses TokenCountingHandler for real usage data when available
    - Falls back to parsing result structures as backup
    """
    usage = {"requests": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    
    # REAL usage data from TokenCountingHandler (preferred method)
    if token_counter is not None:
        usage["requests"] = 1  # Each call to this function represents one workflow execution
        usage["input_tokens"] = token_counter.prompt_llm_token_count
        usage["output_tokens"] = token_counter.completion_llm_token_count  
        usage["total_tokens"] = token_counter.total_llm_token_count
        return usage
    
    # Fallback: Handle LlamaIndex AgentWorkflow result
    if hasattr(result, 'raw') and result.raw:
        raw_data = result.raw
        if isinstance(raw_data, dict) and 'usage' in raw_data and raw_data['usage']:
            r_usage = raw_data['usage']
            usage["requests"] = 1
            usage["input_tokens"] = r_usage.get("prompt_tokens", 0)
            usage["output_tokens"] = r_usage.get("completion_tokens", 0)
            usage["total_tokens"] = r_usage.get("total_tokens", usage["input_tokens"] + usage["output_tokens"])
            return usage
    
    # Fallback: Handle OpenAI agents.Runner result
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

# =============================================================================
# Top-level helpers for state mutation (moved out from dataclasses)
# =============================================================================

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


def update_main_usage(main_state: MainState, usage_dict: Dict[str, int]) -> None:
    """Update token usage counters on MainState."""
    main_state.requests += usage_dict.get("requests", 0)
    main_state.input_tokens += usage_dict.get("input_tokens", 0)
    main_state.output_tokens += usage_dict.get("output_tokens", 0)
    main_state.total_tokens += usage_dict.get("total_tokens", 0)


def calculate_main_cost(main_state: MainState) -> float:
    """Calculate total cost from token usage on MainState."""
    input_cost = (main_state.input_tokens / 1_000_000) * main_state.input_token_price_per_million
    output_cost = (main_state.output_tokens / 1_000_000) * main_state.output_token_price_per_million
    return input_cost + output_cost


def get_main_usage_summary(main_state: MainState) -> str:
    """Get a formatted token usage summary string for MainState."""
    total_cost = calculate_main_cost(main_state)
    return (
        f"📊 Token Usage:\n"
        f"  • Requests: {main_state.requests}\n"
        f"  • Input tokens: {main_state.input_tokens:,}\n"
        f"  • Output tokens: {main_state.output_tokens:,}\n"
        f"  • Total tokens: {main_state.total_tokens:,}\n"
        f"  • Total cost: ${total_cost:.4f}"
    )
