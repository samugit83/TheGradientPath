"""
Code Generator Agents Package
Using openai-agents library patterns for multi-agent code generation
"""

from .types import (
    CodeOutput,
    TestResults,
    TestFailure,
    ReviewOutput,
    TestCodeOutput
)

from .coder import (
    create_coder
)

from .tester import (
    create_tester,
    post_process_test_output
)

from .reviewer import (
    create_reviewer,
    post_process_review_output,
    get_fallback_review
)

from .code_orchestrator_agent import run_code_orchestrator

__all__ = [
    # Type classes
    'CodeOutput',
    'TestResults',
    'TestFailure',
    'ReviewOutput',
    'TestCodeOutput',
    
    # Agent creators
    'create_coder',
    'create_tester',
    'create_reviewer',
    
    # Post-processors
    'post_process_test_output',
    'post_process_review_output',
    'get_fallback_review',
    
    # Orchestrator
    'run_code_orchestrator'
]