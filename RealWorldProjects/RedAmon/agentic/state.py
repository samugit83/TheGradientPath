"""
RedAmon Agent State Management

LangGraph state and Pydantic models for the ReAct agent orchestrator.
Supports iterative Thought-Tool-Output pattern with phase tracking.
"""

from typing import Annotated, TypedDict, Optional, List, Literal, Dict
from datetime import datetime, timezone
import uuid

from params import MAX_ITERATIONS, EXECUTION_TRACE_MEMORY_STEPS


def utc_now() -> datetime:
    """Get current UTC time as timezone-aware datetime."""
    return datetime.now(timezone.utc)

import re
from pydantic import BaseModel, Field, field_validator
from langgraph.graph.message import add_messages


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

Phase = Literal["informational", "exploitation", "post_exploitation"]
TodoStatus = Literal["pending", "in_progress", "completed", "blocked"]
Priority = Literal["high", "medium", "low"]
ApprovalDecision = Literal["approve", "modify", "abort"]
QuestionFormat = Literal["text", "single_choice", "multi_choice"]


# =============================================================================
# PYDANTIC MODELS FOR STRUCTURED DATA
# =============================================================================

class TodoItem(BaseModel):
    """LLM-managed task item for tracking progress."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    description: str
    status: TodoStatus = "pending"
    priority: Priority = "medium"
    notes: Optional[str] = None
    created_at: datetime = Field(default_factory=utc_now)
    completed_at: Optional[datetime] = None

    def mark_complete(self) -> "TodoItem":
        """Mark this todo as completed."""
        return self.model_copy(update={
            "status": "completed",
            "completed_at": utc_now()
        })

    def mark_in_progress(self) -> "TodoItem":
        """Mark this todo as in progress."""
        return self.model_copy(update={"status": "in_progress"})


class ExecutionStep(BaseModel):
    """Single step in the Thought-Tool-Output execution trace."""
    step_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    iteration: int
    timestamp: datetime = Field(default_factory=utc_now)
    phase: Phase

    # Thought (reasoning before action)
    thought: str
    reasoning: str  # Why agent decided to take this action

    # Tool call (if any)
    tool_name: Optional[str] = None
    tool_args: Optional[dict] = None

    # Output (after tool execution)
    tool_output: Optional[str] = None
    output_analysis: Optional[str] = None  # Agent's interpretation of output

    # Status
    success: bool = True
    error_message: Optional[str] = None


class TargetInfo(BaseModel):
    """Accumulated intelligence about the target from graph queries and tools."""
    primary_target: Optional[str] = None  # IP or hostname
    target_type: Optional[Literal["ip", "hostname", "domain", "url"]] = None
    ports: List[int] = Field(default_factory=list)
    services: List[str] = Field(default_factory=list)
    technologies: List[str] = Field(default_factory=list)
    vulnerabilities: List[str] = Field(default_factory=list)  # CVE IDs or vuln descriptions
    credentials: List[dict] = Field(default_factory=list)  # Discovered credentials
    sessions: List[int] = Field(default_factory=list)  # Metasploit session IDs
    # Session details for richer tracking: {session_id: {'type': str, 'connection': str, 'info': str}}
    session_details: Dict[int, dict] = Field(default_factory=dict)

    def merge_from(self, other: "TargetInfo") -> "TargetInfo":
        """Merge new target info into existing, avoiding duplicates."""
        # Merge session_details, with other taking precedence for existing keys
        merged_session_details = {**self.session_details, **other.session_details}
        return TargetInfo(
            primary_target=other.primary_target or self.primary_target,
            target_type=other.target_type or self.target_type,
            ports=list(set(self.ports + other.ports)),
            services=list(set(self.services + other.services)),
            technologies=list(set(self.technologies + other.technologies)),
            vulnerabilities=list(set(self.vulnerabilities + other.vulnerabilities)),
            credentials=self.credentials + [c for c in other.credentials if c not in self.credentials],
            sessions=list(set(self.sessions + other.sessions)),
            session_details=merged_session_details,
        )


class PhaseTransitionRequest(BaseModel):
    """Request for user approval to transition between phases."""
    from_phase: Phase
    to_phase: Phase
    reason: str
    planned_actions: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    requires_approval: bool = True


class PhaseHistoryEntry(BaseModel):
    """Record of a phase transition."""
    phase: Phase
    entered_at: datetime = Field(default_factory=utc_now)
    exited_at: Optional[datetime] = None


# =============================================================================
# USER Q&A MODELS
# =============================================================================

class UserQuestionRequest(BaseModel):
    """Request for user clarification from the agent."""
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    question: str  # The question text to display to user
    context: str  # Why the agent needs this information
    format: QuestionFormat = "text"  # How user should respond
    options: List[str] = Field(default_factory=list)  # For choice formats
    default_value: Optional[str] = None  # Suggested default
    phase: Phase = "informational"  # Phase where question was asked


class UserQuestionAnswer(BaseModel):
    """User's answer to an agent question."""
    question_id: str
    answer: str  # The actual answer text
    timestamp: datetime = Field(default_factory=utc_now)


class QAHistoryEntry(BaseModel):
    """Combined Q&A entry for history tracking."""
    question: UserQuestionRequest
    answer: Optional[UserQuestionAnswer] = None
    answered_at: Optional[datetime] = None


# =============================================================================
# CONVERSATION OBJECTIVES (Multi-Objective Support)
# =============================================================================

class ConversationObjective(BaseModel):
    """Single objective within a continuous conversation."""
    objective_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    content: str  # The user's question/request
    created_at: datetime = Field(default_factory=utc_now)
    completed_at: Optional[datetime] = None
    completion_reason: Optional[str] = None
    required_phase: Optional[Phase] = None  # Hint for which phase this needs


class ObjectiveOutcome(BaseModel):
    """Outcome of a completed objective."""
    objective: ConversationObjective
    execution_steps: List[str] = Field(default_factory=list)  # Step IDs from execution_trace
    findings: dict = Field(default_factory=dict)  # Key findings from target_info
    success: bool = True


# =============================================================================
# LLM RESPONSE MODELS (for structured parsing)
# =============================================================================

ActionType = Literal["use_tool", "transition_phase", "complete", "ask_user"]


class PhaseTransitionDecision(BaseModel):
    """Phase transition details from LLM decision."""
    to_phase: Phase
    reason: str = ""
    planned_actions: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)


class UserQuestionDecision(BaseModel):
    """Question details from LLM decision when action=ask_user."""
    question: str
    context: str
    format: QuestionFormat = "text"
    options: List[str] = Field(default_factory=list)
    default_value: Optional[str] = None


class TodoItemUpdate(BaseModel):
    """Todo item from LLM response (simplified for updates)."""
    id: Optional[str] = None
    description: str
    status: TodoStatus = "pending"
    priority: Priority = "medium"


class LLMDecision(BaseModel):
    """
    Structured response from the ReAct think node.

    The LLM outputs JSON matching this schema to decide its next action.
    """
    thought: str = Field(description="Analysis of current situation")
    reasoning: str = Field(description="Why this action was chosen")
    action: ActionType = Field(default="use_tool", description="Type of action to take")

    # Tool execution fields (when action="use_tool")
    tool_name: Optional[str] = Field(default=None, description="Name of tool to execute")
    tool_args: Optional[dict] = Field(default=None, description="Arguments for the tool")

    # Phase transition fields (when action="transition_phase")
    phase_transition: Optional[PhaseTransitionDecision] = Field(default=None)

    # Completion fields (when action="complete")
    completion_reason: Optional[str] = Field(default=None, description="Why task is complete")

    # User question fields (when action="ask_user")
    user_question: Optional[UserQuestionDecision] = Field(default=None, description="Question to ask user")

    # Todo list updates (always present)
    updated_todo_list: List[TodoItemUpdate] = Field(default_factory=list)


class ExtractedTargetInfo(BaseModel):
    """Target information extracted from tool output analysis."""
    primary_target: Optional[str] = None
    ports: List[int] = Field(default_factory=list)
    services: List[str] = Field(default_factory=list)
    technologies: List[str] = Field(default_factory=list)
    vulnerabilities: List[str] = Field(default_factory=list)
    credentials: List[dict] = Field(default_factory=list)
    sessions: List[int] = Field(default_factory=list)


class OutputAnalysis(BaseModel):
    """
    Structured response from analyzing tool output.

    The LLM outputs JSON matching this schema after a tool executes.
    """
    interpretation: str = Field(description="What the output tells us about the target")
    extracted_info: ExtractedTargetInfo = Field(default_factory=ExtractedTargetInfo)
    actionable_findings: List[str] = Field(default_factory=list)
    recommended_next_steps: List[str] = Field(default_factory=list)


# =============================================================================
# LANGGRAPH STATE
# =============================================================================

class AgentState(TypedDict):
    """
    LangGraph state for the ReAct agent orchestrator.

    This state is maintained in memory via MemorySaver checkpointer.
    All execution history, todos, and phase tracking lives here.
    """
    # Core conversation history (managed by add_messages reducer)
    messages: Annotated[list, add_messages]

    # ReAct loop control
    current_iteration: int
    max_iterations: int
    task_complete: bool
    completion_reason: Optional[str]

    # Phase tracking
    current_phase: Phase
    phase_history: List[dict]  # List of PhaseHistoryEntry.model_dump()
    phase_transition_pending: Optional[dict]  # PhaseTransitionRequest.model_dump() or None

    # Execution trace (Thought-Tool-Output history)
    execution_trace: List[dict]  # List of ExecutionStep.model_dump()

    # LLM-managed todo list
    todo_list: List[dict]  # List of TodoItem.model_dump()

    # Objectives (multi-objective support)
    conversation_objectives: List[dict]  # List of ConversationObjective.model_dump()
    current_objective_index: int
    objective_history: List[dict]  # List of ObjectiveOutcome.model_dump()
    original_objective: str  # DEPRECATED: kept for backward compatibility

    # Target intelligence accumulated from queries
    target_info: dict  # TargetInfo.model_dump()

    # Session context
    user_id: str
    project_id: str
    session_id: str

    # Approval control
    awaiting_user_approval: bool
    user_approval_response: Optional[ApprovalDecision]
    user_modification: Optional[str]  # User's modification if they chose "modify"

    # User Q&A control
    awaiting_user_question: bool
    pending_question: Optional[dict]  # UserQuestionRequest.model_dump() or None
    user_question_answer: Optional[str]  # User's answer text
    qa_history: List[dict]  # List of QAHistoryEntry.model_dump() for context

    # Internal fields for inter-node communication (not persisted long-term)
    _current_step: Optional[dict]  # Current ExecutionStep being processed
    _decision: Optional[dict]  # LLM decision from think node
    _tool_result: Optional[dict]  # Result from tool execution
    _just_transitioned_to: Optional[str]  # Phase we just transitioned to (prevents re-requesting)

    # Metasploit state tracking
    msf_session_reset_done: bool  # True if metasploit was reset at start of this session


# =============================================================================
# RESPONSE MODELS
# =============================================================================

class InvokeResponse(BaseModel):
    """Response from agent invocation - returned by API."""
    # Core response
    answer: str = Field(default="", description="The agent's final answer or current status")
    tool_used: Optional[str] = Field(default=None, description="Name of the tool executed")
    tool_output: Optional[str] = Field(default=None, description="Raw output from the tool")
    error: Optional[str] = Field(default=None, description="Error message if failed")

    # ReAct state
    current_phase: Phase = Field(default="informational", description="Current agent phase")
    iteration_count: int = Field(default=0, description="Current iteration number")
    task_complete: bool = Field(default=False, description="Whether the task is complete")

    # Todo list for frontend display
    todo_list: List[dict] = Field(default_factory=list, description="Current task breakdown")

    # Execution trace summary (last N steps for context)
    execution_trace_summary: List[dict] = Field(
        default_factory=list,
        description="Summary of recent execution steps"
    )

    # Approval flow
    awaiting_approval: bool = Field(default=False, description="True if waiting for user approval")
    approval_request: Optional[dict] = Field(
        default=None,
        description="Phase transition request details if awaiting approval"
    )

    # Q&A flow
    awaiting_question: bool = Field(default=False, description="True if waiting for user answer")
    question_request: Optional[dict] = Field(
        default=None,
        description="Question request details if awaiting_question is True"
    )


class ApprovalRequest(BaseModel):
    """Request model for user approval endpoint."""
    session_id: str
    user_id: str
    project_id: str
    decision: ApprovalDecision
    modification: Optional[str] = None  # User's modification if decision="modify"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_initial_state(
    user_id: str,
    project_id: str,
    session_id: str,
    objective: str,
    max_iterations: int = MAX_ITERATIONS
) -> dict:
    """Create initial state for a new agent session."""
    # Create first objective
    first_objective = ConversationObjective(content=objective).model_dump()

    return {
        "messages": [],
        "current_iteration": 0,
        "max_iterations": max_iterations,
        "task_complete": False,
        "completion_reason": None,
        "current_phase": "informational",
        "phase_history": [PhaseHistoryEntry(phase="informational").model_dump()],
        "phase_transition_pending": None,
        "execution_trace": [],
        "todo_list": [],
        # Multi-objective support
        "conversation_objectives": [first_objective],
        "current_objective_index": 0,
        "objective_history": [],
        "original_objective": objective,  # Kept for backward compatibility
        "target_info": TargetInfo().model_dump(),
        "user_id": user_id,
        "project_id": project_id,
        "session_id": session_id,
        "awaiting_user_approval": False,
        "user_approval_response": None,
        "user_modification": None,
        # Q&A fields
        "awaiting_user_question": False,
        "pending_question": None,
        "user_question_answer": None,
        "qa_history": [],
        # Internal fields
        "_current_step": None,
        "_decision": None,
        "_tool_result": None,
        "_just_transitioned_to": None,
        # Metasploit state
        "msf_session_reset_done": False,
    }


def format_todo_list(todo_list: List[dict]) -> str:
    """Format todo list for display in prompts."""
    if not todo_list:
        return "No tasks defined yet."

    lines = []
    for i, todo in enumerate(todo_list, 1):
        status_icon = {
            "pending": "[ ]",
            "in_progress": "[~]",
            "completed": "[x]",
            "blocked": "[!]"
        }.get(todo.get("status", "pending"), "[ ]")

        priority = todo.get("priority", "medium")
        priority_marker = {"high": "!!!", "medium": "!!", "low": "!"}.get(priority, "!!")

        lines.append(f"{i}. {status_icon} {priority_marker} {todo.get('description', 'No description')}")
        if todo.get("notes"):
            lines.append(f"   Notes: {todo['notes']}")

    return "\n".join(lines)


def format_execution_trace(
    trace: List[dict],
    objectives: List[dict] = None,
    objective_history: List[dict] = None,
    current_objective_index: int = 0,
    last_n: int = None
) -> str:
    """
    Format execution trace with objective grouping.

    Groups steps by objective for better context across multi-objective sessions.
    Uses EXECUTION_TRACE_MEMORY_STEPS from params to control how many steps to show.

    IMPORTANT: This function provides context to the LLM for subsequent decisions.
    Tool outputs must be included so the agent can reference previous results
    (e.g., module paths from 'search CVE-XXX', options from 'info exploit/...').

    Args:
        trace: List of execution step dicts
        objectives: List of conversation objectives
        objective_history: List of completed objective outcomes
        current_objective_index: Index of current objective
        last_n: Override for number of steps (None = use EXECUTION_TRACE_MEMORY_STEPS)
    """
    if not trace:
        return "No steps executed yet."

    # Use configured limit or override
    limit = last_n if last_n is not None else EXECUTION_TRACE_MEMORY_STEPS

    # Apply limit to trace (most recent steps)
    limited_trace = trace[-limit:] if len(trace) > limit else trace

    lines = []

    # If we truncated, show a note
    if len(trace) > limit:
        lines.append(f"[Showing last {limit} of {len(trace)} total steps]")
        lines.append("")

    # Build objective boundaries from objective_history
    # Each completed objective in history has 'execution_steps' (step IDs)
    completed_step_ids = set()
    if objective_history:
        for i, outcome in enumerate(objective_history):
            obj = outcome.get("objective", {})
            step_ids = set(outcome.get("execution_steps", []))

            # Find steps belonging to this objective (that are in our limited trace)
            obj_steps = [s for s in limited_trace if s.get("step_id") in step_ids]

            if obj_steps:
                completed_step_ids.update(step_ids)
                lines.append(f"\n{'='*60}")
                lines.append(f"=== OBJECTIVE {i+1}: {obj.get('content', 'Unknown')[:80]}...")
                lines.append(f"=== Status: COMPLETED")
                lines.append(f"{'='*60}\n")

                for step in obj_steps:
                    lines.extend(_format_single_step(step))

    # Current objective steps (not in completed history)
    current_steps = [s for s in limited_trace if s.get("step_id") not in completed_step_ids]

    if current_steps:
        current_obj_content = "Current objective"
        if objectives and current_objective_index < len(objectives):
            current_obj_content = objectives[current_objective_index].get("content", "Current objective")[:80]

        lines.append(f"\n{'='*60}")
        lines.append(f"=== OBJECTIVE {len(objective_history or []) + 1}: {current_obj_content}...")
        lines.append(f"=== Status: IN PROGRESS")
        lines.append(f"{'='*60}\n")

        for step in current_steps:
            lines.extend(_format_single_step(step))

    return "\n".join(lines)


def _format_single_step(step: dict) -> List[str]:
    """Format a single execution step."""
    lines = []
    iteration = step.get("iteration", "?")
    phase = step.get("phase", "unknown")
    thought = step.get("thought", "No thought recorded")
    tool = step.get("tool_name", "none")
    tool_args = step.get("tool_args", {})
    success = "OK" if step.get("success", True) else "FAILED"
    error_msg = step.get("error_message")

    lines.append(f"--- Step {iteration} [{phase}] - {success} ---")
    lines.append(f"Thought: {thought[:10000]}..." if len(thought) > 10000 else f"Thought: {thought}")

    if tool and tool != "none":
        lines.append(f"Tool: {tool}")
        if tool_args:
            args_str = str(tool_args)
            lines.append(f"Args: {args_str[:10000]}..." if len(args_str) > 10000 else f"Args: {args_str}")

        # CRITICAL: Include full tool output so agent can reference results
        # This is essential for exploitation workflows where search/info results
        # must be used in subsequent exploit commands
        tool_output = step.get("tool_output", "")
        if tool_output:
            max_output_len = 10000
            if len(tool_output) > max_output_len:
                lines.append(f"Output (truncated):\n{tool_output[:max_output_len]}...\n[{len(tool_output) - max_output_len} more chars]")
            else:
                lines.append(f"Output:\n{tool_output}")

        if step.get("output_analysis"):
            analysis = step["output_analysis"]
            lines.append(f"Analysis: {analysis[:10000]}..." if len(analysis) > 10000 else f"Analysis: {analysis}")

    if error_msg:
        lines.append(f"Error: {error_msg}")

    lines.append("")
    return lines


def summarize_trace_for_response(trace: List[dict], last_n: int = None) -> List[dict]:
    """Create a summary of the execution trace for API response."""
    limit = last_n if last_n is not None else EXECUTION_TRACE_MEMORY_STEPS
    recent = trace[-limit:] if len(trace) > limit else trace

    return [
        {
            "iteration": step.get("iteration"),
            "phase": step.get("phase"),
            "thought": step.get("thought", "")[:10000],
            "tool_name": step.get("tool_name"),
            "success": step.get("success", True),
            "output_summary": (step.get("output_analysis") or "")[:10000]
        }
        for step in recent
    ]


def format_qa_history(qa_history: List[dict]) -> str:
    """Format Q&A history for display in prompts."""
    if not qa_history:
        return "No previous questions asked."

    lines = []
    for i, entry in enumerate(qa_history, 1):
        q = entry.get("question", {})
        a = entry.get("answer")

        lines.append(f"Q{i}: {q.get('question', 'Unknown question')}")
        lines.append(f"   Context: {q.get('context', 'No context')}")
        lines.append(f"   Phase: {q.get('phase', 'unknown')}")

        if a:
            lines.append(f"   Answer: {a.get('answer', 'No answer')}")
        else:
            lines.append(f"   Answer: (unanswered)")
        lines.append("")

    return "\n".join(lines)


def format_objective_history(objective_history: List[dict]) -> str:
    """Format completed objectives for display in prompts."""
    if not objective_history:
        return "No previous objectives completed."

    lines = []
    for i, outcome in enumerate(objective_history, 1):
        obj = outcome.get("objective", {})
        lines.append(f"{i}. {obj.get('content', 'Unknown')}")
        lines.append(f"   Status: {'✓ Success' if outcome.get('success') else '✗ Failed'}")

        # Format findings summary
        findings = outcome.get("findings", {})
        vuln_count = len(findings.get("vulnerabilities", []))
        port_count = len(findings.get("ports", []))
        session_count = len(findings.get("sessions", []))

        lines.append(f"   Findings: {vuln_count} vulns, {port_count} ports, {session_count} sessions")
        lines.append("")

    return "\n".join(lines)


def migrate_legacy_objective(state: dict) -> dict:
    """
    Migrate old original_objective to new conversation_objectives format.

    This ensures backward compatibility with sessions created before multi-objective support.
    """
    if "original_objective" in state and "conversation_objectives" not in state:
        original = state.get("original_objective", "")
        if original:
            state["conversation_objectives"] = [
                ConversationObjective(content=original).model_dump()
            ]
            state["current_objective_index"] = 0
            state["objective_history"] = []
    return state
