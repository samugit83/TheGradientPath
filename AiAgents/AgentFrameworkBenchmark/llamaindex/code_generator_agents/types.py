"""
Type definitions for the code generation agents.
Contains all Pydantic models used across the agent system.
"""
from typing import List
from pydantic import BaseModel, Field


class TestFailure(BaseModel):
    """Individual test failure information"""
    name: str
    trace: str


class TestResults(BaseModel):
    """Results from running tests"""
    passed: int = 0
    failed: int = 0
    failures: List[TestFailure] = Field(default_factory=list)
    test_explanation: str = ""


class CodeOutput(BaseModel):
    """Output structure for the Coder agent"""
    code: str
    explanation: str


class ReviewOutput(BaseModel):
    """Output structure for the Reviewer agent"""
    review_notes: List[str]
    analysis: str
    recommendation: str  
    should_continue: bool


class TestCodeOutput(BaseModel):
    """Output structure for the Tester agent when generating test code"""
    test_code: str
    explanation: str
    test_count: int