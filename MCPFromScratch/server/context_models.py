# context_models.py
from __future__ import annotations
from typing import Any, Optional
from fastapi import Request
from pydantic import BaseModel


class LifespanContext:
    def __init__(self, obj: Any):
        self._data = obj

    def __getattr__(self, item):
        return getattr(self._data, item)


class RequestContext(BaseModel):
    request: Optional[Request] = None        # None when called from WebSocket
    lifespan_context: Optional[LifespanContext] = None

    model_config = {'arbitrary_types_allowed': True}


class Context(BaseModel):
    """
    Injected into @tool functions when they declare `ctx: Context`.
    For WebSocket calls we only fill what we can.
    """
    request_context: Optional[RequestContext] = None
