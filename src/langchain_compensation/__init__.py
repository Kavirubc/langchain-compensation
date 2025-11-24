"""LangChain Compensation - Automatic compensation middleware for agents."""

from .agents import create_comp_agent
from .middleware import CompensationLog, CompensationMiddleware, CompensationRecord, SagaCriticalFailure

__version__ = "0.1.0"

__all__ = [
    "create_comp_agent",
    "CompensationMiddleware",
    "CompensationLog",
    "CompensationRecord",
    "SagaCriticalFailure",
]
