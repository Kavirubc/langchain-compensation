"""LangChain Compensation - Automatic compensation middleware for agents.

A composable middleware library for adding automatic compensation (rollback)
to LangChain agents using the Saga pattern.

Key Features:
- Automatic LIFO rollback with DAG-based dependency ordering
- Pluggable error detection and parameter extraction strategies
- Multi-agent support via shared compensation logs
- Optional LLM-based parameter extraction for complex cases
- Optional checkpointing for fault tolerance

Quick Start:
    from langchain_compensation import create_comp_agent

    agent = create_comp_agent(
        model="gpt-4o",
        tools=[book_flight, cancel_flight, book_hotel, cancel_hotel],
        compensation_mapping={
            "book_flight": "cancel_flight",
            "book_hotel": "cancel_hotel",
        },
    )

Advanced Usage:
    from langchain_compensation import (
        create_comp_agent,
        CompensationSchema,
        CompensationLog,
    )

    # Multi-agent with shared compensation
    shared_log = CompensationLog()
    flight_agent = create_comp_agent(
        model="gpt-4o",
        tools=[book_flight, cancel_flight],
        compensation_mapping={"book_flight": "cancel_flight"},
        shared_log=shared_log,
        agent_id="flight-agent",
    )
"""

# Core middleware and classes
from .middleware import (
    CompensationMiddleware,
    CompensationLog,
    CompensationRecord,
    SagaCriticalFailure,
)

# Agent factory
from .agents import create_comp_agent

# Error detection strategies
from .errors import (
    ErrorStrategy,
    ExplicitStatusStrategy,
    ContentDictStrategy,
    ExceptionContentStrategy,
    CompositeErrorStrategy,
    create_error_detector,
)

# Parameter extraction strategies
from .extraction import (
    CompensationSchema,
    ExtractionStrategy,
    StateMappersStrategy,
    SchemaExtractionStrategy,
    HeuristicExtractionStrategy,
    RecursiveSearchStrategy,
    LLMExtractionStrategy,
    PassthroughStrategy,
    CompositeExtractionStrategy,
    create_extraction_strategy,
)

# Checkpoint middleware for fault tolerance
from .checkpoint import CheckpointMiddleware

# Batch execution handling for parallel tool calls
from .batch import (
    BatchContext,
    IntentNode,
    IntentDAG,
    BatchDetector,
    BatchManager,
)

__version__ = "0.5.0"

__all__ = [
    # Core
    "create_comp_agent",
    "CompensationMiddleware",
    "CompensationLog",
    "CompensationRecord",
    "SagaCriticalFailure",
    # Error strategies
    "ErrorStrategy",
    "ExplicitStatusStrategy",
    "ContentDictStrategy",
    "ExceptionContentStrategy",
    "CompositeErrorStrategy",
    "create_error_detector",
    # Extraction strategies
    "CompensationSchema",
    "ExtractionStrategy",
    "StateMappersStrategy",
    "SchemaExtractionStrategy",
    "HeuristicExtractionStrategy",
    "RecursiveSearchStrategy",
    "LLMExtractionStrategy",
    "PassthroughStrategy",
    "CompositeExtractionStrategy",
    "create_extraction_strategy",
    # Checkpoint
    "CheckpointMiddleware",
    # Batch execution
    "BatchContext",
    "IntentNode",
    "IntentDAG",
    "BatchDetector",
    "BatchManager",
]
