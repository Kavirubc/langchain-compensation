"""Agent factory for creating agents with compensation capabilities.

This module provides a simple factory function to create LangChain agents
with automatic compensation (rollback) support using the Saga pattern.

Key Features:
- Drop-in replacement for create_agent() with saga pattern built-in
- Pluggable error detection and parameter extraction strategies
- Multi-agent support via shared compensation logs
- Optional LLM-based parameter extraction for complex cases
- Optional checkpointing for fault tolerance

Example:
    # Simple usage (most common)
    agent = create_comp_agent(
        model="gpt-4o",
        tools=[book_flight, cancel_flight, book_hotel, cancel_hotel],
        compensation_mapping={
            "book_flight": "cancel_flight",
            "book_hotel": "cancel_hotel",
        },
    )

    # With fault tolerance
    agent = create_comp_agent(
        model="gpt-4o",
        tools=[...],
        compensation_mapping={...},
        checkpointer=PostgresSaver(conn_string),
    )

    # With LLM extraction for complex parameter mapping
    agent = create_comp_agent(
        model="gpt-4o",
        tools=[...],
        compensation_mapping={...},
        use_llm_extraction=True,
        extraction_model="gpt-4o-mini",
    )
"""

from collections.abc import Callable, Sequence
from typing import Any, List

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, InterruptOnConfig
from langchain.agents.middleware.types import AgentMiddleware
from langchain.agents.structured_output import ResponseFormat
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.cache.base import BaseCache
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer

from .middleware import CompensationMiddleware, CompensationLog

BASE_AGENT_PROMPT = "In order to complete the objective that the user asks of you, you have access to a number of standard tools."


def create_comp_agent(
    model: str | BaseChatModel,
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
    *,
    compensation_mapping: dict[str, str],
    # Legacy parameters (still supported)
    state_mappers: dict[str, Callable[[Any, dict[str, Any]], dict[str, Any]]] | None = None,
    comp_log_ref: object | None = None,
    # New v2.0 parameters
    shared_log: CompensationLog | None = None,
    agent_id: str | None = None,
    compensation_schemas: dict[str, Any] | None = None,
    error_strategies: List[Any] | None = None,
    extraction_strategies: List[Any] | None = None,
    use_llm_extraction: bool = False,
    extraction_model: str = "gpt-4o-mini",
    use_checkpointing: bool = False,
    # Parallel execution control (v2.1)
    enable_batch_abort: bool = True,
    track_intent: bool = False,
    batch_time_window_ms: float = 50,
    disable_parallel_tool_calls: bool | str = False,
    # Standard agent parameters
    system_prompt: str | None = None,
    middleware: Sequence[AgentMiddleware] = (),
    response_format: ResponseFormat | None = None,
    context_schema: type[Any] | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    interrupt_on: dict[str, bool | InterruptOnConfig] | None = None,
    debug: bool = False,
    name: str | None = None,
    cache: BaseCache | None = None,
) -> CompiledStateGraph:
    """
    Create a LangChain agent with automatic compensation capabilities.

    This factory function creates an agent that automatically rolls back completed actions
    when a failure occurs, using the Saga pattern with LIFO (Last-In-First-Out) ordering.

    Args:
        model: The LLM to use for the agent (e.g., "gpt-4o" or ChatOpenAI instance).
        tools: The tools the agent should have access to.
        compensation_mapping: Dictionary mapping tool names to their compensation tools.
            Example: {"book_flight": "cancel_flight", "book_hotel": "cancel_hotel"}

        # Parameter Extraction Options (choose one or combine):
        state_mappers: Custom functions to extract compensation parameters from results.
            Highest priority. Example: {"book_flight": lambda r, p: {"id": r["booking_id"]}}
        compensation_schemas: CompensationSchema objects for declarative extraction.
            Example: {"book_flight": CompensationSchema(param_mapping={"id": "result.booking_id"})}
        use_llm_extraction: If True, use LLM as fallback for parameter extraction.
            Useful when heuristics fail on complex nested structures.
        extraction_model: Model to use for LLM extraction (default: "gpt-4o-mini").

        # Multi-Agent Support:
        shared_log: Shared CompensationLog for coordinated multi-agent rollback.
            When multiple agents share a log, failure in one triggers rollback across all.
        agent_id: Identifier for this agent instance (used with shared_log).

        # Pluggable Strategies:
        error_strategies: List of ErrorStrategy instances for custom error detection.
        extraction_strategies: List of ExtractionStrategy instances for custom extraction.

        # Fault Tolerance:
        use_checkpointing: If True, adds CheckpointMiddleware for state persistence.
            Requires checkpointer parameter to be set.
        checkpointer: LangGraph checkpointer for state persistence (e.g., PostgresSaver).

        # Parallel Execution Control:
        enable_batch_abort: If True (default), enables fail-fast behavior for parallel
            tool calls. When one tool fails, other tools in the same batch will be
            aborted before execution.
        track_intent: If True, tracks LLM's intended tool calls vs actual execution
            for debugging. Creates IntentDAG for observability.
        batch_time_window_ms: Time window in ms for detecting parallel batches (default: 50).
        disable_parallel_tool_calls: If True or "auto", attempts to disable parallel
            tool calls at the model level. Only supported for OpenAI-compatible models.
            - True: Force disable parallel tool calls
            - False: Allow parallel execution (default)
            - "auto": Auto-detect based on model capabilities

        # Standard Agent Parameters:
        system_prompt: Additional instructions for the agent.
        middleware: Additional middleware to apply after compensation middleware.
        comp_log_ref: Deprecated. Use shared_log instead.
        response_format: A structured output response format for the agent.
        context_schema: The schema of the agent context.
        store: Optional store for persistent storage.
        interrupt_on: Optional mapping of tool names to interrupt configurations.
        debug: Whether to enable debug mode.
        name: The name of the agent.
        cache: The cache to use for the agent.

    Returns:
        CompiledStateGraph: A configured agent with compensation middleware.

    Examples:
        # Basic usage
        >>> agent = create_comp_agent(
        ...     model="gpt-4o",
        ...     tools=[book_flight, cancel_flight],
        ...     compensation_mapping={"book_flight": "cancel_flight"}
        ... )

        # With explicit parameter mapping
        >>> from langchain_compensation import CompensationSchema
        >>> agent = create_comp_agent(
        ...     model="gpt-4o",
        ...     tools=[book_flight, cancel_flight],
        ...     compensation_mapping={"book_flight": "cancel_flight"},
        ...     compensation_schemas={
        ...         "book_flight": CompensationSchema(
        ...             param_mapping={"booking_id": "result.id"}
        ...         )
        ...     },
        ... )

        # Multi-agent with shared compensation
        >>> shared_log = CompensationLog()
        >>> flight_agent = create_comp_agent(
        ...     model="gpt-4o",
        ...     tools=[book_flight, cancel_flight],
        ...     compensation_mapping={"book_flight": "cancel_flight"},
        ...     shared_log=shared_log,
        ...     agent_id="flight-agent",
        ... )
        >>> hotel_agent = create_comp_agent(
        ...     model="gpt-4o",
        ...     tools=[book_hotel, cancel_hotel],
        ...     compensation_mapping={"book_hotel": "cancel_hotel"},
        ...     shared_log=shared_log,
        ...     agent_id="hotel-agent",
        ... )

        # With fault tolerance
        >>> from langgraph.checkpoint.postgres import PostgresSaver
        >>> agent = create_comp_agent(
        ...     model="gpt-4o",
        ...     tools=[...],
        ...     compensation_mapping={...},
        ...     checkpointer=PostgresSaver(conn_string),
        ...     use_checkpointing=True,
        ... )
    """
    # Build extraction strategies if using LLM extraction
    final_extraction_strategies = extraction_strategies
    if use_llm_extraction and extraction_strategies is None:
        from .extraction import (
            StateMappersStrategy,
            SchemaExtractionStrategy,
            HeuristicExtractionStrategy,
            RecursiveSearchStrategy,
            LLMExtractionStrategy,
            PassthroughStrategy,
        )
        final_extraction_strategies = []
        if state_mappers:
            final_extraction_strategies.append(StateMappersStrategy(state_mappers))
        if compensation_schemas:
            final_extraction_strategies.append(SchemaExtractionStrategy(compensation_schemas))
        final_extraction_strategies.extend([
            HeuristicExtractionStrategy(),
            RecursiveSearchStrategy(),
            LLMExtractionStrategy(model=extraction_model),
            PassthroughStrategy(),
        ])

    # Build middleware stack
    agent_middleware = []

    # Add checkpoint middleware first if requested
    if use_checkpointing and checkpointer:
        from .checkpoint import CheckpointMiddleware
        agent_middleware.append(CheckpointMiddleware(checkpointer=checkpointer))

    # Add compensation middleware with parallel execution control
    comp_middleware = CompensationMiddleware(
        compensation_mapping=compensation_mapping,
        tools=tools,
        state_mappers=state_mappers,
        comp_log_ref=comp_log_ref,  # Legacy support
        shared_log=shared_log or comp_log_ref,  # Prefer new name
        agent_id=agent_id,
        compensation_schemas=compensation_schemas,
        error_strategies=error_strategies,
        extraction_strategies=final_extraction_strategies,
        enable_batch_abort=enable_batch_abort,
        track_intent=track_intent,
        batch_time_window_ms=batch_time_window_ms,
    )
    agent_middleware.append(comp_middleware)

    # Handle model-level parallel tool call disabling
    final_model = model
    if disable_parallel_tool_calls:
        should_disable = False

        if disable_parallel_tool_calls == "auto":
            # Auto-detect: check if model supports parallel_tool_calls parameter
            if hasattr(model, "bind_tools"):
                # OpenAI-compatible models typically support this
                model_name = getattr(model, "model_name", "") or str(model)
                should_disable = any(
                    name in model_name.lower()
                    for name in ["gpt-4", "gpt-3.5", "openai"]
                )
        else:
            should_disable = True

        if should_disable and hasattr(model, "bind_tools") and tools:
            try:
                final_model = model.bind_tools(tools, parallel_tool_calls=False)
            except TypeError:
                # Model doesn't support parallel_tool_calls parameter
                pass

    # Add user-provided middleware
    if middleware:
        agent_middleware.extend(middleware)

    # Add human-in-the-loop if configured
    if interrupt_on is not None:
        agent_middleware.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))

    return create_agent(
        final_model,
        system_prompt=(
            system_prompt + "\n\n" + BASE_AGENT_PROMPT if system_prompt else BASE_AGENT_PROMPT
        ),
        tools=tools,
        middleware=agent_middleware,
        response_format=response_format,
        context_schema=context_schema,
        checkpointer=checkpointer,
        store=store,
        debug=debug,
        name=name,
        cache=cache,
    ).with_config({"recursion_limit": 1000})
