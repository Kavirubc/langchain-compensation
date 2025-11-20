"""Agent factory for creating agents with compensation capabilities."""

from collections.abc import Callable, Sequence
from typing import Any

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

from .middleware import CompensationMiddleware

BASE_AGENT_PROMPT = "In order to complete the objective that the user asks of you, you have access to a number of standard tools."


def create_comp_agent(
    model: str | BaseChatModel,
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
    *,
    compensation_mapping: dict[str, str],
    state_mappers: dict[str, Callable[[Any, dict[str, Any]], dict[str, Any]]] | None = None,
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
        model: The LLM to use for the agent.
        tools: The tools the agent should have access to.
        compensation_mapping: Dictionary mapping tool names to their compensation tools.
            Example: {"book_flight": "cancel_flight", "book_hotel": "cancel_hotel"}
        state_mappers: Optional custom functions to extract compensation parameters from results.
            Example: {"book_flight": lambda result, params: {"booking_id": result["id"]}}
        system_prompt: Additional instructions for the agent.
        middleware: Additional middleware to apply after compensation middleware.
        response_format: A structured output response format for the agent.
        context_schema: The schema of the agent context.
        checkpointer: Optional checkpointer for persisting agent state between runs.
        store: Optional store for persistent storage.
        interrupt_on: Optional mapping of tool names to interrupt configurations.
        debug: Whether to enable debug mode.
        name: The name of the agent.
        cache: The cache to use for the agent.

    Returns:
        CompiledStateGraph: A configured agent with compensation middleware.

    Example:
        >>> from langchain_compensation import create_comp_agent
        >>> from langchain_openai import ChatOpenAI
        >>> from langchain_core.tools import tool
        >>>
        >>> @tool
        >>> def book_flight(destination: str) -> str:
        ...     return f"flight_{destination}"
        >>>
        >>> @tool
        >>> def cancel_flight(booking_id: str) -> str:
        ...     return "cancelled"
        >>>
        >>> agent = create_comp_agent(
        ...     model=ChatOpenAI(model="gpt-4"),
        ...     tools=[book_flight, cancel_flight],
        ...     compensation_mapping={"book_flight": "cancel_flight"}
        ... )
    """
    comp_middleware = CompensationMiddleware(
        compensation_mapping=compensation_mapping,
        tools=tools,
        state_mappers=state_mappers,
    )

    agent_middleware = [comp_middleware]

    if middleware:
        agent_middleware.extend(middleware)
    if interrupt_on is not None:
        agent_middleware.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))

    return create_agent(
        model,
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
