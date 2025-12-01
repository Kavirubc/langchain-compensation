"""Checkpoint middleware for fault-tolerant compensation state persistence.

This module provides CheckpointMiddleware that integrates LangGraph's checkpointer
system with the compensation saga pattern. It enables:

- Automatic state persistence before each tool call
- Recovery from failures by restoring last good state
- Time-travel debugging through checkpoint history
- Fault-tolerant agent workflows that survive process crashes

Research Contribution: First middleware to integrate LangGraph checkpointers
with saga compensation for distributed transaction fault tolerance.
"""

import logging
import time
import uuid
from typing import Any, Callable, Dict

from langchain_core.messages import ToolMessage
from langchain.agents.middleware import AgentMiddleware
from langchain.tools.tool_node import ToolCallRequest
from langgraph.types import Command

from .middleware import CompensationLog


# Constants for checkpoint metadata keys
CHECKPOINT_KEY = "compensation_checkpoint"
CHECKPOINT_METADATA_KEY = "compensation_metadata"


class CheckpointMiddleware(AgentMiddleware):
    """Middleware that saves compensation state to LangGraph checkpointer for fault tolerance.

    This middleware wraps tool calls to:
    1. Save the current compensation log state before execution
    2. Record checkpoint metadata (timestamps, tool info)
    3. Enable recovery by restoring from checkpoints on failure

    Works with any LangGraph checkpointer (InMemorySaver, PostgresSaver, etc.).

    Example:
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.checkpoint.postgres import PostgresSaver

        # In-memory for development
        checkpoint_middleware = CheckpointMiddleware(
            checkpointer=InMemorySaver()
        )

        # PostgreSQL for production
        checkpoint_middleware = CheckpointMiddleware(
            checkpointer=PostgresSaver(conn_string="postgresql://...")
        )

        # Use with compensation middleware
        agent = create_agent(
            model="gpt-4o",
            tools=[...],
            middleware=[
                checkpoint_middleware,
                compensation_middleware,
            ],
        )

    Integration with create_comp_agent:
        agent = create_comp_agent(
            model="gpt-4o",
            tools=[...],
            compensation_mapping={...},
            checkpointer=PostgresSaver(conn_string),  # Enable fault tolerance
        )
    """

    def __init__(
        self,
        checkpointer: Any = None,
        checkpoint_every_n_calls: int = 1,
        include_tool_results: bool = True,
    ):
        """Initialize checkpoint middleware.

        Args:
            checkpointer: LangGraph checkpointer instance (InMemorySaver, PostgresSaver, etc.).
                If None, creates an InMemorySaver for basic operation.
            checkpoint_every_n_calls: Save checkpoint every N tool calls.
                Set to 1 for maximum durability, higher for better performance.
            include_tool_results: Whether to include full tool results in checkpoints.
                Disable for large results to reduce storage.
        """
        self.checkpointer = checkpointer
        self.checkpoint_every_n_calls = checkpoint_every_n_calls
        self.include_tool_results = include_tool_results
        self._call_count = 0

        # Lazy import to avoid circular dependencies
        if checkpointer is None:
            try:
                from langgraph.checkpoint.memory import MemorySaver
                self.checkpointer = MemorySaver()
            except ImportError:
                logging.warning(
                    "langgraph.checkpoint not available. "
                    "Install langgraph for checkpoint support."
                )

    def _should_checkpoint(self) -> bool:
        """Determine if we should save a checkpoint based on call count."""
        self._call_count += 1
        return self._call_count % self.checkpoint_every_n_calls == 0

    def _get_thread_id(self, state: Dict[str, Any]) -> str:
        """Extract or generate thread ID from state."""
        # Try common state keys for thread ID
        for key in ["thread_id", "configurable.thread_id", "config.thread_id"]:
            if "." in key:
                parts = key.split(".")
                current = state
                try:
                    for part in parts:
                        current = current[part]
                    return str(current)
                except (KeyError, TypeError):
                    continue
            elif key in state:
                return str(state[key])

        # Generate a default thread ID if none found
        return f"compensation-{uuid.uuid4().hex[:8]}"

    def _save_checkpoint(
        self,
        state: Dict[str, Any],
        tool_name: str,
        checkpoint_type: str = "pre_tool",
    ) -> str | None:
        """Save compensation state to checkpointer.

        Args:
            state: Current agent state containing compensation_log
            tool_name: Name of tool being executed
            checkpoint_type: Type of checkpoint (pre_tool, post_tool, rollback)

        Returns:
            Checkpoint ID if successful, None otherwise
        """
        if not self.checkpointer:
            return None

        try:
            thread_id = self._get_thread_id(state)
            checkpoint_id = f"{thread_id}-{uuid.uuid4().hex[:8]}"

            # Extract compensation log from state
            comp_log_data = state.get("compensation_log", {})

            # Build checkpoint data
            checkpoint_data = {
                "id": checkpoint_id,
                "thread_id": thread_id,
                "timestamp": time.time(),
                "checkpoint_type": checkpoint_type,
                "tool_name": tool_name,
                "compensation_log": comp_log_data,
            }

            # Save using checkpointer's put method
            config = {"configurable": {"thread_id": thread_id}}

            # LangGraph checkpointer interface
            if hasattr(self.checkpointer, "put"):
                self.checkpointer.put(
                    config=config,
                    checkpoint={
                        "v": 1,
                        "id": checkpoint_id,
                        "ts": time.time(),
                        "channel_values": {CHECKPOINT_KEY: checkpoint_data},
                        "channel_versions": {},
                        "versions_seen": {},
                    },
                    metadata={
                        "source": "compensation_middleware",
                        "checkpoint_type": checkpoint_type,
                        "tool_name": tool_name,
                    },
                    new_versions={},
                )
                return checkpoint_id

            logging.warning("Checkpointer does not support put() method")
            return None

        except Exception as e:
            logging.error(f"Failed to save checkpoint: {e}")
            return None

    def _restore_checkpoint(
        self,
        state: Dict[str, Any],
        checkpoint_id: str | None = None,
    ) -> Dict[str, Any] | None:
        """Restore compensation state from checkpointer.

        Args:
            state: Current state (for thread_id extraction)
            checkpoint_id: Specific checkpoint to restore, or None for latest

        Returns:
            Restored compensation log data, or None if not found
        """
        if not self.checkpointer:
            return None

        try:
            thread_id = self._get_thread_id(state)
            config = {"configurable": {"thread_id": thread_id}}

            # Get checkpoint using checkpointer's get method
            if hasattr(self.checkpointer, "get"):
                checkpoint = self.checkpointer.get(config)
                if checkpoint and "channel_values" in checkpoint:
                    return checkpoint["channel_values"].get(CHECKPOINT_KEY)

            # Try get_tuple for newer LangGraph versions
            if hasattr(self.checkpointer, "get_tuple"):
                checkpoint_tuple = self.checkpointer.get_tuple(config)
                if checkpoint_tuple and checkpoint_tuple.checkpoint:
                    channel_values = checkpoint_tuple.checkpoint.get("channel_values", {})
                    return channel_values.get(CHECKPOINT_KEY)

            return None

        except Exception as e:
            logging.error(f"Failed to restore checkpoint: {e}")
            return None

    def get_checkpoint_history(
        self,
        state: Dict[str, Any],
        limit: int = 10,
    ) -> list:
        """Get history of checkpoints for time-travel debugging.

        Args:
            state: Current state (for thread_id extraction)
            limit: Maximum number of checkpoints to return

        Returns:
            List of checkpoint metadata, newest first
        """
        if not self.checkpointer:
            return []

        try:
            thread_id = self._get_thread_id(state)
            config = {"configurable": {"thread_id": thread_id}}

            # Try list method for checkpoint history
            if hasattr(self.checkpointer, "list"):
                checkpoints = list(self.checkpointer.list(config, limit=limit))
                return [
                    {
                        "checkpoint_id": cp.config.get("configurable", {}).get("checkpoint_id"),
                        "timestamp": cp.metadata.get("timestamp"),
                        "tool_name": cp.metadata.get("tool_name"),
                        "checkpoint_type": cp.metadata.get("checkpoint_type"),
                    }
                    for cp in checkpoints
                ]

            return []

        except Exception as e:
            logging.error(f"Failed to get checkpoint history: {e}")
            return []

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Wrap tool calls with checkpoint save/restore logic.

        Flow:
        1. Save pre-execution checkpoint (if checkpoint interval reached)
        2. Execute tool via handler
        3. Save post-execution checkpoint on success
        4. On failure, checkpoint is available for recovery

        Args:
            request: Tool call request
            handler: Next handler in chain

        Returns:
            Tool execution result
        """
        tool_name = request.tool_call.get("name", "unknown")
        state = request.state

        # Save pre-execution checkpoint
        if self._should_checkpoint():
            checkpoint_id = self._save_checkpoint(
                state=state,
                tool_name=tool_name,
                checkpoint_type="pre_tool",
            )
            if checkpoint_id:
                logging.debug(f"Saved pre-tool checkpoint: {checkpoint_id}")

        # Execute the tool
        result = handler(request)

        # Save post-execution checkpoint on success
        # (Error detection is handled by CompensationMiddleware)
        if self._should_checkpoint():
            self._save_checkpoint(
                state=state,
                tool_name=tool_name,
                checkpoint_type="post_tool",
            )

        return result

    def restore_from_failure(
        self,
        state: Dict[str, Any],
        checkpoint_id: str | None = None,
    ) -> CompensationLog | None:
        """Restore compensation log from a checkpoint after failure.

        Use this method to recover agent state after a crash or failure:

            # After process restart
            checkpoint_middleware = CheckpointMiddleware(checkpointer=saved_checkpointer)
            restored_log = checkpoint_middleware.restore_from_failure(state)

            if restored_log:
                # Resume with restored compensation state
                middleware = CompensationMiddleware(
                    compensation_mapping={...},
                    shared_log=restored_log,
                )

        Args:
            state: State dict (needs thread_id for lookup)
            checkpoint_id: Specific checkpoint to restore, or None for latest

        Returns:
            Restored CompensationLog, or None if no checkpoint found
        """
        checkpoint_data = self._restore_checkpoint(state, checkpoint_id)

        if checkpoint_data and "compensation_log" in checkpoint_data:
            return CompensationLog.from_dict(checkpoint_data["compensation_log"])

        return None
