"""Batch execution handling for parallel tool calls.

This module provides thread-safe primitives for detecting and managing
parallel tool execution batches, enabling fail-fast behavior when one
tool in a batch fails.

Key Components:
- BatchContext: Thread-safe batch state with atomic abort flag
- IntentNode: Represents a single intended tool call
- IntentDAG: Tracks LLM's intended tool calls vs actual executions
- BatchDetector: Detects parallel batches via thread patterns
- BatchManager: Manages batch contexts across executions
"""

import hashlib
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple


@dataclass
class BatchContext:
    """Thread-safe context for a batch of parallel tool calls.

    Uses threading.Event for the abort flag, which provides lock-free reads
    via is_set() for maximum performance in the hot path.

    Attributes:
        batch_id: Unique identifier for this batch
        tool_count: Expected number of tools in this batch
        tool_call_ids: List of tool call IDs in this batch
        abort_flag: Event that signals batch should abort
        abort_reason: Human-readable reason for abort
        failed_tool: Name of the tool that triggered abort
        failed_tool_call_id: ID of the tool call that failed

    Example:
        ctx = BatchContext(batch_id="abc123", tool_count=5, tool_call_ids=["1","2","3","4","5"])

        # In worker thread
        if ctx.should_abort():
            return "Aborted"

        # On failure
        ctx.signal_abort("schedule_job", "tc_2", "Resource not available")
    """

    batch_id: str
    tool_count: int
    tool_call_ids: List[str] = field(default_factory=list)
    abort_flag: threading.Event = field(default_factory=threading.Event)
    abort_reason: str | None = None
    failed_tool: str | None = None
    failed_tool_call_id: str | None = None
    _executed_count: int = field(default=0, repr=False)
    _completed_ids: Set[str] = field(default_factory=set, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def signal_abort(self, tool_name: str, tool_call_id: str, reason: str) -> bool:
        """Signal that this batch should abort.

        Thread-safe. Only the first call to signal_abort takes effect;
        subsequent calls are no-ops.

        Args:
            tool_name: Name of the tool that failed
            tool_call_id: ID of the specific tool call
            reason: Human-readable failure reason

        Returns:
            True if this was the first abort signal, False if already aborted
        """
        with self._lock:
            if self.abort_flag.is_set():
                return False  # Already aborted
            self.abort_flag.set()
            self.abort_reason = reason
            self.failed_tool = tool_name
            self.failed_tool_call_id = tool_call_id
            logging.info(f"Batch {self.batch_id} aborted by {tool_name}: {reason}")
            return True

    def should_abort(self) -> bool:
        """Check if batch execution should abort.

        Lock-free read for maximum performance. Safe to call frequently
        in the tool execution hot path.

        Returns:
            True if any tool in batch has failed
        """
        return self.abort_flag.is_set()

    def record_execution(self, tool_call_id: str | None = None) -> int:
        """Record that a tool execution completed.

        Args:
            tool_call_id: Optional ID of completed tool call

        Returns:
            Total number of tools executed so far
        """
        with self._lock:
            self._executed_count += 1
            if tool_call_id:
                self._completed_ids.add(tool_call_id)
            return self._executed_count

    def is_complete(self) -> bool:
        """Check if all tools in batch have been processed."""
        with self._lock:
            return self._executed_count >= self.tool_count

    def get_orphan_executions(self) -> List[str]:
        """Get tool call IDs that completed after abort was signaled.

        These are tools that "slipped through" the abort gate because
        they were already past the check point when abort was signaled.

        Returns:
            List of tool call IDs that executed despite abort
        """
        if not self.should_abort():
            return []

        with self._lock:
            # Orphans are completed tools that aren't the one that failed
            if self.failed_tool_call_id:
                return [
                    tc_id
                    for tc_id in self._completed_ids
                    if tc_id != self.failed_tool_call_id
                ]
            return list(self._completed_ids)


@dataclass
class IntentNode:
    """Represents a single intended tool call in a batch.

    Tracks the lifecycle of a tool call from intent through execution.

    Attributes:
        tool_call_id: Unique ID for this tool call
        tool_name: Name of the tool to execute
        args: Arguments passed to the tool
        status: Current status (PENDING/EXECUTING/COMPLETED/FAILED/ABORTED)
    """

    tool_call_id: str
    tool_name: str
    args: Dict[str, Any] = field(default_factory=dict)
    status: str = "PENDING"
    _status_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # Valid status transitions
    _VALID_TRANSITIONS: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "PENDING": ["EXECUTING", "ABORTED"],
            "EXECUTING": ["COMPLETED", "FAILED", "ABORTED"],
        },
        repr=False,
    )

    def transition(self, new_status: str) -> bool:
        """Attempt thread-safe status transition.

        Args:
            new_status: Target status to transition to

        Returns:
            True if transition was valid and applied, False otherwise
        """
        with self._status_lock:
            valid_next = self._VALID_TRANSITIONS.get(self.status, [])
            if new_status in valid_next:
                self.status = new_status
                return True
            return False


class IntentDAG:
    """Tracks LLM's intended tool calls for observability.

    When `track_intent=True`, this class captures what the LLM intended
    to execute and compares against actual execution. Useful for debugging
    parallel execution issues.

    Example:
        dag = IntentDAG("batch_1", [
            {"id": "tc_1", "name": "schedule_job", "args": {"job_id": "1"}},
            {"id": "tc_2", "name": "schedule_job", "args": {"job_id": "2"}},
        ])

        dag.mark_executing("tc_1")
        dag.mark_completed("tc_1")
        dag.mark_failed("tc_2")
        dag.abort_pending()

        report = dag.get_report()
        # {"completed": ["tc_1"], "failed": ["tc_2"], "aborted": []}
    """

    def __init__(self, batch_id: str, tool_calls: List[Dict[str, Any]]):
        """Initialize intent DAG from tool calls.

        Args:
            batch_id: Unique identifier for this batch
            tool_calls: List of tool call dicts with 'id', 'name', 'args'
        """
        self.batch_id = batch_id
        self.nodes: Dict[str, IntentNode] = {}
        self._lock = threading.Lock()

        for tc in tool_calls:
            node = IntentNode(
                tool_call_id=tc.get("id", f"unknown_{len(self.nodes)}"),
                tool_name=tc.get("name", "unknown"),
                args=tc.get("args", {}),
            )
            self.nodes[node.tool_call_id] = node

    def mark_executing(self, tool_call_id: str) -> bool:
        """Mark a tool as currently executing."""
        with self._lock:
            if tool_call_id in self.nodes:
                return self.nodes[tool_call_id].transition("EXECUTING")
            return False

    def mark_completed(self, tool_call_id: str) -> bool:
        """Mark a tool as successfully completed."""
        with self._lock:
            if tool_call_id in self.nodes:
                return self.nodes[tool_call_id].transition("COMPLETED")
            return False

    def mark_failed(self, tool_call_id: str) -> bool:
        """Mark a tool as failed."""
        with self._lock:
            if tool_call_id in self.nodes:
                return self.nodes[tool_call_id].transition("FAILED")
            return False

    def mark_aborted(self, tool_call_id: str) -> bool:
        """Mark a tool as aborted (didn't execute due to batch abort)."""
        with self._lock:
            if tool_call_id in self.nodes:
                return self.nodes[tool_call_id].transition("ABORTED")
            return False

    def abort_pending(self) -> List[str]:
        """Mark all PENDING tools as ABORTED.

        Called when batch abort is triggered to mark tools that
        never started execution.

        Returns:
            List of tool call IDs that were aborted
        """
        aborted = []
        with self._lock:
            for tool_id, node in self.nodes.items():
                if node.status == "PENDING":
                    if node.transition("ABORTED"):
                        aborted.append(tool_id)
        return aborted

    def get_report(self) -> Dict[str, Any]:
        """Generate execution report for this batch.

        Returns:
            Dict containing batch_id, status counts, and categorized tool IDs
        """
        with self._lock:
            status_counts: Dict[str, int] = {}
            by_status: Dict[str, List[str]] = {
                "completed": [],
                "failed": [],
                "aborted": [],
                "pending": [],
                "executing": [],
            }

            for node in self.nodes.values():
                status_counts[node.status] = status_counts.get(node.status, 0) + 1
                status_key = node.status.lower()
                if status_key in by_status:
                    by_status[status_key].append(node.tool_call_id)

            return {
                "batch_id": self.batch_id,
                "total_tools": len(self.nodes),
                "status_counts": status_counts,
                **by_status,
            }


class BatchDetector:
    """Detects parallel tool batches via thread execution patterns.

    When LangGraph dispatches parallel tool calls, they execute in
    different threads within a short time window. This class detects
    such patterns to identify batches without relying on state access.

    Example:
        detector = BatchDetector(time_window_ms=50)

        # Called from different threads near-simultaneously
        batch_id_1 = detector.record_call("schedule_job", thread_1)  # None (first call)
        batch_id_2 = detector.record_call("schedule_job", thread_2)  # "abc123" (batch detected!)
        batch_id_3 = detector.record_call("schedule_job", thread_3)  # "abc123" (same batch)
    """

    def __init__(self, time_window_ms: float = 50):
        """Initialize batch detector.

        Args:
            time_window_ms: Time window in milliseconds to group tool calls.
                            Calls within this window from different threads
                            are considered part of the same batch.
        """
        self.time_window = time_window_ms / 1000  # Convert to seconds
        # tool_name -> [(thread_id, timestamp, tool_call_id)]
        self._recent_calls: Dict[str, List[Tuple[str, float, str]]] = {}
        # batch_id -> set of tool_call_ids
        self._batch_tool_calls: Dict[str, Set[str]] = {}
        self._lock = threading.Lock()

    def record_call(
        self, tool_name: str, thread_id: str, tool_call_id: str
    ) -> str | None:
        """Record a tool call and detect if it's part of a parallel batch.

        Args:
            tool_name: Name of the tool being called
            thread_id: ID of the executing thread
            tool_call_id: Unique ID for this tool call

        Returns:
            batch_id if parallel batch detected, None for single execution
        """
        now = time.time()

        with self._lock:
            # Initialize if needed
            if tool_name not in self._recent_calls:
                self._recent_calls[tool_name] = []

            # Clean expired entries
            self._recent_calls[tool_name] = [
                (tid, ts, tcid)
                for tid, ts, tcid in self._recent_calls[tool_name]
                if now - ts < self.time_window
            ]

            # Add current call
            self._recent_calls[tool_name].append((thread_id, now, tool_call_id))

            # Check for parallel batch (multiple threads in window)
            calls_in_window = self._recent_calls[tool_name]
            threads_in_window = set(tid for tid, _, _ in calls_in_window)

            if len(threads_in_window) > 1:
                # Generate deterministic batch_id
                sorted_threads = sorted(threads_in_window)
                # Use first timestamp in window for stability
                first_ts = min(ts for _, ts, _ in calls_in_window)
                batch_id = hashlib.md5(
                    f"{tool_name}:{sorted_threads}:{int(first_ts)}".encode()
                ).hexdigest()[:16]

                # Track tool calls in this batch
                if batch_id not in self._batch_tool_calls:
                    self._batch_tool_calls[batch_id] = set()
                for _, _, tcid in calls_in_window:
                    self._batch_tool_calls[batch_id].add(tcid)

                return batch_id

            return None

    def get_batch_tool_calls(self, batch_id: str) -> List[str]:
        """Get all tool call IDs associated with a batch.

        Args:
            batch_id: The batch identifier

        Returns:
            List of tool call IDs in this batch
        """
        with self._lock:
            return list(self._batch_tool_calls.get(batch_id, set()))

    def cleanup_batch(self, batch_id: str) -> None:
        """Remove batch tracking data after completion.

        Args:
            batch_id: The batch to clean up
        """
        with self._lock:
            self._batch_tool_calls.pop(batch_id, None)

    def clear_expired(self) -> None:
        """Clear all expired entries from recent calls tracking."""
        now = time.time()
        with self._lock:
            for tool_name in list(self._recent_calls.keys()):
                self._recent_calls[tool_name] = [
                    (tid, ts, tcid)
                    for tid, ts, tcid in self._recent_calls[tool_name]
                    if now - ts < self.time_window
                ]
                if not self._recent_calls[tool_name]:
                    del self._recent_calls[tool_name]


class BatchManager:
    """Manages batch contexts and intent DAGs across parallel executions.

    Provides a unified interface for batch abort gate functionality,
    coordinating BatchDetector, BatchContext, and IntentDAG.

    Example:
        manager = BatchManager()

        # In wrap_tool_call
        batch_id = manager.detect_batch(tool_name, thread_id, tool_call_id)
        if batch_id:
            ctx = manager.get_or_create_context(batch_id, tool_count=5)
            if ctx.should_abort():
                return "Aborted"

            # On failure
            ctx.signal_abort(tool_name, tool_call_id, "Error")

        # Cleanup
        manager.cleanup_batch(batch_id)
    """

    def __init__(self, time_window_ms: float = 50, track_intent: bool = False):
        """Initialize batch manager.

        Args:
            time_window_ms: Time window for batch detection
            track_intent: Whether to track intent DAGs for observability
        """
        self._detector = BatchDetector(time_window_ms=time_window_ms)
        self._contexts: Dict[str, BatchContext] = {}
        self._intent_dags: Dict[str, IntentDAG] = {}
        self._track_intent = track_intent
        self._lock = threading.Lock()

    def detect_batch(
        self, tool_name: str, thread_id: str, tool_call_id: str
    ) -> str | None:
        """Detect if current tool call is part of a parallel batch.

        Args:
            tool_name: Name of the tool being called
            thread_id: ID of the executing thread
            tool_call_id: Unique ID for this tool call

        Returns:
            batch_id if parallel batch detected, None otherwise
        """
        return self._detector.record_call(tool_name, thread_id, tool_call_id)

    def get_or_create_context(
        self,
        batch_id: str,
        tool_count: int,
        tool_call_ids: List[str] | None = None,
    ) -> BatchContext:
        """Get existing batch context or create new one.

        Args:
            batch_id: Unique identifier for the batch
            tool_count: Expected number of tools in batch
            tool_call_ids: Optional list of tool call IDs

        Returns:
            BatchContext for this batch (existing or newly created)
        """
        with self._lock:
            if batch_id not in self._contexts:
                self._contexts[batch_id] = BatchContext(
                    batch_id=batch_id,
                    tool_count=tool_count,
                    tool_call_ids=tool_call_ids or [],
                )
            return self._contexts[batch_id]

    def get_context(self, batch_id: str) -> BatchContext | None:
        """Get existing batch context if it exists.

        Args:
            batch_id: The batch identifier

        Returns:
            BatchContext if exists, None otherwise
        """
        with self._lock:
            return self._contexts.get(batch_id)

    def create_intent_dag(
        self, batch_id: str, tool_calls: List[Dict[str, Any]]
    ) -> IntentDAG | None:
        """Create intent DAG for tracking if enabled.

        Args:
            batch_id: Unique identifier for the batch
            tool_calls: List of tool call dicts with 'id', 'name', 'args'

        Returns:
            IntentDAG if tracking enabled, None otherwise
        """
        if not self._track_intent:
            return None

        with self._lock:
            if batch_id not in self._intent_dags:
                self._intent_dags[batch_id] = IntentDAG(batch_id, tool_calls)
            return self._intent_dags[batch_id]

    def get_intent_dag(self, batch_id: str) -> IntentDAG | None:
        """Get existing intent DAG if it exists.

        Args:
            batch_id: The batch identifier

        Returns:
            IntentDAG if exists and tracking enabled, None otherwise
        """
        if not self._track_intent:
            return None

        with self._lock:
            return self._intent_dags.get(batch_id)

    def cleanup_batch(self, batch_id: str) -> Dict[str, Any] | None:
        """Clean up batch context and return final report.

        Args:
            batch_id: The batch to clean up

        Returns:
            Final intent DAG report if tracking was enabled, None otherwise
        """
        report = None

        with self._lock:
            # Get intent DAG report before cleanup
            if batch_id in self._intent_dags:
                report = self._intent_dags[batch_id].get_report()

                # Add orphan execution info from context
                if batch_id in self._contexts:
                    ctx = self._contexts[batch_id]
                    report["orphan_executions"] = ctx.get_orphan_executions()

                del self._intent_dags[batch_id]

            # Clean up context
            if batch_id in self._contexts:
                del self._contexts[batch_id]

            # Clean up detector
            self._detector.cleanup_batch(batch_id)

        if report:
            logging.debug(f"Batch {batch_id} report: {report}")

        return report

    def get_active_batch_count(self) -> int:
        """Get number of active batches being tracked."""
        with self._lock:
            return len(self._contexts)
