"""Compensation middleware for agents with automatic LIFO rollback."""

import json
import threading
import time
import uuid
import logging
from typing import Any, Callable, Dict, List

from langchain_core.messages import ToolMessage
from langchain.agents.middleware import AgentMiddleware
from langchain.tools.tool_node import ToolCallRequest
from langgraph.types import Command

from .batch import BatchManager, BatchContext, SequentialExecutionLock


class SagaCriticalFailure(Exception):
    """Raised when a compensation action fails, indicating the system is in an inconsistent state."""
    pass


class CompensationRecord(dict):
    """Tracks a compensatable action. Inherits dict for easy serialization.

    Attributes:
        id: Unique identifier for this record
        tool_name: Name of the tool that was executed
        params: Parameters passed to the tool
        result: Result returned by the tool (set after execution)
        timestamp: Unix timestamp when the action was recorded
        status: One of "PENDING", "COMPLETED", "FAILED"
        compensated: Whether compensation has been executed
        compensation_tool: Name of the tool to call for compensation
        depends_on: List of record IDs this action depends on (for DAG ordering)
        agent_id: Optional identifier for multi-agent scenarios
        thread_id: Optional thread/execution context identifier
    """

    def __init__(
        self,
        id: str,
        tool_name: str,
        params: Dict[str, Any],
        timestamp: float,
        compensation_tool: str,
        result: Any = None,
        status: str = "PENDING",
        compensated: bool = False,
        depends_on: List[str] | None = None,
        agent_id: str | None = None,
        thread_id: str | None = None,
    ):
        super().__init__(
            id=id,
            tool_name=tool_name,
            params=params,
            result=result,
            timestamp=timestamp,
            status=status,
            compensated=compensated,
            compensation_tool=compensation_tool,
            depends_on=depends_on or [],
            agent_id=agent_id,
            thread_id=thread_id,
        )



class CompensationLog:
    """Manages compensation records with LIFO rollback ordering.

    Thread-safe with support for:
    - Atomic batch operations for parallel tool execution
    - Immutable snapshots for safe iteration
    - Multi-agent filtering by agent_id
    - DAG-based rollback ordering with cycle detection

    Example:
        # Single agent usage
        log = CompensationLog()
        log.add(CompensationRecord(...))

        # Multi-agent with shared log
        shared_log = CompensationLog()
        agent1 = create_comp_agent(..., shared_log=shared_log)
        agent2 = create_comp_agent(..., shared_log=shared_log)

        # Parallel tool execution
        log.atomic_batch([
            ("add", record1),
            ("add", record2),
        ])

        # Safe iteration
        for record in log.snapshot().values():
            process(record)
    """

    def __init__(self, records: Dict[str, CompensationRecord] | None = None):
        self._records: Dict[str, CompensationRecord] = records if records is not None else {}
        self._lock = threading.RLock()  # RLock for nested/reentrant calls

    def add(self, record: CompensationRecord) -> None:
        """Add a single compensation record."""
        with self._lock:
            self._records[record["id"]] = record

    def update(self, record_id: str, **kwargs: Any) -> None:
        """Update fields of an existing record."""
        with self._lock:
            if record_id in self._records:
                self._records[record_id].update(kwargs)

    def get(self, record_id: str) -> CompensationRecord | None:
        """Get a record by ID."""
        with self._lock:
            return self._records.get(record_id)

    def snapshot(self) -> Dict[str, CompensationRecord]:
        """Return an immutable deep copy for safe iteration.

        Use this when iterating over records while other threads may modify
        the log. The snapshot is independent and won't reflect changes.

        Returns:
            Deep copy of all records
        """
        import copy
        with self._lock:
            return copy.deepcopy(self._records)

    def atomic_batch(self, operations: List[tuple]) -> None:
        """Execute multiple operations atomically.

        All operations are applied together under a single lock acquisition,
        ensuring consistency for parallel tool execution.

        Args:
            operations: List of (action, record_or_kwargs) tuples where:
                - ("add", CompensationRecord): Add a new record
                - ("update", {"id": str, **updates}): Update existing record
                - ("mark_compensated", record_id): Mark record as compensated

        Example:
            log.atomic_batch([
                ("add", record1),
                ("add", record2),
                ("update", {"id": "abc", "status": "COMPLETED"}),
            ])
        """
        with self._lock:
            for action, data in operations:
                if action == "add":
                    self._records[data["id"]] = data
                elif action == "update":
                    record_id = data.pop("id", None)
                    if record_id and record_id in self._records:
                        self._records[record_id].update(data)
                elif action == "mark_compensated":
                    if data in self._records:
                        self._records[data]["compensated"] = True

    def filter_by_agent(self, agent_id: str) -> List[CompensationRecord]:
        """Get all records for a specific agent.

        Args:
            agent_id: The agent identifier to filter by

        Returns:
            List of records belonging to the specified agent
        """
        with self._lock:
            return [r for r in self._records.values() if r.get("agent_id") == agent_id]

    def get_rollback_plan(self, agent_id: str | None = None) -> List[CompensationRecord]:
        """Generate ordered rollback plan respecting dependencies.

        Uses topological sort on the dependency DAG to ensure actions are
        compensated in the correct order (dependents before dependencies).

        Args:
            agent_id: If provided, only include records from this agent.
                     If None, includes all records (for multi-agent rollback).

        Returns:
            List of records in correct rollback order (most recent first,
            respecting dependency constraints)
        """
        with self._lock:
            # Get candidates for rollback
            candidates = [
                r
                for r in self._records.values()
                if r["status"] == "COMPLETED"
                and not r["compensated"]
                and r["compensation_tool"]
                and (agent_id is None or r.get("agent_id") == agent_id)
            ]
            if not candidates:
                return []

            # Build dependency graph
            id_to_record = {r["id"]: r for r in candidates}
            in_degree = {r["id"]: 0 for r in candidates}
            reverse_deps = {r["id"]: [] for r in candidates}

            for record in candidates:
                for dep_id in record.get("depends_on", []):
                    if dep_id in id_to_record:
                        reverse_deps[dep_id].append(record["id"])
                        in_degree[record["id"]] += 1

            # Topological sort with timestamp tiebreaker
            queue = [r["id"] for r in candidates if in_degree[r["id"]] == 0]
            queue.sort(key=lambda rid: id_to_record[rid]["timestamp"], reverse=True)

            result = []
            while queue:
                current_id = queue.pop(0)
                result.append(id_to_record[current_id])

                for dependent_id in reverse_deps[current_id]:
                    in_degree[dependent_id] -= 1
                    if in_degree[dependent_id] == 0:
                        # Insert maintaining timestamp order
                        idx = 0
                        dep_timestamp = id_to_record[dependent_id]["timestamp"]
                        while idx < len(queue) and id_to_record[queue[idx]]["timestamp"] > dep_timestamp:
                            idx += 1
                        queue.insert(idx, dependent_id)

            # Handle cycles by falling back to timestamp order
            if len(result) != len(candidates):
                logging.warning("Dependency cycle detected! Falling back to timestamp order.")
                result = sorted(candidates, key=lambda x: x["timestamp"], reverse=True)

            return result

    def mark_compensated(self, record_id: str) -> None:
        """Mark a record as compensated."""
        with self._lock:
            if record_id in self._records:
                self._records[record_id]["compensated"] = True

    def clear(self, agent_id: str | None = None) -> None:
        """Clear records from the log.

        Args:
            agent_id: If provided, only clear records for this agent.
                     If None, clears ALL records.
        """
        with self._lock:
            if agent_id is None:
                self._records.clear()
            else:
                self._records = {
                    k: v for k, v in self._records.items()
                    if v.get("agent_id") != agent_id
                }

    def __len__(self) -> int:
        """Return number of records in the log."""
        with self._lock:
            return len(self._records)

    def to_dict(self) -> Dict[str, CompensationRecord]:
        """Export log as dictionary (for state serialization)."""
        with self._lock:
            return dict(self._records)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CompensationLog":
        """Restore log from dictionary (for state deserialization)."""
        if isinstance(data, list):
            return cls()
        return cls(records={k: CompensationRecord(**v) for k, v in data.items()})



class CompensationMiddleware(AgentMiddleware):
    """Middleware that automatically compensates failed tool calls using LIFO rollback.

    This middleware implements the Saga pattern for distributed transactions in agent
    workflows. When a tool call fails, all previously successful compensatable actions
    are automatically rolled back in reverse order (respecting dependencies).

    Features:
    - Automatic LIFO rollback with DAG-based dependency ordering
    - Batch abort gate for parallel tool execution (fail-fast)
    - Pluggable error detection strategies
    - Pluggable parameter extraction strategies
    - Multi-agent support via shared CompensationLog
    - Thread-safe for parallel tool execution
    - Intent DAG tracking for debugging parallel execution issues
    - Explicit compensation schemas for precise control

    Example:
        # Simple usage (batch abort enabled by default)
        middleware = CompensationMiddleware(
            compensation_mapping={"book_flight": "cancel_flight"},
            tools=[book_flight, cancel_flight],
        )

        # With intent tracking for debugging parallel execution
        middleware = CompensationMiddleware(
            compensation_mapping={"book_flight": "cancel_flight"},
            tools=[book_flight, cancel_flight],
            track_intent=True,
        )

        # Advanced with shared log and schemas
        shared_log = CompensationLog()
        middleware = CompensationMiddleware(
            compensation_mapping={"book_flight": "cancel_flight"},
            tools=[book_flight, cancel_flight],
            shared_log=shared_log,
            agent_id="flight-agent",
            compensation_schemas={
                "book_flight": CompensationSchema(
                    param_mapping={"booking_id": "result.id"}
                )
            },
        )
    """

    def __init__(
        self,
        compensation_mapping: Dict[str, str],
        tools: Any = None,
        state_mappers: Dict[str, Callable[[Any, Dict[str, Any]], Dict[str, Any]]] | None = None,
        comp_log_ref: object | None = None,
        # New parameters for v2.0
        shared_log: "CompensationLog | None" = None,
        agent_id: str | None = None,
        compensation_schemas: Dict[str, Any] | None = None,
        error_strategies: List[Any] | None = None,
        extraction_strategies: List[Any] | None = None,
        # Parallel execution control (v2.1)
        enable_batch_abort: bool = True,
        track_intent: bool = False,
        batch_time_window_ms: float = 50,
        sequential_execution: bool = False,
    ):
        """
        Initialize compensation middleware.

        Args:
            compensation_mapping: Maps tool names to their compensation tools
                (e.g., {"book_flight": "cancel_flight"})
            tools: List of tools to cache for compensation execution
            state_mappers: Optional custom mappers to extract params from results
                for compensation (legacy, prefer compensation_schemas)
            comp_log_ref: Deprecated. Use shared_log instead.
            shared_log: Optional shared CompensationLog for multi-agent scenarios.
                When provided, all agents using this log can trigger coordinated
                rollback of each other's actions.
            agent_id: Optional identifier for this middleware instance. Used to
                track which agent performed which action in multi-agent scenarios.
            compensation_schemas: Dict mapping tool names to CompensationSchema
                objects for declarative parameter extraction.
            error_strategies: List of ErrorStrategy instances for pluggable error
                detection. If None, uses default strategies.
            extraction_strategies: List of ExtractionStrategy instances for
                pluggable parameter extraction. If None, uses default strategies.
            enable_batch_abort: If True (default), enables fail-fast behavior for
                parallel tool calls. When one tool fails, other tools in the same
                batch will be aborted before execution.
            track_intent: If True, tracks LLM's intended tool calls vs actual
                execution for debugging. Creates IntentDAG for observability.
            batch_time_window_ms: Time window in milliseconds for detecting
                parallel tool batches. Calls within this window from different
                threads are considered part of the same batch. Default: 50ms.
            sequential_execution: If True, forces sequential execution of
                compensatable tools using a lock. This is the most reliable way
                to prevent parallel execution race conditions, but may impact
                performance. When enabled, parallel tool calls will execute one
                at a time, and if one fails, subsequent tools will be aborted.
        """
        self.compensation_mapping = compensation_mapping
        self.state_mappers = state_mappers or {}
        self.agent_id = agent_id
        self.compensation_schemas = compensation_schemas or {}
        self._tools_cache: Dict[str, Any] = {}

        # Store strategy instances (lazy-loaded if using defaults)
        self._error_strategies = error_strategies
        self._extraction_strategies = extraction_strategies
        self._error_detector = None  # Lazy init
        self._param_extractor = None  # Lazy init

        # Parallel execution control
        self.enable_batch_abort = enable_batch_abort
        self.track_intent = track_intent
        self.sequential_execution = sequential_execution
        self._batch_manager = BatchManager(
            time_window_ms=batch_time_window_ms,
            track_intent=track_intent,
            sequential_execution=sequential_execution,
        )

        # Unified state management: handle both legacy and new parameter names
        self._comp_log: CompensationLog | None = None
        log_ref = shared_log or comp_log_ref  # Prefer new name
        if log_ref is not None:
            if isinstance(log_ref, CompensationLog):
                self._comp_log = log_ref
            elif isinstance(log_ref, dict):
                self._comp_log = CompensationLog(records=log_ref)
            else:
                raise ValueError("shared_log must be a CompensationLog instance or dict")

        # Cache tools for compensation execution
        if tools:
            for tool in tools:
                if hasattr(tool, "name"):
                    self._tools_cache[tool.name] = tool

    def _get_error_detector(self):
        """Lazy-load error detection strategy chain."""
        if self._error_detector is None:
            from .errors import create_error_detector
            self._error_detector = create_error_detector(
                strategies=self._error_strategies,
                default_is_error=False,
            )
        return self._error_detector

    def _get_param_extractor(self):
        """Lazy-load parameter extraction strategy chain."""
        if self._param_extractor is None:
            from .extraction import create_extraction_strategy
            self._param_extractor = create_extraction_strategy(
                state_mappers=self.state_mappers,
                compensation_schemas=self.compensation_schemas,
                raise_on_failure=False,  # Fallback to legacy _map_params
            )
        return self._param_extractor

    def _get_tool(self, tool_name: str, request: ToolCallRequest) -> Any | None:
        """Retrieves tool from request (new API) or cache."""
        if hasattr(request, "tool") and request.tool:
            return request.tool
        return self._tools_cache.get(tool_name)

    def _extract_result(self, msg: ToolMessage) -> Any:
        """Extracts structured result from ToolMessage content."""
        content = msg.content
        if isinstance(content, dict):
            return content
        if isinstance(content, str) and content.startswith("{") and content.endswith("}"):
            try:
                return json.loads(content)
            except Exception:
                pass
        return content

    def _is_error(self, result: ToolMessage) -> bool:
        """Detects if tool result indicates an error using pluggable strategies.

        Uses the configured error detection strategy chain. Default strategies:
        1. ExplicitStatusStrategy: Check ToolMessage.status == 'error'
        2. ContentDictStrategy: Check for {"error": ...}, {"success": false}
        3. ExceptionContentStrategy: Detect exception-like content patterns

        Returns:
            True if any strategy definitively detects an error,
            False otherwise (default behavior assumes success)
        """
        return self._get_error_detector().is_error(result)

    def _extract_values(self, data: Any, visited: set | None = None) -> set:
        """Recursively extract all primitive values from a data structure.
        
        Applies heuristic noise filtering to exclude low-entropy values that
        cause false dependencies (e.g., True, False, 0, 1, "ok", "id").
        
        Args:
            data: The data structure to extract values from
            visited: Set of visited object IDs to prevent infinite recursion
            
        Returns:
            Set of high-entropy primitive values suitable for dependency inference
        """
        if visited is None:
            visited = set()
        
        # Prevent infinite recursion on circular references
        obj_id = id(data)
        if obj_id in visited:
            return set()
        visited.add(obj_id)
        
        values = set()
        
        if isinstance(data, dict):
            for value in data.values():
                values.update(self._extract_values(value, visited))
        elif isinstance(data, (list, tuple)):
            for item in data:
                values.update(self._extract_values(item, visited))
        elif isinstance(data, (str, int, float)):
            # --- HEURISTIC NOISE FILTERING ---
            # Exclude common noise that causes false dependencies:
            # - Booleans (True/False appear everywhere)
            # - Small numbers (0, 1, 100, 200, etc. are configuration, not unique IDs)
            # - Short strings ("ok", "id", "USA" are not unique identifiers)
            
            if isinstance(data, bool):
                # Never include booleans - they create massive false positive graphs
                pass
            elif isinstance(data, (int, float)) and abs(data) < 10000:
                # Assume small numbers are configuration/status codes, not unique IDs
                pass
            elif isinstance(data, str) and len(data) < 5:
                # Assume short strings are not unique identifiers
                pass
            elif data == "" or data is None:
                # Exclude empty/null values
                pass
            else:
                # High-entropy value: likely a unique ID, hash, or meaningful data
                values.add(data)
        
        return values

    def _infer_dependencies(
        self, current_params: Dict[str, Any], comp_log: CompensationLog
    ) -> List[str]:
        """Infer dependencies by matching current params against previous results.
        
        This implements "Data Flow Dependency Inference" to build a true DAG.
        If the current action's parameters contain values that were produced by
        a previous action's result, then we have a data flow dependency.
        
        Args:
            current_params: Parameters for the current tool call
            comp_log: Current compensation log with history
            
        Returns:
            List of record IDs that the current action depends on
        """
        dependencies = []
        
        # Extract all values from current parameters
        param_values = self._extract_values(current_params)
        
        if not param_values:
            return dependencies
        
        # Check each completed action to see if it produced data we're consuming
        for record in comp_log._records.values():
            if record["status"] != "COMPLETED" or record["compensated"]:
                continue
            
            # Extract all values from this record's result
            result_values = self._extract_values(record["result"])
            
            # Check for data flow: do any param values match result values?
            if param_values & result_values:  # Set intersection
                dependencies.append(record["id"])
        
        return dependencies

    def _map_params(self, record: CompensationRecord) -> Any:
        """Maps compensation tool parameters from original result.

        Uses the configured extraction strategy chain:
        1. StateMappersStrategy: Custom mapping functions (highest priority)
        2. SchemaExtractionStrategy: Declarative CompensationSchema
        3. HeuristicExtractionStrategy: Common ID field names
        4. RecursiveSearchStrategy: Deep nested structure search
        5. PassthroughStrategy: Return entire result (last resort)

        Args:
            record: The compensation record containing result and params

        Returns:
            Dict of parameters to pass to the compensation tool
        """
        result = record["result"]
        original_params = record["params"]
        tool_name = record["tool_name"]
        comp_tool_name = record["compensation_tool"]

        # Get compensation tool from cache for schema inspection
        comp_tool = self._tools_cache.get(comp_tool_name)

        # Try the pluggable extraction strategy chain
        extractor = self._get_param_extractor()
        extracted = extractor.extract(
            result=result,
            original_params=original_params,
            compensation_tool=comp_tool,
            tool_name=tool_name,
        )

        if extracted is not None:
            return extracted

        # Fallback: legacy behavior for backwards compatibility
        if isinstance(result, dict):
            for id_field in ["id", "booking_id", "resource_id", "transaction_id"]:
                if id_field in result:
                    return {id_field: result[id_field]}
            return result

        if isinstance(result, str):
            return {"id": result}

        return result

    def _get_comp_log(self, state: Dict[str, Any]) -> CompensationLog:
        """Get compensation log from shared instance or state dict."""
        if self._comp_log is not None:
            return self._comp_log
        return CompensationLog.from_dict(state.get("compensation_log", {}))

    def _sync_comp_log(self, comp_log: CompensationLog, state: Dict[str, Any]) -> None:
        """Sync compensation log to state dict (only if not using shared instance)."""
        if self._comp_log is None:
            state["compensation_log"] = comp_log.to_dict()

    def _execute_compensation(
        self, tool_name: str, params: Any, request: ToolCallRequest
    ) -> ToolMessage:
        """Executes a compensation tool call."""
        tool_call_id = str(uuid.uuid4())
        try:
            tool = self._get_tool(tool_name, request)
            if not tool:
                return ToolMessage(
                    content=f"Error: Tool {tool_name} not found",
                    tool_call_id=tool_call_id,
                    name=tool_name,
                    status="error",
                )

            result = tool.invoke(params)
            return ToolMessage(content=str(result), tool_call_id=tool_call_id, name=tool_name)
        except Exception as e:
            return ToolMessage(
                content=f"Compensation failed: {e}",
                tool_call_id=tool_call_id,
                name=tool_name,
                status="error",
            )


    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Main middleware hook that wraps tool execution with compensation logic.

        This method:
        1. If sequential_execution enabled: acquires lock, checks abort after lock
        2. Otherwise: checks batch abort gate (if enabled)
        3. Records compensatable actions BEFORE execution (with PENDING status)
        4. Executes the tool via the handler
        5. On success: Updates record to COMPLETED
        6. On failure: Signals abort, triggers rollback of COMPLETED actions

        Args:
            request: The tool call request containing tool_call, state, etc.
            handler: The next handler in the chain to execute the tool

        Returns:
            ToolMessage or Command from the tool execution
        """
        tool_name = request.tool_call["name"]
        tool_call_id = request.tool_call.get("id", str(uuid.uuid4()))
        state = request.state
        is_compensatable = tool_name in self.compensation_mapping
        action_id = str(uuid.uuid4())

        # Get current thread ID for parallel execution tracking
        current_thread_id = str(threading.current_thread().ident)

        # === SEQUENTIAL EXECUTION MODE ===
        # If enabled, use lock to force one-at-a-time execution
        sequential_lock = self._batch_manager.get_sequential_lock()
        execution_slot = None

        if sequential_lock and is_compensatable:
            # Acquire execution slot (blocks until available)
            execution_slot = sequential_lock.acquire_execution_slot(tool_call_id)
            slot = execution_slot.__enter__()

            # Check abort AFTER acquiring lock - this is the key fix!
            if slot.should_abort:
                execution_slot.__exit__(None, None, None)
                logging.info(f"Tool '{tool_name}' aborted by sequential lock: {slot.abort_reason}")
                return ToolMessage(
                    content=f"Execution aborted: {slot.abort_reason}",
                    tool_call_id=tool_call_id,
                    name=tool_name,
                    status="error",
                )

        # === BATCH ABORT GATE (fallback for non-sequential mode) ===
        batch_ctx = None
        batch_id = None

        if self.enable_batch_abort and is_compensatable and not sequential_lock:
            # Detect batch via thread pattern
            batch_id = self._batch_manager.detect_batch(
                tool_name, current_thread_id, tool_call_id
            )

            if batch_id:
                batch_ctx = self._batch_manager.get_or_create_context(
                    batch_id=batch_id,
                    tool_count=5,
                    tool_call_ids=[tool_call_id],
                )

                if self.track_intent:
                    intent_dag = self._batch_manager.get_intent_dag(batch_id)
                    if intent_dag:
                        intent_dag.mark_executing(tool_call_id)

                if batch_ctx.should_abort():
                    logging.info(
                        f"Tool '{tool_name}' aborted: batch {batch_id} failed "
                        f"(trigger: {batch_ctx.failed_tool})"
                    )
                    if self.track_intent:
                        intent_dag = self._batch_manager.get_intent_dag(batch_id)
                        if intent_dag:
                            intent_dag.mark_aborted(tool_call_id)
                    return ToolMessage(
                        content=f"Execution aborted: {batch_ctx.abort_reason}",
                        tool_call_id=tool_call_id,
                        name=tool_name,
                        status="error",
                    )

        # Track compensatable action BEFORE execution
        if is_compensatable:
            comp_log = self._get_comp_log(state)
            record = CompensationRecord(
                id=action_id,
                tool_name=tool_name,
                params=request.tool_call.get("args", {}),
                timestamp=time.time(),
                compensation_tool=self.compensation_mapping[tool_name],
                depends_on=self._infer_dependencies(request.tool_call.get("args", {}), comp_log),
                agent_id=self.agent_id,
                thread_id=current_thread_id,
            )
            comp_log.add(record)
            self._sync_comp_log(comp_log, state)

        # Execute the tool with exception handling
        try:
            result = handler(request)
        except Exception as e:
            # Signal abort on exception
            if sequential_lock:
                sequential_lock.signal_abort(tool_name, str(e))
            elif batch_ctx:
                batch_ctx.signal_abort(tool_name, tool_call_id, str(e))

            result = ToolMessage(
                content=f"Tool execution failed: {str(e)}",
                tool_call_id=tool_call_id,
                name=tool_name,
                status="error",
            )

        # Ensure ToolMessage type (handler may return other types)
        if not isinstance(result, ToolMessage):
            result = ToolMessage(
                content=result,
                tool_call_id=tool_call_id,
                name=tool_name,
            )

        is_error = self._is_error(result)

        # Handle error: rollback all completed actions
        if is_error:
            # Signal abort so other parallel tools can fail fast
            if sequential_lock:
                sequential_lock.signal_abort(tool_name, str(result.content)[:200])
            elif batch_ctx:
                batch_ctx.signal_abort(tool_name, tool_call_id, str(result.content)[:200])

            # Update intent DAG
            if self.track_intent and batch_id:
                intent_dag = self._batch_manager.get_intent_dag(batch_id)
                if intent_dag:
                    intent_dag.mark_failed(tool_call_id)
                    intent_dag.abort_pending()

            logging.error(f"Tool '{tool_name}' failed. Initiating rollback...")
            comp_log = self._get_comp_log(state)

            if is_compensatable:
                comp_log.update(action_id, status="FAILED", result=self._extract_result(result))

            for record in comp_log.get_rollback_plan():
                comp_tool = record["compensation_tool"]
                logging.info(f"Rollback: {comp_tool} for {record['tool_name']}")

                comp_result = self._execute_compensation(comp_tool, self._map_params(record), request)

                if self._is_error(comp_result):
                    error_msg = f"Compensation failed for '{record['tool_name']}' using '{comp_tool}'. "
                    error_msg += f"Result: {comp_result.content}. System in inconsistent state."
                    logging.critical(error_msg)
                    # Release lock before raising
                    if execution_slot:
                        execution_slot.__exit__(None, None, None)
                    raise SagaCriticalFailure(error_msg)

                comp_log.mark_compensated(record["id"])

            self._sync_comp_log(comp_log, state)

        elif is_compensatable:
            # Update intent DAG on success
            if self.track_intent and batch_id:
                intent_dag = self._batch_manager.get_intent_dag(batch_id)
                if intent_dag:
                    intent_dag.mark_completed(tool_call_id)

            comp_log = self._get_comp_log(state)
            comp_log.update(action_id, status="COMPLETED", result=self._extract_result(result))
            self._sync_comp_log(comp_log, state)

        # Record batch execution for tracking (non-sequential mode)
        if batch_ctx:
            batch_ctx.record_execution(tool_call_id)
            if batch_ctx.is_complete():
                report = self._batch_manager.cleanup_batch(batch_id)
                if report:
                    logging.debug(f"Batch {batch_id} complete: {report}")

        # Release sequential execution lock
        if execution_slot:
            execution_slot.__exit__(None, None, None)

        return result
