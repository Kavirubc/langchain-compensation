"""Compensation middleware for agents with automatic LIFO rollback."""

import json
import threading
import time
import uuid
from typing import Any, Callable, Dict, List

from langchain_core.messages import ToolMessage
from langchain.agents.middleware.types import AgentMiddleware, ToolCallRequest


class SagaCriticalFailure(Exception):
    """Raised when a compensation action fails, indicating the system is in an inconsistent state."""
    pass


class CompensationRecord(dict):
    """Tracks a compensatable action. Inherits dict for easy serialization."""

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
        )



class CompensationLog:
    """Manages compensation records with LIFO rollback ordering. Now thread-safe."""

    def __init__(self, records: Dict[str, CompensationRecord] | None = None):
        self._records = records if records is not None else {}
        self._lock = threading.Lock()

    def add(self, record: CompensationRecord) -> None:
        with self._lock:
            self._records[record["id"]] = record

    def update(self, record_id: str, **kwargs: Any) -> None:
        with self._lock:
            if record_id in self._records:
                self._records[record_id].update(kwargs)

    def get_rollback_plan(self) -> List[CompensationRecord]:
        with self._lock:
            candidates = [
                r
                for r in self._records.values()
                if r["status"] == "COMPLETED" and not r["compensated"] and r["compensation_tool"]
            ]
            if not candidates:
                return []
            id_to_record = {r["id"]: r for r in candidates}
            in_degree = {r["id"]: 0 for r in candidates}
            reverse_deps = {r["id"]: [] for r in candidates}
            for record in candidates:
                for dep_id in record.get("depends_on", []):
                    if dep_id in id_to_record:
                        reverse_deps[dep_id].append(record["id"])
                        in_degree[record["id"]] += 1
            queue = [r["id"] for r in candidates if in_degree[r["id"]] == 0]
            queue.sort(key=lambda rid: id_to_record[rid]["timestamp"], reverse=True)
            result = []
            while queue:
                current_id = queue.pop(0)
                result.append(id_to_record[current_id])
                for dependent_id in reverse_deps[current_id]:
                    in_degree[dependent_id] -= 1
                    if in_degree[dependent_id] == 0:
                        idx = 0
                        dep_timestamp = id_to_record[dependent_id]["timestamp"]
                        while idx < len(queue) and id_to_record[queue[idx]]["timestamp"] > dep_timestamp:
                            idx += 1
                        queue.insert(idx, dependent_id)
            if len(result) != len(candidates):
                print("WARNING: Dependency cycle detected! Falling back to timestamp order.")
                result = sorted(candidates, key=lambda x: x["timestamp"], reverse=True)
            print(f"DEBUG get_rollback_plan: Returning {len(result)} actions in DAG order:")
            for r in result:
                deps = r.get("depends_on", [])
                dep_info = f" (depends on {len(deps)} actions)" if deps else " (no dependencies)"
                print(f"  - {r['tool_name']}{dep_info}")
            return result

    def mark_compensated(self, record_id: str) -> None:
        with self._lock:
            if record_id in self._records:
                self._records[record_id]["compensated"] = True

    def to_dict(self) -> Dict[str, CompensationRecord]:
        with self._lock:
            return dict(self._records)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CompensationLog":
        if isinstance(data, list):
            return cls()
        return cls(records={k: CompensationRecord(**v) for k, v in data.items()})



class CompensationMiddleware(AgentMiddleware):
    """Middleware that automatically compensates failed tool calls using LIFO rollback."""

    def __init__(
        self,
        compensation_mapping: Dict[str, str],
        tools: Any = None,
        state_mappers: Dict[str, Callable[[Any, Dict[str, Any]], Dict[str, Any]]] | None = None,
        comp_log_ref: dict | None = None,
    ):
        """
        Initialize compensation middleware.

        Args:
            compensation_mapping: Maps tool names to their compensation tools
                (e.g., {"book_flight": "cancel_flight"})
            tools: List of tools to cache for compensation execution
            state_mappers: Optional custom mappers to extract params from results for compensation
            comp_log_ref: Optional external dict to use as the global compensation log (shared across agents)
        """
        self.compensation_mapping = compensation_mapping
        self.state_mappers = state_mappers or {}
        self._tools_cache: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._comp_log_ref = comp_log_ref
        self._shared_comp_log_instance = None

        # Cache all tools upfront for compensation use
        if tools:
            for tool in tools:
                if hasattr(tool, "name"):
                    self._tools_cache[tool.name] = tool

    def _cache_tool_from_request(self, request: ToolCallRequest) -> None:
        """Cache the current tool being executed for later compensation use."""
        tool_name = request.tool_call["name"]

        # Check if we can get the actual tool from the request
        if hasattr(request, "tool") and request.tool:
            self._tools_cache[tool_name] = request.tool
        elif hasattr(request.runtime, "config"):
            config = request.runtime.config
            # Try various config attributes where tools might be stored
            for attr in ["tools", "tools_by_name", "runnable"]:
                if hasattr(config, attr):
                    tools_source = getattr(config, attr)
                    if isinstance(tools_source, dict) and tool_name in tools_source:
                        self._tools_cache[tool_name] = tools_source[tool_name]
                        break

    def _get_tool(self, tool_name: str, request: ToolCallRequest) -> Any | None:
        """Retrieves tool from cache."""
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
        """Detects if tool result indicates an error using strict criteria only.
        
        Returns True only if:
        - ToolMessage has status="error"
        - Content is a dict with {"status": "error"}
        - An actual Exception was caught (handled by caller)
        
        Does NOT use heuristic keyword matching to avoid false positives.
        """
        # Check ToolMessage status attribute
        if hasattr(result, "status") and result.status == "error":
            return True

        # Check content dict for explicit error status
        content = result.content
        if isinstance(content, dict) and content.get("status") == "error":
            return True

        # Default: assume success
        return False

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
                print(f"  Data flow detected: Current action depends on '{record['tool_name']}' (ID: {record['id'][:8]}...)")
        
        return dependencies

    def _map_params(self, record: CompensationRecord) -> Any:
        """Maps compensation tool parameters from original result."""
        # Use custom mapper if available
        if record["tool_name"] in self.state_mappers:
            try:
                return self.state_mappers[record["tool_name"]](
                    record["result"], record["params"]
                )
            except Exception:
                pass

        result = record["result"]

        # Auto-extract common ID fields from dict results
        if isinstance(result, dict):
            for id_field in ["id", "booking_id", "resource_id", "transaction_id"]:
                if id_field in result:
                    return {id_field: result[id_field]}
            return result
        
        # For string results, treat them as the primary ID parameter
        # Common patterns: "flight_id_123", "booking_xyz", etc.
        if isinstance(result, str):
            # Try common parameter names for compensation tools
            for id_param in ["booking_id", "id", "resource_id", "transaction_id"]:
                return {id_param: result}

        return result  # Return raw result for other types

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


    def wrap_tool_call(self, request: ToolCallRequest, handler: Callable) -> ToolMessage:
        """Main middleware hook that wraps tool execution with compensation logic."""
        tool_name = request.tool_call["name"]
        state = request.state

        # Use global/shared compensation log if provided, else fall back to state dict


        def get_comp_log():
            if self._comp_log_ref is not None:
                # Always use a single, live CompensationLog instance for the shared log
                if self._shared_comp_log_instance is None:
                    self._shared_comp_log_instance = CompensationLog(records=self._comp_log_ref)
                    print(f"[DEBUG] Created shared CompensationLog instance id={id(self._shared_comp_log_instance)} records_id={id(self._shared_comp_log_instance._records)}")
                return self._shared_comp_log_instance
            return CompensationLog.from_dict(state.get("compensation_log", {}))

        def set_comp_log(comp_log: CompensationLog):
            if self._comp_log_ref is not None:
                # No-op: all mutations are in-place on the shared log
                print(f"[DEBUG] set_comp_log: shared instance id={id(comp_log)} records_id={id(comp_log._records)}")
                return
            else:
                state["compensation_log"] = comp_log.to_dict()
                print(f"[DEBUG] set_comp_log: state instance id={id(comp_log)} records_id={id(comp_log._records)}")

        # Cache this tool for potential compensation use
        self._cache_tool_from_request(request)


        is_compensatable = tool_name in self.compensation_mapping
        action_id = str(uuid.uuid4())
        # Store action_id in request.state for later update
        request.state["_comp_action_id"] = action_id

        # Track compensatable action before execution
        if is_compensatable:
            with self._lock:
                comp_log = get_comp_log()
                # Data Flow Dependency Inference:
                print(f"Inferring dependencies for '{tool_name}'...")
                depends_on = self._infer_dependencies(
                    request.tool_call.get("args", {}), comp_log
                )
                record = CompensationRecord(
                    id=action_id,
                    tool_name=tool_name,
                    params=request.tool_call.get("args", {}),
                    timestamp=time.time(),
                    compensation_tool=self.compensation_mapping[tool_name],
                    depends_on=depends_on,
                )
                comp_log.add(record)
                print(f"[DEBUG] After add: comp_log id={id(comp_log)} records_id={id(comp_log._records)} keys={list(comp_log._records.keys())}")
                set_comp_log(comp_log)

        # Execute the actual tool - catch exceptions and convert to error ToolMessages
        try:
            result = handler(request)
        except Exception as e:
            # Convert exceptions to ToolMessages with error status
            result = ToolMessage(
                content=f"Tool execution failed: {str(e)}",
                tool_call_id=request.tool_call.get("id", str(uuid.uuid4())),
                name=tool_name,
                status="error"
            )


        if not isinstance(result, ToolMessage):
            result = ToolMessage(
                content=result,
                tool_call_id=request.tool_call.get("id", str(uuid.uuid4())),
                name=tool_name,
            )

        is_error = self._is_error(result)

        # DEBUG: Print compensation log before rollback
        if is_error:
            print("=== COMPENSATION LOG BEFORE ROLLBACK ===")
            try:
                print(json.dumps(get_comp_log().to_dict(), indent=2))
            except Exception as debug_exc:
                print(f"[DEBUG] Failed to print compensation log: {debug_exc}")


        with self._lock:
            comp_log = get_comp_log()
            # Always use the action_id stored in request.state (if present)
            action_id_for_update = request.state.get("_comp_action_id", action_id)
            if is_error:
                print(f"Tool '{tool_name}' failed. Rolling back...")

                # Update failed action status
                if is_compensatable:
                    comp_log.update(
                        action_id_for_update, status="FAILED", result=self._extract_result(result)
                    )
                    print(f"[DEBUG] After FAILED update: comp_log id={id(comp_log)} records_id={id(comp_log._records)} keys={list(comp_log._records.keys())}")

                # Execute rollback plan in dependency order
                plan = comp_log.get_rollback_plan()
                for record in plan:
                    comp_tool = record["compensation_tool"]
                    print(f"Rolling back '{record['tool_name']}' using '{comp_tool}'...")

                    comp_params = self._map_params(record)
                    comp_result = self._execute_compensation(comp_tool, comp_params, request)

                    # Verify compensation succeeded using strict error detection
                    if self._is_error(comp_result):
                        error_msg = f"CRITICAL: Compensation failed for '{record['tool_name']}' using '{comp_tool}'. "
                        error_msg += f"Result: {comp_result.content}. System may be in inconsistent state."
                        raise SagaCriticalFailure(error_msg)

                    # Only mark as compensated if it actually succeeded
                    comp_log.mark_compensated(record["id"])
                    print(f"[DEBUG] After mark_compensated: comp_log id={id(comp_log)} records_id={id(comp_log._records)} keys={list(comp_log._records.keys())}")

                set_comp_log(comp_log)

            elif is_compensatable:
                # Mark successful compensatable action as completed
                comp_log.update(
                    action_id_for_update, status="COMPLETED", result=self._extract_result(result)
                )
                print(f"[DEBUG] Marked action {action_id_for_update} ({tool_name}) as COMPLETED in compensation log.")
                print(f"[DEBUG] After COMPLETED update: comp_log id={id(comp_log)} records_id={id(comp_log._records)} keys={list(comp_log._records.keys())}")
                set_comp_log(comp_log)

        return result
