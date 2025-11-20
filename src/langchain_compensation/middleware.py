"""Compensation middleware for agents with automatic LIFO rollback."""

import json
import threading
import time
import uuid
from typing import Any, Callable, Dict, List

from langchain_core.messages import ToolMessage
from langchain.agents.middleware.types import AgentMiddleware, ToolCallRequest


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
        )


class CompensationLog:
    """Manages compensation records with LIFO rollback ordering."""

    def __init__(self, records: Dict[str, CompensationRecord] | None = None):
        self._records = records or {}

    def add(self, record: CompensationRecord) -> None:
        """Add a new compensation record."""
        self._records[record["id"]] = record

    def update(self, record_id: str, **kwargs: Any) -> None:
        """Update an existing record."""
        if record_id in self._records:
            self._records[record_id].update(kwargs)

    def get_rollback_plan(self) -> List[CompensationRecord]:
        """Returns uncompensated completed actions in reverse chronological order (LIFO)."""
        candidates = [
            r
            for r in self._records.values()
            if r["status"] == "COMPLETED" and not r["compensated"] and r["compensation_tool"]
        ]
        return sorted(candidates, key=lambda x: x["timestamp"], reverse=True)

    def mark_compensated(self, record_id: str) -> None:
        """Mark an action as compensated."""
        if record_id in self._records:
            self._records[record_id]["compensated"] = True

    def to_dict(self) -> Dict[str, CompensationRecord]:
        """Export log as dictionary."""
        return self._records

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CompensationLog":
        """Create log from dictionary."""
        if isinstance(data, list):  # Handle legacy format
            return cls()
        return cls(records={k: CompensationRecord(**v) for k, v in data.items()})


class CompensationMiddleware(AgentMiddleware):
    """Middleware that automatically compensates failed tool calls using LIFO rollback."""

    def __init__(
        self,
        compensation_mapping: Dict[str, str],
        tools: Any = None,
        state_mappers: Dict[str, Callable[[Any, Dict[str, Any]], Dict[str, Any]]] | None = None,
    ):
        """
        Initialize compensation middleware.

        Args:
            compensation_mapping: Maps tool names to their compensation tools
                (e.g., {"book_flight": "cancel_flight"})
            tools: List of tools to cache for compensation execution
            state_mappers: Optional custom mappers to extract params from results for compensation
        """
        self.compensation_mapping = compensation_mapping
        self.state_mappers = state_mappers or {}
        self._tools_cache: Dict[str, Any] = {}
        self._lock = threading.Lock()

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
        """Detects if tool result indicates an error."""
        if hasattr(result, "status") and result.status == "error":
            return True

        content = result.content
        if isinstance(content, dict) and content.get("status") == "error":
            return True

        # Check for error keywords in content
        content_str = str(content).lower()
        return any(
            keyword in content_str
            for keyword in [
                "error:",
                "error",
                "failed",
                "exception",
                "traceback",
                "cannot",
                "unable to",
            ]
        )

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

        # Cache this tool for potential compensation use
        self._cache_tool_from_request(request)

        # Load compensation log from state
        comp_log = CompensationLog.from_dict(state.get("compensation_log", {}))

        is_compensatable = tool_name in self.compensation_mapping
        action_id = str(uuid.uuid4())

        # Track compensatable action before execution
        if is_compensatable:
            with self._lock:
                record = CompensationRecord(
                    id=action_id,
                    tool_name=tool_name,
                    params=request.tool_call.get("args", {}),
                    timestamp=time.time(),
                    compensation_tool=self.compensation_mapping[tool_name],
                )
                comp_log.add(record)
                state["compensation_log"] = comp_log.to_dict()

        # Execute the actual tool
        result = handler(request)

        if not isinstance(result, ToolMessage):
            return result

        is_error = self._is_error(result)

        with self._lock:
            if is_error:
                print(f"Tool '{tool_name}' failed. Rolling back...")

                # Update failed action status
                if is_compensatable:
                    comp_log.update(
                        action_id, status="FAILED", result=self._extract_result(result)
                    )

                # Execute rollback plan in LIFO order
                plan = comp_log.get_rollback_plan()

                for record in plan:
                    comp_tool = record["compensation_tool"]
                    print(f"Rolling back '{record['tool_name']}' using '{comp_tool}'...")

                    comp_params = self._map_params(record)
                    self._execute_compensation(comp_tool, comp_params, request)
                    comp_log.mark_compensated(record["id"])

                state["compensation_log"] = comp_log.to_dict()

            elif is_compensatable:
                # Mark successful compensatable action as completed
                comp_log.update(
                    action_id, status="COMPLETED", result=self._extract_result(result)
                )
                state["compensation_log"] = comp_log.to_dict()

        return result
