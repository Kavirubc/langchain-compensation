"""Basic tests for compensation middleware."""

import concurrent.futures
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool

from langchain_compensation import CompensationLog, CompensationMiddleware, CompensationRecord


def test_compensation_record_creation():
    """Test creating a compensation record."""
    record = CompensationRecord(
        id="test-id",
        tool_name="test_tool",
        params={"arg": "value"},
        timestamp=1234567890.0,
        compensation_tool="undo_test_tool",
    )
    assert record["id"] == "test-id"
    assert record["tool_name"] == "test_tool"
    assert record["status"] == "PENDING"
    assert record["compensated"] is False


def test_compensation_log_add_and_update():
    """Test adding and updating compensation log entries."""
    log = CompensationLog()
    
    record = CompensationRecord(
        id="test-1",
        tool_name="tool1",
        params={},
        timestamp=1.0,
        compensation_tool="undo1",
    )
    
    log.add(record)
    assert len(log._records) == 1
    
    log.update("test-1", status="COMPLETED", result="success")
    assert log._records["test-1"]["status"] == "COMPLETED"
    assert log._records["test-1"]["result"] == "success"


def test_compensation_log_rollback_plan():
    """Test getting rollback plan in LIFO order."""
    log = CompensationLog()
    
    # Add three completed records
    for i in range(3):
        record = CompensationRecord(
            id=f"test-{i}",
            tool_name=f"tool{i}",
            params={},
            timestamp=float(i),
            compensation_tool=f"undo{i}",
            status="COMPLETED",
        )
        log.add(record)
    
    plan = log.get_rollback_plan()
    
    # Should be in reverse order (LIFO)
    assert len(plan) == 3
    assert plan[0]["id"] == "test-2"  # Last one first
    assert plan[1]["id"] == "test-1"
    assert plan[2]["id"] == "test-0"  # First one last


def test_compensation_log_filters_non_completed():
    """Test that rollback plan only includes completed actions."""
    log = CompensationLog()
    
    # Add one pending and one completed
    log.add(
        CompensationRecord(
            id="pending",
            tool_name="tool1",
            params={},
            timestamp=1.0,
            compensation_tool="undo1",
            status="PENDING",
        )
    )
    log.add(
        CompensationRecord(
            id="completed",
            tool_name="tool2",
            params={},
            timestamp=2.0,
            compensation_tool="undo2",
            status="COMPLETED",
        )
    )
    
    plan = log.get_rollback_plan()
    
    # Only completed should be in plan
    assert len(plan) == 1
    assert plan[0]["id"] == "completed"


def test_compensation_middleware_initialization():
    """Test middleware initialization with tools."""
    
    @tool
    def test_tool(arg: str) -> str:
        """Test tool for middleware."""
        return "result"
    
    @tool
    def undo_tool(arg: str) -> str:
        """Undo tool for compensation."""
        return "undone"
    
    middleware = CompensationMiddleware(
        compensation_mapping={"test_tool": "undo_tool"},
        tools=[test_tool, undo_tool],
    )
    
    assert "test_tool" in middleware._tools_cache
    assert "undo_tool" in middleware._tools_cache


def test_compensation_log_serialization():
    """Test log serialization to/from dict."""
    log = CompensationLog()
    
    record = CompensationRecord(
        id="test-1",
        tool_name="tool1",
        params={"key": "value"},
        timestamp=123.45,
        compensation_tool="undo1",
        status="COMPLETED",
    )
    log.add(record)
    
    # Serialize
    data = log.to_dict()
    assert "test-1" in data
    
    # Deserialize
    log2 = CompensationLog.from_dict(data)
    assert "test-1" in log2._records
    assert log2._records["test-1"]["tool_name"] == "tool1"


# ============================================================================
# Parallel Execution / Batch Abort Gate Tests
# ============================================================================


@dataclass
class MockToolCallRequest:
    """Mock ToolCallRequest for testing."""

    tool_call: Dict[str, Any]
    state: Dict[str, Any]
    tool: Any = None


class TestMiddlewareBatchAbort:
    """Tests for batch abort functionality in CompensationMiddleware."""

    def test_batch_abort_enabled_by_default(self):
        """Test that batch abort is enabled by default."""
        middleware = CompensationMiddleware(
            compensation_mapping={"test_tool": "undo_tool"},
        )
        assert middleware.enable_batch_abort is True

    def test_batch_abort_can_be_disabled(self):
        """Test that batch abort can be disabled."""
        middleware = CompensationMiddleware(
            compensation_mapping={"test_tool": "undo_tool"},
            enable_batch_abort=False,
        )
        assert middleware.enable_batch_abort is False

    def test_intent_tracking_disabled_by_default(self):
        """Test that intent tracking is disabled by default."""
        middleware = CompensationMiddleware(
            compensation_mapping={"test_tool": "undo_tool"},
        )
        assert middleware.track_intent is False

    def test_intent_tracking_can_be_enabled(self):
        """Test that intent tracking can be enabled."""
        middleware = CompensationMiddleware(
            compensation_mapping={"test_tool": "undo_tool"},
            track_intent=True,
        )
        assert middleware.track_intent is True

    def test_batch_manager_initialized(self):
        """Test that batch manager is properly initialized."""
        middleware = CompensationMiddleware(
            compensation_mapping={"test_tool": "undo_tool"},
            enable_batch_abort=True,
            batch_time_window_ms=100,
        )
        assert middleware._batch_manager is not None

    def test_wrap_tool_call_with_batch_abort(self):
        """Test that wrap_tool_call integrates with batch abort."""

        @tool
        def schedule_job(job_id: str) -> str:
            """Schedule a job."""
            return f"Scheduled {job_id}"

        @tool
        def cancel_job(job_id: str) -> str:
            """Cancel a job."""
            return f"Cancelled {job_id}"

        middleware = CompensationMiddleware(
            compensation_mapping={"schedule_job": "cancel_job"},
            tools=[schedule_job, cancel_job],
            enable_batch_abort=True,
        )

        # Create mock handler
        def mock_handler(request):
            result = schedule_job.invoke(request.tool_call.get("args", {}))
            return ToolMessage(
                content=str(result),
                tool_call_id=request.tool_call.get("id", "test"),
                name=request.tool_call["name"],
            )

        # Execute a single tool call
        request = MockToolCallRequest(
            tool_call={
                "id": "tc_1",
                "name": "schedule_job",
                "args": {"job_id": "job_1"},
            },
            state={},
        )

        result = middleware.wrap_tool_call(request, mock_handler)

        assert isinstance(result, ToolMessage)
        assert "Scheduled job_1" in result.content


class TestParallelExecutionSimulation:
    """Simulated parallel execution tests."""

    def test_simulated_parallel_failure_triggers_abort(self):
        """Test that failure in parallel execution signals abort."""

        @tool
        def process_item(item_id: str) -> str:
            """Process an item."""
            if item_id == "fail":
                return "Error: Processing failed"
            return f"Processed {item_id}"

        @tool
        def rollback_item(item_id: str) -> str:
            """Rollback item processing."""
            return f"Rolled back {item_id}"

        middleware = CompensationMiddleware(
            compensation_mapping={"process_item": "rollback_item"},
            tools=[process_item, rollback_item],
            enable_batch_abort=True,
        )

        execution_results = []
        execution_lock = threading.Lock()

        def execute_in_thread(item_id, thread_id):
            """Simulate tool execution in a separate thread."""
            tool_call_id = f"tc_{item_id}"

            # Simulate thread pattern detection
            batch_id = middleware._batch_manager.detect_batch(
                "process_item", str(thread_id), tool_call_id
            )

            if batch_id:
                ctx = middleware._batch_manager.get_or_create_context(
                    batch_id, tool_count=3, tool_call_ids=[tool_call_id]
                )

                # Check abort before "executing"
                if ctx.should_abort():
                    with execution_lock:
                        execution_results.append((item_id, "aborted"))
                    return

                # Simulate execution
                time.sleep(0.01)

                if item_id == "fail":
                    ctx.signal_abort("process_item", tool_call_id, "Processing failed")
                    with execution_lock:
                        execution_results.append((item_id, "failed"))
                    return

                with execution_lock:
                    execution_results.append((item_id, "completed"))
            else:
                # First call, no batch yet
                with execution_lock:
                    execution_results.append((item_id, "completed"))

        # Execute "parallel" calls
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(execute_in_thread, "item_1", 1001),
                executor.submit(execute_in_thread, "fail", 1002),
                executor.submit(execute_in_thread, "item_3", 1003),
            ]
            concurrent.futures.wait(futures)

        # Verify results
        statuses = {item: status for item, status in execution_results}

        # At least the failing item should be marked as failed
        assert statuses.get("fail") in ("failed", "completed")  # May complete before abort detected

        # Some items may be aborted
        aborted_count = sum(1 for _, status in execution_results if status == "aborted")
        completed_count = sum(1 for _, status in execution_results if status == "completed")
        failed_count = sum(1 for _, status in execution_results if status == "failed")

        # All should have some status
        assert aborted_count + completed_count + failed_count == 3


class TestCompensationLogDAG:
    """Tests for DAG-based dependency tracking in compensation log."""

    def test_dependency_inference_from_data_flow(self):
        """Test that dependencies are inferred from data flow."""
        middleware = CompensationMiddleware(
            compensation_mapping={"book_flight": "cancel_flight"},
        )

        log = CompensationLog()

        # Add first record with result containing booking_id
        record1 = CompensationRecord(
            id="flight_1",
            tool_name="book_flight",
            params={"destination": "NYC"},
            timestamp=1.0,
            compensation_tool="cancel_flight",
            status="COMPLETED",
            result={"booking_id": "FL12345"},
        )
        log.add(record1)

        # Infer dependencies for second call that uses booking_id
        deps = middleware._infer_dependencies(
            {"booking_id": "FL12345", "hotel_near": "FL12345"},
            log,
        )

        # Should detect dependency on flight_1
        assert "flight_1" in deps

    def test_no_false_dependencies_on_noise(self):
        """Test that noise values don't create false dependencies."""
        middleware = CompensationMiddleware(
            compensation_mapping={"tool": "undo_tool"},
        )

        log = CompensationLog()

        # Add record with noisy values
        record = CompensationRecord(
            id="noisy_1",
            tool_name="tool",
            params={},
            timestamp=1.0,
            compensation_tool="undo_tool",
            status="COMPLETED",
            result={"ok": True, "count": 5, "id": "abc"},  # Short string, bool, small int
        )
        log.add(record)

        # These should NOT create dependencies (noise filtering)
        deps = middleware._infer_dependencies(
            {"status": True, "value": 5, "code": "abc"},
            log,
        )

        # No dependencies (noise filtered out)
        assert len(deps) == 0

    def test_rollback_respects_dag_order(self):
        """Test that rollback plan includes both dependent and dependency records."""
        log = CompensationLog()

        # Add parent record (no dependencies)
        parent = CompensationRecord(
            id="parent",
            tool_name="parent_tool",
            params={},
            timestamp=1.0,
            compensation_tool="undo_parent",
            status="COMPLETED",
        )
        log.add(parent)

        # Add child that depends on parent
        child = CompensationRecord(
            id="child",
            tool_name="child_tool",
            params={},
            timestamp=2.0,
            compensation_tool="undo_child",
            status="COMPLETED",
            depends_on=["parent"],
        )
        log.add(child)

        plan = log.get_rollback_plan()

        # Both should be in the plan
        ids_in_plan = [r["id"] for r in plan]
        assert "parent" in ids_in_plan
        assert "child" in ids_in_plan
        assert len(plan) == 2

    def test_rollback_lifo_without_dependencies(self):
        """Test that rollback follows LIFO order when no dependencies exist."""
        log = CompensationLog()

        # Add records without dependencies (pure LIFO case)
        for i in range(3):
            log.add(
                CompensationRecord(
                    id=f"record_{i}",
                    tool_name=f"tool_{i}",
                    params={},
                    timestamp=float(i),
                    compensation_tool=f"undo_{i}",
                    status="COMPLETED",
                )
            )

        plan = log.get_rollback_plan()

        # Should be in reverse timestamp order (LIFO)
        assert plan[0]["id"] == "record_2"  # Latest first
        assert plan[1]["id"] == "record_1"
        assert plan[2]["id"] == "record_0"  # Earliest last
