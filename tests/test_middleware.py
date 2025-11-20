"""Basic tests for compensation middleware."""

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
