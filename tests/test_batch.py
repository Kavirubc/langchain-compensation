"""Tests for batch execution handling module."""

import concurrent.futures
import threading
import time

import pytest

from langchain_compensation.batch import (
    BatchContext,
    BatchDetector,
    BatchManager,
    IntentDAG,
    IntentNode,
)


class TestBatchContext:
    """Tests for BatchContext class."""

    def test_initial_state(self):
        """Test that batch context starts with correct initial state."""
        ctx = BatchContext(
            batch_id="test_batch",
            tool_count=5,
            tool_call_ids=["tc1", "tc2", "tc3", "tc4", "tc5"],
        )

        assert ctx.batch_id == "test_batch"
        assert ctx.tool_count == 5
        assert len(ctx.tool_call_ids) == 5
        assert not ctx.should_abort()
        assert ctx.abort_reason is None
        assert ctx.failed_tool is None
        assert not ctx.is_complete()

    def test_signal_abort(self):
        """Test that abort signal is set correctly."""
        ctx = BatchContext(batch_id="test", tool_count=3, tool_call_ids=["1", "2", "3"])

        # First abort signal should succeed
        result = ctx.signal_abort("tool_1", "tc_1", "Test failure")
        assert result is True
        assert ctx.should_abort()
        assert ctx.abort_reason == "Test failure"
        assert ctx.failed_tool == "tool_1"
        assert ctx.failed_tool_call_id == "tc_1"

        # Second abort signal should be no-op
        result = ctx.signal_abort("tool_2", "tc_2", "Another failure")
        assert result is False
        assert ctx.abort_reason == "Test failure"  # Still first failure
        assert ctx.failed_tool == "tool_1"

    def test_should_abort_lock_free(self):
        """Test that should_abort() is fast (lock-free read)."""
        ctx = BatchContext(batch_id="test", tool_count=5, tool_call_ids=[])

        # should_abort should be very fast
        start = time.time()
        for _ in range(100000):
            ctx.should_abort()
        elapsed = time.time() - start

        # Should complete in under 100ms for 100k calls
        assert elapsed < 0.1, f"should_abort() too slow: {elapsed}s for 100k calls"

    def test_record_execution(self):
        """Test execution tracking."""
        ctx = BatchContext(batch_id="test", tool_count=3, tool_call_ids=["1", "2", "3"])

        assert ctx.record_execution("tc_1") == 1
        assert ctx.record_execution("tc_2") == 2
        assert not ctx.is_complete()
        assert ctx.record_execution("tc_3") == 3
        assert ctx.is_complete()

    def test_get_orphan_executions(self):
        """Test orphan execution detection."""
        ctx = BatchContext(batch_id="test", tool_count=3, tool_call_ids=["1", "2", "3"])

        # Record some executions
        ctx.record_execution("tc_1")
        ctx.record_execution("tc_2")

        # Signal abort on tc_2
        ctx.signal_abort("tool_2", "tc_2", "Failed")

        # tc_3 executes after abort (orphan)
        ctx.record_execution("tc_3")

        orphans = ctx.get_orphan_executions()

        # tc_1 and tc_3 are orphans (completed but not the failed one)
        assert "tc_1" in orphans
        assert "tc_3" in orphans
        assert "tc_2" not in orphans  # tc_2 is the failed one

    def test_thread_safety(self):
        """Test that batch context is thread-safe under concurrent access."""
        ctx = BatchContext(batch_id="test", tool_count=100, tool_call_ids=[])
        abort_count = [0]

        def worker(tool_id):
            # Simulate some work
            time.sleep(0.001)

            # Check abort flag
            if ctx.should_abort():
                return "aborted"

            # 10% chance to fail
            if tool_id % 10 == 5:
                if ctx.signal_abort(f"tool_{tool_id}", f"tc_{tool_id}", "Simulated fail"):
                    abort_count[0] += 1
                return "failed"

            ctx.record_execution(f"tc_{tool_id}")
            return "completed"

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(worker, i) for i in range(100)]
            results = [f.result() for f in futures]

        # Verify consistency
        assert abort_count[0] == 1  # Only one abort should succeed
        aborted_count = results.count("aborted")
        failed_count = results.count("failed")
        completed_count = results.count("completed")

        # At least some should be aborted
        assert aborted_count >= 0
        assert failed_count >= 1
        assert completed_count >= 0
        assert aborted_count + failed_count + completed_count == 100


class TestIntentNode:
    """Tests for IntentNode class."""

    def test_valid_transitions(self):
        """Test valid status transitions."""
        node = IntentNode(tool_call_id="tc_1", tool_name="test_tool", args={})

        assert node.status == "PENDING"

        # PENDING -> EXECUTING
        assert node.transition("EXECUTING") is True
        assert node.status == "EXECUTING"

        # EXECUTING -> COMPLETED
        assert node.transition("COMPLETED") is True
        assert node.status == "COMPLETED"

    def test_invalid_transitions(self):
        """Test invalid status transitions are rejected."""
        node = IntentNode(tool_call_id="tc_1", tool_name="test_tool", args={})

        # PENDING -> COMPLETED (invalid, must go through EXECUTING)
        assert node.transition("COMPLETED") is False
        assert node.status == "PENDING"

        # PENDING -> FAILED (invalid)
        assert node.transition("FAILED") is False
        assert node.status == "PENDING"

    def test_abort_from_pending(self):
        """Test that PENDING can transition to ABORTED."""
        node = IntentNode(tool_call_id="tc_1", tool_name="test_tool", args={})

        assert node.transition("ABORTED") is True
        assert node.status == "ABORTED"

    def test_abort_from_executing(self):
        """Test that EXECUTING can transition to ABORTED."""
        node = IntentNode(tool_call_id="tc_1", tool_name="test_tool", args={})

        node.transition("EXECUTING")
        assert node.transition("ABORTED") is True
        assert node.status == "ABORTED"


class TestIntentDAG:
    """Tests for IntentDAG class."""

    def test_initialization(self):
        """Test DAG is initialized correctly from tool calls."""
        dag = IntentDAG(
            "batch_1",
            [
                {"id": "tc_1", "name": "tool_1", "args": {"a": 1}},
                {"id": "tc_2", "name": "tool_2", "args": {"b": 2}},
            ],
        )

        assert dag.batch_id == "batch_1"
        assert len(dag.nodes) == 2
        assert "tc_1" in dag.nodes
        assert "tc_2" in dag.nodes

    def test_status_tracking(self):
        """Test status tracking through DAG methods."""
        dag = IntentDAG(
            "batch_1",
            [
                {"id": "tc_1", "name": "tool_1", "args": {}},
                {"id": "tc_2", "name": "tool_2", "args": {}},
            ],
        )

        # Mark tc_1 as executing then completed
        assert dag.mark_executing("tc_1") is True
        assert dag.mark_completed("tc_1") is True

        # Mark tc_2 as executing then failed
        assert dag.mark_executing("tc_2") is True
        assert dag.mark_failed("tc_2") is True

        report = dag.get_report()
        assert report["completed"] == ["tc_1"]
        assert report["failed"] == ["tc_2"]

    def test_abort_pending(self):
        """Test aborting all pending nodes."""
        dag = IntentDAG(
            "batch_1",
            [
                {"id": "tc_1", "name": "tool_1", "args": {}},
                {"id": "tc_2", "name": "tool_2", "args": {}},
                {"id": "tc_3", "name": "tool_3", "args": {}},
            ],
        )

        # Only tc_1 starts executing
        dag.mark_executing("tc_1")

        # Abort pending
        aborted = dag.abort_pending()

        assert "tc_2" in aborted
        assert "tc_3" in aborted
        assert "tc_1" not in aborted  # tc_1 was EXECUTING, not PENDING

    def test_get_report(self):
        """Test report generation."""
        dag = IntentDAG(
            "batch_1",
            [
                {"id": "tc_1", "name": "tool_1", "args": {}},
                {"id": "tc_2", "name": "tool_2", "args": {}},
                {"id": "tc_3", "name": "tool_3", "args": {}},
            ],
        )

        dag.mark_executing("tc_1")
        dag.mark_completed("tc_1")
        dag.mark_executing("tc_2")
        dag.mark_failed("tc_2")
        dag.mark_aborted("tc_3")

        report = dag.get_report()

        assert report["batch_id"] == "batch_1"
        assert report["total_tools"] == 3
        assert report["status_counts"]["COMPLETED"] == 1
        assert report["status_counts"]["FAILED"] == 1
        assert report["status_counts"]["ABORTED"] == 1


class TestBatchDetector:
    """Tests for BatchDetector class."""

    def test_single_call_no_batch(self):
        """Test that single calls don't create batches."""
        detector = BatchDetector(time_window_ms=50)

        batch_id = detector.record_call("tool_1", "thread_1", "tc_1")
        assert batch_id is None  # No batch for single call

    def test_parallel_calls_create_batch(self):
        """Test that parallel calls from different threads create a batch."""
        detector = BatchDetector(time_window_ms=100)

        # Simulate calls from different threads within time window
        batch_id_1 = detector.record_call("tool_1", "thread_1", "tc_1")
        batch_id_2 = detector.record_call("tool_1", "thread_2", "tc_2")
        batch_id_3 = detector.record_call("tool_1", "thread_3", "tc_3")

        # First call returns None, subsequent calls return batch_id
        assert batch_id_1 is None
        assert batch_id_2 is not None
        assert batch_id_3 is not None
        # Note: batch IDs may differ if they're calculated at different timestamps,
        # but both should be non-None indicating parallel batch detection

    def test_different_tools_different_batches(self):
        """Test that different tool names create separate batches."""
        detector = BatchDetector(time_window_ms=100)

        # tool_1 batch
        detector.record_call("tool_1", "thread_1", "tc_1")
        batch_1 = detector.record_call("tool_1", "thread_2", "tc_2")

        # tool_2 batch
        detector.record_call("tool_2", "thread_3", "tc_3")
        batch_2 = detector.record_call("tool_2", "thread_4", "tc_4")

        assert batch_1 is not None
        assert batch_2 is not None
        assert batch_1 != batch_2  # Different batches

    def test_expired_calls_not_batched(self):
        """Test that calls outside time window don't batch."""
        detector = BatchDetector(time_window_ms=10)

        detector.record_call("tool_1", "thread_1", "tc_1")
        time.sleep(0.02)  # Wait for expiry
        batch_id = detector.record_call("tool_1", "thread_2", "tc_2")

        assert batch_id is None  # First call expired, so this is "first" again

    def test_get_batch_tool_calls(self):
        """Test retrieving tool calls for a batch."""
        detector = BatchDetector(time_window_ms=100)

        detector.record_call("tool_1", "thread_1", "tc_1")
        batch_id = detector.record_call("tool_1", "thread_2", "tc_2")

        # Get tool calls for this batch
        tool_calls = detector.get_batch_tool_calls(batch_id)

        # At minimum, tc_1 and tc_2 should be in the batch
        assert "tc_1" in tool_calls
        assert "tc_2" in tool_calls

    def test_cleanup_batch(self):
        """Test batch cleanup."""
        detector = BatchDetector(time_window_ms=100)

        detector.record_call("tool_1", "thread_1", "tc_1")
        batch_id = detector.record_call("tool_1", "thread_2", "tc_2")

        detector.cleanup_batch(batch_id)

        assert detector.get_batch_tool_calls(batch_id) == []


class TestBatchManager:
    """Tests for BatchManager class."""

    def test_detect_batch(self):
        """Test batch detection through manager."""
        manager = BatchManager(time_window_ms=100)

        batch_id_1 = manager.detect_batch("tool_1", "thread_1", "tc_1")
        batch_id_2 = manager.detect_batch("tool_1", "thread_2", "tc_2")

        assert batch_id_1 is None
        assert batch_id_2 is not None

    def test_get_or_create_context(self):
        """Test context creation and retrieval."""
        manager = BatchManager()

        ctx1 = manager.get_or_create_context("batch_1", tool_count=5)
        ctx2 = manager.get_or_create_context("batch_1", tool_count=10)  # Ignored

        assert ctx1 is ctx2
        assert ctx1.tool_count == 5  # Original value preserved

    def test_get_context_nonexistent(self):
        """Test getting non-existent context returns None."""
        manager = BatchManager()

        ctx = manager.get_context("nonexistent")
        assert ctx is None

    def test_intent_dag_tracking_disabled(self):
        """Test that intent DAG is not created when tracking disabled."""
        manager = BatchManager(track_intent=False)

        dag = manager.create_intent_dag("batch_1", [{"id": "tc_1", "name": "tool_1", "args": {}}])
        assert dag is None

    def test_intent_dag_tracking_enabled(self):
        """Test that intent DAG is created when tracking enabled."""
        manager = BatchManager(track_intent=True)

        dag = manager.create_intent_dag("batch_1", [{"id": "tc_1", "name": "tool_1", "args": {}}])
        assert dag is not None
        assert "tc_1" in dag.nodes

    def test_cleanup_batch(self):
        """Test batch cleanup returns report and clears state."""
        manager = BatchManager(track_intent=True)

        # Create context and DAG
        manager.get_or_create_context("batch_1", tool_count=2, tool_call_ids=["tc_1", "tc_2"])
        dag = manager.create_intent_dag("batch_1", [
            {"id": "tc_1", "name": "tool_1", "args": {}},
            {"id": "tc_2", "name": "tool_2", "args": {}},
        ])
        dag.mark_executing("tc_1")
        dag.mark_completed("tc_1")

        # Cleanup
        report = manager.cleanup_batch("batch_1")

        assert report is not None
        assert report["completed"] == ["tc_1"]

        # Context should be cleaned up
        assert manager.get_context("batch_1") is None
        assert manager.get_intent_dag("batch_1") is None

    def test_active_batch_count(self):
        """Test active batch counting."""
        manager = BatchManager()

        assert manager.get_active_batch_count() == 0

        manager.get_or_create_context("batch_1", tool_count=5)
        assert manager.get_active_batch_count() == 1

        manager.get_or_create_context("batch_2", tool_count=3)
        assert manager.get_active_batch_count() == 2

        manager.cleanup_batch("batch_1")
        assert manager.get_active_batch_count() == 1


class TestParallelRaceCondition:
    """Integration tests simulating parallel execution race conditions."""

    def test_abort_prevents_subsequent_executions(self):
        """Test that abort flag prevents tools from executing after failure."""
        ctx = BatchContext(batch_id="test", tool_count=5, tool_call_ids=[])
        execution_order = []
        execution_lock = threading.Lock()

        def execute_tool(tool_id, should_fail=False):
            # Check abort before executing
            if ctx.should_abort():
                with execution_lock:
                    execution_order.append((tool_id, "aborted"))
                return "aborted"

            # Simulate execution time
            time.sleep(0.01)

            if should_fail:
                ctx.signal_abort(f"tool_{tool_id}", f"tc_{tool_id}", "Simulated failure")
                with execution_lock:
                    execution_order.append((tool_id, "failed"))
                return "failed"

            ctx.record_execution(f"tc_{tool_id}")
            with execution_lock:
                execution_order.append((tool_id, "completed"))
            return "completed"

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # Submit 5 tools, tool 1 will fail
            futures = [
                executor.submit(execute_tool, 0),
                executor.submit(execute_tool, 1, should_fail=True),
                executor.submit(execute_tool, 2),
                executor.submit(execute_tool, 3),
                executor.submit(execute_tool, 4),
            ]
            results = [f.result() for f in futures]

        # Verify batch was aborted
        assert ctx.should_abort()

        # Count results
        completed = results.count("completed")
        aborted = results.count("aborted")
        failed = results.count("failed")

        assert failed == 1  # Only tool 1 should fail
        assert aborted >= 0  # Some tools might be aborted
        assert completed + aborted + failed == 5

    def test_intent_dag_captures_execution_state(self):
        """Test that intent DAG correctly captures execution state during race."""
        manager = BatchManager(track_intent=True)

        ctx = manager.get_or_create_context("test_batch", tool_count=3)
        tool_calls = [
            {"id": "tc_0", "name": "tool", "args": {}},
            {"id": "tc_1", "name": "tool", "args": {}},
            {"id": "tc_2", "name": "tool", "args": {}},
        ]
        dag = manager.create_intent_dag("test_batch", tool_calls)

        def execute_tool(tool_id, should_fail=False):
            tool_call_id = f"tc_{tool_id}"
            dag.mark_executing(tool_call_id)

            if ctx.should_abort():
                dag.mark_aborted(tool_call_id)
                return "aborted"

            time.sleep(0.01)

            if should_fail:
                ctx.signal_abort("tool", tool_call_id, "Failed")
                dag.mark_failed(tool_call_id)
                dag.abort_pending()
                return "failed"

            dag.mark_completed(tool_call_id)
            ctx.record_execution(tool_call_id)
            return "completed"

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(execute_tool, 0),
                executor.submit(execute_tool, 1, should_fail=True),
                executor.submit(execute_tool, 2),
            ]
            [f.result() for f in futures]

        report = dag.get_report()

        # Verify all tools have terminal status
        assert report["status_counts"].get("PENDING", 0) == 0
        assert report["status_counts"].get("EXECUTING", 0) == 0

        # At least one failed
        assert len(report["failed"]) >= 1
