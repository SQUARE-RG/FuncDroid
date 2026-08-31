"""Asynchronous trace defect detection execution.

Design motivation:
- `trace_defect_detection.analyze_execution_for_bugs` only reads already-captured screenshots and execution results,
  without touching the device; meanwhile, the next round of `scenario_plan`/`scenario_exec` needs to immediately use the device to continue exploration.
  Moving defect detection to an independent background thread allows overlapping the time spent on "device operations" and "LLM visual review",
  thereby reducing overall execution time without altering the defect detection semantics themselves.
- Defect detection results (detected_bugs/bug_trace_records) do not participate in scenario_plan/ffg_update decisions,
  and are only used for persisted reports; thus, async execution does not affect the correctness of the main loop.
- A single background worker thread sequentially consumes the queue instead of thread pool concurrency: deduplication of detected_bugs relies
  on the "confirmed defect list from the previous round". Concurrent writes across multiple detection requests would cause merge race conditions; sequential processing by a single thread naturally
  avoids this problem and keeps the implementation simple.

Usage:
    from workflow.bug_detection_worker import bug_detection_worker

    bug_detection_worker.configure(bug_detection_llm, on_batch_done=save_callback)
    bug_detection_worker.submit(job_state)   # Returns immediately without blocking the main loop
    ...
    bug_detection_worker.flush()             # Wait for the queue to drain before workflow completion
    detected_bugs, bug_trace_records = bug_detection_worker.snapshot()
"""

from __future__ import annotations

import queue
import os
import threading
import time
import traceback
from typing import Any, Callable

from workflow.state import AgentState
from workflow.trace_defect_detection import analyze_execution_for_bugs


DEFAULT_MAX_PENDING = max(1, int(os.getenv("BUG_DETECTION_MAX_PENDING", "8")))
DEFAULT_SUBMIT_WAIT_SECONDS = float(
    os.getenv("BUG_DETECTION_SUBMIT_WAIT_SECONDS", "0") or 0
)


class BugDetectionWorker:
    """Single background thread + queue, sequentially executing `analyze_execution_for_bugs`."""

    def __init__(self, max_pending: int = DEFAULT_MAX_PENDING) -> None:
        self._lock = threading.Lock()
        self._queue: "queue.Queue[dict[str, Any] | object]" = queue.Queue()
        self._thread: threading.Thread | None = None
        self._llm = None
        self._on_batch_done: Callable[[], None] | None = None
        self._on_job_failed: Callable[[BaseException], None] | None = None
        self._detected_bugs: list[dict[str, Any]] = []
        self._bug_trace_records: dict[str, list[dict[str, Any]]] = {}
        # Number of pending tasks, combined with Condition to implement flush() with timeout.
        self._pending = 0
        self._idle_cond = threading.Condition(self._lock)
        self._max_pending = max(1, int(max_pending))

    # ------------------------------------------------------------------
    # Configuration / Lifecycle
    # ------------------------------------------------------------------
    def configure(
        self,
        llm,
        *,
        on_batch_done: Callable[[], None] | None = None,
        on_job_failed: Callable[[BaseException], None] | None = None,
        initial_bugs: list[dict[str, Any]] | None = None,
        initial_records: dict[str, list[dict[str, Any]]] | None = None,
        max_pending: int | None = None,
    ) -> None:
        """Configure the LLM and persistence callback for this run, and (re)seed existing defects (for resume-from-breakpoint scenarios)."""
        with self._lock:
            self._llm = llm
            self._on_batch_done = on_batch_done
            self._on_job_failed = on_job_failed
            self._detected_bugs = list(initial_bugs or [])
            self._bug_trace_records = dict(initial_records or {})
            if max_pending is not None:
                self._max_pending = max(1, int(max_pending))
        self._ensure_worker_started()

    def _ensure_worker_started(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            thread = threading.Thread(
                target=self._worker_loop,
                name="bug-detection-worker",
                daemon=True,
            )
            self._thread = thread
        thread.start()

    # ------------------------------------------------------------------
    # Submit task
    # ------------------------------------------------------------------
    def submit(self, job_state: dict[str, Any]) -> bool:
        """Submit without dropping evidence; apply backpressure at the memory cap.

        ``BUG_DETECTION_SUBMIT_WAIT_SECONDS=0`` (default) waits until a queue
        slot is available. A positive value permits an explicit bounded wait;
        a negative value restores the old non-blocking/drop behavior.
        """
        self._ensure_worker_started()
        wait_seconds = DEFAULT_SUBMIT_WAIT_SECONDS
        deadline = time.monotonic() + wait_seconds if wait_seconds > 0 else None
        with self._idle_cond:
            announced = False
            while self._pending >= self._max_pending:
                if not announced:
                    print(
                        "[Bug Detection Thread] Pending tasks reached memory limit, waiting for available slot to preserve this review: "
                        f"pending={self._pending}, max_pending={self._max_pending}"
                    )
                    announced = True
                if wait_seconds < 0:
                    return False
                if deadline is None:
                    self._idle_cond.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    print("[Bug Detection Thread] Timed out waiting for available slot; this review was not enqueued.")
                    return False
                self._idle_cond.wait(remaining)
            self._pending += 1
            pending = self._pending
        self._queue.put(job_state)
        print(
            "[Bug Detection Thread] Submitted asynchronous detection task: "
            f"pending={pending}, queue_size={self._queue.qsize()}"
        )
        return True

    # ------------------------------------------------------------------
    # Background thread main loop
    # ------------------------------------------------------------------
    def _worker_loop(self) -> None:
        while True:
            job_state = self._queue.get()
            try:
                self._process_job(job_state)
            except Exception as exc:  # noqa: BLE001 - The background thread needs to swallow exceptions and continue consuming
                print(f"[Bug Detection Thread] Failed to process task, retaining existing defect results: {exc}")
                print(traceback.format_exc())
                callback = self._on_job_failed
                if callback is not None:
                    try:
                        callback(exc)
                    except Exception as callback_exc:  # noqa: BLE001
                        print(f"[Bug Detection Thread] Failure callback itself failed: {callback_exc}")
            finally:
                with self._idle_cond:
                    self._pending = max(0, self._pending - 1)
                    # Wake both flush waiters and producers blocked by the
                    # pending-memory cap whenever one slot becomes available.
                    self._idle_cond.notify_all()

    def _process_job(self, job_state: dict[str, Any]) -> None:
        with self._lock:
            llm = self._llm
            # Overwrite old values in the task snapshot with the current authoritative defect list to ensure sequential merging without losing updates.
            job_state["detected_bugs"] = list(self._detected_bugs)
            job_state["bug_trace_records"] = dict(self._bug_trace_records)
        updates = analyze_execution_for_bugs(llm, job_state)  # type: ignore[arg-type]
        callback: Callable[[], None] | None = None
        with self._lock:
            if isinstance(updates.get("detected_bugs"), list):
                self._detected_bugs = updates["detected_bugs"]
            if isinstance(updates.get("bug_trace_records"), dict):
                self._bug_trace_records = updates["bug_trace_records"]
            callback = self._on_batch_done
        if callback is not None:
            try:
                callback()
            except Exception as exc:  # noqa: BLE001
                print(f"[Bug Detection Thread] Persistence callback failed: {exc}")

    # ------------------------------------------------------------------
    # Read / Wait
    # ------------------------------------------------------------------
    def snapshot(self) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
        """Thread-safely read a snapshot of the current authoritative defect list and trace records."""
        with self._lock:
            return list(self._detected_bugs), dict(self._bug_trace_records)

    def pending_count(self) -> int:
        with self._lock:
            return self._pending

    def flush(self, timeout: float | None = None) -> bool:
        """Wait until all submitted tasks in the queue are processed.

        Returns:
            True indicates completion before timeout; when timeout=None, it blocks until completion.
        """
        with self._idle_cond:
            if timeout is None:
                while self._pending > 0:
                    self._idle_cond.wait()
                return True
            deadline = time.monotonic() + timeout
            while self._pending > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._idle_cond.wait(remaining)
            return True

    def reset(self) -> None:
        """Reset to the unconfigured initial state; for test isolation only, not intended for production calls."""
        with self._lock:
            self._llm = None
            self._on_batch_done = None
            self._on_job_failed = None
            self._detected_bugs = []
            self._bug_trace_records = {}
            self._max_pending = DEFAULT_MAX_PENDING


def build_defect_detection_job(state: AgentState) -> dict[str, Any]:
    """Extract a read-only snapshot required for defect detection from workflow state to avoid sharing mutable references.

    Only copies fields that `analyze_execution_for_bugs` reads; elements in trace (dict/Page)
    are not modified in-place by other workflow nodes once generated, so shallow copying lists/dicts is sufficient to prevent subsequent state updates
    from affecting submitted tasks.
    """
    navigation_trace = state.get("navigation_execution_trace", [])
    scenario_trace = state.get("scenario_execution_trace", [])
    return {
        "navigation_execution_trace": list(navigation_trace) if isinstance(navigation_trace, list) else [],
        "scenario_execution_trace": list(scenario_trace) if isinstance(scenario_trace, list) else [],
        "execution_result": dict(state.get("execution_result", {}) or {}),
        "current_plan": dict(state.get("current_plan", {}) or {}),
        "current_navigation_scenario": str(state.get("current_navigation_scenario", "") or ""),
        "current_test_scenario": str(state.get("current_test_scenario", "") or ""),
        "current_testing_function_id": str(state.get("current_testing_function_id", "") or ""),
        "package_name": str(state.get("package_name", "") or ""),
    }


# In-process singleton, semantically consistent with the global usage registry in `workflow/llm_usage.py`.
bug_detection_worker = BugDetectionWorker()
