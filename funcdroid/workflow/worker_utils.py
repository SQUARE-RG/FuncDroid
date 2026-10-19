"""Shared serial background-worker queue.

Reviewers consume previously captured screenshots without device access, overlapping model calls with GUI execution. A single consumer serializes result merging."""

from __future__ import annotations

import os
import queue
import threading
import time
import traceback
import uuid
from typing import Any, Callable

# Bound queue-slot waiting to avoid blocking the main loop indefinitely; nonpositive values disable the limit.
SUBMIT_WAIT_TIMEOUT_SECONDS = float(os.getenv("WORKER_SUBMIT_WAIT_TIMEOUT_SECONDS", "600") or 600)
# Check worker liveness between bounded waits and restart it if needed.
_SUBMIT_WAIT_SLICE_SECONDS = 1.0
UNFINISHED_ERROR_TYPE = "ReviewUnfinished"


class SubmitTimeoutError(TimeoutError):
    """The queue remained full until the submission timeout."""


def build_trace_job(state: dict[str, Any], *extra_fields: str, with_trace_id: bool = False) -> dict[str, Any]:
    """Capture a read-only review-job snapshot from workflow state.

    Trace elements are immutable after creation, so copying the list is sufficient. Extra fields are copied as strings."""
    navigation_trace = state.get("navigation_execution_trace", [])
    scenario_trace = state.get("scenario_execution_trace", [])
    job: dict[str, Any] = {}
    if with_trace_id:
        job["trace_id"] = uuid.uuid4().hex
    job.update({
        "iteration_count": int(state.get("iteration_count", 0) or 0),
        "navigation_execution_trace": list(navigation_trace) if isinstance(navigation_trace, list) else [],
        "scenario_execution_trace": list(scenario_trace) if isinstance(scenario_trace, list) else [],
        "execution_result": dict(state.get("execution_result", {}) or {}),
        "current_plan": dict(state.get("current_plan", {}) or {}),
    })
    for field in extra_fields:
        job[field] = str(state.get(field, "") or "")
    return job


class SerialWorker:
    """One background consumer with bounded pending count; subclasses implement _process_job."""

    log_prefix = "[Worker]"
    thread_name = "serial-worker"
    default_max_pending = 8
    # Drop and count full-queue submissions only when the work can be retried later.
    drop_when_full = False

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._idle_cond = threading.Condition(self._lock)
        self._queue: "queue.Queue[dict[str, Any]]" = queue.Queue()
        self._thread: threading.Thread | None = None
        self._llm = None
        self._on_batch_done: Callable[[], None] | None = None
        self._failures: list[dict[str, Any]] = []
        self._pending = 0
        self._rejected_count = 0
        self._max_pending = self.default_max_pending
        # Track all pending jobs, including the active job, for timeout reporting.
        self._outstanding: dict[int, dict[str, Any]] = {}
        self._timed_out_ids: set[int] = set()
        self._next_job_id = 0

    def _configure_base(
        self,
        llm,
        on_batch_done: Callable[[], None] | None = None,
        initial_failures: list[dict[str, Any]] | None = None,
    ) -> None:
        with self._lock:
            self._llm = llm
            self._on_batch_done = on_batch_done
            self._failures = list(initial_failures or [])
        self._ensure_worker_started()

    def _ensure_worker_started(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            thread = threading.Thread(target=self._worker_loop, name=self.thread_name, daemon=True)
            self._thread = thread
        thread.start()

    def submit(self, job: dict[str, Any], timeout: float | None = None) -> bool:
        """Submit a job, returning True only after it enters the queue.

        When full, either drop immediately or wait in bounded intervals, restarting a dead worker as needed. Record failure if the overall submission timeout expires."""
        self._ensure_worker_started()
        wait_limit = SUBMIT_WAIT_TIMEOUT_SECONDS if timeout is None else float(timeout)
        deadline = time.monotonic() + wait_limit if wait_limit > 0 else None
        timed_out = False
        with self._idle_cond:
            announced = False
            while self._pending >= self._max_pending:
                if self.drop_when_full:
                    self._rejected_count += 1
                    print(
                        f"{self.log_prefix} Queue is full; submission dropped: "
                        f"pending={self._pending}, total_dropped={self._rejected_count}"
                    )
                    return False
                if not announced:
                    print(f"{self.log_prefix} Pending limit reached; waiting for queue space: pending={self._pending}")
                    announced = True
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    timed_out = True
                    break
                slice_seconds = _SUBMIT_WAIT_SLICE_SECONDS if remaining is None else min(
                    _SUBMIT_WAIT_SLICE_SECONDS, remaining
                )
                self._idle_cond.wait(slice_seconds)
                if self._thread is None or not self._thread.is_alive():
                    # Restart a dead consumer outside the lock, then continue waiting.
                    self._idle_cond.release()
                    try:
                        print(f"{self.log_prefix} Worker stopped; restarting queue consumer")
                        self._ensure_worker_started()
                    finally:
                        self._idle_cond.acquire()
            if not timed_out:
                self._pending += 1
                pending = self._pending
                self._next_job_id += 1
                job_id = self._next_job_id
                self._outstanding[job_id] = job
        if timed_out:
            print(f"{self.log_prefix} Timed out waiting for queue space ({wait_limit:.0f}s); recording submission failure")
            self.record_failure(job, SubmitTimeoutError(f"Timed out waiting for review queue space ({wait_limit:.0f}s)"))
            return False
        self._queue.put((job_id, job))
        print(f"{self.log_prefix} Submitted background job: pending={pending}")
        return True

    def _worker_loop(self) -> None:
        while True:
            job_id, job = self._queue.get()
            try:
                self._process_job(job)
            except Exception as exc:  # noqa: BLE001 - Keep consuming after an individual job exception.
                print(f"{self.log_prefix} Job processing failed: {exc}")
                print(traceback.format_exc())
                self.record_failure(job, exc)
            finally:
                cleared = False
                with self._idle_cond:
                    self._pending = max(0, self._pending - 1)
                    self._outstanding.pop(job_id, None)
                    if job_id in self._timed_out_ids:
                        self._timed_out_ids.discard(job_id)
                        # Remove an unfinished marker after the job finishes, whether successfully or with a recorded failure.
                        before = len(self._failures)
                        self._failures = [
                            failure for failure in self._failures
                            if not (
                                failure.get("error_type") == UNFINISHED_ERROR_TYPE
                                and failure.get("job_id") == job_id
                            )
                        ]
                        cleared = len(self._failures) != before
                    self._idle_cond.notify_all()
                if cleared:
                    self._batch_done()

    def _process_job(self, job: dict[str, Any]) -> None:
        raise NotImplementedError

    def _batch_done(self) -> None:
        with self._lock:
            callback = self._on_batch_done
        if callback is not None:
            try:
                callback()
            except Exception as exc:  # noqa: BLE001
                print(f"{self.log_prefix} Persistence callback failed: {exc}")

    def _failure_record(self, job: dict[str, Any], exc: BaseException) -> dict[str, Any]:
        return {
            "iteration_count": job.get("iteration_count", 0),
            "scenario": job.get("current_test_scenario", ""),
            "plan": dict(job.get("current_plan", {}) or {}),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    def _on_failure_locked(self, job: dict[str, Any], exc: BaseException) -> None:
        """Subclass hook to update result state under the lock."""

    def _on_unfinished_locked(self, job: dict[str, Any], exc: BaseException) -> None:
        """Subclass hook for jobs unfinished at flush timeout; defaults to ordinary failure handling."""
        self._on_failure_locked(job, exc)

    def record_failure(self, job: dict[str, Any], exc: BaseException) -> None:
        """Record a failed or rejected review independently of queue drainage."""
        job = job if isinstance(job, dict) else {}
        with self._lock:
            self._failures.append(self._failure_record(job, exc))
            self._on_failure_locked(job, exc)
        self._batch_done()

    def failures_snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(failure) for failure in self._failures]

    def pending_count(self) -> int:
        with self._lock:
            return self._pending

    def rejected_count(self) -> int:
        with self._lock:
            return self._rejected_count

    def flush(self, timeout: float | None = None) -> bool:
        """Wait for submitted jobs; return False on timeout.

        None waits indefinitely; nonpositive timeout only checks whether the worker is idle."""
        with self._idle_cond:
            if timeout is not None and timeout <= 0:
                return self._pending <= 0
            deadline = None if timeout is None else time.monotonic() + timeout
            while self._pending > 0:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._idle_cond.wait(remaining)
            return True

    def record_unfinished(self, reason: str) -> int:
        """Record unfinished jobs once for traceable reporting.

        A later completion still merges its result and removes the unfinished marker by job_id."""
        with self._lock:
            unfinished = [
                (job_id, job) for job_id, job in self._outstanding.items()
                if job_id not in self._timed_out_ids
            ]
            for job_id, job in unfinished:
                self._timed_out_ids.add(job_id)
                exc = TimeoutError(reason)
                self._failures.append({
                    **self._failure_record(job, exc),
                    "error_type": UNFINISHED_ERROR_TYPE,
                    "job_id": job_id,
                })
                self._on_unfinished_locked(job, exc)
        if unfinished:
            self._batch_done()
        return len(unfinished)

    def reset(self) -> None:
        """Restore unconfigured initial state for isolated tests."""
        with self._lock:
            self._llm = None
            self._on_batch_done = None
            self._failures = []
            self._pending = 0
            self._rejected_count = 0
            self._max_pending = self.default_max_pending
            self._outstanding = {}
            self._timed_out_ids = set()
            self._next_job_id = 0
            # Discard queued jobs; active jobs later clamp the pending count at zero.
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            self._idle_cond.notify_all()
