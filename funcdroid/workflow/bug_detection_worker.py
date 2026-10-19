"""Asynchronous execution-trace bug detection.

Findings are persisted for reports and do not drive planning or FFG updating. One worker serially merges and deduplicates findings. Configure, submit jobs, flush before shutdown, and read snapshot results."""

from __future__ import annotations

from typing import Any, Callable

from workflow.state import AgentState
from workflow.trace_defect_detection import analyze_execution_for_bugs, copy_bug_record
from workflow.worker_utils import SerialWorker, build_trace_job


def _copy_bugs(bugs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [copy_bug_record(bug) if isinstance(bug, dict) else bug for bug in bugs]


def _copy_records(records: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    # Copy the trace list while sharing immutable step records.
    return {
        key: list(value) if isinstance(value, list) else value
        for key, value in records.items()
    }


class BugDetectionWorker(SerialWorker):
    """Run analyze_execution_for_bugs serially."""

    log_prefix = "[Bug detection worker]"
    thread_name = "bug-detection-worker"

    def __init__(self) -> None:
        super().__init__()
        self._detected_bugs: list[dict[str, Any]] = []
        self._bug_trace_records: dict[str, list[dict[str, Any]]] = {}

    def configure(
        self,
        llm,
        *,
        on_batch_done: Callable[[], None] | None = None,
        initial_bugs: list[dict[str, Any]] | None = None,
        initial_records: dict[str, list[dict[str, Any]]] | None = None,
        initial_failures: list[dict[str, Any]] | None = None,
    ) -> None:
        """Configure the model, save callback, and existing findings for a new or resumed run."""
        with self._lock:
            self._detected_bugs = _copy_bugs(list(initial_bugs or []))
            self._bug_trace_records = _copy_records(dict(initial_records or {}))
        self._configure_base(llm, on_batch_done, initial_failures)

    def _process_job(self, job_state: dict[str, Any]) -> None:
        with self._lock:
            llm = self._llm
            # Replace stale snapshot findings with authoritative copies before sequential merging.
            job_state["detected_bugs"] = _copy_bugs(self._detected_bugs)
            job_state["bug_trace_records"] = _copy_records(self._bug_trace_records)
        updates = analyze_execution_for_bugs(llm, job_state)  # type: ignore[arg-type]
        with self._lock:
            # Publish new containers instead of mutating already shared objects.
            if isinstance(updates.get("detected_bugs"), list):
                self._detected_bugs = _copy_bugs(updates["detected_bugs"])
            if isinstance(updates.get("bug_trace_records"), dict):
                self._bug_trace_records = _copy_records(updates["bug_trace_records"])
        self._batch_done()

    def snapshot(self) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
        """Return thread-safe copies of authoritative findings and trace records."""
        with self._lock:
            return _copy_bugs(self._detected_bugs), _copy_records(self._bug_trace_records)

    def reset(self) -> None:
        super().reset()
        with self._lock:
            self._detected_bugs = []
            self._bug_trace_records = {}


def build_defect_detection_job(state: AgentState) -> dict[str, Any]:
    """Capture a read-only detection snapshot; trace elements remain immutable."""
    return build_trace_job(
        state,  # type: ignore[arg-type]
        "current_navigation_scenario",
        "current_test_scenario",
        "current_testing_function_id",
        "package_name",
    )


# Process-local singleton.
bug_detection_worker = BugDetectionWorker()
