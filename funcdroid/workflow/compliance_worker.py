"""Asynchronous compliance review with per-rule idempotent results.

Persist findings and expose read-only checklist snapshots for evidence-driven retries without modifying the FFG."""

from __future__ import annotations

from typing import Any, Callable

from compliance.checkpoints import ComplianceChecklist
from workflow.compliance_detection import analyze_execution_for_compliance
from workflow.worker_utils import SerialWorker, build_trace_job


def plan_check_ids(plan: dict[str, Any] | None) -> list[str]:
    return [
        str(item).strip()
        for item in (plan or {}).get("compliance_check_ids", []) or []
        if str(item).strip()
    ]


class ComplianceDetectionWorker(SerialWorker):
    """Run analyze_execution_for_compliance serially."""

    log_prefix = "[Compliance worker]"
    thread_name = "compliance-detection-worker"

    def __init__(self) -> None:
        super().__init__()
        self._on_trace: Callable[[dict[str, Any]], str] | None = None
        self._checklist = ComplianceChecklist()

    def configure(
        self,
        llm,
        *,
        on_batch_done: Callable[[], None] | None = None,
        on_trace: Callable[[dict[str, Any]], str] | None = None,
        initial_checklist: dict[str, Any] | None = None,
        initial_failures: list[dict[str, Any]] | None = None,
    ) -> None:
        with self._lock:
            self._on_trace = on_trace
            if initial_checklist is not None:
                self._checklist = ComplianceChecklist.from_state(initial_checklist)
        self._configure_base(llm, on_batch_done, initial_failures)

    # Mark scheduling state to prevent duplicate dispatch while the queue is backlogged.
    def mark_executing(
        self,
        rule_ids: list[str],
        evidence_signature: str | dict[str, str] = "",
    ) -> None:
        with self._lock:
            self._checklist.mark_executing(rule_ids, evidence_signature)

    def defer_checks(self, rule_ids: list[str], reason: str) -> None:
        """Return checks without a verdict to pending and refund their attempt budget."""
        with self._lock:
            released = self._checklist.release_scheduled(rule_ids)
        if released:
            print(f"{self.log_prefix} {reason}: returned {released} to pending without consuming retries")

    def reopen_deferred(
        self,
        evidence_signature: str | dict[str, str],
        *,
        force: bool = False,
        rules_path: Any = None,
    ) -> tuple[list[str], dict[str, Any]]:
        """Atomically reopen retryable rules and return their IDs and the new checklist snapshot."""
        with self._lock:
            reopened = self._checklist.reopen_deferred(
                evidence_signature,
                force=force,
                rules_path=rules_path,
            )
            return reopened, self._checklist.to_state()

    def _process_job(self, job_state: dict[str, Any]) -> None:
        with self._lock:
            llm = self._llm
            on_trace = self._on_trace
        trace_path = on_trace(job_state) if on_trace is not None else ""
        updates = analyze_execution_for_compliance(llm, job_state)  # type: ignore[arg-type]
        results = updates.get("check_results")
        if isinstance(results, list):
            if trace_path:
                results = [
                    {**finding, "trace_path": trace_path} if isinstance(finding, dict) else finding
                    for finding in results
                ]
            with self._lock:
                self._checklist.apply_results(results)
        self._batch_done()

    def _failure_record(self, job: dict[str, Any], exc: BaseException) -> dict[str, Any]:
        return {"check_ids": plan_check_ids(job.get("current_plan")), **super()._failure_record(job, exc)}

    def _on_failure_locked(self, job: dict[str, Any], exc: BaseException) -> None:
        # Infrastructure failures produced no verdict: record failure separately, return to pending, and refund the review attempt.
        released = self._checklist.release_scheduled(plan_check_ids(job.get("current_plan")))
        if released:
            print(f"{self.log_prefix} Review produced no verdict ({type(exc).__name__}); returned {released} to pending without consuming retries")

    def snapshot(self) -> dict[str, Any]:
        """Return a thread-safe snapshot of current checkpoint state."""
        with self._lock:
            return self._checklist.to_state()

    def reset(self) -> None:
        super().reset()
        with self._lock:
            self._on_trace = None
            self._checklist = ComplianceChecklist()


def build_compliance_detection_job(state: dict[str, Any]) -> dict[str, Any]:
    """Capture a read-only compliance-review job snapshot."""
    return build_trace_job(
        state,
        "current_test_scenario",
        "compliance_rules_path",
        with_trace_id=True,
    )


# Process-local singleton.
compliance_worker = ComplianceDetectionWorker()
