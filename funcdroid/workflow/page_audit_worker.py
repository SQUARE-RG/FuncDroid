"""Background page control audit worker.

Design motivation:
- Replenishing Page.controls currently relies on the FFG update LLM voluntarily declaring them when reusing pages; small-parameter
  models almost never proactively replenish them, causing a deadlock cycle of "initial recognition omission -> never replenished".
  This worker uses an independent background thread to periodically review the control inventory against page screenshots, turning "replenishing
  controls" into a programmatically triggered deterministic mechanism.
- The audit only mines existing evidence (saved page screenshots) without generating new execution; input is the screenshots and
  existing inventory, output is "interactive controls truly visible in screenshots but missing from the inventory".
- Isomorphic with bug_detection_worker: the background thread only performs LLM calls (compute) without directly writing to
  the FFG; merge actions are completed by the main thread at the ffg_update node entry (write), maintaining a single writer.
- Exit condition: page screenshot set fingerprint unchanged, or consecutively PAGE_AUDIT_ZERO_ADD_LIMIT
  audits with zero additions, then marked saturated to stop auditing; state is stored in-process only, reset on restart.

Usage:
    from workflow.page_audit_worker import page_audit_worker

    page_audit_worker.configure(llm)
    page_audit_worker.submit(job)      # job contains page_id and pre-built messages
    ...                                # ffg_update node entry:
    for job in page_audit_worker.collect_results(): ...
    page_audit_worker.record_result(page_id, added_count)
    page_audit_worker.flush(timeout)   # Wait to drain before run completion
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
import traceback
from typing import Any

from workflow.llm_usage import record_llm_failure, record_llm_request, record_llm_response


DEFAULT_MAX_PENDING = max(1, int(os.getenv("PAGE_AUDIT_MAX_PENDING", "4")))
DEFAULT_BATCH_SIZE = max(1, int(os.getenv("PAGE_AUDIT_BATCH_SIZE", "3")))
DEFAULT_ZERO_ADD_LIMIT = max(1, int(os.getenv("PAGE_AUDIT_ZERO_ADD_LIMIT", "2")))
DEFAULT_PER_PAGE_LIMIT = max(1, int(os.getenv("PAGE_AUDIT_PER_PAGE_LIMIT", "5")))
RESULT_PATTERN = json.JSONDecoder()


class PageAuditWorker:
    """Single background thread + queue, sequentially executing page control audit LLM calls."""

    def __init__(self, max_pending: int = DEFAULT_MAX_PENDING) -> None:
        self._lock = threading.Lock()
        self._queue: "queue.Queue[dict[str, Any]]" = queue.Queue()
        self._thread: threading.Thread | None = None
        self._llm = None
        self._pending = 0
        self._idle_cond = threading.Condition(self._lock)
        self._max_pending = max(1, int(max_pending))
        self._results: list[dict[str, Any]] = []
        # page_id -> {"fingerprint": str, "zero_streak": int, "saturated": bool}
        self._audit_state: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Configuration / Lifecycle
    # ------------------------------------------------------------------
    def configure(self, llm) -> None:
        with self._lock:
            self._llm = llm
        self._ensure_worker_started()

    def _ensure_worker_started(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            thread = threading.Thread(
                target=self._worker_loop,
                name="page-audit-worker",
                daemon=True,
            )
            self._thread = thread
        thread.start()

    # ------------------------------------------------------------------
    # Task submission (called by main thread)
    # ------------------------------------------------------------------
    def filter_candidates(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Filter candidate pages by saturation/fingerprint state, returning the subset allowed for submission.

        Each candidate: {"page_id": str, "fingerprint": str, ...}.
        Candidates with the same fingerprint as the last audit (screenshot set unchanged) or already saturated are skipped.
        """
        allowed: list[dict[str, Any]] = []
        with self._lock:
            for candidate in candidates:
                page_id = str(candidate.get("page_id", "") or "")
                state = self._audit_state.get(page_id)
                if state is None:
                    allowed.append(candidate)
                    continue
                if state.get("saturated"):
                    continue
                if state.get("fingerprint") == str(candidate.get("fingerprint", "")):
                    continue
                allowed.append(candidate)
        return allowed

    def submit(self, job: dict[str, Any]) -> bool:
        """Non-blocking submission of a single audit task; dropped and returns False when queue is saturated."""
        self._ensure_worker_started()
        with self._lock:
            if self._pending >= self._max_pending:
                return False
            self._pending += 1
        self._queue.put(job)
        return True

    # ------------------------------------------------------------------
    # Background thread main loop
    # ------------------------------------------------------------------
    def _worker_loop(self) -> None:
        while True:
            job = self._queue.get()
            try:
                self._process_job(job)
            except Exception as exc:  # noqa: BLE001 - Background thread needs to swallow exceptions and continue consuming
                print(f"[PageAuditThread] Failed to process task: {exc}")
                print(traceback.format_exc())
            finally:
                with self._idle_cond:
                    self._pending = max(0, self._pending - 1)
                    if self._pending == 0:
                        self._idle_cond.notify_all()

    def _process_job(self, job: dict[str, Any]) -> None:
        page_id = str(job.get("page_id", "") or "")
        fingerprint = str(job.get("fingerprint", "") or "")
        messages = job.get("messages") or []
        with self._lock:
            llm = self._llm
        if llm is None or not messages:
            self._record_failure(page_id)
            return
        record_llm_request(llm, operation="[PageAudit]")
        try:
            response = llm.invoke(messages)
        except Exception as exc:  # noqa: BLE001
            record_llm_failure(llm, operation="[PageAudit]")
            print(f"[PageAuditThread] LLM call failed (page={page_id}): {exc}")
            self._record_failure(page_id)
            return
        content = str(getattr(response, "content", "") or "")
        record_llm_response(llm, response, operation="[PageAudit]")
        missing, control_merges, invalid_operations, parse_ok = self._parse_audit_payload_with_status(content)
        if not parse_ok:
            print(f"[PageAuditThread] Invalid LLM response format (page={page_id}), keeping task for retry")
            self._record_failure(page_id)
            return
        with self._lock:
            self._results.append({
                "page_id": page_id,
                "fingerprint": fingerprint,
                "missing_controls": missing,
                "control_merges": control_merges,
                "invalid_operations": invalid_operations,
            })
            # Fingerprint accounting is placed in the result, waiting for main thread merge before record_result determines saturation.
            state = self._audit_state.setdefault(page_id, {})
            state["fingerprint"] = fingerprint
        print(
            f"[PageAuditThread] Page inventory review completed: page={page_id}, "
            f"candidate missing controls={len(missing)}, candidate operations to delete={len(invalid_operations)}"
            f", candidate control merges={len(control_merges)}"
        )

    def _parse_audit_payload(self, content: str) -> tuple[list[str], list[str]]:
        missing, _control_merges, invalid_operations, _parse_ok = self._parse_audit_payload_with_status(content)
        return missing, invalid_operations

    def _parse_audit_payload_with_status(
        self,
        content: str,
    ) -> tuple[list[str], list[dict[str, Any]], list[str], bool]:
        text = str(content or "").strip()
        if not text:
            return [], [], [], False
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return [], [], [], False
        try:
            payload = RESULT_PATTERN.decode(text[start:end + 1])
        except json.JSONDecodeError:
            return [], [], [], False
        if not isinstance(payload, dict):
            return [], [], [], False
        missing = self._clean_text_list(payload.get("missing_controls"))
        control_merges = self._clean_control_merges(payload.get("control_merges"))
        invalid_operations = self._clean_text_list(payload.get("invalid_operations"))
        return (
            missing[:DEFAULT_PER_PAGE_LIMIT],
            control_merges[:DEFAULT_PER_PAGE_LIMIT],
            invalid_operations[:DEFAULT_PER_PAGE_LIMIT],
            True,
        )

    @staticmethod
    def _clean_control_merges(raw: object) -> list[dict[str, Any]]:
        if not isinstance(raw, list):
            return []
        cleaned: list[dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            sources = PageAuditWorker._clean_text_list(item.get("sources"))
            target = str(item.get("target", "") or "").strip()
            if len(sources) >= 2 and target:
                cleaned.append({"sources": sources, "target": target})
        return cleaned

    @staticmethod
    def _clean_text_list(raw: object) -> list[str]:
        if not isinstance(raw, list):
            return []
        cleaned: list[str] = []
        for item in raw:
            value = str(item or "").strip()
            if value and value not in cleaned:
                cleaned.append(value)
        return cleaned

    def _record_state(self, page_id: str, fingerprint: str, added: int) -> None:
        with self._lock:
            state = self._audit_state.setdefault(page_id, {})
            state["fingerprint"] = fingerprint
            streak = int(state.get("zero_streak", 0) or 0)
            streak = 0 if added > 0 else streak + 1
            state["zero_streak"] = streak
            if streak >= DEFAULT_ZERO_ADD_LIMIT:
                state["saturated"] = True

    def _record_failure(self, page_id: str) -> None:
        """Leave failed audits retryable without changing zero-result streaks."""
        with self._lock:
            state = self._audit_state.setdefault(str(page_id or ""), {})
            # Invalidate the prior fingerprint so a transient failure does not
            # make the unchanged page look successfully audited.
            state["fingerprint"] = ""

    # ------------------------------------------------------------------
    # Result reading / Status reporting (called by main thread)
    # ------------------------------------------------------------------
    def collect_results(self) -> list[dict[str, Any]]:
        """Retrieve completed audit results (does not change saturation state, wait for record_result)."""
        with self._lock:
            results = list(self._results)
            self._results = []
        return results

    def record_result(self, page_id: str, added: int) -> None:
        """Report actual added count after main thread merge completes, driving saturation determination."""
        with self._lock:
            state = self._audit_state.setdefault(str(page_id or ""), {})
            streak = int(state.get("zero_streak", 0) or 0)
            streak = 0 if added > 0 else streak + 1
            state["zero_streak"] = streak
            if streak >= DEFAULT_ZERO_ADD_LIMIT:
                state["saturated"] = True
                print(
                    f"[PageAuditThread] Page had no additions for {DEFAULT_ZERO_ADD_LIMIT} consecutive audits, "
                    f"marking saturated to stop auditing: page={page_id}"
                )

    def pending_count(self) -> int:
        with self._lock:
            return self._pending

    def is_configured(self) -> bool:
        """Return whether final-output code may submit additional audits."""
        with self._lock:
            return self._llm is not None

    def flush(self, timeout: float | None = None) -> bool:
        """Wait for all submitted tasks in the queue to finish processing."""
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
        """Restore to unconfigured initial state; for test isolation only."""
        with self._lock:
            self._llm = None
            self._results = []
            self._audit_state = {}
            self._max_pending = DEFAULT_MAX_PENDING


# In-process singleton, semantics aligned with workflow/bug_detection_worker.py.
page_audit_worker = PageAuditWorker()
