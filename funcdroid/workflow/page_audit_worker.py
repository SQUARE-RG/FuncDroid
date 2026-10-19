"""Background audit of controls visible in saved page screenshots.

Audits identify omitted interactive controls without performing new GUI actions. The worker calls the model; the main thread alone merges results into the FFG. Deduplicate in-flight jobs by page and screenshot fingerprint. Stop unchanged, saturated, or repeatedly failing fingerprints until evidence changes. All scheduling state is process-local."""

from __future__ import annotations

import json
import os
from typing import Any

from workflow.llm import invoke_llm_json_with_retry, llm_json_result_failed
from workflow.worker_utils import SerialWorker


DEFAULT_BATCH_SIZE = 3
DEFAULT_ZERO_ADD_LIMIT = 2
DEFAULT_PER_PAGE_LIMIT = 5
# Pause repeated failures for an unchanged fingerprint; new evidence re-enables auditing.
DEFAULT_FAILURE_LIMIT = max(1, int(os.getenv("PAGE_AUDIT_FAILURE_LIMIT", "2")))
# The worker manages failure counts; individual audit calls do not retry.
PAGE_AUDIT_LLM_ATTEMPTS = max(1, int(os.getenv("PAGE_AUDIT_LLM_ATTEMPTS", "1")))


class PageAuditWorker(SerialWorker):
    """Process page-audit model calls serially; a full queue drops candidates for a later round."""

    log_prefix = "[Page audit worker]"
    thread_name = "page-audit-worker"
    default_max_pending = 4
    drop_when_full = True

    def __init__(self) -> None:
        super().__init__()
        self._results: list[dict[str, Any]] = []
        # page_id -> {"fingerprint": str, "zero_streak": int, "saturated": bool,
        #             "inflight": str, "failed_fingerprint": str, "fail_streak": int,
        #             "audited_screenshots": set[str], "pending_screenshots": int}
        self._audit_state: dict[str, dict[str, Any]] = {}

    def configure(self, llm) -> None:
        self._configure_base(llm)

    # ------------------------------------------------------------------
    # Submission from the main thread.
    # ------------------------------------------------------------------
    def filter_candidates(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Select unsaturated, changed page fingerprints not already in flight or repeatedly failing.

        Each candidate contains page_id and fingerprint."""
        allowed: list[dict[str, Any]] = []
        with self._lock:
            for candidate in candidates:
                page_id = str(candidate.get("page_id", "") or "")
                fingerprint = str(candidate.get("fingerprint", ""))
                state = self._audit_state.get(page_id)
                if state is None:
                    allowed.append(candidate)
                    continue
                if state.get("saturated"):
                    continue
                if state.get("inflight") == fingerprint:
                    continue
                if state.get("fingerprint") == fingerprint and not state.get("pending_screenshots"):
                    continue
                if (
                    state.get("failed_fingerprint") == fingerprint
                    and int(state.get("fail_streak", 0) or 0) >= DEFAULT_FAILURE_LIMIT
                ):
                    continue
                allowed.append(candidate)
        return allowed

    def submit(self, job: dict[str, Any], timeout: float | None = None) -> bool:
        # Register in-flight state before submitting; the worker can finish before submit returns.
        page_id = str(job.get("page_id", "") or "")
        fingerprint = str(job.get("fingerprint", "") or "")
        with self._lock:
            state = self._audit_state.setdefault(page_id, {})
            previous = state.get("inflight")
            state["inflight"] = fingerprint
        ok = super().submit(job, timeout=timeout)
        if not ok:
            with self._lock:
                state = self._audit_state.setdefault(page_id, {})
                if state.get("inflight") == fingerprint:
                    if previous:
                        state["inflight"] = previous
                    else:
                        state.pop("inflight", None)
        return ok

    def audited_screenshot_digests(self, page_id: str) -> set[str]:
        with self._lock:
            state = self._audit_state.get(str(page_id or ""), {})
            return set(state.get("audited_screenshots", set()))

    def _process_job(self, job: dict[str, Any]) -> None:
        page_id = str(job.get("page_id", "") or "")
        fingerprint = str(job.get("fingerprint", "") or "")
        messages = job.get("messages") or []
        with self._lock:
            llm = self._llm
        if llm is None or not messages:
            self._record_failure(page_id, fingerprint)
            return
        payload = invoke_llm_json_with_retry(
            llm,
            messages,
            log_prefix="[Page audit]",
            max_attempts=PAGE_AUDIT_LLM_ATTEMPTS,
        )
        if llm_json_result_failed(payload):
            print(f"[Page audit worker] Model call or parsing failed for page={page_id}; retry according to the failure limit")
            self._record_failure(page_id, fingerprint)
            return
        missing, control_merges, invalid_operations = self._clean_audit_payload(payload)
        with self._lock:
            self._results.append({
                "page_id": page_id,
                "fingerprint": fingerprint,
                "missing_controls": missing,
                "control_merges": control_merges,
                "invalid_operations": invalid_operations,
            })
            # Let the main thread determine saturation after merging the result.
            state = self._audit_state.setdefault(page_id, {})
            state["fingerprint"] = fingerprint
            if state.get("inflight") == fingerprint:
                state.pop("inflight", None)
            state["fail_streak"] = 0
            state.pop("failed_fingerprint", None)
            audited = set(state.get("audited_screenshots", set()))
            audited.update(str(item) for item in job.get("screenshot_digests", []) or [])
            state["audited_screenshots"] = audited
            state["pending_screenshots"] = int(job.get("pending_screenshots", 0) or 0)
        print(
            f"[Page audit worker] Page inventory audit completed: page={page_id}, "
            f"missing_control_candidates={len(missing)}, invalid_operation_candidates={len(invalid_operations)}"
            f", control_merge_candidates={len(control_merges)}"
        )

    def _clean_audit_payload(
        self,
        payload: dict[str, Any],
    ) -> tuple[list[str], list[dict[str, Any]], list[str]]:
        missing = self._clean_text_list(payload.get("missing_controls"))
        control_merges = self._clean_control_merges(payload.get("control_merges"))
        invalid_operations = self._clean_text_list(payload.get("invalid_operations"))
        return (
            missing[:DEFAULT_PER_PAGE_LIMIT],
            control_merges[:DEFAULT_PER_PAGE_LIMIT],
            invalid_operations[:DEFAULT_PER_PAGE_LIMIT],
        )

    def _parse_audit_payload_with_status(
        self,
        content: str,
    ) -> tuple[list[str], list[dict[str, Any]], list[str], bool]:
        """Parse an audit response through the compatibility entrypoint."""
        text = str(content or "").strip()
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return [], [], [], False
        try:
            payload = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return [], [], [], False
        if not isinstance(payload, dict):
            return [], [], [], False
        return (*self._clean_audit_payload(payload), True)

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

    def _record_failure(self, page_id: str, fingerprint: str = "") -> None:
        """Keep failed audits retryable until the fingerprint failure limit.

        Invalidate the previous success fingerprint without changing the zero-addition counter."""
        with self._lock:
            state = self._audit_state.setdefault(str(page_id or ""), {})
            state["fingerprint"] = ""
            if fingerprint and state.get("inflight") == fingerprint:
                state.pop("inflight", None)
            if fingerprint and state.get("failed_fingerprint") == fingerprint:
                state["fail_streak"] = int(state.get("fail_streak", 0) or 0) + 1
            else:
                state["failed_fingerprint"] = fingerprint
                state["fail_streak"] = 1
            if fingerprint and state["fail_streak"] >= DEFAULT_FAILURE_LIMIT:
                print(
                    f"[Page audit worker] Page state failed {state['fail_streak']} consecutive audits; "
                    f"pause retries until page evidence changes: page={page_id}"
                )

    # ------------------------------------------------------------------
    # Main-thread result collection and state reporting.
    # ------------------------------------------------------------------
    def collect_results(self) -> list[dict[str, Any]]:
        """Collect completed results; saturation is updated later by record_result."""
        with self._lock:
            results = list(self._results)
            self._results = []
        return results

    def record_result(self, page_id: str, added: int) -> None:
        """Report actual merged additions from the main thread to update saturation."""
        with self._lock:
            state = self._audit_state.setdefault(str(page_id or ""), {})
            streak = int(state.get("zero_streak", 0) or 0)
            streak = 0 if added > 0 else streak + 1
            state["zero_streak"] = streak
            # Do not saturate while any saved screenshot remains unaudited.
            if streak >= DEFAULT_ZERO_ADD_LIMIT and not state.get("pending_screenshots"):
                state["saturated"] = True
                print(
                    f"[Page audit worker] Page had no additions in {DEFAULT_ZERO_ADD_LIMIT} consecutive audits; "
                    f"mark saturated and stop auditing: page={page_id}"
                )

    def is_configured(self) -> bool:
        """Return whether final-output code may submit additional audits."""
        with self._lock:
            return self._llm is not None

    def record_failure(self, job: dict[str, Any], exc: BaseException) -> None:
        job = job or {}
        self._record_failure(
            str(job.get("page_id", "") or ""),
            str(job.get("fingerprint", "") or ""),
        )

    def reset(self) -> None:
        super().reset()
        with self._lock:
            self._results = []
            self._audit_state = {}


# Process-local singleton, matching the bug-detection worker.
page_audit_worker = PageAuditWorker()
