# Multilingual UI and model-response fixtures use Unicode escapes to preserve their exact text.
"""Detection regressions for deduplication, compliance rechecks, worker semantics, thread safety, image numbering, and compression caches."""

from __future__ import annotations

import json
import threading
import time

import numpy as np
import pytest

import workflow.image_compact as image_compact_module
from compliance.checkpoints import (
    CHECK_STATUS_DONE,
    CHECK_STATUS_EXECUTING,
    CHECK_STATUS_PENDING,
    ComplianceChecklist,
)
from model.flow_graph import Page
from tools.utils import encode_image_to_jpg_base64
from workflow import compliance_detection
from workflow.bug_dedupe import bug_identity_key, dedupe_bugs, strip_logcat_prefix
from workflow.compliance_worker import ComplianceDetectionWorker
from workflow.image_compact import clear_compact_cache, compact_screenshot
from workflow.trace_defect_detection import (
    BUG_DETECT_IMAGE_QUALITY,
    _ensure_bug_ids,
    _update_bug_trace_records,
)
from workflow.trace_messages import coerce_step
from workflow.worker_utils import SerialWorker


def _shot(value: int = 180) -> str:
    return encode_image_to_jpg_base64(np.full((48, 32, 3), value, dtype=np.uint8))


# ---------------------------------------------------------------------------
# 1. Strip logcat prefixes from crash identity keys.
# ---------------------------------------------------------------------------

_TIME_FORMAT_CRASH = (
    "{ts} E/AndroidRuntime( {pid}): FATAL EXCEPTION: main\n"
    "{ts} E/AndroidRuntime( {pid}): Process: com.example.app, PID: {pid}\n"
    "{ts} E/AndroidRuntime( {pid}): java.lang.NullPointerException: Attempt to invoke virtual method on a null object reference\n"
    "{ts} E/AndroidRuntime( {pid}): \tat com.example.app.MainActivity.onCreate(MainActivity.java:88)\n"
    "{ts} E/AndroidRuntime( {pid}): \tat android.app.Activity.performCreate(Activity.java:8000)"
)
_THREADTIME_CRASH = (
    "{ts}  {pid}  {tid} E AndroidRuntime: FATAL EXCEPTION: main\n"
    "{ts}  {pid}  {tid} E AndroidRuntime: Process: com.example.app, PID: {pid}\n"
    "{ts}  {pid}  {tid} E AndroidRuntime: java.lang.NullPointerException: Attempt to invoke virtual method on a null object reference\n"
    "{ts}  {pid}  {tid} E AndroidRuntime: \tat com.example.app.MainActivity.onCreate(MainActivity.java:88)"
)


def _crash_bug(actual: str) -> dict:
    return {"type": "crash", "actual": actual, "target_function_id": "func_1", "description": "\u5d29\u6e83"}


def test_strip_logcat_prefix_handles_time_and_threadtime():
    assert strip_logcat_prefix(
        "10-05 12:00:01.123 E/AndroidRuntime( 4321): FATAL EXCEPTION: main"
    ) == "FATAL EXCEPTION: main"
    assert strip_logcat_prefix(
        "10-05 12:00:01.123  4321  4350 E AndroidRuntime: FATAL EXCEPTION: main"
    ) == "FATAL EXCEPTION: main"
    assert strip_logcat_prefix("java.lang.IllegalStateException: boom") == "java.lang.IllegalStateException: boom"


@pytest.mark.parametrize("template", [_TIME_FORMAT_CRASH, _THREADTIME_CRASH])
def test_same_crash_with_different_time_and_pid_has_same_key(template):
    first = _crash_bug(template.format(ts="10-05 12:00:01.123", pid=4321, tid=4350))
    second = _crash_bug(template.format(ts="10-06 09:13:44.987", pid=9876, tid=9901))
    assert bug_identity_key(first) == bug_identity_key(second)
    assert len(dedupe_bugs([first, second])) == 1


def test_time_and_threadtime_formats_of_same_crash_share_key():
    first = _crash_bug(_TIME_FORMAT_CRASH.format(ts="10-05 12:00:01.123", pid=4321, tid=0))
    second = _crash_bug(_THREADTIME_CRASH.format(ts="10-06 09:13:44.987", pid=9876, tid=9901))
    assert bug_identity_key(first) == bug_identity_key(second)


def test_crash_signature_prefers_exception_and_first_frame():
    key = bug_identity_key(_crash_bug(_TIME_FORMAT_CRASH.format(ts="10-05 12:00:01.123", pid=1, tid=0)))
    assert "nullpointerexception" in key
    assert "mainactivityoncreate" in key
    # Use only the first frame, excluding subsequent framework frames.
    assert "performcreate" not in key
    assert "fatalexception" not in key


def test_different_exceptions_stay_separate_with_logcat_prefix():
    first = _crash_bug(_TIME_FORMAT_CRASH.format(ts="10-05 12:00:01.123", pid=1, tid=0))
    other = _crash_bug(
        "10-05 12:00:01.123 E/AndroidRuntime( 1): java.lang.IllegalStateException: boom\n"
        "10-05 12:00:01.123 E/AndroidRuntime( 1): \tat com.example.app.Other.run(Other.java:5)"
    )
    assert len(dedupe_bugs([first, other])) == 2


# ---------------------------------------------------------------------------
# 2. Functional defect identity excludes scenario text.
# ---------------------------------------------------------------------------

def _functional_bug(scenario: str, **extra) -> dict:
    return {
        "type": "logic_error",
        "target_function_id": "func_1",
        "description": "\u4fdd\u5b58\u540e\u6807\u9898\u4e22\u5931",
        "actual": "\u91cd\u65b0\u8fdb\u5165\u7f16\u8f91\u9875\u540e\u6807\u9898\u8f93\u5165\u6846\u4e3a\u7a7a，\u4e4b\u524d\u4fdd\u5b58\u7684\u6807\u9898\u6ca1\u6709\u663e\u793a",
        "scenario": scenario,
        **extra,
    }


def test_functional_key_ignores_regenerated_scenario_text():
    first = _functional_bug("\u7f16\u8f91\u6807\u9898\u540e\u4fdd\u5b58\u5e76\u8fd4\u56de\u5217\u8868\u67e5\u770b")
    second = _functional_bug("\u4fee\u6539\u7b14\u8bb0\u6807\u9898，\u4fdd\u5b58\u540e\u79bb\u5f00\u518d\u8fd4\u56de\u786e\u8ba4\u7ed3\u679c")
    assert bug_identity_key(first) == bug_identity_key(second)
    assert bug_identity_key(first).startswith("functional|func1|")
    assert len(dedupe_bugs([first, second])) == 1


def test_functional_key_still_separates_targets():
    first = _functional_bug("\u573a\u666f")
    second = {**_functional_bug("\u573a\u666f"), "target_function_id": "func_2"}
    assert len(dedupe_bugs([first, second])) == 2


def test_trace_records_match_cross_round_bug_without_scenario():
    existing = {**_functional_bug("\u7b2c\u4e00\u8f6e\u573a\u666f"), "bug_id": "BUG-001"}
    current = _functional_bug("\u7b2c\u4e8c\u8f6e\u91cd\u65b0\u751f\u6210\u7684\u573a\u666f")
    trace = [{"action_name": "click", "current_page": Page(id="p", representative_screenshot=_shot())}]
    detected = [existing]
    records = _update_bug_trace_records({"bug_trace_records": {}}, detected, [current], trace)
    assert records == {"BUG-001": trace}
    assert detected[0]["scenario"] == "\u7b2c\u4e8c\u8f6e\u91cd\u65b0\u751f\u6210\u7684\u573a\u666f"


# ---------------------------------------------------------------------------
# 3. Compliance rechecks: undetermined preserves terminal status and rechecks are bounded.
# ---------------------------------------------------------------------------

def _apply(checklist: ComplianceChecklist, verdict: str, description: str = "") -> None:
    checklist.apply_results([{
        "check_id": "P03",
        "verdict": verdict,
        "description": description or verdict,
        "evidence": [{"global_step": 1, "role": "before"}],
    }])


@pytest.mark.parametrize("verdict", ["compliant", "not_applicable"])
def test_reverify_undetermined_keeps_previous_terminal_verdict(verdict):
    checklist = ComplianceChecklist()
    checklist.mark_executing(["P03"], "v1")
    _apply(checklist, verdict, "\u65e7\u8def\u5f84\u7ed3\u8bba")
    checklist.reopen_deferred({"P03": "v1"})
    assert checklist.reopen_deferred({"P03": "v2"}) == ["P03"]
    checklist.mark_executing(["P03"], "v2")
    _apply(checklist, "undetermined", "\u65b0\u8def\u5f84\u8bc1\u636e\u4e0d\u8db3")

    assert checklist.status["P03"] == CHECK_STATUS_DONE
    finding = checklist.findings[0]
    assert finding["verdict"] == verdict
    assert finding["description"] == "\u65e7\u8def\u5f84\u7ed3\u8bba"
    assert finding["history"][-1]["verdict"] == "undetermined"
    assert finding["history"][-1]["description"] == "\u65b0\u8def\u5f84\u8bc1\u636e\u4e0d\u8db3"


def test_done_reopen_is_capped_and_counter_persists():
    checklist = ComplianceChecklist()
    checklist.mark_executing(["P03"], "v1")
    _apply(checklist, "compliant")
    checklist.reopen_deferred({"P03": "v1"})
    assert checklist.reopen_deferred({"P03": "v2"}) == ["P03"]
    checklist.mark_executing(["P03"], "v2")
    _apply(checklist, "compliant", "\u590d\u6838\u4ecd\u7b26\u5408")
    assert checklist.reverify_counts["P03"] == 1

    restored = ComplianceChecklist.from_state(json.loads(json.dumps(checklist.to_state())))
    assert restored.reverify_counts == {"P03": 1}
    # With the default limit of one, later Functional Flow Graph changes cannot reopen it.
    for signature in ("v3", "v4", "v5"):
        assert restored.reopen_deferred({"P03": signature}) == []
    assert restored.status["P03"] == CHECK_STATUS_DONE


def test_legacy_state_without_reverify_counts_loads():
    restored = ComplianceChecklist.from_state({
        "status": {"P03": "done"},
        "findings": [{"check_id": "P03", "verdict": "compliant", "evidence": []}],
        "attempts": {"P03": 1},
    })
    assert restored.reverify_counts == {}
    assert restored.to_state()["reverify_counts"] == {}


# ---------------------------------------------------------------------------
# 4. Infrastructure failures refund the compliance attempt budget.
# ---------------------------------------------------------------------------

def test_compliance_infrastructure_failure_releases_budget():
    worker = ComplianceDetectionWorker()
    worker.mark_executing(["R014"], "sig")
    assert worker.snapshot()["attempts"]["R014"] == 1
    job = {"iteration_count": 3, "current_plan": {"compliance_check_ids": ["R014"]}}
    worker.record_failure(job, RuntimeError("LLM \u91cd\u8bd5\u8017\u5c3d"))
    snapshot = worker.snapshot()
    assert snapshot["status"]["R014"] == CHECK_STATUS_PENDING
    assert snapshot["attempts"]["R014"] == 0
    assert snapshot["findings"] == []
    failures = worker.failures_snapshot()
    assert failures and failures[0]["check_ids"] == ["R014"]


def test_compliance_unfinished_flush_releases_budget():
    release = threading.Event()

    class Blocking(ComplianceDetectionWorker):
        def _process_job(self, job):
            release.wait(5)

    worker = Blocking()
    worker._configure_base(llm=None)
    worker.mark_executing(["R014"], "sig")
    try:
        assert worker.submit({"current_plan": {"compliance_check_ids": ["R014"]}}) is True
        assert worker.flush(timeout=0.05) is False
        assert worker.record_unfinished("\u6536\u5c3e\u8d85\u65f6") == 1
        snapshot = worker.snapshot()
        assert snapshot["status"]["R014"] == CHECK_STATUS_PENDING
        assert snapshot["attempts"]["R014"] == 0
    finally:
        release.set()
        assert worker.flush(timeout=5)


# ---------------------------------------------------------------------------
# 5. SerialWorker timeout completion, ReviewUnfinished cleanup, flush(0), submit timeout, and reset.
# ---------------------------------------------------------------------------

class _Blocking(SerialWorker):
    log_prefix = "[\u6d4b\u8bd5\u7ebf\u7a0b]"
    thread_name = "test-review-fixes-worker"

    def __init__(self, fail: bool = False):
        super().__init__()
        self.release = threading.Event()
        self.fail = fail
        self.batches = 0

    def _process_job(self, job):
        self.release.wait(5)
        if self.fail:
            raise RuntimeError("boom")

    def _batch_done(self):
        self.batches += 1
        super()._batch_done()


def test_unfinished_record_removed_after_late_success():
    worker = _Blocking()
    worker._configure_base(llm=None)
    worker.submit({"iteration_count": 1})
    assert worker.flush(timeout=0.05) is False
    assert worker.record_unfinished("\u8d85\u65f6") == 1
    failure = worker.failures_snapshot()[0]
    assert failure["error_type"] == "ReviewUnfinished" and failure["job_id"] == 1
    worker.release.set()
    assert worker.flush(timeout=5)
    assert worker.failures_snapshot() == []


def test_unfinished_record_replaced_by_real_failure():
    worker = _Blocking(fail=True)
    worker._configure_base(llm=None)
    worker.submit({"iteration_count": 1})
    assert worker.flush(timeout=0.05) is False
    worker.record_unfinished("\u8d85\u65f6")
    worker.release.set()
    assert worker.flush(timeout=5)
    assert [item["error_type"] for item in worker.failures_snapshot()] == ["RuntimeError"]


def test_flush_zero_timeout_does_not_wait():
    worker = _Blocking()
    worker._configure_base(llm=None)
    worker.submit({})
    try:
        started = time.monotonic()
        assert worker.flush(timeout=0) is False
        assert time.monotonic() - started < 0.5
    finally:
        worker.release.set()
        assert worker.flush(timeout=5)
    assert worker.flush(timeout=0) is True


def test_submit_gives_up_after_timeout_and_records_failure():
    worker = _Blocking()
    worker._configure_base(llm=None)
    worker._max_pending = 1
    try:
        assert worker.submit({"iteration_count": 1}) is True
        started = time.monotonic()
        assert worker.submit({"iteration_count": 2}, timeout=0.2) is False
        assert time.monotonic() - started < 2.0
        failures = worker.failures_snapshot()
        assert failures[-1]["error_type"] == "SubmitTimeoutError"
        assert failures[-1]["iteration_count"] == 2
    finally:
        worker.release.set()
        assert worker.flush(timeout=5)


def test_submit_restarts_dead_worker_thread():
    worker = _Blocking()
    worker.release.set()
    worker._max_pending = 1
    # Simulate a dead background thread whose pending count was not cleared.
    dead = threading.Thread(target=lambda: None)
    dead.start()
    dead.join()
    worker._thread = dead
    worker._pending = 1
    worker._queue.put((99, {}))
    worker._outstanding[99] = {}
    assert worker.submit({}, timeout=5) is True
    assert worker.flush(timeout=5)


def test_reset_clears_pending_queue_and_job_ids():
    worker = _Blocking()
    worker._pending = 3
    worker._next_job_id = 7
    worker._queue.put((1, {}))
    worker.reset()
    assert worker._pending == 0
    assert worker._next_job_id == 0
    assert worker._queue.empty()


# ---------------------------------------------------------------------------
# 6. Snapshots and workers must not share mutable elements.
# ---------------------------------------------------------------------------

def test_ensure_bug_ids_does_not_mutate_input_dicts():
    original = {"type": "ui_error", "description": "x"}
    result = _ensure_bug_ids([original])
    assert "bug_id" not in original and "target_function_id" not in original
    assert result[0]["bug_id"] == "BUG-001"


def test_bug_worker_snapshot_copies_elements():
    from workflow.bug_detection_worker import BugDetectionWorker

    worker = BugDetectionWorker()
    worker._detected_bugs = [{"bug_id": "BUG-001", "evidence": [{"global_step": 1}]}]
    bugs, _records = worker.snapshot()
    bugs[0]["mutated"] = True
    bugs[0]["evidence"].append({"global_step": 2})
    assert "mutated" not in worker._detected_bugs[0]
    assert worker._detected_bugs[0]["evidence"] == [{"global_step": 1}]


def test_checklist_snapshot_not_affected_by_later_apply():
    checklist = ComplianceChecklist()
    checklist.mark_executing(["P03"], "v1")
    _apply(checklist, "violation", "\u8fdd\u89c4")
    snapshot = checklist.to_state()
    frozen = json.dumps(snapshot, ensure_ascii=False, sort_keys=True)
    _apply(checklist, "compliant", "\u540e\u7eed\u8def\u5f84\u7b26\u5408")
    _apply(checklist, "violation", "\u518d\u6b21\u8fdd\u89c4")
    assert json.dumps(snapshot, ensure_ascii=False, sort_keys=True) == frozen
    assert checklist.findings[0]["history"]


# ---------------------------------------------------------------------------
# 7. Compliance OBSERVES_PREVIOUS_RESULT links only adjacent original steps.
# ---------------------------------------------------------------------------

def test_compliance_observes_previous_only_for_adjacent_raw_steps():
    trace = [
        {"action_name": "click", "current_page": {"screenshot": _shot(10)}, "trace_phase": "test"},
        {"action_name": "wait", "current_page": None, "trace_phase": "test"},
        {"action_name": "click", "current_page": {"screenshot": _shot(20)}, "trace_phase": "test"},
        {"action_name": "finished", "current_page": {"screenshot": _shot(30)}, "trace_phase": "test"},
    ]
    steps = compliance_detection._collect_steps(trace)
    content = compliance_detection._build_messages({}, steps, [])[1].content
    rendered = "\n".join(block["text"] for block in content if block.get("type") == "text")
    assert "This image shows the state after STEP 1 was executed." not in rendered
    assert "This image shows the state after STEP 3 was executed." in rendered
    for position, block in enumerate(content):
        if block.get("type") == "text" and "[IMAGE_REF]" in block["text"]:
            assert content[position + 1]["type"] == "image_url"


# ---------------------------------------------------------------------------
# 8. Image compression cache keys and defaults match the Functional Flow Graph path.
# ---------------------------------------------------------------------------

def test_compact_cache_keys_are_digests_not_raw_base64():
    clear_compact_cache()
    screenshot = _shot(77)
    compact_screenshot(screenshot, max_side=32, quality=80)
    keys = list(image_compact_module._cache.keys())
    assert len(keys) == 1
    digest, max_side, quality = keys[0]
    assert len(digest) == 40 and screenshot not in keys[0]
    assert (max_side, quality) == (32, 80)


def test_compact_cache_is_bounded(monkeypatch):
    clear_compact_cache()
    monkeypatch.setattr(image_compact_module, "_CACHE_SIZE", 3)
    for value in range(6):
        compact_screenshot(_shot(value * 10), max_side=32, quality=80)
    assert len(image_compact_module._cache) == 3


def test_bug_detect_default_quality_matches_ffg():
    from workflow.nodes.ffg_update import FFG_UPDATE_IMAGE_QUALITY

    assert BUG_DETECT_IMAGE_QUALITY <= 95
    assert BUG_DETECT_IMAGE_QUALITY == FFG_UPDATE_IMAGE_QUALITY


# ---------------------------------------------------------------------------
# 9. Evidence-chain step parsing accepts numeric strings.
# ---------------------------------------------------------------------------

def test_coerce_step_accepts_numeric_strings_only():
    assert coerce_step(5) == 5
    assert coerce_step(" 5 ") == 5
    for value in (True, 0, -1, "x", None, 1.5):
        assert coerce_step(value) is None


def test_functionality_evidence_accepts_numeric_strings():
    item = {"functionality_evidence": {
        "denial_step": "1", "functional_action_step": "2",
        "result_observation_step": 3, "observed_result": "\u663e\u793a\u7ffb\u8bd1\u7ed3\u679c",
    }}
    steps = [
        {"raw_step_index": index, "action_name": action, "current_page": Page(representative_screenshot=_shot())}
        for index, action in ((1, "click"), (2, "click"), (3, "finished"))
    ]
    assert compliance_detection._validate_functionality_evidence(item, steps)
    item["functionality_evidence"]["denial_step"] = True
    assert not compliance_detection._validate_functionality_evidence(item, steps)


# ---------------------------------------------------------------------------
# 11. Removed dead code.
# ---------------------------------------------------------------------------

def test_dead_checklist_helpers_removed():
    import compliance.checkpoints as checkpoints

    assert not hasattr(ComplianceChecklist, "verdict_of")
    assert not hasattr(checkpoints, "group_priority")
    assert CHECK_STATUS_EXECUTING == "executing"
