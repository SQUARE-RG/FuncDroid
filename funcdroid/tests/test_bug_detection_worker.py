# Multilingual UI and model-response fixtures use Unicode escapes to preserve their exact text.
import threading
import time

import pytest
import numpy as np
from tools.utils import encode_image_to_jpg_base64

from model.flow_graph import Page
from workflow.bug_detection_worker import bug_detection_worker, build_defect_detection_job
from workflow.subgraphs import scenario_exec as scenario_exec_subgraph


@pytest.fixture(autouse=True)
def _isolated_worker():
    """Reset the global worker before and after each test to avoid cross-test contamination."""
    bug_detection_worker.flush()
    bug_detection_worker.reset()
    yield
    bug_detection_worker.flush()
    bug_detection_worker.reset()


def _screenshot(size: int) -> str:
    return encode_image_to_jpg_base64(np.full((size, size, 3), 180, dtype=np.uint8))


def test_scenario_exec_does_not_submit_review_when_detection_is_disabled(monkeypatch):
    monkeypatch.setattr(
        scenario_exec_subgraph,
        "create_scenario_execution_pass",
        lambda _llm, _device: (
            lambda _state: {
                "execution_result": {
                    "navigation_status": "skipped",
                    "scenario_status": "completed",
                },
                "navigation_execution_trace": [],
                "scenario_execution_trace": [],
            }
        ),
    )

    def unexpected_submit(_job):
        raise AssertionError("disabled defect detection must not submit a job")

    monkeypatch.setattr(bug_detection_worker, "submit", unexpected_submit)
    subgraph = scenario_exec_subgraph.create_scenario_exec_subgraph(object(), object())

    result = subgraph.invoke({"bug_detection_enabled": False})

    assert result["execution_result"]["scenario_status"] == "completed"


def test_submit_is_non_blocking_and_snapshot_updates_after_flush():
    release = threading.Event()

    def fake_llm_invoke(*_args, **_kwargs):
        release.wait(timeout=5)
        return type("Response", (), {"content": '{"bugs": []}'})()

    fake_llm = type("Llm", (), {"invoke": staticmethod(fake_llm_invoke)})()
    bug_detection_worker.configure(fake_llm)

    job = build_defect_detection_job({
        "navigation_execution_trace": [],
        "scenario_execution_trace": [{
            "action_name": "finished",
            "trace_phase": "test",
            "current_page": Page(id="page-1", representative_screenshot=_screenshot(10)),
        }],
        "execution_result": {"navigation_status": "skipped", "scenario_status": "completed"},
        "current_test_scenario": "\u4fdd\u5b58\u4fee\u6539",
        "current_plan": {"scenario_type": "intra_test", "target_function_id": "func_1"},
    })

    started_at = time.monotonic()
    bug_detection_worker.submit(job)
    submit_elapsed = time.monotonic() - started_at
    assert submit_elapsed < 1.0, "submit() \u5fc5\u987b\u7acb\u5373\u8fd4\u56de，\u4e0d\u80fd\u7b49\u5f85 LLM \u8c03\u7528\u5b8c\u6210"

    # The task remains unfinished while its fake LLM call is blocked.
    assert bug_detection_worker.pending_count() == 1
    assert bug_detection_worker.flush(timeout=0.2) is False

    release.set()
    assert bug_detection_worker.flush(timeout=5) is True
    assert bug_detection_worker.pending_count() == 0


def test_pending_cap_applies_backpressure_without_dropping_review():
    entered = threading.Event()
    release = threading.Event()

    def fake_llm_invoke(*_args, **_kwargs):
        entered.set()
        release.wait(timeout=5)
        return type("Response", (), {"content": '{"bugs": []}'})()

    fake_llm = type("Llm", (), {"invoke": staticmethod(fake_llm_invoke)})()
    bug_detection_worker.configure(fake_llm)
    bug_detection_worker._max_pending = 2
    job = build_defect_detection_job({
        "scenario_execution_trace": [{
            "action_name": "finished",
            "trace_phase": "test",
            "current_page": Page(id="page-1", representative_screenshot=_screenshot(10)),
        }],
        "execution_result": {"scenario_status": "completed"},
        "current_test_scenario": "\u4fdd\u5b58\u4fee\u6539",
        "current_plan": {"scenario_type": "intra_test"},
    })

    assert bug_detection_worker.submit(dict(job)) is True
    assert entered.wait(timeout=2)
    assert bug_detection_worker.submit(dict(job)) is True
    accepted = []
    submitter = threading.Thread(
        target=lambda: accepted.append(bug_detection_worker.submit(dict(job)))
    )
    submitter.start()
    submitter.join(timeout=0.2)
    assert submitter.is_alive()
    assert bug_detection_worker.pending_count() == 2

    release.set()
    submitter.join(timeout=5)
    assert accepted == [True]
    assert bug_detection_worker.flush(timeout=5) is True


def test_on_batch_done_callback_runs_after_each_job():
    calls = []

    class FakeLlm:
        def invoke(self, _messages):
            return type("Response", (), {"content": '{"bugs": []}'})()

    bug_detection_worker.configure(FakeLlm(), on_batch_done=lambda: calls.append(1))

    for _ in range(3):
        job = build_defect_detection_job({
            "navigation_execution_trace": [],
            "scenario_execution_trace": [{
                "action_name": "finished",
                "trace_phase": "test",
                "current_page": Page(id="page-1", representative_screenshot=_screenshot(10)),
            }],
            "execution_result": {"navigation_status": "skipped", "scenario_status": "completed"},
            "current_test_scenario": "\u4fdd\u5b58\u4fee\u6539",
            "current_plan": {"scenario_type": "intra_test", "target_function_id": "func_1"},
        })
        bug_detection_worker.submit(job)

    assert bug_detection_worker.flush(timeout=5) is True
    assert len(calls) == 3


def test_sequential_jobs_accumulate_bugs_without_losing_earlier_findings(monkeypatch):
    from workflow import trace_defect_detection

    review_headers: list[str] = []
    responses = iter([
        {"bugs": [{
            "type": "ui_error",
            "evidence_category": "rendering_corruption",
            "severity": "high",
            "confidence": "high",
            "description": "\u7b2c\u4e00\u6279\u7f3a\u9677",
            "expected": "\u9875\u9762\u6b63\u5e38\u6e32\u67d3",
            "actual": "\u9875\u9762\u6e32\u67d3\u9519\u4e71",
            "evidence": [{"global_step": 1, "role": "action"}, {"global_step": 1, "role": "after"}],
        }]},
        {"bugs": [{
            "type": "ui_error",
            "evidence_category": "rendering_corruption",
            "severity": "high",
            "confidence": "high",
            "description": "\u7b2c\u4e8c\u6279\u7f3a\u9677",
            "expected": "\u9875\u9762\u6b63\u5e38\u6e32\u67d3",
            "actual": "\u9875\u9762\u6e32\u67d3\u9519\u4e71",
            "evidence": [{"global_step": 1, "role": "action"}, {"global_step": 1, "role": "after"}],
        }]},
    ])

    def fake_invoke(_llm, messages, **_kwargs):
        review_headers.append(messages[1].content[0]["text"])
        return next(responses)

    monkeypatch.setattr(trace_defect_detection, "invoke_llm_json_with_retry", fake_invoke)
    bug_detection_worker.configure(object())

    def make_job(text: str) -> dict:
        return build_defect_detection_job({
            "navigation_execution_trace": [],
            "scenario_execution_trace": [{
                "action_name": "finished",
                "trace_phase": "test",
                "current_page": Page(id="page-1", representative_screenshot=_screenshot(10)),
            }],
            "execution_result": {"navigation_status": "skipped", "scenario_status": "completed"},
            "current_test_scenario": text,
            "current_plan": {"scenario_type": "intra_test", "target_function_id": "func_1"},
        })

    bug_detection_worker.submit(make_job("\u573a\u666f\u4e00"))
    bug_detection_worker.flush(timeout=5)
    bug_detection_worker.submit(make_job("\u573a\u666f\u4e8c"))
    bug_detection_worker.flush(timeout=5)

    detected_bugs, _records = bug_detection_worker.snapshot()
    descriptions = {bug["description"] for bug in detected_bugs}
    assert descriptions == {"\u7b2c\u4e00\u6279\u7f3a\u9677", "\u7b2c\u4e8c\u6279\u7f3a\u9677"}
    assert '"existing_bug_summaries": []' in review_headers[0]
    assert '"description": "\u7b2c\u4e00\u6279\u7f3a\u9677"' in review_headers[1]
