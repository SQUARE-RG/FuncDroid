import threading
import time

import pytest

from model.flow_graph import Page
from workflow.bug_detection_worker import bug_detection_worker, build_defect_detection_job
from workflow.subgraphs import scenario_exec as scenario_exec_subgraph


@pytest.fixture(autouse=True)
def _isolated_worker():
    """每个测试前后都恢复到未配置状态，避免污染其它测试/模块用到的全局单例。"""
    bug_detection_worker.flush()
    bug_detection_worker.reset()
    yield
    bug_detection_worker.flush()
    bug_detection_worker.reset()


def _screenshot(size: int) -> str:
    return "a" * size


def test_scenario_exec_does_not_submit_review_when_detection_is_disabled(monkeypatch):
    monkeypatch.setattr(
        scenario_exec_subgraph,
        "create_scenario_execution_pass",
        lambda _llm, _device: (
            lambda _state: ({
                "execution_result": {
                    "navigation_status": "skipped",
                    "scenario_status": "completed",
                },
                "navigation_execution_trace": [],
                "scenario_execution_trace": [],
            }, None)
        ),
    )

    def unexpected_submit(_job):
        raise AssertionError("disabled defect detection must not submit a job")

    monkeypatch.setattr(bug_detection_worker, "submit", unexpected_submit)
    subgraph = scenario_exec_subgraph.create_scenario_exec_subgraph(object(), object())

    result = subgraph.invoke({"bug_detection_enabled": False})

    assert result["scenario_exec_subgraph_phase"] == "finished"


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
        "current_test_scenario": "保存修改",
        "current_plan": {"scenario_type": "intra_test", "target_function_id": "func_1"},
    })

    started_at = time.monotonic()
    bug_detection_worker.submit(job)
    submit_elapsed = time.monotonic() - started_at
    assert submit_elapsed < 1.0, "submit() 必须立即返回，不能等待 LLM 调用完成"

    # 任务仍卡在 fake LLM 调用里，此时应认为还有未完成任务。
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
    bug_detection_worker.configure(fake_llm, max_pending=2)
    job = build_defect_detection_job({
        "scenario_execution_trace": [{
            "action_name": "finished",
            "trace_phase": "test",
            "current_page": Page(id="page-1", representative_screenshot=_screenshot(10)),
        }],
        "execution_result": {"scenario_status": "completed"},
        "current_test_scenario": "保存修改",
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
            "current_test_scenario": "保存修改",
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
            "description": "第一批缺陷",
            "expected": "页面正常渲染",
            "actual": "页面渲染错乱",
            "evidence": [{"global_step": 1, "role": "action"}, {"global_step": 1, "role": "after"}],
        }]},
        {"bugs": [{
            "type": "ui_error",
            "evidence_category": "rendering_corruption",
            "severity": "high",
            "confidence": "high",
            "description": "第二批缺陷",
            "expected": "页面正常渲染",
            "actual": "页面渲染错乱",
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

    bug_detection_worker.submit(make_job("场景一"))
    bug_detection_worker.flush(timeout=5)
    bug_detection_worker.submit(make_job("场景二"))
    bug_detection_worker.flush(timeout=5)

    detected_bugs, _records = bug_detection_worker.snapshot()
    descriptions = {bug["description"] for bug in detected_bugs}
    assert descriptions == {"第一批缺陷", "第二批缺陷"}
    assert '"existing_bug_summaries": []' in review_headers[0]
    assert '"description": "第一批缺陷"' in review_headers[1]
