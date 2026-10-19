# Multilingual UI and model-response fixtures use Unicode escapes to preserve their exact text.
"""Offline regressions for evidence, termination and completion contracts."""

import base64
import json
import signal
import subprocess
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import main
from model.flow_graph import FunctionalFlowGraph, Page
from scripts import batch_compliance_run, batch_test_apks
from workflow.bug_detection_worker import bug_detection_worker
from workflow.compliance_worker import build_compliance_detection_job, compliance_worker
from workflow.graph import _fallback_scenario_plan, _safe_node
from workflow.llm import invoke_llm_json_with_retry
from workflow.page_audit_worker import page_audit_worker
from workflow.run_options import validate_resume_options
from workflow.subgraphs import scenario_exec
from workflow.trace_artifacts import disable_trace_artifacts
from workflow.trace_defect_detection import DefectReviewError, analyze_execution_for_bugs


@pytest.fixture(autouse=True)
def isolated_workers():
    for worker in (bug_detection_worker, compliance_worker, page_audit_worker):
        assert worker.flush(timeout=5)
        worker.reset()
    yield
    for worker in (bug_detection_worker, compliance_worker, page_audit_worker):
        assert worker.flush(timeout=5)
        worker.reset()
    disable_trace_artifacts()


def options(*extra):
    return main._build_parser().parse_args(["--device-id", "offline-device", *extra])


def image_page():
    encoded = cv2.imencode(".jpg", np.full((12, 12, 3), 120, dtype=np.uint8))[1]
    return Page(id="page_1", activity="com.example.Main", representative_screenshot=base64.b64encode(encoded).decode())


def review_job(status="completed"):
    return {
        "iteration_count": 3,
        "package_name": "com.example",
        "current_test_scenario": "\u89c2\u5bdf\u6807\u9898\u6e32\u67d3",
        "current_plan": {"scenario_type": "intra_test", "target_function_id": "func_1"},
        "execution_result": {"scenario_status": status},
        "scenario_execution_trace": [
            {"action_name": "click", "x": 120, "y": 220, "trace_phase": "test", "current_page": image_page()},
            {"action_name": "capture_screenshot", "terminal_observation": True, "observation_only": True,
             "trace_phase": "test", "current_page": image_page()},
        ],
    }


def resume_directory(root, mode="defect"):
    FunctionalFlowGraph().save(str(root))
    (root / "run_result.json").write_text(json.dumps({
        "status": "success", "mode": mode, "package_name": "com.example",
        "app_name": "Example", "iteration_count": 12, "elapsed_seconds": 600,
    }))
    (root / "bugs.json").write_text("[]")
    (root / "activity_timeline.json").write_text('{"activities": [], "points": []}')


def test_resume_restores_bug_actions_and_images_before_first_snapshot(tmp_path):
    resume_directory(tmp_path)
    trace = review_job()["scenario_execution_trace"]
    bug = {"bug_id": "BUG-001", "confidence": "high", "description": "\u6807\u9898\u91cd\u53e0"}
    (tmp_path / "bugs.json").write_text(json.dumps([bug]))
    main._save_bug_folders({"bug_trace_records": {"BUG-001": trace}}, str(tmp_path), [bug], "test")
    trace_path = tmp_path / "bugs/BUG-001/trace.json"
    old_steps = json.loads(trace_path.read_text())
    old_image = (trace_path.parent / old_steps[0]["page"]["screenshot"]).read_bytes()

    args = options("--resume-output", str(tmp_path))
    validate_resume_options(args)
    restored = main._load_resume_state(args, str(tmp_path), time.time())
    bug_detection_worker.configure(None, initial_bugs=restored["detected_bugs"], initial_records=restored["bug_trace_records"])
    main._save_run_snapshot(restored, str(tmp_path))

    steps = json.loads(trace_path.read_text())
    assert len(steps) == 2
    assert steps[0]["operation"] == old_steps[0]["operation"]
    assert (trace_path.parent / steps[0]["page"]["screenshot"]).read_bytes() == old_image
    assert restored["bug_trace_records"]["BUG-001"][0]["x"] == 120


@pytest.mark.parametrize("record", [None, []])
def test_missing_bug_trace_cannot_overwrite_existing_with_current_scene(tmp_path, record):
    bug = {"bug_id": "BUG-001", "confidence": "high"}
    main._save_bug_folders({"bug_trace_records": {"BUG-001": review_job()["scenario_execution_trace"]}}, str(tmp_path), [bug], "test")
    trace_path = tmp_path / "bugs/BUG-001/trace.json"
    before = trace_path.read_bytes()
    unrelated = [{"action_name": "type", "content": "unrelated", "current_page": image_page()}]
    main._save_bug_folders({"bug_trace_records": {"BUG-001": record}, "scenario_execution_trace": unrelated}, str(tmp_path), [bug], "test")
    assert trace_path.read_bytes() == before


@pytest.mark.parametrize("mode", ["defect", "compliance"])
def test_resume_runtime_failure_preserves_checkpoint_identity(monkeypatch, tmp_path, mode):
    resume_directory(tmp_path, mode)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    monkeypatch.setattr(main, "_init_runtime", lambda _: (_ for _ in ()).throw(ValueError("device disconnected")))
    args = options("--mode", mode, "--resume-output", str(tmp_path))
    assert main._run_single(args) == 1
    assert all(path.read_bytes() == content for path, content in before.items())
    validate_resume_options(args)
    assert args.package_name == "com.example"


def test_corrupt_resume_trace_fails_before_any_device_access(monkeypatch, tmp_path):
    resume_directory(tmp_path)
    folder = tmp_path / "bugs/BUG-001"
    folder.mkdir(parents=True)
    (tmp_path / "bugs.json").write_text('[{"bug_id": "BUG-001"}]')
    (folder / "bug.json").write_text('{"bug_id": "BUG-001"}')
    (folder / "trace.json").write_text("broken")
    original = (tmp_path / "run_result.json").read_bytes()
    monkeypatch.setattr(main, "_init_runtime", lambda _: pytest.fail("corrupt resume must not touch device"))
    assert main._run_single(options("--resume-output", str(tmp_path))) == 1
    assert (tmp_path / "run_result.json").read_bytes() == original


def test_startup_failure_without_output_directory_does_not_write_to_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main, "_init_runtime", lambda _: (_ for _ in ()).throw(ValueError("device disconnected")))
    assert main._run_single(options("--package-name", "com.example")) == 1
    assert list(tmp_path.iterdir()) == []


class FailingModel:
    def invoke(self, _messages):
        raise TimeoutError("offline simulated timeout")


def offline_runtime(monkeypatch, llm, graph):
    monkeypatch.setattr(main, "_init_runtime", lambda _: (llm, llm, llm, llm, object()))
    monkeypatch.setattr(main, "_start_target_app", lambda *args, **kwargs: None)
    monkeypatch.setattr(main, "setup_logcat_capture", lambda *args, **kwargs: None)
    monkeypatch.setattr(main, "build_graph", lambda *args, **kwargs: graph)


def test_detector_timeout_fails_run_and_survives_resume(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_RETRY_ATTEMPTS", "1")

    class Graph:
        def stream(self, state, config):
            bug_detection_worker.submit(review_job())
            yield {"ffg_update": {"ffg": state["ffg"]}}

    offline_runtime(monkeypatch, FailingModel(), Graph())
    assert main._run_single(options("--package-name", "com.example", "--output-dir", str(tmp_path))) == 1
    result = json.loads((tmp_path / "run_result.json").read_text())
    manifest = json.loads((tmp_path / "artifact_manifest.json").read_text())
    assert result["status"] == "failed"
    assert manifest["status"] == "incomplete"
    assert result["pending_reviews"]["defect"] == 0
    assert len(result["defect_review_failures"]) == 1
    assert result["defect_review_failures"][0]["iteration_count"] == 3
    assert any("Bug review" in error for error in manifest["errors"])
    assert len(json.loads((tmp_path / "defect_review_failures.json").read_text())) == 1
    args = options("--resume-output", str(tmp_path))
    validate_resume_options(args)
    restored = main._load_resume_state(args, str(tmp_path), time.time())
    bug_detection_worker.configure(None, initial_failures=restored["defect_review_failures"])
    assert len(bug_detection_worker.failures_snapshot()) == 1


def compliance_review_job():
    return {
        "iteration_count": 5,
        "package_name": "com.example",
        "current_test_scenario": "\u68c0\u67e5\u540c\u610f\u6d41\u7a0b",
        "current_plan": {"scenario_type": "intra_test", "compliance_check_ids": ["R014"]},
        "scenario_execution_trace": [
            {"action_name": "click", "x": 120, "y": 220, "trace_phase": "test", "current_page": image_page()},
            {"action_name": "capture_screenshot", "terminal_observation": True, "observation_only": True,
             "trace_phase": "test", "current_page": image_page()},
        ],
    }


def test_compliance_detector_timeout_fails_run_and_survives_resume(monkeypatch, tmp_path):
    """Exhausted compliance detection retries must mark artifacts incomplete and survive resume; an undetermined fallback must not conceal infrastructure failure behind success."""
    monkeypatch.setenv("LLM_RETRY_ATTEMPTS", "1")

    class Graph:
        def stream(self, state, config):
            compliance_worker.submit(build_compliance_detection_job(compliance_review_job()))
            yield {"ffg_update": {"ffg": state["ffg"]}}

    offline_runtime(monkeypatch, FailingModel(), Graph())
    assert main._run_single(options(
        "--mode", "compliance", "--no-reset-app-data",
        "--package-name", "com.example", "--output-dir", str(tmp_path),
    )) == 1
    result = json.loads((tmp_path / "run_result.json").read_text())
    manifest = json.loads((tmp_path / "artifact_manifest.json").read_text())
    assert result["status"] == "failed"
    assert manifest["status"] == "incomplete"
    assert result["pending_reviews"]["compliance"] == 0
    assert len(result["compliance_review_failures"]) == 1
    assert result["compliance_review_failures"][0]["iteration_count"] == 5
    assert result["compliance_review_failures"][0]["check_ids"] == ["R014"]
    assert result["compliance_review_failures"][0]["error_type"] == "ComplianceReviewError"
    assert "TimeoutError" in result["compliance_review_failures"][0]["error"]
    assert any("Compliance review" in error for error in manifest["errors"])
    assert len(json.loads((tmp_path / "compliance_review_failures.json").read_text())) == 1
    args = options("--mode", "compliance", "--resume-output", str(tmp_path))
    validate_resume_options(args)
    restored = main._load_resume_state(args, str(tmp_path), time.time())
    compliance_worker.configure(None, initial_failures=restored["compliance_review_failures"])
    assert len(compliance_worker.failures_snapshot()) == 1


def test_compliance_review_call_failure_raises_instead_of_masking_as_omitted_clause(monkeypatch):
    from workflow.compliance_detection import ComplianceReviewError, analyze_execution_for_compliance

    monkeypatch.setenv("LLM_RETRY_ATTEMPTS", "1")
    llm = SimpleNamespace(invoke=lambda _: (_ for _ in ()).throw(TimeoutError("offline simulated timeout")))
    with pytest.raises(ComplianceReviewError):
        analyze_execution_for_compliance(llm, compliance_review_job())


@pytest.mark.parametrize("response", ["{}", '{"bugs": {}}', '{"bugs": [null]}'])
def test_invalid_detector_schema_is_a_failure(response):
    llm = SimpleNamespace(invoke=lambda _: SimpleNamespace(content=response))
    with pytest.raises(DefectReviewError):
        analyze_execution_for_bugs(llm, review_job())


def test_valid_empty_bug_list_is_successful():
    llm = SimpleNamespace(invoke=lambda _: SimpleNamespace(content='{"bugs": []}'))
    assert analyze_execution_for_bugs(llm, review_job())["detected_bugs"] == []


def test_failed_review_submission_is_persisted_as_failure(monkeypatch):
    def broken_submit(_job):
        raise RuntimeError("queue unavailable")

    monkeypatch.setattr(scenario_exec, "create_scenario_execution_pass", lambda *_: lambda _: {})
    monkeypatch.setattr(bug_detection_worker, "submit", broken_submit)
    node = scenario_exec.create_scenario_exec_subgraph(object(), object())
    node.invoke({"bug_detection_enabled": True, **review_job()})
    assert len(bug_detection_worker.failures_snapshot()) == 1


def test_sigterm_bypasses_node_recovery_and_model_retry(monkeypatch):
    calls = []

    def terminate(_messages):
        calls.append(1)
        main._raise_run_termination(signal.SIGTERM, None)

    llm = SimpleNamespace(invoke=terminate)
    node = _safe_node("scenario_plan", lambda _: invoke_llm_json_with_retry(llm, [], max_attempts=3), _fallback_scenario_plan)
    with pytest.raises(main.RunTerminationSignal):
        node({})
    assert calls == [1]


def test_sigterm_reaches_top_level_shutdown_and_saves_snapshot(monkeypatch, tmp_path):
    class Graph:
        def stream(self, state, config):
            yield {"scenario_plan": {"iteration_count": 7}}
            _safe_node("scenario_plan", lambda _: main._raise_run_termination(signal.SIGTERM, None), _fallback_scenario_plan)({})

    offline_runtime(monkeypatch, object(), Graph())
    assert main._run_single(options("--package-name", "com.example", "--output-dir", str(tmp_path))) == 143
    result = json.loads((tmp_path / "run_result.json").read_text())
    assert result["exit_code"] == 143
    assert result["iteration_count"] == 7
    assert result["fatal_error"]["type"] == "RunTerminationSignal"
    assert (tmp_path / "ffg.json").is_file()


@pytest.mark.parametrize("returncode,stdout,stderr", [
    (1, "", "error: device not found"), (1, "", "device offline"),
    (0, "Error: package service unavailable", ""),
])
def test_cleanup_query_failure_never_verifies_absence(monkeypatch, returncode, stdout, stderr):
    monkeypatch.setattr(batch_test_apks, "_adb", lambda *args, **kwargs: subprocess.CompletedProcess([], returncode, stdout, stderr))
    result = batch_test_apks.cleanup_package("fake", "com.example")
    assert result["verified_absent"] is False
    assert "package_absence_unverified" in result["errors"]


@pytest.mark.parametrize("output,installed", [("", False), ("package:com.example.extra\n", False), ("package:com.example\n", True)])
def test_installation_query_requires_success_and_exact_package(monkeypatch, output, installed):
    monkeypatch.setattr(batch_test_apks, "_adb", lambda *args, **kwargs: subprocess.CompletedProcess([], 0, output, ""))
    assert batch_test_apks.is_installed("fake", "com.example") is installed


def test_completed_batch_entry_cannot_be_reused_for_another_package(monkeypatch, tmp_path):
    old = batch_compliance_run.AppEntry(1, "Same name", "com.old")
    new = batch_compliance_run.AppEntry(1, "Same name", "com.new")
    folder = tmp_path / old.dir_name
    folder.mkdir()
    (folder / "run_result.json").write_text('{"status": "success", "mode": "compliance", "package_name": "com.old"}')
    (folder / "artifact_manifest.json").write_text('{"status": "complete"}')
    (folder / "compliance_report.json").write_text('{"checks": [], "summary": {"violation": 5}}')
    assert batch_compliance_run.run_artifacts_complete(folder, old.package)
    assert not batch_compliance_run.run_artifacts_complete(folder, new.package)
    batch_compliance_run.build_batch_report(tmp_path, [new])
    report = json.loads((tmp_path / "batch_report.json").read_text())[0]
    assert report["run_status"] == "not_run"
    assert report["compliance"]["available"] is False
    monkeypatch.setattr(batch_compliance_run, "wait_for_device", lambda _: True)
    monkeypatch.setattr(batch_compliance_run, "app_installed", lambda *_: True)
    ran = []
    monkeypatch.setattr(batch_compliance_run, "_run_one", lambda entry, **kwargs: ran.append(entry.package) or {"ok": True})
    # main installs signal handlers for the batch; keep this test isolated.
    monkeypatch.setattr(batch_compliance_run.signal, "signal", lambda *args: None)
    assert batch_compliance_run.main(["--device-id", "fake", "--app", "Same name:com.new", "--output-root", str(tmp_path)]) == 0
    assert ran == ["com.new"]
    assert list(tmp_path.glob(old.dir_name + ".previous-*"))


def test_batch_report_totals_exclude_failed_runs(tmp_path):
    """Exclude failed or unfinished runs from violation totals even if they retain a compliance report; keep totals consistent with the successful-run count."""

    def write_app(entry, *, status: str, violation_count: int) -> None:
        folder = tmp_path / entry.dir_name
        folder.mkdir()
        (folder / "run_result.json").write_text(json.dumps({
            "status": status, "mode": "compliance", "package_name": entry.package,
        }))
        (folder / "artifact_manifest.json").write_text('{"status": "complete"}')
        checks = [
            {"rule_id": f"R{i:03d}", "verdict": "violation", "clause": "\u793a\u4f8b\u6761\u6b3e", "description": "\u793a\u4f8b\u8bf4\u660e"}
            for i in range(violation_count)
        ]
        (folder / "compliance_report.json").write_text(json.dumps({
            "checks": checks,
            "summary": {"gui_total": 10, "processed": 10, "violation": violation_count},
        }))

    ok_app = batch_compliance_run.AppEntry(1, "\u6210\u529fApp", "com.ok")
    failed_app = batch_compliance_run.AppEntry(2, "\u5931\u8d25App", "com.failed")
    write_app(ok_app, status="success", violation_count=2)
    write_app(failed_app, status="failed", violation_count=5)

    markdown_path = batch_compliance_run.build_batch_report(tmp_path, [ok_app, failed_app])
    markdown = markdown_path.read_text(encoding="utf-8")

    # Totals include only the two violations from successful runs, excluding five from a failed run.
    assert "2 total across 2 distinct rules" in markdown
    assert "7 total" not in markdown

    # Keep the failed app visible in detail rows.
    rows = json.loads((tmp_path / "batch_report.json").read_text())
    failed_row = next(row for row in rows if row["package_name"] == "com.failed")
    assert failed_row["run_status"] == "failed"
    assert failed_row["compliance"]["violation"] == 5


@pytest.mark.parametrize("status", ["completed", "incomplete"])
def test_step_limit_with_final_observation_keeps_defect_review(status):
    bug = {"type": "ui_error", "evidence_category": "rendering_corruption", "confidence": "high",
           "description": "\u6807\u9898\u6587\u5b57\u4e0e\u6309\u94ae\u91cd\u53e0", "expected": "\u6807\u9898\u6e05\u6670", "actual": "\u6807\u9898\u91cd\u53e0",
           "evidence": [{"global_step": 2, "role": "after"}]}
    llm = SimpleNamespace(invoke=lambda _: SimpleNamespace(content=json.dumps({"bugs": [bug]})))
    result = analyze_execution_for_bugs(llm, review_job(status))
    assert len(result["detected_bugs"]) == 1


@pytest.mark.parametrize("change", [{"terminal_observation": False}, {"current_page": Page()}, {"screenshot_health_error": "bad frame"}])
def test_step_limit_without_reliable_final_observation_still_skips(change):
    job = review_job("incomplete")
    job["scenario_execution_trace"][-1].update(change)
    llm = SimpleNamespace(invoke=lambda _: pytest.fail("no final evidence"))
    assert analyze_execution_for_bugs(llm, job)["detected_bugs"] == []
