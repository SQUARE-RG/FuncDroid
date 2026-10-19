# Multilingual UI and model-response fixtures use Unicode escapes to preserve their exact text.
import base64
import json
from types import SimpleNamespace

import pytest

import main
from compliance.checkpoints import ComplianceChecklist
from compliance.report import build_compliance_report
from model.flow_graph import Page
from workflow.compliance_worker import ComplianceDetectionWorker, build_compliance_detection_job
from workflow.run_options import validate_resume_options


def options(*extra):
    return main._build_parser().parse_args(["--device-id", "device-1", *extra])


@pytest.mark.parametrize("mode,bug,compliance,reset", [
    ("defect", True, False, False), ("compliance", False, True, True),
])
def test_modes_have_independent_state_and_initialization(mode, bug, compliance, reset):
    args = options("--mode", mode, "--package-name", "com.example")
    state = main._build_initial_state(args, 1.0)
    assert state["run_mode"] == mode
    assert state["bug_detection_enabled"] is bug
    assert state["compliance_detection_enabled"] is compliance
    assert state["reset_app_data"] is reset


@pytest.mark.parametrize("extra", [
    ["--mode", "defect", "--compliance"],
    ["--mode", "compliance", "--bug-detection", "--package-name", "com.example"],
    ["--compliance", "--bug-detection", "--package-name", "com.example"],
    ["--mode", "compliance"],
    ["--resume-output", "previous", "--reset-app-data"],
    ["--mode", "defect", "--rules", "rules.json"],
])
def test_conflicting_or_unsafe_options_rejected_before_runtime(extra):
    with pytest.raises(SystemExit):
        options(*extra)


def test_compliance_can_explicitly_use_current_existing_state():
    args = options("--mode", "compliance", "--no-reset-app-data")
    assert args.compliance and not args.reset_app_data


def test_legacy_compliance_switch_selects_independent_mode():
    args = options("--compliance", "--package-name", "com.example")
    assert args.mode == "compliance" and not args.bug_detection


@pytest.mark.parametrize("changes", [{"mode": "compliance"}, {"package_name": "com.other"}])
def test_resume_rejects_another_mode_or_app(tmp_path, changes):
    (tmp_path / "run_result.json").write_text(json.dumps({"mode": "defect", "package_name": "com.example"}))
    args = options("--resume-output", str(tmp_path))
    for key, value in changes.items():
        setattr(args, key, value)
    with pytest.raises(ValueError):
        validate_resume_options(args)


def test_resume_uses_recorded_package_and_rule_snapshot(tmp_path):
    (tmp_path / "run_result.json").write_text(json.dumps({"mode": "compliance", "package_name": "com.example"}))
    snapshot = tmp_path / "compliance_rules.json"
    snapshot.write_text('{"rules": []}')
    args = options("--mode", "compliance", "--resume-output", str(tmp_path))
    validate_resume_options(args)
    assert args.package_name == "com.example"
    assert args.rules == str(snapshot)
    assert not args.reset_app_data
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps({"rules": [{"rule_id": "R1", "clause": "\u793a\u4f8b\u6761\u6b3e"}]}))
    args.rules = str(changed)
    with pytest.raises(ValueError, match="rule library"):
        validate_resume_options(args)


def test_resume_accepts_same_rules_wrapped_with_version_field(tmp_path):
    """Resume compares the normalized effective rule set, accepting equivalent version wrappers and filtered invalid entries."""
    (tmp_path / "run_result.json").write_text(json.dumps({"mode": "compliance", "package_name": "com.example"}))
    rule = {"rule_id": "R1", "clause": "\u793a\u4f8b\u6761\u6b3e"}
    snapshot = tmp_path / "compliance_rules.json"
    snapshot.write_text(json.dumps({"rules": [rule]}))
    original_library = tmp_path / "rules.json"
    original_library.write_text(json.dumps({
        "version": 1,
        "rules": [rule, {"rule_id": "", "clause": "\u7f3a\u5c11 rule_id \u4f1a\u88ab\u8fc7\u6ee4"}],
    }))
    args = options("--mode", "compliance", "--resume-output", str(tmp_path))
    args.rules = str(original_library)
    validate_resume_options(args)
    assert args.rules == str(snapshot)


def test_invalid_resume_does_not_initialize_device(monkeypatch, tmp_path):
    (tmp_path / "run_result.json").write_text(json.dumps({"mode": "compliance", "package_name": "com.example"}))
    monkeypatch.setattr(main, "_init_runtime", lambda _args: pytest.fail("must validate before device access"))
    with pytest.raises(ValueError, match="mode"):
        main._run_single(options("--resume-output", str(tmp_path)))


@pytest.mark.parametrize("reset", [False, True])
def test_startup_only_clears_data_when_requested(monkeypatch, reset):
    cleared = []
    monkeypatch.setattr(main, "_reset_android_app_data_at_run_start", lambda *args: cleared.append(args) or True)
    monkeypatch.setattr(main.time, "sleep", lambda _: None)
    calls = []
    device = SimpleNamespace(start_app=lambda name: calls.append(("start", name)),
                             stop_app=lambda name: calls.append(("stop", name)))
    main._start_target_app(device, "com.example", reset_app_data=reset)
    assert bool(cleared) is reset
    assert calls == [("stop", "com.example"), ("start", "com.example")]


@pytest.mark.parametrize("mode,grant", [("defect", True), ("compliance", False)])
def test_permission_granting_defaults_follow_the_detector_goal(mode, grant):
    # Defect mode pregrants permissions; compliance mode tests permission requests and does not.
    args = options("--mode", mode, "--package-name", "com.example")
    assert args.grant_permissions is grant


def test_explicit_permission_flags_override_the_mode_default():
    assert options("--mode", "compliance", "--package-name", "com.example",
                   "--grant-permissions").grant_permissions
    assert not options("--mode", "defect", "--package-name", "com.example",
                       "--no-grant-permissions").grant_permissions


def test_granting_permissions_is_skipped_when_the_package_is_unknown():
    # Without --package-name, test the foreground app without a target package for permission grants.
    args = options("--mode", "defect", "--grant-permissions")
    assert args.grant_permissions and not args.package_name


def test_bounded_grant_skips_unknown_package(monkeypatch, capsys):
    monkeypatch.setattr(main.time, "sleep", lambda _: None)
    granted = []
    device = SimpleNamespace(
        start_app=lambda name: None,
        stop_app=lambda name: None,
        grant_app_permissions=lambda name: granted.append(name) or {},
    )
    report = main._start_target_app(device, "", grant_permissions=True)
    assert report == {} and not granted
    assert "skip permission pre-granting" in capsys.readouterr().out


def test_permission_grants_are_idempotent_across_resumed_runs():
    # Resume preserves existing permissions and must not clear app data again.
    args = options("--mode", "defect", "--package-name", "com.example",
                   "--resume-output", "previous", "--no-grant-permissions")
    assert not args.grant_permissions and not args.reset_app_data


@pytest.mark.parametrize("grant", [False, True])
def test_startup_only_grants_permissions_when_requested(monkeypatch, grant):
    granted = []
    monkeypatch.setattr(main.time, "sleep", lambda _: None)
    device = SimpleNamespace(
        start_app=lambda name: None,
        stop_app=lambda name: None,
        grant_app_permissions=lambda name: granted.append(name) or {
            "granted": ["android.permission.CAMERA"], "missing": [], "failed": [], "unavailable": False,
        },
    )
    report = main._start_target_app(device, "com.example", grant_permissions=grant)
    assert bool(granted) is grant
    assert bool(report) is grant


def test_unavailable_permission_granting_does_not_abort_startup(monkeypatch, capsys):
    monkeypatch.setattr(main.time, "sleep", lambda _: None)
    device = SimpleNamespace(
        start_app=lambda name: None,
        stop_app=lambda name: None,
        grant_app_permissions=lambda name: (_ for _ in ()).throw(RuntimeError("no adb")),
    )
    report = main._start_target_app(device, "com.example", grant_permissions=True)
    assert report["unavailable"] is True
    assert "Pre-granting failed; continuing" in capsys.readouterr().out


def test_run_result_records_the_permission_grant_report(tmp_path):
    # Permission grants occur before _prepare_workflow_state reconstructs state; write the report afterward.
    # Otherwise run_result.json retains null and cannot show which permissions were granted.
    state = main._build_initial_state(
        options("--mode", "defect", "--package-name", "com.example"), 1.0
    )
    state["permission_grant_report"] = {
        "granted": ["android.permission.CAMERA"], "missing": [], "failed": [], "unavailable": False,
    }
    main._save_run_result(state, str(tmp_path), exit_code=0, elapsed_seconds=1.0)
    result = json.loads((tmp_path / "run_result.json").read_text())
    assert result["permission_grant_report"]["granted"] == ["android.permission.CAMERA"]
    assert result["grant_permissions"] is True


def test_unsupported_device_cannot_claim_successful_data_reset():
    assert not main._reset_android_app_data_at_run_start(
        SimpleNamespace(operating_system="hdc"), "com.example"
    )


def test_compliance_report_has_round_specific_reproduction_evidence(monkeypatch, tmp_path):
    worker = ComplianceDetectionWorker()
    worker._on_trace = lambda job: main._save_compliance_trace(job, str(tmp_path))
    # Use a real image so persisted evidence can be checked independently.
    import cv2
    import numpy as np
    screenshot = base64.b64encode(cv2.imencode(".png", np.zeros((4, 4, 3), dtype=np.uint8))[1]).decode()
    page = Page(name="permission", representative_screenshot=screenshot)
    job = build_compliance_detection_job({
        "iteration_count": 7, "current_plan": {"compliance_check_ids": ["P03"]},
        "scenario_execution_trace": [{"current_page": page, "action_name": "click", "action_description": "\u62d2\u7edd"}],
    })
    monkeypatch.setattr("workflow.compliance_worker.analyze_execution_for_compliance", lambda *_: {
        "check_results": [{"check_id": "P03", "verdict": "violation", "evidence": [{"global_step": 1, "role": "before"}]}]
    })
    worker._process_job(job)
    report = build_compliance_report(ComplianceChecklist.from_state(worker.snapshot()))
    check = next(item for item in report["checks"] if item["rule_id"] == "P03")
    trace_path = tmp_path / check["trace_path"]
    trace = json.loads(trace_path.read_text())
    assert trace["iteration_count"] == 7
    assert trace["steps"][0]["action_description"] == "\u62d2\u7edd"
    assert (trace_path.parent / trace["steps"][0]["page"]["screenshot"]).is_file()
    second = build_compliance_detection_job({"iteration_count": 7})
    assert second["trace_id"] != job["trace_id"]


def test_failed_submission_releases_executing_checks():
    """Submission/execution failures return checks to pending and refund attempts because no genuine classification occurred."""
    worker = ComplianceDetectionWorker()
    worker.mark_executing(["R014"])
    worker.defer_checks(["R014"], "queue rejected")
    snapshot = worker.snapshot()
    assert snapshot["status"]["R014"] == "pending"
    assert snapshot["attempts"]["R014"] == 0
    assert snapshot["findings"] == []


def test_repeated_submission_failures_do_not_exhaust_retry_budget():
    """Repeated infrastructure submission failures must not exhaust the classification budget or enter discovery-baseline cooldown."""
    from compliance.checkpoints import MAX_COMPLIANCE_ATTEMPTS

    worker = ComplianceDetectionWorker()
    for _ in range(MAX_COMPLIANCE_ATTEMPTS + 1):
        worker.mark_executing(["R014"])
        worker.defer_checks(["R014"], "queue rejected")
    snapshot = worker.snapshot()
    assert snapshot["status"]["R014"] == "pending"
    assert snapshot["attempts"]["R014"] == 0
    assert snapshot["discovery_baselines"] == {}


def test_outputs_only_contain_the_selected_detector(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "_sync_compliance_state", lambda _: None)
    state = {"run_mode": "compliance", "compliance_detection_enabled": True,
             "bug_detection_enabled": False, "compliance_checklist": {}}
    main._save_outputs(state, str(tmp_path), "test")
    assert (tmp_path / "compliance_report.json").is_file()
    assert not (tmp_path / "bugs.json").exists()
    errors = main._validate_final_artifacts(state, str(tmp_path), 0)["errors"]
    assert not any("bugs.json" in error for error in errors)
    (tmp_path / "compliance_report.json").write_text('{"broken"')
    errors = main._validate_final_artifacts(state, str(tmp_path), 0)["errors"]
    assert any("Compliance" in error or "compliance_report" in error for error in errors)


def test_compliance_completion_does_not_reopen_for_unused_time(monkeypatch):
    from workflow.graph import _create_scenario_plan_node_with_iteration
    import time

    planner = _create_scenario_plan_node_with_iteration(lambda _: {"workflow_complete": True})
    result = planner({"run_mode": "compliance", "start_time": time.time(), "total_time_limit": 600})
    assert result["workflow_complete"] is True


@pytest.mark.parametrize("mode", ["defect", "compliance"])
def test_execution_routes_to_only_the_selected_reviewer(monkeypatch, mode):
    from workflow.subgraphs import scenario_exec

    state = main._build_initial_state(options("--mode", mode, "--package-name", "com.example"), 1)
    state["current_plan"] = {"compliance_check_ids": ["R014"]} if mode == "compliance" else {}
    submitted = []
    monkeypatch.setattr(scenario_exec, "create_scenario_execution_pass", lambda *_: lambda _: {})
    monkeypatch.setattr(scenario_exec.bug_detection_worker, "submit", lambda _: submitted.append("defect") or True)
    monkeypatch.setattr(scenario_exec.compliance_worker, "submit", lambda _: submitted.append("compliance") or True)
    scenario_exec.create_scenario_exec_subgraph(object(), object()).invoke(state)
    assert submitted == [mode]


def test_batch_requires_reports_and_manifest_before_skipping(tmp_path):
    from scripts.batch_compliance_run import run_artifacts_complete

    (tmp_path / "run_result.json").write_text('{"status": "success", "mode": "compliance", "package_name": "com.example"}')
    assert not run_artifacts_complete(tmp_path, "com.example")
    (tmp_path / "artifact_manifest.json").write_text('{"status": "complete"}')
    assert not run_artifacts_complete(tmp_path, "com.example")
    (tmp_path / "compliance_report.json").write_text('{"checks": []}')
    assert run_artifacts_complete(tmp_path, "com.example")
    (tmp_path / "artifact_manifest.json").write_text('{"status": "incomplete"}')
    assert not run_artifacts_complete(tmp_path, "com.example")


def test_compliance_rejects_nonexistent_evidence_step(monkeypatch):
    from workflow import compliance_detection

    monkeypatch.setattr(compliance_detection, "_compact_screenshot", lambda s: s)
    monkeypatch.setattr(compliance_detection, "invoke_llm_json_with_retry", lambda *_args, **_kwargs: {
        "check_results": [{"check_id": "R014", "verdict": "compliant", "evidence": [{"global_step": 999, "role": "before"}]}]
    })
    result = compliance_detection.analyze_execution_for_compliance(object(), {
        "current_plan": {"compliance_check_ids": ["R014"]},
        "scenario_execution_trace": [{"current_page": Page(representative_screenshot="fake-image")}],
    })
    assert result["check_results"][0]["verdict"] == "undetermined"
    assert not result["check_results"][0]["evidence"]


def test_startup_config_error_prints_one_line_and_exits_2(monkeypatch, capsys):
    import sys

    import pytest

    monkeypatch.setattr(sys, "argv", ["main.py", "--device-id", "d1", "--output-dir", "/tmp/a",
                                      "--resume-output", "/tmp/b"])
    monkeypatch.setattr(main, "_ensure_device_id", lambda args: None)
    with pytest.raises(SystemExit) as exc_info:
        main.main()
    assert exc_info.value.code == 2
    assert "[Argument error]" in capsys.readouterr().err


def _args_with_package(package_name):
    return SimpleNamespace(package_name=package_name)


def test_stop_target_app_processes_stops_package_at_finish(capsys):
    stopped = []
    resources = main.RunResources(
        device=SimpleNamespace(stop_app=stopped.append))
    main._stop_target_app_processes(_args_with_package("com.example.app"), resources)
    assert stopped == ["com.example.app"]
    assert "Stop target app background processes" in capsys.readouterr().out


def test_stop_target_app_processes_skips_without_device_or_package():
    stopped = []
    # Do not clean up when the device was never initialized.
    main._stop_target_app_processes(_args_with_package("com.example.app"), main.RunResources())
    # Do not clean up without an explicit or detected package name.
    resources_with_device = main.RunResources(device=SimpleNamespace(stop_app=stopped.append))
    main._stop_target_app_processes(_args_with_package(""), resources_with_device)
    assert stopped == []


def test_stop_target_app_processes_swallows_stop_failures(capsys):
    def broken_stop(_package_name):
        raise RuntimeError("device gone")

    resources = main.RunResources(device=SimpleNamespace(stop_app=broken_stop))
    main._stop_target_app_processes(_args_with_package("com.example.app"), resources)
    assert "Failed to stop target app processes; continuing" in capsys.readouterr().out
