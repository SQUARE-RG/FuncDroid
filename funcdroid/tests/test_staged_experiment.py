# Multilingual UI and model-response fixtures use Unicode escapes to preserve their exact text.
import argparse
import base64
import json
import zipfile
from types import SimpleNamespace

import pytest

import main
from model.flow_graph import FunctionalFlowGraph, FunctionalityNode, Page
from scripts import staged_apk_experiment as staged
from scripts.review_compliance_trace import load_trace
from workflow.seed_graph import load_seed_graph


def seed_run(root):
    graph = FunctionalFlowGraph()
    graph.add_node(FunctionalityNode(id="f1", name="\u7ffb\u8bd1", test_status="completed", scenarios=["\u5df2\u6d4b\u8bd5"] ))
    graph.add_page(Page(id="p1", controls=["\u7ffb\u8bd1"], operations=["\u70b9\u51fb\u7ffb\u8bd1"],
                        operation_bindings=[{"operation": "\u70b9\u51fb\u7ffb\u8bd1", "control": "\u7ffb\u8bd1"}],
                        representative_screenshot=base64.b64encode(b"image bytes").decode()), "f1")
    graph.save(str(root))
    (root / "run_result.json").write_text(json.dumps({"package_name": "com.example",
        "status": "success", "mode": "compliance", "elapsed_seconds": 3600}))
    (root / "artifact_manifest.json").write_text('{"status":"complete"}')


def test_seed_resets_coverage_and_preserves_images_without_changing_source(tmp_path):
    seed_run(tmp_path)
    before = {p.name: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    graph, provenance = load_seed_graph(str(tmp_path), "com.example")
    assert graph.pages["p1"].representative_screenshot == base64.b64encode(b"image bytes").decode()
    assert graph.pages["p1"].controls == ["\u7ffb\u8bd1"]
    assert graph.pages["p1"].operations == []
    assert graph.pages["p1"].operation_bindings == []
    assert graph.functions["f1"].test_status == "not_started"
    assert graph.functions["f1"].scenarios == []
    assert provenance["source_elapsed_seconds"] == 3600
    assert before == {p.name: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    output = tmp_path / "new"
    graph.save(str(output))
    assert (output / "screenshots/p1_representative.jpg").read_bytes() == b"image bytes"


@pytest.mark.parametrize("problem", ["wrong_package", "failed", "missing_image", "incomplete"])
def test_invalid_seed_rejected_before_device_access(tmp_path, monkeypatch, problem):
    seed_run(tmp_path)
    if problem == "failed":
        path = tmp_path / "run_result.json"
        value = json.loads(path.read_text())
        value["status"] = "failed"
        path.write_text(json.dumps(value))
    elif problem == "missing_image":
        (tmp_path / "screenshots/p1_representative.jpg").unlink()
    elif problem == "incomplete":
        (tmp_path / "artifact_manifest.json").write_text('{"status":"incomplete"}')
    args = main._build_parser().parse_args(["--device-id", "test", "--seed-ffg", str(tmp_path),
        "--package-name", "wrong" if problem == "wrong_package" else "com.example"])
    monkeypatch.setattr(main, "_init_runtime", lambda *_: pytest.fail("device accessed"))
    assert main._run_single(args) == 1


def test_seed_starts_independent_budget_and_does_not_inherit_findings(tmp_path):
    seed_run(tmp_path)
    graph, provenance = load_seed_graph(str(tmp_path), "com.example")
    args = main._build_parser().parse_args(["--device-id", "test", "--seed-ffg", str(tmp_path),
                                          "--package-name", "com.example", "--total-time", "60"])
    state = main._prepare_workflow_state(args, None, main.RunResources(output_dir=str(tmp_path / "new")),
                                         10000, {"ffg": graph, "seed_provenance": provenance})
    assert state["start_time"] == 10000
    assert state["total_time_limit"] == 3600
    assert state["iteration_count"] == 0
    assert state["detected_bugs"] == []
    assert not state["reset_app_data"]
    assert state["ffg"].pages


@pytest.mark.parametrize("flags", [["--resume-output", "old"], ["--reset-app-data"]])
def test_seed_conflicting_options(flags):
    with pytest.raises(SystemExit):
        main._build_parser().parse_args(["--device-id", "test", "--package-name", "com.example",
                                       "--seed-ffg", "old", *flags])


def test_second_stage_uses_seed_not_resume_and_preserves_data(tmp_path):
    cmd = staged.stage_command(SimpleNamespace(package_name="com.example", app_label="App"),
                               "test", "defect", 60, tmp_path / "defect", tmp_path / "compliance")
    assert "--seed-ffg" in cmd and "--resume-output" not in cmd
    assert "--no-reset-app-data" in cmd


@pytest.mark.parametrize("budget", ["nan", "inf", "0", "-1"])
def test_invalid_stage_budget(budget):
    with pytest.raises(argparse.ArgumentTypeError):
        staged.positive_minutes(budget)


@pytest.mark.parametrize("fail_stage", [None, "compliance", "defect", "interrupt"])
def test_experiment_stages_and_cleanup(tmp_path, monkeypatch, fail_stage):
    info = staged.batch.ApkInfo("example.apk", "hash", "com.example", "1", "1.0", "Example", 21, ())
    monkeypatch.setattr(staged.batch, "inspect_apk", lambda *_: info)
    installed = iter([False, True])
    monkeypatch.setattr(staged.batch, "is_installed", lambda *_: next(installed))
    monkeypatch.setattr(staged.batch, "device_capabilities", lambda *_: (33, set()))
    monkeypatch.setattr(staged, "network_state", lambda *_: {"status": "connected"})
    monkeypatch.setattr(staged.batch, "install_apk", lambda *_: SimpleNamespace(returncode=0, stdout="Success", stderr=""))
    cleanup = []
    monkeypatch.setattr(staged.batch, "cleanup_package", lambda *_: cleanup.append(True) or {"verified_absent": True})
    monkeypatch.setattr(staged.batch, "collect_run_metrics", lambda *_: {})
    monkeypatch.setattr(staged.batch, "run_artifacts_complete", lambda *_: True)
    commands = []
    terminated = []
    monkeypatch.setattr(staged.batch, "_terminate_process_tree", lambda *_: terminated.append(True))

    def child(command, **kwargs):
        commands.append(command)
        mode = command[command.index("--mode") + 1]
        def wait(**_):
            if fail_stage == "interrupt":
                raise staged.batch.BatchInterrupted("test interruption")
            return int(mode == fail_stage)
        return SimpleNamespace(wait=wait)

    monkeypatch.setattr(staged.subprocess, "Popen", child)
    # subprocess.run uses Popen internally; keep git provenance deterministic.
    monkeypatch.setattr(staged.subprocess, "run", lambda command, **_: SimpleNamespace(
        stdout="revision\n" if command[1] == "rev-parse" else b""))
    args = SimpleNamespace(apk="example.apk", device_id="test", output_dir=str(tmp_path / "run"),
                           compliance_minutes=5, defect_minutes=7)
    code = staged.run_experiment(args)
    assert code == int(fail_stage is not None)
    assert cleanup == [True]
    assert len(commands) == (1 if fail_stage in {"compliance", "interrupt"} else 2)
    if len(commands) == 2:
        assert "--seed-ffg" in commands[1]
        assert commands[1][commands[1].index("--total-time") + 1] == "7"
    result = json.loads((tmp_path / "run/experiment.json").read_text())
    assert result["status"] == ("failed" if fail_stage else "completed")
    if fail_stage == "interrupt":
        assert terminated == [True]
        assert result["stages"][0]["finished_at"]
        assert result["stages"][0]["status"] == "interrupted"


def test_preinstalled_app_is_not_cleared(tmp_path, monkeypatch):
    monkeypatch.setattr(staged.batch, "inspect_apk", lambda *_: SimpleNamespace(package_name="com.example"))
    monkeypatch.setattr(staged.batch, "is_installed", lambda *_: True)
    monkeypatch.setattr(staged.batch, "cleanup_package", lambda *_: pytest.fail("existing app cleared"))
    with pytest.raises(ValueError, match="already installed"):
        staged.run_experiment(SimpleNamespace(apk="app.apk", device_id="test", output_dir=str(tmp_path / "run")))
    assert not (tmp_path / "run").exists()


def test_source_snapshot_includes_new_modules_and_excludes_credentials(tmp_path):
    (tmp_path / "main.py").write_text("pass\n")
    (tmp_path / ".env").write_text("API_KEY=do-not-copy")
    (tmp_path / "workflow").mkdir()
    (tmp_path / "workflow/new.py").write_text("pass\n")
    archive = tmp_path / "source.zip"
    hashes = staged.snapshot_sources(tmp_path, archive)
    assert set(hashes) == {"main.py", "workflow/new.py"}
    with zipfile.ZipFile(archive) as source:
        assert set(source.namelist()) == set(hashes)


def test_saved_trace_review_preserves_action_and_resolves_evidence(tmp_path):
    (tmp_path / "shot.jpg").write_bytes(b"image")
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps({"plan": {"compliance_check_ids": ["R062"]}, "scenario": "review",
        "steps": [{"operation": {"action_name": "click", "parameters": {"x": 4, "y": 5}},
                   "action_failed": True, "page": {"screenshot": "shot.jpg"}}]}))
    state = load_trace(trace, tmp_path / "rules.json")
    step = state["scenario_execution_trace"][0]
    assert step["action_name"] == "click" and step["x"] == 4 and step["action_failed"]
    assert step["current_page"]["screenshot"] == base64.b64encode(b"image").decode()


@pytest.mark.parametrize("text,expected", [("Active default network: none", "disconnected"),
    ("Active default network: 101", "connected"), ("", "unknown")])
def test_device_network_state_does_not_use_historical_validated_lines(monkeypatch, text, expected):
    monkeypatch.setattr(staged.batch, "_adb", lambda *_: SimpleNamespace(
        returncode=0, stdout=text + "\nHistorical capabilities: VALIDATED\n"))
    assert staged.network_state("test")["status"] == expected
