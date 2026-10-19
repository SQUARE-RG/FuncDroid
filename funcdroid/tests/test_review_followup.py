# Multilingual UI and model-response fixtures use Unicode escapes to preserve their exact text.
"""Offline regressions for execution, completion and evidence accounting."""

import json
import subprocess
from types import SimpleNamespace

import pytest

from automator.u2 import U2
from compliance.checkpoints import ComplianceChecklist
from compliance.report import build_compliance_report
from model.flow_graph import FlowGraph, FunctionDependency, FunctionNode, Page
from workflow.nodes import scenario_plan as planner
from workflow.nodes import ffg_update as updater


def graph_with_controls():
    graph = FlowGraph()
    node = FunctionNode(id="func_1", name="\u6536\u85cf", test_status="in_progress")
    graph.add_node(node)
    page = Page(id="page_1", name="\u6536\u85cf\u9875", controls=["\u6536\u85cf\u6309\u94ae"], operations=["\u70b9\u51fb\u6536\u85cf\u6309\u94ae"])
    graph.add_page(page, node.id)
    page.add_operation_binding("\u70b9\u51fb\u6536\u85cf\u6309\u94ae", "\u6536\u85cf\u6309\u94ae")
    node.add_scenario("\u70b9\u51fb\u6536\u85cf")
    graph.recompute_all_coverages()
    return graph, node, page


def test_empty_plan_cannot_bypass_verification(monkeypatch):
    graph, node, page = graph_with_controls()
    monkeypatch.setattr(planner, "_ensure_target_app", lambda *a, **kw: page)
    monkeypatch.setattr(planner, "_refresh_current_page_observation", lambda *a: page)
    monkeypatch.setattr(planner, "_invoke_valid_plan", lambda *a, **kw: ({
        "test_scenario": "",
    }, []))
    state = {"ffg": graph, "current_page": page, "run_phase": "intra"}
    plan = planner.create_scenario_plan_node(object(), object())
    # An empty plan gets a template fallback in the same iteration and retains its phase.
    state.update(plan(state))
    assert node.coverage_percent == 100
    assert node.test_status != "completed"
    assert not planner._function_completion_met(graph, node)
    assert state["current_plan"]["function_phase"] == "depth"
    assert state["current_test_scenario"]


@pytest.mark.parametrize("failure", ["exit", "timeout", "missing_binary"])
def test_input_injection_failures_reach_executor(monkeypatch, failure):
    driver = U2.__new__(U2)
    driver._serial = "mock"
    def run(*args, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired("mock", 1)
        if failure == "missing_binary":
            raise FileNotFoundError("mock")
        return SimpleNamespace(returncode=1, stdout="", stderr="injection failed")
    monkeypatch.setattr("automator.u2.subprocess.run", run)
    with pytest.raises(RuntimeError, match="injection failed"):
        driver._shell_input("tap", 10, 20)


@pytest.mark.parametrize("mode", ["defect", "compliance"])
def test_keep_app_reaches_main_as_no_reset(monkeypatch, tmp_path, mode):
    import main
    from scripts import batch_test_apks as batch
    commands = []
    def popen(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(wait=lambda timeout: 0)
    monkeypatch.setattr(batch.subprocess, "Popen", popen)
    monkeypatch.setattr(batch, "is_installed", lambda *a: True)
    monkeypatch.setattr(batch, "collect_run_metrics", lambda *a: {})
    monkeypatch.setattr(batch, "run_artifacts_complete", lambda *a: True)
    def forbidden(*args, **kwargs):
        raise AssertionError("keep_app must preserve the installation")
    monkeypatch.setattr(batch, "cleanup_package", forbidden)
    monkeypatch.setattr(batch, "install_apk", forbidden)
    batch.run_one(batch.ApkInfo("mock.apk", "abc", "com.mock", "1", "1", "Mock", 21, ()),
                  device_id="mock", total_time_minutes=1, max_iterations=1,
                  batch_output_dir=tmp_path, project_root=tmp_path, mode=mode, keep_app=True)
    options = main._build_parser().parse_args(commands[0][2:])
    assert options.reset_app_data is False
    assert options.grant_permissions is (mode == "defect")


def cross_graph():
    graph = FlowGraph()
    for index in (1, 2):
        graph.add_node(FunctionNode(id=f"func_{index}", name=f"\u529f\u80fd{index}"))
    for src, dst, resource in [("func_1", "func_2", "\u8d44\u6e90\u7532"),
                               ("func_2", "func_1", "\u8d44\u6e90\u7532"),
                               ("func_1", "func_2", "\u8d44\u6e90\u4e59")]:
        graph.add_function_dependency(FunctionDependency(src, dst, resource))
    return graph


@pytest.mark.parametrize("outcome", ["done", "failed", "crashed"])
def test_cross_outcomes_preserve_direction_resource_and_evidence(outcome):
    graph = cross_graph()
    state = {"current_plan": {"scenario_type": "cross_test", "src_function_id": "func_1",
                              "dst_function_id": "func_2", "resource_variable": "\u8d44\u6e90\u7532"},
             "current_test_scenario": "\u4fee\u6539\u540e\u8bfb\u56de", "scenario_execution_trace": []}
    result = {"scenario_status": "completed"}
    if outcome == "failed":
        result = {"scenario_status": "failed_goal", "scenario_error": "\u65e0\u6cd5\u6267\u884c"}
    elif outcome == "crashed":
        result = {"scenario_status": "app_crashed", "crash_detected": True}
    counts = updater._record_execution_outcome(graph, state, result)
    assert counts == ({"func_1": 1} if outcome == "crashed" else {})
    restored = FlowGraph.from_dict(json.loads(json.dumps(graph.to_dict())))
    assert restored.cross_test_results[0]["status"] == ("failed" if outcome == "crashed" else outcome)
    # Completed and definitively failed test phases leave the queue without retries.
    assert planner._pending_cross_pairs(restored) == [
        ("func_2", "func_1", "Functional dependency: \u8d44\u6e90\u7532"), ("func_1", "func_2", "Functional dependency: \u8d44\u6e90\u4e59")]
    assert bool(_done_cross_keys(restored)) is (outcome == "done")


def test_legacy_verified_cross_results_load_as_done():
    graph = cross_graph()
    data = graph.to_dict()
    data["cross_test_results"] = [
        {"src_function_id": "func_1", "dst_function_id": "func_2", "resource_variable": "\u8d44\u6e90\u7532",
         "scenario": "\u65e7\u5feb\u7167", "status": status, "verification": {"outcome": "matched"}}
        for status in ("verified",)
    ] + [
        {"src_function_id": "func_2", "dst_function_id": "func_1", "resource_variable": "\u8d44\u6e90\u7532",
         "scenario": "\u65e7\u5feb\u7167", "status": "unverified"}
    ]
    restored = FlowGraph.from_dict(data)
    assert [item["status"] for item in restored.cross_test_results] == ["done", "done"]
    assert all("verification" not in item for item in restored.cross_test_results)


def test_legacy_cross_attempt_is_not_a_cross_result():
    graph = cross_graph()
    graph.add_cross_scenario("func_1", "func_2", "\u81ea\u52a8\u5316\u65e0\u6cd5\u5b8c\u6210")
    assert len(planner._pending_cross_pairs(FlowGraph.from_dict(graph.to_dict()))) == 3
    assert not _done_cross_keys(graph)


def test_cross_failures_are_recorded_per_resource():
    graph = cross_graph()
    state = {"current_plan": {"scenario_type": "cross_test", "src_function_id": "func_1",
                              "dst_function_id": "func_2", "resource_variable": "\u8d44\u6e90\u7532"}}
    result = {"scenario_status": "failed_goal"}
    updater._record_execution_outcome(graph, state, result)
    state["current_plan"]["resource_variable"] = "\u8d44\u6e90\u4e59"
    updater._record_execution_outcome(graph, state, result)
    assert [(item["resource_variable"], item["status"]) for item in graph.cross_test_results] == [
        ("\u8d44\u6e90\u7532", "failed"), ("\u8d44\u6e90\u4e59", "failed")]


def test_functionality_chain_survives_save_reload_report_and_history():
    checklist = ComplianceChecklist()
    chain = {"denial_step": 1, "functional_action_step": 2,
             "result_observation_step": 3, "observed_result": "\u529f\u80fd\u53ef\u7528"}
    checklist.apply_results([{"check_id": "P07", "verdict": "compliant", "functionality_evidence": chain}])
    chain["observed_result"] = "caller mutation"
    restored = ComplianceChecklist.from_state(json.loads(json.dumps(checklist.to_state())))
    row = next(item for item in build_compliance_report(restored)["checks"] if item["rule_id"] == "P07")
    assert row["functionality_evidence"]["observed_result"] == "\u529f\u80fd\u53ef\u7528"
    restored.apply_results([{"check_id": "P07", "verdict": "violation"}])
    assert restored.findings[0]["history"][0]["functionality_evidence"]["denial_step"] == 1


def test_official_mapping_is_candidate_only_and_uses_campaign_rules():
    from scripts.batch_compliance_run import _candidate_issues, _conclusion_text
    from scripts.gen_experiment_report import build_official_records
    assert _candidate_issues({"\u95ee\u9898": ["R039", "R055"]}, {"R039": "violation"}) == ["\u95ee\u9898"]
    text = _conclusion_text({"official": {"version_relation": "unknown"}, "violations": [],
                             "per_issue": {"\u95ee\u9898": ["R039"]}, "candidate_issues": ["\u95ee\u9898"], "extra": []})
    assert "candidate matches" in text and "after remediation" not in text and "reproduced" not in text
    rows = build_official_records([{"pkg": "com.mock", "checks": [{"rule_id": "R039", "verdict": "violation"}]}], {
        "issue_to_rule_mapping": {"\u65b0\u901a\u62a5\u95ee\u9898": ["R039-R040"]},
        "local_apks": [{"package": "com.mock", "app": "Mock", "version_relation": "same",
                        "local_version": "1", "official_version": "1", "official_issues": ["\u65b0\u901a\u62a5\u95ee\u9898"]}]})
    assert rows[0]["comparison_status"] == "pending_review"
    assert rows[0]["candidate_match"] is True
    assert "reproduced" not in rows[0]


def test_cross_results_survive_patch_and_reset_for_seed(tmp_path):
    from workflow.seed_graph import load_seed_graph
    graph = cross_graph()
    graph.record_cross_test("func_1", "func_2", "\u8d44\u6e90\u7532", "\u5931\u8d25\u573a\u666f", "failed", reason="\u4e0d\u53ef\u8fbe")
    updater.apply_ffg_patch(graph, {}, [])
    assert graph.cross_test_status_counts()["failed"] == 1
    graph.get_node("func_1").intra_attempt_count = 9
    graph.cross_test_history = [{"status": "done", "invalidation_reason": "boundary changed"}]
    graph.save(str(tmp_path))
    (tmp_path / "run_result.json").write_text(json.dumps({"package_name": "com.mock", "status": "success"}))
    (tmp_path / "artifact_manifest.json").write_text('{"status":"complete"}')
    seed, _ = load_seed_graph(str(tmp_path), "com.mock")
    assert seed.cross_test_results == []
    assert seed.cross_test_history == []
    assert seed.get_node("func_1").intra_attempt_count == 0
    assert FlowGraph.load(str(tmp_path)).get_node("func_1").intra_attempt_count == 9
    assert len(planner._pending_cross_pairs(seed)) == 3
    assert FlowGraph.load(str(tmp_path)).cross_test_results[0]["status"] == "failed"


def test_batch_report_labels_rule_hits_as_candidates(tmp_path):
    from scripts.batch_compliance_run import AppEntry, build_batch_report
    entry = AppEntry(1, "Mock", "com.mock")
    root = tmp_path / entry.dir_name
    root.mkdir()
    (root / "run_result.json").write_text(json.dumps({"package_name": entry.package, "status": "success"}))
    (root / "compliance_report.json").write_text(json.dumps({
        "summary": {"gui_total": 1, "processed": 1, "violation": 1},
        "checks": [{"rule_id": "R039", "verdict": "violation", "description": "\u62d2\u7edd\u540e\u4e0d\u53ef\u7528"}],
    }))
    official = tmp_path / "official.json"
    official.write_text(json.dumps({"issue_to_rule_mapping": {"\u95ee\u9898": ["R039"]},
        "local_apks": [{"app": "Mock", "version_relation": "unknown", "official_issues": ["\u95ee\u9898", "\u672a\u6620\u5c04\u95ee\u9898"]}]}))
    report = build_batch_report(tmp_path, [entry], official_table=official)
    markdown = report.read_text()
    assert "Candidate match; manual review pending" in markdown
    assert "reproduced" not in markdown and "still present after remediation" not in markdown
    comparison = json.loads((tmp_path / "batch_report.json").read_text())[0]["official_comparison"]
    assert comparison["scope"] == "automatic_rule_match_only"
    assert [item["status"] for item in comparison["issues"]] == ["pending_review", "unmapped"]


def _done_cross_keys(graph):
    """Return done inter-functional combination keys, replacing the removed _tested_cross_pair_keys production helper."""
    return {
        graph.cross_test_key(item["src_function_id"], item["dst_function_id"], item["resource_variable"])
        for item in graph.cross_test_results if item["status"] == "done"
    }
