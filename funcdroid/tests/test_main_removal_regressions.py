"""Offline regressions for the standalone functionality-testing distribution."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

import main
from model.flow_graph import FunctionalFlowGraph, FunctionalityNode, Page
from workflow.run_options import validate_resume_options


def _args(*extra):
    return main._build_parser().parse_args([
        "--device-id", "test-device", "--package-name", "org.example.app", *extra,
    ])


@pytest.fixture
def workers(monkeypatch):
    reviewer = SimpleNamespace(
        snapshot=lambda: ([], {}),
        failures_snapshot=lambda: [],
        pending_count=lambda: 0,
    )
    auditor = SimpleNamespace(pending_count=lambda: 0, rejected_count=lambda: 0)
    monkeypatch.setattr(main, "bug_detection_worker", reviewer)
    monkeypatch.setattr(main, "page_audit_worker", auditor)
    return reviewer, auditor


def _artifact_bundle(root, *, review_enabled=True):
    """Create minimal real artifacts for the final-validator integration checks."""
    graph = FunctionalFlowGraph()
    graph.save(str(root))
    (root / "activity_timeline.json").write_text(json.dumps({
        "activities": [], "points": [{"activity_count": 0}],
    }), encoding="utf-8")
    assert cv2.imwrite(str(root / "activity_timeline.png"), np.zeros((8, 8, 3), dtype=np.uint8))
    (root / "llm_usage.json").write_text("{}", encoding="utf-8")
    (root / "run_result.json").write_text(json.dumps({
        "mode": "defect", "status": "success", "package_name": "org.example.app",
    }), encoding="utf-8")
    (root / "run.log").write_text("Offline validation fixture\n", encoding="utf-8")
    if review_enabled:
        (root / "bugs.json").write_text("[]", encoding="utf-8")
    return {"ffg": graph, "bug_detection_enabled": review_enabled}


def test_exploration_only_runtime_does_not_construct_a_review_model(monkeypatch):
    models = [object(), object(), object()]
    create_model = Mock(side_effect=models)
    device = object()
    create_driver = Mock(return_value=device)
    monkeypatch.setattr(main, "create_llm", create_model)
    monkeypatch.setattr(main, "Driver", create_driver)
    monkeypatch.setenv("FFG_UPDATE_LLM_TIMEOUT", "300")

    result = main._init_runtime(_args("--no-bug-detection"))

    assert result == (*models, None, device)
    assert create_model.call_count == 3
    assert create_model.call_args.kwargs == {"timeout_seconds": 300.0}
    create_driver.assert_called_once_with("test-device")


def test_default_runtime_keeps_a_separate_bug_review_model(monkeypatch):
    models = [object(), object(), object(), object()]
    create_model = Mock(side_effect=models)
    device = object()
    monkeypatch.setattr(main, "create_llm", create_model)
    monkeypatch.setattr(main, "Driver", Mock(return_value=device))

    assert main._init_runtime(_args()) == (*models, device)
    assert create_model.call_count == 4


def test_final_artifact_gate_accepts_a_complete_functionality_run(tmp_path, workers):
    state = _artifact_bundle(tmp_path)

    result = main._validate_final_artifacts(state, str(tmp_path), 0)

    assert result["status"] == "complete"
    assert result["errors"] == []
    assert {item["path"] for item in result["files"]} >= {"ffg.json", "bugs.json", "run_result.json"}


def test_exploration_only_artifacts_do_not_require_bug_outputs(tmp_path, workers):
    state = _artifact_bundle(tmp_path, review_enabled=False)

    result = main._validate_final_artifacts(state, str(tmp_path), 0)

    assert result["status"] == "complete"
    assert not (tmp_path / "bugs.json").exists()


def test_pending_bug_review_still_blocks_artifact_completion(tmp_path, workers):
    state = _artifact_bundle(tmp_path)
    workers[0].pending_count = lambda: 1

    result = main._validate_final_artifacts(state, str(tmp_path), 0)

    assert result["status"] == "incomplete"
    assert any("Bug review" in error and "background jobs" in error for error in result["errors"])


def test_failed_bug_review_still_blocks_artifact_completion(tmp_path, workers):
    state = _artifact_bundle(tmp_path)
    state["defect_review_failures"] = [{"reason": "review request failed"}]

    result = main._validate_final_artifacts(state, str(tmp_path), 0)

    assert result["status"] == "incomplete"
    assert any("Bug review has 1 failed jobs" in error for error in result["errors"])


def test_pending_page_audit_still_blocks_artifact_completion(tmp_path, workers):
    state = _artifact_bundle(tmp_path)
    workers[1].pending_count = lambda: 1

    result = main._validate_final_artifacts(state, str(tmp_path), 0)

    assert result["status"] == "incomplete"
    assert any("Page audit" in error and "background jobs" in error for error in result["errors"])


def test_run_result_retains_defect_failures_and_pending_review_counts(tmp_path, workers, monkeypatch):
    failures = [{"reason": "request timed out"}]
    workers[0].pending_count = lambda: 2
    workers[0].failures_snapshot = lambda: failures
    monkeypatch.setattr(main, "get_llm_usage_totals", lambda: {"requests": 3})
    state = main._build_initial_state(_args(), 1000.0)

    main._save_run_result(state, str(tmp_path), exit_code=1, elapsed_seconds=25.5)
    result = json.loads((tmp_path / "run_result.json").read_text(encoding="utf-8"))

    assert result["mode"] == "defect"
    assert result["status"] == "failed"
    assert result["pending_reviews"] == {"defect": 2}
    assert result["defect_review_failures"] == failures
    assert result["llm_usage"] == {"requests": 3}


def test_resume_keeps_functionality_graph_budget_and_review_failures(tmp_path):
    graph = FunctionalFlowGraph()
    graph.add_node(FunctionalityNode(id="func_1", name="Record editing", successful_depth_scenario_count=1))
    graph.add_page(Page(id="page_1", name="Editor", controls=["Save"]), "func_1")
    graph.save(str(tmp_path))
    (tmp_path / "run_result.json").write_text(json.dumps({
        "mode": "defect", "package_name": "org.example.app", "elapsed_seconds": 40.0,
        "iteration_count": 7, "defect_review_failures": [{"reason": "old snapshot failure"}],
    }), encoding="utf-8")
    (tmp_path / "defect_review_failures.json").write_text(json.dumps([
        {"reason": "authoritative persisted failure"},
    ]), encoding="utf-8")
    args = _args("--resume-output", str(tmp_path))
    validate_resume_options(args)

    resumed = main._load_resume_state(args, str(tmp_path), 1000.0)

    assert resumed["start_time"] == 960.0
    assert resumed["iteration_count"] == 7
    assert resumed["workflow_complete"] is False
    assert resumed["ffg"].get_node("func_1").successful_depth_scenario_count == 1
    assert resumed["defect_review_failures"] == [{"reason": "authoritative persisted failure"}]


def test_resume_rejects_an_unidentified_run_before_device_initialization(tmp_path):
    (tmp_path / "run_result.json").write_text(json.dumps({
        "package_name": "org.example.app", "status": "success",
    }), encoding="utf-8")
    args = _args("--resume-output", str(tmp_path))

    with pytest.raises(ValueError, match="Resume mode mismatch"):
        validate_resume_options(args)


@pytest.mark.parametrize("marker", [
    "Black or unavailable page",
    "\u9ed1\u5c4f\u6216\u9875\u9762\u4e0d\u53ef\u7528",
])
def test_resumed_unavailable_page_is_reused_and_its_failed_path_is_avoided(tmp_path, marker):
    """Old FFG labels remain readable without losing failed-path avoidance."""
    from model.flow_graph import PageTransition
    from workflow.nodes.ffg_update import _record_screenshot_unavailable_page
    from workflow.nodes.scenario_plan import _is_black_screen_page, _matches_black_screen_path

    graph = FunctionalFlowGraph()
    graph.add_node(FunctionalityNode(id="func_1", name="Settings"))
    graph.add_page(Page(id="page_1", name="Settings page"), "func_1")
    graph.add_page(Page(id="page_2", name=marker), "func_1")
    graph.add_page_transition(PageTransition(
        src_page_id="page_1", dst_page_id="page_2",
        operation="Open unstable panel; result: screenshot unavailable",
    ))
    graph.save(str(tmp_path))
    graph = FunctionalFlowGraph.load(str(tmp_path))

    assert _is_black_screen_page(graph, graph.get_page("page_2"))
    assert _matches_black_screen_path(graph, "Open unstable panel and inspect its content")
    assert not _matches_black_screen_path(graph, "Change notification settings")

    owners = _record_screenshot_unavailable_page(graph, {
        "current_plan": {"target_function_id": "func_1"},
        "scenario_execution_trace": [{
            "action_name": "click",
            "action_description": "Open unstable panel",
            "current_page": graph.get_page("page_1"),
        }],
    }, {"scenario_status": "screenshot_unavailable"})

    assert owners == ["func_1"]
    assert len(graph.active_pages()) == 2
    assert graph.get_page("page_2").name == marker
    assert all(edge.dst_page_id == "page_2" for edge in graph.page_transitions)


def test_legacy_unavailable_page_is_excluded_from_screenshot_identity_matching(monkeypatch):
    from workflow.nodes import ffg_update

    graph = FunctionalFlowGraph()
    graph.add_node(FunctionalityNode(id="func_1", name="Settings"))
    graph.add_page(Page(
        id="page_1", name="Settings page", representative_screenshot="valid",
    ), "func_1")
    graph.add_page(Page(
        id="page_2", name="\u9ed1\u5c4f\u6216\u9875\u9762\u4e0d\u53ef\u7528",
        representative_screenshot="unavailable",
    ), "func_1")
    monkeypatch.setattr(ffg_update, "_best_page_similarity", lambda _cache, _shot, page:
                        1.0 if page.id == "page_2" else 0.2)

    assert ffg_update._resolve_observed_page_id(graph, {
        "current_page": Page(representative_screenshot="new-observation"),
    }, 1) == ""
