"""Offline regression checks for the standalone functionality-testing runtime."""

import argparse
import json
from unittest.mock import Mock

import pytest

import main
from model.flow_graph import FlowGraph, FunctionNode, Page
from workflow.graph import _fallback_scenario_exec
from workflow.nodes import ffg_update
from workflow.nodes.scenario_plan import _function_completion_met, _update_function_statuses
from workflow.run_options import validate_resume_options


def test_parser_exposes_only_functional_mode():
    parser = main._build_parser()
    action = next(item for item in parser._actions if item.dest == "mode")
    assert action.choices == ["defect"]
    assert {item.dest for item in parser._actions} == {
        "help", "mode", "reset_app_data", "grant_permissions", "device_id",
        "app_name", "package_name", "max_iterations", "total_time", "output_dir",
        "seed_ffg", "resume_output", "phase", "exploration_mode", "bug_detection",
    }
    args = parser.parse_args(["--device-id", "test-device", "--no-bug-detection"])
    assert args.mode == "defect"
    assert not args.bug_detection
    assert not args.reset_app_data
    assert args.grant_permissions


def test_unsupported_mode_is_rejected_before_runtime(monkeypatch):
    runtime = Mock(side_effect=AssertionError("Device initialization is forbidden"))
    monkeypatch.setattr(main, "_init_runtime", runtime)
    with pytest.raises(SystemExit) as error:
        main._build_parser().parse_args(["--device-id", "test-device", "--mode", "other"])
    assert error.value.code == 2
    runtime.assert_not_called()


@pytest.mark.parametrize("saved_mode", [None, "other"])
def test_resume_requires_matching_explicit_mode(tmp_path, saved_mode):
    result = {"package_name": "example.app", "mode": saved_mode}
    path = tmp_path / "run_result.json"
    path.write_text(json.dumps(result))
    before = path.read_bytes()
    args = argparse.Namespace(resume_output=str(tmp_path), mode="defect", package_name=None)
    with pytest.raises(ValueError, match="Resume mode mismatch"):
        validate_resume_options(args)
    assert path.read_bytes() == before


def test_completion_needs_successful_depth_after_full_widget_coverage():
    graph = FlowGraph()
    node = FunctionNode(id="func_1", name="Save a record", page_ids=["page_1"])
    graph.add_node(node)
    page = Page(id="page_1", controls=["Save"], operations=["click Save"])
    graph.add_page(page, node.id)
    assert page.add_operation_binding("click Save", "Save")
    node.scenarios = ["Save a new test record"]
    graph.recompute_all_coverages()
    assert node.coverage_percent == 100
    node.depth_scenario_count = 1
    assert not _function_completion_met(graph, node)
    node.successful_depth_scenario_count = 1
    assert _function_completion_met(graph, node)
    _update_function_statuses(graph)
    assert node.test_status == "completed"


def test_infrastructure_failure_does_not_consume_scenario_budget(monkeypatch):
    monkeypatch.setattr("workflow.graph._backoff_on_node_error", lambda *args: None, raising=False)
    graph = FlowGraph()
    node = FunctionNode(id="func_1", name="Record management")
    graph.add_node(node)
    state = {
        "ffg": graph,
        "current_plan": {"scenario_type": "intra_test", "function_phase": "depth", "target_function_id": node.id},
    }
    updates = _fallback_scenario_exec(state, RuntimeError("Device disconnected"))
    state.update(updates)
    ffg_update._record_intra_attempt(graph, state)
    assert node.intra_attempt_count == 0
    assert node.depth_scenario_count == 0
    assert node.successful_depth_scenario_count == 0


def test_help_is_available_without_device_or_model_configuration():
    help_text = main._build_parser().format_help()
    assert "Functional" in help_text or "functionality" in help_text
    assert "--device-id" in help_text
    assert "--phase" in help_text
    assert "--seed-ffg" in help_text
