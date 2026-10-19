# Multilingual UI and model-response fixtures use Unicode escapes to preserve their exact text.
"""Regressions for the coverage-accounting and scheduling fixes."""

from __future__ import annotations

import pytest

from model.flow_graph import FlowGraph, FunctionNode, Page
from workflow.nodes.ffg_update import (
    _find_control_signature_page,
    _merge_page_observation,
    _normalize_observation_alignment,
    validate_observation_alignment,
)
from workflow.nodes.scenario_plan import (
    PLAN_FUNCTION_COVERAGE_TARGET,
    _node_intra_attempt_count,
    _normalize_and_reopen_undercovered_functions,
)
from workflow.subgraphs.ffg_update import create_ffg_update_subgraph


@pytest.fixture(autouse=True)
def _legacy_text_matching_enabled(monkeypatch):
    """Enable legacy text matching explicitly to test its false-coverage and mismatch guards. Explicit operation bindings remain authoritative and text fallback is disabled by default."""
    from model import flow_graph as _flow_graph

    monkeypatch.setattr(_flow_graph, "COVERAGE_LEGACY_TEXT_MATCHING", True)


def test_repeated_operations_do_not_inflate_control_coverage():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="\u641c\u7d22"))
    graph.add_page(Page(
        id="page_1",
        controls=[f"\u63a7\u4ef6{index}" for index in range(10)],
        operations=["\u70b9\u51fb\u63a7\u4ef60"] * 9,
    ), "func_1")

    details = graph.function_coverage_details("func_1")

    assert details["total_controls"] == 10
    assert details["executed_operations"] == 1
    assert details["coverage_percent"] < 20


def test_distinct_operations_on_same_control_do_not_inflate_coverage():
    """Four differently worded operations hitting one of four controls yield 25%, not 100%, coverage."""
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="\u641c\u7d22"))
    graph.add_page(Page(
        id="page_1",
        controls=["\u9876\u90e8\u641c\u7d22\u6309\u94ae", "\u5e95\u90e8\u6536\u85cf\u6309\u94ae", "\u5e95\u90e8\u5206\u4eab\u6309\u94ae", "\u5e95\u90e8\u8bc4\u8bba\u6309\u94ae"],
        operations=[
            "\u70b9\u51fb\u9876\u90e8\u641c\u7d22\u6309\u94ae",
            "\u70b9\u51fb\u9876\u90e8\u641c\u7d22\u56fe\u6807",
            "\u70b9\u51fb\u9876\u90e8\u641c\u7d22\u5165\u53e3",
            "\u70b9\u51fb\u9876\u90e8\u641c\u7d22\u6846",
        ],
    ), "func_1")

    details = graph.function_coverage_details("func_1")

    assert details["total_controls"] == 4
    assert details["executed_operations"] == 1
    assert details["coverage_percent"] == 25


def test_uncovered_controls_are_reported_in_full():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="\u8bbe\u7f6e"))
    graph.add_page(Page(
        id="page_1",
        controls=["\u84dd\u7259\u5f00\u5173", "\u97f3\u91cf\u6ed1\u5757", "\u4eae\u5ea6\u6ed1\u5757"],
        operations=["\u70b9\u51fb\u84dd\u7259\u5f00\u5173", "\u70b9\u51fb\u84dd\u7259\u5f00\u5173", "\u70b9\u51fb\u84dd\u7259\u5f00\u5173"],
    ), "func_1")

    details = graph.function_coverage_details("func_1")

    assert details["uncovered_controls"] == ["\u97f3\u91cf\u6ed1\u5757", "\u4eae\u5ea6\u6ed1\u5757"]


def test_numeric_suffix_controls_are_not_treated_as_covered():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="\u5217\u8868"))
    graph.add_page(Page(
        id="page_1",
        controls=["\u63a7\u4ef61", "\u63a7\u4ef612"],
        operations=["\u70b9\u51fb\u63a7\u4ef61"],
    ), "func_1")

    assert graph.function_coverage_details("func_1")["uncovered_controls"] == ["\u63a7\u4ef612"]


def test_navigation_only_entry_page_does_not_contribute_function_coverage():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="\u4e3b\u9875"))
    graph.add_node(FunctionNode(
        id="func_2",
        name="\u76f8\u518c\u9009\u62e9",
        scenarios=["\u4ece\u4e3b\u9875\u70b9\u51fb\u76f8\u518c\u5165\u53e3"],
    ))
    page = Page(
        id="page_1",
        controls=["\u76f8\u518c\u5165\u53e3", "\u8bbe\u7f6e\u5165\u53e3"],
        operations=["\u70b9\u51fb\u76f8\u518c\u5165\u53e3"],
    )
    graph.add_page(page, "func_1")
    graph.add_entry_page(page, "func_2")

    assert graph.function_controls("func_2") == {}
    assert graph.function_executed_operation_keys("func_2") == set()
    assert graph.function_coverage_details("func_2") == {
        "total_controls": 0,
        "executed_operations": 0,
        "coverage_percent": 0,
        "uncovered_controls": [],
    }
    assert graph.function_has_successful_business_operation("func_2") is False


def test_scenario_history_survives_reopen_scan():
    graph = FlowGraph()
    node = FunctionNode(
        id="func_1",
        name="\u4e3b\u9875",
        test_status="completed",
        scenarios=["\u70b9\u51fb\u641c\u7d22\u6309\u94ae\u5e76\u89c2\u5bdf\u7ed3\u679c", "\u70b9\u51fb\u641c\u7d22\u6309\u94ae\u5e76\u786e\u8ba4\u7ed3\u679c"],
    )
    graph.add_node(node)
    graph.add_page(Page(id="page_1", controls=["\u641c\u7d22\u6309\u94ae", "\u83dc\u5355\u6309\u94ae"]), "func_1")

    _normalize_and_reopen_undercovered_functions(graph)

    assert len(graph.get_node("func_1").scenarios) == 2
    assert graph.get_node("func_1").test_status == "in_progress"


def test_legacy_cross_attempt_without_result_record_is_not_tested():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="\u76f8\u518c"))
    graph.add_node(FunctionNode(id="func_2", name="\u804a\u5929"))
    graph.add_cross_scenario("func_1", "func_2", "\u4ece\u76f8\u518c\u9009\u56fe\u5e76\u53d1\u9001\u5230\u4f1a\u8bdd")

    assert not graph.function_dependencies
    tested_keys = _done_cross_keys(graph)
    assert tested_keys == set()




def test_same_activity_and_control_signature_reuses_existing_page():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="\u7b14\u8bb0"))
    graph.add_page(Page(
        id="page_1",
        name="\u7b14\u8bb0\u5217\u8868",
        activity="NotesActivity",
        controls=["\u641c\u7d22\u6309\u94ae", "\u6c49\u5821\u83dc\u5355\u6309\u94ae", "\u65b0\u589e\u7b14\u8bb0\u6309\u94ae", "\u7b14\u8bb0\u6761\u76ee"],
    ), "func_1")

    reused = _find_control_signature_page(
        graph,
        Page(activity="NotesActivity"),
        ["\u641c\u7d22\u6309\u94ae", "\u6c49\u5821\u83dc\u5355\u6309\u94ae", "\u65b0\u589e\u7b14\u8bb0\u6309\u94ae", "\u7b14\u8bb0\u6761\u76ee"],
    )
    distinct = _find_control_signature_page(
        graph,
        Page(activity="NotesActivity"),
        ["\u8fd4\u56de\u6309\u94ae", "\u6807\u9898\u8f93\u5165\u6846", "\u6b63\u6587\u8f93\u5165\u6846", "\u4fdd\u5b58\u6309\u94ae"],
    )

    assert reused is not None and reused.id == "page_1"
    assert distinct is None


def test_same_activity_and_controls_do_not_merge_different_named_entities():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="\u804a\u5929"))
    controls = ["\u8fd4\u56de\u6309\u94ae", "\u6d88\u606f\u5217\u8868", "\u8f93\u5165\u6846", "\u53d1\u9001\u6309\u94ae"]
    graph.add_page(Page(
        id="page_1",
        name="\u6587\u4ef6\u4f20\u8f93\u52a9\u624b\u8be6\u60c5\u9875",
        activity="LauncherUI",
        controls=controls,
    ), "func_1")

    matched = _find_control_signature_page(
        graph,
        Page(activity="LauncherUI"),
        controls,
        declared_name="\u6d4b\u8bd5\u7fa4\u8be6\u60c5\u9875",
    )

    assert matched is None


def test_merge_observation_preserves_stable_representative_screenshot():
    page = Page(id="page_1", representative_screenshot="stable")

    _merge_page_observation(
        page,
        Page(activity="LauncherUI", representative_screenshot="transient-popup"),
    )

    assert page.representative_screenshot == "stable"
    assert "transient-popup" in page.evidence_screenshots


def test_ffg_update_calls_model_once_per_round():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="\u529f\u80fd", test_status="in_progress"))
    graph.add_page(Page(id="page_1", name="\u9996\u9875", activity="HomeActivity"), "func_1")

    class AlwaysContinueLlm:
        calls = 0

        def invoke(self, _messages):
            AlwaysContinueLlm.calls += 1
            return type("Response", (), {"content": (
                '{"update_control":{"decision":"continue","reason":"want more"},'
                '"functions":[],"pages":[],"page_ownership":[],'
                '"function_dependencies":[],"observation_alignment":[]}'
            )})()

    subgraph = create_ffg_update_subgraph(AlwaysContinueLlm())
    subgraph.invoke({
        "ffg": graph,
        "current_page": Page(id="page_1", name="\u9996\u9875", activity="HomeActivity"),
        "current_plan": {"scenario_type": "intra_test", "target_function_id": "func_1"},
        "navigation_execution_trace": [],
        "scenario_execution_trace": [{
            "raw_step_index": 1,
            "action_name": "click",
            "current_page": Page(id="page_1", name="\u9996\u9875", activity="HomeActivity"),
        }],
        "execution_result": {"scenario_status": "completed"},
    })

    assert AlwaysContinueLlm.calls == 1


def test_inherited_coverage_from_shared_page_does_not_complete_barely_tested_function():
    """Sibling operations on shared pages must not complete a functionality with only one own scenario."""
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="\u6587\u672c\u53d1\u9001", test_status="in_progress"))
    graph.add_node(FunctionNode(id="func_2", name="\u8bed\u97f3\u53d1\u9001", test_status="in_progress"))
    shared = Page(
        id="page_1",
        name="\u804a\u5929\u9875",
        activity="ChatActivity",
        controls=["\u8f93\u5165\u6846", "\u53d1\u9001\u6309\u94ae", "\u8868\u60c5\u6309\u94ae", "\u8bed\u97f3\u6309\u94ae", "\u66f4\u591a\u6309\u94ae"],
        operations=["\u70b9\u51fb\u8f93\u5165\u6846", "\u70b9\u51fb\u53d1\u9001\u6309\u94ae", "\u70b9\u51fb\u8868\u60c5\u6309\u94ae", "\u70b9\u51fb\u8bed\u97f3\u6309\u94ae"],
    )
    graph.add_page(shared, "func_1")
    graph.add_page(shared, "func_2")
    graph.get_node("func_1").scenarios = [
        "\u70b9\u51fb\u8f93\u5165\u6846\u8f93\u5165\u6587\u672c\u5e76\u53d1\u9001",
        "\u70b9\u51fb\u8868\u60c5\u6309\u94ae\u5e76\u9009\u62e9\u8868\u60c5\u53d1\u9001",
        "\u70b9\u51fb\u53d1\u9001\u6309\u94ae\u786e\u8ba4\u6d88\u606f\u8fdb\u5165\u804a\u5929\u8bb0\u5f55",
    ]
    # func_2 has one own scenario; all other coverage came from func_1 on shared pages.
    graph.get_node("func_2").scenarios = ["\u70b9\u51fb\u8bed\u97f3\u6309\u94ae\u5e76\u53d1\u9001\u8bed\u97f3"]
    for node_id in ("func_1", "func_2"):
        graph.recompute_function_coverage(node_id)

    node = graph.get_node("func_2")
    assert node.coverage_percent >= PLAN_FUNCTION_COVERAGE_TARGET
    assert graph.function_has_successful_business_operation("func_2") is True

    # Inherited coverage cannot finish a functionality without its own result-verification scenario.
    from workflow.nodes.scenario_plan import _function_completion_met

    assert _node_intra_attempt_count(node) == 1
    assert _function_completion_met(graph, node) is False


def test_function_completion_met_requires_own_evidence():
    """The shared completion predicate requires own evidence even when coverage is inherited."""
    from workflow.nodes.scenario_plan import _function_completion_met

    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="\u804a\u5929\u6d88\u606f\u6536\u53d1"))
    graph.add_node(FunctionNode(id="func_2", name="\u804a\u5929\u6d88\u606f\u6536\u53d1"))
    shared = Page(
        id="page_1",
        name="\u6d88\u606f\u5217\u8868\u9875",
        activity="LauncherUI",
        controls=["\u8f93\u5165\u6846", "\u53d1\u9001\u6309\u94ae", "\u8868\u60c5\u6309\u94ae", "\u8bed\u97f3\u6309\u94ae", "\u66f4\u591a\u6309\u94ae"],
        operations=[
            "\u70b9\u51fb\u8f93\u5165\u6846",
            "\u70b9\u51fb\u53d1\u9001\u6309\u94ae",
            "\u70b9\u51fb\u8868\u60c5\u6309\u94ae",
            "\u70b9\u51fb\u8bed\u97f3\u6309\u94ae",
        ],
    )
    graph.add_page(shared, "func_1")
    graph.add_page(shared, "func_2")
    # func_1 has inherited coverage but no own depth scenario, so it cannot complete.
    graph.get_node("func_1").scenarios = ["\u70b9\u51fb\u8f93\u5165\u6846\u8f93\u5165\u6587\u672c"]
    graph.get_node("func_2").scenarios = ["\u70b9\u51fb\u8bed\u97f3\u6309\u94ae\u5e76\u53d1\u9001\u8bed\u97f3"]
    graph.get_node("func_2").depth_scenario_count = 1
    graph.get_node("func_2").successful_depth_scenario_count = 1
    for node_id in ("func_1", "func_2"):
        graph.recompute_function_coverage(node_id)

    node_1 = graph.get_node("func_1")
    assert node_1.coverage_percent >= PLAN_FUNCTION_COVERAGE_TARGET
    assert graph.function_has_successful_business_operation("func_1") is True
    assert _function_completion_met(graph, node_1) is False

    # func_2 satisfies coverage, successful operations, and own depth scenarios, so it can complete.
    assert _function_completion_met(graph, graph.get_node("func_2")) is True


def test_create_alignment_with_existing_page_id_is_repaired_without_refinement():
    """Normalize create with an existing page_id to reuse as a format repair."""
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="\u9996\u9875"))
    graph.add_page(Page(id="page_1", name="\u9996\u9875", activity="HomeActivity"), "func_1")
    step_page = Page(name="\u9996\u9875", activity="HomeActivity")
    trace = [{"raw_step_index": 1, "current_page": step_page}]
    candidate_map = {1: [{"page_id": "page_1", "score": 0.99, "exact_id": False}]}
    patch = {
        "pages": [],
        "observation_alignment": [
            {"source_step": 1, "page_decision": "create", "page_ref": "page_1"}
        ],
    }

    _normalize_observation_alignment(patch, graph, trace, candidate_map)
    issues = validate_observation_alignment(graph, patch, trace, candidate_map)

    assert patch["observation_alignment"][0]["page_decision"] == "reuse"
    assert issues == []


def test_create_alignment_referencing_unknown_page_is_left_to_the_model():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="\u9996\u9875"))
    graph.add_page(Page(id="page_1", name="\u9996\u9875", activity="HomeActivity"), "func_1")
    trace = [{"raw_step_index": 1, "current_page": Page(name="\u8be6\u60c5", activity="DetailActivity")}]
    patch = {
        "pages": [],
        "observation_alignment": [
            {"source_step": 1, "page_decision": "create", "page_ref": "page_77"}
        ],
    }

    _normalize_observation_alignment(patch, graph, trace, {})
    issues = validate_observation_alignment(graph, patch, trace, {})

    assert patch["observation_alignment"][0]["page_decision"] == "create"
    assert issues == ["create alignment must use new_page_x at source_step 1"]


def test_create_alignment_is_not_downgraded_without_candidate_support():
    """Do not downgrade create to reuse without candidate evidence; a genuinely new page could otherwise merge into the wrong node."""
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="\u9996\u9875"))
    graph.add_page(Page(id="page_1", name="\u9996\u9875", activity="HomeActivity"), "func_1")
    trace = [{"raw_step_index": 1, "current_page": Page(name="\u8be6\u60c5", activity="DetailActivity")}]
    patch = {
        "pages": [],
        "observation_alignment": [
            {"source_step": 1, "page_decision": "create", "page_ref": "page_1"}
        ],
    }

    _normalize_observation_alignment(patch, graph, trace, {})

    assert patch["observation_alignment"][0]["page_decision"] == "create"


def _done_cross_keys(graph):
    """Return done inter-functional combination keys, replacing the removed _tested_cross_pair_keys production helper."""
    return {
        graph.cross_test_key(item["src_function_id"], item["dst_function_id"], item["resource_variable"])
        for item in graph.cross_test_results if item["status"] == "done"
    }
