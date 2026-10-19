# Multilingual UI and model-response fixtures use Unicode escapes to preserve their exact text.
"""Depth-template and inter-functional readiness regression tests."""

from __future__ import annotations

from model.flow_graph import FunctionalFlowGraph, FunctionalityNode, Page
from workflow.nodes.scenario_plan import (
    PLAN_BREADTH_MAX_SCENARIOS,
    _forced_intra_scenario,
    _fallback_intra_functionality_depth_scenario,
    _cross_pair_ready,
)


def build_function(
    total_controls: int, covered_controls: int = 0
) -> tuple[FunctionalFlowGraph, FunctionalityNode]:
    graph = FunctionalFlowGraph()
    node = FunctionalityNode(id="func_1", name="\u6211\u7684\u6536\u85cf", test_status="in_progress")
    graph.add_node(node)
    page = Page(
        id="page_1",
        name="\u6211\u7684\u6536\u85cf\u9875",
        controls=[f"\u63a7\u4ef6{i}" for i in range(total_controls)],
        operations=[],
    )
    graph.add_page(page, "func_1")
    for index in range(min(covered_controls, total_controls)):
        operation = f"\u70b9\u51fb\u63a7\u4ef6{index}"
        page.add_operation(operation)
        assert page.add_operation_binding(operation, f"\u63a7\u4ef6{index}")
    graph.recompute_all_coverages()
    return graph, node


def test_breadth_phase_keeps_control_sweep_template():
    graph, node = build_function(total_controls=4)
    scenario = _forced_intra_scenario(graph, node)
    assert "visible widgets with no execution records" in scenario


def test_depth_phase_uses_business_scenario_template():
    graph, node = build_function(total_controls=1, covered_controls=1)
    scenario = _forced_intra_scenario(graph, node)
    assert scenario.startswith("Execute one complete business operation for \"\u6211\u7684\u6536\u85cf\"")
    assert "leave the page and return to inspect the result" in scenario
    assert "Do not repeat scenarios already executed for this functionality" in scenario
    assert "FuncDroid_" in scenario
    assert "Do not delete accounts, log out, switch accounts, pay, order, or perform identity verification" in scenario
    assert "[verify:" not in scenario


def test_depth_template_mentions_uncovered_controls_when_present():
    graph, node = build_function(total_controls=3)
    node.scenarios = [f"\u5e7f\u5ea6{i}" for i in range(PLAN_BREADTH_MAX_SCENARIOS)]
    scenario = _fallback_intra_functionality_depth_scenario(graph, node)
    assert "uncovered visible widgets" in scenario


def test_depth_template_is_empty_without_covered_pages():
    """Without any observed functionality pages, return an empty template for the existing release path."""
    graph = FunctionalFlowGraph()
    node = FunctionalityNode(id="func_1", name="\u7a7a\u529f\u80fd", test_status="in_progress")
    graph.add_node(node)
    assert _fallback_intra_functionality_depth_scenario(graph, node) == ""
    assert _forced_intra_scenario(graph, node) == ""


def _shared_resource_graph(business_operation: bool) -> FunctionalFlowGraph:
    graph = FunctionalFlowGraph()
    for index, name in enumerate(("\u4e2a\u4eba\u4e2d\u5fc3", "\u8d26\u53f7\u767b\u5f55"), start=1):
        node = FunctionalityNode(id=f"func_{index}", name=name, test_status="in_progress")
        graph.add_node(node)
        page = Page(
            id=f"page_{index}",
            name=f"{name}\u9875",
            controls=[f"{name}\u63a7\u4ef6"],
            operations=[],
        )
        graph.add_page(page, f"func_{index}")
        node.resource_variables = ["\u7528\u6237\u8d26\u53f7"]
        node.scenarios = [f"{name}\u5e7f\u5ea6\u573a\u666f"]
        if business_operation:
            page.add_operation(f"\u70b9\u51fb{name}\u63a7\u4ef6")
    graph.recompute_all_coverages()
    return graph


def test_cross_pair_waits_for_both_functions_to_have_business_evidence():
    """An inter-functional pair needs successful business evidence from both endpoints as reusable prerequisites."""
    graph = _shared_resource_graph(business_operation=False)
    assert _cross_pair_ready(graph, "func_1", "func_2") is False


def test_cross_pair_is_ready_without_waiting_for_unrelated_breadth():
    """Schedule a pair once both endpoints have own evidence without waiting for unrelated breadth exploration."""
    graph = _shared_resource_graph(business_operation=True)
    graph.add_node(FunctionalityNode(id="func_3", name="\u672a\u89e6\u78b0\u529f\u80fd", test_status="not_started"))
    assert _cross_pair_ready(graph, "func_1", "func_2") is True
