# Multilingual UI and model-response fixtures use Unicode escapes to preserve their exact text.
"""Deterministic tests for functionality breadth/depth phases and navigation levels. Derive phases from coverage and counters, and construct coverage with authoritative Page.operation_bindings."""

from __future__ import annotations

from model.flow_graph import FlowGraph, FunctionNode, Page, PageTransition
from workflow.nodes.scenario_plan import (
    FUNCTION_PHASE_BREADTH,
    FUNCTION_PHASE_DEPTH,
    NAV_LEVEL_BACKTRACK,
    NAV_LEVEL_EXPLORE,
    NAV_LEVEL_FORWARD,
    NAV_LEVEL_HERE,
    NAV_LEVEL_RESTART,
    PLAN_BREADTH_MAX_SCENARIOS,
    PLAN_FUNCTION_COVERAGE_TARGET,
    PLAN_MAX_DEPTH_SCENARIOS,
    PLAN_MIN_DEPTH_SCENARIOS,
    PLAN_SCENARIO_CAP_MARGIN,
    _describe_function_phase,
    _function_completion_met,
    _function_phase,
    _function_scenario_cap,
    _intra_sort_key,
    _navigation_to_function,
    _required_depth_count,
)


def build_function(total_controls: int, covered_controls: int = 0) -> tuple[FlowGraph, FunctionNode]:
    """Build a functionality with total controls and explicit execution bindings for the first covered controls."""
    graph = FlowGraph()
    node = FunctionNode(id="func_1", name="\u5e94\u7528\u8bbe\u7f6e", test_status="in_progress")
    graph.add_node(node)
    page = Page(
        id="page_1",
        name="\u8bbe\u7f6e\u9875",
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


def test_unobserved_function_stays_in_breadth():
    """Keep an unobserved functionality in breadth because its controls and verification targets are still unknown."""
    graph, node = build_function(total_controls=0)
    assert _function_phase(graph, node) == FUNCTION_PHASE_BREADTH


def test_uncovered_controls_keep_function_in_breadth():
    graph, node = build_function(total_controls=5, covered_controls=1)
    assert node.coverage_percent < PLAN_FUNCTION_COVERAGE_TARGET
    assert _function_phase(graph, node) == FUNCTION_PHASE_BREADTH


def test_breadth_gate_converts_to_depth_after_budget():
    """The breadth scenario budget advances to depth so entry exploration cannot continue indefinitely."""
    graph, node = build_function(total_controls=5)
    node.scenarios = [f"\u5e7f\u5ea6\u573a\u666f{i}" for i in range(PLAN_BREADTH_MAX_SCENARIOS)]
    assert _function_phase(graph, node) == FUNCTION_PHASE_DEPTH


def test_fully_covered_function_converts_to_depth():
    graph, node = build_function(total_controls=3, covered_controls=3)
    assert _function_phase(graph, node) == FUNCTION_PHASE_DEPTH


def test_depth_target_returns_function_to_breadth():
    """After reaching the depth target, return to breadth to cover remaining controls."""
    graph, node = build_function(total_controls=5)
    node.scenarios = [f"\u5e7f\u5ea6\u573a\u666f{i}" for i in range(PLAN_BREADTH_MAX_SCENARIOS)]
    node.intra_attempt_count = PLAN_BREADTH_MAX_SCENARIOS
    assert _function_phase(graph, node) == FUNCTION_PHASE_DEPTH
    node.depth_scenario_count = _required_depth_count(graph, node)
    node.intra_attempt_count += node.depth_scenario_count
    assert _function_phase(graph, node) == FUNCTION_PHASE_DEPTH
    node.successful_depth_scenario_count = node.depth_scenario_count
    assert _function_phase(graph, node) == FUNCTION_PHASE_BREADTH


def test_depth_target_scales_with_controls_and_is_clamped():
    assert _required_depth_count(*build_function(total_controls=0)) == PLAN_MIN_DEPTH_SCENARIOS
    assert _required_depth_count(*build_function(total_controls=5)) == 1
    assert _required_depth_count(*build_function(total_controls=6)) == 2
    assert _required_depth_count(*build_function(total_controls=12)) == 3
    assert _required_depth_count(*build_function(total_controls=200)) == PLAN_MAX_DEPTH_SCENARIOS


def test_scenario_cap_is_breadth_plus_depth_target_plus_margin():
    graph, node = build_function(total_controls=12)
    assert _function_scenario_cap(graph, node) == PLAN_BREADTH_MAX_SCENARIOS + 3 + PLAN_SCENARIO_CAP_MARGIN


def test_legacy_verify_markers_migrate_to_depth_count():
    """Migrate legacy verify/mismatch scenario markers into depth counts when loading the graph."""
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="\u6211\u7684\u6536\u85cf"))
    data = graph.to_dict()
    function = next(iter(data["functions"].values())) if isinstance(data["functions"], dict) else data["functions"][0]
    function.pop("depth_scenario_count", None)
    function["scenarios"] = ["\u5e7f\u5ea6\u573a\u666f", "[verify:func_1] \u79fb\u9664\u540e\u91cd\u8fdb\u4e0d\u56de\u5f39", "[mismatch:func_1] \u4fdd\u5b58\u540e\u8bfb\u56de"]

    restored = FlowGraph.from_dict(data).get_node("func_1")
    assert restored.scenarios == ["\u5e7f\u5ea6\u573a\u666f", "\u79fb\u9664\u540e\u91cd\u8fdb\u4e0d\u56de\u5f39", "\u4fdd\u5b58\u540e\u8bfb\u56de"]
    assert restored.depth_scenario_count == 2
    assert FlowGraph.from_dict(FlowGraph.from_dict(data).to_dict()).get_node("func_1").depth_scenario_count == 2


def test_completion_requires_depth_target():
    """Sufficient coverage and successful operations do not complete a functionality below its depth target."""
    graph, node = build_function(total_controls=1, covered_controls=1)
    node.scenarios = ["\u70b9\u51fb\u63a7\u4ef60\u5e76\u89c2\u5bdf\u7ed3\u679c"]
    assert _function_completion_met(graph, node) is False

    node.depth_scenario_count = _required_depth_count(graph, node)
    assert _function_completion_met(graph, node) is False
    node.successful_depth_scenario_count = node.depth_scenario_count
    assert _function_completion_met(graph, node) is True


def test_reaching_scenario_cap_without_depth_defers():
    from workflow.nodes.scenario_plan import _update_function_statuses

    graph, node = build_function(total_controls=1, covered_controls=1)
    node.intra_attempt_count = _function_scenario_cap(graph, node)
    _update_function_statuses(graph)
    assert graph.get_node_test_status("func_1") == "deferred"
    assert node.status_reason == "scenario_cap"


def test_completed_function_reopens_when_depth_target_rises():
    """New controls can raise the depth target and reopen a completed functionality."""
    from workflow.nodes.scenario_plan import _update_function_statuses

    graph, node = build_function(total_controls=5, covered_controls=5)
    node.scenarios = ["\u5e7f\u5ea6\u573a\u666f", "\u6df1\u5ea6\u573a\u666f"]
    node.intra_attempt_count = 2
    node.depth_scenario_count = 1
    node.successful_depth_scenario_count = 1
    _update_function_statuses(graph)
    assert graph.get_node_test_status("func_1") == "completed"

    page = graph.get_page("page_1")
    for index in range(5, 10):
        page.controls.append(f"\u63a7\u4ef6{index}")
        operation = f"\u70b9\u51fb\u63a7\u4ef6{index}"
        page.add_operation(operation)
        page.add_operation_binding(operation, f"\u63a7\u4ef6{index}")
    _update_function_statuses(graph)
    assert graph.get_node_test_status("func_1") == "in_progress"
    assert _function_phase(graph, node) == FUNCTION_PHASE_DEPTH


def test_phase_summary_is_human_readable():
    graph, node = build_function(total_controls=3)
    summary = _describe_function_phase(graph, node)
    assert FUNCTION_PHASE_BREADTH in summary
    assert "depth 0/" in summary
    assert _describe_function_phase(graph, None) == "No target functionality"


def test_depth_phase_functions_are_scheduled_before_untouched_breadth_functions():
    """Prioritize depth-ready functionalities over untouched breadth work to prevent starvation when many new functionalities exist."""
    graph = FlowGraph()
    depth_node = FunctionNode(id="func_1", name="\u5df2\u8fdb\u5165\u6df1\u5ea6", test_status="in_progress")
    untouched_a = FunctionNode(id="func_2", name="\u4ece\u672a\u89e6\u78b0A", test_status="not_started")
    untouched_b = FunctionNode(id="func_3", name="\u4ece\u672a\u89e6\u78b0B", test_status="not_started")
    for node in (depth_node, untouched_a, untouched_b):
        graph.add_node(node)
    graph.add_page(Page(id="page_1", name="\u98751", controls=["a"], operations=[]), "func_1")
    graph.add_page(Page(id="page_2", name="\u98752", controls=["b"], operations=[]), "func_2")
    graph.add_page(Page(id="page_3", name="\u98753", controls=["c"], operations=[]), "func_3")
    depth_node.scenarios = [f"\u5e7f\u5ea6\u573a\u666f{i}" for i in range(PLAN_BREADTH_MAX_SCENARIOS)]
    graph.recompute_all_coverages()

    assert _function_phase(graph, depth_node) == FUNCTION_PHASE_DEPTH

    ordered = sorted(
        graph.functions.values(), key=lambda node: _intra_sort_key(graph, node, [], {})
    )
    assert ordered[0].id == "func_1"


def test_failure_penalty_still_outranks_depth_priority():
    """Failure penalties outrank depth priority so the scheduler cannot remain trapped in a repeatedly failing functionality."""
    graph = FlowGraph()
    depth_node = FunctionNode(id="func_1", name="\u5df2\u8fdb\u5165\u6df1\u5ea6\u4f46\u521a\u5931\u8d25", test_status="in_progress")
    untouched = FunctionNode(id="func_2", name="\u4ece\u672a\u89e6\u78b0", test_status="not_started")
    for node in (depth_node, untouched):
        graph.add_node(node)
    graph.add_page(Page(id="page_1", name="\u98751", controls=["a"], operations=[]), "func_1")
    graph.add_page(Page(id="page_2", name="\u98752", controls=["b"], operations=[]), "func_2")
    depth_node.scenarios = [f"\u5e7f\u5ea6\u573a\u666f{i}" for i in range(PLAN_BREADTH_MAX_SCENARIOS)]
    graph.recompute_all_coverages()

    ordered = sorted(
        graph.functions.values(),
        key=lambda node: _intra_sort_key(graph, node, [], {"func_1": 1}),
    )
    assert ordered[0].id == "func_2"


def _navigation_graph() -> FlowGraph:
    """Home to list to detail; detail belongs to func_2 and the isolated page to func_3."""
    graph = FlowGraph()
    for index, name in ((1, "\u9996\u9875"), (2, "\u8be6\u60c5"), (3, "\u5b64\u7acb\u529f\u80fd")):
        graph.add_node(FunctionNode(id=f"func_{index}", name=name, test_status="in_progress"))
    graph.add_page(Page(id="page_1", name="\u9996\u9875"), "func_1")
    graph.add_page(Page(id="page_2", name="\u5217\u8868\u9875"), "func_1")
    graph.add_page(Page(id="page_3", name="\u8be6\u60c5\u9875"), "func_2")
    graph.add_page(Page(id="page_4", name="\u5b64\u7acb\u9875"), "func_3")
    graph.add_page_transition(PageTransition(src_page_id="page_1", dst_page_id="page_2", operation="\u70b9\u51fb\u5217\u8868"))
    graph.add_page_transition(PageTransition(src_page_id="page_2", dst_page_id="page_3", operation="\u70b9\u51fb\u6761\u76ee"))
    return graph


def test_navigation_is_empty_when_already_in_scope():
    graph = _navigation_graph()
    text, _, level = _navigation_to_function(graph, graph.get_page("page_3"), graph.get_node("func_2"))
    assert (text, level) == ("", NAV_LEVEL_HERE)


def test_navigation_levels_forward_backtrack_restart():
    graph = _navigation_graph()
    target = graph.get_node("func_2")

    text, _, level = _navigation_to_function(graph, graph.get_page("page_1"), target)
    assert level == NAV_LEVEL_FORWARD and "\u70b9\u51fb\u5217\u8868" in text and "\u70b9\u51fb\u6761\u76ee" in text

    graph.add_page(Page(id="page_5", name="\u5b50\u9875"), "func_1")
    graph.add_page_transition(PageTransition(src_page_id="page_2", dst_page_id="page_5", operation="\u70b9\u51fb\u5b50\u9875"))
    text, _, level = _navigation_to_function(graph, graph.get_page("page_5"), target)
    assert level == NAV_LEVEL_BACKTRACK and "try back once" in text

    graph.add_page(Page(id="page_6", name="\u5916\u90e8\u9875"), "func_3")
    text, _, level = _navigation_to_function(graph, graph.get_page("page_6"), target)
    assert level == NAV_LEVEL_RESTART and text.startswith("Execute open_app")


def test_navigation_without_recorded_path_is_exploratory_not_blocking():
    """Without a recorded route, use level-D exploratory navigation rather than immediately deferring the functionality."""
    graph = _navigation_graph()
    text, _, level = _navigation_to_function(graph, graph.get_page("page_1"), graph.get_node("func_3"))
    assert level == NAV_LEVEL_EXPLORE
    assert "exploratory navigation" in text and "\u5b64\u7acb\u9875" in text
    assert graph.get_node_test_status("func_3") == "in_progress"


def test_deferred_function_gets_exactly_one_closing_retry():
    from workflow.nodes.scenario_plan import _select_deferred_retry

    graph, node = build_function(total_controls=2)
    node.test_status = "deferred"
    page = graph.get_page("page_1")

    assert _select_deferred_retry(graph, page, []).id == "func_1"
    assert _select_deferred_retry(graph, page, ["func_1"]) is None


def test_deferred_retry_skips_functions_only_reachable_by_exploration():
    from workflow.nodes.scenario_plan import _select_deferred_retry

    graph = _navigation_graph()
    graph.get_node("func_3").test_status = "deferred"
    assert _select_deferred_retry(graph, graph.get_page("page_1"), []) is None
    graph.get_node("func_2").test_status = "deferred"
    assert _select_deferred_retry(graph, graph.get_page("page_1"), []).id == "func_2"


def test_deferred_function_is_promoted_once_completion_is_met():
    from workflow.nodes.scenario_plan import _update_function_statuses

    graph, node = build_function(total_controls=1, covered_controls=1)
    node.scenarios = ["\u70b9\u51fb\u63a7\u4ef60\u5e76\u89c2\u5bdf\u7ed3\u679c"]
    node.depth_scenario_count = 1
    node.successful_depth_scenario_count = 1
    node.test_status = "deferred"

    _update_function_statuses(graph)

    assert graph.get_node_test_status("func_1") == "completed"
