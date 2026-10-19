# Multilingual UI and model-response fixtures use Unicode escapes to preserve their exact text.
"""Review regressions for execution facts, state semantics, inter-functional retries, depth counts, navigation, and detection diagnostics."""

import threading

from model.flow_graph import FlowGraph, FunctionDependency, FunctionNode, Page, PageTransition
from workflow.graph import _fallback_scenario_exec
from workflow.nodes import ffg_update as updater
from workflow.nodes import scenario_plan as planner
from workflow.worker_utils import SerialWorker


def _intra_graph() -> FlowGraph:
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="\u8bb0\u5f55\u6d3b\u52a8", test_status="in_progress"))
    graph.add_page(Page(id="page_1", name="\u6d3b\u52a8\u4e3b\u9875"), "func_1")
    return graph


def _typed_trace(content: str) -> list[dict]:
    return [
        {"action_name": "type", "content": content, "current_page": Page(representative_screenshot="img")},
        {"action_name": "finished", "current_page": Page(representative_screenshot="img")},
    ]


# Issue 1: Record execution facts even without a structural patch.

def test_execution_facts_are_recorded_without_structural_patch():
    graph = _intra_graph()
    state = {
        "current_plan": {"scenario_type": "intra_test", "target_function_id": "func_1"},
        "current_test_scenario": "\u65b0\u589e\u4e00\u6761\u8bb0\u5f55",
        "scenario_execution_trace": _typed_trace("FuncDroid_func_1"),
    }

    result = updater._apply_non_recordable_execution_outcome(
        graph, state, {"navigation_status": "completed", "scenario_status": "completed"},
    )

    assert result["ffg"] is graph
    assert "\u65b0\u589e\u4e00\u6761\u8bb0\u5f55" in graph.get_node("func_1").scenarios
    assert [item["value"] for item in graph.function_test_data("func_1")] == ["FuncDroid_func_1"]


# Issue 3: Bounded inter-functional retries with traceable test data.

def _cross_graph() -> FlowGraph:
    graph = FlowGraph()
    for index in (1, 2):
        graph.add_node(FunctionNode(id=f"func_{index}", name=f"\u529f\u80fd{index}"))
    graph.add_function_dependency(FunctionDependency("func_1", "func_2", "\u8d44\u6e90\u7532"))
    return graph


def _cross_state() -> dict:
    return {
        "current_plan": {"scenario_type": "cross_test", "src_function_id": "func_1",
                         "dst_function_id": "func_2", "resource_variable": "\u8d44\u6e90\u7532"},
        "current_test_scenario": "\u5199\u5165\u540e\u8bfb\u53d6",
        "scenario_execution_trace": [],
    }


def test_navigation_failed_cross_pair_is_retried_until_attempt_cap():
    graph = _cross_graph()
    navigation_failure = {"navigation_status": "stalled", "scenario_status": "skipped"}

    updater._record_execution_outcome(graph, _cross_state(), navigation_failure)
    assert graph.cross_test_results[0]["retryable"] is True
    assert planner._pending_cross_pairs(graph) == [("func_1", "func_2", "Functional dependency: \u8d44\u6e90\u7532")]

    for _ in range(planner.PLAN_MAX_CROSS_ATTEMPTS - 1):
        updater._record_execution_outcome(graph, _cross_state(), navigation_failure)
    assert graph.cross_test_results[0]["attempts"] == planner.PLAN_MAX_CROSS_ATTEMPTS
    assert planner._pending_cross_pairs(graph) == []


def test_test_phase_cross_failure_is_not_retried():
    graph = _cross_graph()
    updater._record_execution_outcome(
        graph, _cross_state(), {"navigation_status": "completed", "scenario_status": "failed_goal"},
    )
    assert graph.cross_test_results[0]["retryable"] is False
    assert planner._pending_cross_pairs(graph) == []


def test_cross_scenario_reuses_source_test_data():
    graph = _cross_graph()
    graph.record_test_data("func_1", ["FuncDroid_func_1"], iteration=3)

    scenario = planner._forced_cross_scenario(graph, "func_1", "func_2", "\u529f\u80fd\u4f9d\u8d56: \u8d44\u6e90\u7532")

    assert "FuncDroid_func_1" in scenario
    restored = FlowGraph.from_dict(graph.to_dict())
    assert restored.function_test_data("func_1")[0]["iteration"] == 3


def test_test_data_instruction_uses_stable_function_suffix():
    graph = _intra_graph()
    text = planner._test_data_instruction(graph, "intra_test", graph.get_node("func_1"), None)
    assert "FuncDroid_func_1" in text


# Depth attempts count on entering the test phase, including failed attempts.

def _depth_state(status: str, **plan) -> dict:
    return {
        "current_plan": {"scenario_type": "intra_test", "target_function_id": "func_1",
                         "function_phase": "depth", **plan},
        "current_test_scenario": "\u65b0\u5efa\u6761\u76ee\u540e\u79bb\u5f00\u518d\u8fd4\u56de\u67e5\u770b",
        "execution_result": {"navigation_status": "completed", "scenario_status": status},
    }


def test_depth_attempt_counts_even_when_test_phase_fails():
    graph = _intra_graph()
    for status in ("completed", "failed_goal"):
        updater._record_intra_attempt(graph, _depth_state(status))
    node = graph.get_node("func_1")
    assert node.depth_scenario_count == 2
    assert node.intra_attempt_count == 2


def test_navigation_skip_and_breadth_do_not_count_as_depth():
    graph = _intra_graph()
    updater._record_intra_attempt(graph, _depth_state("skipped"))
    breadth = _depth_state("completed")
    breadth["current_plan"]["function_phase"] = "breadth"
    updater._record_intra_attempt(graph, breadth)
    node = graph.get_node("func_1")
    assert node.depth_scenario_count == 0
    assert node.intra_attempt_count == 1


# Attempt level-D exploratory navigation only once.

def test_exploratory_navigation_failure_defers_immediately():
    graph = _intra_graph()
    state = _depth_state("skipped", navigation_level="D")
    updater._record_execution_outcome(
        graph, state, {"navigation_status": "stalled", "scenario_status": "skipped"},
    )
    node = graph.get_node("func_1")
    assert graph.get_node_test_status("func_1") == "deferred"
    assert node.status_reason == "exploratory_navigation_failed"


def test_recorded_path_navigation_failure_waits_for_limit():
    graph = _intra_graph()
    state = _depth_state("skipped", navigation_level="A")
    updater._record_execution_outcome(
        graph, state, {"navigation_status": "stalled", "scenario_status": "skipped"},
    )
    assert graph.get_node_test_status("func_1") == "in_progress"
    assert graph.get_node("func_1").navigation_failure_count == 1


def test_exploratory_cross_navigation_failure_is_not_retried():
    graph = _cross_graph()
    state = _cross_state()
    state["current_plan"]["navigation_level"] = "D"
    updater._record_execution_outcome(
        graph, state, {"navigation_status": "stalled", "scenario_status": "skipped"},
    )
    assert graph.cross_test_results[0]["retryable"] is False
    assert planner._pending_cross_pairs(graph) == []


# Issue 5: Executor infrastructure exceptions do not count as functionality attempts.

def test_exec_fallback_clears_scenario_and_marks_infrastructure_error():
    result = _fallback_scenario_exec(
        {"current_test_scenario": "\u672a\u6267\u884c\u7684\u573a\u666f", "current_plan": {"scenario_type": "intra_test"}},
        RuntimeError("device lost"),
    )
    assert result["current_test_scenario"] == ""
    assert result["execution_result"]["infrastructure_error"] is True


def test_infrastructure_error_records_no_scenario_or_attempt():
    graph = _intra_graph()
    state = {
        "current_plan": {"scenario_type": "intra_test", "target_function_id": "func_1"},
        "current_test_scenario": "\u672a\u6267\u884c\u7684\u573a\u666f",
        "execution_result": {"scenario_status": "error", "infrastructure_error": True},
    }
    updater._record_intra_attempt(graph, state)
    updater._record_execution_facts(graph, state, state["execution_result"])

    node = graph.get_node("func_1")
    assert node.scenarios == []
    assert node.intra_attempt_count == 0


# Issue 6: Compliance relevance does not override failure yielding.

def test_compliance_relevance_prefers_functions_without_recent_failure():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="\u9690\u79c1\u8bbe\u7f6e", test_status="in_progress"))
    graph.add_node(FunctionNode(id="func_2", name="\u9690\u79c1\u653f\u7b56", test_status="in_progress"))
    graph.add_page(Page(id="page_1", name="\u9690\u79c1\u8bbe\u7f6e\u9875"), "func_1")
    graph.add_page(Page(id="page_2", name="\u9690\u79c1\u653f\u7b56\u9875"), "func_2")
    graph.add_page_transition(PageTransition(src_page_id="page_1", dst_page_id="page_2", operation="\u70b9\u51fb\u9690\u79c1\u653f\u7b56"))

    selected = planner._select_intra_target(
        graph, graph.get_page("page_1"), {"func_1": 1}, ("\u9690\u79c1",),
    )

    assert selected is not None and selected.id == "func_2"


# Issue 7: Observe a zero-control functionality before deferring it.

def test_zero_control_function_gets_observation_template():
    graph = _intra_graph()
    scenario = planner._forced_intra_scenario(graph, graph.get_node("func_1"))
    assert "\u6d3b\u52a8\u4e3b\u9875" in scenario and "Observe" in scenario


# Depth scenario targets scale with control counts.

def _graph_with_controls(control_count: int) -> FlowGraph:
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="\u8bb0\u8d26", test_status="in_progress"))
    graph.add_page(Page(id="page_1", name="\u8bb0\u8d26\u9875",
                        controls=[f"\u63a7\u4ef6{index}" for index in range(control_count)]), "func_1")
    return graph


def test_depth_target_is_ceil_of_controls_over_five():
    for controls, expected in ((1, 1), (5, 1), (6, 2), (11, 3), (30, 6), (60, 6)):
        graph = _graph_with_controls(controls)
        assert planner._required_depth_count(graph, graph.get_node("func_1")) == expected


def test_function_stays_in_depth_until_depth_target():
    graph = _graph_with_controls(8)
    node = graph.get_node("func_1")
    node.intra_attempt_count = planner.PLAN_BREADTH_MAX_SCENARIOS
    assert planner._function_phase(graph, node) == planner.FUNCTION_PHASE_DEPTH

    node.intra_attempt_count += 1
    node.depth_scenario_count = 1
    assert planner._function_phase(graph, node) == planner.FUNCTION_PHASE_DEPTH

    node.intra_attempt_count += 1
    node.depth_scenario_count = 2
    # Attempts without effective depth completion keep the functionality in its depth phase.
    assert planner._function_phase(graph, node) == planner.FUNCTION_PHASE_DEPTH
    node.successful_depth_scenario_count = 2
    assert planner._function_phase(graph, node) == planner.FUNCTION_PHASE_BREADTH


def test_depth_template_avoids_repeating_recorded_scenarios():
    graph = _graph_with_controls(4)
    scenario = planner._forced_depth_scenario(graph, graph.get_node("func_1"))
    assert "Do not repeat scenarios already executed for this functionality" in scenario
    assert "[verify:" not in scenario and "verification checkpoint" not in scenario


# Synonym merge: two labels referring to one control.

def test_synonym_control_merge_accepts_two_sources_with_existing_name_as_target():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="\u641c\u7d22"))
    page = Page(id="page_1", name="\u9996\u9875", controls=["\u641c\u7d22\u6309\u94ae", "\u9876\u90e8\u641c\u7d22\u56fe\u6807", "\u8bbe\u7f6e"])
    graph.add_page(page, "func_1")
    updater._apply_page_audit_control_merges(
        page, [{"sources": ["\u641c\u7d22\u6309\u94ae", "\u9876\u90e8\u641c\u7d22\u56fe\u6807"], "target": "\u641c\u7d22\u6309\u94ae"}],
    )
    assert page.controls == ["\u641c\u7d22\u6309\u94ae", "\u8bbe\u7f6e"]


def test_two_source_merge_with_new_target_name_is_rejected():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="\u641c\u7d22"))
    page = Page(id="page_1", name="\u9996\u9875", controls=["\u641c\u7d22\u6309\u94ae", "\u9876\u90e8\u641c\u7d22\u56fe\u6807", "\u8bbe\u7f6e"])
    graph.add_page(page, "func_1")
    updater._apply_page_audit_control_merges(
        page, [{"sources": ["\u641c\u7d22\u6309\u94ae", "\u9876\u90e8\u641c\u7d22\u56fe\u6807"], "target": "\u641c\u7d22"}],
    )
    assert page.controls == ["\u641c\u7d22\u6309\u94ae", "\u9876\u90e8\u641c\u7d22\u56fe\u6807", "\u8bbe\u7f6e"]



def test_synonym_merge_of_toggle_state_names_is_rejected():
    """Do not merge opposing enable/disable toggle labels as synonyms."""
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="\u624b\u7535\u7b52"))
    controls = ["\u5f00\u542f\u624b\u7535\u7b52\u6309\u94ae", "\u5173\u95ed\u624b\u7535\u7b52\u6309\u94ae", "\u8fd4\u56de\u6309\u94ae"]
    page = Page(id="page_1", name="\u624b\u7535\u7b52\u9875", controls=list(controls))
    graph.add_page(page, "func_1")
    merged = updater._apply_page_audit_control_merges(
        page, [{"sources": ["\u5f00\u542f\u624b\u7535\u7b52\u6309\u94ae", "\u5173\u95ed\u624b\u7535\u7b52\u6309\u94ae"], "target": "\u5f00\u542f\u624b\u7535\u7b52\u6309\u94ae"}],
    )
    assert merged == []
    assert page.controls == controls

# Issue 9: Preserve diagnostics for detection unfinished at flush timeout.

class _BlockingWorker(SerialWorker):
    log_prefix = "[\u6d4b\u8bd5\u7ebf\u7a0b]"
    thread_name = "test-blocking-worker"

    def __init__(self):
        super().__init__()
        self.release = threading.Event()
        self.failed_jobs: list[dict] = []

    def _process_job(self, job):
        self.release.wait(5)

    def _on_failure_locked(self, job, exc):
        self.failed_jobs.append(job)


def test_flush_timeout_records_unfinished_reviews_once():
    worker = _BlockingWorker()
    worker._configure_base(llm=None)
    worker.submit({"iteration_count": 7, "current_test_scenario": "\u573a\u666fA", "current_plan": {}})
    try:
        assert worker.flush(timeout=0.05) is False
        assert worker.record_unfinished("\u6536\u5c3e\u524d\u7b49\u5f85\u8d85\u65f6") == 1
        assert worker.record_unfinished("\u6536\u5c3e\u524d\u7b49\u5f85\u8d85\u65f6") == 0
        failures = worker.failures_snapshot()
        assert failures[0]["error_type"] == "ReviewUnfinished"
        assert failures[0]["iteration_count"] == 7
        assert len(worker.failed_jobs) == 1
    finally:
        worker.release.set()
        assert worker.flush(timeout=5)


def test_every_image_block_uses_a_jpeg_data_url():
    """Every planner image block needs a valid JPEG data URL to avoid request rejection."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    offenders = []
    for path in list((root / "workflow").rglob("*.py")) + list((root / "compliance").rglob("*.py")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r'"url":\s*f"', line) and 'f"data:image/jpeg;base64,{' not in line:
                offenders.append(f"{path.relative_to(root)}:{number}")
    assert offenders == []


def test_enumerated_control_names_are_detected_generically():
    enumerating = [
        "\u5de5\u5177\u7bb1\u5de5\u5177\u9879（\u523b\u5ea6\u5c3a、\u624b\u6301\u5f39\u5e55、\u6307\u5357\u9488、\u624b\u7535\u7b52）",
        "\u529f\u80fd\u5165\u53e3(A/B/C)",
        "\u626b\u63cf、\u6e05\u7406、\u52a0\u901f\u5165\u53e3",
        "\u8bbe\u7f6e\u9879（\u901a\u77e5，\u9690\u79c1，\u5173\u4e8e）",
    ]
    single = [
        "\u9884\u8bbe\u80cc\u666f\u56fe\u9879（\u4ee3\u8868：\u84dd\u5929）",
        "\u529f\u80fd\u5361\u7247（\u5982\u5929\u6c14、\u65e5\u5386\u7b49）",
        "\u4fdd\u5b58\u6309\u94ae（\u53f3\u4e0a\u89d2）",
        "\u6e05\u7406、\u52a0\u901f\u5165\u53e3",
        "\u666e\u901a\u6d88\u606f\u6761\u76ee",
    ]
    assert all(updater._control_name_enumerates_items(name) for name in enumerating)
    assert not any(updater._control_name_enumerates_items(name) for name in single)


def test_enumerated_control_is_not_written_and_is_reported():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="\u5de5\u5177\u7bb1"))
    graph.add_page(Page(id="page_1", name="\u5de5\u5177\u7bb1\u9875", controls=["\u8fd4\u56de\u6309\u94ae"]), "func_1")
    trace = [{"raw_step_index": 2, "current_page": Page(name="\u5de5\u5177\u7bb1\u9875")}]
    patch = {
        "observation_alignment": [{"source_step": 2, "page_decision": "reuse", "page_ref": "page_1"}],
        "pages": [{
            "page_ref": "page_1",
            "function_refs": ["func_1"],
            "source_step": 2,
            "controls": ["\u5de5\u5177\u9879（\u523b\u5ea6\u5c3a、\u6307\u5357\u9488、\u6c34\u5e73\u4eea）", "\u6307\u5357\u9488\u5165\u53e3"],
        }],
    }

    result = updater.apply_ffg_patch(graph, patch, trace)

    controls = graph.pages["page_1"].controls
    assert "\u6307\u5357\u9488\u5165\u53e3" in controls
    assert not any("\u523b\u5ea6\u5c3a" in control for control in controls)
    issues = result["control_naming_issues"]
    assert len(issues) == 1 and "page_1" in issues[0] and "source_step 2" in issues[0]
    # A naming issue does not prevent applying other content.
    assert "\u6307\u5357\u9488\u5165\u53e3" in graph.pages["page_1"].controls
