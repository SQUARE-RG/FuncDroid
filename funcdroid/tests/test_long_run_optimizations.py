# Multilingual UI and model-response fixtures use Unicode escapes to preserve their exact text.
"""Long-run regressions for state separation, failure yielding, budgets, and audit deduplication."""

import pytest

from model.flow_graph import (
    FunctionalFlowGraph,
    FunctionDependency,
    FunctionalityNode,
    Page,
    PageTransition,
)
from workflow.nodes.ffg_update import (
    _record_completed_intra_scenario,
    _record_execution_outcome,
)
from workflow.nodes.scenario_plan import (
    PLAN_FUNCTION_COVERAGE_TARGET,
    _function_audit_signature,
    _function_scenario_cap,
    _node_uncovered_control_hints,
    _normalize_and_reopen_undercovered_functions,
    _pending_cross_pairs,
    _intra_sort_key,
    _select_intra_target,
    create_scenario_plan_node,
)


@pytest.fixture(autouse=True)
def _legacy_text_matching_enabled(monkeypatch):
    """Explicitly enable legacy text fallback to preserve these pre-binding tests' intended coverage semantics."""
    from model import flow_graph as _flow_graph

    monkeypatch.setattr(_flow_graph, "COVERAGE_LEGACY_TEXT_MATCHING", True)


def _intra_state(function_id, execution_result, failure_counts=None):
    return {
        "current_plan": {
            "scenario_type": "intra_test",
            "target_function_id": function_id,
        },
        "function_execution_failure_counts": failure_counts or {},
    }, execution_result


# P0: State separation.

def test_function_node_accepts_terminal_statuses():
    for status in ("blocked", "deferred", "unavailable"):
        node = FunctionalityNode(id="func_1", test_status=status)
        assert node.test_status == status


def test_flow_graph_set_status_accepts_terminal_statuses():
    graph = FunctionalFlowGraph()
    graph.add_node(FunctionalityNode(id="func_1"))
    for status in ("blocked", "deferred", "unavailable"):
        graph.set_node_test_status("func_1", status)
        assert graph.get_node_test_status("func_1") == status


def test_stalled_execution_flags_function_but_stays_in_progress():
    graph = FunctionalFlowGraph()
    graph.add_node(FunctionalityNode(id="func_1", test_status="in_progress"))
    state, execution_result = _intra_state(
        "func_1",
        {"scenario_status": "stalled", "scenario_error": "\u8fde\u7eed\u591a\u6b65\u9875\u9762\u65e0\u53d8\u5316"},
    )

    counts = _record_execution_outcome(graph, state, execution_result)

    assert counts == {"func_1": 1}
    assert graph.get_node_test_status("func_1") == "in_progress"


def test_test_phase_blocked_marks_function_unavailable_with_reason():
    graph = FunctionalFlowGraph()
    graph.add_node(FunctionalityNode(id="func_1", test_status="in_progress"))
    state, execution_result = _intra_state(
        "func_1",
        {
            "navigation_status": "completed",
            "scenario_status": "blocked",
            "scenario_error": "\u6ed1\u5757\u9a8c\u8bc1\u65e0\u6cd5\u7531\u65e0\u4eba\u503c\u5b88\u6d4b\u8bd5\u5b8c\u6210",
        },
        failure_counts={"func_1": 1},
    )

    counts = _record_execution_outcome(graph, state, execution_result)

    assert graph.get_node_test_status("func_1") == "unavailable"
    assert graph.get_node("func_1").status_reason == "blocked"
    assert counts == {}


def test_navigation_blocked_counts_navigation_failure_instead_of_unavailable():
    graph = FunctionalFlowGraph()
    graph.add_node(FunctionalityNode(id="func_1", test_status="in_progress"))
    state, execution_result = _intra_state(
        "func_1",
        {"navigation_status": "blocked", "scenario_status": "skipped"},
    )

    counts = _record_execution_outcome(graph, state, execution_result)

    assert graph.get_node_test_status("func_1") == "in_progress"
    assert graph.get_node("func_1").navigation_failure_count == 1
    assert counts == {"func_1": 1}


def test_repeated_navigation_failures_defer_function_with_reopenable_reason():
    from workflow.nodes.ffg_update import NAVIGATION_FAILURE_DEFER_LIMIT

    graph = FunctionalFlowGraph()
    graph.add_node(FunctionalityNode(id="func_1", test_status="in_progress"))
    state, execution_result = _intra_state(
        "func_1",
        {"navigation_status": "stalled", "scenario_status": "skipped"},
    )
    for _ in range(NAVIGATION_FAILURE_DEFER_LIMIT):
        _record_execution_outcome(graph, state, execution_result)

    node = graph.get_node("func_1")
    assert node.test_status == "deferred"
    assert node.status_reason == "navigation_failed"


def test_infrastructure_error_does_not_change_scheduling_state():
    graph = FunctionalFlowGraph()
    graph.add_node(FunctionalityNode(id="func_1", test_status="in_progress"))
    state, execution_result = _intra_state(
        "func_1",
        {"scenario_status": "error", "infrastructure_error": True},
    )

    counts = _record_execution_outcome(graph, state, execution_result)

    assert counts == {}
    assert graph.get_node_test_status("func_1") == "in_progress"


def test_set_node_test_status_rejects_unknown_status():
    graph = FunctionalFlowGraph()
    graph.add_node(FunctionalityNode(id="func_1"))
    with pytest.raises(ValueError):
        graph.set_node_test_status("func_1", "done")


def test_crash_flags_function_without_marking_unavailable():
    graph = FunctionalFlowGraph()
    graph.add_node(FunctionalityNode(id="func_1", test_status="in_progress"))
    state, execution_result = _intra_state(
        "func_1",
        {"scenario_status": "app_crashed", "crash_detected": True},
    )

    counts = _record_execution_outcome(graph, state, execution_result)

    assert counts == {"func_1": 1}
    assert graph.get_node_test_status("func_1") == "in_progress"


def test_crashed_cross_test_flags_source_without_marking_unavailable():
    graph = FunctionalFlowGraph()
    graph.add_node(FunctionalityNode(id="func_1", test_status="in_progress"))
    graph.add_node(FunctionalityNode(id="func_2", test_status="in_progress"))
    state = {
        "current_plan": {"scenario_type": "cross_test", "src_function_id": "func_1", "dst_function_id": "func_2"},
        "current_test_scenario": "\u8de8\u529f\u80fd\u5d29\u6e83\u573a\u666f",
    }
    execution_result = {"scenario_status": "app_crashed", "crash_detected": True}

    counts = _record_execution_outcome(graph, state, execution_result)

    assert counts == {"func_1": 1}
    assert graph.get_node_test_status("func_1") == "in_progress"
    assert graph.cross_test_results[0]["status"] == "failed"


def test_screenshot_unavailable_is_not_counted_as_failure():
    graph = FunctionalFlowGraph()
    graph.add_node(FunctionalityNode(id="func_1", test_status="in_progress"))
    state, execution_result = _intra_state("func_1", {"scenario_status": "screenshot_unavailable"})

    assert _record_execution_outcome(graph, state, execution_result) == {}
    assert graph.get_node_test_status("func_1") == "in_progress"


# P1a: Failure yielding and priority.

def _two_reachable_functions() -> tuple[FunctionalFlowGraph, Page]:
    graph = FunctionalFlowGraph()
    graph.add_node(FunctionalityNode(id="func_1", name="\u529f\u80fd\u4e00", test_status="in_progress"))
    graph.add_node(FunctionalityNode(id="func_2", name="\u529f\u80fd\u4e8c", test_status="not_started"))
    page_1 = Page(id="page_1", name="\u529f\u80fd\u4e00\u5165\u53e3")
    page_2 = Page(id="page_2", name="\u529f\u80fd\u4e8c\u5165\u53e3")
    graph.add_page(page_1, "func_1")
    graph.add_page(page_2, "func_2")
    graph.add_page_transition(PageTransition("page_1", "page_2", "\u70b9\u51fb\u529f\u80fd\u4e8c\u5165\u53e3"))
    return graph, page_1


def test_priority_unfinished_nodes_puts_failed_last():
    graph, _ = _two_reachable_functions()
    graph.get_node("func_2").test_status = "in_progress"

    order = [
        n.id
        for n in sorted(
            graph.functions.values(),
            key=lambda n: _intra_sort_key(graph, n, [], {"func_1": 1}),
        )
    ]

    assert order == ["func_2", "func_1"]


def test_select_intra_target_yields_recently_failed_function():
    graph, current_page = _two_reachable_functions()

    target = _select_intra_target(graph, current_page, {"func_1": 1})

    assert target.id == "func_2"


def test_select_intra_target_keeps_actual_current_function_before_unstarted_function():
    graph, current_page = _two_reachable_functions()

    target = _select_intra_target(graph, current_page, {})

    assert target.id == "func_1"


def test_select_intra_target_prioritizes_unstarted_function_sharing_current_entry_page():
    graph = FunctionalFlowGraph()
    graph.add_node(FunctionalityNode(
        id="func_1",
        name="\u4f1a\u8bdd\u6d88\u606f\u6d4f\u89c8\u4e0e\u6536\u53d1",
        test_status="in_progress",
    ))
    graph.add_node(FunctionalityNode(
        id="func_2",
        name="\u4ece\u76f8\u518c\u9009\u62e9\u5e76\u53d1\u9001\u5a92\u4f53",
        test_status="not_started",
    ))
    entry_page = Page(id="page_1", name="\u4f1a\u8bdd\u804a\u5929\u9875\u9762")
    graph.add_page(entry_page, "func_1")
    graph.add_page(entry_page, "func_2")
    graph.get_node("func_1").add_scenario("\u6d4f\u89c8\u4f1a\u8bdd\u6d88\u606f")

    target = _select_intra_target(graph, entry_page, {})

    assert target.id == "func_2"


# P1a: Per-functionality scenario budget.

def test_scenario_budget_does_not_complete_when_program_coverage_is_low():
    graph = FunctionalFlowGraph()
    node = FunctionalityNode(id="func_1", name="\u5e9e\u5927\u529f\u80fd", test_status="in_progress")
    graph.add_node(node)
    graph.add_page(Page(id="page_1", name="\u5165\u53e3", controls=["\u641c\u7d22\u6309\u94ae"]), "func_1")
    for index in range(_function_scenario_cap(graph, node)):
        node.add_scenario(f"\u573a\u666f {index}")

    class Llm:
        def invoke(self, _messages):
            return type("Response", (), {"content": (
                '{"scenario_type":"intra_test","target_function":"func_1",'
                '"src_function":"","dst_function":"","navigation_scenario":"",'
                '"test_scenario":"\u70b9\u51fb\u641c\u7d22\u6309\u94ae → \u89c2\u5bdf\u641c\u7d22\u9875\u9762\u540e\u7ed3\u675f"}'
            )})()

    class Device:
        def current_package(self):
            return ""

    result = create_scenario_plan_node(Llm(), Device())({
        "ffg": graph,
        "current_page": graph.get_page("page_1"),
        "current_testing_function_id": "func_1",
        "current_testing_function_status": "in_progress",
        "run_phase": "all",
        "current_plan": {},
    })

    # At the scenario cap, defer; finalization adds one attempt without restoring in_progress.
    assert graph.get_node_test_status("func_1") == "deferred"
    assert result["current_plan"]["scenario_type"] == "intra_test"
    assert result["deferred_retry_ids"] == ["func_1"]


def test_coverage_target_forces_completion_without_llm(monkeypatch):
    # Production retry logic treats the fake LLM AssertionError as a failed call and backs off.
    # Use one attempt with zero delay to avoid retrying exhausted scripted responses.
    monkeypatch.setenv("LLM_RETRY_ATTEMPTS", "1")
    monkeypatch.setenv("LLM_RETRY_DELAY", "0")
    monkeypatch.setenv("LLM_RETRY_MAX_DELAY", "0")
    graph = FunctionalFlowGraph()
    node = FunctionalityNode(
        id="func_1",
        name="\u5df2\u5145\u5206\u8986\u76d6\u529f\u80fd",
        test_status="in_progress",
    )
    node.add_scenario("\u5b8c\u6210\u5f53\u524d\u9875\u9762\u7684\u4ee3\u8868\u6027\u4e1a\u52a1\u95ed\u73af")
    graph.add_node(node)
    graph.add_page(Page(
        id="page_1",
        name="\u5f53\u524d\u9875\u9762",
        controls=["\u641c\u7d22\u6309\u94ae", "\u5185\u5bb9\u5361\u7247"],
        operations=["\u70b9\u51fb\u641c\u7d22\u6309\u94ae", "\u70b9\u51fb\u5185\u5bb9\u5361\u7247"],
    ), "func_1")
    # Coverage and depth evidence are already sufficient; finalization needs no planner call.
    node.depth_scenario_count = 1
    node.successful_depth_scenario_count = 1
    graph.recompute_all_coverages()

    class Llm:
        def invoke(self, _messages):
            raise AssertionError("\u8fbe\u5230\u8986\u76d6\u7387\u9608\u503c\u540e\u4e0d\u5f97\u518d\u8c03\u7528\u89c4\u5212 LLM")

    class Device:
        def current_package(self):
            return ""

    create_scenario_plan_node(Llm(), Device())({
        "ffg": graph,
        "current_page": graph.get_page("page_1"),
        "current_testing_function_id": "func_1",
        "current_testing_function_status": "in_progress",
        "run_phase": "all",
        "current_plan": {},
    })

    assert graph.get_node_test_status("func_1") == "completed"


def test_expanded_entry_placeholder_is_renamed_and_reopened():
    graph = FunctionalFlowGraph()
    node = FunctionalityNode(
        id="func_1",
        name="\u670d\u52a1\u53f7\u5165\u53e3",
        test_status="completed",
        coverage_percent=100,
        scenarios=["\u70b9\u51fb\u670d\u52a1\u53f7\u6761\u76ee\u5e76\u8fdb\u5165\u9875\u9762"],
    )
    graph.add_node(node)
    graph.add_page(Page(id="page_1", name="\u804a\u5929\u5217\u8868"), "func_1")
    graph.add_page(Page(id="page_2", name="\u670d\u52a1\u53f7\u9875\u9762"), "func_1")

    reopened = _normalize_and_reopen_undercovered_functions(graph)

    assert reopened == ["func_1"]
    assert node.name == "\u670d\u52a1\u53f7"
    assert node.test_status == "in_progress"
    assert node.coverage_percent == 0


def test_multi_page_coverage_needs_representative_scenario_per_page():
    graph = FunctionalFlowGraph()
    node = FunctionalityNode(
        id="func_1",
        name="\u5185\u5bb9\u6d4f\u89c8",
        test_status="in_progress",
        coverage_percent=PLAN_FUNCTION_COVERAGE_TARGET,
        scenarios=["\u8fdb\u5165\u5185\u5bb9\u9875\u9762"],
    )
    graph.add_node(node)
    graph.add_page(Page(id="page_1", name="\u5165\u53e3\u9875\u9762"), "func_1")
    graph.add_page(Page(id="page_2", name="\u5185\u5bb9\u9875\u9762", controls=["\u641c\u7d22\u6309\u94ae"]), "func_1")

    class Llm:
        calls = 0

        def invoke(self, _messages):
            self.calls += 1
            return type("Response", (), {"content": (
                '{"scenario_type":"intra_test","target_function":"func_1",'
                '"src_function":"","dst_function":"","navigation_scenario":"",'
                '"test_scenario":"\u70b9\u51fb\u641c\u7d22\u6309\u94ae → \u770b\u5230\u641c\u7d22\u9875\u9762\u540e\u7ed3\u675f"}'
            )})()

    class Device:
        def current_package(self):
            return ""

    llm = Llm()
    result = create_scenario_plan_node(llm, Device())({
        "ffg": graph,
        "current_page": graph.get_page("page_2"),
        "current_testing_function_id": "func_1",
        "current_testing_function_status": "in_progress",
        "run_phase": "all",
        "current_plan": {},
    })

    assert llm.calls == 1
    assert graph.get_node_test_status("func_1") == "in_progress"
    assert result["current_test_scenario"].startswith("\u70b9\u51fb\u641c\u7d22\u6309\u94ae")


def test_repeated_entry_observation_does_not_count_as_new_scenario():
    graph = FunctionalFlowGraph()
    node = FunctionalityNode(
        id="func_1",
        name="\u670d\u52a1\u53f7",
        test_status="completed",
        coverage_percent=79,
        scenarios=[
            "\u70b9\u51fb\u670d\u52a1\u53f7\u5165\u53e3\u6761\u76ee → \u8fdb\u5165\u670d\u52a1\u53f7\u9875\u9762\u540e\u7ed3\u675f",
            "\u70b9\u51fb\u670d\u52a1\u53f7\u5165\u53e3\u6761\u76ee → \u7b49\u5f85\u5185\u5bb9\u52a0\u8f7d → \u770b\u5230\u5185\u5bb9\u5361\u7247\u540e\u7ed3\u675f",
        ],
    )
    graph.add_node(node)
    graph.add_page(
        Page(id="page_1", name="\u804a\u5929\u5217\u8868", operations=["\u70b9\u51fb\u670d\u52a1\u53f7\u5165\u53e3\u6761\u76ee"]),
        "func_1",
    )
    graph.add_page(
        Page(
            id="page_2",
            name="\u670d\u52a1\u53f7\u9875\u9762",
            controls=["\u9876\u90e8\u8fd4\u56de\u6309\u94ae", "\u9875\u9762\u6807\u9898“\u670d\u52a1\u53f7”", "\u641c\u7d22\u6309\u94ae", "\u5173\u6ce8\u8d26\u53f7\u5185\u5bb9\u5361\u7247"],
            operations=["\u70b9\u51fb\u9875\u9762\u5de6\u4e0a\u89d2\u7684\u8fd4\u56de\u7bad\u5934"],
        ),
        "func_1",
    )

    assert _node_uncovered_control_hints(graph, node) == [
        "\u9875\u9762\u6807\u9898“\u670d\u52a1\u53f7”",
        "\u641c\u7d22\u6309\u94ae",
        "\u5173\u6ce8\u8d26\u53f7\u5185\u5bb9\u5361\u7247",
    ]

    reopened = _normalize_and_reopen_undercovered_functions(graph)

    assert reopened == ["func_1"]
    assert node.test_status == "in_progress"


def test_completed_scenario_is_recorded_on_planned_target_function():
    """Credit scenarios to the planned target functionality, using the endpoint only as fallback. Shared endpoint pages previously stole own-scenario evidence and blocked completion despite sufficient coverage."""
    graph, _ = _two_reachable_functions()

    owner_id = _record_completed_intra_scenario(
        graph,
        {"scenario_type": "intra_test", "target_function_id": "func_1"},
        "\u70b9\u51fb\u5165\u53e3 → \u5b8c\u6210\u76ee\u6807\u80fd\u529b\u540e\u7ed3\u675f",
    )

    assert owner_id == "func_1"
    assert graph.get_node("func_1").scenarios == ["\u70b9\u51fb\u5165\u53e3 → \u5b8c\u6210\u76ee\u6807\u80fd\u529b\u540e\u7ed3\u675f"]
    assert graph.get_node("func_2").scenarios == []


# P2: Deduplicate audit signatures.

def test_blocked_function_is_excluded_from_pending_cross_pairs():
    graph = FunctionalFlowGraph()
    graph.add_node(FunctionalityNode(id="func_1", name="\u6e90\u529f\u80fd", test_status="blocked"))
    graph.add_node(FunctionalityNode(id="func_2", name="\u76ee\u6807\u529f\u80fd", test_status="completed"))
    graph.add_function_dependency(FunctionDependency("func_1", "func_2", "\u4efb\u52a1\u8bb0\u5f55"))

    assert _pending_cross_pairs(graph) == []


def test_audit_signature_ignores_scenario_count_but_tracks_structure():
    graph = FunctionalFlowGraph()
    graph.add_node(FunctionalityNode(id="func_1", test_status="completed"))
    graph.add_page(
        Page(id="page_1", name="\u5165\u53e3", controls=["\u6309\u94ae"], operations=["\u70b9\u51fb\u6309\u94ae"]),
        "func_1",
    )
    before = _function_audit_signature(graph, graph.get_node("func_1"))

    graph.add_scenario("func_1", "\u65b0\u573a\u666f")

    assert _function_audit_signature(graph, graph.get_node("func_1")) == before

    # Audit exploration adds operation records; counting them in signatures causes repeated audits.
    graph.get_page("page_1").add_operation("\u70b9\u51fb\u65b0\u5165\u53e3")

    assert _function_audit_signature(graph, graph.get_node("func_1")) == before

    # New controls are new structural coverage facts and require another audit.
    graph.get_page("page_1").controls.append("\u65b0\u5165\u53e3")

    assert _function_audit_signature(graph, graph.get_node("func_1")) != before


# H2 regression: audit execution failures must preserve terminal functionality states.

@pytest.mark.parametrize(
    "execution_result",
    [
        {"scenario_status": "blocked", "scenario_error": "\u6ed1\u5757\u9a8c\u8bc1\u65e0\u6cd5\u7531\u65e0\u4eba\u503c\u5b88\u6d4b\u8bd5\u5b8c\u6210"},
        {"scenario_status": "app_crashed", "crash_detected": True},
        {"scenario_status": "stalled", "scenario_error": "\u8fde\u7eed\u591a\u6b65\u9875\u9762\u65e0\u53d8\u5316"},
    ],
)
def test_terminal_status_is_not_flipped_by_execution_failure(execution_result):
    """An execution failure during a coverage audit must not change a completed functionality or mark it failed."""
    graph = FunctionalFlowGraph()
    graph.add_node(FunctionalityNode(id="func_1", test_status="completed"))
    state, execution_result = _intra_state("func_1", execution_result)

    assert _record_execution_outcome(graph, state, execution_result) == {}
    assert graph.get_node_test_status("func_1") == "completed"


# H1 regression: persistent screenshot failures must have bounded restarts.

def test_screenshot_failure_restart_loop_is_bounded(monkeypatch):
    """Bound screenshot-failure recovery even when restart_app cannot help. Failures add no trace steps, so the step budget alone cannot prevent an infinite restart loop."""
    from workflow.nodes import scenario_exec

    class Device:
        def __init__(self):
            self.restarts = 0

        def restart_app(self, _package_name):
            self.restarts += 1

        def display_size(self):
            raise AssertionError("display_size must not be reached")

    class Llm:
        def invoke(self, _messages):
            raise AssertionError("llm must not be called when screenshots fail")

    monkeypatch.setattr(scenario_exec, "_ensure_target_app", lambda *_args: True)
    monkeypatch.setattr(scenario_exec, "_target_process_pid", lambda *_args: "123")
    monkeypatch.setattr(
        scenario_exec,
        "_target_app_foreground_state",
        lambda *_args: ("foreground", "com.example"),
    )

    def _raise_screenshot_failure(*_args, **_kwargs):
        raise RuntimeError("screenshot transport down")

    monkeypatch.setattr(scenario_exec, "capture_runtime_page", _raise_screenshot_failure)
    monkeypatch.setattr(
        scenario_exec, "_append_terminal_observation", lambda *_a, **_k: (None, "")
    )
    monkeypatch.setattr(scenario_exec.time, "sleep", lambda _seconds: None)

    device = Device()
    trace, _page, status, error_message, _crash = scenario_exec._run_single_scenario(
        Llm(),
        device,
        "\u793a\u4f8b\u5e94\u7528",
        "com.example",
        "\u70b9\u51fb\u767b\u5f55\u6309\u94ae",
        Page(id="page_1", name="\u4e3b\u9875"),
        clear_logcat=False,
        trace_phase="test",
    )

    assert status == "error"
    assert "screenshot failures" in error_message
    assert device.restarts == scenario_exec.APP_RESTART_ATTEMPTS
    assert trace == []
