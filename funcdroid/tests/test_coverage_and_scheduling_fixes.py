"""Regressions for the coverage-accounting and scheduling fixes."""

from __future__ import annotations

import pytest

from model.flow_graph import FlowGraph, FunctionNode, Page
from workflow.nodes.ffg_update import (
    _find_control_signature_page,
    _merge_page_observation,
    _normalize_observation_alignment,
    partition_alignment_issues,
    validate_observation_alignment,
)
from workflow.nodes.scenario_plan import (
    PLAN_FUNCTION_COVERAGE_TARGET,
    PLAN_MIN_SCENARIOS_BEFORE_COMPLETE,
    PLAN_UNPRODUCTIVE_RELEASE_LIMIT,
    _force_progress_after_planner_noop,
    _is_similar_scenario,
    _node_intra_attempt_count,
    _normalize_and_reopen_undercovered_functions,
)
from workflow.subgraphs.ffg_update import create_ffg_update_subgraph


@pytest.fixture(autouse=True)
def _legacy_text_matching_enabled(monkeypatch):
    """本文件用例写于操作-控件显式绑定之前，守护文本匹配引擎的边界行为。

    显式绑定是覆盖率的权威来源且文本兜底默认关闭
    （COVERAGE_LEGACY_TEXT_MATCHING）；这些用例断言的是旧引擎的防
    虚增/防误配守卫，需显式开启兜底。
    """
    from model import flow_graph as _flow_graph

    monkeypatch.setattr(_flow_graph, "COVERAGE_LEGACY_TEXT_MATCHING", True)


def test_repeated_operations_do_not_inflate_control_coverage():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="搜索"))
    graph.add_page(Page(
        id="page_1",
        controls=[f"控件{index}" for index in range(10)],
        operations=["点击控件0"] * 9,
    ), "func_1")

    details = graph.function_coverage_details("func_1")

    assert details["total_controls"] == 10
    assert details["executed_operations"] == 1
    assert details["coverage_percent"] < 20


def test_distinct_operations_on_same_control_do_not_inflate_coverage():
    """操作数顶替命中数的回归：4 条不同表述只命中 1 个控件时覆盖率 25% 而非 100%。"""
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="搜索"))
    graph.add_page(Page(
        id="page_1",
        controls=["顶部搜索按钮", "底部收藏按钮", "底部分享按钮", "底部评论按钮"],
        operations=[
            "点击顶部搜索按钮",
            "点击顶部搜索图标",
            "点击顶部搜索入口",
            "点击顶部搜索框",
        ],
    ), "func_1")

    details = graph.function_coverage_details("func_1")

    assert details["total_controls"] == 4
    assert details["executed_operations"] == 1
    assert details["coverage_percent"] == 25


def test_uncovered_controls_are_reported_in_full():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="设置"))
    graph.add_page(Page(
        id="page_1",
        controls=["蓝牙开关", "音量滑块", "亮度滑块"],
        operations=["点击蓝牙开关", "点击蓝牙开关", "点击蓝牙开关"],
    ), "func_1")

    details = graph.function_coverage_details("func_1")

    assert details["uncovered_controls"] == ["音量滑块", "亮度滑块"]


def test_numeric_suffix_controls_are_not_treated_as_covered():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="列表"))
    graph.add_page(Page(
        id="page_1",
        controls=["控件1", "控件12"],
        operations=["点击控件1"],
    ), "func_1")

    assert graph.function_coverage_details("func_1")["uncovered_controls"] == ["控件12"]


def test_navigation_only_entry_page_does_not_contribute_function_coverage():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="主页"))
    graph.add_node(FunctionNode(
        id="func_2",
        name="相册选择",
        scenarios=["从主页点击相册入口"],
    ))
    page = Page(
        id="page_1",
        controls=["相册入口", "设置入口"],
        operations=["点击相册入口"],
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
        name="主页",
        test_status="completed",
        scenarios=["点击搜索按钮并观察结果", "点击搜索按钮并确认结果"],
    )
    graph.add_node(node)
    graph.add_page(Page(id="page_1", controls=["搜索按钮", "菜单按钮"]), "func_1")

    _normalize_and_reopen_undercovered_functions(graph)

    assert len(graph.get_node("func_1").scenarios) == 2
    assert graph.get_node("func_1").test_status == "in_progress"


def test_partially_overlapping_scenarios_are_not_duplicates():
    left = "在设置页，点击“通知”开关，再点击“保存”按钮，观察结果。"
    right = "在设置页，点击“定位”开关，再点击“保存”按钮，观察结果。"

    assert _is_similar_scenario(left, right) is False


def test_cross_pair_is_tested_without_a_dependency_edge():
    from workflow.nodes.scenario_plan import _tested_cross_pair_keys

    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="相册"))
    graph.add_node(FunctionNode(id="func_2", name="聊天"))
    graph.add_cross_scenario("func_1", "func_2", "从相册选图并发送到会话")

    assert not graph.function_dependencies
    tested_keys = _tested_cross_pair_keys(graph)
    assert frozenset(("func_1", "func_2")) in tested_keys
    assert frozenset(("func_2", "func_1")) in tested_keys


def test_repeatedly_unproductive_function_gets_a_terminal_status():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="空功能", test_status="in_progress"))

    result = _force_progress_after_planner_noop(
        graph,
        None,
        "core",
        "",
        "not_started",
        {
            "ffg": graph,
            "run_phase": "all",
            "function_execution_failure_counts": {
                "func_1": PLAN_UNPRODUCTIVE_RELEASE_LIMIT - 1
            },
        },
        "无可执行场景",
        1,
    )

    assert result is not None
    assert graph.get_node_test_status("func_1") == "unavailable"
    assert result["function_execution_failure_counts"] == {}


def test_same_activity_and_control_signature_reuses_existing_page():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="笔记"))
    graph.add_page(Page(
        id="page_1",
        name="笔记列表",
        activity="NotesActivity",
        controls=["搜索按钮", "汉堡菜单按钮", "新增笔记按钮", "笔记条目"],
    ), "func_1")

    reused = _find_control_signature_page(
        graph,
        Page(activity="NotesActivity"),
        ["搜索按钮", "汉堡菜单按钮", "新增笔记按钮", "笔记条目"],
    )
    distinct = _find_control_signature_page(
        graph,
        Page(activity="NotesActivity"),
        ["返回按钮", "标题输入框", "正文输入框", "保存按钮"],
    )

    assert reused is not None and reused.id == "page_1"
    assert distinct is None


def test_same_activity_and_controls_do_not_merge_different_named_entities():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="聊天"))
    controls = ["返回按钮", "消息列表", "输入框", "发送按钮"]
    graph.add_page(Page(
        id="page_1",
        name="文件传输助手详情页",
        activity="LauncherUI",
        controls=controls,
    ), "func_1")

    matched = _find_control_signature_page(
        graph,
        Page(activity="LauncherUI"),
        controls,
        declared_name="测试群详情页",
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


def test_refinement_stops_when_structure_is_valid():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="功能", test_status="in_progress"))
    graph.add_page(Page(id="page_1", name="首页", activity="HomeActivity"), "func_1")

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
        "current_page": Page(id="page_1", name="首页", activity="HomeActivity"),
        "current_plan": {"scenario_type": "intra_test", "target_function_id": "func_1"},
        "navigation_execution_trace": [],
        "scenario_execution_trace": [{
            "raw_step_index": 1,
            "action_name": "click",
            "current_page": Page(id="page_1", name="首页", activity="HomeActivity"),
        }],
        "execution_result": {"scenario_status": "completed"},
    })

    assert AlwaysContinueLlm.calls == 1


def test_inherited_coverage_from_shared_page_does_not_complete_barely_tested_function():
    """共享页面的兄弟功能操作不得让一个仅测过一次的功能直接收尾。"""
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="文本发送", test_status="in_progress"))
    graph.add_node(FunctionNode(id="func_2", name="语音发送", test_status="in_progress"))
    shared = Page(
        id="page_1",
        name="聊天页",
        activity="ChatActivity",
        controls=["输入框", "发送按钮", "表情按钮", "语音按钮", "更多按钮"],
        operations=["点击输入框", "点击发送按钮", "点击表情按钮", "点击语音按钮"],
    )
    graph.add_page(shared, "func_1")
    graph.add_page(shared, "func_2")
    graph.get_node("func_1").scenarios = [
        "点击输入框输入文本并发送",
        "点击表情按钮并选择表情发送",
        "点击发送按钮确认消息进入聊天记录",
    ]
    # func_2 只跑过一个自己的场景，其余覆盖全部来自 func_1 在共享页面上的操作。
    graph.get_node("func_2").scenarios = ["点击语音按钮并发送语音"]
    for node_id in ("func_1", "func_2"):
        graph.recompute_function_coverage(node_id)

    node = graph.get_node("func_2")
    assert node.coverage_percent >= PLAN_FUNCTION_COVERAGE_TARGET
    assert graph.function_has_successful_business_operation("func_2") is True

    details = graph.function_coverage_details("func_2")
    intra_attempts = _node_intra_attempt_count(node)
    own_evidence = (
        not details["uncovered_controls"]
        or intra_attempts >= PLAN_MIN_SCENARIOS_BEFORE_COMPLETE
    )

    assert intra_attempts == 1
    assert details["uncovered_controls"] == ["更多按钮"]
    # 旧闸门（仅覆盖率 + 有成功业务操作）会在此直接判 completed。
    assert own_evidence is False


def test_function_completion_met_requires_own_evidence():
    """批量完成判定复用同一闸门：继承覆盖率但无自身证据时不得收尾。"""
    from workflow.nodes.scenario_plan import _function_completion_met

    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="聊天消息收发"))
    graph.add_node(FunctionNode(id="func_2", name="聊天消息收发"))
    shared = Page(
        id="page_1",
        name="消息列表页",
        activity="LauncherUI",
        controls=["输入框", "发送按钮", "表情按钮", "语音按钮", "更多按钮"],
        operations=[
            "点击输入框",
            "点击发送按钮",
            "点击表情按钮",
            "点击语音按钮",
        ],
    )
    graph.add_page(shared, "func_1")
    graph.add_page(shared, "func_2")
    graph.get_node("func_1").scenarios = ["点击输入框输入文本并发送"]
    graph.get_node("func_2").scenarios = ["点击语音按钮并发送语音"]
    for node_id in ("func_1", "func_2"):
        graph.recompute_function_coverage(node_id)

    # func_1：1 个自身场景，仍有未覆盖控件"更多按钮" → 自身证据不足。
    # 但覆盖率和成功业务操作都达标（全部继承自共享页面）→ 不得收尾。
    node_1 = graph.get_node("func_1")
    assert node_1.coverage_percent >= PLAN_FUNCTION_COVERAGE_TARGET
    assert graph.function_has_successful_business_operation("func_1") is True
    assert _function_completion_met(graph, node_1) is False

    # func_2：场景全部录满自身证据（达到预算或覆盖全部控件）→ 应可收尾。
    node_2 = graph.get_node("func_2")
    graph.get_node("func_2").scenarios = [
        "点击输入框并发送文本",
        "点击表情按钮并选择表情",
        "点击发送按钮确认发送",
        "点击语音按钮发送语音",
    ]
    graph.recompute_function_coverage("func_2")
    assert _function_completion_met(graph, node_2) is True


def test_create_alignment_with_existing_page_id_is_repaired_without_refinement():
    """create + 已有 page_id 属纯格式错误，应由代码归一化为 reuse。"""
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="首页"))
    graph.add_page(Page(id="page_1", name="首页", activity="HomeActivity"), "func_1")
    step_page = Page(name="首页", activity="HomeActivity")
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
    graph.add_node(FunctionNode(id="func_1", name="首页"))
    graph.add_page(Page(id="page_1", name="首页", activity="HomeActivity"), "func_1")
    trace = [{"raw_step_index": 1, "current_page": Page(name="详情", activity="DetailActivity")}]
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


def test_only_candidate_shortlist_warnings_are_advisory():
    blocking, advisory = partition_alignment_issues([
        "reuse alignment 2->page_7 is outside retrieved candidates",
        "missing observation_alignment for source_step 1",
        "create alignment must use new_page_x at source_step 3",
    ])

    assert advisory == ["reuse alignment 2->page_7 is outside retrieved candidates"]
    assert blocking == [
        "missing observation_alignment for source_step 1",
        "create alignment must use new_page_x at source_step 3",
    ]


def test_create_alignment_is_not_downgraded_without_candidate_support():
    """无检索候选佐证时不得改判 reuse，否则真正的新页面会被并进错误页面。"""
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="首页"))
    graph.add_page(Page(id="page_1", name="首页", activity="HomeActivity"), "func_1")
    trace = [{"raw_step_index": 1, "current_page": Page(name="详情", activity="DetailActivity")}]
    patch = {
        "pages": [],
        "observation_alignment": [
            {"source_step": 1, "page_decision": "create", "page_ref": "page_1"}
        ],
    }

    _normalize_observation_alignment(patch, graph, trace, {})

    assert patch["observation_alignment"][0]["page_decision"] == "create"
