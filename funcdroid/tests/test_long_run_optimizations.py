"""针对长时间运行优化的回归测试：状态拆分、失败让出、预算与审计去冗余。"""

import pytest

from model.flow_graph import (
    FlowGraph,
    FunctionDependency,
    FunctionNode,
    Page,
    PageTransition,
)
from workflow.nodes.ffg_update import (
    FUNCTION_FAILURE_BLOCK_THRESHOLD,
    _close_crashed_cross_test_scope,
    _endpoint_function_state,
    _record_completed_intra_scenario,
    _update_function_execution_progress,
)
from workflow.nodes.scenario_plan import (
    PLAN_FUNCTION_COVERAGE_TARGET,
    PLAN_MAX_SCENARIOS_PER_FUNCTION,
    _function_audit_signature,
    _node_uncovered_control_hints,
    _normalize_and_reopen_undercovered_functions,
    _pending_cross_pairs,
    _priority_unfinished_nodes,
    _select_intra_target,
    create_scenario_plan_node,
)


@pytest.fixture(autouse=True)
def _legacy_text_matching_enabled(monkeypatch):
    """本文件用例写于操作-控件显式绑定之前，依赖文本匹配计入覆盖。

    显式绑定是覆盖率的权威来源且文本兜底默认关闭
    （COVERAGE_LEGACY_TEXT_MATCHING），此处显式开启以保持用例原语义。
    """
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


# ── P0: 状态拆分 ────────────────────────────────────────────────

def test_function_node_accepts_terminal_statuses():
    for status in ("blocked", "deferred", "unavailable"):
        node = FunctionNode(id="func_1", test_status=status)
        assert node.test_status == status


def test_flow_graph_set_status_accepts_terminal_statuses():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1"))
    for status in ("blocked", "deferred", "unavailable"):
        graph.set_node_test_status("func_1", status)
        assert graph.get_node_test_status("func_1") == status


def test_iter_unfinished_nodes_excludes_terminal_statuses():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", test_status="in_progress"))
    graph.add_node(FunctionNode(id="func_2", test_status="blocked"))
    graph.add_node(FunctionNode(id="func_3", test_status="deferred"))
    graph.add_node(FunctionNode(id="func_4", test_status="unavailable"))
    graph.add_node(FunctionNode(id="func_5", test_status="completed"))

    assert [node.id for node in graph.iter_unfinished_nodes()] == ["func_1"]


def test_generic_repeated_failure_yields_but_stays_in_progress():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", test_status="in_progress"))
    state, execution_result = _intra_state(
        "func_1",
        {"scenario_status": "stalled", "scenario_error": "检测到连续 3 次重复动作: click"},
        failure_counts={"func_1": FUNCTION_FAILURE_BLOCK_THRESHOLD - 1},
    )

    _update_function_execution_progress(graph, state, execution_result, "func_1", "in_progress")

    assert graph.get_node_test_status("func_1") == "in_progress"


def test_verification_block_marks_function_unavailable_immediately():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", test_status="in_progress"))
    state, execution_result = _intra_state(
        "func_1",
        {
            "scenario_status": "blocked_by_verification",
            "scenario_error": "滑块验证无法由无人值守测试完成",
        },
    )

    failure_counts, current_id, current_status = _update_function_execution_progress(
        graph,
        state,
        execution_result,
        "func_1",
        "in_progress",
    )

    assert graph.get_node_test_status("func_1") == "unavailable"
    assert failure_counts == {}
    assert current_id == ""
    assert current_status == "not_started"


def test_navigation_verification_also_marks_target_unavailable():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", test_status="in_progress"))
    state, execution_result = _intra_state(
        "func_1",
        {
            "navigation_status": "blocked_by_verification",
            "navigation_error": "图形验证码无法由无人值守测试完成",
            "scenario_status": "skipped_due_navigation_failure",
        },
    )

    _counts, current_id, current_status = _update_function_execution_progress(
        graph,
        state,
        execution_result,
        "func_1",
        "in_progress",
    )

    assert graph.get_node_test_status("func_1") == "unavailable"
    assert current_id == ""
    assert current_status == "not_started"


def test_single_page_loop_yields_without_marking_function_unavailable():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", test_status="in_progress"))
    state, execution_result = _intra_state(
        "func_1",
        {
            "scenario_status": "stalled",
            "scenario_error": "相同点击意图重试一次后页面仍无明显变化",
        },
    )

    _counts, current_id, _status = _update_function_execution_progress(
        graph,
        state,
        execution_result,
        "func_1",
        "in_progress",
    )

    assert graph.get_node_test_status("func_1") == "in_progress"
    assert current_id == ""


def test_confirmed_unchanged_page_stall_marks_function_unavailable():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", test_status="in_progress"))
    state, execution_result = _intra_state(
        "func_1",
        {
            "scenario_status": "stalled",
            "scenario_error": "连续点击目标控件后页面持续无明显变化，确认界面卡死无法继续",
        },
    )

    _counts, current_id, _status = _update_function_execution_progress(
        graph,
        state,
        execution_result,
        "func_1",
        "in_progress",
    )

    assert graph.get_node_test_status("func_1") == "unavailable"
    assert current_id == ""


def test_crash_yields_function_without_marking_unavailable():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", test_status="in_progress"))
    state, execution_result = _intra_state(
        "func_1",
        {"scenario_status": "target_app_crashed", "crash_detected": True},
    )

    _update_function_execution_progress(graph, state, execution_result, "func_1", "in_progress")

    assert graph.get_node_test_status("func_1") == "in_progress"


def test_crashed_cross_scope_yields_without_marking_unavailable():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", test_status="in_progress"))
    graph.add_node(FunctionNode(id="func_2", test_status="in_progress"))
    state = {
        "current_plan": {"scenario_type": "cross_test", "src_function_id": "func_1", "dst_function_id": "func_2"},
        "current_test_scenario": "跨功能崩溃场景",
    }
    execution_result = {"scenario_status": "target_app_crashed", "crash_detected": True}

    _close_crashed_cross_test_scope(
        graph, state, execution_result, {}, "func_1", "in_progress"
    )

    assert graph.get_node_test_status("func_1") == "in_progress"


def test_screenshot_unavailable_does_not_downgrade_terminal_status():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", test_status="blocked"))
    state, execution_result = _intra_state(
        "func_1",
        {"scenario_status": "screenshot_unavailable"},
    )

    _update_function_execution_progress(graph, state, execution_result, "func_1", "blocked")

    # 已 block 的功能不得被黑屏降级回 in_progress。
    assert graph.get_node_test_status("func_1") == "blocked"


# ── P1a: 失败让出与优先级 ──────────────────────────────────────

def _two_reachable_functions() -> tuple[FlowGraph, Page]:
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="功能一", test_status="in_progress"))
    graph.add_node(FunctionNode(id="func_2", name="功能二", test_status="not_started"))
    page_1 = Page(id="page_1", name="功能一入口")
    page_2 = Page(id="page_2", name="功能二入口")
    graph.add_page(page_1, "func_1")
    graph.add_page(page_2, "func_2")
    graph.add_page_transition(PageTransition("page_1", "page_2", "点击功能二入口"))
    return graph, page_1


def test_priority_unfinished_nodes_puts_failed_last():
    graph, _ = _two_reachable_functions()
    graph.get_node("func_2").test_status = "in_progress"

    order = [n.id for n in _priority_unfinished_nodes(graph, {"func_1": 1})]

    assert order == ["func_2", "func_1"]


def test_select_intra_target_keeps_actual_current_function_after_recent_failure():
    graph, current_page = _two_reachable_functions()

    target = _select_intra_target(graph, current_page, "func_1", {"func_1": 1})

    assert target.id == "func_1"


def test_select_intra_target_keeps_actual_current_function_before_unstarted_function():
    graph, current_page = _two_reachable_functions()

    target = _select_intra_target(graph, current_page, "func_1", {})

    assert target.id == "func_1"


def test_select_intra_target_prioritizes_unstarted_function_sharing_current_entry_page():
    graph = FlowGraph()
    graph.add_node(FunctionNode(
        id="func_1",
        name="会话消息浏览与收发",
        test_status="in_progress",
    ))
    graph.add_node(FunctionNode(
        id="func_2",
        name="从相册选择并发送媒体",
        test_status="not_started",
    ))
    entry_page = Page(id="page_1", name="会话聊天页面")
    graph.add_page(entry_page, "func_1")
    graph.add_page(entry_page, "func_2")

    target = _select_intra_target(graph, entry_page, "func_1", {})

    assert target.id == "func_2"


# ── P1a: 单功能场景预算 ────────────────────────────────────────

def test_scenario_budget_does_not_complete_when_program_coverage_is_low():
    graph = FlowGraph()
    node = FunctionNode(id="func_1", name="庞大功能", test_status="in_progress")
    graph.add_node(node)
    graph.add_page(Page(id="page_1", name="入口", controls=["搜索按钮"]), "func_1")
    for index in range(PLAN_MAX_SCENARIOS_PER_FUNCTION):
        node.add_scenario(f"场景 {index}")

    class Llm:
        def invoke(self, _messages):
            return type("Response", (), {"content": (
                '{"scenario_type":"intra_test","target_function":"func_1",'
                '"src_function":"","dst_function":"","navigation_scenario":"",'
                '"test_scenario":"点击搜索按钮 → 观察搜索页面后结束"}'
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

    assert result["current_plan"]["scenario_type"] == "intra_test"
    assert graph.get_node_test_status("func_1") == "in_progress"


def test_coverage_target_forces_completion_without_llm():
    graph = FlowGraph()
    node = FunctionNode(
        id="func_1",
        name="已充分覆盖功能",
        test_status="in_progress",
    )
    node.add_scenario("完成当前页面的代表性业务闭环")
    graph.add_node(node)
    graph.add_page(Page(
        id="page_1",
        name="当前页面",
        controls=["搜索按钮", "内容卡片"],
        operations=["点击搜索按钮", "点击内容卡片"],
    ), "func_1")
    graph.recompute_all_coverages()

    class Llm:
        def invoke(self, _messages):
            raise AssertionError("达到覆盖率阈值后不得再调用规划 LLM")

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
    graph = FlowGraph()
    node = FunctionNode(
        id="func_1",
        name="服务号入口",
        test_status="completed",
        coverage_percent=100,
        scenarios=["点击服务号条目并进入页面"],
    )
    graph.add_node(node)
    graph.add_page(Page(id="page_1", name="聊天列表"), "func_1")
    graph.add_page(Page(id="page_2", name="服务号页面"), "func_1")

    reopened = _normalize_and_reopen_undercovered_functions(graph)

    assert reopened == ["func_1"]
    assert node.name == "服务号"
    assert node.test_status == "in_progress"
    assert node.coverage_percent == 0


def test_multi_page_coverage_needs_representative_scenario_per_page():
    graph = FlowGraph()
    node = FunctionNode(
        id="func_1",
        name="内容浏览",
        test_status="in_progress",
        coverage_percent=PLAN_FUNCTION_COVERAGE_TARGET,
        scenarios=["进入内容页面"],
    )
    graph.add_node(node)
    graph.add_page(Page(id="page_1", name="入口页面"), "func_1")
    graph.add_page(Page(id="page_2", name="内容页面", controls=["搜索按钮"]), "func_1")

    class Llm:
        calls = 0

        def invoke(self, _messages):
            self.calls += 1
            return type("Response", (), {"content": (
                '{"scenario_type":"intra_test","target_function":"func_1",'
                '"src_function":"","dst_function":"","navigation_scenario":"",'
                '"test_scenario":"点击搜索按钮 → 看到搜索页面后结束"}'
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
    assert result["current_test_scenario"].startswith("点击搜索按钮")


def test_repeated_entry_observation_does_not_count_as_new_scenario():
    graph = FlowGraph()
    node = FunctionNode(
        id="func_1",
        name="服务号",
        test_status="completed",
        coverage_percent=79,
        scenarios=[
            "点击服务号入口条目 → 进入服务号页面后结束",
            "点击服务号入口条目 → 等待内容加载 → 看到内容卡片后结束",
        ],
    )
    graph.add_node(node)
    graph.add_page(
        Page(id="page_1", name="聊天列表", operations=["点击服务号入口条目"]),
        "func_1",
    )
    graph.add_page(
        Page(
            id="page_2",
            name="服务号页面",
            controls=["顶部返回按钮", "页面标题“服务号”", "搜索按钮", "关注账号内容卡片"],
            operations=["点击页面左上角的返回箭头"],
        ),
        "func_1",
    )

    assert _node_uncovered_control_hints(graph, node) == [
        "页面标题“服务号”",
        "搜索按钮",
        "关注账号内容卡片",
    ]

    reopened = _normalize_and_reopen_undercovered_functions(graph)

    assert reopened == ["func_1"]
    assert node.test_status == "in_progress"


def test_execution_endpoint_becomes_next_current_function():
    graph, _ = _two_reachable_functions()
    function_id, status = _endpoint_function_state(
        graph,
        graph.get_page("page_2"),
        {"scenario_type": "intra_test", "target_function_id": "func_1"},
        "func_1",
        ["func_2"],
    )

    assert function_id == "func_2"
    assert status == "not_started"


def test_completed_scenario_is_recorded_on_planned_target_function():
    """P3 回归：场景学分归属规划目标功能，执行终点功能只作兜底。

    旧行为（按终点功能归属）会系统性偷走目标功能的自身场景计数：
    终点落在共享页面时，共享该页面的旧功能拿走学分，目标功能
    自身场景恒为 0，覆盖率达标也无法收尾。
    """
    graph, _ = _two_reachable_functions()

    owner_id = _record_completed_intra_scenario(
        graph,
        {"scenario_type": "intra_test", "target_function_id": "func_1"},
        "点击入口 → 完成目标能力后结束",
        "func_2",
    )

    assert owner_id == "func_1"
    assert graph.get_node("func_1").scenarios == ["点击入口 → 完成目标能力后结束"]
    assert graph.get_node("func_2").scenarios == []


def test_completed_scenario_falls_back_to_endpoint_when_target_unresolvable():
    graph, _ = _two_reachable_functions()

    owner_id = _record_completed_intra_scenario(
        graph,
        {"scenario_type": "intra_test", "target_function_id": ""},
        "点击入口 → 完成目标能力后结束",
        "func_2",
    )

    assert owner_id == "func_2"
    assert graph.get_node("func_2").scenarios == ["点击入口 → 完成目标能力后结束"]
    assert graph.get_node_test_status("func_2") == "in_progress"


def test_shared_endpoint_prefers_single_newly_touched_function_owner():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="聊天列表", test_status="in_progress"))
    graph.add_node(FunctionNode(id="func_2", name="消息草稿编辑"))
    shared_page = Page(id="page_1", name="聊天页面")
    graph.add_page(shared_page, "func_1")
    graph.add_page(shared_page, "func_2")

    function_id, status = _endpoint_function_state(
        graph,
        shared_page,
        {"scenario_type": "intra_test", "target_function_id": "func_1"},
        "func_1",
        ["func_1", "func_2"],
    )

    assert function_id == "func_2"
    assert status == "not_started"


def test_cross_execution_prefers_destination_owner_on_shared_page():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="源功能", test_status="in_progress"))
    graph.add_node(FunctionNode(id="func_2", name="目标功能", test_status="in_progress"))
    shared = Page(id="page_shared", name="共享结果页")
    graph.add_page(shared, "func_1")
    graph.add_page(shared, "func_2")

    function_id, _ = _endpoint_function_state(
        graph,
        shared,
        {
            "scenario_type": "cross_test",
            "src_function_id": "func_1",
            "dst_function_id": "func_2",
        },
        "func_1",
    )

    assert function_id == "func_2"


# ── P2: 审计签名去冗余 ─────────────────────────────────────────

def test_blocked_function_is_excluded_from_pending_cross_pairs():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="源功能", test_status="blocked"))
    graph.add_node(FunctionNode(id="func_2", name="目标功能", test_status="completed"))
    graph.add_function_dependency(FunctionDependency("func_1", "func_2", "任务记录"))

    assert _pending_cross_pairs(graph) == []


def test_audit_signature_ignores_scenario_count_but_tracks_structure():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", test_status="completed"))
    graph.add_page(
        Page(id="page_1", name="入口", controls=["按钮"], operations=["点击按钮"]),
        "func_1",
    )
    before = _function_audit_signature(graph, graph.get_node("func_1"))

    graph.add_scenario("func_1", "新场景")

    assert _function_audit_signature(graph, graph.get_node("func_1")) == before

    graph.get_page("page_1").add_operation("点击新入口")

    assert _function_audit_signature(graph, graph.get_node("func_1")) != before


# ── H2 回归：终端状态不得被覆盖审计轮的执行失败翻转 ──────────────

def test_verification_block_keeps_completed_status():
    """覆盖审计轮遇到验证阻塞不得把 completed 功能改写为 unavailable。"""
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", test_status="completed"))
    state, execution_result = _intra_state(
        "func_1",
        {
            "scenario_status": "blocked_by_verification",
            "scenario_error": "滑块验证无法由无人值守测试完成",
        },
    )

    failure_counts, current_id, current_status = _update_function_execution_progress(
        graph, state, execution_result, "func_1", "completed",
    )

    assert graph.get_node_test_status("func_1") == "completed"
    assert failure_counts == {}
    assert current_id == ""
    assert current_status == "not_started"


def test_crash_keeps_completed_status():
    """覆盖审计轮目标 App 崩溃重启不得把 completed 功能改写为 in_progress。"""
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", test_status="completed"))
    state, execution_result = _intra_state(
        "func_1",
        {"scenario_status": "target_app_crashed", "crash_detected": True},
    )

    _update_function_execution_progress(
        graph, state, execution_result, "func_1", "completed",
    )

    assert graph.get_node_test_status("func_1") == "completed"


def test_confirmed_stall_keeps_completed_status():
    """覆盖审计轮确认页面卡死不得把 completed 功能改写为 unavailable。"""
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", test_status="completed"))
    state, execution_result = _intra_state(
        "func_1",
        {
            "scenario_status": "stalled",
            "scenario_error": "连续点击目标控件后页面持续无明显变化，确认界面卡死无法继续",
        },
    )

    _update_function_execution_progress(
        graph, state, execution_result, "func_1", "completed",
    )

    assert graph.get_node_test_status("func_1") == "completed"


def test_unconfirmed_stall_keeps_completed_status():
    """覆盖审计轮普通停滞不得把 completed 功能改写为 in_progress。"""
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", test_status="completed"))
    state, execution_result = _intra_state(
        "func_1",
        {
            "scenario_status": "stalled",
            "scenario_error": "相同点击意图重试一次后页面仍无明显变化",
        },
    )

    _update_function_execution_progress(
        graph, state, execution_result, "func_1", "completed",
    )

    assert graph.get_node_test_status("func_1") == "completed"


# ── H1 回归：截图持续失败的重启必须有限次 ────────────────────────

def test_screenshot_failure_restart_loop_is_bounded(monkeypatch):
    """截图持续失败且 restart_app 无法恢复时，场景必须终止而非无限重启。

    历史缺陷：截图失败不增长 trace，步数上限永远不可达，设备持续故障时
    “失败 3 次-重启-再失败”会无限占用运行预算。
    """
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
    monkeypatch.setattr(
        scenario_exec,
        "_recover_target_app_at_scene_end",
        lambda *_a, **_k: (False, "", ""),
    )
    monkeypatch.setattr(scenario_exec, "_is_target_app_foreground", lambda *_a: True)
    monkeypatch.setattr(scenario_exec.time, "sleep", lambda _seconds: None)

    device = Device()
    trace, _page, status, error_message, _crash, _ctx = scenario_exec._run_single_scenario(
        Llm(),
        device,
        "示例应用",
        "com.example",
        "点击登录按钮",
        Page(id="page_1", name="主页"),
        clear_logcat=False,
        trace_phase="test",
    )

    assert status == "error"
    assert "截图失败" in error_message
    assert device.restarts == scenario_exec.APP_RESTART_ATTEMPTS
    assert trace == []
