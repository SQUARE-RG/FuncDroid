import time

import pytest

from model.flow_graph import (
    FlowGraph,
    FunctionDependency,
    FunctionNode,
    Page,
    PageTransition,
)
from workflow.nodes import scenario_plan as scenario_plan_module
from workflow.graph import _fallback_scenario_exec, _route_after_plan
from workflow.nodes.ffg_update import _resolve_plan_target_function_id
from workflow.nodes.scenario_plan import (
    _audit_exploration_response,
    _audit_function_for_gap,
    _build_audit_navigation_scenario,
    _forced_intra_scenario,
    _force_progress_after_planner_noop,
    _is_similar_scenario,
    _candidate_cross_pairs,
    _pending_cross_pairs,
    _planner_status_update_response,
    _navigation_to_function,
    _run_completion_audit,
    _validate_audit_result,
    _validate_plan_result,
    create_scenario_plan_node,
    _bootstrap_initial_ffg,
)
from workflow.nodes.prompt import (
    BOOTSTRAP_FFG_SYSTEM_PROMPT,
    FFG_AUDIT_SYSTEM_PROMPT,
    FFG_UPDATE_SYSTEM_PROMPT,
    SCENARIO_EXEC_SYSTEM_PROMPT_TEMPLATE,
    UNIFIED_SCENARIO_PLAN_SYSTEM_PROMPT,
)


@pytest.fixture(autouse=True)
def _legacy_text_matching_enabled(monkeypatch):
    """本文件用例写于操作-控件显式绑定之前，依赖文本匹配计入覆盖。

    显式绑定是覆盖率的权威来源且文本兜底默认关闭
    （COVERAGE_LEGACY_TEXT_MATCHING），此处显式开启以保持用例原语义。
    """
    from model import flow_graph as _flow_graph

    monkeypatch.setattr(_flow_graph, "COVERAGE_LEGACY_TEXT_MATCHING", True)


def _graph_with_two_functions() -> FlowGraph:
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="源功能"))
    graph.add_node(FunctionNode(id="func_2", name="目标功能"))
    return graph


def test_navigation_reaches_function_page_instead_of_stopping_at_entry_page():
    graph = _graph_with_two_functions()
    root = Page(id="page_1", name="微信首页")
    detail = Page(id="page_2", name="文件传输助手详情页")
    graph.add_page(root, "func_1")
    graph.add_entry_page(root, "func_2")
    graph.add_page(detail, "func_2")
    graph.add_page_transition(PageTransition("page_1", "page_2", "点击文件传输助手"))

    scenario, _reason, reachable = _navigation_to_function(
        graph,
        root,
        graph.get_node("func_2"),
    )

    assert reachable is True
    assert "点击文件传输助手" in scenario
    assert "确认到达“文件传输助手详情页”" in scenario


def test_completion_audit_batches_cover_every_page_image(monkeypatch):
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="多页面功能"))
    for index in range(5):
        graph.add_page(Page(
            id=f"page_{index + 1}",
            name=f"页面{index + 1}",
            representative_screenshot=f"image-{index + 1}",
        ), "func_1")

    class Response:
        content = (
            '{"has_exploration_gap":false,"entry_page_ref":"",'
            '"entry_name":"","test_scenario":""}'
        )

    class Llm:
        def __init__(self):
            self.image_counts = []

        def invoke(self, messages):
            content = messages[1].content
            self.image_counts.append(sum(
                1 for block in content if block.get("type") == "image_url"
            ))
            return Response()

    monkeypatch.setattr(scenario_plan_module, "MAX_AUDIT_IMAGES_PER_FUNCTION", 2)
    llm = Llm()

    result = _audit_function_for_gap(
        llm,
        graph,
        graph.get_node("func_1"),
        "core",
    )

    assert result["has_exploration_gap"] is False
    assert llm.image_counts == [2, 2, 1]


def test_time_limit_blocks_new_normal_scenario_execution():
    state = {
        "workflow_complete": False,
        "iteration_count": 1,
        "max_iterations": 0,
        "start_time": time.time() - 10,
        "total_time_limit": 1,
        "current_plan": {"scenario_type": "intra_test"},
    }

    assert _route_after_plan(state) == "end"


def test_planner_self_loop_stops_at_max_iterations():
    state = {
        "iteration_count": 3,
        "max_iterations": 3,
        "workflow_complete": False,
        "current_plan": {"scenario_type": "status_update"},
    }

    assert _route_after_plan(state) == "end"


def test_executor_prompt_explains_scroll_direction_semantics():
    assert "查看页面下方/更靠后的内容用 up" in SCENARIO_EXEC_SYSTEM_PROMPT_TEMPLATE
    assert "返回上方/更靠前的内容用 down" in SCENARIO_EXEC_SYSTEM_PROMPT_TEMPLATE


def test_prompts_do_not_claim_unsupported_physical_inputs():
    assert "不能提供真人语音" in UNIFIED_SCENARIO_PLAN_SYSTEM_PROMPT
    assert "不规划识别、拖动、提交或验证成功等后续场景" in UNIFIED_SCENARIO_PLAN_SYSTEM_PROMPT
    assert "人机验证入口最多探索到准备页/挑战出现" in FFG_AUDIT_SYSTEM_PROMPT
    assert "你不能发出真人语音" in SCENARIO_EXEC_SYSTEM_PROMPT_TEMPLATE
    assert "不得尝试图形验证码、滑块/拼图或其他人机验证" in UNIFIED_SCENARIO_PLAN_SYSTEM_PROMPT
    assert "挑战出现后必须结束" in UNIFIED_SCENARIO_PLAN_SYSTEM_PROMPT
    assert "不得要求自动化识别、拖动、提交或完成验证" in FFG_AUDIT_SYSTEM_PROMPT
    assert "看到验证挑战后先执行一次 press_back 退出" in SCENARIO_EXEC_SYSTEM_PROMPT_TEMPLATE
    assert "验证阻塞，已退出" in SCENARIO_EXEC_SYSTEM_PROMPT_TEMPLATE


def test_prompts_require_compact_exploration_directions():
    assert "范围明确的探索方向" in UNIFIED_SCENARIO_PLAN_SYSTEM_PROMPT
    assert "开放式探索（开放结尾）" in UNIFIED_SCENARIO_PLAN_SYSTEM_PROMPT
    assert "批量覆盖" in UNIFIED_SCENARIO_PLAN_SYSTEM_PROMPT
    assert "不要把未知下游写成完整动作脚本" in UNIFIED_SCENARIO_PLAN_SYSTEM_PROMPT
    assert "不要复述任务、历史动作、页面控件清单" in SCENARIO_EXEC_SYSTEM_PROMPT_TEMPLATE
    assert "建议不超过 40 个汉字" in SCENARIO_EXEC_SYSTEM_PROMPT_TEMPLATE
    assert "探索方向而不是固定动作脚本" in SCENARIO_EXEC_SYSTEM_PROMPT_TEMPLATE
    assert "中间结果页不是默认终点" in SCENARIO_EXEC_SYSTEM_PROMPT_TEMPLATE
    assert "没有新的有价值方向时" in UNIFIED_SCENARIO_PLAN_SYSTEM_PROMPT
    assert "两个场景字段都留空" in UNIFIED_SCENARIO_PLAN_SYSTEM_PROMPT
    assert "仅再次进入、等待加载、改写文本或重复同一路径不算新场景" in UNIFIED_SCENARIO_PLAN_SYSTEM_PROMPT


def test_ffg_prompts_treat_visible_entries_as_business_functions():
    assert "不使用“xx入口”这类控件位置名称" in BOOTSTRAP_FFG_SYSTEM_PROMPT
    assert "不能继续保留“xx入口”" in FFG_UPDATE_SYSTEM_PROMPT
    assert "entry_function_refs" in BOOTSTRAP_FFG_SYSTEM_PROMPT
    assert "entry_function_refs" in FFG_UPDATE_SYSTEM_PROMPT
    assert "成功和失败操作由程序从真实 trace 记录" in FFG_UPDATE_SYSTEM_PROMPT
    assert "不得把下游能力并入原列表 Function" in FFG_UPDATE_SYSTEM_PROMPT
    assert "仅到达页面也足以确认下游能力存在" in FFG_UPDATE_SYSTEM_PROMPT
    assert "coverage_percent" not in FFG_UPDATE_SYSTEM_PROMPT
    assert "coverage_percent" not in BOOTSTRAP_FFG_SYSTEM_PROMPT


def test_ffg_prompt_requires_splitting_sub_functions_of_large_functions():
    """方案 1+2：大功能下的子功能必须拆分为单独功能节点，优先部分拆分。"""
    assert "大功能下的子功能都应拆分为单独的功能节点" in FFG_UPDATE_SYSTEM_PROMPT
    assert "共享同一入口（如设置、我的、管理页）" in FFG_UPDATE_SYSTEM_PROMPT
    assert "必须拆分" in FFG_UPDATE_SYSTEM_PROMPT
    assert "retire_source=false 的部分拆分" in FFG_UPDATE_SYSTEM_PROMPT
    assert "不必一次性全拆" in FFG_UPDATE_SYSTEM_PROMPT


def test_prompt_json_examples_do_not_use_literal_union_placeholders():
    assert '"has_exploration_gap": true/false' not in FFG_AUDIT_SYSTEM_PROMPT
    assert "target_function_status" not in UNIFIED_SCENARIO_PLAN_SYSTEM_PROMPT
    assert "scenario_goal" not in UNIFIED_SCENARIO_PLAN_SYSTEM_PROMPT
    assert "不得输出竖线枚举" in UNIFIED_SCENARIO_PLAN_SYSTEM_PROMPT
    assert "场景结束缺陷复核" not in SCENARIO_EXEC_SYSTEM_PROMPT_TEMPLATE


def test_audit_protocol_rejects_empty_or_ambiguous_results():
    assert _validate_audit_result({}) == ["empty_or_non_object_result"]
    assert "has_exploration_gap_must_be_boolean" in _validate_audit_result({
        "has_exploration_gap": "false",
        "entry_page_ref": "",
        "entry_name": "",
        "test_scenario": "",
    })
    assert "no_gap_requires_empty_detail_fields" in _validate_audit_result({
        "has_exploration_gap": False,
        "entry_page_ref": "page_1",
        "entry_name": "设置",
        "test_scenario": "",
    })


def test_invalid_audit_result_cannot_mark_completion(monkeypatch):
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="设置", test_status="completed"))
    graph.add_page(Page(id="page_1", name="设置页"), "func_1")
    monkeypatch.setattr(
        scenario_plan_module,
        "_audit_function_for_gap",
        lambda *_args, **_kwargs: {},
    )

    audit_result, audit_complete, signatures = _run_completion_audit(
        object(),
        graph,
        "core",
    )

    assert audit_result is None
    assert audit_complete is False
    assert signatures == {}


def test_ffg_prompt_can_update_existing_function_resource_variables():
    assert "新增资源时写入 functions，并沿用原 func_x" in FFG_UPDATE_SYSTEM_PROMPT
    assert "空数组不删除已有内容" in FFG_UPDATE_SYSTEM_PROMPT
    assert '"function_ref": "new_func_1"' in FFG_UPDATE_SYSTEM_PROMPT
    assert '"function_dependencies": [{' in FFG_UPDATE_SYSTEM_PROMPT


def test_ffg_prompt_separates_planning_scope_from_function_ownership():
    assert "scenario_type 只表示调度范围" in FFG_UPDATE_SYSTEM_PROMPT
    assert "禁止默认继承 current_plan.target_function" in FFG_UPDATE_SYSTEM_PROMPT
    assert '"function_refs": ["new_func_1"]' in FFG_UPDATE_SYSTEM_PROMPT
    assert '"page_ownership": [{' in FFG_UPDATE_SYSTEM_PROMPT
    assert "可见业务入口可直接创建 Function" in FFG_UPDATE_SYSTEM_PROMPT
    assert "仅提供导航入口用 add_entry_function_refs" in FFG_UPDATE_SYSTEM_PROMPT
    assert "实际承载行为用 add_function_refs" in FFG_UPDATE_SYSTEM_PROMPT
    assert '"test_status": "unavailable"' not in FFG_UPDATE_SYSTEM_PROMPT
    assert "不输出 test_status" in FFG_UPDATE_SYSTEM_PROMPT


def test_ffg_prompt_leaves_trace_edges_to_deterministic_code():
    assert '"operations": [' not in FFG_UPDATE_SYSTEM_PROMPT
    assert '"page_transitions": [' not in FFG_UPDATE_SYSTEM_PROMPT


def test_bootstrap_does_not_rewrite_function_by_package_specific_rule():
    graph = FlowGraph()
    current_page = Page(
        name="微信会话列表",
        description="显示当前会话条目",
    )

    class BootstrapLlm:
        class Response:
            content = (
                '{"functions":[{"function_ref":"new_func_1","source_step":1,'
                '"name":"微信会话","description":"微信中的所有会话能力",'
                '"entry_page_ref":"new_page_1"}],"pages":[{'
                '"page_ref":"new_page_1","function_refs":["new_func_1"],'
                '"source_step":1,"name":"微信会话列表",'
                '"description":"显示当前会话条目","controls":[]}]}'
            )

        def invoke(self, _messages):
            return self.Response()

    _bootstrap_initial_ffg(
        BootstrapLlm(),
        graph,
        current_page,
        app_name="微信",
        package_name="com.tencent.mm",
    )

    node = graph.get_node("func_1")
    assert node.name == "微信会话"
    assert node.description == "微信中的所有会话能力"


def test_bootstrap_visible_business_entries_share_the_observed_page():
    graph = FlowGraph()
    current_page = Page(
        name="会话聊天页面",
        description="显示消息和扩展功能面板",
    )

    class BootstrapLlm:
        class Response:
            content = (
                '{"functions":['
                '{"function_ref":"new_func_1","source_step":1,'
                '"name":"会话消息浏览与收发","description":"浏览和发送消息","coverage_percent":0},'
                '{"function_ref":"new_func_2","source_step":1,'
                '"name":"从相册选择并发送媒体","description":"由相册入口发现，流程待确认","coverage_percent":0},'
                '{"function_ref":"new_func_3","source_step":1,'
                '"name":"选择并发送文件","description":"由文件入口发现，流程待确认","coverage_percent":0}'
                '],"pages":[{'
                '"page_ref":"new_page_1",'
                '"function_refs":["new_func_1","new_func_2","new_func_3"],'
                '"source_step":1,"name":"会话聊天页面",'
                '"description":"显示消息和扩展功能面板",'
                '"controls":["相册入口","文件入口"]}]}'
            )

        def invoke(self, _messages):
            return self.Response()

    _bootstrap_initial_ffg(
        BootstrapLlm(),
        graph,
        current_page,
        app_name="测试应用",
        package_name="com.example.app",
    )

    assert list(graph.functions) == ["func_1", "func_2", "func_3"]
    assert graph.get_page_owners("page_1") == ["func_1", "func_2", "func_3"]
    assert all(graph.get_node(function_id).page_ids == ["page_1"] for function_id in graph.functions)
    assert graph.get_node("func_1").entry_page_ids == []
    assert graph.get_node("func_2").entry_page_ids == ["page_1"]
    assert graph.get_node("func_3").entry_page_ids == ["page_1"]
    assert graph.get_node("func_2").coverage_percent == 0
    assert graph.get_node("func_3").test_status == "not_started"


def test_last_executable_plan_is_not_skipped_by_planner_limit():
    state = {
        "iteration_count": 3,
        "max_iterations": 3,
        "workflow_complete": False,
        "current_plan": {"scenario_type": "intra_test"},
    }

    assert _route_after_plan(state) == "scenario_exec"


def test_cross_candidates_ignore_navigation_transition_and_prefer_dependency():
    graph = _graph_with_two_functions()
    graph.add_function_dependency(FunctionDependency("func_1", "func_2", "已创建记录"))

    candidates = _candidate_cross_pairs(graph)

    assert candidates == [("func_1", "func_2", "功能依赖: 已创建记录")]


def test_cross_candidates_fall_back_to_shared_resource_variables():
    graph = _graph_with_two_functions()
    graph.get_node("func_1").resource_variables = ["任务记录"]
    graph.get_node("func_2").resource_variables = ["任务记录"]

    candidates = _candidate_cross_pairs(graph)

    assert candidates == [("func_1", "func_2", "共享资源变量: 任务记录")]


def test_navigation_transition_alone_does_not_create_cross_test_candidate():
    graph = _graph_with_two_functions()

    assert _candidate_cross_pairs(graph) == []


def test_shared_resource_fallback_uses_stable_function_order():
    graph = _graph_with_two_functions()
    graph.get_node("func_1").resource_variables = ["任务记录"]
    graph.get_node("func_2").resource_variables = ["任务记录"]
    candidates = _candidate_cross_pairs(graph)

    assert candidates == [(
        "func_1",
        "func_2",
        "共享资源变量: 任务记录",
    )]


def test_cross_candidates_dedupe_reverse_directions():
    graph = _graph_with_two_functions()
    graph.add_function_dependency(FunctionDependency("func_1", "func_2", "已创建记录"))

    candidates = _candidate_cross_pairs(graph)

    assert candidates == [("func_1", "func_2", "功能依赖: 已创建记录")]


def test_cross_pair_tested_in_either_direction_blocks_reverse():
    from workflow.nodes.scenario_plan import _tested_cross_pair_keys

    graph = _graph_with_two_functions()
    graph.add_function_dependency(FunctionDependency("func_1", "func_2", "详情记录"))
    graph.add_cross_scenario("func_2", "func_1", "已验证创建到列表")
    graph.get_node("func_1").test_status = "completed"
    graph.get_node("func_2").test_status = "completed"

    assert _pending_cross_pairs(graph) == []
    tested_keys = _tested_cross_pair_keys(graph)
    assert frozenset(("func_1", "func_2")) in tested_keys
    assert frozenset(("func_2", "func_1")) in tested_keys


def test_completion_audit_returns_execution_candidate_without_mutating_ffg():
    graph = FlowGraph()
    node = FunctionNode(id="func_1", name="主导航", test_status="completed")
    graph.add_node(node)
    graph.add_page(Page(
        id="page_1",
        name="首页",
        description="包含已记录的导入入口",
        representative_screenshot="stored-screenshot",
    ), "func_1")

    class AuditLlm:
        class Response:
            content = (
                '{"has_exploration_gap":true,"entry_page_ref":"page_1",'
                '"test_scenario":"进入首页，点击已记录的导入入口，并观察记录实际出现的页面"}'
            )

        def invoke(self, _messages):
            return self.Response()

    before_ids = list(graph.functions)
    result, audit_complete, _signatures = _run_completion_audit(
        AuditLlm(), graph, "core"
    )

    assert audit_complete is False
    assert result["owner_function_id"] == "func_1"
    assert result["entry_page_id"] == "page_1"
    assert list(graph.functions) == before_ids
    assert "functions" not in result
    assert "只返回一次入口探索" in FFG_AUDIT_SYSTEM_PROMPT


def test_completion_audit_retries_a_non_executable_gap_candidate():
    graph = FlowGraph()
    node = FunctionNode(id="func_1", name="主导航", test_status="completed")
    graph.add_node(node)
    graph.add_page(Page(
        id="page_1",
        name="首页",
        description="包含尚未探索的导入入口",
    ), "func_1")

    class AuditLlm:
        def __init__(self):
            self.call_count = 0

        def invoke(self, _messages):
            self.call_count += 1
            content = (
                '{"has_exploration_gap":true,"entry_page_ref":"page_1",'
                '"entry_name":"导入入口","test_scenario":"首页"}'
                if self.call_count == 1
                else '{"has_exploration_gap":true,"entry_page_ref":"page_1",'
                '"entry_name":"导入入口","test_scenario":'
                '"进入首页，点击导入入口，并观察记录实际出现的页面"}'
            )
            return type("Response", (), {"content": content})()

    llm = AuditLlm()
    result, audit_complete, _signatures = _run_completion_audit(llm, graph, "core")

    assert llm.call_count == 2
    assert audit_complete is False
    assert result["test_scenario"].startswith("进入首页，点击导入入口")


def test_completion_audit_allows_wechat_conversation_entry():
    graph = FlowGraph()
    node = FunctionNode(id="func_1", name="微信首页", test_status="completed")
    graph.add_node(node)
    graph.add_page(Page(
        id="page_1",
        name="微信首页",
        description="包含文件传输助手会话",
    ), "func_1")

    class AuditLlm:
        class Response:
            content = (
                '{"has_exploration_gap":true,"entry_page_ref":"page_1",'
                '"entry_name":"文件传输助手会话","test_scenario":'
                '"进入微信首页，点击文件传输助手会话，并观察实际页面"}'
            )

        def invoke(self, _messages):
            return self.Response()

    result, audit_complete, _signatures = _run_completion_audit(
        AuditLlm(),
        graph,
        "core",
        package_name="com.tencent.mm",
    )

    assert result is not None
    assert result["entry_page_id"] == "page_1"
    assert audit_complete is False


def test_new_unfinished_function_runs_before_resuming_completed_function_audit():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="已完成入口", test_status="completed"))
    graph.add_page(Page(id="page_1", name="首页", representative_screenshot="stored"), "func_1")
    graph.add_node(FunctionNode(id="func_2", name="新发现功能", test_status="not_started"))
    graph.add_page(Page(id="page_2", name="新页面", representative_screenshot="stored"), "func_2")

    class PlanningLlm:
        calls = 0

        def invoke(self, _messages):
            self.calls += 1
            return type("Response", (), {"content": (
                '{"workflow_complete":false,"scenario_type":"intra_test",'
                '"target_function":"func_2",'
                '"src_function":"","dst_function":"",'
                '"navigation_scenario":"","test_scenario":"点击新页面中的已记录入口 → 确认结果后结束"}'
            )})()

    llm = PlanningLlm()
    planner = create_scenario_plan_node(llm, object())
    result = planner({
        "ffg": graph,
        "current_page": graph.get_page("page_2"),
        "current_plan": {
            "scenario_type": "intra_test",
            "target_function_id": "func_1",
            "audit_exploration": True,
        },
        "current_testing_function_id": "func_1",
        "current_testing_function_status": "completed",
        "run_phase": "all",
        "completion_audit_attempts": {"func_1": 1},
    })

    assert llm.calls == 1
    assert result["current_plan"]["target_function_id"] == "func_2"
    assert result["current_plan"].get("audit_exploration") is not True


def test_same_parameterized_control_family_is_a_duplicate_scenario():
    cinema = "在“活动主页”页面，点击“Cinema”活动按钮，观察当前活动和持续时间。"
    woodworking = "在“活动主页”页面，点击“Woodworking”活动按钮，观察当前活动和持续时间。"
    search = "在“活动主页”页面，点击“搜索”图标，观察搜索输入框。"

    assert _is_similar_scenario(cinema, woodworking) is True
    assert _is_similar_scenario(cinema, search) is False


def test_different_named_menu_items_are_not_duplicate_scenarios():
    diary = (
        "到达 Navigation Drawer 页面，点击 'Diary' 菜单项，"
        "观察并记录实际出现的页面或弹窗。"
    )
    settings = (
        "到达 Navigation Drawer 页面，点击 'Settings' 菜单项，"
        "观察并记录实际出现的页面或弹窗。"
    )

    assert _is_similar_scenario(diary, settings) is False
    assert _is_similar_scenario(diary, diary) is True


def test_different_unquoted_audit_entries_are_not_duplicate_scenarios():
    photo = (
        "到达 Create Note Type Selection Menu 页面，"
        "点击 Photo，并观察记录实际出现的页面或弹窗。"
    )
    checklist = (
        "到达 Create Note Type Selection Menu 页面，"
        "点击 Checklist，并观察记录实际出现的页面或弹窗。"
    )

    assert _is_similar_scenario(photo, checklist) is False
    assert _is_similar_scenario(photo, photo) is True


def test_unquoted_button_target_drives_duplicate_matching_not_result_paraphrase():
    share = "到达公众号文章详情页 → 点击分享按钮 → 确认实际出现的分享结果后结束"
    more_a = "到达公众号文章详情页 → 点击更多操作按钮 → 确认实际出现的操作结果后结束"
    more_b = "到达公众号文章详情页 → 点击更多操作按钮 → 确认实际出现的操作菜单或结果后结束"

    assert _is_similar_scenario(share, more_a) is False
    assert _is_similar_scenario(more_a, more_b) is True


def test_slash_style_chinese_audit_entries_are_not_duplicate_scenarios():
    search = (
        "进入已记录的 Notes 页面，点击/打开/选择 Search，"
        "观察并记录实际出现的页面或弹窗。"
    )
    drawer = (
        "进入已记录的 Notes 页面，点击/打开/选择 汉堡菜单按钮，"
        "观察并记录实际出现的页面或弹窗。"
    )

    assert _is_similar_scenario(search, drawer) is False
    assert _is_similar_scenario(drawer, drawer) is True


def test_different_quoted_audit_entries_are_not_duplicate_scenarios():
    photo = "到达 Create Note Options Menu 页面，点击 'Photo'，并观察记录实际出现的页面或弹窗。"
    checklist = "到达 Create Note Options Menu 页面，点击 'Checklist'，并观察记录实际出现的页面或弹窗。"
    text_option = "在创建菜单中点击“Text note”选项，观察编辑器。"
    checklist_option = "在创建菜单中点击“Checklist”选项，观察编辑器。"

    assert _is_similar_scenario(photo, checklist) is False
    assert _is_similar_scenario(text_option, checklist_option) is False
    assert _is_similar_scenario(photo, photo) is True


def test_different_visible_quoted_audit_entries_are_not_duplicate_scenarios():
    photo = "进入新增笔记类型菜单，点击真实可见的“Photo”入口，并观察实际流程。"
    checklist = "进入新增笔记类型菜单，点击真实可见的“Checklist”入口，并观察实际流程。"

    assert _is_similar_scenario(photo, checklist) is False


def test_different_named_icons_are_not_duplicate_scenarios():
    camera = "到达 Activity List 页面，点击真实可见的 camera icon，观察实际结果。"
    pencil = "到达 Activity List 页面，点击真实可见的 pencil icon，观察实际结果。"

    assert _is_similar_scenario(camera, pencil) is False
    assert _is_similar_scenario(camera, camera) is True


def test_different_chinese_named_icons_are_not_duplicate_scenarios():
    plus = "在笔记列表页，点击右下角浮动操作按钮（加号图标），观察实际结果。"
    search = "在笔记列表页，点击右上角的搜索图标，观察实际结果。"
    drawer = "在笔记列表页，点击左上角的汉堡菜单图标，观察实际结果。"

    assert _is_similar_scenario(plus, search) is False
    assert _is_similar_scenario(search, drawer) is False
    assert _is_similar_scenario(search, search) is True


def test_same_form_submission_with_different_input_values_is_duplicate():
    normal = (
        "在 'New Activity' 页面，在 'Activity Title' 输入框中输入文本 'My Activity'。"
        "点击 'checkmark' 保存图标，观察是否返回主页。"
    )
    special = (
        "在 'New Activity' 页面，在 'Activity Title' 输入框中输入文本 '!@#$%^&*'。"
        "点击 'checkmark' 保存图标，观察特殊字符活动是否保存并返回主页。"
    )
    color = (
        "在 'New Activity' 页面，在 'Activity Title' 输入框中输入文本 'Colorful'。"
        "点击 'Activity Color' 区域选择紫色，再点击 'checkmark' 保存图标。"
    )

    assert _is_similar_scenario(normal, special) is True
    assert _is_similar_scenario(normal, color) is False


def test_existing_tested_scenarios_do_not_bypass_program_limits():
    graph = FlowGraph()
    node = FunctionNode(id="func_1", name="主页功能", test_status="in_progress")
    graph.add_node(node)
    graph.add_page(Page(
        id="page_1",
        name="主页",
        description="包含尚未点击的侧边菜单入口",
        controls=["可点击的‘侧边菜单’按钮"],
    ), "func_1")
    for index in range(5):
        node.add_scenario(f"已完成的不同场景 {index}")

    class Llm:
        calls = 0

        def invoke(self, _messages):
            self.calls += 1
            content = (
                '{"workflow_complete":false,"scenario_type":"intra_test",'
                '"target_function":"func_1",'
                '"src_function":"","dst_function":"",'
                '"navigation_scenario":"","test_scenario":'
                '"在主页点击‘侧边菜单’按钮，观察并记录实际出现的页面或抽屉"}'
            )
            return type("Response", (), {"content": content})()

    llm = Llm()
    planner = create_scenario_plan_node(llm, object())
    result = planner({
        "ffg": graph,
        "current_page": graph.get_page("page_1"),
        "current_testing_function_id": "func_1",
        "current_testing_function_status": "in_progress",
        "current_plan": {"scenario_type": "intra_test"},
        "run_phase": "all",
        "completion_audit_updates": 0,
        "completion_audit_signatures": {},
    })

    assert llm.calls == 1
    assert result["current_plan"]["scenario_type"] == "intra_test"
    assert result["current_plan"]["target_function_id"] == "func_1"
    assert graph.get_node_test_status("func_1") == "in_progress"


def test_duplicate_planner_feedback_is_injected_into_the_next_request():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="记录活动", test_status="in_progress"))
    graph.add_page(Page(
        id="page_1",
        name="活动主页",
        description="包含活动按钮、搜索、添加和菜单入口",
    ), "func_1")

    class Llm:
        def __init__(self):
            self.messages = []

        def invoke(self, messages):
            self.messages = messages
            content = (
                '{"workflow_complete":false,"scenario_type":"intra_test",'
                '"target_function":"func_1",'
                '"src_function":"","dst_function":"",'
                '"navigation_scenario":"",'
                '"test_scenario":"在活动主页点击搜索图标，观察搜索输入框"}'
            )
            return type("Response", (), {"content": content})()

    llm = Llm()
    node = create_scenario_plan_node(llm, object())
    result = node({
        "ffg": graph,
        "current_page": graph.get_page("page_1"),
        "current_testing_function_id": "func_1",
        "current_testing_function_status": "in_progress",
        "run_phase": "all",
        "planner_noop_count": 1,
        "planner_feedback": "检测到相似历史场景，要求改选未覆盖入口",
        "planner_rejected_scenario": "点击 Woodworking 活动按钮",
    })

    text_parts = [
        item.get("text", "")
        for message in llm.messages
        for item in getattr(message, "content", [])
        if isinstance(item, dict)
    ]
    assert "同类参数化入口的其他取值" in "".join(text_parts)
    assert "点击搜索图标" in result["current_test_scenario"]


def test_repeated_duplicate_noop_releases_function_without_marking_it_completed():
    graph = FlowGraph()
    node = FunctionNode(id="func_1", name="记录活动", test_status="in_progress")
    graph.add_node(node)
    graph.add_page(Page(id="page_1", name="活动主页"), "func_1")

    result = _planner_status_update_response(
        graph,
        graph.get_page("page_1"),
        "core",
        "func_1",
        "in_progress",
        "检测到 LLM 生成了同功能重复控件或相似历史场景，要求改选未覆盖入口",
        {"planner_noop_count": 2, "run_phase": "all"},
        rejected_scenario="点击 Woodworking 活动按钮",
    )

    assert graph.get_node("func_1").test_status == "in_progress"
    assert result["current_plan"]["scenario_type"] == "status_update"
    assert result["current_test_scenario"] == ""
    assert result["current_testing_function_id"] == ""
    assert result["function_execution_failure_counts"] == {"func_1": 1}
    assert result["planner_noop_count"] == 0


def test_low_coverage_empty_plan_immediately_generates_representative_scenario():
    graph = FlowGraph()
    node = FunctionNode(id="func_1", name="记录活动", test_status="in_progress")
    graph.add_node(node)
    graph.add_page(Page(
        id="page_1",
        name="活动主页",
        controls=["搜索按钮"],
    ), "func_1")

    result = _planner_status_update_response(
        graph,
        graph.get_page("page_1"),
        "core",
        "func_1",
        "in_progress",
        "程序派生覆盖率不足且仍有未处理控件，不能直接完成",
        {"planner_noop_count": 0, "run_phase": "all"},
    )

    assert result["current_plan"]["scenario_type"] == "intra_test"
    assert result["current_plan"]["target_function_id"] == "func_1"
    assert "搜索按钮" in result["current_test_scenario"]
    assert result["planner_noop_count"] == 0


def test_shared_resource_fallback_does_not_depend_on_navigation_direction():
    graph = _graph_with_two_functions()
    graph.get_node("func_1").resource_variables = ["任务记录"]
    graph.get_node("func_2").resource_variables = ["任务记录"]
    candidates = _candidate_cross_pairs(graph)

    assert candidates == [(
        "func_1",
        "func_2",
        "共享资源变量: 任务记录",
    )]


def test_planner_uses_empty_intra_scenario_as_programmatic_completion():
    intra_result = {
        "workflow_complete": False,
        "scenario_type": "intra_test",
        "target_function": "func_1",
        "src_function": "",
        "dst_function": "",
        "navigation_scenario": "",
        "test_scenario": "",
    }
    cross_result = {
        **intra_result,
        "scenario_type": "cross_test",
        "target_function": "",
        "src_function": "func_1",
        "dst_function": "func_2",
    }
    null_completion = {
        **intra_result,
        "navigation_scenario": None,
        "test_scenario": None,
    }

    assert _validate_plan_result(intra_result) == []
    assert "cross_test_requires_nonempty_test_scenario" in _validate_plan_result(cross_result)
    assert _validate_plan_result(null_completion) == []


def test_done_plan_is_rejected_because_completion_is_programmatic():
    done_result = {
        "workflow_complete": True,
        "scenario_type": "done",
        "target_function": None,
        "src_function": None,
        "dst_function": None,
        "navigation_scenario": "",
        "test_scenario": "",
    }

    assert "invalid_scenario_type='done'" in _validate_plan_result(done_result)


def test_planner_no_longer_rejects_transient_control_mentions():
    """瞬时控件校验已删除；UNDO/Toast 等词出现在场景中不再被程序拒绝。"""
    standalone_undo = {
        "workflow_complete": False,
        "scenario_type": "intra_test",
        "target_function": "func_1",
        "src_function": "",
        "dst_function": "",
        "navigation_scenario": "",
        "test_scenario": "探索编辑功能，若出现 UNDO 按钮则观察其效果",
    }
    immediate_undo = {
        **standalone_undo,
        "test_scenario": "点击活动按钮启动记录，紧接着点击 UNDO 按钮并观察结果",
    }

    assert not any(
        error.startswith("test_scenario_requires_live_transient_control=")
        for error in _validate_plan_result(standalone_undo)
    )
    assert not any(
        error.startswith("test_scenario_requires_live_transient_control=")
        for error in _validate_plan_result(immediate_undo)
    )


def test_planner_accepts_open_ended_conditional_scenario():
    """方案 B：下游结果未记录时，允许条件开放结尾（若出现则继续，否则结束）。"""
    search_scenario = {
        "workflow_complete": False,
        "scenario_type": "intra_test",
        "target_function": "func_2",
        "src_function": "",
        "dst_function": "",
        "navigation_scenario": "",
        "test_scenario": (
            "点击搜索框并输入关键词，若出现搜索结果，点击首个结果查看详情；"
            "若结果为空则结束"
        ),
    }
    conditional_nav_scenario = {
        **search_scenario,
        "test_scenario": (
            "点击打开相机入口，若出现相机取景页面，点击拍照按钮确认进入；"
            "若未出现则结束"
        ),
    }

    assert _validate_plan_result(search_scenario) == []
    assert _validate_plan_result(conditional_nav_scenario) == []
    # 开放结尾不应触发穷尽式控件拒绝或动作数超限
    errors = _validate_plan_result(search_scenario)
    assert "test_scenario_exhaustive_control_family" not in errors
    assert "test_scenario_too_many_actions" not in errors


def test_open_ended_scenario_no_longer_action_counted():
    """探索方向语义下不再按动作数量限制场景；长方向场景应放行。"""
    padded = {
        "workflow_complete": False,
        "scenario_type": "intra_test",
        "target_function": "func_1",
        "src_function": "",
        "dst_function": "",
        "navigation_scenario": "",
        "test_scenario": "，".join(["点击按钮"] * 8) + "，若出现结果则点击查看",
    }

    assert not any(
        error.startswith("test_scenario_too_many_actions=")
        for error in _validate_plan_result(padded)
    )


def test_plan_prompt_mentions_open_ending_support():
    """规划 system prompt 应包含泛化的开放结尾条件分支规则。"""
    assert "开放结尾" in UNIFIED_SCENARIO_PLAN_SYSTEM_PROMPT
    assert "下游" in UNIFIED_SCENARIO_PLAN_SYSTEM_PROMPT
    assert "若出现明确且未执行的相关结果/入口则继续" in UNIFIED_SCENARIO_PLAN_SYSTEM_PROMPT
    assert "具体动作由执行器根据每一步截图决定" in UNIFIED_SCENARIO_PLAN_SYSTEM_PROMPT
    assert "输入关键词后，若出现搜索结果" not in UNIFIED_SCENARIO_PLAN_SYSTEM_PROMPT


def test_plan_prompt_splits_independent_stable_flows():
    assert "一个场景只探索一个独立业务目标" in UNIFIED_SCENARIO_PLAN_SYSTEM_PROMPT
    assert "会打开不同稳定流程的入口绝不打包" in UNIFIED_SCENARIO_PLAN_SYSTEM_PROMPT
    assert "相册、位置、文件、联系人" in UNIFIED_SCENARIO_PLAN_SYSTEM_PROMPT
    assert "一旦进入新的全屏页面、独立 Activity" in UNIFIED_SCENARIO_PLAN_SYSTEM_PROMPT


def test_executor_stops_at_first_independent_stable_flow():
    assert "场景边界按“独立稳定流程”判断" in SCENARIO_EXEC_SYSTEM_PROMPT_TEMPLATE
    assert "任务只要求进入设置就停在设置页" in SCENARIO_EXEC_SYSTEM_PROMPT_TEMPLATE
    assert "不能再顺带进入搜索、媒体、文件、位置、联系人" in SCENARIO_EXEC_SYSTEM_PROMPT_TEMPLATE


def test_plan_transport_failure_does_not_multiply_semantic_retries(monkeypatch):
    class FailingLlm:
        calls = 0

        def invoke(self, _messages):
            self.calls += 1
            raise RuntimeError("temporary gateway failure")

    llm = FailingLlm()
    monkeypatch.setattr(scenario_plan_module, "PLAN_VALIDATION_RETRY_ATTEMPTS", 3)
    monkeypatch.setenv("LLM_RETRY_ATTEMPTS", "1")
    result, errors = scenario_plan_module._invoke_valid_plan(llm, "prompt", [])

    assert result == {}
    assert errors == ["llm_request_or_parse_failed"]
    assert llm.calls == 1


def test_unfinished_current_function_is_dispatched_before_pending_cross_test():
    graph = _graph_with_two_functions()
    graph.add_page(Page(id="page_1", name="任务编辑"), "func_1")
    graph.add_page(Page(id="page_2", name="任务记录"), "func_2")
    graph.add_page_transition(PageTransition("page_1", "page_2", "打开任务记录"))
    graph.set_node_test_status("func_1", "in_progress")
    graph.add_function_dependency(FunctionDependency("func_1", "func_2", "任务记录"))

    class Response:
        content = (
            '{"workflow_complete":false,"scenario_type":"intra_test",'
            '"target_function":"func_1",'
            '"src_function":"","dst_function":"",'
            '"navigation_scenario":"",'
            '"test_scenario":"在源功能编辑并保存一条任务，确认保存结果"}'
        )

    class Llm:
        def __init__(self):
            self.calls = 0

        def invoke(self, _messages):
            self.calls += 1
            return Response()

    llm = Llm()
    node = create_scenario_plan_node(llm, object())
    result = node({
        "ffg": graph,
        "current_page": Page(id="page_1", name="任务编辑"),
        "current_testing_function_id": "func_1",
        "current_testing_function_status": "in_progress",
        "run_phase": "all",
        "planner_noop_count": 0,
    })

    assert llm.calls == 1
    assert result["current_plan"]["scenario_type"] == "intra_test"
    assert result["current_testing_function_id"] == "func_1"
    assert "编辑并保存" in result["current_test_scenario"]


def test_planner_unfinished_function_is_dispatched_before_completed_function_audit():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="主导航", test_status="completed"))
    graph.add_node(FunctionNode(id="func_2", name="待测功能", test_status="not_started"))
    graph.add_page(Page(
        id="page_1",
        name="Navigation Drawer",
        description="包含 Diary、Settings 和 About 菜单项",
    ), "func_1")
    graph.add_page(Page(
        id="page_2",
        name="待测页面",
        description="包含待测入口",
    ), "func_2")
    graph.add_page_transition(PageTransition("page_1", "page_2", "点击待测功能入口"))

    class AuditLlm:
        calls = 0

        def invoke(self, _messages):
            self.calls += 1
            return type("Response", (), {"content": (
                '{"workflow_complete":false,"scenario_type":"intra_test",'
                '"target_function":"func_2",'
                '"src_function":"","dst_function":"",'
                '"navigation_scenario":"","test_scenario":'
                '"点击待测页面中的待测入口 → 确认结果后结束"}'
            )})()

    llm = AuditLlm()
    node = create_scenario_plan_node(llm, object())
    result = node({
        "ffg": graph,
        "current_page": Page(id="page_1", name="Navigation Drawer"),
        "current_plan": {
            "scenario_type": "status_update",
            "target_function_id": "func_1",
        },
        "current_testing_function_id": "",
        "current_testing_function_status": "not_started",
        "run_phase": "all",
        "planner_noop_count": 0,
    })

    assert llm.calls == 1
    assert result["current_plan"].get("audit_exploration") is not True
    assert result["current_plan"]["target_function_id"] == "func_2"
    assert "待测入口" in result["current_test_scenario"]


def test_completed_coverage_finishes_early_when_frontier_is_exhausted():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="主功能", test_status="completed"))
    graph.add_page(Page(id="page_1", name="主页", operations=["点击刷新按钮"]), "func_1")

    class DoneLlm:
        def invoke(self, _messages):
            return type("Response", (), {"content": (
                '{"has_exploration_gap":false,"entry_page_ref":"",'
                '"entry_name":"","test_scenario":""}'
            )})()

    node = create_scenario_plan_node(DoneLlm(), object())
    result = node({
        "ffg": graph,
        "current_page": Page(id="page_1", name="主页"),
        "current_plan": {},
        "current_testing_function_id": "",
        "current_testing_function_status": "not_started",
        "run_phase": "all",
        "start_time": time.time(),
        "total_time_limit": 1800,
        "iteration_count": 3,
    })

    assert result["workflow_complete"] is True
    assert result["current_plan"]["scenario_type"] == "status_update"
    assert result["current_test_scenario"] == ""


def test_completion_audit_gets_recorded_navigation_to_its_entry_page():
    graph = FlowGraph()
    node = FunctionNode(id="func_1", name="笔记列表")
    graph.add_node(node)
    graph.add_page(Page(id="page_1", name="笔记首页"), "func_1")
    graph.add_page(Page(id="page_2", name="新增类型菜单"), "func_1")
    graph.add_page(Page(id="page_3", name="笔记编辑页"), "func_1")
    graph.add_page_transition(PageTransition("page_1", "page_2", "点击新增按钮"))

    navigation = _build_audit_navigation_scenario(
        graph,
        graph.get_page("page_3"),
        "func_1",
        "page_2",
    )

    assert "先执行一次 back" in navigation
    assert "直接回到“新增类型菜单”" in navigation
    assert "再执行一次 open_app" in navigation
    assert "点击新增按钮" in navigation
    assert "不要在导航阶段点击本轮待验证入口" in navigation
    assert "主动打开的菜单属于导航路径，不要再把它关闭" in navigation
    assert _build_audit_navigation_scenario(
        graph,
        graph.get_page("page_2"),
        "func_1",
        "page_2",
    ) == ""


def test_completion_audit_uses_open_app_first_for_unrelated_current_page():
    graph = FlowGraph()
    node = FunctionNode(id="func_1", name="笔记列表")
    graph.add_node(node)
    graph.add_node(FunctionNode(id="func_2", name="其他功能"))
    graph.add_page(Page(id="page_1", name="笔记首页"), "func_1")
    graph.add_page(Page(id="page_2", name="新增类型菜单"), "func_1")
    graph.add_page(Page(id="page_other", name="其他功能页面"), "func_2")
    graph.add_page_transition(PageTransition("page_1", "page_2", "点击新增按钮"))

    navigation = _build_audit_navigation_scenario(
        graph,
        graph.get_page("page_other"),
        "func_1",
        "page_2",
    )

    assert "先执行一次 open_app" in navigation
    assert "先执行一次 back" not in navigation


def test_audit_exploration_replaces_internal_ids_in_user_facing_scenario():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="消息会话"))
    graph.add_page(Page(id="page_1", name="聊天页面"), "func_1")

    result = _audit_exploration_response(
        graph,
        graph.get_page("page_1"),
        {
            "owner_function_id": "func_1",
            "entry_page_id": "page_1",
            "test_scenario": "到达page_1 → 点击表情搜索入口 → 确认结果后结束",
        },
        0,
        {},
    )

    assert "page_1" not in result["current_test_scenario"]
    assert "聊天页面" in result["current_test_scenario"]


def test_forced_progress_selects_only_one_uncovered_control():
    graph = FlowGraph()
    node = FunctionNode(id="func_1", name="笔记列表")
    graph.add_node(node)
    graph.add_page(Page(
        id="page_1",
        name="笔记首页",
        controls=[
            "可点击的‘搜索’按钮",
            "可点击的‘排序’按钮",
            "可点击的‘侧边栏’按钮",
        ],
        operations=["点击搜索按钮", "打开侧边栏"],
    ), "func_1")

    scenario = _forced_intra_scenario(graph, node, hint_index=1)

    assert "排序" in scenario
    assert "搜索" not in scenario
    assert "侧边栏" not in scenario
    assert "尚无执行记录" in scenario


def test_forced_progress_may_choose_wechat_conversation_control():
    graph = FlowGraph()
    node = FunctionNode(id="func_1", name="微信首页")
    graph.add_node(node)
    graph.add_page(Page(
        id="page_1",
        name="微信首页",
        controls=[
            "可点击的‘文件传输助手会话’",
            "可点击的‘通讯录标签’",
        ],
    ), "func_1")

    scenario = _forced_intra_scenario(
        graph,
        node,
        package_name="com.tencent.mm",
    )

    assert "文件传输助手" in scenario


def test_exec_fallback_marks_current_page_unknown():
    state = {
        "current_page": Page(id="page_old"),
        "current_test_scenario": "测试保存",
        "current_plan": {"scenario_type": "intra_test"},
    }

    updates = _fallback_scenario_exec(state, RuntimeError("boom"))

    assert updates["current_page"] is None
    assert updates["execution_result"]["scenario_status"] == "error"
    assert updates["execution_result"]["scenario_error"] == "场景执行节点异常: boom"
    assert updates["execution_result"]["scenario_trace_length"] == 0
    assert not {
        "test_status",
        "test_error",
        "test_trace_length",
    }.intersection(updates["execution_result"])


def test_missing_plan_target_does_not_fall_back_to_previous_function():
    graph = _graph_with_two_functions()

    assert _resolve_plan_target_function_id(graph, {}) == ""
    assert _resolve_plan_target_function_id(
        graph,
        {"target_function_id": "func_2"},
    ) == "func_2"


def test_finished_with_partial_success_is_not_a_failure():
    from workflow.nodes.scenario_exec import _finished_declares_failure

    thought = (
        "当前已成功返回Notes笔记列表页面，列表最顶部已显示标题为“FuncDroid_29”的清单笔记，"
        "已完成标题输入、确认默认待办项“New item”及其复选框存在的步骤，"
        "仅新增额外待办项的操作未成功触发，核心任务的主要验证点已覆盖，可结束本次执行任务。"
    )

    assert _finished_declares_failure(thought) is False


def test_force_progress_releases_current_function_without_uncovered_controls():
    graph = _graph_with_two_functions()
    graph.add_page(Page(
        id="page_2",
        name="目标页",
        controls=["可点击的‘新入口’按钮"],
    ), "func_2")
    graph.get_node("func_1").add_scenario("已尝试过一次")
    graph.get_node("func_1").test_status = "in_progress"
    graph.get_node("func_2").test_status = "in_progress"

    state = {"run_phase": "all", "current_plan": {}}
    result = _force_progress_after_planner_noop(
        graph,
        None,
        "core",
        "func_1",
        "in_progress",
        state,
        "连续空规划达到阈值",
        3,
    )

    assert result["current_testing_function_id"] == ""
    assert result["current_plan"]["target_function_id"] == "func_1"
    assert result["current_plan"]["scenario_type"] == "status_update"
    assert graph.get_node("func_1").test_status == "in_progress"
    assert result["function_execution_failure_counts"] == {"func_1": 1}
    assert "target_function_status" not in result["current_plan"]


def test_intra_target_prefers_reachable_unpenalized_function():
    graph = _graph_with_two_functions()
    graph.add_page(Page(id="page_1", name="源页面"), "func_1")
    graph.add_page(Page(id="page_2", name="目标页面"), "func_2")
    graph.add_page_transition(PageTransition("page_1", "page_2", "进入目标功能"))

    selected = scenario_plan_module._select_intra_target(
        graph,
        graph.get_page("page_1"),
        "",
        {"func_1": 1},
    )

    assert selected is not None
    assert selected.id == "func_2"
