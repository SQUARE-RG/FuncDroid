"""操作-控件显式绑定（覆盖率权威来源）的回归测试。

背景：纯文本家族匹配对“控件清单命名 vs 操作措辞”的措辞差系统性失配
（微信实测 13 控件 + 9 条真实操作命中 0），覆盖率恒为 0。显式绑定由
FFG 更新 patch 声明并做清单原文校验，绑定命中即为覆盖。
"""

from __future__ import annotations

import pytest

from model import flow_graph as flow_graph_module
from model.flow_graph import FlowGraph, FunctionNode, Page
from workflow.nodes import ffg_update as ffg_update_module
from workflow.nodes.ffg_update import (
    _apply_operation_bindings,
    _auto_bind_unambiguous_trace_operations,
)


def _wechat_page() -> Page:
    """微信_11 实测 page_1 的精简复刻：真实控件命名 + 真实操作措辞。"""
    return Page(
        id="page_1",
        name="微信主界面",
        controls=[
            "聊天列表项-微信团队",
            "聊天列表项-公众号",
            "底部导航-通讯录Tab",
            "底部导航-发现Tab",
            "底部导航-我Tab",
            "左上角返回按钮",
        ],
        operations=[
            "点击聊天列表中的“微信团队”条目",
            "点击底部导航栏中的“通讯录”Tab",
            "在聊天列表中的“公众号”条目上执行左滑操作",
            "点击底部导航栏中的“发现”Tab",
        ],
    )


def _graph_with_wechat_page() -> tuple[FlowGraph, Page]:
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="消息列表浏览"))
    page = _wechat_page()
    graph.add_page(page, "func_1")
    return graph, page


def test_text_matching_alone_leaves_wechat_page_uncovered():
    """措辞失配基线：无绑定时这些真实操作命中不了对应控件。"""
    graph, _page = _graph_with_wechat_page()
    handled = graph.function_handled_controls("func_1")
    # “聊天列表项-微信团队”与“点击聊天列表中的微信团队条目”家族冲突；
    # “底部导航-通讯录Tab”与“点击底部导航栏中的通讯录Tab”噪声剥离不对称。
    assert ("page_1", flow_graph_module._text_key("聊天列表项-微信团队")) not in handled
    assert ("page_1", flow_graph_module._text_key("底部导航-通讯录Tab")) not in handled


def test_bindings_cover_controls_that_text_matching_misses():
    graph, page = _graph_with_wechat_page()
    assert page.add_operation_binding(
        "点击聊天列表中的“微信团队”条目",
        "聊天列表项-微信团队",
    )
    assert page.add_operation_binding(
        "点击底部导航栏中的“通讯录”Tab",
        "底部导航-通讯录Tab",
    )
    assert page.add_operation_binding(
        "在聊天列表中的“公众号”条目上执行左滑操作",
        "聊天列表项-公众号",
    )

    handled = graph.function_handled_controls("func_1")
    for control in ("聊天列表项-微信团队", "底部导航-通讯录Tab", "聊天列表项-公众号"):
        assert ("page_1", flow_graph_module._text_key(control)) in handled

    details = graph.function_coverage_details("func_1")
    assert details["executed_operations"] == 3
    assert "聊天列表项-微信团队" not in details["uncovered_controls"]
    assert details["coverage_percent"] > 0


def test_binding_rejects_texts_not_in_page_lists():
    """幻觉控件名/操作文本必须被拒绝（防模型编造绑定抬高覆盖率）。"""
    page = _wechat_page()
    assert page.add_operation_binding("点击不存在的操作", "聊天列表项-微信团队") is False
    assert page.add_operation_binding("点击聊天列表中的“微信团队”条目", "不存在的控件") is False
    assert page.operation_bindings == []


def test_binding_matches_survive_lenient_normalization():
    page = _wechat_page()
    # 归一化一致（全角引号/空白差异）也应接受，降低逐字复制的脆弱性。
    assert page.add_operation_binding(
        "点击聊天列表中的“微信团队”条目 ",  # 尾部空白
        "聊天列表项-微信团队",
    )
    assert page.operation_bindings == [{
        "operation": "点击聊天列表中的“微信团队”条目",
        "control": "聊天列表项-微信团队",
    }]


def test_binding_deduplicates_repeated_declarations():
    page = _wechat_page()
    assert page.add_operation_binding("点击聊天列表中的“微信团队”条目", "聊天列表项-微信团队")
    assert page.add_operation_binding("点击聊天列表中的“微信团队”条目", "聊天列表项-微信团队") is False
    assert len(page.operation_bindings) == 1


def test_one_operation_cannot_cover_two_different_controls():
    page = Page(
        id="page_1",
        controls=["更多扩展功能按钮", "更多信息按钮"],
        operations=["点击更多按钮"],
    )

    assert page.add_operation_binding("点击更多按钮", "更多扩展功能按钮")
    assert page.add_operation_binding("点击更多按钮", "更多信息按钮") is False
    assert page.operation_bindings == [{
        "operation": "点击更多按钮",
        "control": "更多扩展功能按钮",
    }]


def test_recovery_and_failure_bindings_do_not_count(monkeypatch):
    graph, page = _graph_with_wechat_page()
    page.add_operation("自动恢复：打开微信主页面")
    page.add_operation("尝试点击底部导航-发现Tab失败：超时")
    assert page.add_operation_binding("自动恢复：打开微信主页面", "聊天列表项-微信团队")
    assert page.add_operation_binding("尝试点击底部导航-发现Tab失败：超时", "底部导航-发现Tab")
    # 关闭文本兜底，隔离验证“恢复/失败操作的绑定不计覆盖”。
    monkeypatch.setattr(flow_graph_module, "COVERAGE_LEGACY_TEXT_MATCHING", False)
    assert graph.function_handled_controls("func_1") == set()


def test_legacy_text_matching_can_be_disabled(monkeypatch):
    """绑定数据充分后可关闭文本兜底：只认绑定。"""
    graph, page = _graph_with_wechat_page()
    page.add_operation_binding("点击底部导航栏中的“发现”Tab", "底部导航-发现Tab")
    monkeypatch.setattr(flow_graph_module, "COVERAGE_LEGACY_TEXT_MATCHING", False)

    handled = graph.function_handled_controls("func_1")
    assert handled == {("page_1", flow_graph_module._text_key("底部导航-发现Tab"))}


def test_legacy_text_matching_is_off_by_default(monkeypatch):
    """实测发现的假阳性回归：泛化家族一致不得把未操作控件计入覆盖。

    “点击服务通知列表项”与“列表项-微信团队”实例名剥离后家族同为
    “列表项”，文本兜底会把从未被操作过的微信团队/测试条目误标为已
    覆盖；默认只认显式绑定。
    """
    monkeypatch.delenv("FFG_COVERAGE_LEGACY_MATCHING", raising=False)
    import importlib

    importlib.reload(flow_graph_module)
    assert flow_graph_module.COVERAGE_LEGACY_TEXT_MATCHING is False

    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="消息列表浏览"))
    page = Page(
        id="page_1",
        controls=["列表项-微信团队", "列表项-服务通知", "列表项-测试"],
        operations=["点击服务通知列表项"],
    )
    graph.add_page(page, "func_1")
    assert graph.function_handled_controls("func_1") == set()


def test_legacy_text_matching_can_be_reenabled(monkeypatch):
    """显式开启时文本兜底仍可用（向后兼容开关）。"""
    monkeypatch.setattr(flow_graph_module, "COVERAGE_LEGACY_TEXT_MATCHING", True)
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="消息列表浏览"))
    page = Page(
        id="page_1",
        controls=["发送按钮"],
        operations=["点击发送按钮"],
    )
    graph.add_page(page, "func_1")
    assert graph.function_handled_controls("func_1") == {
        ("page_1", flow_graph_module._text_key("发送按钮")),
    }


def test_remove_operation_drops_its_bindings():
    page = _wechat_page()
    page.add_operation_binding("点击底部导航栏中的“通讯录”Tab", "底部导航-通讯录Tab")
    assert page.remove_operation("点击底部导航栏中的“通讯录”Tab") is True
    assert page.operation_bindings == []


def test_operation_merge_keeps_bindings_consistent():
    """家族归并替换操作原文时，绑定应迁移到保留文本上。"""
    page = Page(
        id="page_1",
        controls=["发送按钮"],
        operations=["点击发送按钮", "点击位于聊天列表底部的绿色的圆形发送按钮"],
        operation_bindings=[{"operation": "点击发送按钮", "control": "发送按钮"}],
    )
    # 更丰富表述（长度差超过 12 字）在家族归并中胜出。
    assert page.operations == ["点击位于聊天列表底部的绿色的圆形发送按钮"]
    assert page.operation_bindings == [{
        "operation": "点击位于聊天列表底部的绿色的圆形发送按钮",
        "control": "发送按钮",
    }]


def test_bindings_roundtrip_through_serialization():
    graph, page = _graph_with_wechat_page()
    page.add_operation_binding("点击底部导航栏中的“通讯录”Tab", "底部导航-通讯录Tab")
    data = graph.to_dict()
    restored = FlowGraph.from_dict(data)
    restored_page = restored.get_page("page_1")
    assert restored_page is not None
    assert restored_page.operation_bindings == [{
        "operation": "点击底部导航栏中的“通讯录”Tab",
        "control": "底部导航-通讯录Tab",
    }]
    assert restored.function_handled_controls("func_1") == {
        ("page_1", flow_graph_module._text_key("底部导航-通讯录Tab")),
    }


def test_invalid_bindings_dropped_on_deserialization():
    page = Page(
        id="page_1",
        controls=["保存按钮"],
        operations=["点击保存按钮"],
        operation_bindings=[
            {"operation": "点击保存按钮", "control": "保存按钮"},
            {"operation": "幻觉操作", "control": "保存按钮"},
            {"operation": "点击保存按钮", "control": "幻觉控件"},
        ],
    )
    assert page.operation_bindings == [
        {"operation": "点击保存按钮", "control": "保存按钮"},
    ]


def _trace_step(
    raw_step: int,
    action_name: str,
    action_description: str,
    **extra,
) -> dict:
    step = {
        "raw_step_index": raw_step,
        "action_name": action_name,
        "action_description": action_description,
    }
    step.update(extra)
    return step


def test_apply_operation_bindings_resolves_step_and_control():
    """patch 用 source_step 声明绑定：操作原文由程序按 trace 口径推导。"""
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="消息列表浏览"))
    page = _wechat_page()
    graph.add_page(page, "func_1")
    trace = [
        _trace_step(1, "click", "点击底部导航栏中的“通讯录”Tab"),
    ]
    trace_page_map = {1: "page_1"}

    applied = _apply_operation_bindings(
        graph,
        {"operation_bindings": [{"source_step": 1, "control": "底部导航-通讯录Tab"}]},
        trace,
        trace_page_map,
    )

    assert applied == 1
    assert page.operation_bindings == [{
        "operation": "点击底部导航栏中的“通讯录”Tab",
        "control": "底部导航-通讯录Tab",
    }]
    assert ("page_1", flow_graph_module._text_key("底部导航-通讯录Tab")) in (
        graph.function_handled_controls("func_1")
    )


def test_apply_operation_bindings_drops_invalid_entries():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="消息列表浏览"))
    page = _wechat_page()
    graph.add_page(page, "func_1")
    trace = [
        _trace_step(1, "click", "点击底部导航栏中的“通讯录”Tab"),
        _trace_step(2, "click", "点击聊天列表中的“微信团队”条目"),
    ]
    trace_page_map = {1: "page_1", 2: "page_1"}

    applied = _apply_operation_bindings(
        graph,
        {"operation_bindings": [
            {"source_step": 99, "control": "底部导航-通讯录Tab"},  # 不存在的步骤
            {"source_step": 2},                                    # 缺 control
            {"operation": "幻觉操作文本", "control": "底部导航-通讯录Tab"},  # 无 source_step 且操作不在清单
        ]},
        trace,
        trace_page_map,
    )

    assert applied == 0
    assert page.operation_bindings == []


def test_apply_operation_bindings_rejects_missing_control():
    """绑定不能借真实操作自动注册模型臆造或归错页面的控件。"""
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="消息列表浏览"))
    page = _wechat_page()
    graph.add_page(page, "func_1")
    # 与真实流程一致：trace 操作先落图，绑定随后声明。
    page.add_operation("点击分享面板中的“取消”按钮")
    trace = [_trace_step(1, "click", "点击分享面板中的“取消”按钮")]
    trace_page_map = {1: "page_1"}

    applied = _apply_operation_bindings(
        graph,
        {"operation_bindings": [{"source_step": 1, "control": "取消按钮"}]},
        trace,
        trace_page_map,
    )

    assert applied == 0
    assert "取消按钮" not in page.controls
    assert page.operation_bindings == []


def test_unambiguous_existing_control_is_bound_without_model_declaration():
    graph, page = _graph_with_wechat_page()
    trace = [_trace_step(1, "click", "点击底部导航栏中的“通讯录”Tab")]

    applied = _auto_bind_unambiguous_trace_operations(
        graph,
        trace,
        {1: "page_1"},
    )

    assert applied == 1
    assert page.operation_bindings == [{
        "operation": "点击底部导航栏中的“通讯录”Tab",
        "control": "底部导航-通讯录Tab",
    }]


def test_scenario_credit_goes_to_planned_target_function():
    """P3 回归：场景学分归属规划目标功能，不被执行终点功能偷走。

    微信实测：func_10 的场景终点落在共享 page_2（属 func_2），按
    “实际到达功能”归属后 func_10 自身场景恒为 0，92% 覆盖仍无法收尾。
    """
    from workflow.nodes.ffg_update import _record_completed_intra_scenario

    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_2", name="执行会话浏览"))
    graph.add_node(FunctionNode(id="func_10", name="测试会话浏览"))
    graph.get_node("func_10").test_status = "in_progress"

    owner = _record_completed_intra_scenario(
        graph,
        {"target_function_id": "func_10"},
        "操作测试会话列表项并查看详情",
        endpoint_function_id="func_2",
    )

    assert owner == "func_10"
    assert "操作测试会话列表项并查看详情" in graph.get_node("func_10").scenarios
    assert graph.get_node("func_2").scenarios == []


def test_page_audit_removal_preserves_bound_operations():
    """P4 回归：已绑定控件的操作是有效业务动作，审计不得删除。

    微信实测 page_8“点击语音输入按钮”有绑定仍被审计删除，连带丢失
    绑定导致 func_7 覆盖率受损。
    """
    from workflow.nodes.ffg_update import _apply_page_audit_operation_removals

    page = _wechat_page()
    page.add_operation("点击输入法弹窗中的“微信输入法”选项")
    page.add_operation_binding(
        "点击底部导航栏中的“通讯录”Tab", "底部导航-通讯录Tab"
    )

    removed = _apply_page_audit_operation_removals(
        page,
        [
            "点击底部导航栏中的“通讯录”Tab",   # 已绑定：拒绝删除
            "点击输入法弹窗中的“微信输入法”选项",  # 未绑定：可删
            "清单里不存在的操作",                # 原文不一致：丢弃
        ],
    )

    assert removed == ["点击输入法弹窗中的“微信输入法”选项"]
    assert "点击底部导航栏中的“通讯录”Tab" in page.operations
    assert page.operation_bindings  # 绑定完好


def test_apply_operation_bindings_redeclaration_is_not_validation_failure():
    """refinement 重复声明已记录的绑定属正常现象，不是校验失败。

    微信实测：第 1 次尝试记录绑定成功后，第 2、3 次尝试重复声明同样
    绑定，曾被误报为“操作或控件不在本页清单”。
    """
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="消息列表浏览"))
    page = _wechat_page()
    graph.add_page(page, "func_1")
    trace = [_trace_step(1, "click", "点击底部导航栏中的“通讯录”Tab")]
    trace_page_map = {1: "page_1"}
    patch_payload = {"operation_bindings": [{"source_step": 1, "control": "底部导航-通讯录Tab"}]}

    assert _apply_operation_bindings(graph, patch_payload, trace, trace_page_map) == 1
    # 重复声明：不再新增，但也不是校验失败（不应计入丢弃告警）。
    assert _apply_operation_bindings(graph, patch_payload, trace, trace_page_map) == 0
    assert len(page.operation_bindings) == 1


def test_apply_operation_bindings_ignores_press_back_steps():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="消息列表浏览"))
    page = _wechat_page()
    graph.add_page(page, "func_1")
    trace = [_trace_step(1, "press_back", "返回上一页")]
    trace_page_map = {1: "page_1"}

    applied = _apply_operation_bindings(
        graph,
        {"operation_bindings": [{"source_step": 1, "control": "底部导航-通讯录Tab"}]},
        trace,
        trace_page_map,
    )

    assert applied == 0
    assert page.operation_bindings == []


def test_ffg_update_prompt_declares_operation_bindings():
    assert "operation_bindings" in ffg_update_module.FFG_UPDATE_SYSTEM_PROMPT
    assert "source_step" in ffg_update_module.FFG_UPDATE_SYSTEM_PROMPT


@pytest.mark.parametrize(
    "raw,expected_x,expected_y",
    [
        ('{"x": 226, 190}', 226, 190),
        ('{"x": 54, 69 }', 54, 69),
        ('{"x": 54,\n 69}', 54, 69),
        ('{"x": [682, 954]}', 682, 954),
    ],
)
def test_llm_json_repair_fixes_missing_coordinate_key(raw, expected_x, expected_y):
    import json

    from workflow.llm import _repair_common_json_typos

    repaired = _repair_common_json_typos(raw)
    payload = json.loads(repaired)
    assert payload["x"] == expected_x
    assert payload["y"] == expected_y


def test_llm_json_repair_leaves_valid_text_untouched():
    from workflow.llm import _repair_common_json_typos

    original = '{"x": 100, "y": 200, "action_name": "click"}'
    assert _repair_common_json_typos(original) == original
    # 超出坐标范围（0-1000）的普通数字对不修复。
    assert _repair_common_json_typos('{"x": 2000, 3000}') == '{"x": 2000, 3000}'


def test_action_json_parser_repairs_missing_y_key():
    """场景执行解析路径（_parse_action_json）必须接入统一修复层。

    微信实测第 3 轮：模型输出 "x": 949, 537（缺 "y": 键名）导致整步
    解析失败重试；该路径此前未调用 _repair_common_json_typos。
    """
    from workflow.nodes.scenario_exec import _parse_action_json

    raw = (
        "```json\n"
        "{\n"
        '  "thought": "点击底部加号按钮",\n'
        '  "action_name": "click",\n'
        '  "is_interruption": false,\n'
        '  "x": 949, 537\n'
        "}\n"
        "```"
    )
    plan = _parse_action_json(raw)
    assert plan["x"] == 949
    assert plan["y"] == 537
    assert plan["action_name"] == "click"


def test_action_plan_accepts_action_type_alias():
    """微信实测第 2 轮：模型输出 action_type 而非 action_name。"""
    from workflow.nodes.scenario_exec import _normalize_action_plan

    plan = _normalize_action_plan({
        "action_type": "press_back",
        "action_description": "点击左上角返回箭头",
        "is_interruption": False,
    })
    assert plan["action_name"] == "press_back"
