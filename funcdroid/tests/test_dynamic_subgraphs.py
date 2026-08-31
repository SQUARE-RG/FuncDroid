from __future__ import annotations

import json

import pytest

from model.flow_graph import (
    FlowGraph,
    FunctionDependency,
    FunctionNode,
    Page,
    PageTransition,
    adjusted_control_coverage,
    control_family_key,
    operation_family_key,
)
from workflow.subgraphs import ffg_update as ffg_update_subgraph_module
from workflow.nodes import scenario_plan as scenario_plan_module
from workflow.page_audit_worker import page_audit_worker
from workflow.nodes.ffg_update import (
    _build_ffg_summary,
    _build_observation_candidate_map,
    _combined_execution_trace,
    _normalize_observation_alignment,
    _select_granularity_review_function,
    apply_ffg_patch,
    validate_observation_alignment,
    validate_ffg_invariants,
)


@pytest.fixture(autouse=True)
def _legacy_text_matching_enabled(monkeypatch):
    """本文件用例写于操作-控件显式绑定之前，守护文本匹配引擎的边界行为。

    显式绑定是覆盖率的权威来源且文本兜底默认关闭
    （COVERAGE_LEGACY_TEXT_MATCHING）；这些用例断言的是旧引擎的防
    虚增/防误配守卫，需显式开启兜底。
    """
    from model import flow_graph as _flow_graph

    monkeypatch.setattr(_flow_graph, "COVERAGE_LEGACY_TEXT_MATCHING", True)


def test_navigation_and_scenario_traces_are_both_used_for_ffg_updates():
    state = {
        "navigation_execution_trace": [{"action_name": "click"}],
        "scenario_execution_trace": [{"action_name": "finished"}],
    }

    trace = _combined_execution_trace(state)

    assert [step["trace_phase"] for step in trace] == ["navigation", "test"]


def test_update_summary_keeps_one_hop_boundary_page_and_transition():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="首页"))
    graph.add_node(FunctionNode(id="func_2", name="收藏"))
    graph.add_page(Page(id="page_1", name="首页", activity="HomeActivity"), "func_1")
    graph.add_page(Page(id="page_2", name="收藏列表", activity="CollectionActivity"), "func_2")
    graph.add_page_transition(PageTransition("page_1", "page_2", "点击收藏入口"))
    trace = [{
        "raw_step_index": 1,
        "current_page": Page(id="page_1", name="首页", activity="HomeActivity"),
    }]

    summary = _build_ffg_summary(
        graph,
        {"target_function_id": "func_1"},
        trace,
    )
    facts = json.loads(summary.split("\n", 1)[1])

    assert facts["boundary_pages"] == [{
        "id": "page_2",
        "name": "收藏列表",
        "activity": "CollectionActivity",
        "owners": ["func_2"],
    }]
    assert facts["boundary_transitions"] == [{
        "src_page_id": "page_1",
        "dst_page_id": "page_2",
        "operation": "点击收藏入口",
    }]


def test_observation_candidates_rank_matching_activity_and_semantics_first():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="首页"))
    graph.add_node(FunctionNode(id="func_2", name="编辑"))
    graph.add_page(Page(id="page_1", name="首页", activity="HomeActivity"), "func_1")
    graph.add_page(Page(
        id="page_2",
        name="编辑内容",
        description="编辑标题和正文",
        controls=["标题输入框", "正文输入框"],
        activity="EditActivity",
    ), "func_2")
    trace = [{
        "raw_step_index": 4,
        "action_name": "finished",
        "current_page": Page(
            name="编辑内容",
            description="编辑标题和正文",
            controls=["标题输入框"],
            activity="EditActivity",
        ),
    }]

    candidate_map, _step_similarity_cache = _build_observation_candidate_map(
        graph,
        {"target_function_id": "func_2"},
        trace,
    )

    assert candidate_map[4][0]["page_id"] == "page_2"
    assert candidate_map[4][0]["activity_match"] is True
    assert candidate_map[4][0]["scope_match"] is True


def test_observation_alignment_maps_reused_page_without_pages_patch():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="首页"))
    graph.add_page(Page(
        id="page_1",
        name="首页",
        representative_screenshot="image-stable",
    ), "func_1")
    trace = [{
        "raw_step_index": 2,
        "current_page": Page(
            name="首页",
            representative_screenshot="image-expanded-panel",
        ),
    }]
    patch = {
        "observation_alignment": [{
            "source_step": 2,
            "page_decision": "reuse",
            "page_ref": "page_1",
        }],
    }

    result = apply_ffg_patch(graph, patch, trace)

    assert result["trace_page_map"] == {2: "page_1"}
    assert result["persisted_trace_images"] == 1
    assert graph.get_page("page_1").representative_screenshot == "image-stable"
    assert graph.get_page("page_1").evidence_screenshots == ["image-expanded-panel"]
    assert validate_observation_alignment(
        graph,
        patch,
        trace,
        {2: [{"page_id": "page_1"}]},
    ) == []


def test_observation_alignment_reports_missing_observed_step():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="首页"))
    graph.add_page(Page(id="page_1", name="首页"), "func_1")
    trace = [{"raw_step_index": 1, "current_page": Page(name="未知页面")}]

    issues = validate_observation_alignment(graph, {}, trace, {})

    assert issues == ["missing observation_alignment for source_step 1"]


def test_alignment_reuse_of_same_patch_page_is_normalized_to_create():
    patch = {
        "pages": [{"page_ref": "new_page_1", "source_step": 1}],
        "observation_alignment": [{
            "source_step": 1,
            "page_decision": "reuse",
            "page_ref": "new_page_1",
        }],
    }

    alignments = _normalize_observation_alignment(
        patch,
        FlowGraph(),
        [{"raw_step_index": 1, "current_page": Page(name="新页面")}],
        {},
    )

    assert alignments[0]["page_decision"] == "create"


def test_non_ignorable_observation_with_page_patch_is_normalized_to_create():
    patch = {
        "pages": [{"page_ref": "new_page_1", "source_step": 1}],
        "observation_alignment": [{
            "source_step": 1,
            "page_decision": "ignore",
            "page_ref": "",
        }],
    }

    alignments = _normalize_observation_alignment(
        patch,
        FlowGraph(),
        [{"raw_step_index": 1, "current_page": Page(activity="NovelActivity")}],
        {},
    )

    assert alignments[0] == {
        "source_step": 1,
        "page_decision": "create",
        "page_ref": "new_page_1",
    }


def test_external_interruption_ignore_is_not_forced_to_create():
    patch = {
        "observation_alignment": [{
            "source_step": 1,
            "page_decision": "ignore",
            "page_ref": "",
        }],
    }

    alignments = _normalize_observation_alignment(
        patch,
        FlowGraph(),
        [{
            "raw_step_index": 1,
            "current_page": Page(activity="ExternalActivity"),
            "is_interruption": True,
            "external_context": True,
        }],
        {},
    )

    assert alignments[0]["page_decision"] == "ignore"


def test_new_activity_ignore_still_requires_refinement_without_page_patch():
    graph = FlowGraph()
    trace = [{
        "raw_step_index": 1,
        "current_page": Page(activity="NovelActivity"),
    }]
    patch = {
        "observation_alignment": [{
            "source_step": 1,
            "page_decision": "ignore",
            "page_ref": "",
        }],
    }

    _normalize_observation_alignment(patch, graph, trace, {})

    assert validate_observation_alignment(graph, patch, trace, {}) == [
        "ignore alignment lacks an ignorable observation at source_step 1"
    ]


def test_ffg_subgraph_rolls_back_unresolved_alignment_issue(monkeypatch):
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="首页"))
    graph.add_page(Page(id="page_1", name="首页", activity="HomeActivity"), "func_1")

    class MissingAlignmentLlm:
        class Response:
            content = (
                '{"update_control":{"decision":"finish","reason":"done"},'
                '"functions":[],"pages":[{"page_ref":"new_page_1",'
                '"function_refs":["func_1"],"source_step":1,'
                '"name":"详情页","description":"详情内容"}],'
                '"page_ownership":[],"function_dependencies":[]}'
            )

        def invoke(self, _messages):
            return self.Response()

    page_audit_worker.reset()
    with page_audit_worker._lock:
        page_audit_worker._results.append({
            "page_id": "page_1",
            "fingerprint": "fp",
            "missing_controls": ["右上角更多按钮"],
            "control_merges": [],
            "invalid_operations": [],
        })
    monkeypatch.setattr(ffg_update_subgraph_module, "FFG_UPDATE_MAX_STEPS", 1)
    monkeypatch.setattr(
        "workflow.nodes.ffg_update._submit_page_audit_batch",
        lambda *_args, **_kwargs: 0,
    )
    try:
        subgraph = ffg_update_subgraph_module.create_ffg_update_subgraph(MissingAlignmentLlm())
        result = subgraph.invoke({
            "ffg": graph,
            "current_page": Page(id="page_1", name="首页", activity="HomeActivity"),
            "current_plan": {
                "scenario_type": "intra_test",
                "target_function_id": "func_1",
            },
            "navigation_execution_trace": [],
            "scenario_execution_trace": [{
                "raw_step_index": 1,
                "action_name": "finished",
                "current_page": Page(name="详情页", activity="DetailActivity"),
            }],
            "execution_result": {
                "navigation_status": "skipped",
                "scenario_status": "completed",
            },
            "current_testing_function_id": "func_1",
            "current_testing_function_status": "in_progress",
        })
    finally:
        page_audit_worker.reset()

    assert list(result["ffg"].pages) == ["page_1"]
    assert result["ffg"].get_page("page_1").controls == ["右上角更多按钮"]
    assert result["ffg_update_validation_issues"] == [
        "missing observation_alignment for source_step 1"
    ]


def test_planner_refreshes_stale_page_id_from_live_visual(monkeypatch):
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="消息列表"))
    graph.add_page(Page(
        id="page_1",
        name="消息列表",
        activity="LauncherUI",
        representative_screenshot="home-image",
    ), "func_1")
    graph.add_page(Page(
        id="page_2",
        name="聊天页",
        activity="LauncherUI",
        representative_screenshot="chat-image",
    ), "func_1")
    monkeypatch.setattr(
        scenario_plan_module,
        "capture_current_page",
        lambda _device: Page(
            activity="LauncherUI",
            representative_screenshot="home-image",
        ),
    )

    refreshed = scenario_plan_module._refresh_current_page_observation(
        graph,
        object(),
        Page(id="page_2", activity="LauncherUI", representative_screenshot="chat-image"),
    )

    assert refreshed.id == "page_1"
    assert refreshed.name == "消息列表"


def test_planner_does_not_force_unmatched_visual_to_only_activity_page(monkeypatch):
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="消息列表"))
    graph.add_page(Page(
        id="page_1",
        name="消息列表",
        activity="LauncherUI",
        representative_screenshot="home-image",
    ), "func_1")
    monkeypatch.setattr(
        scenario_plan_module,
        "capture_current_page",
        lambda _device: Page(
            activity="LauncherUI",
            representative_screenshot="unknown-chat-image",
        ),
    )

    refreshed = scenario_plan_module._refresh_current_page_observation(
        graph,
        object(),
        Page(id="page_1", activity="LauncherUI", representative_screenshot="home-image"),
    )

    assert refreshed.id is None


def test_existing_page_merges_changed_state_without_replacing_stable_image():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="任务列表"))
    graph.add_page(Page(
        id="page_1",
        name="任务列表",
        representative_screenshot="image-closed",
        controls=["一直可见的‘排序’按钮"],
        operations=["点击排序按钮"],
    ), "func_1")
    trace = [
        {
            "raw_step_index": 1,
            "action_name": "click",
            "action_description": "点击排序按钮",
            "current_page": Page(representative_screenshot="image-closed"),
        },
        {
            "raw_step_index": 2,
            "action_name": "finished",
            "current_page": Page(representative_screenshot="image-dropdown"),
        },
    ]
    patch = {
        "pages": [{
            "page_ref": "page_1",
            "function_refs": ["func_1"],
            "source_step": 2,
            "name": "任务列表",
            "controls": ["展开排序下拉框后可见的‘按时间’选项"],
        }],
    }

    first = apply_ffg_patch(graph, patch, trace, {"target_function_id": "func_1"})
    second = apply_ffg_patch(graph, patch, trace, {"target_function_id": "func_1"})

    page = graph.get_page("page_1")
    assert page is not None
    assert page.representative_screenshot == "image-closed"
    assert page.evidence_screenshots == ["image-dropdown"]
    assert page.controls == [
        "一直可见的‘排序’按钮",
        "展开排序下拉框后可见的‘按时间’选项",
    ]
    assert page.operations == ["点击排序按钮"]
    assert graph.page_transitions == []
    assert first["validation_issues"] == []
    assert second["validation_issues"] == []
    assert validate_ffg_invariants(graph) == []


def test_new_independent_function_entry_is_removed_from_parent_shell_owner():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="微信会话"))
    graph.add_page(Page(id="page_1", name="聊天页"), "func_1")
    trace = [{
        "raw_step_index": 2,
        "current_page": Page(name="表情商店", activity="EmojiStoreActivity"),
    }]

    result = apply_ffg_patch(graph, {
        "functions": [{
            "function_ref": "new_func_1",
            "source_step": 2,
            "name": "表情包浏览与管理",
            "description": "浏览、添加和预览表情包",
            "coverage_percent": 10,
        }],
        "pages": [{
            "page_ref": "new_page_1",
            "function_refs": ["new_func_1", "func_1"],
            "source_step": 2,
            "name": "表情商店",
            "description": "浏览表情包",
        }],
    }, trace, {"target_function_id": "func_1"})

    new_function_id = result["function_refs"]["new_func_1"]
    new_page_id = result["page_refs"]["new_page_1"]
    assert graph.get_node(new_function_id).name == "表情包浏览与管理"
    assert graph.get_page_owners(new_page_id) == [new_function_id]
    assert graph.get_node(new_function_id).page_ids == [new_page_id]
    assert graph.get_node(new_function_id).coverage_percent == 0
    assert result["validation_issues"] == []


def test_named_dynamic_instances_are_not_auto_merged_by_app_specific_vocabulary():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="微信会话列表浏览与进入"))
    graph.add_node(FunctionNode(
        id="func_2",
        name="文件传输助手会话消息浏览与收发",
        description="浏览会话消息并发送内容",
    ))
    graph.add_page(Page(id="page_1", name="微信会话列表"), "func_1")
    graph.add_page(Page(
        id="page_2",
        name="文件传输助手聊天页面",
        activity=".ui.LauncherUI",
    ), "func_2")
    trace = [{
        "raw_step_index": 2,
        "current_page": Page(
            name="联系人聊天页面",
            activity=".ui.LauncherUI",
        ),
    }]

    result = apply_ffg_patch(graph, {
        "functions": [{
            "function_ref": "new_func_1",
            "source_step": 2,
            "name": "联系人会话消息浏览与收发",
            "description": "浏览联系人消息并发送内容",
        }],
        "pages": [{
            "page_ref": "new_page_1",
            "function_refs": ["func_1", "new_func_1"],
            "source_step": 2,
            "name": "联系人聊天页面",
            "description": "联系人消息收发页面",
        }],
    }, trace, {"target_function_id": "func_1"})

    new_function_id = result["function_refs"]["new_func_1"]
    new_page_id = result["page_refs"]["new_page_1"]
    assert new_function_id != "func_2"
    assert new_page_id != "page_2"
    assert list(graph.functions) == ["func_1", "func_2", new_function_id]
    assert list(graph.pages) == ["page_1", "page_2", new_page_id]
    assert graph.get_page_owners("page_2") == ["func_2"]
    assert graph.get_page_owners(new_page_id) == [new_function_id]
    assert result["validation_issues"] == []


def test_dynamic_chat_pages_stay_separate_when_functions_are_already_distinct():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="文件传输助手会话"))
    graph.add_node(FunctionNode(id="func_2", name="测试群会话"))
    graph.add_page(Page(
        id="page_1",
        name="文件传输助手聊天页面",
        activity=".ui.LauncherUI",
    ), "func_1")
    trace = [{
        "raw_step_index": 1,
        "current_page": Page(name="联系人聊天页面", activity=".ui.LauncherUI"),
    }]

    result = apply_ffg_patch(graph, {
        "pages": [{
            "page_ref": "new_page_1",
            "function_refs": ["func_2"],
            "source_step": 1,
            "name": "联系人聊天页面",
        }],
    }, trace, {"target_function_id": "func_2"})

    new_page_id = result["page_refs"]["new_page_1"]
    assert new_page_id != "page_1"
    assert graph.get_page_owners("page_1") == ["func_1"]
    assert graph.get_page_owners(new_page_id) == ["func_2"]


def test_page_ownership_patch_can_split_page_from_coarse_function():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="微信会话"))
    graph.add_page(Page(id="page_1", name="会话列表"), "func_1")
    graph.add_page(Page(id="page_2", name="聊天记录查找"), "func_1")
    trace = [{
        "raw_step_index": 3,
        "current_page": Page(id="page_2", name="聊天记录查找"),
    }]

    result = apply_ffg_patch(graph, {
        "functions": [{
            "function_ref": "new_func_1",
            "source_step": 3,
            "name": "聊天记录搜索",
            "description": "按关键词或内容类型查找聊天记录",
        }],
        "page_ownership": [{
            "page_ref": "page_2",
            "add_function_refs": ["new_func_1"],
            "remove_function_refs": ["func_1"],
            "source_step": 3,
            "reason": "直接观察到独立搜索闭环",
        }],
    }, trace, {"target_function_id": "func_1"})

    new_function_id = result["function_refs"]["new_func_1"]
    assert graph.get_page_owners("page_2") == [new_function_id]
    assert graph.get_node("func_1").page_ids == ["page_1"]
    assert graph.get_node(new_function_id).page_ids == ["page_2"]
    assert graph.get_node(new_function_id).entry_page_ids == []
    assert result["validation_issues"] == []


def test_oversized_function_can_be_atomically_replaced_from_persisted_pages():
    graph = FlowGraph()
    graph.add_node(FunctionNode(
        id="func_1",
        name="个人中心与设置",
        scenarios=[
            "点击设置进入设置页",
            "点击账号安全查看设备",
            "点击收藏进入收藏页",
            "点击卡包查看卡券",
        ],
        test_status="in_progress",
    ))
    for index, name in enumerate((
        "个人中心页",
        "设置页",
        "账号安全页",
        "收藏页",
        "卡包页",
    ), start=1):
        graph.add_page(Page(
            id=f"page_{index}",
            name=name,
            activity=f"Activity{index}",
            operations=[f"点击{name}入口"],
        ), "func_1")
    graph.add_node(FunctionNode(id="func_2", name="首页"))
    graph.add_page(Page(id="page_0", name="微信首页"), "func_2")
    graph.add_entry_page(graph.get_page("page_0"), "func_1")

    result = apply_ffg_patch(graph, {
        "functions": [{
            "function_ref": "new_func_1",
            "name": "账号与系统设置",
            "description": "管理账号安全和系统设置",
        }, {
            "function_ref": "new_func_2",
            "name": "个人内容与资产",
            "description": "查看收藏、卡包等个人内容",
        }],
        "function_restructures": [{
            "source_function_ref": "func_1",
            "replacements": [{
                "function_ref": "new_func_1",
                "page_refs": ["page_1", "page_2", "page_3"],
                "entry_page_refs": ["page_0"],
                "scenario_refs": ["scenario_1", "scenario_2"],
            }, {
                "function_ref": "new_func_2",
                "page_refs": ["page_4", "page_5"],
                "entry_page_refs": ["page_0"],
                "scenario_refs": ["scenario_3", "scenario_4"],
            }],
            "retire_source": True,
            "reason": "设置和个人资产已形成独立闭环",
        }],
    }, [{
        "raw_step_index": 1,
        "current_page": Page(id="page_1", name="个人中心页"),
    }], {"target_function_id": "func_1"})

    settings_id = result["function_refs"]["new_func_1"]
    assets_id = result["function_refs"]["new_func_2"]
    assert graph.get_node("func_1") is None
    assert graph.get_node(settings_id).coverage_page_ids == [
        "page_1", "page_2", "page_3",
    ]
    assert graph.get_node(assets_id).coverage_page_ids == ["page_4", "page_5"]
    assert graph.get_node(assets_id).entry_page_ids == ["page_0"]
    assert graph.get_node(settings_id).scenarios == [
        "点击设置进入设置页",
        "点击账号安全查看设备",
    ]
    assert graph.get_node(assets_id).scenarios == [
        "点击收藏进入收藏页",
        "点击卡包查看卡券",
    ]
    assert result["function_replacements"] == {
        "func_1": [settings_id, assets_id],
    }
    assert result["validation_issues"] == []
    assert validate_ffg_invariants(graph) == []


def test_full_function_split_rolls_back_when_scenario_evidence_would_be_lost():
    graph = FlowGraph()
    graph.add_node(FunctionNode(
        id="func_1",
        name="粗粒度功能",
        scenarios=["场景一", "场景二"],
    ))
    graph.add_page(Page(id="page_1", name="页面一"), "func_1")
    graph.add_page(Page(id="page_2", name="页面二"), "func_1")

    result = apply_ffg_patch(graph, {
        "functions": [
            {"function_ref": "new_func_1", "name": "功能一", "description": "页面一"},
            {"function_ref": "new_func_2", "name": "功能二", "description": "页面二"},
        ],
        "function_restructures": [{
            "source_function_ref": "func_1",
            "replacements": [{
                "function_ref": "new_func_1",
                "page_refs": ["page_1"],
                "scenario_refs": ["scenario_1"],
            }, {
                "function_ref": "new_func_2",
                "page_refs": ["page_2"],
                "scenario_refs": [],
            }],
            "retire_source": True,
        }],
    }, [{"raw_step_index": 1, "current_page": Page(id="page_1")}])

    assert list(graph.functions) == ["func_1"]
    assert graph.get_node("func_1").page_ids == ["page_1", "page_2"]
    assert any("scenario_2" in issue for issue in result["validation_issues"])


def test_existing_functions_can_be_merged_with_pages_scenarios_and_dependencies():
    graph = FlowGraph()
    graph.add_node(FunctionNode(
        id="func_1",
        name="收藏入口浏览",
        scenarios=["从个人中心进入收藏"],
        resource_variables=["收藏内容"],
        test_status="completed",
    ))
    graph.add_node(FunctionNode(
        id="func_2",
        name="收藏列表查看",
        scenarios=["查看收藏列表"],
        resource_variables=["收藏内容"],
        test_status="in_progress",
    ))
    graph.add_node(FunctionNode(id="func_3", name="个人中心"))
    graph.add_page(Page(id="page_1", name="个人中心"), "func_3")
    graph.add_page(Page(
        id="page_2",
        name="收藏入口页",
        operations=["点击收藏入口"],
    ), "func_1")
    graph.add_entry_page(graph.get_page("page_1"), "func_1")
    graph.add_page(Page(
        id="page_3",
        name="收藏列表页",
        operations=["点击收藏条目"],
    ), "func_2")
    graph.add_function_dependency(FunctionDependency(
        src_function_id="func_1",
        dst_function_id="func_2",
        resource_variable="收藏内容",
    ))

    result = apply_ffg_patch(graph, {
        "function_merges": [{
            "source_function_refs": ["func_1", "func_2"],
            "target_function_ref": "func_1",
            "reason": "两段已确认属于同一收藏目标",
        }],
    }, [{"raw_step_index": 1, "current_page": Page(id="page_2")}])

    assert graph.get_node("func_2") is None
    merged = graph.get_node("func_1")
    assert merged is not None
    assert merged.page_ids == ["page_2", "page_1", "page_3"]
    assert merged.entry_page_ids == ["page_1"]
    assert merged.scenarios == ["从个人中心进入收藏", "查看收藏列表"]
    assert merged.test_status == "in_progress"
    assert graph.function_dependencies == []
    assert result["function_replacements"] == {"func_2": ["func_1"]}
    assert result["validation_issues"] == []
    assert validate_ffg_invariants(graph) == []


def test_granularity_review_is_spaced_as_function_grows(monkeypatch):
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="个人中心与设置"))
    for index in range(1, 13):
        graph.add_page(Page(
            id=f"page_{index}",
            name=f"子页面{index}",
            activity=f"Activity{index}",
        ), "func_1")

    monkeypatch.setattr(
        "workflow.nodes.ffg_update.FFG_UPDATE_GRANULARITY_MIN_PAGES", 6
    )
    first_id, first_metrics = _select_granularity_review_function(
        graph,
        {"target_function_id": "func_1"},
        {},
    )
    second_id, second_metrics = _select_granularity_review_function(
        graph,
        {"target_function_id": "func_1"},
        {"func_1": 1},
    )
    exhausted_id, _ = _select_granularity_review_function(
        graph,
        {"target_function_id": "func_1"},
        {"func_1": 2},
    )

    assert first_id == second_id == "func_1"
    assert first_metrics["page_threshold"] == 6
    assert second_metrics["page_threshold"] == 12
    assert exhausted_id == ""


def test_visible_business_entry_creates_function_with_shared_entry_page():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="会话消息浏览与收发"))
    graph.add_page(Page(
        id="page_1",
        name="会话聊天页面",
        controls=["相册入口（扩展功能面板展开时可见）"],
    ), "func_1")
    trace = [{
        "raw_step_index": 2,
        "current_page": Page(
            id="page_1",
            name="会话聊天页面",
            controls=["相册入口（扩展功能面板展开时可见）"],
        ),
    }]

    result = apply_ffg_patch(graph, {
        "functions": [{
            "function_ref": "new_func_1",
            "source_step": 2,
            "name": "从相册选择并发送媒体",
            "description": "从聊天扩展面板中的相册入口进入，具体流程待执行确认。",
            "coverage_percent": 0,
        }],
        "page_ownership": [{
            "page_ref": "page_1",
            "add_function_refs": ["new_func_1"],
            "remove_function_refs": [],
            "source_step": 2,
            "reason": "页面直接显示名称明确的相册业务入口",
        }],
    }, trace, {"target_function_id": "func_1"})

    new_function_id = result["function_refs"]["new_func_1"]
    assert graph.get_page_owners("page_1") == ["func_1", new_function_id]
    assert graph.get_node(new_function_id).page_ids == ["page_1"]
    assert graph.get_node(new_function_id).entry_page_ids == ["page_1"]
    assert graph.get_node(new_function_id).coverage_page_ids == []
    assert graph.get_node(new_function_id).test_status == "not_started"
    assert graph.get_node(new_function_id).coverage_percent == 0
    assert list(graph.pages) == ["page_1"]
    assert graph.page_transitions == []
    assert result["validation_issues"] == []


def test_existing_function_name_and_description_update_require_direct_evidence():
    graph = FlowGraph()
    graph.add_node(FunctionNode(
        id="func_1",
        name="相册相关功能",
        description="从相册入口发现，具体流程待确认。",
    ))
    graph.add_page(Page(id="page_1", name="聊天页面"), "func_1")

    apply_ffg_patch(graph, {
        "functions": [{
            "function_ref": "func_1",
            "source_step": 3,
            "name": "从相册选择并发送图片或视频",
            "description": "选择设备中的图片或视频并发送到当前会话。",
        }],
    }, [{
        "raw_step_index": 3,
        "current_page": Page(id="page_1", name="聊天页面"),
    }])

    node = graph.get_node("func_1")
    assert node.name == "从相册选择并发送图片或视频"
    assert node.description == "选择设备中的图片或视频并发送到当前会话。"

    apply_ffg_patch(graph, {
        "functions": [{
            "function_ref": "func_1",
            "source_step": 99,
            "name": "没有证据的重命名",
            "description": "不应写入。",
        }],
    }, [{
        "raw_step_index": 4,
        "current_page": Page(id="page_1", name="聊天页面"),
    }])

    assert node.name == "从相册选择并发送图片或视频"
    assert node.description == "选择设备中的图片或视频并发送到当前会话。"


def test_ffg_patch_cannot_write_function_terminal_status():
    graph = FlowGraph()
    graph.add_node(FunctionNode(
        id="func_1",
        name="从相册选择并发送媒体",
        test_status="in_progress",
    ))
    graph.add_page(Page(id="page_1", name="会话聊天页面"), "func_1")

    apply_ffg_patch(graph, {
        "functions": [{
            "function_ref": "func_1",
            "source_step": 2,
            "test_status": "unavailable",
        }],
    }, [{
        "raw_step_index": 2,
        "current_page": Page(id="page_1", name="会话聊天页面"),
    }])

    assert graph.get_node_test_status("func_1") == "in_progress"

    apply_ffg_patch(graph, {
        "functions": [{
            "function_ref": "func_1",
            "source_step": 99,
            "test_status": "blocked",
        }],
    }, [{
        "raw_step_index": 3,
        "current_page": Page(id="page_1", name="会话聊天页面"),
    }])

    assert graph.get_node_test_status("func_1") == "in_progress"


def test_existing_function_resource_variables_are_merged_from_direct_evidence():
    graph = FlowGraph()
    graph.add_node(FunctionNode(
        id="func_1",
        name="文章阅读",
        resource_variables=["阅读进度"],
    ))
    graph.add_page(Page(id="page_1", name="文章详情"), "func_1")
    trace = [{
        "raw_step_index": 2,
        "current_page": Page(id="page_1", name="文章详情"),
    }]

    first = apply_ffg_patch(graph, {
        "functions": [{
            "function_ref": "func_1",
            "source_step": 2,
            "resource_variables": ["收藏内容", "收藏内容"],
        }],
    }, trace)
    second = apply_ffg_patch(graph, {
        "functions": [{
            "function_ref": "func_1",
            "source_step": 2,
            "resource_variables": [],
        }],
    }, trace)

    assert graph.get_node("func_1").resource_variables == ["阅读进度", "收藏内容"]
    assert first["validation_issues"] == []
    assert second["validation_issues"] == []


def test_existing_function_resource_update_requires_direct_trace_step():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="文章阅读"))

    apply_ffg_patch(graph, {
        "functions": [{
            "function_ref": "func_1",
            "source_step": 9,
            "resource_variables": ["收藏内容"],
        }],
    }, [{"raw_step_index": 1, "current_page": Page(name="文章详情")}])

    assert graph.get_node("func_1").resource_variables == []


def test_function_dependency_syncs_resource_to_both_functions():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="文章阅读"))
    graph.add_node(FunctionNode(id="func_2", name="收藏列表"))
    graph.add_page(Page(id="page_1", name="文章详情"), "func_1")
    graph.add_page(Page(id="page_2", name="收藏列表"), "func_2")

    result = apply_ffg_patch(graph, {
        "function_dependencies": [{
            "src_function_ref": "func_1",
            "dst_function_ref": "func_2",
            "resource_variable": "收藏内容",
            "description": "收藏后在收藏列表可见",
        }],
    }, [{"raw_step_index": 1, "current_page": Page(name="收藏列表")}])

    assert graph.get_node("func_1").resource_variables == ["收藏内容"]
    assert graph.get_node("func_2").resource_variables == ["收藏内容"]
    assert len(graph.function_dependencies) == 1
    assert result["validation_issues"] == []

    graph.get_node("func_1").resource_variables.clear()
    graph.get_node("func_2").resource_variables.clear()
    graph.add_function_dependency(graph.function_dependencies[0])

    assert graph.get_node("func_1").resource_variables == ["收藏内容"]
    assert graph.get_node("func_2").resource_variables == ["收藏内容"]
    assert len(graph.function_dependencies) == 1


def test_page_control_union_preserves_similar_but_distinct_dropdown_options():
    page = Page(controls=[
        "展开排序下拉框后可见的‘按时间’选项",
        "展开排序下拉框后可见的‘按名称’选项",
    ])

    assert page.controls == [
        "展开排序下拉框后可见的‘按时间’选项",
        "展开排序下拉框后可见的‘按名称’选项",
    ]


def test_page_round_trip_has_only_screenshot_and_control_lists():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="主页"))
    graph.add_page(Page(
        id="page_1",
        representative_screenshot="image-a",
        evidence_screenshots=["image-b"],
        controls=["可点击的‘新增’按钮"],
    ), "func_1")

    data = graph.to_dict()
    page_data = data["pages"][0]

    assert page_data["representative_screenshot"] == "image-a"
    assert page_data["evidence_screenshots"] == ["image-b"]
    assert page_data["controls"] == ["可点击的‘新增’按钮"]
    assert "screenshot" not in page_data
    assert "status" not in page_data
    assert "alias_to" not in page_data
    assert "correction_log" not in data
    restored = FlowGraph.from_dict(data)
    assert restored.get_page("page_1").representative_screenshot == "image-a"
    assert restored.get_page("page_1").evidence_screenshots == ["image-b"]


def test_entry_page_role_survives_flow_graph_round_trip():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="主页"))
    graph.add_node(FunctionNode(id="func_2", name="相册选择"))
    page = Page(id="page_1", name="主页", controls=["相册入口"])
    graph.add_page(page, "func_1")
    graph.add_entry_page(page, "func_2")

    restored = FlowGraph.from_dict(graph.to_dict())

    assert restored.get_node("func_2").page_ids == ["page_1"]
    assert restored.get_node("func_2").entry_page_ids == ["page_1"]
    assert restored.get_node("func_2").coverage_page_ids == []


def test_page_owners_are_derived_and_global_transitions_can_cross_functions():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="列表"))
    graph.add_node(FunctionNode(id="func_2", name="详情"))
    shared = Page(id="page_shared", name="共享入口")
    graph.add_page(shared, "func_1")
    graph.add_page(shared, "func_2")
    graph.add_page(Page(id="page_detail", name="详情页"), "func_2")
    graph.add_page_transition(PageTransition(
        src_page_id="page_shared",
        dst_page_id="page_detail",
        operation="打开详情",
    ))

    assert graph.page_owners == {
        "page_shared": ["func_1", "func_2"],
        "page_detail": ["func_2"],
    }
    assert graph.get_node("func_2").page_ids == ["page_shared", "page_detail"]
    assert graph.page_transitions[0].dst_page_id == "page_detail"
    assert "page_owners" not in graph.to_dict()


def test_flow_graph_load_rejects_removed_legacy_structure(tmp_path):
    legacy = {
        "function_nodes": [{
            "id": "func_1",
            "name": "主页",
            "entry_page_id": "page_1",
            "test_depth": 7,
            "page_ids": ["page_1", "page_2"],
            "page_transitions": [{
                "src_page_id": "page_1",
                "dst_page_id": "page_2",
                "operation": "打开详情",
            }],
            "test_scenarios": [{"description": "浏览详情"}],
            "attempted_scenarios": [{"description": "点击筛选"}],
        }],
        "page_registry": [{
            "id": "page_1",
            "screenshots": ["representative", "evidence"],
        }, {
            "id": "page_2",
        }],
        "page_owners": {"page_1": ["func_1"], "page_2": ["func_1"]},
    }

    (tmp_path / "ffg.json").write_text(
        json.dumps(legacy, ensure_ascii=False),
        encoding="utf-8",
    )

    try:
        FlowGraph.load(str(tmp_path))
    except ValueError as exc:
        assert "functions" in str(exc)
    else:
        raise AssertionError("旧 FFG 结构不应被静默加载")


def test_function_coverage_percent_is_clamped_and_serialized():
    below = FunctionNode(id="func_1", coverage_percent=-1)
    above = FunctionNode(id="func_2", coverage_percent=120)

    assert below.coverage_percent == 0
    assert above.coverage_percent == 100
    assert below.to_dict()["coverage_percent"] == 0
    assert above.to_dict()["coverage_percent"] == 100


def test_llm_function_coverage_is_ignored_and_program_derived():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="搜索", coverage_percent=10))
    graph.add_page(Page(id="page_1", name="搜索页"), "func_1")
    trace = [{
        "raw_step_index": 2,
        "current_page": Page(id="page_1", name="搜索页"),
    }]

    apply_ffg_patch(graph, {
        "functions": [{
            "function_ref": "func_1",
            "source_step": 2,
            "coverage_percent": 35,
        }],
    }, trace)
    assert graph.get_node("func_1").coverage_percent == 0

    apply_ffg_patch(graph, {
        "functions": [{
            "function_ref": "func_1",
            "source_step": 99,
            "coverage_percent": 90,
        }],
    }, trace)
    assert graph.get_node("func_1").coverage_percent == 0


def test_controls_use_simple_exact_dedup_while_operations_remain_deduplicated():
    page = Page(
        controls=[
            "文件传输助手会话条目",
            "测试会话条目",
            "搜索按钮",
            "公众号入口条目",
            "服务号入口条目",
        ],
        operations=[
            "点击底部通讯录标签",
            "点击底部导航栏的通讯录按钮",
            "尝试点击搜索按钮失败：控件无响应",
        ],
    )

    assert page.controls == [
        "文件传输助手会话条目",
        "测试会话条目",
        "搜索按钮",
        "公众号入口条目",
        "服务号入口条目",
    ]
    assert page.operations == [
        "点击底部通讯录标签",
        "尝试点击搜索按钮失败：控件无响应",
    ]
    assert operation_family_key(page.operations[0]) == "click:通讯录"


def test_picker_instances_and_visual_modifiers_share_operation_families():
    assert control_family_key("表情分类与表情项") == "表情选择项"
    assert operation_family_key("点击所有表情分类下第一个微笑表情") == "click:表情选择项"
    assert operation_family_key("点击小鸡表情分类中的“拜托”表情") == "click:表情选择项"
    assert operation_family_key("点击绿色的发送按钮") == "click:发送"


def test_instance_conversation_operations_merge_into_one_family():
    """实例级会话表述与"会话条目"同族，重复实例操作在入库时归并为一条。"""
    assert control_family_key("张三的会话") == "会话条目"
    assert control_family_key("某个会话") == "会话条目"
    assert control_family_key("会话列表") == "会话条目"
    # 非实例表述不受影响
    assert control_family_key("会话标题") != "会话条目"
    assert (
        operation_family_key("点击张三的会话")
        == operation_family_key("点击李四的会话")
        == "click:会话条目"
    )
    page = Page(id="p", controls=["会话条目"])
    page.add_operation("点击张三的会话")
    page.add_operation("点击李四的会话")
    page.add_operation("点击某个会话")
    assert page.operations == ["点击张三的会话"]


def test_function_coverage_counts_all_page_controls_without_name_filtering():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="发送表情消息"))
    graph.add_page(Page(
        id="page_1",
        controls=[
            "返回按钮",
            "会话标题",
            "文本消息",
            "图片消息",
            "表情消息",
            "消息文本输入框",
            "表情按钮",
            "表情选择面板",
            "表情分类与表情项",
            "发送按钮",
        ],
        operations=[
            "点击表情按钮",
            "点击小鸡表情分类中的“拜托”表情",
            "点击绿色的发送按钮",
        ],
    ), "func_1")

    details = graph.function_coverage_details("func_1")

    # 命中控件口径："点击表情按钮"命中表情按钮；"点击小鸡表情分类中的拜托
    # 表情"命中表情选择面板与表情分类与表情项（同族）；"点击绿色的发送
    # 按钮"命中发送按钮。"表情消息"虽是"表情"的子串扩展，但 family 不同
    # （表情消息 vs 表情），不再计入。
    assert details["total_controls"] == 10
    assert details["executed_operations"] == 4
    assert details["coverage_percent"] == 45


def test_failed_operation_counts_as_executed_but_not_successful():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="搜索"))
    graph.add_page(Page(
        id="page_1",
        controls=["搜索按钮"],
        operations=["尝试点击搜索按钮失败：控件无响应"],
    ), "func_1")
    graph.recompute_all_coverages()

    details = graph.function_coverage_details("func_1")
    assert details["executed_operations"] == 1
    assert details["coverage_percent"] == 100
    assert not graph.function_has_successful_business_operation("func_1")


def test_many_recorded_operations_do_not_drop_to_zero_from_function_name_filtering():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="微信聊天消息收发"))
    graph.add_page(Page(
        id="page_1",
        controls=[f"控件{i}" for i in range(22)],
        operations=[f"点击控件{i}" for i in range(12)],
    ), "func_1")

    details = graph.function_coverage_details("func_1")

    assert details["total_controls"] == 22
    assert details["executed_operations"] == 12
    assert details["coverage_percent"] == 70


def test_adjusted_control_coverage_requires_all_for_small_functions():
    assert adjusted_control_coverage(3, 2) == 67
    assert adjusted_control_coverage(3, 3) == 100
    assert adjusted_control_coverage(8, 6) == 84
    assert adjusted_control_coverage(8, 7) == 94
    assert adjusted_control_coverage(20, 14) == 90
    assert adjusted_control_coverage(20, 20) == 100
