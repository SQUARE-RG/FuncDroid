from langchain_core.messages import HumanMessage, SystemMessage

from model.flow_graph import Page
from workflow.graph import build_graph
from workflow import trace_defect_detection


def test_workflow_contains_only_three_business_nodes():
    compiled = build_graph(object(), object(), object())

    node_names = set(compiled.get_graph().nodes) - {"__start__", "__end__"}

    assert node_names == {"scenario_plan", "scenario_exec", "ffg_update"}


def test_navigation_crash_is_detected_from_execution_trace_without_llm():
    page = Page(id="page-1", name="首页")
    trace = [{
        "action_name": "click",
        "trace_phase": "navigation",
        "current_page": page,
        "caused_target_crash": True,
    }]
    state = {
        "package_name": "com.example.app",
        "current_navigation_scenario": "进入设置页",
        "navigation_execution_trace": trace,
        "scenario_execution_trace": [],
        "execution_result": {
            "navigation_status": "target_app_crashed",
            "navigation_error": "目标应用崩溃",
            "scenario_status": "skipped_due_target_app_crashed",
            "crash_logcat": (
                "FATAL EXCEPTION: main\n"
                "Process: com.example.app, PID: 1234\n"
                "java.lang.IllegalStateException: boom"
            ),
        },
        "detected_bugs": [],
        "bug_trace_records": {},
        "current_plan": {"scenario_type": "intra_test"},
    }

    result = trace_defect_detection.analyze_execution_for_bugs(object(), state)

    assert len(result["detected_bugs"]) == 1
    detected = result["detected_bugs"][0]
    assert detected["type"] == "crash"
    assert detected["source"] == "runtime_monitor"
    assert result["bug_trace_records"][detected["bug_id"]] == trace


def test_llm_receives_navigation_and_test_phases_as_one_trace(monkeypatch):
    captured = {}

    def fake_invoke(_llm, messages, **_kwargs):
        captured["content"] = messages[1].content
        return {"bugs": []}

    monkeypatch.setattr(trace_defect_detection, "invoke_llm_json_with_retry", fake_invoke)
    state = {
        "current_navigation_scenario": "进入编辑页",
        "current_test_scenario": "保存修改",
        "navigation_execution_trace": [{
            "action_name": "click",
            "trace_phase": "navigation",
            "current_page": Page(id="page-1"),
        }],
        "scenario_execution_trace": [{
            "action_name": "click",
            "trace_phase": "test",
            "current_page": Page(id="page-2"),
        }],
        "execution_result": {
            "navigation_status": "completed",
            "scenario_status": "completed",
        },
        "detected_bugs": [],
        "bug_trace_records": {},
        "current_plan": {"scenario_type": "intra_test"},
    }

    result = trace_defect_detection.analyze_execution_for_bugs(object(), state)

    step_text = "\n".join(
        item["text"]
        for item in captured["content"]
        if item.get("type") == "text"
    )
    assert "phase=navigation" in step_text
    assert "phase=test" in step_text
    assert result == {"detected_bugs": [], "bug_trace_records": {}}


def test_stalled_trace_still_receives_semantic_bug_review(monkeypatch):
    calls = []

    def fake_invoke(_llm, _messages, **_kwargs):
        calls.append(1)
        return {"bugs": []}

    monkeypatch.setattr(trace_defect_detection, "invoke_llm_json_with_retry", fake_invoke)
    state = {
        "current_test_scenario": "点击保存并确认结果",
        "navigation_execution_trace": [],
        "scenario_execution_trace": [{
            "action_name": "click",
            "thought": "已点击保存，但页面持续显示错误提示",
            "trace_phase": "test",
            "current_page": Page(id="page-1"),
        }],
        "execution_result": {"scenario_status": "stalled"},
        "detected_bugs": [],
        "bug_trace_records": {},
        "current_plan": {"scenario_type": "intra_test"},
    }

    result = trace_defect_detection.analyze_execution_for_bugs(object(), state)

    assert calls == [1]
    assert result == {"detected_bugs": [], "bug_trace_records": {}}


def test_bug_detection_sends_full_trace_with_all_screenshots_independently(monkeypatch):
    """异步/独立线程设计要求：每次检测都基于完整轨迹，不复用执行模型的对话上下文。"""
    captured = {}
    execution_model = object()

    def fake_invoke(llm, messages, **_kwargs):
        captured["llm"] = llm
        captured["messages"] = messages
        return {"bugs": []}

    monkeypatch.setattr(trace_defect_detection, "invoke_llm_json_with_retry", fake_invoke)
    state = {
        "current_navigation_scenario": "进入编辑页",
        "current_test_scenario": "保存修改并确认结果",
        "navigation_execution_trace": [{
            "action_name": "click",
            "trace_phase": "navigation",
            "current_page": Page(id="page-1", name="首页", representative_screenshot="early-image"),
        }],
        "scenario_execution_trace": [{
            "action_name": "finished",
            "thought": "保存后结果已显示",
            "trace_phase": "test",
            "current_page": Page(id="page-2", name="结果页", representative_screenshot="final-image"),
        }],
        "execution_result": {
            "navigation_status": "completed",
            "scenario_status": "completed",
        },
        "detected_bugs": [{
            "bug_id": "BUG-007",
            "type": "logic_error",
            "evidence_category": "persisted_result_error",
            "target_function_id": "func_1",
            "description": "已有问题",
            "expected": "保存后显示新值",
            "actual": "保存后仍显示旧值",
            "confidence": "high",
        }],
        "bug_trace_records": {},
        "current_plan": {
            "scenario_type": "intra_test",
            "target_function_id": "func_1",
        },
    }

    result = trace_defect_detection.analyze_execution_for_bugs(execution_model, state)

    messages = captured["messages"]
    assert captured["llm"] is execution_model
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    review_content = messages[1].content
    assert isinstance(review_content, list)
    header_text = review_content[0]["text"]
    assert "【完整执行轨迹缺陷复核】" in header_text
    assert "保存修改并确认结果" in header_text
    assert "测试场景文本是测试意图，不是产品承诺" in header_text
    assert '"existing_bug_summaries"' in header_text
    assert '"bug_id": "BUG-007"' in header_text
    assert '"description": "已有问题"' in header_text
    assert '"actual": "保存后仍显示旧值"' in header_text
    assert "与其中任一缺陷重复时不要再次输出" in header_text
    step_text = "\n".join(
        block["text"]
        for block in review_content
        if block.get("type") == "text"
    )
    assert "phase=navigation" in step_text
    assert "phase=test" in step_text
    images = [block for block in review_content if block.get("type") == "image_url"]
    # 两个轨迹步骤都带截图，独立发送、不采样、不省略。
    assert len(images) == 2
    assert result["detected_bugs"][0]["description"] == "已有问题"


def test_llm_bug_without_direct_evidence_category_is_rejected(monkeypatch):
    def fake_invoke(_llm, _messages, **_kwargs):
        return {
            "bugs": [{
                "type": "logic_error",
                "severity": "high",
                "confidence": "high",
                "description": "按钮标签在不同步骤发生变化",
                "expected": "标签保持不变",
                "actual": "标签随页面状态变化",
            }]
        }

    monkeypatch.setattr(trace_defect_detection, "invoke_llm_json_with_retry", fake_invoke)
    state = {
        "current_test_scenario": "执行计算",
        "navigation_execution_trace": [],
        "scenario_execution_trace": [{
            "action_name": "click",
            "trace_phase": "test",
            "current_page": Page(id="page-1"),
        }],
        "execution_result": {"scenario_status": "completed"},
        "detected_bugs": [],
        "bug_trace_records": {},
        "current_plan": {"scenario_type": "intra_test"},
    }

    result = trace_defect_detection.analyze_execution_for_bugs(object(), state)

    assert result == {"detected_bugs": [], "bug_trace_records": {}}


def test_speculative_navigation_and_spatial_association_bugs_are_rejected():
    navigation_bug = {
        "evidence_category": "single_action_resource_mutation",
        "description": "点击编辑图标后未触发预期跳转，反而弹出备注对话框",
        "expected": "页面应跳转到编辑页面",
        "actual": "页面弹出备注对话框",
    }
    navigation_scenario = "点击铅笔图标，观察页面是否跳转到编辑页面。"
    association_bug = {
        "evidence_category": "persisted_result_error",
        "description": "备注被错误关联到当前活动，而非目标列表项",
        "expected": "备注应关联到Woodworking活动",
        "actual": "备注显示在Cinema活动下",
    }
    association_scenario = "点击Woodworking活动旁边的铅笔图标，输入备注并保存。"

    assert trace_defect_detection._bug_uses_speculative_or_automation_expectation(
        navigation_bug,
        navigation_scenario,
    ) is True
    assert trace_defect_detection._bug_uses_speculative_or_automation_expectation(
        association_bug,
        association_scenario,
    ) is True


def test_unverified_boundary_rules_and_failed_automation_input_are_rejected():
    required_bug = {
        "evidence_category": "persisted_result_error",
        "description": "空标题未执行必填校验",
        "expected": "必填字段为空时应当阻止保存",
        "actual": "空标题被保存",
    }
    empty_scenario = "不输入任何文本到Activity Title输入框，点击保存并观察是否允许保存。"
    failed_input_bug = {
        "evidence_category": "simultaneous_state_contradiction",
        "description": "输入框为空时显示名称重复提示",
        "expected": "输入框应显示特殊字符",
        "actual": "输入框仍为空并显示重复提示",
    }
    input_scenario = "在输入框中输入文本 '!@#$%^&*'，点击保存。"

    assert trace_defect_detection._bug_uses_speculative_or_automation_expectation(
        required_bug,
        empty_scenario,
    ) is True
    assert trace_defect_detection._bug_uses_speculative_or_automation_expectation(
        failed_input_bug,
        input_scenario,
    ) is True


def test_direct_persisted_result_error_is_not_filtered():
    bug = {
        "evidence_category": "persisted_result_error",
        "description": "保存Task A后结果页显示Task B",
        "expected": "保存后的结果页显示Task A",
        "actual": "结果页显示Task B",
    }
    scenario = "在标题输入框中输入文本 'Task A'，点击保存并观察结果页。"

    assert trace_defect_detection._bug_uses_speculative_or_automation_expectation(
        bug,
        scenario,
    ) is False
