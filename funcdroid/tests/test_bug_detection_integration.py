# Multilingual UI and model-response fixtures use Unicode escapes to preserve their exact text.
from langchain_core.messages import HumanMessage, SystemMessage

from model.flow_graph import Page
from workflow.graph import build_graph
from workflow import trace_defect_detection


def test_workflow_contains_only_three_business_nodes():
    compiled = build_graph(object(), object(), object())

    node_names = set(compiled.get_graph().nodes) - {"__start__", "__end__"}

    assert node_names == {"scenario_plan", "scenario_exec", "ffg_update"}


def test_navigation_crash_is_detected_from_execution_trace_without_llm():
    page = Page(id="page-1", name="\u9996\u9875")
    trace = [{
        "action_name": "click",
        "trace_phase": "navigation",
        "current_page": page,
        "caused_target_crash": True,
    }]
    state = {
        "package_name": "com.example.app",
        "current_navigation_scenario": "\u8fdb\u5165\u8bbe\u7f6e\u9875",
        "navigation_execution_trace": trace,
        "scenario_execution_trace": [],
        "execution_result": {
            "navigation_status": "app_crashed",
            "navigation_error": "\u76ee\u6807\u5e94\u7528\u5d29\u6e83",
            "scenario_status": "skipped",
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
        "current_navigation_scenario": "\u8fdb\u5165\u7f16\u8f91\u9875",
        "current_test_scenario": "\u4fdd\u5b58\u4fee\u6539",
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
        "current_test_scenario": "\u70b9\u51fb\u4fdd\u5b58\u5e76\u786e\u8ba4\u7ed3\u679c",
        "navigation_execution_trace": [],
        "scenario_execution_trace": [{
            "action_name": "click",
            "thought": "\u5df2\u70b9\u51fb\u4fdd\u5b58，\u4f46\u9875\u9762\u6301\u7eed\u663e\u793a\u9519\u8bef\u63d0\u793a",
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
    """Each independent detection receives the complete trace and does not reuse executor conversation context."""
    captured = {}
    execution_model = object()

    def fake_invoke(llm, messages, **_kwargs):
        captured["llm"] = llm
        captured["messages"] = messages
        return {"bugs": []}

    monkeypatch.setattr(trace_defect_detection, "invoke_llm_json_with_retry", fake_invoke)
    state = {
        "current_navigation_scenario": "\u8fdb\u5165\u7f16\u8f91\u9875",
        "current_test_scenario": "\u4fdd\u5b58\u4fee\u6539\u5e76\u786e\u8ba4\u7ed3\u679c",
        "navigation_execution_trace": [{
            "action_name": "click",
            "trace_phase": "navigation",
            "current_page": Page(id="page-1", name="\u9996\u9875", representative_screenshot="early-image"),
        }],
        "scenario_execution_trace": [{
            "action_name": "finished",
            "thought": "\u4fdd\u5b58\u540e\u7ed3\u679c\u5df2\u663e\u793a",
            "trace_phase": "test",
            "current_page": Page(id="page-2", name="\u7ed3\u679c\u9875", representative_screenshot="final-image"),
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
            "description": "\u5df2\u6709\u95ee\u9898",
            "expected": "\u4fdd\u5b58\u540e\u663e\u793a\u65b0\u503c",
            "actual": "\u4fdd\u5b58\u540e\u4ecd\u663e\u793a\u65e7\u503c",
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
    assert "[Complete execution-trace bug review]" in header_text
    assert "\u4fdd\u5b58\u4fee\u6539\u5e76\u786e\u8ba4\u7ed3\u679c" in header_text
    assert "do not treat test intent as a product promise" in header_text
    assert '"existing_bug_summaries"' in header_text
    assert '"bug_id": "BUG-007"' in header_text
    assert '"description": "\u5df2\u6709\u95ee\u9898"' in header_text
    assert '"actual": "\u4fdd\u5b58\u540e\u4ecd\u663e\u793a\u65e7\u503c"' in header_text
    assert "do not emit a duplicate finding" in header_text
    step_text = "\n".join(
        block["text"]
        for block in review_content
        if block.get("type") == "text"
    )
    assert "phase=navigation" in step_text
    assert "phase=test" in step_text
    images = [block for block in review_content if block.get("type") == "image_url"]
    # Send both screenshot-bearing trace steps independently without sampling or omission.
    assert len(images) == 2
    assert result["detected_bugs"][0]["description"] == "\u5df2\u6709\u95ee\u9898"


def test_llm_bug_without_direct_evidence_category_is_rejected(monkeypatch):
    def fake_invoke(_llm, _messages, **_kwargs):
        return {
            "bugs": [{
                "type": "logic_error",
                "severity": "high",
                "confidence": "high",
                "description": "\u6309\u94ae\u6807\u7b7e\u5728\u4e0d\u540c\u6b65\u9aa4\u53d1\u751f\u53d8\u5316",
                "expected": "\u6807\u7b7e\u4fdd\u6301\u4e0d\u53d8",
                "actual": "\u6807\u7b7e\u968f\u9875\u9762\u72b6\u6001\u53d8\u5316",
            }]
        }

    monkeypatch.setattr(trace_defect_detection, "invoke_llm_json_with_retry", fake_invoke)
    state = {
        "current_test_scenario": "\u6267\u884c\u8ba1\u7b97",
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
        "description": "\u70b9\u51fb\u7f16\u8f91\u56fe\u6807\u540e\u672a\u89e6\u53d1\u9884\u671f\u8df3\u8f6c，\u53cd\u800c\u5f39\u51fa\u5907\u6ce8\u5bf9\u8bdd\u6846",
        "expected": "\u9875\u9762\u5e94\u8df3\u8f6c\u5230\u7f16\u8f91\u9875\u9762",
        "actual": "\u9875\u9762\u5f39\u51fa\u5907\u6ce8\u5bf9\u8bdd\u6846",
    }
    navigation_scenario = "\u70b9\u51fb\u94c5\u7b14\u56fe\u6807，\u89c2\u5bdf\u9875\u9762\u662f\u5426\u8df3\u8f6c\u5230\u7f16\u8f91\u9875\u9762。"
    association_bug = {
        "evidence_category": "persisted_result_error",
        "description": "\u5907\u6ce8\u88ab\u9519\u8bef\u5173\u8054\u5230\u5f53\u524d\u6d3b\u52a8，\u800c\u975e\u76ee\u6807\u5217\u8868\u9879",
        "expected": "\u5907\u6ce8\u5e94\u5173\u8054\u5230Woodworking\u6d3b\u52a8",
        "actual": "\u5907\u6ce8\u663e\u793a\u5728Cinema\u6d3b\u52a8\u4e0b",
    }
    association_scenario = "\u70b9\u51fbWoodworking\u6d3b\u52a8\u65c1\u8fb9\u7684\u94c5\u7b14\u56fe\u6807，\u8f93\u5165\u5907\u6ce8\u5e76\u4fdd\u5b58。"

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
        "description": "\u7a7a\u6807\u9898\u672a\u6267\u884c\u5fc5\u586b\u6821\u9a8c",
        "expected": "\u5fc5\u586b\u5b57\u6bb5\u4e3a\u7a7a\u65f6\u5e94\u5f53\u963b\u6b62\u4fdd\u5b58",
        "actual": "\u7a7a\u6807\u9898\u88ab\u4fdd\u5b58",
    }
    empty_scenario = "\u4e0d\u8f93\u5165\u4efb\u4f55\u6587\u672c\u5230Activity Title\u8f93\u5165\u6846，\u70b9\u51fb\u4fdd\u5b58\u5e76\u89c2\u5bdf\u662f\u5426\u5141\u8bb8\u4fdd\u5b58。"
    failed_input_bug = {
        "evidence_category": "simultaneous_state_contradiction",
        "description": "\u8f93\u5165\u6846\u4e3a\u7a7a\u65f6\u663e\u793a\u540d\u79f0\u91cd\u590d\u63d0\u793a",
        "expected": "\u8f93\u5165\u6846\u5e94\u663e\u793a\u7279\u6b8a\u5b57\u7b26",
        "actual": "\u8f93\u5165\u6846\u4ecd\u4e3a\u7a7a\u5e76\u663e\u793a\u91cd\u590d\u63d0\u793a",
    }
    input_scenario = "\u5728\u8f93\u5165\u6846\u4e2d\u8f93\u5165\u6587\u672c '!@#$%^&*'，\u70b9\u51fb\u4fdd\u5b58。"

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
        "description": "\u4fdd\u5b58Task A\u540e\u7ed3\u679c\u9875\u663e\u793aTask B",
        "expected": "\u4fdd\u5b58\u540e\u7684\u7ed3\u679c\u9875\u663e\u793aTask A",
        "actual": "\u7ed3\u679c\u9875\u663e\u793aTask B",
    }
    scenario = "\u5728\u6807\u9898\u8f93\u5165\u6846\u4e2d\u8f93\u5165\u6587\u672c 'Task A'，\u70b9\u51fb\u4fdd\u5b58\u5e76\u89c2\u5bdf\u7ed3\u679c\u9875。"

    assert trace_defect_detection._bug_uses_speculative_or_automation_expectation(
        bug,
        scenario,
    ) is False


def test_one_shot_task_result_reset_on_reentry_is_not_a_persistence_bug():
    """Reentering a completed one-shot battery repair task may legitimately reset its task screen."""
    bug = {
        "evidence_category": "persisted_result_error",
        "description": "\u7535\u6c60\u4fee\u590d\u6210\u529f\u540e\u79bb\u5f00\u518d\u6b21\u8fdb\u5165\u7535\u6c60\u4fee\u590d\u9875，\u9875\u9762\u53c8\u56de\u5230\u5145\u7535\u4e2d\u5e76\u91cd\u65b0\u51fa\u73b0“\u5f00\u59cb\u4fee\u590d”\u6309\u94ae，\u4fee\u590d\u5b8c\u6210\u7ed3\u679c\u672a\u88ab\u4fdd\u7559。",
        "expected": "\u4f9d\u636e“\u7535\u6c60\u5df2\u7ecf\u4fee\u590d\u6210\u529f！”，\u518d\u6b21\u8fdb\u5165\u5e94\u4fdd\u7559\u5df2\u4fee\u590d\u72b6\u6001。",
        "actual": "\u518d\u6b21\u8fdb\u5165\u540e\u663e\u793a“\u5145\u7535\u4e2d...”\u548c“\u5f00\u59cb\u4fee\u590d”\u6309\u94ae，\u4e0e\u4fee\u590d\u524d\u4e00\u81f4。",
    }
    scenario = "\u70b9\u51fb\u5f00\u59cb\u4fee\u590d\u6309\u94ae，\u8fdb\u5ea6\u5230100%\u540e\u8fd4\u56de\u9996\u9875，\u518d\u6b21\u8fdb\u5165\u7535\u6c60\u4fee\u590d\u9875\u89c2\u5bdf\u662f\u5426\u4fdd\u7559\u5b8c\u6210\u6001。"

    assert trace_defect_detection._bug_uses_speculative_or_automation_expectation(
        bug,
        scenario,
    ) is True


def test_setting_state_lost_on_reentry_is_still_reported():
    bug = {
        "evidence_category": "persisted_result_error",
        "description": "\u5f00\u542f\u7701\u7535\u6a21\u5f0f\u5f00\u5173\u540e\u79bb\u5f00\u518d\u6b21\u8fdb\u5165，\u5f00\u5173\u6062\u590d\u4e3a\u5173\u95ed\u72b6\u6001。",
        "expected": "\u5f00\u5173\u5e94\u4fdd\u6301\u5f00\u542f。",
        "actual": "\u518d\u6b21\u8fdb\u5165\u540e\u5f00\u5173\u4e3a\u5173\u95ed。",
    }
    scenario = "\u5f00\u542f\u7701\u7535\u6a21\u5f0f\u5f00\u5173，\u8fd4\u56de\u9996\u9875\u540e\u518d\u6b21\u8fdb\u5165\u7701\u7535\u6a21\u5f0f\u9875。"

    assert trace_defect_detection._bug_uses_speculative_or_automation_expectation(
        bug,
        scenario,
    ) is False


def test_progress_animation_frame_mismatch_is_not_a_state_contradiction():
    bug = {
        "evidence_category": "simultaneous_state_contradiction",
        "description": "\u8fdb\u5ea6\u9875\u663e\u793a 86%、\u6587\u6848\u4e3a“\u6b63\u5728\u5904\u7406\u4e2d…”，\u4f46\u540c\u5c4f\u7684\u8fdb\u5ea6\u56fe\u5f62\u586b\u5145\u53ea\u6709\u7ea6\u56db\u5206\u4e4b\u4e00。",
        "expected": "\u56fe\u5f62\u586b\u5145\u5e94\u4e0e\u8fdb\u5ea6\u6570\u503c\u4e00\u81f4。",
        "actual": "\u56fe\u5f62\u586b\u5145\u5c11\u4e8e\u6570\u503c\u6240\u793a\u8fdb\u5ea6。",
    }

    assert trace_defect_detection._bug_uses_speculative_or_automation_expectation(
        bug,
        "\u70b9\u51fb\u5f00\u59cb\u6309\u94ae\u5e76\u7b49\u5f85\u4efb\u52a1\u5b8c\u6210。",
    ) is True


def test_static_text_contradiction_is_still_reported():
    bug = {
        "evidence_category": "simultaneous_state_contradiction",
        "description": "\u8bbe\u7f6e\u9875\u540c\u4e00\u622a\u56fe\u4e2d\u5f00\u5173\u663e\u793a\u4e3a\u5f00\u542f，\u800c\u4e0b\u65b9\u8bf4\u660e\u6587\u5b57\u5199\u7740“\u5f53\u524d\u5df2\u5173\u95ed”。",
        "expected": "\u5f00\u5173\u72b6\u6001\u4e0e\u8bf4\u660e\u6587\u5b57\u4e00\u81f4。",
        "actual": "\u5f00\u5173\u4e3a\u5f00\u542f，\u6587\u5b57\u4e3a\u5df2\u5173\u95ed。",
    }

    assert trace_defect_detection._bug_uses_speculative_or_automation_expectation(
        bug,
        "\u6253\u5f00\u8bbe\u7f6e\u9875\u67e5\u770b\u5f00\u5173\u72b6\u6001。",
    ) is False
