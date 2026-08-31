import base64
import json
import time

import cv2
import numpy as np
import pytest

from automator.h2 import H2
from model import flow_graph as flow_graph_module
from main import (
    RunResources,
    _build_parser,
    _clear_android_background_tasks,
    _graph_recursion_limit,
    _print_graph_step_status,
    _run_graph,
    _resolve_target_app,
    _save_artifact_manifest,
    _save_activity_timeline,
    _save_bug_folders,
    _save_final_outputs,
    _save_run_result,
)
from model.flow_graph import (
    FlowGraph,
    FunctionNode,
    Page,
    PageTransition,
)
from tools.actions import (
    ScreenshotUnavailableError,
    _perform_action,
    capture_current_page,
    capture_runtime_page,
)
from workflow.graph import _create_scenario_plan_node_with_iteration
from workflow.llm import invoke_llm_json_with_retry
from workflow import trace_defect_detection
from workflow.trace_defect_detection import (
    _bug_has_required_trace_evidence,
    _runtime_crash_bug,
)
from workflow.nodes.ffg_update import (
    FFG_UPDATE_IMAGE_MAX_SIDE,
    FFG_UPDATE_IMAGE_QUALITY,
    FFG_UPDATE_SUMMARY_MAX_CHARS,
    _build_ffg_summary as _build_update_ffg_summary,
    _filter_trace,
    _needs_trace_screenshot,
    _unrepresented_trace_activities,
    apply_ffg_patch,
    create_ffg_update_node,
)
from workflow.nodes.prompt import (
    BOOTSTRAP_FFG_SYSTEM_PROMPT,
    BUG_DETECT_SYSTEM_PROMPT,
    EXPLORATION_MODE_INSTRUCTIONS,
    FFG_AUDIT_SYSTEM_PROMPT,
    FFG_UPDATE_SYSTEM_PROMPT,
    SCENARIO_EXEC_SYSTEM_PROMPT_TEMPLATE,
    UNIFIED_SCENARIO_PLAN_SYSTEM_PROMPT,
)
from workflow.nodes._shared import (
    has_exhaustive_control_family_intent,
    representative_control_family_scenario,
)
from workflow.nodes import scenario_exec as scenario_exec_module
from workflow.nodes.scenario_exec import (
    _detect_stall,
    _finished_declares_failure,
    _normalize_action_plan,
    _parse_action_json,
    _proposed_action_text,
    _restart_target_app,
    _run_single_scenario,
    _scale_plan_coordinates,
    _validate_action_plan,
    _verification_attempt_reason,
    _verification_finish_reason,
)
from workflow.nodes.scenario_plan import (
    _attach_current_page_identity,
    _build_global_planning_summary,
    _ensure_target_app,
    _validate_plan_result,
    create_scenario_plan_node,
)
from workflow.wechat_safety import (
    wechat_action_policy_violation,
    wechat_read_only_violation,
)


@pytest.fixture(autouse=True)
def _legacy_text_matching_enabled(monkeypatch):
    """本文件用例写于操作-控件显式绑定之前，依赖文本匹配计入覆盖。

    显式绑定是覆盖率的权威来源且文本兜底默认关闭
    （COVERAGE_LEGACY_TEXT_MATCHING），此处显式开启以保持用例原语义。
    """
    monkeypatch.setattr(flow_graph_module, "COVERAGE_LEGACY_TEXT_MATCHING", True)


def _screenshot(marker_width: int) -> str:
    image = np.full((256, 256, 3), 255, dtype=np.uint8)
    if marker_width:
        image[20:40, 20:20 + marker_width] = 0
    success, encoded = cv2.imencode(".jpg", image)
    assert success
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _repeated_click_trace(screenshots: list[str]) -> list[dict]:
    return [
        {
            "action_name": "click",
            "x": 100,
            "y": 200,
            "current_page": Page(id="page_1", representative_screenshot=screenshot),
        }
        for screenshot in screenshots
    ]


def test_iteration_limit_is_checked_before_planner_invocation():
    calls = []
    wrapped = _create_scenario_plan_node_with_iteration(
        lambda state: calls.append(state) or {"workflow_complete": False}
    )

    updates = wrapped({
        "iteration_count": 3,
        "max_iterations": 3,
        "run_phase": "all",
    })

    assert calls == []
    assert updates["iteration_count"] == 3
    assert updates["workflow_complete"] is True


def test_page_evidence_limit_of_one_keeps_only_representative(monkeypatch):
    monkeypatch.setattr(flow_graph_module, "MAX_EVIDENCE_SCREENSHOTS_PER_PAGE", 0)

    page = Page(
        representative_screenshot="representative",
        evidence_screenshots=["old-1", "old-2"],
    )

    assert page.evidence_screenshots == []


def test_h2_reads_foreground_package_and_ability_from_driver():
    automator = object.__new__(H2)

    class FakeDriver:
        def current_app(self):
            return "com.example.harmony", "EntryAbility"

    automator._driver = FakeDriver()

    assert automator.current_package() == "com.example.harmony"
    assert automator.current_activity() == "EntryAbility"


def test_terminal_observation_records_last_action_result(monkeypatch):
    result_page = Page(id="runtime-final", name="保存结果", activity="ResultActivity")
    monkeypatch.setattr(
        scenario_exec_module,
        "capture_runtime_page",
        lambda _device: (result_page, ""),
    )
    trace = [{
        "action_name": "click",
        "action_description": "点击保存按钮",
        "current_page": Page(id="page_1", name="编辑页面"),
        "trace_phase": "test",
    }]

    page, health_error = scenario_exec_module._append_terminal_observation(
        trace,
        object(),
        "test",
    )

    assert page is result_page
    assert health_error == ""
    assert trace[-1]["action_name"] == "capture_screenshot"
    assert trace[-1]["observation_only"] is True
    assert trace[-1]["terminal_observation"] is True
    assert trace[-1]["current_page"] is result_page


def test_repeated_click_is_not_stalled_when_small_screen_region_changes():
    trace = _repeated_click_trace([
        _screenshot(10),
        _screenshot(20),
        _screenshot(30),
    ])

    assert _detect_stall(trace) == ""


def test_two_repeated_clicks_are_tolerated_when_screen_is_unchanged():
    screenshot = _screenshot(10)
    trace = _repeated_click_trace([screenshot, screenshot])

    assert _detect_stall(trace) == ""


def test_three_repeated_clicks_are_stalled_when_screen_is_unchanged():
    screenshot = _screenshot(10)
    trace = _repeated_click_trace([screenshot, screenshot, screenshot])

    assert "连续 3 次重复点击" in _detect_stall(trace)


def test_same_click_intent_with_changed_coordinates_stalls_after_one_retry():
    screenshot = _screenshot(10)
    trace = [
        {
            "action_name": "click",
            "x": x,
            "y": 900,
            "thought": "当前在服务通知页面，找到观看直播按钮，点击该按钮进入对应页面",
            "current_page": Page(id="page_1", representative_screenshot=screenshot),
        }
        for x in (120, 520)
    ]

    assert "相同点击意图" in _detect_stall(trace)


def test_navigation_same_scroll_search_intent_stalls_after_one_retry(monkeypatch):
    monkeypatch.setattr(scenario_exec_module, "NAVIGATION_SCROLL_INTENT_REPEAT_WINDOW", 2)
    trace = [
        {
            "action_name": "scroll",
            "direction": "left",
            "trace_phase": "navigation",
            "thought": thought,
            "current_page": Page(activity="SearchResultUI"),
        }
        for thought in (
            "向左滑动顶部标签栏寻找网络搜索入口",
            "继续左滑顶部标签栏找到网络搜索入口",
        )
    ]

    reason = _detect_stall(trace)
    assert "连续 2 次向左滚动" in reason
    assert "寻找" in reason


def test_execution_prompt_keeps_exploring_after_intermediate_result():
    assert "中间结果页不是默认终点" in SCENARIO_EXEC_SYSTEM_PROMPT_TEMPLATE
    assert "探索方向已经完成" in SCENARIO_EXEC_SYSTEM_PROMPT_TEMPLATE
    assert "首次看到结果后应根据截图选择一个代表项继续" in SCENARIO_EXEC_SYSTEM_PROMPT_TEMPLATE
    assert "屏幕内虚拟按键" in SCENARIO_EXEC_SYSTEM_PROMPT_TEMPLATE


def test_audit_prompt_uses_one_representative_for_same_result_controls():
    assert "分类、筛选、排序和样式选项" in FFG_AUDIT_SYSTEM_PROMPT
    assert "不换名称逐项枚举" in FFG_AUDIT_SYSTEM_PROMPT


def test_single_device_action_failure_can_recover_with_alternate_action(monkeypatch):
    screenshot = _screenshot(10)

    class Response:
        def __init__(self, content):
            self.content = content

    class Llm:
        def __init__(self):
            self.responses = iter([
                Response('{"thought":"输入数字","action_name":"type","content":"1234"}'),
                Response('{"thought":"改用可见键盘后目标已达成","action_name":"finished"}'),
            ])

        def invoke(self, _messages):
            return next(self.responses)

    class Device:
        def display_size(self):
            return 1200, 2652

    monkeypatch.setattr(scenario_exec_module, "_ensure_target_app", lambda *_args: True)
    monkeypatch.setattr(scenario_exec_module, "_target_process_pid", lambda *_args: "")
    monkeypatch.setattr(
        scenario_exec_module,
        "_target_app_foreground_state",
        lambda *_args: ("foreground", "com.example"),
    )
    monkeypatch.setattr(
        scenario_exec_module,
        "capture_runtime_page",
        lambda *_args, **_kwargs: (
            Page(id="page_1", activity="MainActivity", representative_screenshot=screenshot),
            "",
        ),
    )
    monkeypatch.setattr(
        scenario_exec_module,
        "_perform_action",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("input has no focus")),
    )
    monkeypatch.setattr(
        scenario_exec_module,
        "_recover_target_app_at_scene_end",
        lambda *_args, **_kwargs: (False, "", ""),
    )
    monkeypatch.setattr(scenario_exec_module.time, "sleep", lambda _seconds: None)

    trace, _, status, error, _, _ = scenario_exec_module._run_single_scenario(
        llm=Llm(),
        device=Device(),
        app_name="Example",
        package_name="com.example",
        scenario="输入四位数字",
        initial_page=Page(id="page_1"),
        clear_logcat=False,
        trace_phase="test",
    )

    assert status == "completed"
    assert error == ""
    assert len(trace) == 2
    assert trace[0]["action_failed"] is True
    assert trace[1]["action_name"] == "finished"


def test_action_parser_uses_last_valid_corrected_json_object():
    raw = (
        '{"action_name":"click","x":52, 70}\n修正：\n'
        '{"thought":"返回","action_name":"click","x":52,"y":70}'
    )

    parsed = _parse_action_json(raw)

    assert parsed["action_name"] == "click"
    assert parsed["x"] == 52
    assert parsed["y"] == 70


def test_executor_stops_immediately_after_unavailable_screenshot(monkeypatch):
    screenshot = _screenshot(10)

    class Llm:
        calls = 0

        class Response:
            content = (
                '{"thought":"点击入口","action_name":"click",'
                '"action_description":"点击危险入口","x":100,"y":200}'
            )

        def invoke(self, _messages):
            self.calls += 1
            return self.Response()

    class Device:
        def display_size(self):
            return 1200, 2652

    llm = Llm()
    actions = []
    captures = iter([
        (Page(id="page_1", activity="MainActivity", representative_screenshot=screenshot), ""),
        ScreenshotUnavailableError("黑屏"),
    ])
    monkeypatch.setattr(scenario_exec_module, "_ensure_target_app", lambda *_args: True)
    monkeypatch.setattr(scenario_exec_module, "_target_process_pid", lambda *_args: "")
    monkeypatch.setattr(
        scenario_exec_module,
        "_target_app_foreground_state",
        lambda *_args: ("foreground", "com.example"),
    )

    def capture(*_args, **_kwargs):
        result = next(captures)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(scenario_exec_module, "capture_runtime_page", capture)
    monkeypatch.setattr(
        scenario_exec_module,
        "_perform_action",
        lambda **kwargs: actions.append(kwargs),
    )
    monkeypatch.setattr(scenario_exec_module.time, "sleep", lambda _seconds: None)

    trace, _, status, error, _, _ = scenario_exec_module._run_single_scenario(
        llm=llm,
        device=Device(),
        app_name="Example",
        package_name="com.example",
        scenario="点击危险入口并确认结果",
        initial_page=Page(id="page_1"),
        clear_logcat=False,
        trace_phase="test",
    )

    assert status == "screenshot_unavailable"
    assert "当前场景已停止" in error
    assert llm.calls == 1
    assert len(actions) == 1
    assert len(trace) == 1
    assert trace[0]["action_description"] == "点击危险入口"


def test_runtime_capture_preserves_black_image_but_strict_capture_rejects_it(monkeypatch):
    class Device:
        def screenshot(self):
            return np.zeros((120, 120, 3), dtype=np.uint8)

        def current_activity(self):
            return "MainActivity"

    monkeypatch.setattr("tools.actions.time.sleep", lambda _seconds: None)

    runtime_page, health_error = capture_runtime_page(Device())

    assert runtime_page.representative_screenshot
    assert "黑屏" in health_error
    with pytest.raises(ScreenshotUnavailableError, match="黑屏"):
        capture_current_page(Device())


def test_repeated_runtime_black_screen_restarts_and_resets_counter(monkeypatch):
    black_page = Page(activity="MainActivity", representative_screenshot=_screenshot(20))
    reset_page = Page(id="page_1", activity="MainActivity", representative_screenshot=_screenshot(10))
    scenario_results = iter([
        ([], reset_page, "skipped", "", "", None),
        ([{
            "action_name": "capture_screenshot",
            "current_page": black_page,
            "observation_only": True,
            "screenshot_health_error": "截图疑似黑屏",
        }], black_page, "screenshot_unavailable", "黑屏", "", None),
    ])
    restart_calls = []
    monkeypatch.setattr(scenario_exec_module, "_ensure_target_app", lambda *_args: True)
    monkeypatch.setattr(
        scenario_exec_module,
        "_run_single_scenario",
        lambda *_args, **_kwargs: next(scenario_results),
    )

    def restart(trace, _device, _package_name, trace_phase):
        restart_calls.append(trace_phase)
        trace.append({
            "action_name": "restart_app",
            "current_page": reset_page,
            "is_interruption": True,
        })
        return reset_page, ""

    monkeypatch.setattr(
        scenario_exec_module,
        "_restart_after_repeated_black_screen",
        restart,
    )
    execution_pass = scenario_exec_module.create_scenario_execution_pass(object(), object())
    updates, _ = execution_pass({
        "app_name": "Example",
        "package_name": "com.example",
        "current_page": reset_page,
        "current_navigation_scenario": "",
        "current_test_scenario": "恢复后测试其它入口",
        "current_plan": {"scenario_type": "intra_test"},
        "screenshot_unavailable_count": 1,
    })

    assert restart_calls == ["test"]
    assert updates["current_page"] is reset_page
    assert updates["screenshot_unavailable_count"] == 0
    assert updates["scenario_execution_trace"][-1]["action_name"] == "restart_app"
    assert updates["execution_result"]["scenario_status"] == "screenshot_unavailable"
    assert updates["execution_result"]["scenario_error"] == "黑屏"
    assert updates["execution_result"]["scenario_trace_length"] == 2
    assert "test_status" not in updates["execution_result"]
    assert "test_error" not in updates["execution_result"]
    assert "test_trace_length" not in updates["execution_result"]


def test_action_parser_repairs_anonymous_coordinate_between_x_and_y():
    raw = """{
      "thought": "返回",
      "action_name": "click",
      "is_interruption": false,
      "x": 52, 70,
      "y": 70
    }"""

    parsed = _parse_action_json(raw)

    assert parsed["action_name"] == "click"
    assert parsed["x"] == 52
    assert parsed["y"] == 70


def test_action_parser_does_not_broadly_repair_other_invalid_json():
    with pytest.raises(ValueError, match="没有合法 JSON"):
        _parse_action_json('{"action_name":"click","x":52,"y":70, 80}')


def test_action_normalization_handles_string_boolean_and_case():
    plan = _normalize_action_plan({
        "action_name": "CLICK",
        "is_interruption": "false",
        "x": "100",
        "y": "200",
    })

    assert plan["action_name"] == "click"
    assert plan["is_interruption"] is False
    _validate_action_plan(plan)


def test_action_validation_rejects_out_of_range_coordinate_and_direction():
    with pytest.raises(ValueError, match="0–1000"):
        _validate_action_plan({
            "action_name": "click",
            "is_interruption": False,
            "x": 1001,
            "y": 20,
        })
    with pytest.raises(ValueError, match="direction 非法"):
        _validate_action_plan({
            "action_name": "scroll",
            "is_interruption": False,
            "direction": "forward",
        })


def test_coordinate_1000_maps_to_last_valid_pixel():
    plan = {"x": 1000, "y": 1000}

    _scale_plan_coordinates(plan, 1080, 2400)

    assert plan == {"x": 1079, "y": 2399}


def test_interruption_recovery_abab_cycle_is_stalled_early():
    trace = [
        {
            "action_name": "click",
            "x": 700,
            "y": 1000,
            "is_interruption": True,
            "current_page": Page(activity="MainActivity"),
        },
        {
            "action_name": "click",
            "x": 970,
            "y": 680,
            "is_interruption": False,
            "current_page": Page(activity="MainActivity"),
        },
        {
            "action_name": "click",
            "x": 702,
            "y": 1002,
            "is_interruption": True,
            "current_page": Page(activity="MainActivity"),
        },
        {
            "action_name": "click",
            "x": 972,
            "y": 682,
            "is_interruption": False,
            "current_page": Page(activity="MainActivity"),
        },
    ]

    assert "干扰恢复" in _detect_stall(trace)


def test_visual_abab_cycle_in_same_activity_is_stalled_early():
    closed = _screenshot(10)
    opened = _screenshot(80)
    trace = [
        {
            "action_name": "click",
            "x": x,
            "y": y,
            "current_page": Page(activity="LauncherUI", representative_screenshot=screenshot),
        }
        for x, y, screenshot in (
            (1100, 180, closed),
            (500, 1100, opened),
            (1110, 185, closed),
            (480, 1120, opened),
        )
    ]

    assert "界面状态往返循环" in _detect_stall(trace)


def test_persisted_result_bug_requires_real_persistence_action():
    bug = {
        "evidence_category": "persisted_result_error",
        "evidence": [
            {"global_step": 1, "role": "action"},
            {"global_step": 2, "role": "after"},
        ],
    }
    observe_only_trace = [
        {
            "raw_step_index": 1,
            "trace_phase": "test",
            "action_name": "click",
            "thought": "点击活动按钮并观察计时",
            "action_description": "选择活动",
        },
        {"action_name": "wait", "thought": "等待界面变化"},
    ]
    save_trace = [
        {
            "raw_step_index": 1,
            "trace_phase": "test",
            "action_name": "click",
            "thought": "点击右上角对勾提交新建活动",
            "action_description": "保存活动",
        },
        {
            "raw_step_index": 2,
            "trace_phase": "test",
            "action_name": "finished",
            "thought": "观察保存后的活动列表",
        },
    ]

    assert _bug_has_required_trace_evidence(bug, observe_only_trace) is False
    assert _bug_has_required_trace_evidence(bug, save_trace) is True


def test_finished_via_different_entry_is_not_treated_as_success():
    thought = "原铅笔入口不符合任务预期，改用替代入口加号后目标已达成。"

    assert _finished_declares_failure(thought) is True


def test_finished_with_success_and_unreachable_subgoal_is_failure():
    thought = "目标已达成；但文件发送入口不可达，剩余验证未完成。"

    assert _finished_declares_failure(thought) is True


def test_finished_with_all_validation_points_satisfied_is_success_despite_negative_notice():
    thought = "出现了 CAN'T SAVE AN EMPTY NOTE 的提示，但不影响跳转结果，任务的所有验证点均已满足。"

    assert _finished_declares_failure(thought) is False


def test_finished_with_all_task_goals_reached_is_success_despite_negative_notice():
    thought = (
        "顶部提示 CAN'T SAVE AN EMPTY NOTE 符合空笔记无法保存的预期，"
        "所有任务目标均达成。"
    )

    assert _finished_declares_failure(thought) is False


def test_coverage_audit_keeps_distinct_creation_content_types():
    assert "文本、清单、照片/相机、文件" in FFG_AUDIT_SYSTEM_PROMPT
    assert "分别实际点击验证" in FFG_AUDIT_SYSTEM_PROMPT


def test_planning_output_clears_previous_execution_payload():
    planning_node = _create_scenario_plan_node_with_iteration(
        lambda _state: {
            "current_plan": {"scenario_type": "status_update"},
            "workflow_complete": False,
        }
    )

    result = planning_node({
        "execution_result": {"scenario_status": "completed"},
        "navigation_execution_trace": [{"action_name": "click"}],
        "scenario_execution_trace": [{"action_name": "finished"}],
    })

    assert result["execution_result"] == {}
    assert result["navigation_execution_trace"] == []
    assert result["scenario_execution_trace"] == []


def test_graph_step_status_prints_compact_node_specific_summaries(capsys):
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", test_status="in_progress"))
    graph.add_page(Page(id="page_1", activity="MainActivity"), "func_1")
    state = {
        "ffg": graph,
        "current_page": graph.get_page("page_1"),
        "current_testing_function_id": "func_1",
        "current_testing_function_status": "in_progress",
        "current_plan": {
            "scenario_type": "intra_test",
            "target_function_id": "func_1",
        },
        "current_test_scenario": "点击保存按钮并观察结果",
        "execution_result": {
            "navigation_status": "skipped",
            "navigation_trace_length": 0,
            "scenario_status": "completed",
            "scenario_trace_length": 2,
            "current_scenario": "不应重复打印的完整场景",
        },
        "ffg_update_decision": "finish",
    }

    _print_graph_step_status("scenario_plan", state)
    _print_graph_step_status("scenario_exec", state)
    _print_graph_step_status("ffg_update", state)

    output = capsys.readouterr().out
    assert "[状态] scenario_plan | type=intra_test | scope=func_1" in output
    assert "[计划] 场景: 点击保存按钮并观察结果" in output
    assert "navigation=skipped(0步) | scenario=completed(2步) | page=page_1" in output
    assert "decision=finish | graph=1F/1P | current_function=func_1(in_progress)" in output
    assert "不应重复打印的完整场景" not in output
    assert "执行结果: {" not in output


def test_status_transition_is_resolved_inside_the_same_iteration():
    calls = []

    def planner(state):
        calls.append(dict(state))
        if len(calls) == 1:
            return {
                "current_plan": {
                    "scenario_type": "status_update",
                    "target_function_id": "func_1",
                },
                "current_testing_function_id": "",
                "current_testing_function_status": "not_started",
                "planner_noop_count": 1,
                "planner_feedback": "切换目标",
                "workflow_complete": False,
            }
        return {
            "current_plan": {
                "scenario_type": "intra_test",
                "target_function_id": "func_2",
            },
            "current_test_scenario": "点击待测入口并观察结果",
            "workflow_complete": False,
        }

    wrapped = _create_scenario_plan_node_with_iteration(planner)
    result = wrapped({"iteration_count": 0, "run_phase": "all"})

    assert len(calls) == 2
    assert result["iteration_count"] == 1
    assert result["current_plan"]["scenario_type"] == "intra_test"
    assert result["current_test_scenario"] == "点击待测入口并观察结果"


def test_current_page_identity_uses_closest_recorded_screenshot():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1"))
    graph.add_page(
        Page(
            id="page_1",
            name="模式一",
            activity="Calculator",
            representative_screenshot=_screenshot(10),
        ),
        "func_1",
    )
    graph.add_page(
        Page(
            id="page_2",
            name="模式二",
            activity="Calculator",
            representative_screenshot=_screenshot(30),
        ),
        "func_1",
    )
    current_page = Page(
        activity="Calculator",
        representative_screenshot=_screenshot(30),
    )

    resolved = _attach_current_page_identity(graph, current_page)

    assert resolved is current_page
    assert resolved.id == "page_2"
    assert resolved.name == "模式二"


def test_background_cleanup_counts_only_successful_task_removals(capsys):
    class Connector:
        def shell(self, command: str):
            if command == "dumpsys activity recents":
                return "Recent #0: TaskRecord{abc #42 A=example U=0}"
            if command == "cmd activity remove-task 42":
                raise RuntimeError("unsupported")
            return ""

    class Device:
        operating_system = "adb"
        connector = Connector()

    _clear_android_background_tasks(Device(), "com.example")

    assert "removed_recent_tasks=0/1" in capsys.readouterr().out


def test_cli_package_name_is_optional():
    parser = _build_parser()
    current_args = parser.parse_args(["--device-id", "emulator-5554"])
    specified_args = parser.parse_args([
        "--device-id", "emulator-5554",
        "--package-name", "com.example.app",
    ])

    assert current_args.package_name is None
    assert specified_args.package_name == "com.example.app"


def test_cli_allows_disabling_iteration_limit():
    args = _build_parser().parse_args([
        "--device-id", "emulator-5554",
        "--max-iterations", "0",
    ])

    assert args.max_iterations == 0


def test_cli_allows_disabling_bug_detection():
    parser = _build_parser()

    default_args = parser.parse_args(["--device-id", "emulator-5554"])
    disabled_args = parser.parse_args([
        "--device-id", "emulator-5554",
        "--no-bug-detection",
    ])

    assert default_args.bug_detection is True
    assert disabled_args.bug_detection is False


def test_graph_stream_gets_long_run_recursion_budget(tmp_path):
    class Graph:
        config = None

        def stream(self, _state, *, config):
            self.config = config
            yield {"scenario_plan": {}}

    graph = Graph()
    _run_graph(
        graph,
        {"max_iterations": 0},
        RunResources(output_dir=str(tmp_path)),
    )

    assert graph.config == {"recursion_limit": _graph_recursion_limit({"max_iterations": 0})}
    assert graph.config["recursion_limit"] >= 1_000_000
    assert (tmp_path / "bugs.json").is_file()
    assert (tmp_path / "activity_timeline.json").is_file()
    assert (tmp_path / "activity_timeline.png").is_file()
    running_result = json.loads((tmp_path / "run_result.json").read_text(encoding="utf-8"))
    assert running_result["status"] == "running"


def test_time_budget_keeps_one_ffg_alive_after_planner_reports_done(monkeypatch):
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="主功能", test_status="completed"))

    def planner(state):
        assert state["ffg"] is graph
        return {
            "ffg": graph,
            "workflow_complete": True,
            "current_plan": {"scenario_type": "done"},
        }

    monkeypatch.setattr("workflow.graph.LONG_RUN_IDLE_SECONDS", 0)
    wrapped = _create_scenario_plan_node_with_iteration(planner)
    result = wrapped({
        "ffg": graph,
        "iteration_count": 4,
        "start_time": time.time(),
        "total_time_limit": 3600,
        "max_iterations": 0,
    })

    assert result["ffg"] is graph
    assert result["iteration_count"] == 5
    assert result["workflow_complete"] is False
    assert result["long_run_idle_count"] == 1


def test_final_artifact_set_is_complete(tmp_path):
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="主功能"))
    graph.add_page(Page(
        id="page_1",
        representative_screenshot=_screenshot(12),
        controls=["可点击的‘新增’按钮"],
    ), "func_1")
    state = {
        "ffg": graph,
        "detected_bugs": [],
        "iteration_count": 1,
        "workflow_complete": True,
    }
    (tmp_path / "run.log").write_text("complete\n", encoding="utf-8")
    (tmp_path / "llm_usage.json").write_text("{}\n", encoding="utf-8")

    _save_final_outputs(state, str(tmp_path))
    state["_activity_timeline"] = _save_activity_timeline(
        state,
        str(tmp_path),
        force=True,
    )
    _save_run_result(state, str(tmp_path), exit_code=0, elapsed_seconds=1.0)
    manifest = _save_artifact_manifest(state, str(tmp_path), exit_code=0)

    assert manifest["status"] == "complete"
    assert manifest["errors"] == []
    assert (tmp_path / "screenshots" / "page_1_representative.jpg").is_file()
    assert (tmp_path / "activity_timeline.json").is_file()
    assert cv2.imread(str(tmp_path / "activity_timeline.png")) is not None


def test_bug_folders_skip_unchanged_artifacts(monkeypatch, tmp_path):
    bug = {
        "bug_id": "bug_incremental",
        "confidence": "high",
        "title": "按钮无响应",
    }
    trace = [{
        "action_name": "click",
        "x": 100,
        "y": 200,
        "current_page": Page(
            id="page_1",
            name="详情页",
            representative_screenshot=_screenshot(8),
        ),
    }]
    state = {
        "bug_trace_records": {bug["bug_id"]: trace},
        "scenario_execution_trace": trace,
    }

    _save_bug_folders(state, str(tmp_path), [bug], "[测试]")
    bug_dir = tmp_path / "bugs" / bug["bug_id"]
    assert (bug_dir / ".artifact-fingerprint").is_file()
    assert (bug_dir / "trace.json").is_file()
    assert (bug_dir / "screenshots" / "step_001.jpg").is_file()

    monkeypatch.setattr(
        "main._atomic_write_json",
        lambda *_args, **_kwargs: pytest.fail("未变化 bug 不应重写 JSON"),
    )
    monkeypatch.setattr(
        "main._atomic_write_bytes",
        lambda *_args, **_kwargs: pytest.fail("未变化 bug 不应重写截图或指纹"),
    )

    _save_bug_folders(state, str(tmp_path), [bug], "[测试]")


def test_activity_timeline_accumulates_unique_discovered_activities(tmp_path):
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="主功能"))
    graph.add_page(Page(id="page_1", activity="LauncherUI"), "func_1")
    state = {
        "start_time": time.time() - 60,
        "ffg": graph,
        "current_page": Page(activity="SearchUI"),
        "navigation_execution_trace": [
            {"current_page": Page(activity="ContactInfoUI")},
        ],
        "scenario_execution_trace": [
            {"current_page": Page(activity="LauncherUI")},
        ],
    }

    state["_activity_timeline"] = _save_activity_timeline(
        state,
        str(tmp_path),
        force=True,
    )
    _save_run_result(state, str(tmp_path), exit_code=None, elapsed_seconds=60)

    timeline = json.loads((tmp_path / "activity_timeline.json").read_text(encoding="utf-8"))
    result = json.loads((tmp_path / "run_result.json").read_text(encoding="utf-8"))
    assert timeline["activities"] == ["ContactInfoUI", "LauncherUI", "SearchUI"]
    assert timeline["points"][-1]["activity_count"] == 3
    assert result["status"] == "running"
    assert result["activity_discovery"]["count"] == 3
    assert cv2.imread(str(tmp_path / "activity_timeline.png")) is not None


def test_app_specific_account_guard_is_not_hardcoded_in_prompts():
    for prompt in (
        UNIFIED_SCENARIO_PLAN_SYSTEM_PROMPT,
        BOOTSTRAP_FFG_SYSTEM_PROMPT,
        SCENARIO_EXEC_SYSTEM_PROMPT_TEMPLATE,
        FFG_UPDATE_SYSTEM_PROMPT,
        FFG_AUDIT_SYSTEM_PROMPT,
        BUG_DETECT_SYSTEM_PROMPT,
        *EXPLORATION_MODE_INSTRUCTIONS.values(),
    ):
        assert "com.tencent.mm" not in prompt
        assert "微信规则" not in prompt


def test_missing_package_uses_current_foreground_app():
    args = _build_parser().parse_args([
        "--device-id", "emulator-5554",
    ])

    class Device:
        def current_package(self):
            return "com.example.current"

    assert _resolve_target_app(args, Device()) is False
    assert args.package_name == "com.example.current"
    assert args.app_name == "com.example.current"


def test_target_open_app_restarts_when_app_is_foreground():
    class Device:
        def __init__(self):
            self.started = []
            self.restarted = []
            self.foreground = "com.tencent.mm"

        def current_package(self):
            return self.foreground

        def start_app(self, package):
            self.started.append(package)

        def restart_app(self, package):
            self.restarted.append(package)

    device = Device()

    _perform_action(
        device,
        "open_app",
        app_name="微信",
        package_name="com.tencent.mm",
    )

    # 应用在前台但可能停在深层 Activity：重启以回到入口页。
    assert device.restarted == ["com.tencent.mm"]
    assert device.started == []


def test_target_open_app_merely_starts_when_app_not_foreground():
    class Device:
        def __init__(self):
            self.started = []
            self.restarted = []
            self.foreground = "com.hihonor.android.launcher"

        def current_package(self):
            return self.foreground

        def start_app(self, package):
            self.started.append(package)

        def restart_app(self, package):
            self.restarted.append(package)

    device = Device()

    _perform_action(
        device,
        "open_app",
        app_name="微信",
        package_name="com.tencent.mm",
    )

    # 目标应用不在前台（桌面/其他应用/冷启动中）：只拉起不强杀，
    # 避免模型在冷启动期间重复 open_app 形成 force-stop 重启风暴。
    assert device.started == ["com.tencent.mm"]
    assert device.restarted == []


def test_generic_open_app_keeps_foreground_start_behavior():
    class Device:
        def __init__(self):
            self.started = []
            self.restarted = []

        def start_app(self, package):
            self.started.append(package)

        def restart_app(self, package):
            self.restarted.append(package)

    device = Device()

    _perform_action(device, "open_app", app_name="com.example")

    assert device.started == ["com.example"]
    assert device.restarted == []


def test_specified_package_does_not_depend_on_foreground_app():
    args = _build_parser().parse_args([
        "--device-id", "emulator-5554",
        "--package-name", "com.example.app",
    ])

    class Device:
        def current_package(self):
            raise AssertionError("specified package must not inspect foreground")

    assert _resolve_target_app(args, Device()) is True
    assert args.package_name == "com.example.app"


def test_missing_package_requires_a_visible_foreground_app():
    args = _build_parser().parse_args(["--device-id", "emulator-5554"])

    class Device:
        def current_package(self):
            return ""

    with pytest.raises(RuntimeError, match="请先在设备上打开目标 App"):
        _resolve_target_app(args, Device())


def test_planner_rejects_screen_when_restarted_app_is_not_foreground(monkeypatch):
    class Device:
        def current_package(self):
            return "com.android.launcher3"

        def restart_app(self, _package_name):
            return None

    monkeypatch.setattr("workflow.nodes.scenario_plan.time.sleep", lambda _seconds: None)

    assert _ensure_target_app(Device(), "com.example", None) is None


def test_planner_restarts_target_and_captures_runtime_page(monkeypatch):
    class Device:
        def __init__(self):
            self.package = "com.google.android.packageinstaller"
            self.restarted = False

        def current_package(self):
            return self.package

        def restart_app(self, _package_name):
            self.restarted = True
            self.package = "com.example"

    device = Device()
    expected = Page(id="runtime", name="主页")
    monkeypatch.setattr("workflow.nodes.scenario_plan.time.sleep", lambda _seconds: None)
    monkeypatch.setattr(
        "workflow.nodes.scenario_plan.capture_current_page",
        lambda *_args, **_kwargs: expected,
    )

    assert _ensure_target_app(device, "com.example", None) is expected
    assert device.restarted is True


def test_planner_restarts_foreground_target_after_black_screenshot(monkeypatch):
    class Device:
        restarts = 0
        backs = 0

        def current_package(self):
            return "com.example"

        def back(self):
            self.backs += 1

        def restart_app(self, _package_name):
            self.restarts += 1

    device = Device()
    expected = Page(id="runtime", name="主页")
    captures = iter([ScreenshotUnavailableError("黑屏"), expected])
    monkeypatch.setattr("workflow.nodes.scenario_plan.time.sleep", lambda _seconds: None)

    def capture(*_args, **_kwargs):
        result = next(captures)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr("workflow.nodes.scenario_plan.capture_current_page", capture)

    assert _ensure_target_app(device, "com.example", None) is expected
    assert device.backs == 1
    assert device.restarts == 0


def test_planner_restarts_after_back_cannot_recover_black_screen(monkeypatch):
    class Device:
        restarts = 0
        backs = 0

        def current_package(self):
            return "com.example"

        def back(self):
            self.backs += 1

        def restart_app(self, _package_name):
            self.restarts += 1

    device = Device()
    expected = Page(id="runtime", name="主页")
    captures = iter([
        ScreenshotUnavailableError("初始黑屏"),
        ScreenshotUnavailableError("返回后仍黑屏"),
        expected,
    ])
    monkeypatch.setattr("workflow.nodes.scenario_plan.PLAN_SCREENSHOT_BACK_ATTEMPTS", 1)
    monkeypatch.setattr("workflow.nodes.scenario_plan.time.sleep", lambda _seconds: None)

    def capture(*_args, **_kwargs):
        result = next(captures)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr("workflow.nodes.scenario_plan.capture_current_page", capture)

    assert _ensure_target_app(device, "com.example", None) is expected
    assert device.backs == 1
    assert device.restarts == 1


def test_planner_uses_black_screen_as_current_page_and_generates_recovery_navigation():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="设置功能", test_status="in_progress"))
    graph.add_page(Page(
        id="page_1",
        name="设置页",
        controls=["可点击的通知设置入口"],
    ), "func_1")
    graph.add_page(Page(
        id="page_2",
        name="黑屏或页面不可用",
        description="异常终点",
    ), "func_1")
    graph.add_page_transition(PageTransition(
        src_page_id="page_1",
        dst_page_id="page_2",
        operation="点击危险入口；结果：黑屏或页面不可用（截图健康检查失败）",
    ))

    class Device:
        def current_package(self):
            return "com.example"

    class Llm:
        class Response:
            content = json.dumps({
                "scenario_type": "intra_test",
                "target_function": "func_1",
                "src_function": "",
                "dst_function": "",
                "navigation_scenario": "先按返回键退出当前黑屏界面",
                "test_scenario": "点击通知设置入口 → 切换一次通知状态 → 确认状态变化后结束",
            }, ensure_ascii=False)

        def invoke(self, _messages):
            return self.Response()

    current_page = Page(id="page_2", representative_screenshot=_screenshot(20))
    result = create_scenario_plan_node(Llm(), Device())({
        "ffg": graph,
        "package_name": "com.example",
        "current_page": current_page,
        "current_testing_function_id": "func_1",
        "current_testing_function_status": "in_progress",
        "run_phase": "all",
    })

    assert result["current_plan"]["scenario_type"] == "intra_test"
    assert "返回" in result["current_navigation_scenario"]
    assert "恢复后" not in result["current_navigation_scenario"]
    assert "通知设置入口" in result["current_test_scenario"]
    assert result["current_page"].representative_screenshot == current_page.representative_screenshot


def test_planner_rejects_scenario_that_repeats_known_black_screen_path():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="设置功能", test_status="in_progress"))
    graph.add_page(Page(id="page_1", name="设置页"), "func_1")
    graph.add_page(Page(id="page_2", name="黑屏或页面不可用"), "func_1")
    graph.add_page_transition(PageTransition(
        src_page_id="page_1",
        dst_page_id="page_2",
        operation="点击危险入口；结果：黑屏或页面不可用（截图健康检查失败）",
    ))

    class Device:
        def current_package(self):
            return "com.example"

    class Llm:
        class Response:
            content = json.dumps({
                "scenario_type": "intra_test",
                "target_function": "func_1",
                "src_function": "",
                "dst_function": "",
                "navigation_scenario": "先按返回键退出当前黑屏界面",
                "test_scenario": "再次点击危险入口并确认结果",
            }, ensure_ascii=False)

        def invoke(self, _messages):
            return self.Response()

    result = create_scenario_plan_node(Llm(), Device())({
        "ffg": graph,
        "package_name": "com.example",
        "current_page": Page(id="page_2", representative_screenshot=_screenshot(20)),
        "current_testing_function_id": "func_1",
        "current_testing_function_status": "in_progress",
        "run_phase": "all",
    })

    assert result["current_plan"]["scenario_type"] == "status_update"
    assert "导致黑屏" in result["planner_feedback"]


def test_executor_restart_requires_target_app_to_reach_foreground(monkeypatch):
    class Device:
        def restart_app(self, _package_name):
            return None

        def current_package(self):
            return "com.android.launcher3"

    monkeypatch.setattr("workflow.nodes.scenario_exec.time.sleep", lambda _seconds: None)

    assert _restart_target_app(Device(), "com.example") is False


def test_executor_restart_retries_until_foreground(monkeypatch):
    class Device:
        restarts = 0

        def restart_app(self, _package_name):
            self.restarts += 1

    device = Device()
    monkeypatch.setattr("workflow.nodes.scenario_exec.time.sleep", lambda _seconds: None)
    monkeypatch.setattr(
        scenario_exec_module,
        "_is_target_app_foreground",
        lambda current, _package_name: current.restarts >= 2,
    )

    assert _restart_target_app(device, "com.example") is True
    assert device.restarts == 2


def test_untrusted_external_app_is_recovered_and_scene_stops(monkeypatch):
    # 真外部跳转：前台状态复检（_settle_foreground_state，最多 4 次调用）
    # 后仍为非目标应用，才定性为 external_transition_blocked。
    states = iter([("background", "com.untrusted.app")] * 8)

    monkeypatch.setattr(scenario_exec_module, "_ensure_target_app", lambda *_args: True)
    monkeypatch.setattr(scenario_exec_module, "_target_process_pid", lambda *_args: "")
    monkeypatch.setattr(
        scenario_exec_module,
        "_target_app_foreground_state",
        lambda *_args: next(states, ("foreground", "com.example")),
    )
    monkeypatch.setattr(scenario_exec_module, "_read_crash_logcat", lambda *_args: "")
    monkeypatch.setattr(scenario_exec_module, "_perform_action", lambda **_kwargs: None)
    monkeypatch.setattr(scenario_exec_module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        scenario_exec_module,
        "capture_current_page",
        lambda *_args, **_kwargs: Page(id="page_recovered"),
    )

    trace, _, status, error, _, _ = scenario_exec_module._run_single_scenario(
        llm=object(),
        device=object(),
        app_name="Example",
        package_name="com.example",
        scenario="点击设置按钮并观察结果",
        initial_page=Page(id="page_start"),
        clear_logcat=False,
        trace_phase="test",
    )

    assert status == "external_transition_blocked"
    assert error == scenario_exec_module.EXTERNAL_TRANSITION_BLOCKED_END
    assert trace[-1]["foreground_recovery"] is True
    assert trace[-1]["recovery_method"] == "press_back"
    assert trace[-2]["external_context"] is True
    assert trace[-2]["external_package"] == "com.untrusted.app"
    assert trace[-2]["observation_only"] is True


@pytest.mark.parametrize(
    "scenario",
    [
        "到达公众号文章详情页并打开更多操作面板 → 点击‘在浏览器中打开’ → 确认实际结果后结束",
        "点击链接并跳转到默认浏览器查看网页",
        "使用 browser 打开页面并确认标题",
    ],
)
def test_explicit_browser_scenarios_allow_controlled_external_context(scenario):
    assert scenario_exec_module._scenario_allows_external_app(scenario) is True


def test_target_app_recovery_wording_does_not_allow_external_context():
    scenarios = [
        "先执行一次 open_app 直接打开目标应用主页面；若离开目标应用则再次 open_app",
        "为到达目标页而主动打开的菜单属于导航路径，不要再把它关闭",
    ]
    assert all(
        scenario_exec_module._scenario_allows_external_app(scenario) is False
        for scenario in scenarios
    )


def test_untrusted_external_app_reports_failed_restart(monkeypatch):
    monkeypatch.setattr(scenario_exec_module, "_ensure_target_app", lambda *_args: True)
    monkeypatch.setattr(scenario_exec_module, "_target_process_pid", lambda *_args: "")
    monkeypatch.setattr(
        scenario_exec_module,
        "_target_app_foreground_state",
        lambda *_args: ("background", "com.untrusted.app"),
    )
    monkeypatch.setattr(scenario_exec_module, "_read_crash_logcat", lambda *_args: "")
    monkeypatch.setattr(scenario_exec_module, "_recover_target_app_by_back", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(scenario_exec_module, "_restart_target_app", lambda *_args: False)

    _, _, status, error, _, _ = scenario_exec_module._run_single_scenario(
        llm=object(),
        device=object(),
        app_name="Example",
        package_name="com.example",
        scenario="点击设置按钮并观察结果",
        initial_page=Page(id="page_start"),
        clear_logcat=False,
        trace_phase="test",
    )

    assert status == "target_app_recovery_failed"
    assert error == scenario_exec_module.TARGET_APP_RECOVERY_FAILED_END


def test_emulator_egl_failure_is_not_reported_as_app_bug():
    crash_logcat = (
        "E/OpenGLRenderer: Failed to choose config, error = EGL_NOT_INITIALIZED\n"
        "F/libc: Fatal signal 6 in RenderThread\n"
        "F/DEBUG: >>> com.example <<<"
    )

    bug = _runtime_crash_bug(
        {
            "navigation_status": "target_app_crashed",
            "scenario_status": "skipped_due_target_app_crashed",
            "crash_logcat": crash_logcat,
        },
        [{"action_name": "click"}],
        "启动应用",
        "com.example",
    )

    assert bug is None


def test_target_java_crash_is_reported_even_with_egl_noise():
    crash_logcat = (
        "E/OpenGLRenderer: Failed to choose config, error = EGL_NOT_INITIALIZED\n"
        "E/AndroidRuntime: FATAL EXCEPTION: main\n"
        "E/AndroidRuntime: Process: com.example, PID: 123\n"
        "E/AndroidRuntime: java.lang.RuntimeException: Unable to start activity\n"
    )

    bug = _runtime_crash_bug(
        {
            "navigation_status": "skipped",
            "scenario_status": "target_app_crashed",
            "crash_detected": True,
            "crash_logcat": crash_logcat,
        },
        [{"action_name": "click"}],
        "打开设置",
        "com.example",
    )

    assert bug is not None
    assert bug["type"] == "crash"
    assert bug["source"] == "runtime_monitor"


def test_planner_allows_scenario_regardless_of_action_count():
    """探索方向语义下不再按动作数量限制场景。"""
    result = {
        "workflow_complete": False,
        "scenario_type": "intra_test",
        "target_function": "func_1",
        "src_function": "",
        "dst_function": "",
        "navigation_scenario": "",
        "test_scenario": "，".join(["点击按钮"] * 9),
    }

    assert not any(
        error.startswith("test_scenario_too_many_actions=")
        for error in _validate_plan_result(result)
    )


@pytest.mark.parametrize(
    "scenario",
    [
        "点击所有表情网格中的表情 → 点击发送按钮",
        "逐个选择每个筛选标签",
    ],
)
def test_planner_rejects_exhaustive_control_family_scenarios(scenario):
    result = {
        "workflow_complete": False,
        "scenario_type": "intra_test",
        "target_function": "func_1",
        "src_function": "",
        "dst_function": "",
        "navigation_scenario": "",
        "test_scenario": scenario,
    }

    assert "test_scenario_exhaustive_control_family" in _validate_plan_result(result)


def test_planner_allows_explicit_one_shot_bulk_operation():
    result = {
        "workflow_complete": False,
        "scenario_type": "intra_test",
        "target_function": "func_1",
        "src_function": "",
        "dst_function": "",
        "navigation_scenario": "",
        "test_scenario": "点击全选按钮 → 确认全部内容被选中",
    }

    assert "test_scenario_exhaustive_control_family" not in _validate_plan_result(result)


def test_planner_allows_small_batch_of_exact_named_controls():
    result = {
        "workflow_complete": False,
        "scenario_type": "intra_test",
        "target_function": "func_1",
        "src_function": "",
        "dst_function": "",
        "navigation_scenario": "",
        "test_scenario": "依次点击“通知”、“隐私”和“通用”，分别观察直接结果后返回",
    }

    assert "test_scenario_exhaustive_control_family" not in _validate_plan_result(result)


def test_executor_fallback_rewrites_exhaustive_family_to_representative_item():
    scenario = "点击所有表情网格中的表情 → 点击绿色发送按钮"

    normalized = representative_control_family_scenario(scenario)

    assert has_exhaustive_control_family_intent(scenario)
    assert "同类控件只操作一个代表项" in normalized
    assert "一个代表性的表情" in normalized
    assert "首次结果出现后立即结束" in normalized


def test_planner_requires_stable_function_ids_for_executable_plans():
    intra_plan = {
        "workflow_complete": False,
        "scenario_type": "intra_test",
        "target_function": "科学计算器",
        "src_function": "",
        "dst_function": "",
        "navigation_scenario": "",
        "test_scenario": "点击数字并观察结果",
    }
    cross_plan = {
        **intra_plan,
        "scenario_type": "cross_test",
        "target_function": "",
        "src_function": "func_1",
        "dst_function": "设置功能",
    }

    assert "intra_test_requires_func_id='科学计算器'" in _validate_plan_result(intra_plan)
    assert "cross_test_requires_dst_func_id='设置功能'" in _validate_plan_result(cross_plan)


def test_wechat_policy_allows_conversation_state_mutation():
    plan = {
        "workflow_complete": False,
        "scenario_type": "intra_test",
        "target_function": "func_1",
        "src_function": "",
        "dst_function": "",
        "navigation_scenario": "",
        "test_scenario": "长按会话并点击不显示该聊天",
    }

    errors = _validate_plan_result(plan, package_name="com.tencent.mm")

    assert not any(error.startswith("wechat_read_only_policy=") for error in errors)
    assert not wechat_read_only_violation("com.example.app", plan["test_scenario"])


def test_scenario_plan_no_longer_rejects_unknown_dynamic_resource():
    """探索方向语义下，程序不再校验动态资源；约束交给 prompt 与执行器。"""
    plan = {
        "scenario_type": "intra_test",
        "target_function": "func_1",
        "src_function": "",
        "dst_function": "",
        "navigation_scenario": "",
        "test_scenario": "探索搜索入口并查看一个直接结果，无明确后续时结束",
    }

    assert not any(
        error.startswith("unknown_dynamic_resource=")
        for error in _validate_plan_result(plan)
    )
    assert not any(
        error.startswith("test_scenario_too_many_actions=")
        for error in _validate_plan_result(plan)
    )


def test_executor_blocks_wechat_logout_before_touching_device():
    result = _run_single_scenario(
        None,
        None,
        "微信",
        "com.tencent.mm",
        "进入设置并点击退出登录",
        None,
        False,
        "test",
    )

    trace, _page, status, error, _logcat, _context = result
    assert trace == []
    assert status == "blocked_by_safety_policy"
    assert "退出登录或删除账户" in error


def test_executor_blocks_verification_attempt_but_allows_exit_action():
    assert _verification_attempt_reason(
        "scroll",
        "拖动拼图滑块通过安全验证",
    )
    assert _verification_attempt_reason(
        "click",
        "点击右上角关闭滑块验证弹窗",
    ) == ""
    assert _verification_finish_reason("验证码阻塞，已退出当前页面")
    assert _verification_finish_reason("已成功通过验证并进入下一页") == ""


def test_blocked_page_exit_presses_back_once_and_records_recovery(monkeypatch):
    actions: list[str] = []
    recovered_page = Page(id="page_recovered", name="恢复页面")
    monkeypatch.setattr(
        scenario_exec_module,
        "_perform_action",
        lambda *, action_name, **_kwargs: actions.append(action_name),
    )
    monkeypatch.setattr(
        scenario_exec_module,
        "capture_current_page",
        lambda _device: recovered_page,
    )
    monkeypatch.setattr(scenario_exec_module.time, "sleep", lambda _seconds: None)
    trace = [{
        "action_name": "scroll",
        "action_failed": True,
        "current_page": Page(id="page_blocked"),
    }]

    result = scenario_exec_module._exit_blocked_page(trace, object(), "test")

    assert result is recovered_page
    assert actions == ["press_back"]
    assert trace[-1]["blocked_page_exit"] is True
    assert trace[-1]["observation_after_recovery"] is True


def test_finished_success_with_unread_conversation_text_is_not_failure():
    thought = (
        "所有指定会话条目和未读标记均已确认，页面无加载异常，"
        "已完成全部基础状态校验。"
    )

    assert not _finished_declares_failure(thought)


def test_wecreate_action_alias_and_bounding_box_are_normalized():
    plan = _normalize_action_plan({
        "action": "click",
        "thought": "点击返回",
        "x": [40, 80],
        "y": [50, 90],
    })

    assert plan["action_name"] == "click"
    assert plan["x"] == 60
    assert plan["y"] == 70
    assert "action" not in plan


def test_wecreate_next_action_alias_and_target_are_normalized():
    plan = _normalize_action_plan({
        "next_action": "click",
        "target": "设置",
        "x": 100,
        "y": 200,
    })

    assert plan["action_name"] == "click"
    assert plan["action_description"] == "click 设置"


def test_incomplete_next_action_is_rejected_before_device_execution():
    plan = _normalize_action_plan({
        "next_action": "click",
        "target": "设置",
    })

    with pytest.raises(ValueError, match="缺少 x/y 坐标"):
        _validate_action_plan(plan)


def test_action_intent_excludes_unrelated_visible_page_description():
    intent = _proposed_action_text({
        "thought": (
            "页面包含消息通知、语音和视频通话通知等设置项。"
            "下一步需要点击顶部左侧返回按钮。"
        ),
    })

    assert "返回按钮" in intent
    assert "视频通话" not in intent


def test_wechat_policy_allows_setting_switch_and_visible_call_text():
    assert not wechat_read_only_violation(
        "com.tencent.mm",
        "点击使用听筒播放语音消息开关，验证开关状态发生切换",
    )
    assert not wechat_read_only_violation(
        "com.tencent.mm",
        "页面包含消息通知、语音和视频通话通知等设置项，下一步点击左上角返回",
    )


def test_wechat_policy_allows_setting_profile_status_but_blocks_account_deletion():
    assert not wechat_read_only_violation("com.tencent.mm", "点击+ 状态并设置微信状态")
    assert wechat_read_only_violation("com.tencent.mm", "进入账号设置并删除账户")


def test_wechat_action_policy_only_blocks_logout_or_account_deletion():
    assert not wechat_action_policy_violation(
        "com.tencent.mm",
        "click",
        "点击底部导航栏的通讯录标签",
    )
    assert not wechat_action_policy_violation(
        "com.tencent.mm",
        "click",
        "点击发现",
    )
    assert not wechat_action_policy_violation(
        "com.tencent.mm",
        "click",
        "点击文件传输助手会话",
    )
    assert not wechat_action_policy_violation(
        "com.tencent.mm",
        "type",
        "在搜索框输入文本",
    )
    assert wechat_action_policy_violation(
        "com.tencent.mm",
        "click",
        "点击退出登录",
    )


def test_incomplete_max_step_trace_skips_semantic_bug_detection(monkeypatch):
    monkeypatch.setattr(
        trace_defect_detection,
        "invoke_llm_json_with_retry",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("incomplete trace must not reach LLM bug detection")
        ),
    )
    state = {
        "package_name": "com.example",
        "current_test_scenario": "超长场景",
        "scenario_execution_trace": [{
            "action_name": "click",
            "current_page": Page(id="page_1", representative_screenshot=_screenshot(10)),
        }],
        "navigation_execution_trace": [],
        "execution_result": {
            "navigation_status": "skipped",
            "scenario_status": "max_steps_reached",
        },
        "detected_bugs": [],
        "bug_trace_records": {},
        "current_plan": {"scenario_type": "intra_test", "target_function_id": "func_1"},
    }

    result = trace_defect_detection.analyze_execution_for_bugs(object(), state)

    assert result == {"detected_bugs": [], "bug_trace_records": {}}


def test_entry_exploration_skips_semantic_bug_detection(monkeypatch):
    monkeypatch.setattr(
        trace_defect_detection,
        "invoke_llm_json_with_retry",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("entry exploration must not reach semantic bug detection")
        ),
    )
    state = {
        "package_name": "com.example",
        "current_test_scenario": "点击入口并确认页面",
        "scenario_execution_trace": [{
            "action_name": "finished",
            "current_page": Page(id="page_2", representative_screenshot=_screenshot(10)),
        }],
        "navigation_execution_trace": [],
        "execution_result": {
            "navigation_status": "skipped",
            "scenario_status": "completed",
        },
        "detected_bugs": [],
        "bug_trace_records": {},
        "current_plan": {
            "scenario_type": "intra_test",
            "target_function_id": "func_1",
            "audit_exploration": True,
        },
    }

    result = trace_defect_detection.analyze_execution_for_bugs(object(), state)

    assert result == {"detected_bugs": [], "bug_trace_records": {}}


def test_ffg_update_keeps_navigation_images_until_visual_match_is_confirmed():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="主功能"))
    graph.add_page(Page(id="page_1", activity="KnownActivity"), "func_1")

    assert _needs_trace_screenshot(graph, {
        "trace_phase": "navigation",
        "current_page": Page(activity="KnownActivity", representative_screenshot=_screenshot(10)),
    })
    assert _needs_trace_screenshot(graph, {
        "trace_phase": "navigation",
        "current_page": Page(activity="NovelActivity", representative_screenshot=_screenshot(20)),
    })
    assert _needs_trace_screenshot(graph, {
        "trace_phase": "test",
        "current_page": Page(activity="KnownActivity", representative_screenshot=_screenshot(30)),
    })


def test_json_retry_helper_rejects_top_level_array():
    class Response:
        content = "[]"

    class Llm:
        def invoke(self, _messages):
            return Response()

    result = invoke_llm_json_with_retry(
        Llm(),
        [],
        max_attempts=1,
        retry_delay_seconds=0,
    )

    assert result == {}


def test_json_retry_helper_accepts_object_surrounded_by_local_model_prose():
    class Response:
        content = '分析如下：\n```json\n{"decision":"finish"}\n```\n以上。'

    class Llm:
        def invoke(self, _messages):
            return Response()

    result = invoke_llm_json_with_retry(
        Llm(),
        [],
        max_attempts=1,
        retry_delay_seconds=0,
    )

    assert result == {"decision": "finish"}


def test_json_retry_helper_does_not_start_after_deadline():
    class Llm:
        calls = 0

        def invoke(self, _messages):
            self.calls += 1
            raise AssertionError("expired request must not be invoked")

    llm = Llm()
    result = invoke_llm_json_with_retry(
        llm,
        [],
        deadline_monotonic=time.monotonic() - 1,
    )

    assert result == {}
    assert llm.calls == 0


def test_json_retry_helper_marks_exhausted_failure_but_not_valid_empty_object():
    from workflow.llm import llm_json_result_failed

    class FailureLlm:
        def invoke(self, _messages):
            raise RuntimeError("gateway unavailable")

    class EmptyObjectLlm:
        def invoke(self, _messages):
            return type("Response", (), {"content": "{}"})()

    failed = invoke_llm_json_with_retry(
        FailureLlm(), [], max_attempts=1, retry_delay_seconds=0
    )
    valid_empty = invoke_llm_json_with_retry(
        EmptyObjectLlm(), [], max_attempts=1, retry_delay_seconds=0
    )

    assert llm_json_result_failed(failed)
    assert not llm_json_result_failed(valid_empty)


def test_planning_summary_exposes_stable_ids_and_shared_page_owners():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="主导航"))
    graph.add_node(FunctionNode(id="func_2", name="计算"))
    shared_page = Page(id="page_1", name="计算器")
    graph.add_page(shared_page, "func_1")
    graph.add_page(shared_page, "func_2")

    summary = _build_global_planning_summary(graph, "all")

    assert "func_1 | 主导航" in summary
    assert "func_2 | 计算" in summary
    # 同一页面被两个功能共享时，两侧都必须看到该页面归属。
    assert summary.count("pages=1") == 2
    assert graph.get_page_owners("page_1") == ["func_1", "func_2"]


def test_ffg_update_context_separates_plan_facts_and_before_step_evidence():
    screenshot = _screenshot(20)
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="科学计算"))
    graph.add_page(Page(id="page_1", name="计算器", representative_screenshot=screenshot), "func_1")

    class Response:
        content = (
            '{"functions":[],"pages":[],"function_dependencies":[]}'
        )

    class CapturingLlm:
        def __init__(self):
            self.messages = []

        def invoke(self, messages):
            self.messages = messages
            return Response()

    llm = CapturingLlm()
    node = create_ffg_update_node(llm)
    node({
        "ffg": graph,
        "current_plan": {
            "scenario_type": "intra_test",
            "target_function_id": "func_1",
            "src_function_id": "",
            "dst_function_id": "",
        },
        "current_test_scenario": "点击数字并观察显示",
        "scenario_execution_trace": [
            {
                "action_name": "click",
                "x": 100,
                "y": 200,
                "action_description": "点击数字 1",
                "thought": "数字键位于左下区域",
                "current_page": Page(
                    id="page_1",
                    name="计算器",
                    activity="Calculator",
                    representative_screenshot=screenshot,
                ),
            },
            {
                "action_name": "finished",
                "action_description": "结束并记录结果",
                "current_page": Page(
                    id="page_1",
                    name="计算器",
                    activity="Calculator",
                    representative_screenshot=screenshot,
                ),
            },
        ],
        "navigation_execution_trace": [],
        "execution_result": {
            "navigation_status": "skipped",
            "scenario_status": "completed",
        },
        "current_testing_function_id": "func_1",
        "current_testing_function_status": "in_progress",
    })

    human_content = llm.messages[1].content
    text_content = "\n".join(
        block["text"]
        for block in human_content
        if block.get("type") == "text"
    )

    assert '"owners":[' in text_content
    assert '"func_1"' in text_content
    assert "【本轮权威规划作用域】" in text_content
    assert '"id": "func_1"' in text_content
    assert "coordinate=(100,200)" in text_content
    assert "executor_target_hint=点击数字 1" in text_content
    assert "[FFG_REFERENCE_IMAGE] page_id=page_1" in text_content
    assert "[OBSERVATION] BEFORE_STEP_1" in text_content
    assert "source_step=3" in text_content
    assert "finished 只是一张最终状态观测" in text_content


def test_ffg_update_rejects_filtered_ordinal_as_source_step():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="主功能"))
    trace = [
        {"raw_step_index": 1, "current_page": Page(representative_screenshot=_screenshot(10))},
        {"raw_step_index": 3, "current_page": Page(representative_screenshot=_screenshot(30))},
    ]

    result = apply_ffg_patch(
        graph,
        {
            "pages": [{
                "page_ref": "new_page_1",
                "function_refs": ["func_1"],
                "source_step": 2,
                "name": "错误映射页面",
                "description": "不应按过滤后序号映射",
            }],
        },
        trace,
    )

    assert result["page_refs"] == {}
    assert graph.pages == {}


def test_ffg_update_does_not_auto_reuse_page_across_functions():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="文本笔记"))
    graph.add_node(FunctionNode(id="func_2", name="待办清单"))
    screenshot = _screenshot(20)
    graph.add_page(Page(id="page_1", name="文本编辑页", representative_screenshot=screenshot), "func_1")

    result = apply_ffg_patch(
        graph,
        {
            "pages": [{
                "page_ref": "new_page_1",
                "function_refs": ["func_2"],
                "source_step": 1,
                "name": "待办清单编辑页",
                "description": "显示待办项目复选框",
            }],
        },
        [{"raw_step_index": 1, "current_page": Page(representative_screenshot=screenshot)}],
    )

    new_page_id = result["page_refs"]["new_page_1"]
    assert new_page_id != "page_1"
    assert graph.get_page_owners("page_1") == ["func_1"]
    assert graph.get_page_owners(new_page_id) == ["func_2"]


def test_ffg_update_uses_screenshots_and_natural_language_controls():
    # Long-running multimodal updates use a bandwidth-conscious image size while
    # retaining enough detail for mobile controls and text to remain legible.
    assert FFG_UPDATE_IMAGE_MAX_SIDE >= 640
    assert FFG_UPDATE_IMAGE_QUALITY >= 65
    assert "Page.controls 保存真实可见且可能交互的控件" in FFG_UPDATE_SYSTEM_PROMPT
    assert "同功能同结果的重复控件只保留一到两个代表" in FFG_UPDATE_SYSTEM_PROMPT


def test_obsolete_correction_patch_is_ignored():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="稳定功能"))

    result = apply_ffg_patch(
        graph,
        {"corrections": [{"entity_ref": "func_1", "operation": "deprecate"}]},
        [{"raw_step_index": 1, "current_page": Page(representative_screenshot=_screenshot(10))}],
    )

    assert "corrections_patch" not in result
    assert graph.get_node("func_1").name == "稳定功能"
def test_large_ffg_context_uses_global_index_and_bounded_focused_details():
    graph = FlowGraph()
    long_description = "页面控件和历史说明" * 120
    for index in range(1, 81):
        function_id = f"func_{index}"
        page_id = f"page_{index}"
        graph.add_node(FunctionNode(
            id=function_id,
            name=f"功能 {index}",
            description=long_description,
        ))
        graph.add_page(Page(
            id=page_id,
            name=f"页面 {index}",
            description=long_description,
            activity=f"Activity{index}",
            operations=[long_description] * 8,
        ), function_id)

    planning_summary = _build_global_planning_summary(graph, "all")
    update_summary = _build_update_ffg_summary(
        graph,
        {"scenario_type": "intra_test", "target_function_id": "func_80"},
        [{"raw_step_index": 1, "current_page": graph.get_page("page_80")}],
    )

    assert "【全局 FFG 摘要】" in planning_summary
    # 功能数超过 PLAN_MAX_GLOBAL_SUMMARY_FUNCTIONS（60）时，全局摘要
    # 只列前 60 个并标注省略数量；被省略功能由作用域摘要承载详情。
    assert "func_60 | 功能 60" in planning_summary
    assert "已省略 20 个功能" in planning_summary
    assert "func_80 | 功能 80" not in planning_summary
    assert "global_function_index" in update_summary
    assert '"focused_function_ids":["func_80"' in update_summary
    assert len(update_summary) <= FFG_UPDATE_SUMMARY_MAX_CHARS + 40


def test_ffg_update_respects_model_decision_to_create_new_page():
    screenshot = _screenshot(24)
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="主功能"))
    graph.add_page(Page(
        id="page_1",
        name="已记录页面",
        activity="MainActivity",
        representative_screenshot=screenshot,
    ), "func_1")

    result = apply_ffg_patch(
        graph,
        {
            "pages": [{
                "page_ref": "new_page_1",
                "function_refs": ["func_1"],
                "source_step": 3,
                "name": "模型误判的新页面",
                "description": "视觉证据实际与 page_1 相同",
            }],
        },
        [{
            "raw_step_index": 3,
            "current_page": Page(
                activity="MainActivity",
                representative_screenshot=screenshot,
            ),
        }],
    )

    assert result["page_refs"] == {"new_page_1": "page_2"}
    assert list(graph.pages) == ["page_1", "page_2"]


def test_empty_ffg_patch_without_device_action_keeps_function_in_progress():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="主功能"))
    graph.add_page(Page(id="page_1", representative_screenshot=_screenshot(10)), "func_1")

    class EmptyPatchLlm:
        class Response:
            content = "{}"

        def invoke(self, _messages):
            return self.Response()

    result = create_ffg_update_node(EmptyPatchLlm())({
        "ffg": graph,
        "current_plan": {
            "scenario_type": "intra_test",
            "target_function_id": "func_1",
        },
        "current_test_scenario": "进入主功能并完成一次结果验证",
        "scenario_execution_trace": [{
            "action_name": "finished",
            "current_page": Page(id="page_1", representative_screenshot=_screenshot(10)),
        }],
        "navigation_execution_trace": [],
        "execution_result": {
            "navigation_status": "skipped",
            "scenario_status": "completed",
        },
        "current_testing_function_id": "func_1",
        "current_testing_function_status": "in_progress",
    })

    assert graph.get_node_test_status("func_1") == "in_progress"
    assert graph.get_node("func_1").scenarios == []
    assert result["current_testing_function_status"] == "in_progress"


def test_ffg_transport_failure_short_circuits_alignment_refinement(monkeypatch):
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="主功能", test_status="in_progress"))
    graph.add_page(Page(
        id="page_1",
        representative_screenshot=_screenshot(10),
    ), "func_1")

    class FailingLlm:
        calls = 0

        def invoke(self, _messages):
            self.calls += 1
            raise RuntimeError("gateway unavailable")

    monkeypatch.setenv("LLM_RETRY_ATTEMPTS", "1")
    llm = FailingLlm()
    result = create_ffg_update_node(llm)({
        "ffg": graph,
        "current_page": Page(id="page_1", representative_screenshot=_screenshot(10)),
        "current_plan": {
            "scenario_type": "intra_test",
            "target_function_id": "func_1",
        },
        "current_test_scenario": "点击一个尚未建模的入口并观察结果",
        "scenario_execution_trace": [{
            "action_name": "click",
            "action_description": "点击尚未建模的入口",
            "current_page": Page(id="page_1", representative_screenshot=_screenshot(10)),
        }],
        "navigation_execution_trace": [],
        "execution_result": {
            "navigation_status": "skipped",
            "scenario_status": "completed",
        },
        "current_testing_function_id": "func_1",
        "current_testing_function_status": "in_progress",
    })

    assert llm.calls == 1
    assert result["ffg_update_decision"] == "finish"
    assert result["ffg_update_reason"] == "model patch unavailable; raw trace preserved"
    assert result["ffg_update_validation_issues"] == []


def test_failed_intra_execution_is_recorded_as_attempted_but_not_successful():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="主功能", test_status="in_progress"))
    graph.add_page(Page(
        id="page_1",
        representative_screenshot=_screenshot(10),
        controls=["临时撤销按钮"],
    ), "func_1")

    class EmptyPatchLlm:
        class Response:
            content = "{}"

        def invoke(self, _messages):
            return self.Response()

    create_ffg_update_node(EmptyPatchLlm())({
        "ffg": graph,
        "current_plan": {
            "scenario_type": "intra_test",
            "target_function_id": "func_1",
        },
        "current_test_scenario": "点击已经消失的临时撤销按钮",
        "scenario_execution_trace": [{
            "action_name": "click",
            "action_description": "点击临时撤销按钮",
            "action_failed": True,
            "action_error": "目标控件不可达",
            "current_page": Page(id="page_1", representative_screenshot=_screenshot(10)),
        }],
        "navigation_execution_trace": [],
        "execution_result": {
            "navigation_status": "skipped",
            "scenario_status": "stalled",
            "scenario_error": "目标控件不可达",
        },
        "current_testing_function_id": "func_1",
        "current_testing_function_status": "in_progress",
    })

    node = graph.get_node("func_1")
    assert node.scenarios == [
        "点击已经消失的临时撤销按钮"
    ]
    assert graph.get_page("page_1").operations == [
        "尝试点击临时撤销按钮失败：目标控件不可达"
    ]
    assert graph.get_node("func_1").coverage_percent == 100
    assert graph.get_node_test_status("func_1") == "in_progress"


def test_finished_only_trace_does_not_record_a_test_scenario():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="主功能", test_status="in_progress"))
    graph.add_page(Page(id="page_1", controls=["搜索按钮"]), "func_1")

    class EmptyPatchLlm:
        class Response:
            content = "{}"

        def invoke(self, _messages):
            return self.Response()

    create_ffg_update_node(EmptyPatchLlm())({
        "ffg": graph,
        "current_plan": {
            "scenario_type": "intra_test",
            "target_function_id": "func_1",
        },
        "current_test_scenario": "点击搜索按钮并观察结果",
        "scenario_execution_trace": [{
            "action_name": "finished",
            "current_page": Page(id="page_1"),
        }],
        "navigation_execution_trace": [],
        "execution_result": {
            "navigation_status": "skipped",
            "scenario_status": "completed",
        },
        "current_testing_function_id": "func_1",
        "current_testing_function_status": "in_progress",
    })

    assert graph.get_node("func_1").scenarios == []
    assert graph.get_node("func_1").coverage_percent == 0


def test_failed_intra_execution_without_trace_is_still_remembered():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="主功能", test_status="in_progress"))

    class NeverCalledLlm:
        def invoke(self, _messages):
            raise AssertionError("empty trace must not invoke FFG update LLM")

    create_ffg_update_node(NeverCalledLlm())({
        "ffg": graph,
        "current_plan": {
            "scenario_type": "intra_test",
            "target_function_id": "func_1",
        },
        "current_test_scenario": "点击真实账号安全策略禁止的入口",
        "navigation_execution_trace": [],
        "scenario_execution_trace": [],
        "execution_result": {
            "navigation_status": "skipped",
            "scenario_status": "blocked_by_safety_policy",
        },
        "current_testing_function_id": "func_1",
        "current_testing_function_status": "in_progress",
    })

    assert graph.get_node("func_1").scenarios == ["点击真实账号安全策略禁止的入口"]


def test_black_screen_trace_updates_ffg_and_reuses_single_error_page():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="主功能", test_status="in_progress"))
    graph.add_page(Page(id="page_1", name="主页面", representative_screenshot=_screenshot(10)), "func_1")

    class EmptyPatchLlm:
        class Response:
            content = "{}"

        def invoke(self, _messages):
            return self.Response()

    node = create_ffg_update_node(EmptyPatchLlm())
    base_state = {
        "ffg": graph,
        "current_plan": {
            "scenario_type": "intra_test",
            "target_function_id": "func_1",
        },
        "current_testing_function_id": "func_1",
        "current_testing_function_status": "in_progress",
        "navigation_execution_trace": [],
    }

    black_runtime_page = Page(activity="MainActivity", representative_screenshot=_screenshot(20))
    result = node({
        **base_state,
        "current_page": black_runtime_page,
        "current_test_scenario": "点击危险入口并确认结果",
        "scenario_execution_trace": [
            {
                "action_name": "click",
                "action_description": "点击危险入口",
                "x": 100,
                "y": 200,
                "current_page": Page(id="page_1", representative_screenshot=_screenshot(10)),
            },
            {
                "action_name": "capture_screenshot",
                "action_description": "观察到黑屏或页面不可用",
                "current_page": black_runtime_page,
                "observation_only": True,
                "screenshot_health_error": "截图疑似黑屏",
            },
        ],
        "execution_result": {
            "navigation_status": "skipped",
            "scenario_status": "screenshot_unavailable",
            "scenario_error": "黑屏",
        },
    })

    error_pages = [page for page in graph.active_pages() if page.name == "黑屏或页面不可用"]
    assert len(error_pages) == 1
    error_page = error_pages[0]
    assert error_page.representative_screenshot == ""
    assert error_page.id in graph.get_node("func_1").page_ids
    assert graph.get_node_test_status("func_1") == "in_progress"
    assert result["current_testing_function_id"] == "func_1"
    assert result["current_page"].id == error_page.id
    assert result["current_page"].representative_screenshot == black_runtime_page.representative_screenshot
    assert graph.get_node("func_1").scenarios == ["点击危险入口并确认结果"]
    assert any(
        transition.src_page_id == "page_1"
        and transition.dst_page_id == error_page.id
        and "点击危险入口" in transition.operation
        for transition in graph.page_transitions
    )
    assert any("点击危险入口" in operation for operation in graph.get_page("page_1").operations)

    node({
        **base_state,
        "current_test_scenario": "再次触发异常路径",
        "scenario_execution_trace": [],
        "execution_result": {
            "navigation_status": "skipped",
            "scenario_status": "screenshot_unavailable",
            "scenario_error": "仍然黑屏",
        },
    })

    assert len([page for page in graph.active_pages() if page.name == "黑屏或页面不可用"]) == 1


def test_ffg_update_keeps_failed_external_and_recovery_trace_steps():
    trace = [
        {
            "action_name": "click",
            "action_failed": True,
            "current_page": Page(id="page_1"),
        },
        {
            "action_name": "press_back",
            "is_interruption": True,
            "external_context": True,
            "current_page": Page(id="page_external"),
        },
        {
            "action_name": "restart_app",
            "foreground_recovery": True,
            "current_page": Page(id="page_2"),
        },
        {
            "action_name": "press_back",
            "action_failed": True,
            "action_error": "设备连接暂时不可用",
        },
    ]

    filtered = _filter_trace(trace)

    assert len(filtered) == 4
    assert [step["raw_step_index"] for step in filtered] == [1, 2, 3, 4]


def test_ffg_records_black_observation_even_when_model_recovers_in_same_scenario():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="主功能", test_status="in_progress"))
    graph.add_page(Page(id="page_1", name="主页面", representative_screenshot=_screenshot(10)), "func_1")

    class EmptyPatchLlm:
        class Response:
            content = "{}"

        def invoke(self, _messages):
            return self.Response()

    result = create_ffg_update_node(EmptyPatchLlm())({
        "ffg": graph,
        "current_page": Page(id="page_1", representative_screenshot=_screenshot(10)),
        "current_plan": {
            "scenario_type": "intra_test",
            "target_function_id": "func_1",
        },
        "current_test_scenario": "从异常界面返回后测试其它入口",
        "navigation_execution_trace": [{
            "action_name": "press_back",
            "action_description": "从黑屏界面返回",
            "current_page": Page(representative_screenshot=_screenshot(20)),
            "screenshot_health_error": "截图疑似黑屏",
        }],
        "scenario_execution_trace": [{
            "action_name": "finished",
            "current_page": Page(id="page_1", representative_screenshot=_screenshot(10)),
        }],
        "execution_result": {
            "navigation_status": "completed",
            "scenario_status": "completed",
        },
        "current_testing_function_id": "func_1",
        "current_testing_function_status": "in_progress",
    })

    assert any(page.name == "黑屏或页面不可用" for page in graph.active_pages())
    assert result["current_page"].id == "page_1"


def test_ffg_deterministically_records_operations_and_direct_transitions():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="主功能", test_status="in_progress"))
    graph.add_page(Page(id="page_1", name="列表页", representative_screenshot=_screenshot(10)), "func_1")
    graph.add_page(Page(id="page_2", name="详情页", representative_screenshot=_screenshot(20)), "func_1")

    class EmptyPatchLlm:
        class Response:
            content = "{}"

        def invoke(self, _messages):
            return self.Response()

    create_ffg_update_node(EmptyPatchLlm())({
        "ffg": graph,
        "current_page": Page(id="page_2", representative_screenshot=_screenshot(20)),
        "current_plan": {
            "scenario_type": "intra_test",
            "target_function_id": "func_1",
        },
        "current_test_scenario": "点击列表项进入详情",
        "navigation_execution_trace": [],
        "scenario_execution_trace": [
            {
                "action_name": "click",
                "action_description": "点击列表项",
                "current_page": Page(id="page_1", representative_screenshot=_screenshot(10)),
            },
            {
                "action_name": "finished",
                "current_page": Page(id="page_2", representative_screenshot=_screenshot(20)),
            },
        ],
        "execution_result": {
            "navigation_status": "skipped",
            "scenario_status": "completed",
        },
        "current_testing_function_id": "func_1",
        "current_testing_function_status": "in_progress",
    })

    assert "点击列表项" in graph.get_page("page_1").operations
    assert any(
        transition.src_page_id == "page_1"
        and transition.dst_page_id == "page_2"
        and transition.operation == "点击列表项"
        for transition in graph.page_transitions
    )


def test_new_activity_in_scenario_trace_must_be_represented_by_a_page():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="主功能"))
    graph.add_page(Page(id="page_1", activity="LauncherUI"), "func_1")

    missing = _unrepresented_trace_activities(
        graph,
        [{"current_page": Page(activity="ContactInfoUI")}],
        "com.tencent.mm",
        "点击通讯录标签并打开联系人资料页",
    )

    assert missing == ["ContactInfoUI"]


def test_model_can_map_transient_activity_to_existing_logical_page():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="主功能"))
    graph.add_page(Page(id="page_1", activity="ArticleUI"), "func_1")

    missing = _unrepresented_trace_activities(
        graph,
        [{"raw_step_index": 3, "current_page": Page(activity="WebViewStubProxyUI")}],
        "com.tencent.mm",
        "收藏文章后确认提示",
        {3: "page_1"},
    )

    assert missing == []


def test_navigation_only_trace_does_not_mark_test_scenario_attempted():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="主功能", test_status="in_progress"))
    graph.add_page(Page(id="page_1", representative_screenshot=_screenshot(10)), "func_1")

    class EmptyPatchLlm:
        class Response:
            content = "{}"

        def invoke(self, _messages):
            return self.Response()

    create_ffg_update_node(EmptyPatchLlm())({
        "ffg": graph,
        "current_plan": {
            "scenario_type": "intra_test",
            "target_function_id": "func_1",
        },
        "current_test_scenario": "点击主功能里的保存按钮",
        "navigation_execution_trace": [{
            "action_name": "click",
            "current_page": Page(id="page_1", representative_screenshot=_screenshot(10)),
        }],
        "scenario_execution_trace": [],
        "execution_result": {
            "navigation_status": "stalled",
            "scenario_status": "skipped_due_navigation_failure",
        },
        "current_testing_function_id": "func_1",
        "current_testing_function_status": "in_progress",
    })

    assert graph.get_node("func_1").scenarios == []
