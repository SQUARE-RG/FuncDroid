# Multilingual UI and model-response fixtures use Unicode escapes to preserve their exact text.
import base64
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from automator.h2 import H2
from model import flow_graph as flow_graph_module
from main import (
    RunResources,
    _build_parser,
    _graph_recursion_limit,
    _print_graph_step_status,
    _run_graph,
    _resolve_target_app,
    _reset_android_app_data_at_run_start,
    _save_artifact_manifest,
    _save_activity_timeline,
    _save_bug_folders,
    _save_final_outputs,
    _save_run_result,
)
from model.flow_graph import (
    FunctionalFlowGraph,
    FunctionalityNode,
    Page,
    PageTransition,
)
from tools.actions import (
    ScreenshotUnavailableError,
    _perform_action,
    _screenshot_health_error,
    capture_current_page,
    capture_runtime_page,
)
from tools.utils import encode_image_to_jpg_base64
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
from workflow.nodes import ffg_update as ffg_update_module
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
    _unfinished_safe_external_reason,
)
from workflow.nodes.scenario_plan import (
    _attach_current_page_identity,
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
    """Explicitly enable legacy text fallback to preserve these pre-binding tests' intended coverage semantics."""
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
    result_page = Page(id="runtime-final", name="\u4fdd\u5b58\u7ed3\u679c", activity="ResultActivity")
    monkeypatch.setattr(
        scenario_exec_module,
        "capture_runtime_page",
        lambda _device: (result_page, ""),
    )
    trace = [{
        "action_name": "click",
        "action_description": "\u70b9\u51fb\u4fdd\u5b58\u6309\u94ae",
        "current_page": Page(id="page_1", name="\u7f16\u8f91\u9875\u9762"),
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

    assert "3 consecutive clicks" in _detect_stall(trace)


def test_same_click_intent_with_changed_coordinates_tolerates_one_retry():
    """Allow one retry of an ineffective click with changed coordinates before declaring stagnation."""
    screenshot = _screenshot(10)
    trace = [
        {
            "action_name": "click",
            "x": x,
            "y": 900,
            "thought": "\u5f53\u524d\u5728\u670d\u52a1\u901a\u77e5\u9875\u9762，\u627e\u5230\u89c2\u770b\u76f4\u64ad\u6309\u94ae，\u70b9\u51fb\u8be5\u6309\u94ae\u8fdb\u5165\u5bf9\u5e94\u9875\u9762",
            "current_page": Page(id="page_1", representative_screenshot=screenshot),
        }
        for x in (120, 520)
    ]

    assert _detect_stall(trace) == ""


def test_same_click_intent_stall_requires_every_recent_click_to_match():
    """Every recent click must share the same intent; changing targets is not repeated blind probing."""
    screenshot = _screenshot(10)
    trace = [
        {
            "action_name": "click",
            "x": x,
            "y": 900,
            "thought": thought,
            "current_page": Page(id="page_1", representative_screenshot=screenshot),
        }
        for x, thought in (
            (120, "\u627e\u5230\u89c2\u770b\u76f4\u64ad\u6309\u94ae，\u70b9\u51fb\u8be5\u6309\u94ae\u8fdb\u5165\u5bf9\u5e94\u9875\u9762"),
            (520, "\u627e\u5230\u89c2\u770b\u76f4\u64ad\u6309\u94ae，\u70b9\u51fb\u8be5\u6309\u94ae\u8fdb\u5165\u5bf9\u5e94\u9875\u9762"),
            (300, "\u627e\u5230\u8bbe\u7f6e\u5165\u53e3，\u70b9\u51fb\u8be5\u6309\u94ae\u8fdb\u5165\u8bbe\u7f6e\u9875\u9762"),
        )
    ]

    assert _detect_stall(trace) == ""


def test_swipe_coordinates_are_scaled_like_click_coordinates():
    plan = {"action_name": "swipe", "x1": 0, "y1": 0, "x2": 722, "y2": 1599}

    _scale_plan_coordinates(plan, 1080, 2400, (723, 1600))

    assert plan == {
        "action_name": "swipe",
        "x1": 0,
        "y1": 0,
        "x2": 1079,
        "y2": 2399,
    }


def test_validate_action_plan_requires_all_swipe_endpoints():
    with pytest.raises(ValueError, match="swipe requires y2"):
        _validate_action_plan({
            "action_name": "swipe",
            "is_interruption": False,
            "x1": 500,
            "y1": 0,
            "x2": 500,
        }, image_size=(723, 1600))


def test_validate_action_plan_rejects_degenerate_swipe():
    with pytest.raises(ValueError, match="start and end coordinates must differ"):
        _validate_action_plan({
            "action_name": "swipe",
            "is_interruption": False,
            "x1": 500,
            "y1": 300,
            "x2": 500,
            "y2": 300,
        }, image_size=(723, 1600))


def test_wait_and_swipe_duration_defaults_and_clamp():
    wait_plan = {"action_name": "wait", "is_interruption": False}
    _validate_action_plan(wait_plan, image_size=(723, 1600))
    assert wait_plan["duration"] == scenario_exec_module.DEFAULT_WAIT_SECONDS

    capped_wait = {"action_name": "wait", "is_interruption": False, "duration": 999}
    _validate_action_plan(capped_wait, image_size=(723, 1600))
    assert capped_wait["duration"] == scenario_exec_module.MAX_WAIT_SECONDS

    capped_swipe = {
        "action_name": "swipe",
        "is_interruption": False,
        "x1": 500,
        "y1": 0,
        "x2": 500,
        "y2": 700,
        "duration": 999,
    }
    _validate_action_plan(capped_swipe, image_size=(723, 1600))
    assert capped_swipe["duration"] == scenario_exec_module.MAX_SWIPE_DURATION


def test_repeated_same_direction_swipes_are_stalled():
    screenshot = _screenshot(10)
    trace = [
        {
            "action_name": "swipe",
            "x1": 500,
            "y1": 2000,
            "x2": 500,
            "y2": 600,
            "current_page": Page(id="page_1", representative_screenshot=screenshot),
        }
        for _ in range(4)
    ]

    assert "exceeding same-direction budget" in _detect_stall(trace)


def test_swipe_and_scroll_share_same_direction_budget():
    screenshot = _screenshot(10)
    trace = [
        {
            "action_name": "scroll",
            "direction": "up",
            "current_page": Page(id="page_1", representative_screenshot=screenshot),
        },
        {
            "action_name": "scroll",
            "direction": "up",
            "current_page": Page(id="page_1", representative_screenshot=screenshot),
        },
        {
            "action_name": "swipe",
            "x1": 500,
            "y1": 2000,
            "x2": 500,
            "y2": 600,
            "current_page": Page(id="page_1", representative_screenshot=screenshot),
        },
        {
            "action_name": "swipe",
            "x1": 500,
            "y1": 2000,
            "x2": 500,
            "y2": 600,
            "current_page": Page(id="page_1", representative_screenshot=screenshot),
        },
    ]

    assert "exceeding same-direction budget" in _detect_stall(trace)


def test_execution_prompt_keeps_exploring_after_intermediate_result():
    assert "An intermediate page is not automatically the endpoint" in SCENARIO_EXEC_SYSTEM_PROMPT_TEMPLATE
    assert "direction is complete" in SCENARIO_EXEC_SYSTEM_PROMPT_TEMPLATE
    assert "select a representative result to continue" in SCENARIO_EXEC_SYSTEM_PROMPT_TEMPLATE
    assert "visible virtual numeric/password keypad keys" in SCENARIO_EXEC_SYSTEM_PROMPT_TEMPLATE


def test_audit_prompt_uses_one_representative_for_same_result_controls():
    assert "categories, filters, sorting or styles" in FFG_AUDIT_SYSTEM_PROMPT
    assert "Use one representative for equivalent" in FFG_AUDIT_SYSTEM_PROMPT


def test_single_device_action_failure_can_recover_with_alternate_action(monkeypatch):
    screenshot = _screenshot(10)

    class Response:
        def __init__(self, content):
            self.content = content

    class Llm:
        def __init__(self):
            self.responses = iter([
                Response('{"thought":"\u8f93\u5165\u6570\u5b57","action_name":"type","content":"1234"}'),
                Response('{"thought":"\u6539\u7528\u53ef\u89c1\u952e\u76d8\u540e\u76ee\u6807\u5df2\u8fbe\u6210","action_name":"finished"}'),
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
    monkeypatch.setattr(scenario_exec_module.time, "sleep", lambda _seconds: None)

    trace, _, status, error, _ = scenario_exec_module._run_single_scenario(
        llm=Llm(),
        device=Device(),
        app_name="Example",
        package_name="com.example",
        scenario="\u8f93\u5165\u56db\u4f4d\u6570\u5b57",
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
        '{"action_name":"click","x":52, 70}\n\u4fee\u6b63：\n'
        '{"thought":"\u8fd4\u56de","action_name":"click","x":52,"y":70}'
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
                '{"thought":"\u70b9\u51fb\u5165\u53e3","action_name":"click",'
                '"action_description":"\u70b9\u51fb\u5371\u9669\u5165\u53e3","x":100,"y":200}'
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
        ScreenshotUnavailableError("\u9ed1\u5c4f"),
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

    trace, _, status, error, _ = scenario_exec_module._run_single_scenario(
        llm=llm,
        device=Device(),
        app_name="Example",
        package_name="com.example",
        scenario="\u70b9\u51fb\u5371\u9669\u5165\u53e3\u5e76\u786e\u8ba4\u7ed3\u679c",
        initial_page=Page(id="page_1"),
        clear_logcat=False,
        trace_phase="test",
    )

    assert status == "screenshot_unavailable"
    assert "scenario stopped" in error
    assert llm.calls == 1
    assert len(actions) == 1
    assert len(trace) == 1
    assert trace[0]["action_description"] == "\u70b9\u51fb\u5371\u9669\u5165\u53e3"


def test_runtime_capture_preserves_black_image_but_strict_capture_rejects_it(monkeypatch):
    class Device:
        def screenshot(self):
            return np.zeros((120, 120, 3), dtype=np.uint8)

        def current_activity(self):
            return "MainActivity"

    monkeypatch.setattr("tools.actions.time.sleep", lambda _seconds: None)

    runtime_page, health_error = capture_runtime_page(Device())

    assert runtime_page.representative_screenshot
    assert "black" in health_error
    with pytest.raises(ScreenshotUnavailableError, match="black"):
        capture_current_page(Device())


def test_screenshot_health_check_ignores_status_bar_on_protected_screen():
    # Screenshot-protected app content can be black while the system status bar remains visible.
    # Full-image mean/std statistics previously missed this condition and reported an app defect.
    # Exclude the status bar to detect the protected black content area.
    image = np.zeros((2652, 1200, 3), dtype=np.uint8)
    image[:100, :, :] = 200
    assert "black" in _screenshot_health_error(image)


def test_screenshot_health_check_passes_normal_page_with_dark_status_bar():
    image = np.full((2652, 1200, 3), 180, dtype=np.uint8)
    image[:100, :, :] = 20
    assert _screenshot_health_error(image) == ""


def test_bug_evidence_rejects_screenshot_health_error_steps():
    # Unhealthy screenshots cannot support a defect evidence chain.
    bug = dict(evidence_category="persisted_result_error", evidence=[
        dict(global_step=1, role="action"), dict(global_step=2, role="after")])
    steps = [
        dict(action_name="click", trace_phase="test", action_description="\u70b9\u51fb\u5bc6\u7801\u8f93\u5165\u6846"),
        dict(action_name="wait", trace_phase="test", screenshot_health_error="\u622a\u56fe\u7591\u4f3c\u9ed1\u5c4f"),
    ]
    for step in steps:
        step["current_page"] = Page(representative_screenshot=encode_image_to_jpg_base64(
            np.full((32, 32, 3), 180, dtype=np.uint8)))
    assert not _bug_has_required_trace_evidence(bug, steps)


def test_repeated_runtime_black_screen_restarts_and_resets_counter(monkeypatch):
    black_page = Page(activity="MainActivity", representative_screenshot=_screenshot(20))
    reset_page = Page(id="page_1", activity="MainActivity", representative_screenshot=_screenshot(10))
    scenario_results = iter([
        ([], reset_page, "skipped", "", ""),
        ([{
            "action_name": "capture_screenshot",
            "current_page": black_page,
            "observation_only": True,
            "screenshot_health_error": "\u622a\u56fe\u7591\u4f3c\u9ed1\u5c4f",
        }], black_page, "screenshot_unavailable", "\u9ed1\u5c4f", ""),
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
    updates = execution_pass({
        "app_name": "Example",
        "package_name": "com.example",
        "current_page": reset_page,
        "current_navigation_scenario": "",
        "current_test_scenario": "\u6062\u590d\u540e\u6d4b\u8bd5\u5176\u5b83\u5165\u53e3",
        "current_plan": {"scenario_type": "intra_test"},
        "screenshot_unavailable_count": 1,
    })

    assert restart_calls == ["test"]
    assert updates["current_page"] is reset_page
    assert updates["screenshot_unavailable_count"] == 0
    assert updates["scenario_execution_trace"][-1]["action_name"] == "restart_app"
    assert updates["execution_result"]["scenario_status"] == "screenshot_unavailable"
    assert updates["execution_result"]["scenario_error"] == "\u9ed1\u5c4f"
    assert updates["execution_result"]["scenario_trace_length"] == 2
    assert "test_status" not in updates["execution_result"]
    assert "test_error" not in updates["execution_result"]
    assert "test_trace_length" not in updates["execution_result"]


def test_action_parser_repairs_anonymous_coordinate_between_x_and_y():
    raw = """{
      "thought": "\u8fd4\u56de",
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
    with pytest.raises(ValueError, match="no valid JSON"):
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
    _validate_action_plan(plan, image_size=(723, 1600))


def test_action_validation_rejects_out_of_range_coordinate_and_direction():
    with pytest.raises(ValueError, match="image pixel range"):
        _validate_action_plan({
            "action_name": "click",
            "is_interruption": False,
            "x": 1001,
            "y": 20,
        }, image_size=(723, 1600))
    with pytest.raises(ValueError, match="Invalid scroll direction"):
        _validate_action_plan({
            "action_name": "scroll",
            "is_interruption": False,
            "direction": "forward",
        }, image_size=(723, 1600))


def test_last_image_pixel_maps_to_last_screen_pixel():
    plan = {"x": 722, "y": 1599}

    _scale_plan_coordinates(plan, 1080, 2400, (723, 1600))

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

    assert "interruption recovery" in _detect_stall(trace)


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

    assert "Visual-state alternation" in _detect_stall(trace)


def test_persisted_result_bug_requires_executed_action_not_save_word():
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
            "thought": "\u70b9\u51fb\u6d3b\u52a8\u6309\u94ae\u5e76\u89c2\u5bdf\u8ba1\u65f6",
            "action_description": "\u9009\u62e9\u6d3b\u52a8",
        },
        {"action_name": "wait", "thought": "\u7b49\u5f85\u754c\u9762\u53d8\u5316"},
    ]
    save_trace = [
        {
            "raw_step_index": 1,
            "trace_phase": "test",
            "action_name": "click",
            "thought": "\u70b9\u51fb\u53f3\u4e0a\u89d2\u5bf9\u52fe\u63d0\u4ea4\u65b0\u5efa\u6d3b\u52a8",
            "action_description": "\u4fdd\u5b58\u6d3b\u52a8",
        },
        {
            "raw_step_index": 2,
            "trace_phase": "test",
            "action_name": "finished",
            "thought": "\u89c2\u5bdf\u4fdd\u5b58\u540e\u7684\u6d3b\u52a8\u5217\u8868",
        },
    ]

    for step in observe_only_trace + save_trace:
        step["current_page"] = Page(representative_screenshot=_screenshot(10))
    assert _bug_has_required_trace_evidence(bug, observe_only_trace) is True
    observe_only_trace[0]["action_name"] = "wait"
    assert _bug_has_required_trace_evidence(bug, observe_only_trace) is False
    assert _bug_has_required_trace_evidence(bug, save_trace) is True


def test_finished_via_different_entry_is_not_treated_as_success():
    thought = "\u539f\u94c5\u7b14\u5165\u53e3\u4e0d\u7b26\u5408\u4efb\u52a1\u9884\u671f，\u6539\u7528\u66ff\u4ee3\u5165\u53e3\u52a0\u53f7\u540e\u76ee\u6807\u5df2\u8fbe\u6210。"

    assert _finished_declares_failure(thought) is True


def test_finished_with_success_and_unreachable_subgoal_is_failure():
    thought = "\u76ee\u6807\u5df2\u8fbe\u6210；\u4f46\u6587\u4ef6\u53d1\u9001\u5165\u53e3\u4e0d\u53ef\u8fbe，\u5269\u4f59\u9a8c\u8bc1\u672a\u5b8c\u6210。"

    assert _finished_declares_failure(thought) is True


def test_finished_with_all_validation_points_satisfied_is_success_despite_negative_notice():
    thought = "\u51fa\u73b0\u4e86 CAN'T SAVE AN EMPTY NOTE \u7684\u63d0\u793a，\u4f46\u4e0d\u5f71\u54cd\u8df3\u8f6c\u7ed3\u679c，\u4efb\u52a1\u7684\u6240\u6709\u9a8c\u8bc1\u70b9\u5747\u5df2\u6ee1\u8db3。"

    assert _finished_declares_failure(thought) is False


def test_finished_with_all_task_goals_reached_is_success_despite_negative_notice():
    thought = (
        "\u9876\u90e8\u63d0\u793a CAN'T SAVE AN EMPTY NOTE \u7b26\u5408\u7a7a\u7b14\u8bb0\u65e0\u6cd5\u4fdd\u5b58\u7684\u9884\u671f，"
        "\u6240\u6709\u4efb\u52a1\u76ee\u6807\u5747\u8fbe\u6210。"
    )

    assert _finished_declares_failure(thought) is False


def test_coverage_audit_keeps_distinct_creation_content_types():
    assert "Text, checklists, camera/photos and files" in FFG_AUDIT_SYSTEM_PROMPT
    assert "need separate actions when they open different flows" in FFG_AUDIT_SYSTEM_PROMPT


def test_planning_output_clears_previous_execution_payload():
    planning_node = _create_scenario_plan_node_with_iteration(
        lambda _state: {
            "current_plan": {"scenario_type": "skip"},
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
    graph = FunctionalFlowGraph()
    graph.add_node(FunctionalityNode(id="func_1", test_status="in_progress"))
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
        "current_test_scenario": "\u70b9\u51fb\u4fdd\u5b58\u6309\u94ae\u5e76\u89c2\u5bdf\u7ed3\u679c",
        "execution_result": {
            "navigation_status": "skipped",
            "navigation_trace_length": 0,
            "scenario_status": "completed",
            "scenario_trace_length": 2,
            "current_scenario": "\u4e0d\u5e94\u91cd\u590d\u6253\u5370\u7684\u5b8c\u6574\u573a\u666f",
        },
        "ffg_update_decision": "finish",
    }

    _print_graph_step_status("scenario_plan", state)
    _print_graph_step_status("scenario_exec", state)
    _print_graph_step_status("ffg_update", state)

    output = capsys.readouterr().out
    assert "[State] scenario_plan | type=intra_test | scope=func_1" in output
    assert "[Plan] Scenario: \u70b9\u51fb\u4fdd\u5b58\u6309\u94ae\u5e76\u89c2\u5bdf\u7ed3\u679c" in output
    assert "navigation=skipped(0 steps) | scenario=completed(2 steps) | page=page_1" in output
    assert "decision=finish | graph=1F/1P | current_function=func_1(in_progress)" in output
    assert "\u4e0d\u5e94\u91cd\u590d\u6253\u5370\u7684\u5b8c\u6574\u573a\u666f" not in output
    assert "Execution result: {" not in output


def test_skip_round_is_not_replanned_inside_the_same_iteration():
    """Apply state transitions at planning entry without replanning a skipped iteration outside the node."""
    calls = []

    def planner(state):
        calls.append(dict(state))
        return {
            "current_plan": {
                "scenario_type": "skip",
                "target_function_id": "func_1",
            },
            "current_testing_function_id": "",
            "current_testing_function_status": "not_started",
            "workflow_complete": False,
        }

    wrapped = _create_scenario_plan_node_with_iteration(planner)
    result = wrapped({"iteration_count": 0, "run_phase": "all"})

    assert len(calls) == 1
    assert result["iteration_count"] == 1
    assert result["current_plan"]["scenario_type"] == "skip"


def test_current_page_identity_uses_closest_recorded_screenshot():
    graph = FunctionalFlowGraph()
    graph.add_node(FunctionalityNode(id="func_1"))
    graph.add_page(
        Page(
            id="page_1",
            name="\u6a21\u5f0f\u4e00",
            activity="Calculator",
            representative_screenshot=_screenshot(10),
        ),
        "func_1",
    )
    graph.add_page(
        Page(
            id="page_2",
            name="\u6a21\u5f0f\u4e8c",
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
    assert resolved.name == "\u6a21\u5f0f\u4e8c"


def test_app_data_reset_is_explicitly_confirmed_at_run_start():
    class Connector:
        def __init__(self, output):
            self.output = output
            self.commands = []

        def shell(self, command: str):
            self.commands.append(command)
            return self.output

    class Device:
        operating_system = "adb"

        def __init__(self, output):
            self.connector = Connector(output)

    success = Device("Success")
    assert _reset_android_app_data_at_run_start(success, "com.example") is True
    assert success.connector.commands == ["pm clear com.example"]

    failed = Device("Failed")
    assert _reset_android_app_data_at_run_start(failed, "com.example") is False


def test_compliance_runtime_has_no_pm_clear_path():
    source = Path(scenario_exec_module.__file__).read_text(encoding="utf-8")
    assert "pm clear" not in source


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

    # Unified CLI defaults to an independent defect run.
    assert default_args.bug_detection is True
    assert disabled_args.bug_detection is False
    assert default_args.compliance is False
    assert disabled_args.compliance is False


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
    graph = FunctionalFlowGraph()
    graph.add_node(FunctionalityNode(id="func_1", name="\u4e3b\u529f\u80fd", test_status="completed"))

    def planner(state):
        assert state["ffg"] is graph
        return {
            "ffg": graph,
            "workflow_complete": True,
            "current_plan": {"scenario_type": "done"},
        }

    monkeypatch.setattr("workflow.graph.LONG_RUN_IDLE_SECONDS", 0)
    # Wait for pending page audits because they may discover new omissions.
    monkeypatch.setattr("workflow.graph._merge_page_audit_results", lambda graph: 0)
    monkeypatch.setattr("workflow.graph._submit_page_audit_batch", lambda graph, **_: 0)
    monkeypatch.setattr("workflow.graph.page_audit_worker.pending_count", lambda: 1)
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


def test_time_budget_run_ends_early_once_audits_are_exhausted(monkeypatch):
    """Finish early when audit work is exhausted instead of idling for the remaining time budget."""
    graph = FunctionalFlowGraph()
    graph.add_node(FunctionalityNode(id="func_1", name="\u4e3b\u529f\u80fd", test_status="completed"))

    def planner(state):
        return {
            "ffg": graph,
            "workflow_complete": True,
            "current_plan": {"scenario_type": "done"},
        }

    monkeypatch.setattr("workflow.graph.LONG_RUN_IDLE_SECONDS", 0)
    monkeypatch.setattr("workflow.graph._merge_page_audit_results", lambda graph: 0)
    monkeypatch.setattr("workflow.graph._submit_page_audit_batch", lambda graph, **_: 0)
    monkeypatch.setattr("workflow.graph.page_audit_worker.pending_count", lambda: 0)
    wrapped = _create_scenario_plan_node_with_iteration(planner)
    result = wrapped({
        "ffg": graph,
        "iteration_count": 4,
        "start_time": time.time(),
        "total_time_limit": 3600,
        "max_iterations": 0,
    })

    assert result["workflow_complete"] is True
    assert result["long_run_idle_count"] == 0


def test_time_budget_run_continues_when_audit_just_changed_the_graph(monkeypatch):
    graph = FunctionalFlowGraph()
    graph.add_node(FunctionalityNode(id="func_1", name="\u4e3b\u529f\u80fd", test_status="completed"))

    def planner(state):
        return {"ffg": graph, "workflow_complete": True, "current_plan": {"scenario_type": "done"}}

    monkeypatch.setattr("workflow.graph.LONG_RUN_IDLE_SECONDS", 0)
    monkeypatch.setattr("workflow.graph._merge_page_audit_results", lambda graph: 2)
    monkeypatch.setattr("workflow.graph._submit_page_audit_batch", lambda graph, **_: 0)
    monkeypatch.setattr("workflow.graph.page_audit_worker.pending_count", lambda: 0)
    result = _create_scenario_plan_node_with_iteration(planner)({
        "ffg": graph,
        "iteration_count": 4,
        "start_time": time.time(),
        "total_time_limit": 3600,
        "max_iterations": 0,
    })

    assert result["workflow_complete"] is False


def test_final_artifact_set_is_complete(tmp_path):
    graph = FunctionalFlowGraph()
    graph.add_node(FunctionalityNode(id="func_1", name="\u4e3b\u529f\u80fd"))
    graph.add_page(Page(
        id="page_1",
        representative_screenshot=_screenshot(12),
        controls=["\u53ef\u70b9\u51fb\u7684‘\u65b0\u589e’\u6309\u94ae"],
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
        "title": "\u6309\u94ae\u65e0\u54cd\u5e94",
    }
    trace = [{
        "action_name": "click",
        "x": 100,
        "y": 200,
        "current_page": Page(
            id="page_1",
            name="\u8be6\u60c5\u9875",
            representative_screenshot=_screenshot(8),
        ),
    }]
    state = {
        "bug_trace_records": {bug["bug_id"]: trace},
        "scenario_execution_trace": trace,
    }

    _save_bug_folders(state, str(tmp_path), [bug], "[\u6d4b\u8bd5]")
    bug_dir = tmp_path / "bugs" / bug["bug_id"]
    assert (bug_dir / ".artifact-fingerprint").is_file()
    assert (bug_dir / "trace.json").is_file()
    assert (bug_dir / "screenshots" / "step_001.jpg").is_file()

    monkeypatch.setattr(
        "main._atomic_write_json",
        lambda *_args, **_kwargs: pytest.fail("\u672a\u53d8\u5316 bug \u4e0d\u5e94\u91cd\u5199 JSON"),
    )
    monkeypatch.setattr(
        "main._atomic_write_bytes",
        lambda *_args, **_kwargs: pytest.fail("\u672a\u53d8\u5316 bug \u4e0d\u5e94\u91cd\u5199\u622a\u56fe\u6216\u6307\u7eb9"),
    )

    _save_bug_folders(state, str(tmp_path), [bug], "[\u6d4b\u8bd5]")


def test_activity_timeline_accumulates_unique_discovered_activities(tmp_path):
    graph = FunctionalFlowGraph()
    graph.add_node(FunctionalityNode(id="func_1", name="\u4e3b\u529f\u80fd"))
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
        assert "WeChat rules" not in prompt


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
        app_name="\u5fae\u4fe1",
        package_name="com.tencent.mm",
    )

    # Restart an already foreground app to leave a deep Activity and return to its entry.
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
        app_name="\u5fae\u4fe1",
        package_name="com.tencent.mm",
    )

    # If the app is not foreground, launch without force-stopping it.
    # This prevents repeated open_app calls during cold start from creating a restart storm.
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

    with pytest.raises(RuntimeError, match="open the target app on the device first"):
        _resolve_target_app(args, Device())


def test_planner_rejects_screen_when_restarted_app_is_not_foreground(monkeypatch):
    class Device:
        def current_package(self):
            return "com.android.launcher3"

        def restart_app(self, _package_name):
            return None

    monkeypatch.setattr("workflow.nodes.scenario_plan.time.sleep", lambda _seconds: None)
    # Foreground polling uses monotonic deadlines, so mocking sleep alone still spins to the limit.
    # Reduce the limit to 0.2 seconds without changing the behavior under test.
    monkeypatch.setattr("workflow.nodes.scenario_plan.APP_START_MAX_WAIT", 0.2)

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
    expected = Page(id="runtime", name="\u4e3b\u9875")
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
    expected = Page(id="runtime", name="\u4e3b\u9875")
    captures = iter([ScreenshotUnavailableError("\u9ed1\u5c4f"), expected])
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
    expected = Page(id="runtime", name="\u4e3b\u9875")
    captures = iter([
        ScreenshotUnavailableError("\u521d\u59cb\u9ed1\u5c4f"),
        ScreenshotUnavailableError("\u8fd4\u56de\u540e\u4ecd\u9ed1\u5c4f"),
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
    graph = FunctionalFlowGraph()
    graph.add_node(FunctionalityNode(id="func_1", name="\u8bbe\u7f6e\u529f\u80fd", test_status="in_progress"))
    graph.add_page(Page(
        id="page_1",
        name="\u8bbe\u7f6e\u9875",
        controls=["\u53ef\u70b9\u51fb\u7684\u901a\u77e5\u8bbe\u7f6e\u5165\u53e3"],
    ), "func_1")
    graph.add_page(Page(
        id="page_2",
        name="\u9ed1\u5c4f\u6216\u9875\u9762\u4e0d\u53ef\u7528",
        description="\u5f02\u5e38\u7ec8\u70b9",
    ), "func_1")
    graph.add_page_transition(PageTransition(
        src_page_id="page_1",
        dst_page_id="page_2",
        operation="\u70b9\u51fb\u5371\u9669\u5165\u53e3；\u7ed3\u679c：\u9ed1\u5c4f\u6216\u9875\u9762\u4e0d\u53ef\u7528（\u622a\u56fe\u5065\u5eb7\u68c0\u67e5\u5931\u8d25）",
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
                "navigation_scenario": "\u5148\u6309\u8fd4\u56de\u952e\u9000\u51fa\u5f53\u524d\u9ed1\u5c4f\u754c\u9762",
                "test_scenario": "\u70b9\u51fb\u901a\u77e5\u8bbe\u7f6e\u5165\u53e3 → \u5207\u6362\u4e00\u6b21\u901a\u77e5\u72b6\u6001 → \u786e\u8ba4\u72b6\u6001\u53d8\u5316\u540e\u7ed3\u675f",
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
    assert "press_back" in result["current_navigation_scenario"]
    assert "after recovery" not in result["current_navigation_scenario"].lower()
    assert "\u901a\u77e5\u8bbe\u7f6e\u5165\u53e3" in result["current_test_scenario"]
    assert result["current_page"].representative_screenshot == current_page.representative_screenshot


def test_planner_rejects_scenario_that_repeats_known_black_screen_path():
    graph = FunctionalFlowGraph()
    graph.add_node(FunctionalityNode(id="func_1", name="\u8bbe\u7f6e\u529f\u80fd", test_status="in_progress"))
    graph.add_page(Page(id="page_1", name="\u8bbe\u7f6e\u9875"), "func_1")
    graph.add_page(Page(id="page_2", name="\u9ed1\u5c4f\u6216\u9875\u9762\u4e0d\u53ef\u7528"), "func_1")
    graph.add_page_transition(PageTransition(
        src_page_id="page_1",
        dst_page_id="page_2",
        operation="\u70b9\u51fb\u5371\u9669\u5165\u53e3；\u7ed3\u679c：\u9ed1\u5c4f\u6216\u9875\u9762\u4e0d\u53ef\u7528（\u622a\u56fe\u5065\u5eb7\u68c0\u67e5\u5931\u8d25）",
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
                "navigation_scenario": "\u5148\u6309\u8fd4\u56de\u952e\u9000\u51fa\u5f53\u524d\u9ed1\u5c4f\u754c\u9762",
                "test_scenario": "\u518d\u6b21\u70b9\u51fb\u5371\u9669\u5165\u53e3\u5e76\u786e\u8ba4\u7ed3\u679c",
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

    # Reject a repeated black-screen route; use an observation template for a zero-control functionality.
    assert result["current_plan"]["scenario_type"] == "intra_test"
    assert "\u5371\u9669\u5165\u53e3" not in result["current_test_scenario"]
    assert "Observe" in result["current_test_scenario"]


def test_executor_restart_requires_target_app_to_reach_foreground(monkeypatch):
    class Device:
        def restart_app(self, _package_name):
            return None

        def current_package(self):
            return "com.android.launcher3"

    monkeypatch.setattr("workflow.nodes.scenario_exec.time.sleep", lambda _seconds: None)
    # The executor passes max_wait=None and uses app_foreground.APP_START_MAX_WAIT.
    # Reduce that fallback to 0.2 seconds to avoid idle waiting.
    monkeypatch.setattr("workflow.app_foreground.APP_START_MAX_WAIT", 0.2)

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
        "_wait_target_app_foreground",
        lambda current, _package_name: current.restarts >= 2,
    )

    assert _restart_target_app(device, "com.example") is True
    assert device.restarts == 2


def test_untrusted_external_app_is_recovered_and_scene_stops(monkeypatch):
    # Classify a true external transition only after foreground settling (at most four checks)
    # still reports an app other than the target.
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

    trace, _, status, error, _ = scenario_exec_module._run_single_scenario(
        llm=object(),
        device=object(),
        app_name="Example",
        package_name="com.example",
        scenario="\u70b9\u51fb\u8bbe\u7f6e\u6309\u94ae\u5e76\u89c2\u5bdf\u7ed3\u679c",
        initial_page=Page(id="page_start"),
        clear_logcat=False,
        trace_phase="test",
    )

    assert status == "left_app"
    assert error == scenario_exec_module.EXTERNAL_TRANSITION_BLOCKED_END
    assert trace[-1]["foreground_recovery"] is True
    assert trace[-1]["recovery_method"] == "press_back"
    assert trace[-2]["external_context"] is True
    assert trace[-2]["external_package"] == "com.untrusted.app"
    assert trace[-2]["observation_only"] is True


@pytest.mark.parametrize(
    "scenario",
    [
        "\u5230\u8fbe\u516c\u4f17\u53f7\u6587\u7ae0\u8be6\u60c5\u9875\u5e76\u6253\u5f00\u66f4\u591a\u64cd\u4f5c\u9762\u677f → \u70b9\u51fb‘\u5728\u6d4f\u89c8\u5668\u4e2d\u6253\u5f00’ → \u786e\u8ba4\u5b9e\u9645\u7ed3\u679c\u540e\u7ed3\u675f",
        "\u70b9\u51fb\u94fe\u63a5\u5e76\u8df3\u8f6c\u5230\u9ed8\u8ba4\u6d4f\u89c8\u5668\u67e5\u770b\u7f51\u9875",
        "\u4f7f\u7528 browser \u6253\u5f00\u9875\u9762\u5e76\u786e\u8ba4\u6807\u9898",
    ],
)
def test_explicit_browser_scenarios_allow_controlled_external_context(scenario):
    assert scenario_exec_module._scenario_allows_external_app(scenario) is True


def test_target_app_recovery_wording_does_not_allow_external_context():
    scenarios = [
        "\u5148\u6267\u884c\u4e00\u6b21 open_app \u76f4\u63a5\u6253\u5f00\u76ee\u6807\u5e94\u7528\u4e3b\u9875\u9762；\u82e5\u79bb\u5f00\u76ee\u6807\u5e94\u7528\u5219\u518d\u6b21 open_app",
        "\u4e3a\u5230\u8fbe\u76ee\u6807\u9875\u800c\u4e3b\u52a8\u6253\u5f00\u7684\u83dc\u5355\u5c5e\u4e8e\u5bfc\u822a\u8def\u5f84，\u4e0d\u8981\u518d\u628a\u5b83\u5173\u95ed",
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

    _, _, status, error, _ = scenario_exec_module._run_single_scenario(
        llm=object(),
        device=object(),
        app_name="Example",
        package_name="com.example",
        scenario="\u70b9\u51fb\u8bbe\u7f6e\u6309\u94ae\u5e76\u89c2\u5bdf\u7ed3\u679c",
        initial_page=Page(id="page_start"),
        clear_logcat=False,
        trace_phase="test",
    )

    assert status == "left_app"
    assert error == scenario_exec_module.TARGET_APP_RECOVERY_FAILED_END


def test_emulator_egl_failure_is_not_reported_as_app_bug():
    crash_logcat = (
        "E/OpenGLRenderer: Failed to choose config, error = EGL_NOT_INITIALIZED\n"
        "F/libc: Fatal signal 6 in RenderThread\n"
        "F/DEBUG: >>> com.example <<<"
    )

    bug = _runtime_crash_bug(
        {
            "navigation_status": "app_crashed",
            "scenario_status": "skipped",
            "crash_logcat": crash_logcat,
        },
        [{"action_name": "click"}],
        "\u542f\u52a8\u5e94\u7528",
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
            "scenario_status": "app_crashed",
            "crash_detected": True,
            "crash_logcat": crash_logcat,
        },
        [{"action_name": "click"}],
        "\u6253\u5f00\u8bbe\u7f6e",
        "com.example",
    )

    assert bug is not None
    assert bug["type"] == "crash"
    assert bug["source"] == "runtime_monitor"


def test_planner_allows_scenario_regardless_of_action_count():
    """Exploration directions are not rejected by counting their actions."""
    result = {
        "workflow_complete": False,
        "scenario_type": "intra_test",
        "target_function": "func_1",
        "src_function": "",
        "dst_function": "",
        "navigation_scenario": "",
        "test_scenario": "，".join(["\u70b9\u51fb\u6309\u94ae"] * 9),
    }

    assert not any(
        error.startswith("test_scenario_too_many_actions=")
        for error in _validate_plan_result(result)
    )


@pytest.mark.parametrize(
    "scenario",
    [
        "\u70b9\u51fb\u6240\u6709\u8868\u60c5\u7f51\u683c\u4e2d\u7684\u8868\u60c5 → \u70b9\u51fb\u53d1\u9001\u6309\u94ae",
        "\u9010\u4e2a\u9009\u62e9\u6bcf\u4e2a\u7b5b\u9009\u6807\u7b7e",
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
        "test_scenario": "\u70b9\u51fb\u5168\u9009\u6309\u94ae → \u786e\u8ba4\u5168\u90e8\u5185\u5bb9\u88ab\u9009\u4e2d",
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
        "test_scenario": "\u4f9d\u6b21\u70b9\u51fb“\u901a\u77e5”、“\u9690\u79c1”\u548c“\u901a\u7528”，\u5206\u522b\u89c2\u5bdf\u76f4\u63a5\u7ed3\u679c\u540e\u8fd4\u56de",
    }

    assert "test_scenario_exhaustive_control_family" not in _validate_plan_result(result)


def test_plan_result_only_requires_test_scenario():
    # The program fixes the scope; the LLM returns only test_scenario.
    assert _validate_plan_result({"test_scenario": "\u70b9\u51fb\u6570\u5b57\u5e76\u89c2\u5bdf\u7ed3\u679c"}) == []
    assert "missing_fields=['test_scenario']" in _validate_plan_result({"note": ""})
    assert _validate_plan_result({"test_scenario": ""}, scenario_type="cross_test")


def test_wechat_policy_allows_conversation_state_mutation():
    plan = {
        "workflow_complete": False,
        "scenario_type": "intra_test",
        "target_function": "func_1",
        "src_function": "",
        "dst_function": "",
        "navigation_scenario": "",
        "test_scenario": "\u957f\u6309\u4f1a\u8bdd\u5e76\u70b9\u51fb\u4e0d\u663e\u793a\u8be5\u804a\u5929",
    }

    errors = _validate_plan_result(plan, package_name="com.tencent.mm")

    assert not any(error.startswith("wechat_read_only_policy=") for error in errors)
    assert not wechat_read_only_violation("com.example.app", plan["test_scenario"])


def test_scenario_plan_no_longer_rejects_unknown_dynamic_resource():
    """Leave dynamic-resource constraints to the prompt and executor rather than rejecting them in planning code."""
    plan = {
        "scenario_type": "intra_test",
        "target_function": "func_1",
        "src_function": "",
        "dst_function": "",
        "navigation_scenario": "",
        "test_scenario": "\u63a2\u7d22\u641c\u7d22\u5165\u53e3\u5e76\u67e5\u770b\u4e00\u4e2a\u76f4\u63a5\u7ed3\u679c，\u65e0\u660e\u786e\u540e\u7eed\u65f6\u7ed3\u675f",
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
        "\u5fae\u4fe1",
        "com.tencent.mm",
        "\u8fdb\u5165\u8bbe\u7f6e\u5e76\u70b9\u51fb\u9000\u51fa\u767b\u5f55",
        None,
        False,
        "test",
    )

    trace, _page, status, error, _logcat = result
    assert trace == []
    assert status == "blocked"
    assert "logout or account deletion" in error


def test_executor_blocks_verification_attempt_but_allows_exit_action():
    assert _verification_attempt_reason(
        "scroll",
        "\u62d6\u52a8\u62fc\u56fe\u6ed1\u5757\u901a\u8fc7\u5b89\u5168\u9a8c\u8bc1",
    )
    assert _verification_attempt_reason(
        "click",
        "\u70b9\u51fb\u53f3\u4e0a\u89d2\u5173\u95ed\u6ed1\u5757\u9a8c\u8bc1\u5f39\u7a97",
    ) == ""
    assert _verification_finish_reason("\u9a8c\u8bc1\u7801\u963b\u585e，\u5df2\u9000\u51fa\u5f53\u524d\u9875\u9762")
    assert _verification_finish_reason("\u5df2\u6210\u529f\u901a\u8fc7\u9a8c\u8bc1\u5e76\u8fdb\u5165\u4e0b\u4e00\u9875") == ""


def test_ordinary_slider_control_is_not_a_verification_challenge():
    """An ordinary ruler slider must not be mistaken for human verification or make the functionality unavailable."""
    assert _verification_attempt_reason(
        "swipe",
        "\u5f53\u524d\u5728\u523b\u5ea6\u5c3a\u9875，\u6ed1\u5757\u4f4d\u4e8e0\u7aef，\u8bfb\u65700cm/0inch，\u5148\u62d6\u52a8\u6ed1\u5757\u5230\u4e2d\u95f4\u523b\u5ea6\u89c2\u5bdf\u8054\u52a8",
    ) == ""
    assert _verification_attempt_reason("swipe", "\u62d6\u52a8\u4eae\u5ea6\u6ed1\u5757\u5230\u6700\u53f3\u4fa7") == ""
    assert _verification_finish_reason("\u6ed1\u5757\u5df2\u62d6\u5230\u6700\u5927\u523b\u5ea6，\u8bfb\u6570\u6b63\u5e38") == ""
    assert _verification_attempt_reason("swipe", "\u62d6\u52a8\u6ed1\u5757\u9a8c\u8bc1\u8bfb\u6570\u8054\u52a8") == ""
    assert _verification_attempt_reason("swipe", "\u62d6\u52a8\u5b57\u4f53\u5927\u5c0f\u6ed1\u5757，\u518d\u8fd4\u56de\u7f16\u8f91\u9875\u9a8c\u8bc1\u914d\u7f6e\u4fdd\u7559") == ""
    assert _verification_attempt_reason("swipe", "\u5411\u53f3\u62d6\u52a8\u6ed1\u5757\u5b8c\u6210\u9a8c\u8bc1")
    assert _verification_attempt_reason("swipe", "\u51fa\u73b0\u6ed1\u5757\u9a8c\u8bc1，\u5c1d\u8bd5\u62d6\u52a8")
    assert _verification_attempt_reason("swipe", "\u62d6\u52a8\u62fc\u56fe\u5230\u7f3a\u53e3\u5904")
    assert _verification_finish_reason("\u51fa\u73b0\u6ed1\u52a8\u9a8c\u8bc1\u7801，\u65e0\u6cd5\u7ee7\u7eed")


def test_executor_cannot_finish_on_required_system_dialog():
    reason = _unfinished_safe_external_reason(
        "finished",
        "com.android.permissioncontroller",
    )
    assert "before finished" in reason
    assert _unfinished_safe_external_reason(
        "click",
        "com.android.permissioncontroller",
    ) == ""
    assert _unfinished_safe_external_reason("finished", "com.example.other") == ""


def test_blocked_page_exit_presses_back_once_and_records_recovery(monkeypatch):
    actions: list[str] = []
    recovered_page = Page(id="page_recovered", name="\u6062\u590d\u9875\u9762")
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
        "\u6240\u6709\u6307\u5b9a\u4f1a\u8bdd\u6761\u76ee\u548c\u672a\u8bfb\u6807\u8bb0\u5747\u5df2\u786e\u8ba4，\u9875\u9762\u65e0\u52a0\u8f7d\u5f02\u5e38，"
        "\u5df2\u5b8c\u6210\u5168\u90e8\u57fa\u7840\u72b6\u6001\u6821\u9a8c。"
    )

    assert not _finished_declares_failure(thought)


def test_wecreate_action_alias_and_bounding_box_are_normalized():
    plan = _normalize_action_plan({
        "action": "click",
        "thought": "\u70b9\u51fb\u8fd4\u56de",
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
        "target": "\u8bbe\u7f6e",
        "x": 100,
        "y": 200,
    })

    assert plan["action_name"] == "click"
    assert plan["action_description"] == "click \u8bbe\u7f6e"


def test_incomplete_next_action_is_rejected_before_device_execution():
    plan = _normalize_action_plan({
        "next_action": "click",
        "target": "\u8bbe\u7f6e",
    })

    with pytest.raises(ValueError, match="requires x/y coordinates"):
        _validate_action_plan(plan, image_size=(723, 1600))


def test_action_intent_excludes_unrelated_visible_page_description():
    intent = _proposed_action_text({
        "thought": (
            "\u9875\u9762\u5305\u542b\u6d88\u606f\u901a\u77e5、\u8bed\u97f3\u548c\u89c6\u9891\u901a\u8bdd\u901a\u77e5\u7b49\u8bbe\u7f6e\u9879。"
            "\u4e0b\u4e00\u6b65\u9700\u8981\u70b9\u51fb\u9876\u90e8\u5de6\u4fa7\u8fd4\u56de\u6309\u94ae。"
        ),
    })

    assert "\u8fd4\u56de\u6309\u94ae" in intent
    assert "\u89c6\u9891\u901a\u8bdd" not in intent


def test_wechat_policy_allows_setting_switch_and_visible_call_text():
    assert not wechat_read_only_violation(
        "com.tencent.mm",
        "\u70b9\u51fb\u4f7f\u7528\u542c\u7b52\u64ad\u653e\u8bed\u97f3\u6d88\u606f\u5f00\u5173，\u9a8c\u8bc1\u5f00\u5173\u72b6\u6001\u53d1\u751f\u5207\u6362",
    )
    assert not wechat_read_only_violation(
        "com.tencent.mm",
        "\u9875\u9762\u5305\u542b\u6d88\u606f\u901a\u77e5、\u8bed\u97f3\u548c\u89c6\u9891\u901a\u8bdd\u901a\u77e5\u7b49\u8bbe\u7f6e\u9879，\u4e0b\u4e00\u6b65\u70b9\u51fb\u5de6\u4e0a\u89d2\u8fd4\u56de",
    )


def test_wechat_policy_allows_setting_profile_status_but_blocks_account_deletion():
    assert not wechat_read_only_violation("com.tencent.mm", "\u70b9\u51fb+ \u72b6\u6001\u5e76\u8bbe\u7f6e\u5fae\u4fe1\u72b6\u6001")
    assert wechat_read_only_violation("com.tencent.mm", "\u8fdb\u5165\u8d26\u53f7\u8bbe\u7f6e\u5e76\u5220\u9664\u8d26\u6237")


def test_wechat_action_policy_only_blocks_logout_or_account_deletion():
    assert not wechat_action_policy_violation(
        "com.tencent.mm",
        "click",
        "\u70b9\u51fb\u5e95\u90e8\u5bfc\u822a\u680f\u7684\u901a\u8baf\u5f55\u6807\u7b7e",
    )
    assert not wechat_action_policy_violation(
        "com.tencent.mm",
        "click",
        "\u70b9\u51fb\u53d1\u73b0",
    )
    assert not wechat_action_policy_violation(
        "com.tencent.mm",
        "click",
        "\u70b9\u51fb\u6587\u4ef6\u4f20\u8f93\u52a9\u624b\u4f1a\u8bdd",
    )
    assert not wechat_action_policy_violation(
        "com.tencent.mm",
        "type",
        "\u5728\u641c\u7d22\u6846\u8f93\u5165\u6587\u672c",
    )
    assert wechat_action_policy_violation(
        "com.tencent.mm",
        "click",
        "\u70b9\u51fb\u9000\u51fa\u767b\u5f55",
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
        "current_test_scenario": "\u8d85\u957f\u573a\u666f",
        "scenario_execution_trace": [{
            "action_name": "click",
            "current_page": Page(id="page_1", representative_screenshot=_screenshot(10)),
        }],
        "navigation_execution_trace": [],
        "execution_result": {
            "navigation_status": "skipped",
            "scenario_status": "incomplete",
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
        "current_test_scenario": "\u70b9\u51fb\u5165\u53e3\u5e76\u786e\u8ba4\u9875\u9762",
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
        content = '\u5206\u6790\u5982\u4e0b：\n```json\n{"decision":"finish"}\n```\n\u4ee5\u4e0a。'

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


def test_json_retry_helper_keeps_outer_object_instead_of_last_nested_evidence():
    class Response:
        content = (
            '\u5224\u5b9a\u5982\u4e0b：\n'
            '{"check_results":[{"check_id":"R014","verdict":"compliant",'
            '"evidence":[{"global_step":1,"role":"before"}]}]}\n'
            '\u4ee5\u4e0a。'
        )

    class Llm:
        def invoke(self, _messages):
            return Response()

    result = invoke_llm_json_with_retry(
        Llm(),
        [],
        max_attempts=1,
        retry_delay_seconds=0,
    )

    assert result["check_results"][0]["check_id"] == "R014"


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


def test_ffg_update_context_separates_plan_facts_and_before_step_evidence():
    screenshot = _screenshot(20)
    graph = FunctionalFlowGraph()
    graph.add_node(FunctionalityNode(id="func_1", name="\u79d1\u5b66\u8ba1\u7b97"))
    graph.add_page(Page(id="page_1", name="\u8ba1\u7b97\u5668", representative_screenshot=screenshot), "func_1")

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
        "current_test_scenario": "\u70b9\u51fb\u6570\u5b57\u5e76\u89c2\u5bdf\u663e\u793a",
        "scenario_execution_trace": [
            {
                "action_name": "click",
                "x": 100,
                "y": 200,
                "action_description": "\u70b9\u51fb\u6570\u5b57 1",
                "thought": "\u6570\u5b57\u952e\u4f4d\u4e8e\u5de6\u4e0b\u533a\u57df",
                "current_page": Page(
                    id="page_1",
                    name="\u8ba1\u7b97\u5668",
                    activity="Calculator",
                    representative_screenshot=screenshot,
                ),
            },
            {
                "action_name": "finished",
                "action_description": "\u7ed3\u675f\u5e76\u8bb0\u5f55\u7ed3\u679c",
                "current_page": Page(
                    id="page_1",
                    name="\u8ba1\u7b97\u5668",
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
    assert "[Authoritative planning scope]" in text_content
    assert '"id": "func_1"' in text_content
    assert "coordinate=(100,200)" in text_content
    assert "executor_target_hint=\u70b9\u51fb\u6570\u5b57 1" in text_content
    assert "[FFG_REFERENCE_IMAGE] page_id=page_1" in text_content
    assert "[OBSERVATION] BEFORE_STEP_1" in text_content
    assert "source_step=3" in text_content
    assert "finished is a final observation" in text_content


def test_ffg_update_rejects_filtered_ordinal_as_source_step():
    graph = FunctionalFlowGraph()
    graph.add_node(FunctionalityNode(id="func_1", name="\u4e3b\u529f\u80fd"))
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
                "name": "\u9519\u8bef\u6620\u5c04\u9875\u9762",
                "description": "\u4e0d\u5e94\u6309\u8fc7\u6ee4\u540e\u5e8f\u53f7\u6620\u5c04",
            }],
        },
        trace,
    )

    assert result["page_refs"] == {}
    assert graph.pages == {}


def test_ffg_update_does_not_auto_reuse_page_across_functions():
    graph = FunctionalFlowGraph()
    graph.add_node(FunctionalityNode(id="func_1", name="\u6587\u672c\u7b14\u8bb0"))
    graph.add_node(FunctionalityNode(id="func_2", name="\u5f85\u529e\u6e05\u5355"))
    screenshot = _screenshot(20)
    graph.add_page(Page(id="page_1", name="\u6587\u672c\u7f16\u8f91\u9875", representative_screenshot=screenshot), "func_1")

    result = apply_ffg_patch(
        graph,
        {
            "pages": [{
                "page_ref": "new_page_1",
                "function_refs": ["func_2"],
                "source_step": 1,
                "name": "\u5f85\u529e\u6e05\u5355\u7f16\u8f91\u9875",
                "description": "\u663e\u793a\u5f85\u529e\u9879\u76ee\u590d\u9009\u6846",
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
    assert "one or two representatives among equivalent interactive controls" in FFG_UPDATE_SYSTEM_PROMPT
    assert "Equivalent thumbnails/messages/contacts/files/themes/backgrounds/avatars aggregate by type" in FFG_UPDATE_SYSTEM_PROMPT


def test_obsolete_correction_patch_is_ignored():
    graph = FunctionalFlowGraph()
    graph.add_node(FunctionalityNode(id="func_1", name="\u7a33\u5b9a\u529f\u80fd"))

    result = apply_ffg_patch(
        graph,
        {"corrections": [{"entity_ref": "func_1", "operation": "deprecate"}]},
        [{"raw_step_index": 1, "current_page": Page(representative_screenshot=_screenshot(10))}],
    )

    assert "corrections_patch" not in result
    assert graph.get_node("func_1").name == "\u7a33\u5b9a\u529f\u80fd"
def test_large_ffg_context_uses_global_index_and_bounded_focused_details():
    graph = FunctionalFlowGraph()
    long_description = "\u9875\u9762\u63a7\u4ef6\u548c\u5386\u53f2\u8bf4\u660e" * 120
    for index in range(1, 81):
        function_id = f"func_{index}"
        page_id = f"page_{index}"
        graph.add_node(FunctionalityNode(
            id=function_id,
            name=f"\u529f\u80fd {index}",
            description=long_description,
        ))
        graph.add_page(Page(
            id=page_id,
            name=f"\u9875\u9762 {index}",
            description=long_description,
            activity=f"Activity{index}",
            operations=[long_description] * 8,
        ), function_id)

    update_summary = _build_update_ffg_summary(
        graph,
        {"scenario_type": "intra_test", "target_function_id": "func_80"},
        [{"raw_step_index": 1, "current_page": graph.get_page("page_80")}],
    )

    assert "global_function_index" in update_summary
    assert '"focused_function_ids":["func_80"' in update_summary
    assert len(update_summary) <= FFG_UPDATE_SUMMARY_MAX_CHARS + 40


def test_ffg_update_respects_model_decision_to_create_new_page():
    screenshot = _screenshot(24)
    graph = FunctionalFlowGraph()
    graph.add_node(FunctionalityNode(id="func_1", name="\u4e3b\u529f\u80fd"))
    graph.add_page(Page(
        id="page_1",
        name="\u5df2\u8bb0\u5f55\u9875\u9762",
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
                "name": "\u6a21\u578b\u8bef\u5224\u7684\u65b0\u9875\u9762",
                "description": "\u89c6\u89c9\u8bc1\u636e\u5b9e\u9645\u4e0e page_1 \u76f8\u540c",
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
    graph = FunctionalFlowGraph()
    graph.add_node(FunctionalityNode(id="func_1", name="\u4e3b\u529f\u80fd", test_status="in_progress"))
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
        "current_test_scenario": "\u8fdb\u5165\u4e3b\u529f\u80fd\u5e76\u5b8c\u6210\u4e00\u6b21\u7ed3\u679c\u9a8c\u8bc1",
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


def test_ffg_transport_failure_applies_trace_without_model_patch(monkeypatch):
    graph = FunctionalFlowGraph()
    graph.add_node(FunctionalityNode(id="func_1", name="\u4e3b\u529f\u80fd", test_status="in_progress"))
    graph.add_page(Page(
        id="page_1",
        representative_screenshot=_screenshot(10),
    ), "func_1")

    class FailingLlm:
        calls = 0

        def invoke(self, _messages):
            self.calls += 1
            raise RuntimeError("gateway unavailable")

    monkeypatch.setattr(ffg_update_module, "FFG_UPDATE_LLM_RETRY_ATTEMPTS", 1)
    llm = FailingLlm()
    result = create_ffg_update_node(llm)({
        "ffg": graph,
        "current_page": Page(id="page_1", representative_screenshot=_screenshot(10)),
        "current_plan": {
            "scenario_type": "intra_test",
            "target_function_id": "func_1",
        },
        "current_test_scenario": "\u70b9\u51fb\u4e00\u4e2a\u5c1a\u672a\u5efa\u6a21\u7684\u5165\u53e3\u5e76\u89c2\u5bdf\u7ed3\u679c",
        "scenario_execution_trace": [{
            "action_name": "click",
            "action_description": "\u70b9\u51fb\u5c1a\u672a\u5efa\u6a21\u7684\u5165\u53e3",
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
    assert result["ffg_update_reason"] == "model patch unavailable; trace applied by program"
    # If the model is unavailable, reuse pages by screenshot and record executed operations in the FFG.
    assert "\u70b9\u51fb\u5c1a\u672a\u5efa\u6a21\u7684\u5165\u53e3" in graph.get_page("page_1").operations


def test_failed_intra_execution_is_recorded_as_attempted_but_not_successful():
    graph = FunctionalFlowGraph()
    graph.add_node(FunctionalityNode(id="func_1", name="\u4e3b\u529f\u80fd", test_status="in_progress"))
    graph.add_page(Page(
        id="page_1",
        representative_screenshot=_screenshot(10),
        controls=["\u4e34\u65f6\u64a4\u9500\u6309\u94ae"],
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
        "current_test_scenario": "\u70b9\u51fb\u5df2\u7ecf\u6d88\u5931\u7684\u4e34\u65f6\u64a4\u9500\u6309\u94ae",
        "scenario_execution_trace": [{
            "action_name": "click",
            "action_description": "\u70b9\u51fb\u4e34\u65f6\u64a4\u9500\u6309\u94ae",
            "action_failed": True,
            "action_error": "\u76ee\u6807\u63a7\u4ef6\u4e0d\u53ef\u8fbe",
            "current_page": Page(id="page_1", representative_screenshot=_screenshot(10)),
        }],
        "navigation_execution_trace": [],
        "execution_result": {
            "navigation_status": "skipped",
            "scenario_status": "stalled",
            "scenario_error": "\u76ee\u6807\u63a7\u4ef6\u4e0d\u53ef\u8fbe",
        },
        "current_testing_function_id": "func_1",
        "current_testing_function_status": "in_progress",
    })

    node = graph.get_node("func_1")
    assert node.scenarios == [
        "\u70b9\u51fb\u5df2\u7ecf\u6d88\u5931\u7684\u4e34\u65f6\u64a4\u9500\u6309\u94ae"
    ]
    assert graph.get_page("page_1").operations == [
        "Failed attempt: \u70b9\u51fb\u4e34\u65f6\u64a4\u9500\u6309\u94ae; error: \u76ee\u6807\u63a7\u4ef6\u4e0d\u53ef\u8fbe"
    ]
    assert graph.get_node("func_1").coverage_percent == 100
    assert graph.get_node_test_status("func_1") == "in_progress"


def test_finished_only_trace_does_not_record_a_test_scenario():
    graph = FunctionalFlowGraph()
    graph.add_node(FunctionalityNode(id="func_1", name="\u4e3b\u529f\u80fd", test_status="in_progress"))
    graph.add_page(Page(id="page_1", controls=["\u641c\u7d22\u6309\u94ae"]), "func_1")

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
        "current_test_scenario": "\u70b9\u51fb\u641c\u7d22\u6309\u94ae\u5e76\u89c2\u5bdf\u7ed3\u679c",
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
    graph = FunctionalFlowGraph()
    graph.add_node(FunctionalityNode(id="func_1", name="\u4e3b\u529f\u80fd", test_status="in_progress"))

    class NeverCalledLlm:
        def invoke(self, _messages):
            raise AssertionError("empty trace must not invoke FFG update LLM")

    create_ffg_update_node(NeverCalledLlm())({
        "ffg": graph,
        "current_plan": {
            "scenario_type": "intra_test",
            "target_function_id": "func_1",
        },
        "current_test_scenario": "\u70b9\u51fb\u771f\u5b9e\u8d26\u53f7\u5b89\u5168\u7b56\u7565\u7981\u6b62\u7684\u5165\u53e3",
        "navigation_execution_trace": [],
        "scenario_execution_trace": [],
        "execution_result": {
            "navigation_status": "skipped",
            "scenario_status": "blocked",
        },
        "current_testing_function_id": "func_1",
        "current_testing_function_status": "in_progress",
    })

    assert graph.get_node("func_1").scenarios == ["\u70b9\u51fb\u771f\u5b9e\u8d26\u53f7\u5b89\u5168\u7b56\u7565\u7981\u6b62\u7684\u5165\u53e3"]


def test_black_screen_trace_updates_ffg_and_reuses_single_error_page():
    graph = FunctionalFlowGraph()
    graph.add_node(FunctionalityNode(id="func_1", name="\u4e3b\u529f\u80fd", test_status="in_progress"))
    graph.add_page(Page(id="page_1", name="\u4e3b\u9875\u9762", representative_screenshot=_screenshot(10)), "func_1")

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
        "current_test_scenario": "\u70b9\u51fb\u5371\u9669\u5165\u53e3\u5e76\u786e\u8ba4\u7ed3\u679c",
        "scenario_execution_trace": [
            {
                "action_name": "click",
                "action_description": "\u70b9\u51fb\u5371\u9669\u5165\u53e3",
                "x": 100,
                "y": 200,
                "current_page": Page(id="page_1", representative_screenshot=_screenshot(10)),
            },
            {
                "action_name": "capture_screenshot",
                "action_description": "\u89c2\u5bdf\u5230\u9ed1\u5c4f\u6216\u9875\u9762\u4e0d\u53ef\u7528",
                "current_page": black_runtime_page,
                "observation_only": True,
                "screenshot_health_error": "\u622a\u56fe\u7591\u4f3c\u9ed1\u5c4f",
            },
        ],
        "execution_result": {
            "navigation_status": "skipped",
            "scenario_status": "screenshot_unavailable",
            "scenario_error": "\u9ed1\u5c4f",
        },
    })

    error_pages = [page for page in graph.active_pages() if page.name == "Black or unavailable page"]
    assert len(error_pages) == 1
    error_page = error_pages[0]
    assert error_page.representative_screenshot == ""
    assert error_page.id in graph.get_node("func_1").page_ids
    assert graph.get_node_test_status("func_1") == "in_progress"
    assert result["current_page"].id == error_page.id
    assert result["current_page"].representative_screenshot == black_runtime_page.representative_screenshot
    assert graph.get_node("func_1").scenarios == ["\u70b9\u51fb\u5371\u9669\u5165\u53e3\u5e76\u786e\u8ba4\u7ed3\u679c"]
    assert any(
        transition.src_page_id == "page_1"
        and transition.dst_page_id == error_page.id
        and "\u70b9\u51fb\u5371\u9669\u5165\u53e3" in transition.operation
        for transition in graph.page_transitions
    )
    assert any("\u70b9\u51fb\u5371\u9669\u5165\u53e3" in operation for operation in graph.get_page("page_1").operations)

    node({
        **base_state,
        "current_test_scenario": "\u518d\u6b21\u89e6\u53d1\u5f02\u5e38\u8def\u5f84",
        "scenario_execution_trace": [],
        "execution_result": {
            "navigation_status": "skipped",
            "scenario_status": "screenshot_unavailable",
            "scenario_error": "\u4ecd\u7136\u9ed1\u5c4f",
        },
    })

    assert len([page for page in graph.active_pages() if page.name == "Black or unavailable page"]) == 1


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
            "action_error": "\u8bbe\u5907\u8fde\u63a5\u6682\u65f6\u4e0d\u53ef\u7528",
        },
    ]

    filtered = _filter_trace(trace)

    assert len(filtered) == 4
    assert [step["raw_step_index"] for step in filtered] == [1, 2, 3, 4]


def test_ffg_records_black_observation_even_when_model_recovers_in_same_scenario():
    graph = FunctionalFlowGraph()
    graph.add_node(FunctionalityNode(id="func_1", name="\u4e3b\u529f\u80fd", test_status="in_progress"))
    graph.add_page(Page(id="page_1", name="\u4e3b\u9875\u9762", representative_screenshot=_screenshot(10)), "func_1")

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
        "current_test_scenario": "\u4ece\u5f02\u5e38\u754c\u9762\u8fd4\u56de\u540e\u6d4b\u8bd5\u5176\u5b83\u5165\u53e3",
        "navigation_execution_trace": [{
            "action_name": "press_back",
            "action_description": "\u4ece\u9ed1\u5c4f\u754c\u9762\u8fd4\u56de",
            "current_page": Page(representative_screenshot=_screenshot(20)),
            "screenshot_health_error": "\u622a\u56fe\u7591\u4f3c\u9ed1\u5c4f",
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

    assert any(page.name == "Black or unavailable page" for page in graph.active_pages())
    assert result["current_page"].id == "page_1"


def test_ffg_deterministically_records_operations_and_direct_transitions():
    graph = FunctionalFlowGraph()
    graph.add_node(FunctionalityNode(id="func_1", name="\u4e3b\u529f\u80fd", test_status="in_progress"))
    graph.add_page(Page(id="page_1", name="\u5217\u8868\u9875", representative_screenshot=_screenshot(10)), "func_1")
    graph.add_page(Page(id="page_2", name="\u8be6\u60c5\u9875", representative_screenshot=_screenshot(20)), "func_1")

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
        "current_test_scenario": "\u70b9\u51fb\u5217\u8868\u9879\u8fdb\u5165\u8be6\u60c5",
        "navigation_execution_trace": [],
        "scenario_execution_trace": [
            {
                "action_name": "click",
                "action_description": "\u70b9\u51fb\u5217\u8868\u9879",
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

    assert "\u70b9\u51fb\u5217\u8868\u9879" in graph.get_page("page_1").operations
    assert any(
        transition.src_page_id == "page_1"
        and transition.dst_page_id == "page_2"
        and transition.operation == "\u70b9\u51fb\u5217\u8868\u9879"
        for transition in graph.page_transitions
    )


def test_new_activity_in_scenario_trace_must_be_represented_by_a_page():
    graph = FunctionalFlowGraph()
    graph.add_node(FunctionalityNode(id="func_1", name="\u4e3b\u529f\u80fd"))
    graph.add_page(Page(id="page_1", activity="LauncherUI"), "func_1")

    missing = _unrepresented_trace_activities(
        graph,
        [{"current_page": Page(activity="ContactInfoUI")}],
    )

    assert missing == ["ContactInfoUI"]


def test_model_can_map_transient_activity_to_existing_logical_page():
    graph = FunctionalFlowGraph()
    graph.add_node(FunctionalityNode(id="func_1", name="\u4e3b\u529f\u80fd"))
    graph.add_page(Page(id="page_1", activity="ArticleUI"), "func_1")

    missing = _unrepresented_trace_activities(
        graph,
        [{"raw_step_index": 3, "current_page": Page(activity="WebViewStubProxyUI")}],
        {3: "page_1"},
    )

    assert missing == []


def test_navigation_only_trace_does_not_mark_test_scenario_attempted():
    graph = FunctionalFlowGraph()
    graph.add_node(FunctionalityNode(id="func_1", name="\u4e3b\u529f\u80fd", test_status="in_progress"))
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
        "current_test_scenario": "\u70b9\u51fb\u4e3b\u529f\u80fd\u91cc\u7684\u4fdd\u5b58\u6309\u94ae",
        "navigation_execution_trace": [{
            "action_name": "click",
            "current_page": Page(id="page_1", representative_screenshot=_screenshot(10)),
        }],
        "scenario_execution_trace": [],
        "execution_result": {
            "navigation_status": "stalled",
            "scenario_status": "skipped",
        },
        "current_testing_function_id": "func_1",
        "current_testing_function_status": "in_progress",
    })

    assert graph.get_node("func_1").scenarios == []
