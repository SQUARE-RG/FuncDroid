# Multilingual UI and model-response fixtures use Unicode escapes to preserve their exact text.
"""Regressions for the throughput fixes: blind-probe early stop, zero-yield
scheduling release, and batched coverage scenarios."""

from __future__ import annotations

import base64

import cv2
import numpy as np
import pytest

from model.flow_graph import FunctionalFlowGraph, FunctionalityNode, Page
from workflow.nodes.ffg_update import (
    ZERO_YIELD_MIN_COVERAGE,
    _record_execution_outcome,
)
from workflow.nodes.scenario_exec import (
    NO_PROGRESS_STALL_WINDOW,
    _detect_no_progress_stall,
    _detect_stall,
)
from workflow.nodes.scenario_plan import (
    PLAN_SCENARIO_CONTROL_BATCH,
    _forced_intra_scenario,
)


@pytest.fixture(autouse=True)
def _legacy_text_matching_enabled(monkeypatch):
    """Explicitly enable legacy text fallback to preserve these pre-binding tests' intended coverage semantics."""
    from model import flow_graph as _flow_graph

    monkeypatch.setattr(_flow_graph, "COVERAGE_LEGACY_TEXT_MATCHING", True)


def _screenshot(marker_width: int) -> str:
    image = np.full((256, 256, 3), 255, dtype=np.uint8)
    if marker_width:
        image[20:40, 20:20 + marker_width] = 0
    success, encoded = cv2.imencode(".jpg", image)
    assert success
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _full_screen_screenshot(level: int) -> str:
    """Simulate full-screen list pagination. Global grayscale similarity distinguishes these states while small local expansions remain similar and can indicate blind probing."""
    image = np.full((256, 256, 3), max(0, 255 - level * 40), dtype=np.uint8)
    success, encoded = cv2.imencode(".jpg", image)
    assert success
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _probe_step(action_name: str, screenshot: str, **fields) -> dict:
    step = {
        "action_name": action_name,
        "current_page": Page(
            id="",
            activity=".plugin.gallery.ui.AlbumPreviewUI",
            representative_screenshot=screenshot,
        ),
    }
    step.update(fields)
    return step


def _blind_probe_trace() -> list[dict]:
    """Mixed-direction probing that every pre-existing detector misses."""
    collapsed = _screenshot(10)
    expanded = _screenshot(200)
    plan = [
        ("click", collapsed, {"x": 719, "y": 186}),
        ("click", expanded, {"x": 600, "y": 1591}),
        ("scroll", collapsed, {"direction": "up"}),
        ("scroll", collapsed, {"direction": "left"}),
        ("scroll", collapsed, {"direction": "right"}),
        ("click", expanded, {"x": 719, "y": 186}),
    ]
    return [
        _probe_step(action, screenshot, **fields)
        for action, screenshot, fields in plan[:NO_PROGRESS_STALL_WINDOW]
    ]


def test_mixed_direction_blind_probing_is_detected_as_stalled():
    trace = _blind_probe_trace()

    reason = _detect_no_progress_stall(trace)

    assert reason
    assert "Blind probing" in reason
    # Apply the behavior through the common entry point, not only a standalone helper.
    assert _detect_stall(trace) == reason


def test_blind_probe_detection_needs_the_full_window():
    trace = _blind_probe_trace()[: NO_PROGRESS_STALL_WINDOW - 1]

    assert _detect_no_progress_stall(trace) == ""


def test_genuine_list_browsing_is_not_treated_as_stalled():
    """Full-screen changes prevent fallback stagnation detection; the directional scroll budget applies separately."""
    trace = [
        _probe_step("scroll", _full_screen_screenshot(index), direction="up")
        for index in range(NO_PROGRESS_STALL_WINDOW)
    ]

    # Full-screen changes at each step must not trigger the fallback no-progress detector.
    assert _detect_no_progress_stall(trace) == ""
    # The common entry still enforces the same-direction scroll budget and requires a new direction.
    stall_reason = _detect_stall(trace)
    assert stall_reason
    assert "same-direction budget" in stall_reason


def test_two_state_ping_pong_across_full_screens_is_stalled():
    """Alternating between only two full-screen states still indicates no progress."""
    first = _full_screen_screenshot(0)
    second = _full_screen_screenshot(4)
    trace = [
        _probe_step("click", first if index % 2 == 0 else second, x=100, y=200 + index)
        for index in range(NO_PROGRESS_STALL_WINDOW)
    ]

    assert _detect_no_progress_stall(trace)


def test_activity_change_within_window_is_not_treated_as_stalled():
    trace = _blind_probe_trace()
    trace[-1]["current_page"] = Page(
        id="",
        activity=".ui.LauncherUI",
        representative_screenshot=_screenshot(10),
    )

    assert _detect_no_progress_stall(trace) == ""


def test_interruption_recovery_is_left_to_its_own_detector():
    trace = _blind_probe_trace()
    trace[-1]["is_interruption"] = True

    assert _detect_no_progress_stall(trace) == ""


def _zero_yield_graph(coverage_controls: int, handled: int) -> FunctionalFlowGraph:
    graph = FunctionalFlowGraph()
    graph.add_node(FunctionalityNode(id="func_1", name="\u516c\u4f17\u53f7\u56fe\u6587\u5185\u5bb9\u8be6\u60c5\u67e5\u770b"))
    graph.add_page(Page(
        id="page_1",
        activity=".plugin.brandservice.ui.timeline.preload.ui.TmplWebViewMMUI",
        controls=[f"\u6b63\u6587\u6bb5\u843d{index}" for index in range(coverage_controls)],
        operations=[f"\u70b9\u51fb\u6b63\u6587\u6bb5\u843d{index}" for index in range(handled)],
    ), "func_1")
    graph.set_node_test_status("func_1", "in_progress")
    return graph


def _zero_yield_state() -> dict:
    return {
        "current_plan": {"scenario_type": "intra_test", "target_function_id": "func_1"},
        "execution_result": {"scenario_status": "completed"},
    }


def _outcome(graph, state, yielded_structure):
    return _record_execution_outcome(graph, state, state["execution_result"], yielded_structure)


def test_successful_round_without_new_structure_flags_the_function():
    graph = _zero_yield_graph(coverage_controls=4, handled=3)
    assert graph.function_coverage_details("func_1")["coverage_percent"] >= ZERO_YIELD_MIN_COVERAGE

    assert _outcome(graph, _zero_yield_state(), yielded_structure=False) == {"func_1": 1}
    assert graph.get_node_test_status("func_1") == "in_progress"


def test_structural_yield_clears_the_flag():
    graph = _zero_yield_graph(coverage_controls=4, handled=3)
    state = _zero_yield_state()
    state["function_execution_failure_counts"] = {"func_1": 1}

    assert _outcome(graph, state, yielded_structure=True) == {}


def test_early_exploration_is_not_penalised_for_zero_yield():
    """Do not penalize initial low-coverage exploration before it has enough attempts to establish progress."""
    graph = _zero_yield_graph(coverage_controls=20, handled=1)
    assert graph.function_coverage_details("func_1")["coverage_percent"] < ZERO_YIELD_MIN_COVERAGE

    assert _outcome(graph, _zero_yield_state(), yielded_structure=False) == {}


def test_forced_intra_scenario_batches_multiple_uncovered_controls():
    graph = FunctionalFlowGraph()
    node = FunctionalityNode(id="func_1", name="\u804a\u5929\u4fe1\u606f\u7ba1\u7406")
    graph.add_node(node)
    graph.add_page(Page(
        id="page_1",
        name="\u804a\u5929\u4fe1\u606f\u9875",
        controls=["\u67e5\u627e\u804a\u5929\u5185\u5bb9", "\u8bbe\u7f6e\u804a\u5929\u80cc\u666f", "\u6e05\u7a7a\u804a\u5929\u8bb0\u5f55\u5165\u53e3", "\u6d88\u606f\u514d\u6253\u6270"],
        operations=[],
    ), "func_1")

    scenario = _forced_intra_scenario(graph, graph.get_node("func_1"))

    assert scenario
    hints = [
        hint for hint in ("\u67e5\u627e\u804a\u5929\u5185\u5bb9", "\u8bbe\u7f6e\u804a\u5929\u80cc\u666f", "\u6d88\u606f\u514d\u6253\u6270")
        if hint in scenario
    ]
    assert len(hints) >= min(2, PLAN_SCENARIO_CONTROL_BATCH)
    assert "visible widgets with no execution records in order" in scenario
    # The old wording trapped execution on one control.
    assert "Do not continue sequentially exploring other controls" not in scenario


def test_forced_intra_scenario_degrades_to_single_control():
    graph = FunctionalFlowGraph()
    graph.add_node(FunctionalityNode(id="func_1", name="\u626b\u7801"))
    graph.add_page(Page(
        id="page_1",
        name="\u626b\u7801\u9875",
        controls=["\u76f8\u518c"],
        operations=[],
    ), "func_1")

    scenario = _forced_intra_scenario(graph, graph.get_node("func_1"))

    assert "\u76f8\u518c" in scenario
    assert "visible widgets with no execution records in order" not in scenario


def _scroll_step(direction: str, phase: str = "") -> dict:
    return _probe_step(
        "scroll",
        _full_screen_screenshot(1),
        direction=direction,
        trace_phase=phase,
    )


def test_consecutive_same_direction_scroll_counting():
    from workflow.nodes.scenario_exec import (
        _consecutive_same_direction_scroll,
    )

    # Three consecutive downward scrolls count as three.
    trace = [_scroll_step("down") for _ in range(3)]
    assert _consecutive_same_direction_scroll(trace) == ("down", 3)
    # An intervening action resets the counter.
    trace.append(_probe_step("click", _full_screen_screenshot(2), x=100, y=200))
    trace.append(_scroll_step("down"))
    assert _consecutive_same_direction_scroll(trace) == ("down", 1)
    # A direction change resets the counter.
    trace.extend([_scroll_step("up") for _ in range(2)])
    assert _consecutive_same_direction_scroll(trace) == ("up", 2)
    # A non-scroll final action returns zero.
    trace.append(_probe_step("wait", _full_screen_screenshot(2)))
    assert _consecutive_same_direction_scroll(trace) == ("", 0)


def test_scroll_budget_triggers_stall_on_fourth_same_direction_scroll():
    """The common stall entry rejects the fourth consecutive scroll in one direction."""
    trace = [_scroll_step("down") for _ in range(4)]

    reason = _detect_stall(trace)

    assert reason
    assert "same-direction budget" in reason
    assert "down" in reason


def test_scroll_budget_allows_three_same_direction_scrolls():
    trace = [_scroll_step("down") for _ in range(3)]

    assert _detect_stall(trace) == ""


def test_scroll_budget_resets_after_other_actions():
    """A tap between scroll sequences resets the directional scroll budget."""
    trace = [_scroll_step("down") for _ in range(3)]
    trace.append(_probe_step("click", _full_screen_screenshot(2), x=100, y=200))
    trace.append(_scroll_step("down"))

    assert _detect_stall(trace) == ""


def test_scroll_budget_skips_navigation_phase():
    """Navigation keeps its existing intent detection and is exempt from the directional scroll cap."""
    trace = [_scroll_step("down", phase="navigation") for _ in range(4)]

    assert _detect_stall(trace) == ""


def test_step_input_warns_when_scroll_budget_nearly_exhausted():
    from workflow.nodes.scenario_exec import _build_step_input

    trace = [_scroll_step("down") for _ in range(3)]
    content = _build_step_input(trace, _screenshot(10), 4, 15)

    text_blocks = [
        item["text"] for item in content if item.get("type") == "text"
    ]
    assert any("Scroll budget" in text for text in text_blocks)
    assert any("Do not continue" in text for text in text_blocks)


def _phone_screenshot(switch_on: bool = False, clock: int = 0) -> str:
    """Create a 723x1600 solid-color page with a small right-side switch and variable status-bar clock."""
    image = np.full((1600, 723, 3), 230, dtype=np.uint8)
    image[20:50, 20:20 + 10 * clock] = 0  # Status bar is excluded from interface-change comparison.
    image[1010:1040, 560:620] = 40 if switch_on else 200
    success, encoded = cv2.imencode(".jpg", image)
    assert success
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _click(screenshot: str, x: int, y: int) -> dict:
    return {
        "action_name": "click",
        "model_coordinates": {"x": x, "y": y},
        "current_page": Page(representative_screenshot=screenshot),
    }


def test_small_toggle_change_is_reported_as_local_effect_not_no_change():
    from workflow.nodes.scenario_exec import _action_effect

    kind, text = _action_effect(
        _phone_screenshot(switch_on=False, clock=1),
        _phone_screenshot(switch_on=True, clock=3),
        (723, 1600),
    )
    assert kind == "local"
    assert "changed region" in text


def test_status_bar_only_change_counts_as_no_change():
    from workflow.nodes.scenario_exec import _action_effect

    kind, _ = _action_effect(_phone_screenshot(clock=1), _phone_screenshot(clock=4), (723, 1600))
    assert kind == "none"


def test_repeated_unresponsive_click_gets_strategy_change_note():
    from workflow.nodes.scenario_exec import _build_step_input

    same = _phone_screenshot()
    trace = [_click(same, 361, 760), _click(same, 362, 758)]
    content = _build_step_input(trace, same, 3, 15, image_size=(723, 1600))
    text = "\n".join(item["text"] for item in content if item.get("type") == "text")

    assert "2 consecutive clicks" in text and "the control is currently unresponsive" in text
    assert "Effect: No visible change" in text
    assert "Image coordinates: (361,760)" in text


def test_single_unresponsive_click_asks_to_recheck_position_first():
    from workflow.nodes.scenario_exec import _build_step_input

    same = _phone_screenshot()
    content = _build_step_input([_click(same, 361, 760)], same, 2, 15, image_size=(723, 1600))
    text = "\n".join(item["text"] for item in content if item.get("type") == "text")

    assert "No visible change after clicking (361,760)" in " ".join(text.split())
    assert "the control is currently unresponsive" not in text


def test_click_that_changed_the_screen_gets_no_warning():
    from workflow.nodes.scenario_exec import _build_step_input

    trace = [_click(_phone_screenshot(False), 590, 1025)]
    content = _build_step_input(trace, _phone_screenshot(True), 2, 15, image_size=(723, 1600))
    text = "\n".join(item["text"] for item in content if item.get("type") == "text")

    assert "[Previous effect]" not in text
    assert "Effect: Local visible change" in text


def test_type_result_must_be_checked_against_the_field_content():
    from workflow.nodes.scenario_exec import _build_step_input

    trace = [{
        "action_name": "type",
        "content": "FuncDroid_1",
        "current_page": Page(representative_screenshot=_phone_screenshot(False)),
    }]
    content = _build_step_input(trace, _phone_screenshot(True), 2, 15, image_size=(723, 1600))
    text = "\n".join(item["text"] for item in content if item.get("type") == "text")

    assert "character/length restrictions" in text
