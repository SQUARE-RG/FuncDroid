# Multilingual UI and model-response fixtures use Unicode escapes to preserve their exact text.
"""Verify that model-image coordinates reach ADB as the intended device pixels through the full dispatch chain."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from automator.u2 import U2
from device import Driver
from tools.actions import _perform_action
from tools.utils import encode_image_to_jpg_base64, screenshot_dimensions
from workflow.nodes.scenario_exec import (
    _scale_plan_coordinates, _validate_action_plan, _build_step_input, _parse_action_json,
    _normalize_action_plan,
)


def _offline_driver(screen):
    automator = object.__new__(U2)
    automator.display_size = lambda: screen
    commands = []
    automator._shell_input = lambda *args: commands.append(args)
    driver = object.__new__(Driver)
    driver.automator = automator
    return driver, commands


@pytest.mark.parametrize("action", ["click", "long_press"])
@pytest.mark.parametrize("screen, image, point, expected", [
    ((1200, 2652), (723, 1600), (0, 0), (0, 0)),
    ((1200, 2652), (723, 1600), (0.6, 0.6), (1, 1)),
    ((1200, 2652), (723, 1600), (361, 800), (600, 1326)),
    ((1200, 2652), (723, 1600), (722, 1599), (1199, 2651)),
    ((2652, 1200), (1600, 723), (800, 361), (1326, 600)),
    ((320, 640), (320, 640), (1, 1), (1, 1)),
])
def test_model_point_is_scaled_once_before_adb(action, screen, image, point, expected):
    driver, commands = _offline_driver(screen)
    plan = {"action_name": action, "x": point[0], "y": point[1]}
    _validate_action_plan(plan, image)
    _scale_plan_coordinates(plan, *driver.display_size(), image)
    _perform_action(driver, **plan)
    assert len(commands) == 1
    assert commands[0][1:3] == expected
    if action == "long_press":
        assert commands[0][3:5] == expected


def test_model_swipe_keeps_one_pixel_start_and_last_pixel_end():
    driver, commands = _offline_driver((1200, 2652))
    plan = {"action_name": "swipe", "x1": 0.6, "y1": 0.6,
            "x2": 722, "y2": 1599}
    _validate_action_plan(plan, (723, 1600))
    _scale_plan_coordinates(plan, *driver.display_size(), (723, 1600))
    _perform_action(driver, **plan)
    assert commands == [("swipe", 1, 1, 1199, 2651, 500)]


def test_relative_scroll_still_uses_device_dimensions():
    driver, commands = _offline_driver((1200, 2652))
    _perform_action(driver, action_name="scroll", direction="up")
    assert commands == [("swipe", 600, 1326, 600, 265, 500)]


def test_float_relative_endpoint_does_not_exceed_screen():
    driver, commands = _offline_driver((1200, 2652))
    driver.click(1.0, 1.0)
    assert commands == [("tap", 1199, 2651)]


@pytest.mark.parametrize("point", [(723, 100), (100, 1600), (-1, 3), (False, 3), (3, float('inf'))])
def test_out_of_image_coordinates_are_rejected(point):
    with pytest.raises(ValueError):
        _validate_action_plan({'action_name': 'click', 'x': point[0], 'y': point[1]}, (723, 1600))


def test_unknown_dimensions_are_not_guessed():
    with pytest.raises(ValueError, match='screenshot dimensions'):
        _validate_action_plan({'action_name': 'click', 'x': 12, 'y': 34})
    assert screenshot_dimensions('not-an-image') is None


@pytest.mark.parametrize('raw', ['{"action_name":"click","x":200, 1500}',
                                 '{"action_name":"click","x":[200,1500]}'])
def test_json_repair_preserves_large_image_coordinates(raw):
    plan = _normalize_action_plan(_parse_action_json(raw))
    _validate_action_plan(plan, (723, 1600))
    assert (plan['x'], plan['y']) == (200, 1500)


def test_real_payload_dimensions_and_compliance_image_match():
    from model.flow_graph import Page
    from workflow.compliance_detection import _build_messages

    shot = encode_image_to_jpg_base64(np.full((2652, 1200, 3), 180, dtype=np.uint8), max_side=1600)
    assert screenshot_dimensions(shot) == (723, 1600)
    content = _build_step_input([], shot, 1, 3)
    text = '\n'.join(b['text'] for b in content if b['type'] == 'text')
    assert 'width=723, height=1600' in text
    assert '0-722' in text and '0-1599' in text
    review = _build_messages({}, [{'current_page': Page(representative_screenshot=shot)}], [])[1].content

    def image(blocks):
        return [b['image_url']['url'] for b in blocks if b['type'] == 'image_url']

    assert image(content) == image(review)


@pytest.mark.parametrize("action", ["click", "long_press"])
def test_executor_records_both_coordinate_spaces(monkeypatch, tmp_path, action):
    from model.flow_graph import Page
    from workflow.nodes import scenario_exec as executor
    from workflow.compliance_detection import _build_messages
    from main import _serialize_trace

    shot = encode_image_to_jpg_base64(np.full((1600, 723, 3), 180, dtype=np.uint8))
    page = Page(id='page_1', activity='MainActivity', representative_screenshot=shot)
    replies = iter([
        {'action_name': action, 'x': 361, 'y': 1500, 'thought': '\u70b9\u51fb\u76ee\u6807'},
        {'action_name': 'finished', 'thought': '\u76ee\u6807\u5df2\u8fbe\u6210'},
    ])
    llm = SimpleNamespace(invoke=lambda messages: SimpleNamespace(content=json.dumps(next(replies))))
    driver, commands = _offline_driver((1200, 2652))
    # Taps do not read the control tree; nearby controls cannot rewrite injected coordinates.
    driver.automator._driver = SimpleNamespace(
        dump_hierarchy=lambda: pytest.fail("click must not read the UI hierarchy"),
    )
    monkeypatch.setattr(executor, '_ensure_target_app', lambda *a: True)
    monkeypatch.setattr(executor, '_target_process_pid', lambda *a: '')
    monkeypatch.setattr(executor, '_target_app_foreground_state', lambda *a: ('foreground', 'com.example'))
    monkeypatch.setattr(executor, 'capture_runtime_page', lambda *a, **kw: (page, ''))
    monkeypatch.setattr(executor.time, 'sleep', lambda *a: None)
    monkeypatch.setattr(executor, 'save_trace_screenshot', lambda *a, **kw: '')
    trace, _, status, error, _ = executor._run_single_scenario(
        llm, driver, 'App', 'com.example', '\u70b9\u51fb\u76ee\u6807', page, False, 'test',
    )
    assert status == 'completed', error
    expected = (600, 2487)
    assert commands[0][1:3] == expected
    assert commands[0][0] == ('tap' if action == 'click' else 'swipe')
    assert trace[0]['model_coordinates'] == {'x': 361, 'y': 1500}
    assert trace[0]['image_size'] == {'width': 723, 'height': 1600}
    saved = _serialize_trace(trace, str(tmp_path))[0]['operation']['parameters']
    assert saved['model_coordinate_space'] == 'image_pixels'
    assert saved['coordinate_space'] == 'screen_pixels'
    assert (saved['x'], saved['y']) == expected
    assert 'coordinate_snapped' not in saved
    review = _build_messages({}, trace, [])[1].content
    assert any('COORDINATE_CONTEXT' in block.get('text', '') for block in review)


def test_executor_prompt_describes_image_only_clicks_and_type_clearing():
    from workflow.nodes.prompt import SCENARIO_EXEC_SYSTEM_PROMPT_TEMPLATE as prompt

    assert "program does not snap them to nearby controls" in prompt
    assert "type clears the focused field" in prompt
    assert "rather than repeatedly clicking" in prompt
