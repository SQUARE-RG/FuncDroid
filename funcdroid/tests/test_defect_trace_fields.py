# Multilingual UI and model-response fixtures use Unicode escapes to preserve their exact text.
"""Detection input includes executor observations and step flags so the model can distinguish app defects, automation problems, and external events."""

from __future__ import annotations

import json

from model.flow_graph import Page
from workflow.trace_defect_detection import _build_full_trace_messages


def _messages_text(state: dict, trace: list[dict]) -> str:
    messages = _build_full_trace_messages(state, trace)
    human = messages[-1]
    parts = []
    for block in human.content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block["text"])
    return "\n".join(parts)


def _step(**overrides) -> dict:
    step = {
        "action_name": "click",
        "raw_step_index": 1,
        "trace_phase": "test",
        "current_page": Page(id="page_1", name="\u8bbe\u7f6e\u9875"),
    }
    step.update(overrides)
    return step


def test_step_carries_description_observation_and_parameters():
    text = _messages_text(
        {"current_plan": {}, "package_name": "com.example.app"},
        [
            _step(
                action_description="\u70b9\u51fb\u63a5\u6536\u63a8\u9001\u6d88\u606f\u5f00\u5173",
                thought="\u5f00\u5173\u70b9\u4e86\u4e00\u4e0b\u6ca1\u6709\u53cd\u5e94，\u53ef\u80fd\u672a\u751f\u6548",
                x=120,
                y=340,
            )
        ],
    )
    assert "[ACTION_DESCRIPTION] \u70b9\u51fb\u63a5\u6536\u63a8\u9001\u6d88\u606f\u5f00\u5173" in text
    assert "[AGENT_OBSERVATION] \u5f00\u5173\u70b9\u4e86\u4e00\u4e0b\u6ca1\u6709\u53cd\u5e94，\u53ef\u80fd\u672a\u751f\u6548" in text
    assert "[ACTION_PARAMETERS]" in text and "x=120" in text and "y=340" in text


def test_step_carries_flags_and_failure_result():
    text = _messages_text(
        {"current_plan": {}},
        [
            _step(
                action_name="press_back",
                action_failed=True,
                action_error="\u5750\u6807\u8d8a\u754c",
                is_interruption=True,
                foreground_recovery=True,
            )
        ],
    )
    assert "[STEP_FLAGS]" in text
    assert "is_interruption" in text and "foreground_recovery" in text
    assert "[ACTION_RESULT] failed_or_blocked: \u5750\u6807\u8d8a\u754c" in text


def test_step_without_optional_fields_omits_them():
    text = _messages_text({"current_plan": {}}, [_step()])
    # Inspect step bodies only; the header itself documents these markers.
    step_text = text.split("[STEP] index=1", 1)[1]
    assert "[ACTION_DESCRIPTION]" not in step_text
    assert "[AGENT_OBSERVATION]" not in step_text
    assert "[STEP_FLAGS]" not in step_text
    assert "[ACTION_RESULT]" not in step_text


def test_context_exposes_function_phase_but_not_test_expectation():
    """Expose the phase without injecting test expectations: test intent is not a product promise."""
    state = {
        "current_plan": {"scenario_type": "intra_test", "function_phase": "depth"},
        "current_test_scenario": "\u9a8c\u8bc1\u79fb\u9664\u540e\u91cd\u8fdb\u4e0d\u56de\u5f39",
    }
    text = _messages_text(state, [_step()])
    assert '"function_phase": "depth"' in text
    # Scenario text remains execution context but must not be treated as an expected-behavior source.
    assert "Scenario text expresses test intent" in text
    assert "function_phase=depth" in text


def test_context_omits_phase_when_absent():
    text = _messages_text({"current_plan": {}}, [_step()])
    # Inspect the context JSON because the header already describes function_phase.
    assert '"function_phase"' not in text


def _shot_page() -> Page:
    import numpy as np

    from tools.utils import encode_image_to_jpg_base64

    shot = encode_image_to_jpg_base64(np.full((32, 32, 3), 180, dtype=np.uint8))
    return Page(id="page_1", name="\u8bbe\u7f6e\u9875", representative_screenshot=shot)


def test_image_ref_still_present_per_step():
    # The old test expected IMAGE_REF for a step without an image, reproducing the misleading marker bug.
    # Only steps with attached images now emit IMAGE_REF.
    text = _messages_text(
        {"current_plan": {}},
        [_step(current_page=_shot_page()), _step(raw_step_index=2, current_page=_shot_page())],
    )
    assert "[IMAGE_REF] BEFORE_STEP_1" in text
    assert "The result of STEP 1 is normally visible in BEFORE_STEP_2" in text


def test_step_without_screenshot_is_not_labeled_as_image():
    messages = _build_full_trace_messages(
        {"current_plan": {}},
        [_step(current_page=_shot_page()), _step(raw_step_index=2), _step(raw_step_index=3, current_page=_shot_page())],
    )
    blocks = messages[-1].content
    text = "\n".join(block["text"] for block in blocks if block.get("type") == "text")
    assert "[IMAGE_REF] BEFORE_STEP_2" not in text
    assert "[NO_SCREENSHOT] STEP 2" in text
    # Do not claim a result appears in BEFORE_STEP_2 when the next step has no image.
    assert "visible in BEFORE_STEP_2" not in text
    assert sum(1 for block in blocks if block.get("type") == "image_url") == 2
    # Every IMAGE_REF text block is immediately followed by an image.
    for position, block in enumerate(blocks):
        if block.get("type") == "text" and "[IMAGE_REF]" in block["text"]:
            assert blocks[position + 1]["type"] == "image_url"


def test_plan_json_is_embedded_for_traceability():
    state = {"current_plan": {"scenario_type": "intra_test", "function_phase": "breadth"}}
    text = _messages_text(state, [_step()])
    payload = json.loads(text.split("[Authoritative context]", 1)[1].split("\n\n", 1)[0])
    assert payload["current_plan"]["function_phase"] == "breadth"
