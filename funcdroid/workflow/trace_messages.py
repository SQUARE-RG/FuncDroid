"""Shared trace-message construction and evidence normalization.

Keep step text, coordinates, outcomes, and image references consistent across reviewers."""

from __future__ import annotations

import json
from typing import Any

EVIDENCE_ROLES = {"before", "action", "after"}
ACTION_PARAMETER_FIELDS = ("x", "y", "x1", "y1", "x2", "y2", "direction", "app_name")
STEP_FLAG_FIELDS = (
    "is_interruption",
    "action_failed",
    "external_context",
    "foreground_recovery",
    "external_package",
    "observation_after_recovery",
)


def page_screenshot(page: Any) -> str:
    """Return a Page or dictionary screenshot, or an empty string."""
    if isinstance(page, dict):
        return str(page.get("screenshot", "") or page.get("screenshot_base64", "") or "")
    if hasattr(page, "representative_screenshot"):
        return str(getattr(page, "representative_screenshot", "") or "")
    return ""


def step_screenshot(step: dict[str, Any]) -> str:
    return page_screenshot(step.get("current_page")) if isinstance(step, dict) else ""


def raw_step_index(step: dict[str, Any], fallback: int) -> int:
    return int(step.get("raw_step_index", fallback) or fallback)


def coerce_step(value: Any) -> int | None:
    """Accept positive integer step numbers, including numeric strings; reject booleans and nonpositive values."""
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value.strip())
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        return None
    return value


def normalize_evidence_items(items: Any) -> list[dict[str, Any]]:
    """Normalize evidence entries with global_step and before/action/after roles."""
    evidence: list[dict[str, Any]] = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        step = coerce_step(item.get("global_step"))
        if step is None:
            continue
        role = str(item.get("role", "") or "").strip().lower()
        if role not in EVIDENCE_ROLES:
            continue
        evidence.append({"global_step": step, "role": role})
    return evidence


def build_step_text(
    step: dict[str, Any],
    raw_index: int,
    *,
    coordinate_note: str,
    include_flags: bool = False,
) -> str:
    """Build step metadata, observations, action parameters, context flags, and outcomes.

    The caller adds image_ref_text only after deciding whether an image is attached."""
    action_name = str(step.get("action_name", "")).strip()
    action_description = str(step.get("action_description", "") or "").strip()
    thought = str(step.get("thought", "") or "").strip()
    phase = str(step.get("trace_phase", "")).strip() or "unknown"
    text = f"[STEP] index={raw_index}, phase={phase}, action={action_name}"
    # Include automation and external-context flags so reviewers can distinguish application defects from infrastructure events.
    if action_description:
        text += f"\n[ACTION_DESCRIPTION] {action_description}"
    if thought:
        text += f"\n[AGENT_OBSERVATION] {thought}"
    parameters: list[str] = []
    for field in ACTION_PARAMETER_FIELDS:
        value = step.get(field)
        if value is not None and str(value).strip():
            parameters.append(f"{field}={value}")
    content = str(step.get("content", "")).strip()
    if content:
        parameters.append(f"content={content}")
    if parameters:
        text += "\n[ACTION_PARAMETERS] " + ", ".join(parameters)
    if step.get("model_coordinate_space") == "image_pixels":
        text += (
            "\n[COORDINATE_CONTEXT] Coordinates in ACTION_PARAMETERS are screen pixels passed to the device driver; "
            f"screen_size={json.dumps(step.get('screen_size', {}), ensure_ascii=False)}；"
            f"executor_image_size={json.dumps(step.get('image_size', {}), ensure_ascii=False)}；"
            f"original_model_image_coordinates={json.dumps(step.get('model_coordinates', {}), ensure_ascii=False)}。"
            + coordinate_note
        )
    if include_flags:
        flags = [name for name in STEP_FLAG_FIELDS if bool(step.get(name, False))]
        if flags:
            text += "\n[STEP_FLAGS] " + ", ".join(flags)
    if bool(step.get("action_failed", False)):
        text += (
            "\n[ACTION_RESULT] failed_or_blocked: "
            + str(step.get("action_error", "") or "").strip()
        )
    return text


def image_ref_text(raw_index: int, *, has_image: bool, image_note: str) -> str:
    """Emit IMAGE_REF only when an image follows, otherwise NO_SCREENSHOT.

    Use raw_step_index consistently with evidence validation even when intermediate steps are filtered."""
    if has_image:
        return f"\n[IMAGE_REF] BEFORE_STEP_{raw_index}: The following image belongs to STEP {raw_index} {image_note}"
    return (
        f"\n[NO_SCREENSHOT] STEP {raw_index} has no usable screenshot; no image follows this step. "
        f"Do not treat another step's image as the state of STEP {raw_index}."
    )


def image_block(payload: str) -> dict[str, Any]:
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{payload}"},
    }
