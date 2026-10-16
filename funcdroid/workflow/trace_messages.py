"""缺陷判定与合规判定共用的轨迹消息构造与证据归一化。

两条判定管线对每个步骤输出相同结构的文本块（STEP 头、动作描述/观察/参数、
坐标上下文、动作结果、图片编号），差异只在各自的补充说明文案；这里集中实现，
避免两处各自演化后图片编号/步号语义分叉。
"""

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
    """读取 Page / dict 页面上的截图 base64；没有时返回空串。"""
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
    """把模型给出的步号统一为正整数；接受 5 与 "5"，拒绝 bool/0/负数/其它。"""
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value.strip())
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        return None
    return value


def normalize_evidence_items(items: Any) -> list[dict[str, Any]]:
    """归一化模型输出的 evidence 列表：[{global_step:int, role:before/action/after}]。"""
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
    """单步文本：STEP 头 + 动作描述/观察/参数 + 坐标上下文 + [标记] + 动作结果。

    不含图片编号；图片编号由调用方在确认确实附图后通过 image_ref_text 追加。
    """
    action_name = str(step.get("action_name", "")).strip()
    action_description = str(step.get("action_description", "") or "").strip()
    thought = str(step.get("thought", "") or "").strip()
    phase = str(step.get("trace_phase", "")).strip() or "unknown"
    text = f"[STEP] index={raw_index}, phase={phase}, action={action_name}"
    # 提示词要求区分自动化/外部环境事件，这些标记必须真的随步骤发出，
    # 否则模型无从判断一次异常是 App 缺陷还是自动化问题。
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
            "\n[COORDINATE_CONTEXT] ACTION_PARAMETERS 中的坐标为传给设备驱动的屏幕像素；"
            f"屏幕尺寸={json.dumps(step.get('screen_size', {}), ensure_ascii=False)}；"
            f"执行器输入图片尺寸={json.dumps(step.get('image_size', {}), ensure_ascii=False)}；"
            f"模型在执行器图片上的原始像素坐标={json.dumps(step.get('model_coordinates', {}), ensure_ascii=False)}。"
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
    """图片编号说明；只有确实紧随一张图片时才输出 [IMAGE_REF]，否则输出 [NO_SCREENSHOT]。

    图片编号必须与 [STEP] index 同源（raw_step_index）：证据校验按 raw_step_index
    建索引表，若改用连续序号，轨迹里有被过滤的步骤时两套编号就会分叉。
    """
    if has_image:
        return f"\n[IMAGE_REF] BEFORE_STEP_{raw_index}: 紧随其后的图片是 STEP {raw_index} {image_note}"
    return (
        f"\n[NO_SCREENSHOT] STEP {raw_index} 没有可用截图，本步之后不附图片；"
        f"不要把其它步骤的图片当作 STEP {raw_index} 的页面状态。"
    )


def image_block(payload: str) -> dict[str, Any]:
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{payload}"},
    }
