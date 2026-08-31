"""Constants and lightweight helpers shared between nodes."""

from __future__ import annotations

import base64
import re
from typing import Any

import cv2
import numpy as np

JSON_BLOCK_PATTERN = re.compile(r"^```(?:json)?\s*|\s*```$")

EXHAUSTIVE_CONTROL_FAMILY_PATTERN = re.compile(
    r"(?:\u6240\u6709|\u5168\u90e8|\u6bcf\u4e2a|\u9010\u4e2a|\u9010\u4e00|\u904d\u5386)"
    r"[^\u2192\uff0c,.\u3002\uff1b;]{0,28}?"
    r"(?:\u8868\u60c5|\u9009\u9879|\u6761\u76ee|\u6309\u94ae|\u63a7\u4ef6|\u6807\u7b7e|\u5206\u7c7b|\u83dc\u5355\u9879|\u5bfc\u822a\u9879|\u5217\u8868\u9879|\u56fe\u6807|\u9879\u76ee|\u7ed3\u679c)"
    r"|\b(?:all|every|each)\b.{0,28}\b"
    r"(?:items?|options?|buttons?|tabs?|emojis?|entries?|results?)\b",
    re.IGNORECASE,
)

EXPLICIT_BULK_OPERATION_PATTERN = re.compile(
    r"\u5168\u9009\u6309\u94ae|\u70b9\u51fb\u5168\u9009|\u9009\u62e9\u5168\u90e8(?:\u5185\u5bb9|\u6587\u672c|\u6587\u4ef6)|\u5220\u9664\u5168\u90e8|\u5168\u90e8\u5220\u9664|"
    r"\u6e05\u7a7a\u5168\u90e8|\u5168\u90e8\u6e05\u9664|\u4e00\u952e(?:\u6e05\u7a7a|\u5220\u9664)|\bselect all\b|\bdelete all\b|\bclear all\b",
    re.IGNORECASE,
)


def has_exhaustive_control_family_intent(value: object) -> bool:
    """Identify scenarios requiring exhaustive interaction with duplicate controls of the same function, while preserving genuine single batch operations."""
    text = str(value or "").strip()
    if not text:
        return False
    candidate = EXPLICIT_BULK_OPERATION_PATTERN.sub("", text)
    return EXHAUSTIVE_CONTROL_FAMILY_PATTERN.search(candidate) is not None


def representative_control_family_scenario(value: object) -> str:
    """Shrink exhaustive scenarios that bypassed planning validation to a single representative item."""
    text = str(value or "").strip()
    if not has_exhaustive_control_family_intent(text):
        return text

    normalized = re.sub(r"\u9010\u4e2a|\u9010\u4e00|\u904d\u5386", "", text)
    normalized = re.sub(r"\u4f9d\u6b21(?=\u70b9\u51fb|\u9009\u62e9|\u6253\u5f00|\u52fe\u9009)", "", normalized)
    normalized = re.sub(r"\u6240\u6709|\u5168\u90e8|\u6bcf\u4e2a", "\u4e00\u4e2a\u4ee3\u8868\u6027\u7684", normalized)
    normalized = re.sub(r"\b(?:all|every|each)\b", "one representative", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return f"Operate on only one representative item for similar controls, and finish immediately after the first result appears; {normalized}"


def screenshot_thumbnail(screenshot: str, size: int = 64) -> np.ndarray | None:
    """Decode a screenshot string into a grayscale thumbnail; return None if decoding fails.

    Extracted as a public function so callers (such as FFG update nodes) can cache
    the thumbnail of the same persisted page screenshot, avoiding repeated base64 decoding + imdecode in nested loops.
    """
    if not screenshot:
        return None
    if "," in screenshot and screenshot.lstrip().startswith("data:image"):
        screenshot = screenshot.split(",", 1)[1]
    try:
        image_bytes = base64.b64decode(screenshot, validate=False)
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)
    except Exception:
        return None
    if image is None:
        return None
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def screenshot_similarity(left: str, right: str, size: int = 64) -> float | None:
    """Return a lightweight visual similarity score in [0, 1]."""
    left_thumb = screenshot_thumbnail(left, size)
    right_thumb = screenshot_thumbnail(right, size)
    if left_thumb is None or right_thumb is None:
        return None
    diff = np.mean(
        np.abs(left_thumb.astype(np.float32) - right_thumb.astype(np.float32))
    ) / 255.0
    return 1.0 - float(diff)


def normalize_llm_text(content: Any) -> str:
    """Normalize LLM response content to plain text.

    Most models return str, but some models return list-of-blocks (e.g. [{"type":"text","text":...}]).
    Calling .strip() directly on a list raises an exception, which gets swallowed by upper-level broad except and misjudged as parsing failure,
    causing a structurally valid response to be discarded. Text is extracted first uniformly here.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                value = block.get("text") or block.get("content") or ""
                if isinstance(value, str):
                    parts.append(value)
        return "".join(parts).strip()
    return str(content).strip()
