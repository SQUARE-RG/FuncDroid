"""节点间共享的常量与轻量 helper。"""

from __future__ import annotations

import base64
import re
from typing import Any

import cv2
import numpy as np

JSON_BLOCK_PATTERN = re.compile(r"^```(?:json)?\s*|\s*```$")

EXHAUSTIVE_CONTROL_FAMILY_PATTERN = re.compile(
    r"(?:所有|全部|每个|逐个|逐一|遍历)"
    r"[^→，,.\u3002；;]{0,28}?"
    r"(?:表情|选项|条目|按钮|控件|标签|分类|菜单项|导航项|列表项|图标|项目|结果)"
    r"|\b(?:all|every|each)\b.{0,28}\b"
    r"(?:items?|options?|buttons?|tabs?|emojis?|entries?|results?)\b",
    re.IGNORECASE,
)

EXPLICIT_BULK_OPERATION_PATTERN = re.compile(
    r"全选按钮|点击全选|选择全部(?:内容|文本|文件)|删除全部|全部删除|"
    r"清空全部|全部清除|一键(?:清空|删除)|\bselect all\b|\bdelete all\b|\bclear all\b",
    re.IGNORECASE,
)


def has_exhaustive_control_family_intent(value: object) -> bool:
    """识别要求穷尽同功能重复控件的场景，但保留真正的单次批量操作。"""
    text = str(value or "").strip()
    if not text:
        return False
    candidate = EXPLICIT_BULK_OPERATION_PATTERN.sub("", text)
    return EXHAUSTIVE_CONTROL_FAMILY_PATTERN.search(candidate) is not None


def representative_control_family_scenario(value: object) -> str:
    """将漏过规划校验的穷尽式场景收缩为一个代表项。"""
    text = str(value or "").strip()
    if not has_exhaustive_control_family_intent(text):
        return text

    normalized = re.sub(r"逐个|逐一|遍历", "", text)
    normalized = re.sub(r"依次(?=点击|选择|打开|勾选)", "", normalized)
    normalized = re.sub(r"所有|全部|每个", "一个代表性的", normalized)
    normalized = re.sub(r"\b(?:all|every|each)\b", "one representative", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return f"同类控件只操作一个代表项，首次结果出现后立即结束；{normalized}"


def screenshot_thumbnail(screenshot: str, size: int = 64) -> np.ndarray | None:
    """把截图字符串解码为灰度缩略图；无法解码时返回 None。

    独立成公共函数是为了让调用方（如 FFG 更新节点）能把同一张已持久化
    页面截图的缩略图缓存起来，避免在多重循环里反复 base64 解码 + imdecode。
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
    """把 LLM 响应的 content 归一化为纯文本。

    多数模型返回 str，但部分模型返回 list-of-blocks（如 [{"type":"text","text":...}]）。
    直接对 list 调 .strip() 会抛异常，被上层的宽泛 except 吞掉后误判为解析失败，
    导致一次结构合法的响应被丢弃。这里统一先抽取文本。
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
