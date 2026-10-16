"""节点间共享的常量与轻量 helper。"""

from __future__ import annotations

import base64
import hashlib
import os
import re
from collections import OrderedDict
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


class BoundedCache:
    """按插入/访问顺序淘汰的有界 LRU 缓存（单线程场景使用）。"""

    def __init__(self, max_entries: int):
        self.max_entries = max(1, int(max_entries))
        self._items: OrderedDict = OrderedDict()

    def get(self, key, default=None):
        if key not in self._items:
            return default
        self._items.move_to_end(key)
        return self._items[key]

    def __contains__(self, key) -> bool:
        return key in self._items

    def put(self, key, value) -> None:
        self._items[key] = value
        self._items.move_to_end(key)
        while len(self._items) > self.max_entries:
            self._items.popitem(last=False)

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)


# 截图解码缓存：同一张截图在规划（当前页匹配）、执行（动作效果、三个停滞
# 检测器）中会被反复解码。按截图内容哈希缓存，避免持有大字符串本身。
SCREENSHOT_DECODE_CACHE_SIZE = max(8, int(os.getenv("SCREENSHOT_DECODE_CACHE_SIZE", "512")))
_THUMBNAIL_CACHE = BoundedCache(SCREENSHOT_DECODE_CACHE_SIZE)
_CHANGE_IMAGE_CACHE = BoundedCache(max(8, SCREENSHOT_DECODE_CACHE_SIZE // 8))


def screenshot_digest(screenshot: str) -> str:
    """截图内容哈希，用作解码缓存键。"""
    return hashlib.blake2b(str(screenshot or "").encode("utf-8"), digest_size=16).hexdigest()


def _decode_thumbnail(screenshot: str, size: int) -> np.ndarray | None:
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


def screenshot_thumbnail(screenshot: str, size: int = 64) -> np.ndarray | None:
    """把截图字符串解码为灰度缩略图；无法解码时返回 None。

    结果按截图内容哈希缓存（有界 LRU），返回的数组只读，调用方需要修改时自行复制。
    """
    if not screenshot:
        return None
    key = (screenshot_digest(screenshot), int(size))
    if key in _THUMBNAIL_CACHE:
        return _THUMBNAIL_CACHE.get(key)
    thumbnail = _decode_thumbnail(screenshot, size)
    if thumbnail is not None:
        thumbnail.flags.writeable = False
    _THUMBNAIL_CACHE.put(key, thumbnail)
    return thumbnail


# 动作效果比较：顶部系统状态栏（时钟、信号）会自行变化，不计入界面变化；
# 单像素灰度差超过阈值才算“变了”，以免 JPEG 噪声和光标闪烁被当成响应。
CHANGE_DIFF_WIDTH = 240
CHANGE_STATUS_BAR_RATIO = 0.04
CHANGE_PIXEL_THRESHOLD = 40


def _gray_image(screenshot: str) -> np.ndarray | None:
    if not screenshot:
        return None
    if "," in screenshot and screenshot.lstrip().startswith("data:image"):
        screenshot = screenshot.split(",", 1)[1]
    try:
        image_bytes = base64.b64decode(screenshot, validate=False)
        return cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    except Exception:
        return None


def _change_diff_image(screenshot: str) -> tuple[int, int, np.ndarray] | None:
    """返回 (原图高, 原图宽, 按 CHANGE_DIFF_WIDTH 等比缩放的灰度图)，按内容哈希缓存。"""
    if not screenshot:
        return None
    key = screenshot_digest(screenshot)
    if key in _CHANGE_IMAGE_CACHE:
        return _CHANGE_IMAGE_CACHE.get(key)
    image = _gray_image(screenshot)
    entry = None
    if image is not None:
        height, width = image.shape[:2]
        if width:
            small_height = max(1, int(round(CHANGE_DIFF_WIDTH * height / width)))
            small = cv2.resize(image, (CHANGE_DIFF_WIDTH, small_height), interpolation=cv2.INTER_AREA)
            small.flags.writeable = False
            entry = (height, width, small)
    _CHANGE_IMAGE_CACHE.put(key, entry)
    return entry


def _change_small_image(
    screenshot: str,
    cached: tuple[int, int, np.ndarray],
    height: int,
) -> np.ndarray | None:
    """取缩放到 (CHANGE_DIFF_WIDTH, height) 的灰度图；与缓存尺寸一致时直接复用。"""
    if cached[2].shape[0] == height:
        return cached[2]
    image = _gray_image(screenshot)
    if image is None:
        return None
    return cv2.resize(image, (CHANGE_DIFF_WIDTH, height), interpolation=cv2.INTER_AREA)


def screenshot_change_region(
    before: str,
    after: str,
) -> tuple[float, tuple[float, float, float, float] | None] | None:
    """比较动作前后截图，返回 (变化像素占比, 变化区域外接框)。

    外接框用 0–1 的相对坐标 (left, top, right, bottom) 表示，调用方按当前
    图片尺寸换算；无变化时外接框为 None。无法解码或比例不一致时返回 None。
    缩放后的灰度图按截图内容哈希缓存，最近几步效果反复计算时不再全尺寸解码。
    """
    left_cached = _change_diff_image(before)
    right_cached = _change_diff_image(after)
    if left_cached is None or right_cached is None:
        return None
    left_h, left_w = left_cached[:2]
    right_h, right_w = right_cached[:2]
    if not left_w or not right_w or abs(left_h / left_w - right_h / right_w) > 0.02:
        return None
    width = CHANGE_DIFF_WIDTH
    height = max(1, int(round(width * right_h / right_w)))
    left_small = _change_small_image(before, left_cached, height)
    right_small = _change_small_image(after, right_cached, height)
    if left_small is None or right_small is None:
        return None
    top = int(height * CHANGE_STATUS_BAR_RATIO)
    changed = (
        np.abs(left_small[top:].astype(np.int16) - right_small[top:].astype(np.int16))
        > CHANGE_PIXEL_THRESHOLD
    )
    ratio = float(changed.mean()) if changed.size else 0.0
    if not changed.any():
        return ratio, None
    rows = np.where(changed.any(axis=1))[0]
    cols = np.where(changed.any(axis=0))[0]
    return ratio, (
        float(cols[0]) / width,
        float(rows[0] + top) / height,
        float(cols[-1] + 1) / width,
        float(rows[-1] + top + 1) / height,
    )


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
