"""Shared constants and lightweight node utilities."""

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

# Unicode escapes preserve multilingual UI and legacy scenario matching.
EXHAUSTIVE_CONTROL_FAMILY_PATTERN = re.compile(
    r"(?:\u6240\u6709|\u5168\u90e8|\u6bcf\u4e2a|\u9010\u4e2a|\u9010\u4e00|\u904d\u5386)"
    r"[^→，,.\u3002；;]{0,28}?"
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
    """Identify exhaustive same-functionality control exploration while retaining single bulk operations."""
    text = str(value or "").strip()
    if not text:
        return False
    candidate = EXPLICIT_BULK_OPERATION_PATTERN.sub("", text)
    return EXHAUSTIVE_CONTROL_FAMILY_PATTERN.search(candidate) is not None


class BoundedCache:
    """Bounded least-recently-used cache for single-threaded execution."""

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


# Cache decoded screenshots shared by planning, action comparison, and stagnation detection. Use content hashes instead of retaining large screenshot strings.
SCREENSHOT_DECODE_CACHE_SIZE = max(8, int(os.getenv("SCREENSHOT_DECODE_CACHE_SIZE", "512")))
_THUMBNAIL_CACHE = BoundedCache(SCREENSHOT_DECODE_CACHE_SIZE)
_CHANGE_IMAGE_CACHE = BoundedCache(max(8, SCREENSHOT_DECODE_CACHE_SIZE // 8))


def screenshot_digest(screenshot: str) -> str:
    """Hash screenshot content for the decoding cache key."""
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
    """Decode a screenshot into a grayscale thumbnail, or return None on failure.

    Cache by content hash in a bounded LRU. Cached arrays are read-only; copy before editing."""
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


# Ignore the changing system status bar. Apply a per-pixel threshold so JPEG noise and cursor blinking do not count as action responses.
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
    """Return original height, original width, and grayscale image scaled to CHANGE_DIFF_WIDTH; cache by content hash."""
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
    """Return grayscale pixels scaled to (CHANGE_DIFF_WIDTH, height), reusing matching cached dimensions."""
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
    """Compare screenshots and return the changed-pixel ratio and enclosing region.

    The region uses relative (left, top, right, bottom) coordinates in [0, 1].
    Return a None region when unchanged, or None for decoding or aspect-ratio failures.
    Cache scaled grayscale pixels by content hash to avoid repeated full-resolution decoding."""
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
    """Normalize LLM response content into plain text.

    Accept strings and lists of text blocks. Extract text before parsing so a valid
    block response is not discarded by calling string methods on a list."""
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
