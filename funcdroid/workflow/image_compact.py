"""Bounded screenshot-compaction cache shared across consumers.

Cache the pure transformation of source image, maximum edge, and JPEG quality. Keys contain only the source SHA-1 and parameters; a bounded process-local LRU avoids retaining original images."""

from __future__ import annotations

import base64
import hashlib
import threading
from collections import OrderedDict

import cv2
import numpy as np

from tools.utils import resize_image

# A 128-entry cache spans several trace rounds while retaining compact images and digests only.
_CACHE_SIZE = 128

_cache: "OrderedDict[tuple[str, int, int], str]" = OrderedDict()
_cache_lock = threading.Lock()


def _compress(screenshot: str, max_side: int, quality: int) -> str:
    image_bytes = base64.b64decode(screenshot, validate=False)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        return screenshot
    image = resize_image(image, max_side=max_side)
    success, buffer = cv2.imencode(
        ".jpg",
        image,
        [int(cv2.IMWRITE_JPEG_QUALITY), quality],
    )
    if not success:
        return screenshot
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def _cache_key(screenshot: str, max_side: int, quality: int) -> tuple[str, int, int]:
    digest = hashlib.sha1(screenshot.encode("utf-8", errors="ignore")).hexdigest()
    return digest, int(max_side), int(quality)


def _compact_cached(screenshot: str, max_side: int, quality: int) -> str:
    """Bounded LRU using digest/parameter keys; never cache failures."""
    key = _cache_key(screenshot, max_side, quality)
    with _cache_lock:
        cached = _cache.get(key)
        if cached is not None:
            _cache.move_to_end(key)
            return cached
    # Compress outside the lock; simultaneous misses may perform duplicate work without blocking consumers.
    result = _compress(screenshot, int(max_side), int(quality))
    # Return uncached originals on failure instead of retaining large source images.
    if result is screenshot:
        return result
    with _cache_lock:
        _cache[key] = result
        _cache.move_to_end(key)
        while len(_cache) > _CACHE_SIZE:
            _cache.popitem(last=False)
    return result


def clear_compact_cache() -> None:
    """Clear the process-wide cache for test isolation."""
    with _cache_lock:
        _cache.clear()


def compact_screenshot(
    screenshot: str,
    *,
    max_side: int,
    quality: int,
    source_label: str = "Screenshot compaction",
) -> str:
    """Compact through the shared cache, returning the original image on failure.

    Failures are retried on subsequent calls and logged with the consumer label."""
    screenshot = str(screenshot or "").strip()
    if not screenshot:
        return ""
    if screenshot.startswith("data:image") and "," in screenshot:
        screenshot = screenshot.split(",", 1)[1]
    try:
        return _compact_cached(screenshot, int(max_side), int(quality))
    except Exception as exc:  # noqa: BLE001 - Fall back to the original on decode/encode failure without interrupting consumers.
        print(f"[{source_label}] Screenshot compaction failed; using original image: {exc}")
        return screenshot
