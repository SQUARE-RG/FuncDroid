"""跨消费方共享的截图压缩缓存。

FFG 更新、缺陷判定、合规判定与页面审计会按各自参数压缩同一批轨迹截图；此前各自维护
缓存或直接重复压缩，同一张原图在一轮内最多被解码/缩放/编码 3 次。压缩是
(原始 base64, max_side, quality) 的纯函数，这里提供进程级有界 LRU：相同输入
直接返回缓存结果；缓存有界，避免长跑期间全部历史压缩图常驻内存。

缓存键只保存原图 base64 的 sha1 摘要 + 参数，不保存原图本身：原图单张数百 KB，
若直接作键，128 项会让原图与压缩图一起常驻内存。
"""

from __future__ import annotations

import base64
import hashlib
import threading
from collections import OrderedDict

import cv2
import numpy as np

from tools.utils import resize_image

# 工作集约为单轮轨迹（≤25 步）+ 参考图；128 项可覆盖数轮历史，常驻的只有
# 压缩后的结果（每项数十 KB 级）与 40 字节摘要。
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
    """有界 LRU（键为摘要 + 参数）；异常不缓存，下次调用重试。"""
    key = _cache_key(screenshot, max_side, quality)
    with _cache_lock:
        cached = _cache.get(key)
        if cached is not None:
            _cache.move_to_end(key)
            return cached
    # 压缩在锁外进行，避免多个后台线程互相串行等待；并发首次命中至多重复压缩一次。
    result = _compress(screenshot, int(max_side), int(quality))
    # 压缩失败时原样返回原图：不缓存，免得把原图放进缓存常驻。
    if result is screenshot:
        return result
    with _cache_lock:
        _cache[key] = result
        _cache.move_to_end(key)
        while len(_cache) > _CACHE_SIZE:
            _cache.popitem(last=False)
    return result


def clear_compact_cache() -> None:
    """清空进程级压缩缓存；供测试隔离使用。"""
    with _cache_lock:
        _cache.clear()


def compact_screenshot(
    screenshot: str,
    *,
    max_side: int,
    quality: int,
    source_label: str = "截图压缩",
) -> str:
    """带全局缓存的压缩入口；无法压缩时原样返回输入（与既有各处行为一致）。

    缓存不保存异常：解码/编码失败会每次重试并在调用方日志中标注来源。
    """
    screenshot = str(screenshot or "").strip()
    if not screenshot:
        return ""
    if screenshot.startswith("data:image") and "," in screenshot:
        screenshot = screenshot.split(",", 1)[1]
    try:
        return _compact_cached(screenshot, int(max_side), int(quality))
    except Exception as exc:  # noqa: BLE001 - 解码/编码失败时回退原图，不中断消费方
        print(f"[{source_label}] 压缩截图失败，使用原图: {exc}")
        return screenshot
