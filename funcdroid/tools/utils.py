"""工作流工具的公共辅助函数。"""

import base64
import os
from io import BytesIO

import cv2
import numpy as np
from PIL import Image


SCREENSHOT_MAX_SIDE = max(160, int(os.getenv("SCREENSHOT_MAX_SIDE", "1600")))
SCREENSHOT_QUALITY = max(30, min(100, int(os.getenv("SCREENSHOT_QUALITY", "95"))))


def decode_screenshot(screenshot: str):
    """Decode a trace image, returning None for missing or invalid data."""
    value = str(screenshot or "").strip()
    if value.startswith("data:image") and "," in value:
        value = value.split(",", 1)[1]
    if not value:
        return None
    try:
        image = cv2.imdecode(np.frombuffer(base64.b64decode(value), dtype=np.uint8), cv2.IMREAD_COLOR)
    except (ValueError, cv2.error):
        return None
    return image


def screenshot_dimensions(screenshot: str) -> tuple[int, int] | None:
    """读取实际上传图片的宽高；无法解码时不猜测尺寸。

    只解析图片头（PIL 惰性打开，不解码像素），避免每步为读尺寸整图解码一次 JPEG。
    """
    value = str(screenshot or "").strip()
    if value.startswith("data:image") and "," in value:
        value = value.split(",", 1)[1]
    if not value:
        return None
    try:
        raw = base64.b64decode(value)
        with Image.open(BytesIO(raw)) as image:
            width, height = image.size
    except (ValueError, OSError, Image.DecompressionBombError):
        return None
    if width <= 0 or height <= 0:
        return None
    return int(width), int(height)


def encode_image_to_jpg_base64(
    image, max_side: int = SCREENSHOT_MAX_SIDE, quality: int = SCREENSHOT_QUALITY,
) -> str:
    """统一编码执行与合规取证截图；后续合规判定直接复用编码结果。"""
    image = resize_image(image, max_side=max_side)
    success, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        raise ValueError("截图编码失败")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def resize_image(image, max_side: int):
    """缩放图片，使其最长边不超过 max_side。"""
    height, width = image.shape[:2]
    longest_side = max(height, width)
    if longest_side <= max_side:
        return image

    scale = max_side / longest_side
    resized_width = max(1, int(width * scale))
    resized_height = max(1, int(height * scale))
    return cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
