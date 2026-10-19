"""Shared workflow utilities."""

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
    """Read actual uploaded-image dimensions; never guess after decoding failure.

    Parse only the image header with lazy PIL loading to avoid decoding every JPEG just to read its size."""
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
    """Encode execution evidence once so asynchronous reviewers can reuse the same image."""
    image = resize_image(image, max_side=max_side)
    success, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        raise ValueError("Screenshot encoding failed")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def resize_image(image, max_side: int):
    """Resize an image so its longest edge is at most max_side."""
    height, width = image.shape[:2]
    longest_side = max(height, width)
    if longest_side <= max_side:
        return image

    scale = max_side / longest_side
    resized_width = max(1, int(width * scale))
    resized_height = max(1, int(height * scale))
    return cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
