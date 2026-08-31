"""Common helper functions for workflow tools."""

import base64

import cv2


def encode_image_to_jpg_base64(image, max_side: int = 960, quality: int = 70) -> str:
    """Resize the image as needed, encode it as JPEG, and return a base64 string."""
    image = resize_image(image, max_side=max_side)
    success, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        raise ValueError("Failed to encode screenshot")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def resize_image(image, max_side: int):
    """Resize the image so that its longest side does not exceed max_side."""
    height, width = image.shape[:2]
    longest_side = max(height, width)
    if longest_side <= max_side:
        return image

    scale = max_side / longest_side
    resized_width = max(1, int(width * scale))
    resized_height = max(1, int(height * scale))
    return cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
