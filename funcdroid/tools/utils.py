"""工作流工具的公共辅助函数。"""

import base64

import cv2


def encode_image_to_jpg_base64(image, max_side: int = 960, quality: int = 70) -> str:
    """将图片按需缩放后编码为 JPEG，并返回 base64 字符串。"""
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
