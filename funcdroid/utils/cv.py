import cv2
import base64
from PIL import Image, ImageDraw
import numpy as np
import io, base64

def read(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    return img

def write(img_path, img):
    cv2.imwrite(img_path, img)

def _crop(img, bound):
    (x1, y1), (x2, y2) = bound
    return img[y1:y2, x1:x2]

def encode_image(image, quality=85, max_size=(800, 1400)):
    # 获取原始图像尺寸
    height, width = image.shape[:2]
    
    # 计算缩放比例，保持宽高比
    if width > max_size[0] or height > max_size[1]:
        scale_w = max_size[0] / width
        scale_h = max_size[1] / height
        scale = min(scale_w, scale_h)
        
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # 缩放图像
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    else:
        resized_image = image
    
    # 使用JPEG编码并设置质量参数
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, buffer = cv2.imencode('.jpeg', resized_image, encode_params)
    encoded_image = base64.b64encode(buffer).decode('utf-8')
    return encoded_image


def combine_images_horizontal(img_list, show=False):
    """
    将多张 cv2 图片水平拼接为一张，中间添加粗箭头分隔，可选是否显示。
    """
    if not img_list:
        raise ValueError("img_list 为空")

    # cv2 -> PIL
    pil_images = []
    for img in img_list:
        if img is None:
            raise ValueError("存在 None 图片，请检查路径是否正确")
        if isinstance(img, np.ndarray):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_images.append(Image.fromarray(img))
        elif isinstance(img, Image.Image):
            pil_images.append(img)
        else:
            raise TypeError("图像类型错误，必须是 cv2 读取的 numpy.ndarray 或 PIL.Image")

    # 统一高度
    min_height = min(img.height for img in pil_images)
    resized = []
    for img in pil_images:
        ratio = min_height / img.height
        new_w = int(img.width * ratio)
        resized.append(img.resize((new_w, min_height)))

    # 箭头宽度
    arrow_width = 100
    total_width = sum(img.width for img in resized) + arrow_width * (len(resized) - 1)
    new_img = Image.new("RGB", (total_width, min_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(new_img)

    # 拼接 + 画箭头
    x_offset = 0
    for idx, img in enumerate(resized):
        new_img.paste(img, (x_offset, 0))
        x_offset += img.width

        if idx < len(resized) - 1:
            center_y = min_height // 2
            arrow_start = x_offset + 10
            arrow_end = arrow_start + arrow_width - 20
            draw.line((arrow_start, center_y, arrow_end, center_y), fill=(0, 0, 0), width=8)
            draw.polygon([
                (arrow_end, center_y),
                (arrow_end - 20, center_y - 12),
                (arrow_end - 20, center_y + 12)
            ], fill=(0, 0, 0))
            x_offset += arrow_width

    # 可选显示
    if show:
        new_img.show()

    # 转为 base64
    buf = io.BytesIO()
    new_img.save(buf, format="JPEG")
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return img_base64





