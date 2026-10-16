"""操作执行工具。"""

import os
import time

import cv2

from model.flow_graph import Page
from tools.utils import encode_image_to_jpg_base64

SCREENSHOT_CAPTURE_ATTEMPTS = max(1, int(os.getenv("SCREENSHOT_CAPTURE_ATTEMPTS", "3")))
SCREENSHOT_CAPTURE_RETRY_DELAY = max(0.0, float(os.getenv("SCREENSHOT_CAPTURE_RETRY_DELAY", "0.5")))
SCREENSHOT_BLACK_MEAN_THRESHOLD = float(os.getenv("SCREENSHOT_BLACK_MEAN_THRESHOLD", "6"))
SCREENSHOT_BLACK_STD_THRESHOLD = float(os.getenv("SCREENSHOT_BLACK_STD_THRESHOLD", "4"))
SWIPE_DEFAULT_DURATION = max(0.05, float(os.getenv("EXEC_SWIPE_DEFAULT_DURATION", "0.5")))


class ScreenshotUnavailableError(RuntimeError):
    """Raised when the device repeatedly returns invalid or unusable screenshots."""


def _screenshot_health_error(image) -> str:
    if image is None:
        return "截图为空"
    shape = getattr(image, "shape", None)
    if not shape or len(shape) < 2:
        return "截图格式无效"
    height, width = int(shape[0] or 0), int(shape[1] or 0)
    if height <= 1 or width <= 1:
        return f"截图尺寸异常: {width}x{height}"

    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(shape) >= 3 else image
        # 状态栏由系统渲染，防截屏页面（FLAG_SECURE 等）下依然可见，会把全黑
        # 画面的均值/方差抬到阈值之上造成漏检；跳过顶部状态栏区域再判定。
        top_skip = height // 12
        body = gray[top_skip:, :] if height > top_skip * 2 else gray
        mean, stddev = cv2.meanStdDev(body)
        mean_value = float(mean[0][0])
        std_value = float(stddev[0][0])
    except Exception as exc:
        return f"截图像素检查失败: {exc}"

    if mean_value <= SCREENSHOT_BLACK_MEAN_THRESHOLD and std_value <= SCREENSHOT_BLACK_STD_THRESHOLD:
        return f"截图疑似黑屏: mean={mean_value:.2f}, std={std_value:.2f}"
    return ""


def _perform_action(
    device,
    action_name: str,
    x: int | None = None,
    y: int | None = None,
    content: str | None = None,
    direction: str | None = None,
    app_name: str | None = None,
    package_name: str | None = None,
    x1: int | None = None,
    y1: int | None = None,
    x2: int | None = None,
    y2: int | None = None,
    duration: float | None = None,
):
    if action_name == "click":
        _require_xy(x, y, action_name)
        return device.click(x, y)

    if action_name == "long_press":
        _require_xy(x, y, action_name)
        return device.long_click(x, y)

    if action_name == "type":
        if not content:
            raise ValueError("type 操作必须提供 content")
        device.input(content)
        return

    if action_name == "scroll":
        if not direction:
            raise ValueError("scroll 操作必须提供 direction")
        device.swipe_ext(direction)
        return

    if action_name == "swipe":
        # 精确手势：用于 scroll 无法完成的系统级动作（通知栏下拉、侧滑抽屉、
        # 精确拖动）。坐标已由执行器换算为屏幕像素。
        for key, value in (("x1", x1), ("y1", y1), ("x2", x2), ("y2", y2)):
            if value is None:
                raise ValueError(f"swipe 操作必须提供 {key}")
        start_x, start_y, end_x, end_y = int(x1), int(y1), int(x2), int(y2)
        if start_x == end_x and start_y == end_y:
            raise ValueError("swipe 操作起点与终点不能相同")
        device.swipe(start_x, start_y, end_x, end_y, float(duration or SWIPE_DEFAULT_DURATION))
        return

    if action_name == "open_app":
        package_to_open = (package_name or app_name or "").strip()
        if not package_to_open:
            raise ValueError("open_app 操作必须提供 package_name 或 app_name")
        # Scenario execution supplies ``package_name`` when this is a recovery
        # action for the target app. Restart only when the app IS foreground
        # but stuck in a deep Activity (restart reliably returns to the
        # launcher entry). When it is NOT foreground (desktop / another app /
        # still cold-starting), merely start it: a force-stop restart here
        # used to feed back into itself — the model saw the launcher during
        # the slow cold start, re-issued open_app, and each retry force-killed
        # the app again. Generic tool calls that only provide app_name keep
        # the ordinary non-destructive start behavior.
        if package_name:
            try:
                current_package = str(device.current_package() or "").strip()
            except Exception:
                current_package = ""
            if current_package == package_to_open:
                device.restart_app(package_to_open)
            else:
                device.start_app(package_to_open)
        else:
            device.start_app(package_to_open)
        return

    if action_name == "press_back":
        device.back()
        return

    if action_name in {"finished", "wait"}:
        return

    raise ValueError(f"不支持的操作类型: {action_name}")


def capture_current_page(
    device,
    page_id: str | None = None,
) -> Page:
    """严格截取当前页面；启动和重启恢复时拒绝黑屏或不可用截图。"""
    page, health_error = capture_page_observation(device, page_id=page_id)
    if health_error:
        raise ScreenshotUnavailableError(health_error)
    return page


def capture_runtime_page(
    device,
    page_id: str | None = None,
) -> tuple[Page, str]:
    """运行时截取页面，同时保留可见黑屏供执行器和下一轮规划判断。"""
    return capture_page_observation(device, page_id=page_id)


def capture_page_observation(
    device,
    page_id: str | None = None,
) -> tuple[Page, str]:
    """返回最后一次实际截图和健康错误；完全无法取得图片时才抛异常。

    Args:
        device: 设备驱动
        page_id: 页面 ID（可选）
    """
    last_error = ""
    image = None
    for attempt in range(1, SCREENSHOT_CAPTURE_ATTEMPTS + 1):
        try:
            image = device.screenshot()
        except Exception as exc:
            last_error = f"截图采集异常: {exc}"
            print(f"[截图健康检查] {last_error} ({attempt}/{SCREENSHOT_CAPTURE_ATTEMPTS})")
        else:
            last_error = _screenshot_health_error(image)
            if not last_error:
                break
            print(f"[截图健康检查] {last_error} ({attempt}/{SCREENSHOT_CAPTURE_ATTEMPTS})")
            if attempt < SCREENSHOT_CAPTURE_ATTEMPTS and "黑屏" in last_error:
                # 黑屏最常见的原因是设备息屏：重试前先尽力唤醒屏幕。
                _wake_screen(device)
        if attempt < SCREENSHOT_CAPTURE_ATTEMPTS:
            time.sleep(SCREENSHOT_CAPTURE_RETRY_DELAY)
    if image is None:
        raise ScreenshotUnavailableError(last_error or "连续截图不可用")

    page = Page(
        id=page_id,
        activity=device.current_activity(),
        representative_screenshot=encode_image_to_jpg_base64(image),
    )
    return page, last_error


def _wake_screen(device) -> None:
    wake = getattr(device, "wake_screen", None)
    if not callable(wake):
        return
    try:
        wake()
    except Exception as exc:  # noqa: BLE001 - 唤醒只是尽力而为，不影响截图重试
        print(f"[截图健康检查] 唤醒屏幕失败: {exc}")


def _require_xy(x: int | None, y: int | None, action_name: str):
    if x is None or y is None:
        raise ValueError(f"{action_name} 操作必须提供 x 和 y")
