"""操作执行工具。"""

import os
import time
from typing import Any, Literal

import cv2
from langchain_core.tools import tool

from model.flow_graph import Page
from tools.utils import encode_image_to_jpg_base64

ACTION_NAMES = Literal[
    "click",
    "long_press",
    "type",
    "scroll",
    "open_app",
    "press_back",
    "finished",
    "wait",
    "capture_screenshot",
]
SCREENSHOT_CAPTURE_ATTEMPTS = max(1, int(os.getenv("SCREENSHOT_CAPTURE_ATTEMPTS", "3")))
SCREENSHOT_CAPTURE_RETRY_DELAY = max(0.0, float(os.getenv("SCREENSHOT_CAPTURE_RETRY_DELAY", "0.5")))
SCREENSHOT_BLACK_MEAN_THRESHOLD = float(os.getenv("SCREENSHOT_BLACK_MEAN_THRESHOLD", "6"))
SCREENSHOT_BLACK_STD_THRESHOLD = float(os.getenv("SCREENSHOT_BLACK_STD_THRESHOLD", "4"))


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
        mean, stddev = cv2.meanStdDev(gray)
        mean_value = float(mean[0][0])
        std_value = float(stddev[0][0])
    except Exception as exc:
        return f"截图像素检查失败: {exc}"

    if mean_value <= SCREENSHOT_BLACK_MEAN_THRESHOLD and std_value <= SCREENSHOT_BLACK_STD_THRESHOLD:
        return f"截图疑似黑屏: mean={mean_value:.2f}, std={std_value:.2f}"
    return ""


def create_device_tool(device):
    """创建一个统一的设备工具，支持动作执行和截图。"""

    @tool
    def execute_device_action(
        action_name: ACTION_NAMES,
        x: int | None = None,
        y: int | None = None,
        content: str | None = None,
        direction: Literal["up", "down", "left", "right"] | None = None,
        app_name: str | None = None,
    ) -> dict:
        """执行一次移动端动作，或直接获取当前页面截图。"""
        if action_name != "capture_screenshot":
            _perform_action(
                device=device,
                action_name=action_name,
                x=x,
                y=y,
                content=content,
                direction=direction,
                app_name=app_name,
            )
        current_page = capture_current_page(device)
        trace_entry = _build_trace_entry(
            action_name=action_name,
            x=x,
            y=y,
            content=content,
            direction=direction,
            app_name=app_name,
            current_page=current_page,
        )
        return {
            "current_page": current_page,
            "scenario_execution_trace": [trace_entry],
        }

    return execute_device_action


def _perform_action(
    device,
    action_name: str,
    x: int | None = None,
    y: int | None = None,
    content: str | None = None,
    direction: str | None = None,
    app_name: str | None = None,
    package_name: str | None = None,
):
    if action_name == "click":
        _require_xy(x, y, action_name)
        device.click(x, y)
        return

    if action_name == "long_press":
        _require_xy(x, y, action_name)
        device.long_click(x, y)
        return

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


def _build_trace_entry(
    action_name: str,
    current_page: Page,
    x: int | None = None,
    y: int | None = None,
    content: str | None = None,
    direction: str | None = None,
    app_name: str | None = None,
) -> dict[str, Any]:
    trace_entry: dict[str, Any] = {
        "action_name": action_name,
        "current_page": current_page,
    }

    optional_fields = {
        "x": x,
        "y": y,
        "content": content,
        "direction": direction,
        "app_name": app_name,
    }
    for key, value in optional_fields.items():
        if value is not None:
            trace_entry[key] = value

    return trace_entry


def _require_xy(x: int | None, y: int | None, action_name: str):
    if x is None or y is None:
        raise ValueError(f"{action_name} 操作必须提供 x 和 y")
