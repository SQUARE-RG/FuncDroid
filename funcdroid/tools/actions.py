"""Action execution tools."""

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
        return "Screenshot is empty"
    shape = getattr(image, "shape", None)
    if not shape or len(shape) < 2:
        return "Invalid screenshot format"
    height, width = int(shape[0] or 0), int(shape[1] or 0)
    if height <= 1 or width <= 1:
        return f"Abnormal screenshot dimensions: {width}x{height}"

    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(shape) >= 3 else image
        mean, stddev = cv2.meanStdDev(gray)
        mean_value = float(mean[0][0])
        std_value = float(stddev[0][0])
    except Exception as exc:
        return f"Screenshot pixel check failed: {exc}"

    if mean_value <= SCREENSHOT_BLACK_MEAN_THRESHOLD and std_value <= SCREENSHOT_BLACK_STD_THRESHOLD:
        return f"Screenshot appears to be a black screen: mean={mean_value:.2f}, std={std_value:.2f}"
    return ""


def create_device_tool(device):
    """Create a unified device tool that supports action execution and screenshot capture."""

    @tool
    def execute_device_action(
        action_name: ACTION_NAMES,
        x: int | None = None,
        y: int | None = None,
        content: str | None = None,
        direction: Literal["up", "down", "left", "right"] | None = None,
        app_name: str | None = None,
    ) -> dict:
        """Execute a mobile action, or directly capture the current page screenshot."""
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
            raise ValueError("The type action must provide content")
        device.input(content)
        return

    if action_name == "scroll":
        if not direction:
            raise ValueError("The scroll action must provide direction")
        device.swipe_ext(direction)
        return

    if action_name == "open_app":
        package_to_open = (package_name or app_name or "").strip()
        if not package_to_open:
            raise ValueError("The open_app action must provide package_name or app_name")
        # Scenario execution supplies ``package_name`` when this is a recovery
        # action for the target app. Restart only when the app IS foreground
        # but stuck in a deep Activity (restart reliably returns to the
        # launcher entry). When it is NOT foreground (desktop / another app /
        # still cold-starting), merely start it: a force-stop restart here
        # used to feed back into itself - the model saw the launcher during
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

    raise ValueError(f"Unsupported action type: {action_name}")


def capture_current_page(
    device,
    page_id: str | None = None,
) -> Page:
    """Strictly capture the current page; reject black screens or unavailable screenshots during startup and restart recovery."""
    page, health_error = capture_page_observation(device, page_id=page_id)
    if health_error:
        raise ScreenshotUnavailableError(health_error)
    return page


def capture_runtime_page(
    device,
    page_id: str | None = None,
) -> tuple[Page, str]:
    """Capture the page at runtime while retaining visible black screens for executor and next-round planning evaluation."""
    return capture_page_observation(device, page_id=page_id)


def capture_page_observation(
    device,
    page_id: str | None = None,
) -> tuple[Page, str]:
    """Return the last actual screenshot and health error; raise an exception only when an image cannot be captured at all.

    Args:
        device: Device driver
        page_id: Page ID (optional)
    """
    last_error = ""
    image = None
    for attempt in range(1, SCREENSHOT_CAPTURE_ATTEMPTS + 1):
        try:
            image = device.screenshot()
        except Exception as exc:
            last_error = f"Screenshot capture exception: {exc}"
            print(f"[Screenshot Health Check] {last_error} ({attempt}/{SCREENSHOT_CAPTURE_ATTEMPTS})")
        else:
            last_error = _screenshot_health_error(image)
            if not last_error:
                break
            print(f"[Screenshot Health Check] {last_error} ({attempt}/{SCREENSHOT_CAPTURE_ATTEMPTS})")
        if attempt < SCREENSHOT_CAPTURE_ATTEMPTS:
            time.sleep(SCREENSHOT_CAPTURE_RETRY_DELAY)
    if image is None:
        raise ScreenshotUnavailableError(last_error or "Consecutive screenshots are unavailable")

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
        raise ValueError(f"The {action_name} action must provide x and y")
