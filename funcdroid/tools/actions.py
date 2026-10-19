"""Device action execution and screenshot capture."""

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
        return "Screenshot is empty"
    shape = getattr(image, "shape", None)
    if not shape or len(shape) < 2:
        return "Invalid screenshot format"
    height, width = int(shape[0] or 0), int(shape[1] or 0)
    if height <= 1 or width <= 1:
        return f"Invalid screenshot dimensions: {width}x{height}"

    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(shape) >= 3 else image
        # Exclude the system status bar: it can remain visible above secure black content and inflate image statistics.
        top_skip = height // 12
        body = gray[top_skip:, :] if height > top_skip * 2 else gray
        mean, stddev = cv2.meanStdDev(body)
        mean_value = float(mean[0][0])
        std_value = float(stddev[0][0])
    except Exception as exc:
        return f"Screenshot pixel check failed: {exc}"

    if mean_value <= SCREENSHOT_BLACK_MEAN_THRESHOLD and std_value <= SCREENSHOT_BLACK_STD_THRESHOLD:
        return f"Screenshot appears black: mean={mean_value:.2f}, std={std_value:.2f}"
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
            raise ValueError("type requires content")
        device.input(content)
        return

    if action_name == "scroll":
        if not direction:
            raise ValueError("scroll requires direction")
        device.swipe_ext(direction)
        return

    if action_name == "swipe":
        # Exact screen-pixel gestures support system panels, drawers, and dragging that scroll cannot express.
        for key, value in (("x1", x1), ("y1", y1), ("x2", x2), ("y2", y2)):
            if value is None:
                raise ValueError(f"swipe requires {key}")
        start_x, start_y, end_x, end_y = int(x1), int(y1), int(x2), int(y2)
        if start_x == end_x and start_y == end_y:
            raise ValueError("swipe endpoints must differ")
        device.swipe(start_x, start_y, end_x, end_y, float(duration or SWIPE_DEFAULT_DURATION))
        return

    if action_name == "open_app":
        package_to_open = (package_name or app_name or "").strip()
        if not package_to_open:
            raise ValueError("open_app requires package_name or app_name")
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

    raise ValueError(f"Unsupported action: {action_name}")


def capture_current_page(
    device,
    page_id: str | None = None,
) -> Page:
    """Capture a healthy page; reject black or unavailable screenshots during startup/recovery."""
    page, health_error = capture_page_observation(device, page_id=page_id)
    if health_error:
        raise ScreenshotUnavailableError(health_error)
    return page


def capture_runtime_page(
    device,
    page_id: str | None = None,
) -> tuple[Page, str]:
    """Capture a runtime page, retaining visible black frames for executor and planner decisions."""
    return capture_page_observation(device, page_id=page_id)


def capture_page_observation(
    device,
    page_id: str | None = None,
) -> tuple[Page, str]:
    """Return the last captured image and health error; raise only if no image was obtained.

    Args:
        device: Automation driver.
        page_id: Optional page identifier."""
    last_error = ""
    image = None
    for attempt in range(1, SCREENSHOT_CAPTURE_ATTEMPTS + 1):
        try:
            image = device.screenshot()
        except Exception as exc:
            last_error = f"Screenshot capture error: {exc}"
            print(f"[Screenshot health] {last_error} ({attempt}/{SCREENSHOT_CAPTURE_ATTEMPTS})")
        else:
            last_error = _screenshot_health_error(image)
            if not last_error:
                break
            print(f"[Screenshot health] {last_error} ({attempt}/{SCREENSHOT_CAPTURE_ATTEMPTS})")
            if attempt < SCREENSHOT_CAPTURE_ATTEMPTS and "black" in last_error:
                # Wake the screen before retrying a black frame; screen sleep is a common cause.
                _wake_screen(device)
        if attempt < SCREENSHOT_CAPTURE_ATTEMPTS:
            time.sleep(SCREENSHOT_CAPTURE_RETRY_DELAY)
    if image is None:
        raise ScreenshotUnavailableError(last_error or "Repeated screenshot unavailability")

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
    except Exception as exc:  # noqa: BLE001 - Wake is best effort and must not prevent screenshot retries.
        print(f"[Screenshot health] Could not wake screen: {exc}")


def _require_xy(x: int | None, y: int | None, action_name: str):
    if x is None or y is None:
        raise ValueError(f"{action_name} requires x and y")
