import os
import re
import shlex
import struct
import subprocess
import time

import cv2
import numpy as np
import uiautomator2
from loguru import logger

from .base import Automator, parse_activity_top, validate_package_name


ADB_SCREENSHOT_TIMEOUT = max(1.0, float(os.getenv("ADB_SCREENSHOT_TIMEOUT", "12")))
ADB_WM_SIZE_TIMEOUT = max(1.0, float(os.getenv("ADB_WM_SIZE_TIMEOUT", "5")))
ADB_INPUT_TIMEOUT = max(3.0, float(os.getenv("ADB_INPUT_TIMEOUT", "10")))
LONG_PRESS_DURATION_MS = max(500, int(os.getenv("LONG_PRESS_DURATION_MS", "1500")))
# Minimum shell swipe duration prevents short gestures from being interpreted as flings.
MIN_SWIPE_DURATION_MS = max(50, int(os.getenv("MIN_SWIPE_DURATION_MS", "100")))
# Runtime permission entries require a dotted name, excluding colon-bearing subsection labels.
RUNTIME_PERMISSION_LINE_PATTERN = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_]*\.[A-Za-z0-9_.]+)\s*:")
# Locate requested-permission headings by text because dumpsys indentation varies across devices.
REQUESTED_PERMISSIONS_HEADER_PATTERN = re.compile(
    r"^[ \t]*requested permissions:[^\n]*\[([^\]]*)\]", re.MULTILINE
)
RUNTIME_PERMISSIONS_HEADER_PATTERN = re.compile(r"^[ \t]*runtime permissions:", re.MULTILINE)
# Peer headings end the current section.
TOP_LEVEL_SECTION_PATTERN = re.compile(r"^[ \t]{0,3}\S", re.MULTILINE)
PERMISSION_GRANT_TIMEOUT = max(3.0, float(os.getenv("ADB_PERMISSION_TIMEOUT", "10")))
# am start -W waits for cold start and needs a longer timeout.
ADB_START_TIMEOUT = max(5.0, float(os.getenv("ADB_START_TIMEOUT", "30")))
ADB_POWER_TIMEOUT = max(1.0, float(os.getenv("ADB_POWER_TIMEOUT", "5")))
ADB_FOREGROUND_TIMEOUT = max(1.0, float(os.getenv("ADB_FOREGROUND_TIMEOUT", "10")))
# Prefer uncompressed screencap; set to 0 to force PNG capture.
ADB_SCREENSHOT_RAW = os.getenv("ADB_SCREENSHOT_RAW", "1").strip().lower() not in {"0", "false", "no", "off"}
# These output markers indicate launch failure even when am exits with code zero.
AM_START_FAILURE_MARKERS = ("Error", "does not exist", "Exception")

# Cache launcher resolution per serial/package and invalidate it after a failed launch.
_LAUNCHER_ACTIVITY_CACHE: dict[tuple[str, str], str] = {}


def _section(dump_output: str, header_pattern: re.Pattern[str]) -> str:
    """Extract a top-level dumpsys package section up to the next peer heading."""
    text = str(dump_output or "")
    match = header_pattern.search(text)
    if not match:
        return ""
    # Skip the heading remainder so its own newline is not treated as the next heading.
    remainder = text[match.end():].lstrip("\r\n")
    next_section = TOP_LEVEL_SECTION_PATTERN.search(remainder)
    return remainder[: next_section.start()] if next_section else remainder


def _dedupe(names: list[str]) -> list[str]:
    seen: list[str] = []
    for name in names:
        if name and name not in seen:
            seen.append(name)
    return seen


def parse_runtime_permissions(dump_output: str) -> list[str]:
    """Extract unique runtime permission names in their recorded order.

    This section contains dangerous runtime permissions, unlike the requested-permission list, and therefore supplies candidates for pm grant."""
    names = [
        match.group(1)
        for line in _section(dump_output, RUNTIME_PERMISSIONS_HEADER_PATTERN).splitlines()
        for match in [RUNTIME_PERMISSION_LINE_PATTERN.match(line)]
        if match
    ]
    return _dedupe(names)


def parse_requested_permissions(dump_output: str) -> list[str]:
    """Extract all requested permissions as a fallback when runtime permissions are absent.

    The device rejects non-runtime permissions without side effects; let pm grant determine which are grantable."""
    text = str(dump_output or "")
    match = REQUESTED_PERMISSIONS_HEADER_PATTERN.search(text)
    if not match:
        return []
    return _dedupe([raw.strip() for raw in match.group(1).split(",")])
WM_SIZE_PATTERN = re.compile(r"(?:Physical|Override) size:\s*(\d+)x(\d+)")
LAUNCHER_COMPONENT_PATTERN = re.compile(
    r"^\s*[A-Za-z0-9._]+/([A-Za-z0-9._$]+)\s*$",
    re.MULTILINE,
)
SURFACE_ORIENTATION_PATTERN = re.compile(r"SurfaceOrientation:\s*(\d)")
DISPLAY_ORIENTATION_PATTERN = re.compile(r"mCurrentOrientation=(\d)")

# Android PixelFormat values: 1=RGBA_8888, 2=RGBX_8888, 5=BGRA_8888.
_RAW_PIXEL_CONVERSIONS = {
    1: cv2.COLOR_RGBA2BGR,
    2: cv2.COLOR_RGBA2BGR,
    5: cv2.COLOR_BGRA2BGR,
}
# Legacy headers are 12 bytes; Android 9 adds a 4-byte colorspace field.
_RAW_HEADER_SIZES = (16, 12)
_RAW_MAX_SIDE = 16384


def expand_activity_name(package_name: str, activity: str) -> str:
    """Expand relative or component activity names to fully qualified class names."""
    value = str(activity or "").strip()
    if "/" in value:
        component_package, _, value = value.partition("/")
        package_name = component_package or package_name
    if value.startswith("."):
        return f"{package_name}{value}"
    return value


def parse_raw_screencap(data: bytes):
    """Decode a raw screencap frame to BGR, returning None for unsupported layouts.

    Validate the little-endian width, height, format, optional colorspace header and exact pixel length. Accept 12- or 16-byte headers; malformed frames fall back to PNG."""
    if not data or len(data) < 12:
        return None
    width, height, pixel_format = struct.unpack_from("<III", data, 0)
    if not (0 < width <= _RAW_MAX_SIDE and 0 < height <= _RAW_MAX_SIDE):
        return None
    conversion = _RAW_PIXEL_CONVERSIONS.get(pixel_format)
    if conversion is None:
        return None
    pixel_bytes = width * height * 4
    header_size = next((size for size in _RAW_HEADER_SIZES if len(data) == size + pixel_bytes), None)
    if header_size is None:
        return None
    pixels = np.frombuffer(data, dtype=np.uint8, count=pixel_bytes, offset=header_size)
    return cv2.cvtColor(pixels.reshape(height, width, 4), conversion)


def orient_size(size: tuple[int, int], rotation: int | None) -> tuple[int, int]:
    """Rotate natural wm size dimensions by swapping width/height at 90 or 270 degrees."""
    width, height = size
    if rotation in (1, 3):
        return height, width
    return width, height


class U2(Automator):
    def __init__(self, device):
        super().__init__(device)
        self._serial = device.serial
        self._driver = uiautomator2.connect(self._serial)
        self._display_size_cache: tuple[int, int] | None = None
        self._last_frame_size: tuple[int, int] | None = None
        logger.debug("uiautomator2 connected: {}", self._serial)

    # Device shell transport.
    def _adb_device(self):
        """Reuse the adbutils connection, or return None to use an adb subprocess."""
        driver = getattr(self, "_driver", None)
        if driver is None:
            return None
        try:
            return getattr(driver, "adb_device", None)
        except Exception:  # noqa: BLE001 - Fall back to the subprocess transport if the connection object is unavailable.
            return None

    def _adb_shell(self, args, timeout: float) -> tuple[int, str, str]:
        """Run a device shell command and return (exit_code, stdout, stderr).

        Reuse the adbutils connection and quote arguments individually. Normalize transport failures and timeouts to RuntimeError."""
        args = [str(arg) for arg in args]
        device = self._adb_device()
        if device is not None and hasattr(device, "shell2"):
            try:
                result = device.shell2(args, timeout=timeout)
            except Exception as exc:  # noqa: BLE001 - adbutils transport and timeout exception types vary.
                raise RuntimeError(f"adb shell failed: {exc}") from exc
            return int(result.returncode), str(result.output or ""), ""
        command = ["adb", "-s", self._serial, "shell", shlex.join(args)]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError(f"adb shell failed: {exc}") from exc
        return int(result.returncode), str(result.stdout or ""), str(result.stderr or "")

    # Application lifecycle.
    def start_app(self, package_name: str):
        package_name = validate_package_name(package_name)
        logger.info("u2 launch application: {}", package_name)
        activity = self._preferred_launcher_activity(package_name)
        if activity:
            logger.debug("u2 launcher activity: {}", activity)
            started, output = self._am_start(package_name, activity)
            if started:
                return output
            # Check am output explicitly; invalidate stale activity resolution before falling back to the default monkey launch.
            _LAUNCHER_ACTIVITY_CACHE.pop((self._serial, package_name), None)
            logger.warning("u2 activity launch failed; cleared cache and retrying default launch: {}", output.strip()[:300])
        return self._driver.app_start(package_name)

    def _am_start(self, package_name: str, activity: str) -> tuple[bool, str]:
        component = f"{package_name}/{activity}"
        command = [
            "am", "start", "-W",
            "-a", "android.intent.action.MAIN",
            "-c", "android.intent.category.LAUNCHER",
            "-n", component,
        ]
        try:
            exit_code, stdout, stderr = self._adb_shell(command, ADB_START_TIMEOUT)
        except RuntimeError as exc:
            return False, str(exc)
        output = f"{stdout}\n{stderr}".strip()
        failed = exit_code != 0 or any(marker in output for marker in AM_START_FAILURE_MARKERS)
        return not failed, output

    def _preferred_launcher_activity(self, package_name: str) -> str:
        cache_key = (self._serial, package_name)
        cached = _LAUNCHER_ACTIVITY_CACHE.get(cache_key)
        if cached:
            return cached
        try:
            result = subprocess.run(
                [
                    "adb", "-s", self._serial, "shell", "cmd", "package",
                    "query-activities", "--brief", "-a", "android.intent.action.MAIN",
                    "-c", "android.intent.category.LAUNCHER", package_name,
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.warning("u2 launcher lookup failed; using default launch: {}", exc)
            return ""

        # Expand relative activity names before namespace scoring to avoid preferring unrelated library components.
        activities = [
            expand_activity_name(package_name, activity)
            for activity in LAUNCHER_COMPONENT_PATTERN.findall(result.stdout or "")
        ]
        if not activities:
            return ""

        package_parts = package_name.split(".")

        def namespace_score(activity: str) -> int:
            activity_parts = activity.split(".")
            return next(
                (index for index, pair in enumerate(zip(package_parts, activity_parts)) if pair[0] != pair[1]),
                min(len(package_parts), len(activity_parts)),
            )

        resolved = max(activities, key=namespace_score)
        _LAUNCHER_ACTIVITY_CACHE[cache_key] = resolved
        return resolved

    def stop_app(self, package_name: str):
        package_name = validate_package_name(package_name)
        logger.info("u2 stop application: {}", package_name)
        return self._driver.app_stop(package_name)

    def _package_dump(self, package_name: str) -> str:
        command = ["adb", "-s", self._serial, "shell", "dumpsys", "package", package_name]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=PERMISSION_GRANT_TIMEOUT,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.warning("Could not read application permissions: {}", exc)
            return ""
        return result.stdout or ""

    def grant_app_permissions(self, package_name: str) -> dict:
        """Grant requested runtime permissions that are not yet granted.

        Granting before functionality testing avoids consuming the scenario budget with permission dialogs. Return granted/missing/failed lists and tolerate individual pm grant failures."""
        dump_output = self._package_dump(package_name)
        if not dump_output.strip():
            return {"granted": [], "missing": [], "failed": [], "unavailable": True}
        runtime_permissions = parse_runtime_permissions(dump_output)
        candidates = runtime_permissions or parse_requested_permissions(dump_output)
        granted: list[str] = []
        missing: list[str] = []
        failed: list[str] = []
        for permission in candidates:
            command = ["adb", "-s", self._serial, "shell", "pm", "grant", package_name, permission]
            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=PERMISSION_GRANT_TIMEOUT,
                    check=False,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                logger.warning("Could not grant permission {}: {}", permission, exc)
                failed.append(permission)
                continue
            if result.returncode == 0:
                granted.append(permission)
            elif "not a changeable permission" in (result.stdout or "") + (result.stderr or ""):
                # Normal and system permissions may reject pm grant; this is expected.
                missing.append(permission)
            else:
                failed.append(permission)
        return {
            "granted": granted,
            "missing": missing,
            "failed": failed,
            "unavailable": False,
        }

    # Inject taps, long presses, and Back through shell input; some vendors discard AccessibilityService-injected events.
    def _shell_input(self, *args) -> str:
        try:
            exit_code, stdout, stderr = self._adb_shell(["input", *args], ADB_INPUT_TIMEOUT)
        except RuntimeError as exc:
            raise RuntimeError(f"shell input injection failed: {exc}") from exc
        if exit_code != 0:
            raise RuntimeError(
                f"shell input injection failed (exit={exit_code}): "
                f"{(stderr or stdout or '').strip()}"
            )
        return stdout

    def click(self, x, y):
        # Use screenshot coordinates exactly; do not alter them using the widget hierarchy.
        px, py = self._normalize_point(x, y)
        self._shell_input("tap", int(px), int(py))
        return {"x": int(px), "y": int(py)}

    def long_click(self, x, y):
        px, py = self._normalize_point(x, y)
        # A swipe with identical endpoints implements long press; duration is in milliseconds.
        self._shell_input("swipe", int(px), int(py), int(px), int(py), LONG_PRESS_DURATION_MS)
        return {"x": int(px), "y": int(py)}

    def swipe(self, x1, y1, x2, y2, duration=0.5):
        # Use the same shell injection transport for scroll/swipe and taps to avoid vendor accessibility restrictions.
        sx, sy = self._normalize_point(x1, y1)
        ex, ey = self._normalize_point(x2, y2)
        duration_ms = max(MIN_SWIPE_DURATION_MS, int(round(float(duration) * 1000)))
        return self._shell_input("swipe", int(sx), int(sy), int(ex), int(ey), duration_ms)

    def input(self, text, x=None, y=None, clear: bool = True):
        if x is not None and y is not None:
            logger.debug("u2 coordinate input: ({}, {})", x, y)
            self.click(x, y)
            self._wait_for_input_focus()
        logger.debug("u2 focused input(clear={}): {}", clear, text)
        return self._driver.send_keys(text, clear=bool(clear))

    def back(self):
        return self._shell_input("keyevent", "KEYCODE_BACK")

    # Screen power.
    def keep_awake(self) -> None:
        # Keep the screen awake while USB is connected to avoid black frames in long runs.
        try:
            exit_code, stdout, stderr = self._adb_shell(["svc", "power", "stayon", "usb"], ADB_POWER_TIMEOUT)
        except RuntimeError as exc:
            logger.warning("Could not keep screen awake; continuing: {}", exc)
            return
        if exit_code != 0:
            logger.warning("Could not keep screen awake; continuing: exit={}, {}", exit_code, (stderr or stdout).strip())

    def wake_screen(self) -> None:
        try:
            self._adb_shell(["input", "keyevent", "KEYCODE_WAKEUP"], ADB_POWER_TIMEOUT)
        except RuntimeError as exc:
            logger.warning("Could not wake screen; continuing: {}", exc)

    # Foreground application.
    def current_app(self) -> tuple[str, str]:
        """Query the foreground package and activity together.

        Call the unwrapped adbutils implementation once, avoiding repeated internal retries, then fall back once to dumpsys activity top."""
        try:
            package, activity = self._query_app_current()
            if package:
                return package, activity
        except Exception as exc:  # noqa: BLE001 - uiautomator2 and adbutils exception types vary.
            logger.warning("u2 foreground query failed; trying dumpsys activity top: {}", exc)
        try:
            exit_code, stdout, _stderr = self._adb_shell(["dumpsys", "activity", "top"], ADB_FOREGROUND_TIMEOUT)
            if exit_code == 0:
                return parse_activity_top(stdout)
        except RuntimeError as exc:
            logger.warning("u2 dumpsys activity top foreground query failed: {}", exc)
        return "", ""

    def _query_app_current(self) -> tuple[str, str]:
        device = self._adb_device()
        decorated = getattr(type(device), "app_current", None) if device is not None else None
        single_attempt = getattr(decorated, "__wrapped__", None)
        if callable(single_attempt):
            info = single_attempt(device)
            return str(getattr(info, "package", "") or ""), str(getattr(info, "activity", "") or "")
        current = self._driver.app_current()
        if isinstance(current, dict):
            return str(current.get("package", "") or ""), str(current.get("activity", "") or "")
        return "", ""

    def current_activity(self):
        return self.current_app()[1]

    def current_package(self):
        return self.current_app()[0]

    # Screenshots and display dimensions.
    def screenshot(self, path=""):
        image = self._adb_screenshot()
        if image is None:
            image = self._driver.screenshot(format="opencv")
        self._record_frame_size(image)
        if path:
            logger.debug("u2 save screenshot: {}", path)
            cv2.imwrite(path, image)
        return image

    def _resolve_display_size(self):
        u2_error = None
        try:
            # uiautomator displayWidth/displayHeight already reflect current orientation.
            info = self._driver.info
            width = int(info["displayWidth"])
            height = int(info["displayHeight"])
            if width > 0 and height > 0:
                return width, height
            u2_error = ValueError(f"invalid uiautomator2 display size: {width}x{height}")
        except Exception as exc:
            u2_error = exc
            logger.warning("u2 display-size query failed; trying adb wm size: {}", exc)

        wm_size = self._adb_wm_size()
        if wm_size is not None:
            return orient_size(wm_size, self._adb_rotation())

        logger.warning("adb wm size failed; trying screenshot dimensions")
        image = self._adb_screenshot()
        if image is not None and len(image.shape) >= 2:
            self._record_frame_size(image)
            height, width = image.shape[:2]
            if width > 0 and height > 0:
                return int(width), int(height)

        raise RuntimeError("Cannot resolve display dimensions: uiautomator2, adb wm size, and screenshot fallbacks all failed") from u2_error

    def _adb_rotation(self) -> int | None:
        for command, pattern in (
            (["dumpsys", "input"], SURFACE_ORIENTATION_PATTERN),
            (["dumpsys", "display"], DISPLAY_ORIENTATION_PATTERN),
        ):
            try:
                exit_code, stdout, _stderr = self._adb_shell(command, ADB_WM_SIZE_TIMEOUT)
            except RuntimeError as exc:
                logger.debug("Could not read display orientation: {}", exc)
                continue
            match = pattern.search(stdout) if exit_code == 0 else None
            if match:
                return int(match.group(1))
        return None

    def _adb_screenshot(self):
        if ADB_SCREENSHOT_RAW:
            image = self._raw_screenshot()
            if image is not None:
                return image
        return self._png_screenshot()

    def _raw_screenshot(self):
        """Transfer an uncompressed screencap frame through adbutils and convert it locally to BGR."""
        device = self._adb_device()
        try:
            if device is not None and hasattr(device, "open_shell"):
                data = self._read_shell_stream(device, ["screencap"], ADB_SCREENSHOT_TIMEOUT)
            else:
                data = subprocess.run(
                    ["adb", "-s", self._serial, "exec-out", "screencap"],
                    capture_output=True,
                    timeout=ADB_SCREENSHOT_TIMEOUT,
                    check=True,
                ).stdout
        except Exception as exc:  # noqa: BLE001 - Fall back to PNG after any raw-capture failure.
            logger.debug("Raw screenshot failed; falling back to PNG: serial={}, error={}", self._serial, exc)
            return None
        image = parse_raw_screencap(bytes(data or b""))
        if image is None:
            logger.debug("Unsupported raw screenshot format; falling back to PNG: serial={}, bytes={}", self._serial, len(data or b""))
        return image

    @staticmethod
    def _read_shell_stream(device, args, timeout: float) -> bytes:
        """Read complete binary shell output with an overall timeout.

        Accumulate 1 MB blocks in a bytearray to avoid quadratic concatenation of large raw frames."""
        deadline = time.monotonic() + timeout
        connection = device.open_shell(shlex.join(str(arg) for arg in args))
        try:
            sock = connection.conn
            buffer = bytearray()
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Timed out reading shell output ({timeout:g}s)")
                sock.settimeout(remaining)
                chunk = sock.recv(1 << 20)
                if not chunk:
                    return bytes(buffer)
                buffer += chunk
        finally:
            connection.close()

    def _png_screenshot(self):
        cmd = ["adb", "-s", self._serial, "exec-out", "screencap", "-p"]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=ADB_SCREENSHOT_TIMEOUT,
                check=True,
            )
        except subprocess.TimeoutExpired:
            logger.warning("adb screenshot timed out: serial={}, timeout={}s", self._serial, ADB_SCREENSHOT_TIMEOUT)
            return None
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode(errors="ignore").strip() if exc.stderr else ""
            logger.warning("adb screenshot failed: serial={}, stderr={}", self._serial, stderr or "unknown")
            return None
        except Exception as exc:
            logger.warning("adb screenshot error: serial={}, error={}", self._serial, exc)
            return None

        raw = result.stdout or b""
        if not raw:
            logger.warning("adb screenshot was empty: serial={}", self._serial)
            return None

        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            logger.warning("adb screenshot decoding failed: serial={}", self._serial)
        return image

    def _adb_wm_size(self):
        """Return natural physical display dimensions; the caller applies orientation."""
        cmd = ["adb", "-s", self._serial, "shell", "wm", "size"]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=ADB_WM_SIZE_TIMEOUT,
                check=True,
            )
        except subprocess.TimeoutExpired:
            logger.warning("adb wm size timed out: serial={}, timeout={}s", self._serial, ADB_WM_SIZE_TIMEOUT)
            return None
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            logger.warning("adb wm size failed: serial={}, stderr={}", self._serial, stderr or "unknown")
            return None
        except Exception as exc:
            logger.warning("adb wm size error: serial={}, error={}", self._serial, exc)
            return None

        matches = WM_SIZE_PATTERN.findall(result.stdout or "")
        if not matches:
            logger.warning("Cannot parse adb wm size output: serial={}, output={}", self._serial, (result.stdout or "").strip())
            return None
        width, height = matches[-1]
        return int(width), int(height)
