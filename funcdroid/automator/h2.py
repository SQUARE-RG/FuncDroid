import json
import logging
import os
import re
import shutil
import tempfile
import uuid

import cv2
from hmdriver2.driver import Driver
from loguru import logger

from .base import Automator, validate_package_name

h2_logger = logging.getLogger("hmdriver2")
h2_logger.setLevel(logging.CRITICAL)

# HarmonyOS key codes: Back, Backspace, left Ctrl, A, and End.
KEYCODE_BACK = 2
KEYCODE_DEL = 2055
KEYCODE_CTRL_LEFT = 2072
KEYCODE_A = 2017
KEYCODE_MOVE_END = 2082

HDC_SCREENSHOT_TIMEOUT = max(1.0, float(os.getenv("HDC_SCREENSHOT_TIMEOUT", "15")))
HDC_APP_TIMEOUT = max(3.0, float(os.getenv("HDC_APP_TIMEOUT", "30")))
HDC_SHORT_TIMEOUT = max(1.0, float(os.getenv("HDC_SHORT_TIMEOUT", "10")))
# Optional extra backspaces for controls that ignore Ctrl+A; each requires an HDC command and is disabled by default.
H2_CLEAR_BACKSPACE_COUNT = max(0, int(os.getenv("H2_CLEAR_BACKSPACE_COUNT", "0")))
HDC_FAILURE_MARKERS = ("error:", "[fail]")

ABILITY_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")


def parse_foreground_mission(output: str) -> tuple[str, str]:
    """Parse aa dump -l into the foreground (bundle, ability), matching hmdriver2."""
    for block in re.findall(r"Mission ID #[\s\S]*?isKeepAlive: \w+\s*}", str(output or "")):
        if "state #FOREGROUND" not in block:
            continue
        bundle = re.search(r"bundle name \[(.*?)\]", block)
        ability = re.search(r"main name \[(.*?)\]", block)
        if bundle and ability:
            return bundle.group(1).strip(), ability.group(1).strip()
    return "", ""


def select_main_ability(bm_dump_output: str) -> str:
    """Select the launcher ability, then module main ability, then main module, matching hmdriver2."""
    text = str(bm_dump_output or "")
    start, end = text.find("{"), text.rfind("}") + 1
    if start < 0 or end <= start:
        return ""
    info = json.loads(text[start:end])
    main_entry = info.get("mainEntry")
    candidates = []
    for module in info.get("hapModuleInfos") or []:
        module_main = module.get("mainAbility")
        for ability in module.get("abilityInfos") or []:
            name = str(ability.get("name") or "")
            if not name:
                continue
            is_launcher = bool(ability.get("skills"))
            score = int(name == module_main) + int(ability.get("moduleName") == main_entry)
            candidates.append((not is_launcher, -score, len(candidates), name))
    return min(candidates)[3] if candidates else ""


class H2(Automator):
    """HarmonyOS automation backend.

    Use this project's HDC connector with argument lists and bounded timeouts for device commands. Retain hmdriver2 RPC only for GUI input injection."""

    def __init__(self, device):
        super().__init__(device)
        self._serial = device.serial
        self._driver = Driver(self._serial)
        self._display_size_cache = None
        self._last_frame_size = None
        logger.debug("hmdriver2 connected: {}", self._serial)

    # Internal helpers.
    def _connector(self):
        connector = getattr(getattr(self, "device", None), "connector", None)
        if connector is None:
            raise RuntimeError("h2 requires an HDC connector for device commands")
        return connector

    def _hdc_shell(self, args, timeout: float) -> str:
        output = str(self._connector().shell([str(arg) for arg in args], timeout=timeout) or "")
        lowered = output.lower()
        if any(marker in lowered for marker in HDC_FAILURE_MARKERS):
            raise RuntimeError(f"hdc shell failed: {' '.join(map(str, args))}: {output.strip()[:300]}")
        return output

    def _rpc(self, name: str, *args):
        """Invoke hmdriver2 GUI methods without the fixed 0.6-second delay decorator.

        The workflow already controls action delays. Fall back to the wrapped method if __wrapped__ is unavailable."""
        method = getattr(type(self._driver), name, None)
        raw = getattr(method, "__wrapped__", None)
        if callable(raw):
            return raw(self._driver, *args)
        return getattr(self._driver, name)(*args)

    def _press_key(self, key_code: int) -> None:
        self._hdc_shell(["uitest", "uiInput", "keyEvent", int(key_code)], HDC_SHORT_TIMEOUT)

    # Application lifecycle.
    def start_app(self, package_name: str):
        package_name = validate_package_name(package_name)
        logger.info("h2 launch application: {}", package_name)
        ability = self._main_ability(package_name)
        return self._hdc_shell(["aa", "start", "-a", ability, "-b", package_name], HDC_APP_TIMEOUT)

    def _main_ability(self, package_name: str) -> str:
        try:
            output = str(self._connector().shell(["bm", "dump", "-n", package_name], timeout=HDC_APP_TIMEOUT) or "")
            ability = select_main_ability(output)
        except Exception as exc:  # noqa: BLE001 - Use the hmdriver2 default ability if parsing fails.
            logger.warning("h2 main ability lookup failed; using MainAbility: {}", exc)
            ability = ""
        if not ABILITY_NAME_PATTERN.fullmatch(ability):
            ability = "MainAbility"
        return ability

    def stop_app(self, package_name: str):
        package_name = validate_package_name(package_name)
        logger.info("h2 stop application: {}", package_name)
        return self._hdc_shell(["aa", "force-stop", package_name], HDC_APP_TIMEOUT)

    # Input injection through hmdriver2 RPC.
    def click(self, x, y):
        px, py = self._normalize_point(x, y)
        self._rpc("click", int(px), int(py))
        return {"x": int(px), "y": int(py)}

    def long_click(self, x, y):
        px, py = self._normalize_point(x, y)
        self._rpc("long_click", int(px), int(py))
        return {"x": int(px), "y": int(py)}

    def swipe(self, x1, y1, x2, y2, duration=0.5):
        sx, sy = self._normalize_point(x1, y1)
        ex, ey = self._normalize_point(x2, y2)
        speed = self._duration_to_speed(sx, sy, ex, ey, duration)
        return self._rpc("swipe", int(sx), int(sy), int(ex), int(ey), speed)

    def input(self, text, x=None, y=None, clear: bool = True):
        if x is not None and y is not None:
            logger.debug("h2 coordinate input: ({}, {})", x, y)
            self.click(x, y)
            self._wait_for_input_focus()
        if clear:
            self._clear_focused_text()
        logger.debug("h2 focused input(clear={}): {}", clear, text)
        return self._rpc("input_text", text)

    def _clear_focused_text(self) -> None:
        """Clear the focused field with the same replacement semantics as Android.

        Prefer Component.clearText; otherwise use Ctrl+A and delete, optionally followed by end-of-field backspaces."""
        try:
            focused = self._driver(focused=True)
            # Query once without the default one-second retry delay.
            if not focused.find_component(retries=1, wait_time=0):
                raise LookupError("Focused control not found")
            method = getattr(type(focused), "clear_text", None)
            raw = getattr(method, "__wrapped__", None)
            raw(focused) if callable(raw) else focused.clear_text()
            return
        except Exception as exc:  # noqa: BLE001 - Fall back to keyboard-based clearing.
            logger.debug("h2 focused clearText failed; clearing with keys: {}", exc)
        try:
            self._hdc_shell(
                ["uitest", "uiInput", "keyEvent", KEYCODE_CTRL_LEFT, KEYCODE_A], HDC_SHORT_TIMEOUT
            )
            self._press_key(KEYCODE_DEL)
            if H2_CLEAR_BACKSPACE_COUNT:
                self._press_key(KEYCODE_MOVE_END)
                for _ in range(H2_CLEAR_BACKSPACE_COUNT):
                    self._press_key(KEYCODE_DEL)
        except Exception as exc:  # noqa: BLE001 - Continue typing after clear failure; let the executor observe the result.
            logger.warning("h2 field clearing failed; continuing input: {}", exc)

    def back(self):
        return self._press_key(KEYCODE_BACK)

    # Screen power.
    def keep_awake(self) -> None:
        self.wake_screen()

    def wake_screen(self) -> None:
        try:
            self._hdc_shell(["power-shell", "wakeup"], HDC_SHORT_TIMEOUT)
        except Exception as exc:  # noqa: BLE001 - Best effort.
            logger.warning("h2 Could not wake screen; continuing: {}", exc)

    # Foreground application.
    def current_activity(self):
        _package_name, page_name = self.current_app()
        return page_name

    def current_package(self):
        package_name, _page_name = self.current_app()
        return package_name

    def current_app(self) -> tuple[str, str]:
        if getattr(getattr(self, "device", None), "connector", None) is not None:
            try:
                return parse_foreground_mission(self._hdc_shell(["aa", "dump", "-l"], HDC_SHORT_TIMEOUT))
            except Exception as exc:  # noqa: BLE001
                logger.warning("h2 foreground query failed: {}", exc)
                return "", ""
        return self._current_app_from_driver()

    def _current_app(self):
        return self.current_app()

    def _current_app_from_driver(self):
        try:
            current = self._driver.current_app()
        except Exception as exc:
            logger.warning("h2 foreground query failed: {}", exc)
            return "", ""
        if isinstance(current, (tuple, list)) and len(current) >= 2:
            return str(current[0] or "").strip(), str(current[1] or "").strip()
        if isinstance(current, dict):
            package_name = current.get("package_name") or current.get("bundle_name") or current.get("package")
            page_name = current.get("page_name") or current.get("ability_name") or current.get("ability")
            return str(package_name or "").strip(), str(page_name or "").strip()
        logger.warning("Unrecognized h2 foreground response: {!r}", current)
        return "", ""

    # Screenshots and display dimensions.
    def screenshot(self, path=""):
        remote_path = f"/data/local/tmp/_funcdroid_{uuid.uuid4().hex}.jpeg"
        fd, local_path = tempfile.mkstemp(prefix="_funcdroid_h2_", suffix=".jpeg")
        os.close(fd)
        connector = self._connector()
        try:
            self._hdc_shell(["snapshot_display", "-f", remote_path], HDC_SCREENSHOT_TIMEOUT)
            output = str(connector.run_cmd(
                ["file", "recv", remote_path, local_path], timeout=HDC_SCREENSHOT_TIMEOUT
            ) or "")
            if any(marker in output.lower() for marker in HDC_FAILURE_MARKERS):
                raise RuntimeError(f"h2 screenshot transfer failed: {output.strip()[:300]}")
            image = cv2.imread(local_path)
            if image is None:
                raise RuntimeError(f"Cannot decode h2 screenshot file: {local_path}")
            self._record_frame_size(image)
            if path:
                logger.debug("h2 save screenshot: {}", path)
                shutil.copyfile(local_path, path)
            return image
        finally:
            try:
                self._hdc_shell(["rm", "-f", remote_path], HDC_SHORT_TIMEOUT)
            except Exception as exc:  # noqa: BLE001 - Temporary-file cleanup failure must not discard a captured screenshot.
                logger.debug("Could not remove temporary h2 screenshot from device: {}", exc)
            if os.path.isfile(local_path):
                os.remove(local_path)

    def _resolve_display_size(self):
        # Invalidate cached display_size before reading it so rotation is reflected.
        invalidate = getattr(self._driver, "_invalidate_cache", None)
        if callable(invalidate):
            invalidate("display_size")
        width, height = self._driver.display_size
        return int(width), int(height)

    def _duration_to_speed(self, x1, y1, x2, y2, duration):
        if duration is None or duration <= 0:
            return 2000

        width, height = self.display_size()

        def absolute(value, extent):
            # Only floats in [0, 1] are relative coordinates; integer 0/1 remains pixels.
            return value * extent if isinstance(value, float) and 0 <= value <= 1 else value

        start_x, start_y = absolute(x1, width), absolute(y1, height)
        end_x, end_y = absolute(x2, width), absolute(y2, height)
        distance = ((end_x - start_x) ** 2 + (end_y - start_y) ** 2) ** 0.5

        speed = int(distance / duration)
        return min(max(speed, 200), 40000)
