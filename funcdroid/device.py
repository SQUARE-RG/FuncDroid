import os
import time

from automator import H2, U2, SwipeDirection
from connector import ADB, HDC
from loguru import logger


# Cache foreground queries briefly to avoid repeated dumpsys calls within one step; every action invalidates the cache.
FOREGROUND_CACHE_TTL = max(0.0, float(os.getenv("FOREGROUND_CACHE_TTL", "0.3")))


class Driver:
    BACKENDS = {
        "android": (ADB, U2),
        "adb": (ADB, U2),
        "harmony": (HDC, H2),
        "hdc": (HDC, H2),
    }
    def __init__(self, device_serial: str, operating_system: str | None = None):
        self.serial = device_serial
        self.operating_system = self._resolve_operating_system(device_serial, operating_system)
        logger.info("Initialize driver: serial={}, backend={}", self.serial, self.operating_system)
        try:
            connector_cls, automator_cls = self.BACKENDS[self.operating_system]
        except KeyError as exc:
            logger.error("Unsupported operating system: {}", self.operating_system)
            raise ValueError(f"Unsupported operating system: {self.operating_system}") from exc
        self.connector = connector_cls(self)
        self.automator = automator_cls(self)
        self._foreground_cache = None
        self.keep_awake()

    # Foreground cache.
    def _invalidate_foreground(self) -> None:
        self._foreground_cache = None

    def current_app(self) -> tuple[str, str]:
        """Return foreground (package, activity), cached for FOREGROUND_CACHE_TTL seconds."""
        cached = getattr(self, "_foreground_cache", None)
        now = time.monotonic()
        if cached is not None and FOREGROUND_CACHE_TTL > 0 and now - cached[0] < FOREGROUND_CACHE_TTL:
            return cached[1]
        current_app = getattr(self.automator, "current_app", None)
        try:
            if callable(current_app):
                package, activity = current_app()
            else:
                package, activity = self.automator.current_package(), self.automator.current_activity()
        except Exception as exc:  # noqa: BLE001 - Treat query failure as unknown foreground.
            logger.warning("Could not query foreground application: {}", exc)
            package, activity = "", ""
        value = (str(package or ""), str(activity or ""))
        # Do not cache failures; allow the next call to retry.
        self._foreground_cache = (now, value) if value[0] else None
        return value

    def current_activity(self):
        return self.current_app()[1]

    def current_package(self):
        return self.current_app()[0]

    # Actions invalidate foreground cache.
    def _act(self, action, *args, **kwargs):
        """Execute a GUI-changing action and invalidate foreground state even on failure."""
        try:
            return action(*args, **kwargs)
        finally:
            self._invalidate_foreground()

    def start_app(self, package_name: str):
        return self._act(self.automator.start_app, package_name)

    def stop_app(self, package_name: str):
        return self._act(self.automator.stop_app, package_name)

    def grant_app_permissions(self, package_name: str) -> dict:
        return self.automator.grant_app_permissions(package_name)

    def restart_app(self, package_name: str):
        return self._act(self.automator.restart_app, package_name)

    def click(self, x, y=None):
        if y is None:
            raise ValueError("Coordinate clicks require y")
        return self._act(self.automator.click, x, y)

    def long_click(self, x, y=None):
        if y is None:
            raise ValueError("Coordinate long presses require y")
        return self._act(self.automator.long_click, x, y)

    def swipe(self, x1, y1, x2, y2, duration=0.5):
        return self._act(self.automator.swipe, x1, y1, x2, y2, duration)

    def swipe_ext(self, direction: SwipeDirection | str, scale=0.4):
        return self._act(self.automator.swipe_ext, direction, scale)

    def input(self, value, arg2=None, arg3=None, clear: bool = True):
        # Support input(text) and input(x, y, text); clear=True replaces existing content.
        if isinstance(value, (int, float)) and isinstance(arg2, (int, float)):
            if arg3 is None:
                raise ValueError("Coordinate input requires text")
            return self._act(self.automator.input, arg3, x=value, y=arg2, clear=clear)
        if value is None:
            raise ValueError("text is required")
        return self._act(self.automator.input, value, clear=clear)

    def back(self):
        return self._act(self.automator.back)

    def wake_screen(self):
        """Best-effort wake using Android KEYCODE_WAKEUP or HarmonyOS power-shell wakeup."""
        wake = getattr(self.automator, "wake_screen", None)
        if not callable(wake):
            return
        try:
            self._act(wake)
        except Exception as exc:  # noqa: BLE001 - Best effort.
            logger.warning("Could not wake screen; continuing: {}", exc)

    def keep_awake(self):
        """Request keep-awake during initialization; log failures without aborting."""
        try:
            keep = getattr(self.automator, "keep_awake", None)
            if callable(keep):
                keep()
        except Exception as exc:  # noqa: BLE001 - Best effort.
            logger.warning("Could not keep screen awake; continuing: {}", exc)

    def screenshot(self, path=""):
        return self.automator.screenshot(path)

    def display_size(self):
        return self.automator.display_size()

    @classmethod
    def _resolve_operating_system(cls, device_serial: str, operating_system: str | None):
        if operating_system:
            logger.debug("Using explicitly selected backend {} for {}", operating_system, device_serial)
            return operating_system.lower()

        try:
            adb_devices = ADB.devices()
        except Exception:
            adb_devices = []

        try:
            hdc_devices = HDC.devices()
        except Exception:
            hdc_devices = []

        in_adb = device_serial in adb_devices
        in_hdc = device_serial in hdc_devices
        logger.debug(
            "Resolve backend for {}: in_adb={}, in_hdc={}",
            device_serial,
            in_adb,
            in_hdc,
        )

        if in_adb and in_hdc:
            logger.error("Device {} appears in both adb and hdc inventories", device_serial)
            raise ValueError(f"device_serial appears in both adb and hdc inventories: {device_serial}")
        if in_adb:
            return "adb"
        if in_hdc:
            return "hdc"
        logger.error("Cannot infer backend for device_serial={}", device_serial)
        raise ValueError(f"Cannot infer operating system from device_serial: {device_serial}")
