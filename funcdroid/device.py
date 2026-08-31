from automator import H2, U2, SwipeDirection
from connector import ADB, HDC
from loguru import logger


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
        logger.info("Initializing driver: serial={}, backend={}", self.serial, self.operating_system)
        try:
            connector_cls, automator_cls = self.BACKENDS[self.operating_system]
        except KeyError as exc:
            logger.error("Unsupported OS type: {}", self.operating_system)
            raise ValueError(f"Unsupported OS type: {self.operating_system}") from exc
        self.connector = connector_cls(self)
        self.automator = automator_cls(self)

    def start_app(self, package_name: str):
        return self.automator.start_app(package_name)

    def stop_app(self, package_name: str):
        return self.automator.stop_app(package_name)

    def restart_app(self, package_name: str):
        return self.automator.restart_app(package_name)

    def click(self, x, y=None):
        if y is None:
            raise ValueError("y must be provided when the click target is coordinates")
        return self.automator.click(x, y)

    def long_click(self, x, y=None):
        if y is None:
            raise ValueError("y must be provided when the long press target is coordinates")
        return self.automator.long_click(x, y)

    def swipe(self, x1, y1, x2, y2, duration=0.5):
        return self.automator.swipe(x1, y1, x2, y2, duration)

    def swipe_ext(self, direction: SwipeDirection | str, scale=0.4):
        return self.automator.swipe_ext(direction, scale)

    def input(self, value, arg2=None, arg3=None):
        # Supports input(text) and input(x, y, text).
        if isinstance(value, (int, float)) and isinstance(arg2, (int, float)):
            if arg3 is None:
                raise ValueError("text must be provided when the input target is coordinates")
            return self.automator.input(arg3, x=value, y=arg2)
        if value is None:
            raise ValueError("text must be provided")
        return self.automator.input(value)

    def current_activity(self):
        return self.automator.current_activity()

    def current_package(self):
        current_package = getattr(self.automator, "current_package", None)
        if callable(current_package):
            try:
                package = current_package()
                if package:
                    return package
            except Exception:
                pass

        try:
            output = self.connector.shell("dumpsys activity top")
            for line in output.splitlines():
                line = line.strip()
                if "ACTIVITY " not in line:
                    continue
                activity_part = line.split("ACTIVITY ", 1)[1].split()[0]
                if "/" in activity_part:
                    return activity_part.split("/", 1)[0]
                return activity_part
        except Exception:
            pass
        return ""

    def screenshot(self, path=""):
        return self.automator.screenshot(path)

    def display_size(self):
        return self.automator.display_size()

    def back(self):
        return self.automator.back()

    @classmethod
    def _resolve_operating_system(cls, device_serial: str, operating_system: str | None):
        if operating_system:
            logger.debug("Using explicitly specified backend {} for {}", operating_system, device_serial)
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
            "Resolving backend for {}: in_adb={}, in_hdc={}",
            device_serial,
            in_adb,
            in_hdc,
        )

        if in_adb and in_hdc:
            logger.error("Device {} exists in both adb and hdc device lists", device_serial)
            raise ValueError(f"device_serial exists in both adb and hdc lists: {device_serial}")
        if in_adb:
            return "adb"
        if in_hdc:
            return "hdc"
        logger.error("Unable to infer backend type from device_serial={}", device_serial)
        raise ValueError(f"Unable to infer OS type from device_serial: {device_serial}")
