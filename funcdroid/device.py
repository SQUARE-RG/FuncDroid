import os
import time

from automator import H2, U2, SwipeDirection
from connector import ADB, HDC
from loguru import logger


# 前台应用查询的短 TTL 缓存（秒）：同一步内 current_package()/current_activity()
# 会被调用多次，每次都是一轮 dumpsys；任何动作方法都会让缓存立即失效。
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
        logger.info("初始化驱动: serial={}, backend={}", self.serial, self.operating_system)
        try:
            connector_cls, automator_cls = self.BACKENDS[self.operating_system]
        except KeyError as exc:
            logger.error("不支持的系统类型: {}", self.operating_system)
            raise ValueError(f"不支持的系统类型: {self.operating_system}") from exc
        self.connector = connector_cls(self)
        self.automator = automator_cls(self)
        self._foreground_cache = None
        self.keep_awake()

    # ---- 前台应用缓存 ---------------------------------------------------
    def _invalidate_foreground(self) -> None:
        self._foreground_cache = None

    def current_app(self) -> tuple[str, str]:
        """前台 (package, activity)，带 FOREGROUND_CACHE_TTL 秒的短缓存。"""
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
        except Exception as exc:  # noqa: BLE001 - 查询失败按“未知前台”处理
            logger.warning("获取前台应用失败: {}", exc)
            package, activity = "", ""
        value = (str(package or ""), str(activity or ""))
        # 失败结果不缓存，下一次调用立即重试。
        self._foreground_cache = (now, value) if value[0] else None
        return value

    def current_activity(self):
        return self.current_app()[1]

    def current_package(self):
        return self.current_app()[0]

    # ---- 动作（全部使前台缓存失效） -------------------------------------
    def _act(self, action, *args, **kwargs):
        """执行一个会改变界面的动作；无论成败都让前台缓存失效。"""
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
            raise ValueError("当点击目标为坐标时，必须提供 y")
        return self._act(self.automator.click, x, y)

    def long_click(self, x, y=None):
        if y is None:
            raise ValueError("当长按目标为坐标时，必须提供 y")
        return self._act(self.automator.long_click, x, y)

    def swipe(self, x1, y1, x2, y2, duration=0.5):
        return self._act(self.automator.swipe, x1, y1, x2, y2, duration)

    def swipe_ext(self, direction: SwipeDirection | str, scale=0.4):
        return self._act(self.automator.swipe_ext, direction, scale)

    def input(self, value, arg2=None, arg3=None, clear: bool = True):
        # 支持 input(text) 和 input(x, y, text)；clear=True 表示先清空再输入。
        if isinstance(value, (int, float)) and isinstance(arg2, (int, float)):
            if arg3 is None:
                raise ValueError("当输入目标为坐标时，必须提供 text")
            return self._act(self.automator.input, arg3, x=value, y=arg2, clear=clear)
        if value is None:
            raise ValueError("必须提供 text")
        return self._act(self.automator.input, value, clear=clear)

    def back(self):
        return self._act(self.automator.back)

    def wake_screen(self):
        """尽力点亮屏幕（Android 发送 KEYCODE_WAKEUP，HarmonyOS 调用 power-shell wakeup）。"""
        wake = getattr(self.automator, "wake_screen", None)
        if not callable(wake):
            return
        try:
            self._act(wake)
        except Exception as exc:  # noqa: BLE001 - 尽力而为
            logger.warning("唤醒屏幕失败（忽略继续）: {}", exc)

    def keep_awake(self):
        """初始化时尽力让屏幕保持常亮，失败只记录日志。"""
        try:
            keep = getattr(self.automator, "keep_awake", None)
            if callable(keep):
                keep()
        except Exception as exc:  # noqa: BLE001 - 尽力而为
            logger.warning("设置屏幕常亮失败（忽略继续）: {}", exc)

    def screenshot(self, path=""):
        return self.automator.screenshot(path)

    def display_size(self):
        return self.automator.display_size()

    @classmethod
    def _resolve_operating_system(cls, device_serial: str, operating_system: str | None):
        if operating_system:
            logger.debug("为 {} 使用显式指定的后端 {}", device_serial, operating_system)
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
            "解析 {} 的后端: in_adb={}, in_hdc={}",
            device_serial,
            in_adb,
            in_hdc,
        )

        if in_adb and in_hdc:
            logger.error("设备 {} 同时存在于 adb 和 hdc 设备列表中", device_serial)
            raise ValueError(f"device_serial 同时存在于 adb 和 hdc 列表中: {device_serial}")
        if in_adb:
            return "adb"
        if in_hdc:
            return "hdc"
        logger.error("无法根据 device_serial={} 推断后端类型", device_serial)
        raise ValueError(f"无法根据 device_serial 推断系统类型: {device_serial}")
