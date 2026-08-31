import os
import re
import subprocess

import cv2
import numpy as np
import uiautomator2
from loguru import logger

from .base import Automator


ADB_SCREENSHOT_TIMEOUT = max(1.0, float(os.getenv("ADB_SCREENSHOT_TIMEOUT", "12")))
ADB_WM_SIZE_TIMEOUT = max(1.0, float(os.getenv("ADB_WM_SIZE_TIMEOUT", "5")))
WM_SIZE_PATTERN = re.compile(r"(?:Physical|Override) size:\s*(\d+)x(\d+)")
LAUNCHER_COMPONENT_PATTERN = re.compile(
    r"^\s*[A-Za-z0-9._]+/([A-Za-z0-9._$]+)\s*$",
    re.MULTILINE,
)


class U2(Automator):
    def __init__(self, device):
        super().__init__(device)
        self._serial = device.serial
        self._driver = uiautomator2.connect(self._serial)
        logger.debug("uiautomator2 已连接: {}", self._serial)

    def start_app(self, package_name: str):
        logger.info("u2 启动应用: {}", package_name)
        activity = self._preferred_launcher_activity(package_name)
        if activity:
            logger.debug("u2 使用启动 Activity: {}", activity)
            return self._driver.app_start(package_name, activity=activity)
        return self._driver.app_start(package_name)

    def _preferred_launcher_activity(self, package_name: str) -> str:
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
            logger.warning("u2 查询启动 Activity 失败，使用默认启动方式: {}", exc)
            return ""

        activities = LAUNCHER_COMPONENT_PATTERN.findall(result.stdout or "")
        if not activities:
            return ""

        package_parts = package_name.split(".")

        def namespace_score(activity: str) -> int:
            activity_parts = activity.lstrip(".").split(".")
            return next(
                (index for index, pair in enumerate(zip(package_parts, activity_parts)) if pair[0] != pair[1]),
                min(len(package_parts), len(activity_parts)),
            )

        return max(activities, key=namespace_score)

    def stop_app(self, package_name: str):
        logger.info("u2 停止应用: {}", package_name)
        return self._driver.app_stop(package_name)

    def click(self, x, y):
        px, py = self._normalize_point(x, y)
        return self._driver.click(px, py)

    def long_click(self, x, y):
        px, py = self._normalize_point(x, y)
        return self._driver.long_click(px, py, 1.5)

    def swipe(self, x1, y1, x2, y2, duration=0.5):
        sx, sy = self._normalize_point(x1, y1)
        ex, ey = self._normalize_point(x2, y2)
        return self._driver.swipe(sx, sy, ex, ey, duration)

    def input(self, text, x=None, y=None):
        if x is not None and y is not None:
            logger.debug("u2 通过坐标输入: ({}, {})", x, y)
            self.click(x, y)
        logger.debug("u2 通过焦点控件输入: {}", text)
        return self._driver.send_keys(text, True)

    def current_activity(self):
        try:
            current = self._driver.app_current()
            if isinstance(current, dict):
                return current.get("activity", "") or ""
        except Exception as exc:
            logger.warning("u2 获取当前 activity 失败，尝试 adb fallback: {}", exc)

        try:
            output = self.device.connector.shell("dumpsys activity top")
            for line in output.splitlines():
                line = line.strip()
                if "ACTIVITY " not in line:
                    continue
                activity_part = line.split("ACTIVITY ", 1)[1].split()[0]
                if "/" in activity_part:
                    return activity_part.split("/", 1)[1]
                return activity_part
        except Exception as exc:
            logger.warning("u2 adb fallback 获取当前 activity 失败: {}", exc)
        return ""

    def current_package(self):
        try:
            current = self._driver.app_current()
            if isinstance(current, dict):
                return current.get("package", "") or ""
        except Exception as exc:
            logger.warning("u2 获取当前 package 失败，尝试 adb fallback: {}", exc)

        try:
            output = self.device.connector.shell("dumpsys activity top")
            for line in output.splitlines():
                line = line.strip()
                if "ACTIVITY " not in line:
                    continue
                activity_part = line.split("ACTIVITY ", 1)[1].split()[0]
                if "/" in activity_part:
                    return activity_part.split("/", 1)[0]
                return activity_part
        except Exception as exc:
            logger.warning("u2 adb fallback 获取当前 package 失败: {}", exc)
        return ""

    def screenshot(self, path=""):
        image = self._adb_screenshot()
        if image is None:
            image = self._driver.screenshot(format="opencv")
        if path:
            logger.debug("u2 保存截图: {}", path)
            cv2.imwrite(path, image)
        return image

    def display_size(self):
        u2_error = None
        try:
            info = self._driver.info
            width = int(info["displayWidth"])
            height = int(info["displayHeight"])
            if width > 0 and height > 0:
                return width, height
            u2_error = ValueError(f"invalid uiautomator2 display size: {width}x{height}")
        except Exception as exc:
            u2_error = exc
            logger.warning("u2 获取屏幕尺寸失败，尝试 adb wm size fallback: {}", exc)

        wm_size = self._adb_wm_size()
        if wm_size is not None:
            return wm_size

        logger.warning("adb wm size fallback 失败，尝试 adb 截图尺寸 fallback")
        image = self._adb_screenshot()
        if image is not None and len(image.shape) >= 2:
            height, width = image.shape[:2]
            if width > 0 and height > 0:
                return int(width), int(height)

        raise RuntimeError("无法获取屏幕尺寸：uiautomator2、adb wm size 和截图尺寸 fallback 均失败") from u2_error

    def back(self):
        return self._driver.press("back")

    def _normalize_point(self, x, y):
        if 0 <= x <= 1 and 0 <= y <= 1:
            width, height = self.display_size()
            return int(x * width), int(y * height)
        return x, y

    def _adb_screenshot(self):
        cmd = ["adb", "-s", self._serial, "exec-out", "screencap", "-p"]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=ADB_SCREENSHOT_TIMEOUT,
                check=True,
            )
        except subprocess.TimeoutExpired:
            logger.warning("adb 截图超时: serial={}, timeout={}s", self._serial, ADB_SCREENSHOT_TIMEOUT)
            return None
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode(errors="ignore").strip() if exc.stderr else ""
            logger.warning("adb 截图失败: serial={}, stderr={}", self._serial, stderr or "unknown")
            return None
        except Exception as exc:
            logger.warning("adb 截图异常: serial={}, error={}", self._serial, exc)
            return None

        raw = result.stdout or b""
        if not raw:
            logger.warning("adb 截图为空: serial={}", self._serial)
            return None

        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            logger.warning("adb 截图解码失败: serial={}", self._serial)
        return image

    def _adb_wm_size(self):
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
            logger.warning("adb wm size 超时: serial={}, timeout={}s", self._serial, ADB_WM_SIZE_TIMEOUT)
            return None
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            logger.warning("adb wm size 失败: serial={}, stderr={}", self._serial, stderr or "unknown")
            return None
        except Exception as exc:
            logger.warning("adb wm size 异常: serial={}, error={}", self._serial, exc)
            return None

        matches = WM_SIZE_PATTERN.findall(result.stdout or "")
        if not matches:
            logger.warning("adb wm size 输出不可解析: serial={}, output={}", self._serial, (result.stdout or "").strip())
            return None
        width, height = matches[-1]
        return int(width), int(height)
