import logging
import os
import shutil
import uuid

import cv2
from hmdriver2.driver import Driver
from loguru import logger

from .base import Automator

h2_logger = logging.getLogger("hmdriver2")
h2_logger.setLevel(logging.CRITICAL)


class H2(Automator):
    def __init__(self, device):
        super().__init__(device)
        self._serial = device.serial
        self._driver = Driver(self._serial)
        logger.debug("hmdriver2 已连接: {}", self._serial)

    def start_app(self, package_name: str):
        logger.info("h2 启动应用: {}", package_name)
        return self._driver.start_app(package_name)

    def stop_app(self, package_name: str):
        logger.info("h2 停止应用: {}", package_name)
        return self._driver.stop_app(package_name)

    def click(self, x, y):
        return self._driver.click(x, y)

    def long_click(self, x, y):
        return self._driver.long_click(x, y)

    def swipe(self, x1, y1, x2, y2, duration=0.5):
        speed = self._duration_to_speed(x1, y1, x2, y2, duration)
        return self._driver.swipe(x1, y1, x2, y2, speed)

    def input(self, text, x=None, y=None):
        if x is not None and y is not None:
            logger.debug("h2 通过坐标输入: ({}, {})", x, y)
            self.click(x, y)
        logger.debug("h2 通过焦点控件输入: {}", text)
        return self._driver.input_text(text)

    def current_activity(self):
        _package_name, page_name = self._current_app()
        return page_name

    def current_package(self):
        package_name, _page_name = self._current_app()
        return package_name

    def _current_app(self):
        try:
            current = self._driver.current_app()
        except Exception as exc:
            logger.warning("h2 获取当前前台应用失败: {}", exc)
            return "", ""
        if isinstance(current, (tuple, list)) and len(current) >= 2:
            return str(current[0] or "").strip(), str(current[1] or "").strip()
        if isinstance(current, dict):
            package_name = current.get("package_name") or current.get("bundle_name") or current.get("package")
            page_name = current.get("page_name") or current.get("ability_name") or current.get("ability")
            return str(package_name or "").strip(), str(page_name or "").strip()
        logger.warning("h2 当前前台应用返回格式不可识别: {!r}", current)
        return "", ""

    def screenshot(self, path=""):
        temp_path = f"_tmp_{uuid.uuid4().hex}.jpeg"
        local_path = ""
        try:
            local_path = str(self._driver.screenshot(temp_path) or temp_path)
            image = cv2.imread(local_path)
            if image is None:
                raise RuntimeError(f"h2 截图文件无法解码: {local_path}")
            if path:
                logger.debug("h2 保存截图: {}", path)
                shutil.copyfile(local_path, path)
            return image
        finally:
            cleanup_paths = {temp_path}
            if local_path and os.path.basename(local_path) == os.path.basename(temp_path):
                cleanup_paths.add(local_path)
            for cleanup_path in cleanup_paths:
                if os.path.isfile(cleanup_path):
                    os.remove(cleanup_path)

    def display_size(self):
        return self._driver.display_size

    def back(self):
        return self._driver.go_back()

    def _duration_to_speed(self, x1, y1, x2, y2, duration):
        if duration is None or duration <= 0:
            return 2000

        width, height = self._driver.display_size
        start_x = x1 * width if 0 <= x1 <= 1 else x1
        start_y = y1 * height if 0 <= y1 <= 1 else y1
        end_x = x2 * width if 0 <= x2 <= 1 else x2
        end_y = y2 * height if 0 <= y2 <= 1 else y2
        distance = ((end_x - start_x) ** 2 + (end_y - start_y) ** 2) ** 0.5

        speed = int(distance / duration)
        return min(max(speed, 200), 40000)
