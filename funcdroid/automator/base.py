from abc import ABC, abstractmethod
from enum import Enum


class SwipeDirection(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"


class Automator(ABC):
    def __init__(self, device):
        self.device = device

    @abstractmethod
    def start_app(self, package_name: str):
        pass

    @abstractmethod
    def stop_app(self, package_name: str):
        pass

    def restart_app(self, package_name: str):
        self.stop_app(package_name)
        self.start_app(package_name)

    @abstractmethod
    def click(self, x, y):
        pass

    @abstractmethod
    def long_click(self, x, y):
        pass

    @abstractmethod
    def swipe(self, x1, y1, x2, y2, duration=0.5):
        pass

    def swipe_ext(self, direction, scale=0.4):
        direction = SwipeDirection(direction)
        if direction == SwipeDirection.LEFT:
            return self.swipe(0.5, 0.5, 0.5 - scale, 0.5)
        if direction == SwipeDirection.RIGHT:
            return self.swipe(0.5, 0.5, 0.5 + scale, 0.5)
        if direction == SwipeDirection.UP:
            return self.swipe(0.5, 0.5, 0.5, 0.5 - scale)
        return self.swipe(0.5, 0.5, 0.5, 0.5 + scale)

    @abstractmethod
    def input(self, text, x=None, y=None):
        pass

    @abstractmethod
    def current_activity(self):
        pass

    def current_package(self):
        return ""

    @abstractmethod
    def screenshot(self, path=""):
        pass

    @abstractmethod
    def display_size(self):
        pass

    @abstractmethod
    def back(self):
        pass
