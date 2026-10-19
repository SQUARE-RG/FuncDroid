import os
import re
import time
from abc import ABC, abstractmethod
from enum import Enum


# Allow focus and the keyboard to settle after a coordinate tap before typing.
INPUT_FOCUS_DELAY = max(0.0, float(os.getenv("INPUT_FOCUS_DELAY", "0.3")))

# Validate reverse-domain package/bundle names before inserting them into device commands.
PACKAGE_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)+$")

ACTIVITY_LINE_PATTERN = re.compile(r"^[ \t]*ACTIVITY[ \t]+(\S+)", re.MULTILINE)


def validate_package_name(package_name) -> str:
    value = str(package_name or "").strip()
    if not PACKAGE_NAME_PATTERN.fullmatch(value):
        raise ValueError(f"Invalid application package name: {package_name!r}")
    return value


def parse_activity_top(output: str) -> tuple[str, str]:
    """Parse dumpsys activity top into the foreground (package, activity).

    Tasks appear bottom to top: prefer the last resumed ACTIVITY section, otherwise the last ACTIVITY. Preserve relative activity names to match adbutils and avoid duplicate page identities."""
    text = str(output or "")
    matches = list(ACTIVITY_LINE_PATTERN.finditer(text))
    if not matches:
        return "", ""
    chosen = matches[-1]
    for index, match in enumerate(matches):
        block_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        if "mResumed=true" in text[match.end():block_end]:
            chosen = match
    component = chosen.group(1)
    package, separator, activity = component.partition("/")
    if not separator:
        return component, ""
    return package, activity


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

    def grant_app_permissions(self, package_name: str) -> dict:
        """Grant runtime permissions; unsupported backends return unavailable."""
        return {"granted": [], "missing": [], "failed": [], "unavailable": True}

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
    def input(self, text, x=None, y=None, clear: bool = True):
        """Type into the focused control.

        clear=True replaces existing text consistently on both backends. When coordinates are supplied, tap first and wait for focus."""

    @abstractmethod
    def current_activity(self):
        pass

    def current_package(self):
        return ""

    def current_app(self) -> tuple[str, str]:
        """Return foreground (package, activity/ability), preferably in one backend query."""
        return str(self.current_package() or ""), str(self.current_activity() or "")

    @abstractmethod
    def screenshot(self, path=""):
        pass

    def display_size(self):
        """Return (width, height) in the current orientation.

        Prefer the latest screenshot dimensions so rotation cannot stale coordinate conversion; query the backend only before the first screenshot."""
        frame_size = getattr(self, "_last_frame_size", None)
        if frame_size:
            return frame_size
        # Cache backend dimensions only until the first screenshot supplies current orientation.
        cached = getattr(self, "_display_size_cache", None)
        if cached:
            return cached
        resolved = tuple(self._resolve_display_size())
        self._display_size_cache = resolved
        return resolved

    def _resolve_display_size(self):
        raise NotImplementedError

    def _record_frame_size(self, image) -> None:
        shape = getattr(image, "shape", None)
        if shape is not None and len(shape) >= 2:
            height, width = int(shape[0]), int(shape[1])
            if width > 0 and height > 0:
                self._last_frame_size = (width, height)
                self._display_size_cache = None

    def _normalize_point(self, x, y):
        # Preserve integer pixel coordinates, including (1, 1); only floating-point values in [0, 1] denote relative coordinates.
        if (isinstance(x, float) or isinstance(y, float)) and 0 <= x <= 1 and 0 <= y <= 1:
            width, height = self.display_size()
            return min(width - 1, int(x * width)), min(height - 1, int(y * height))
        return x, y

    def _wait_for_input_focus(self) -> None:
        if INPUT_FOCUS_DELAY > 0:
            time.sleep(INPUT_FOCUS_DELAY)

    def keep_awake(self) -> None:
        """Best-effort keep-awake request; unsupported backends do nothing."""

    def wake_screen(self) -> None:
        """Best-effort screen wake request; unsupported backends do nothing."""

    @abstractmethod
    def back(self):
        pass
