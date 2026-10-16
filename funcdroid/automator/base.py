import os
import re
import time
from abc import ABC, abstractmethod
from enum import Enum


# 坐标点击后到输入前的焦点就绪等待（秒）：输入框获得焦点、软键盘弹出需要一点时间，
# 立即输入可能落到旧焦点或被丢弃。
INPUT_FOCUS_DELAY = max(0.0, float(os.getenv("INPUT_FOCUS_DELAY", "0.3")))

# Android 包名 / HarmonyOS bundle 名：反向域名，至少两段，只允许字母数字下划线。
# 包名会被拼进设备 shell 命令，严格校验可以杜绝命令注入与误把应用标签当包名。
PACKAGE_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)+$")

ACTIVITY_LINE_PATTERN = re.compile(r"^[ \t]*ACTIVITY[ \t]+(\S+)", re.MULTILINE)


def validate_package_name(package_name) -> str:
    value = str(package_name or "").strip()
    if not PACKAGE_NAME_PATTERN.fullmatch(value):
        raise ValueError(f"非法的应用包名: {package_name!r}")
    return value


def parse_activity_top(output: str) -> tuple[str, str]:
    """解析 `dumpsys activity top`，返回前台（top resumed）Activity 的 (package, activity)。

    该命令会按“从底到顶”的顺序列出各可见任务栈的栈顶 Activity，因此最后一个
    ACTIVITY 才是最上层的那个（adbutils 的 app_current 同样取最后一个匹配；旧实现
    取第一个，在分屏/悬浮窗/桌面残留时会拿到底层任务）。每个 ACTIVITY 段内的
    “Local Activity … mResumed=true” 标记更直接地说明它处于 resumed 状态，存在时
    优先取最后一个带该标记的段。activity 保持 dumpsys 原样（可能是 .Xxx 相对名），
    与 app_current 的返回格式一致，避免同一页面在两种来源下被当成不同 Activity。
    """
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
        """授予应用运行期权限；不支持的平台返回 unavailable，由调用方跳过。"""
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
        """向焦点控件输入文本。

        clear=True（默认）表示先清空再输入（替换语义），两个平台保持一致；
        给出坐标时先点击该位置并等待焦点就绪。
        """

    @abstractmethod
    def current_activity(self):
        pass

    def current_package(self):
        return ""

    def current_app(self) -> tuple[str, str]:
        """前台应用 (package, activity/ability)；子类应以一次查询同时取得两者。"""
        return str(self.current_package() or ""), str(self.current_activity() or "")

    @abstractmethod
    def screenshot(self, path=""):
        pass

    def display_size(self):
        """当前方向下的屏幕尺寸 (width, height)。

        优先使用最近一帧截图的尺寸：截图总是反映当前方向，横竖屏切换后坐标换算
        仍然正确；尚未截图时才回退到平台查询（_resolve_display_size）。
        """
        frame_size = getattr(self, "_last_frame_size", None)
        if frame_size:
            return frame_size
        # 尚无截图时的平台查询结果只缓存到第一帧到来为止（之后始终以帧尺寸为准）。
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
        # 执行器已经把模型的图片坐标换算成整数屏幕像素，(1, 1) 也必须保留。
        # 仅为 swipe_ext 等调用保留浮点数 0–1 比例坐标的兼容入口。
        if (isinstance(x, float) or isinstance(y, float)) and 0 <= x <= 1 and 0 <= y <= 1:
            width, height = self.display_size()
            return min(width - 1, int(x * width)), min(height - 1, int(y * height))
        return x, y

    def _wait_for_input_focus(self) -> None:
        if INPUT_FOCUS_DELAY > 0:
            time.sleep(INPUT_FOCUS_DELAY)

    def keep_awake(self) -> None:
        """尽力让屏幕保持常亮；不支持的平台什么都不做。"""

    def wake_screen(self) -> None:
        """尽力点亮屏幕；不支持的平台什么都不做。"""

    @abstractmethod
    def back(self):
        pass
