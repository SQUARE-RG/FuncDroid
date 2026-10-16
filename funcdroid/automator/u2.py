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
# shell input swipe 的最小时长：过短会被部分设备识别为 fling 而丢帧。
MIN_SWIPE_DURATION_MS = max(50, int(os.getenv("MIN_SWIPE_DURATION_MS", "100")))
# `dumpsys package` 的 "runtime permissions:" 段，形如 "android.permission.CAMERA: granted=false"。
# 要求名字至少含一个点，排除 enabledComponents 这类凑巧带冒号的段内子标题。
RUNTIME_PERMISSION_LINE_PATTERN = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_]*\.[A-Za-z0-9_.]+)\s*:")
# "requested permissions:" 段所在的段头行；带方括号时一次性列出全部权限名。
# dumpsys 的同一段落在不同 Android 版本/机型上缩进层级不一致，因此只按段头文本定位。
REQUESTED_PERMISSIONS_HEADER_PATTERN = re.compile(
    r"^[ \t]*requested permissions:[^\n]*\[([^\]]*)\]", re.MULTILINE
)
RUNTIME_PERMISSIONS_HEADER_PATTERN = re.compile(r"^[ \t]*runtime permissions:", re.MULTILINE)
# 同级段头（缩进不超过段头的行），用于把一个段落切到下一个段落之前。
TOP_LEVEL_SECTION_PATTERN = re.compile(r"^[ \t]{0,3}\S", re.MULTILINE)
PERMISSION_GRANT_TIMEOUT = max(3.0, float(os.getenv("ADB_PERMISSION_TIMEOUT", "10")))
# `am start -W` 会等待冷启动完成，需要比普通 shell 命令更长的超时。
ADB_START_TIMEOUT = max(5.0, float(os.getenv("ADB_START_TIMEOUT", "30")))
ADB_POWER_TIMEOUT = max(1.0, float(os.getenv("ADB_POWER_TIMEOUT", "5")))
ADB_FOREGROUND_TIMEOUT = max(1.0, float(os.getenv("ADB_FOREGROUND_TIMEOUT", "10")))
# 默认走 raw screencap（不在设备上做 PNG 编码）；设为 0 可强制回到 -p PNG。
ADB_SCREENSHOT_RAW = os.getenv("ADB_SCREENSHOT_RAW", "1").strip().lower() not in {"0", "false", "no", "off"}
# am start 输出中出现这些标记即视为启动失败（am 自身退出码通常仍为 0）。
AM_START_FAILURE_MARKERS = ("Error", "does not exist", "Exception")

# launcher Activity 对同一 (serial, package) 是稳定事实；每次 start_app 都跑
# query-activities 要多付一次 adb 子进程往返。缓存成功解析结果，启动失败时清除。
_LAUNCHER_ACTIVITY_CACHE: dict[tuple[str, str], str] = {}


def _section(dump_output: str, header_pattern: re.Pattern[str]) -> str:
    """截取 `dumpsys package` 输出中某个顶层段落（从段头到下一个顶层行）。"""
    text = str(dump_output or "")
    match = header_pattern.search(text)
    if not match:
        return ""
    # 跳过段头所在行的剩余部分，避免把紧随其后的换行本身当成下一个段头。
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
    """从 `dumpsys package <pkg>` 的 runtime permissions 段提取权限名（保持顺序去重）。

    只有 Android 的 dangerous 运行期权限会出现在该段，普通权限（INTERNET 等）
    不在其中，因此这里天然就是可授予候选集；段落切分也避免了误把
    requested permissions 的权限清单当成带状态的条目。
    """
    names = [
        match.group(1)
        for line in _section(dump_output, RUNTIME_PERMISSIONS_HEADER_PATTERN).splitlines()
        for match in [RUNTIME_PERMISSION_LINE_PATTERN.match(line)]
        if match
    ]
    return _dedupe(names)


def parse_requested_permissions(dump_output: str) -> list[str]:
    """从 `dumpsys package <pkg>` 的 requested permissions 段提取全部申请权限名。

    仅用于运行期权限段缺失时的回退：`pm grant` 对非运行期权限会直接失败且
    无副作用，因此不做 dangerous 判定，由设备侧决定哪些真正可授予。
    """
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

# screencap 原始帧格式（android PixelFormat）：1=RGBA_8888，2=RGBX_8888，5=BGRA_8888。
_RAW_PIXEL_CONVERSIONS = {
    1: cv2.COLOR_RGBA2BGR,
    2: cv2.COLOR_RGBA2BGR,
    5: cv2.COLOR_BGRA2BGR,
}
# 头部长度：旧版本为 width/height/format 共 12 字节，Android 9 起追加 4 字节 colorspace。
_RAW_HEADER_SIZES = (16, 12)
_RAW_MAX_SIDE = 16384


def expand_activity_name(package_name: str, activity: str) -> str:
    """把 query-activities 给出的相对名（.MainActivity）或组件名（pkg/.Act）展开为全限定类名。"""
    value = str(activity or "").strip()
    if "/" in value:
        component_package, _, value = value.partition("/")
        package_name = component_package or package_name
    if value.startswith("."):
        return f"{package_name}{value}"
    return value


def parse_raw_screencap(data: bytes):
    """解析 `screencap`（不带 -p）的原始帧，返回 BGR 图像；格式不符时返回 None。

    头部为小端 uint32：width、height、format[、colorspace]。不同 Android 版本头长
    12 或 16 字节，这里用“头长 + width*height*4 == 总长度”精确判定，任何不一致
    （未知像素格式、PTY 换行改写导致的长度变化等）都返回 None，由调用方回退 PNG。
    """
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
    """`wm size` 给的是自然方向（未旋转）的物理尺寸；90/270 度时交换宽高。"""
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
        logger.debug("uiautomator2 已连接: {}", self._serial)

    # ---- 设备 shell 通道 -------------------------------------------------
    def _adb_device(self):
        """uiautomator2 已持有的 adbutils 设备连接；不可用时返回 None 走 adb 子进程。"""
        driver = getattr(self, "_driver", None)
        if driver is None:
            return None
        try:
            return getattr(driver, "adb_device", None)
        except Exception:  # noqa: BLE001 - 连接对象异常时退回子进程通道
            return None

    def _adb_shell(self, args, timeout: float) -> tuple[int, str, str]:
        """执行设备 shell 命令，返回 (exit_code, stdout, stderr)。

        优先复用 uiautomator2 的 adbutils 连接（不再为每条命令拉起 adb 客户端进程），
        参数逐个 shell 转义；任何传输错误/超时统一抛 RuntimeError。
        """
        args = [str(arg) for arg in args]
        device = self._adb_device()
        if device is not None and hasattr(device, "shell2"):
            try:
                result = device.shell2(args, timeout=timeout)
            except Exception as exc:  # noqa: BLE001 - adbutils 的超时/连接异常类型不统一
                raise RuntimeError(f"adb shell 执行失败: {exc}") from exc
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
            raise RuntimeError(f"adb shell 执行失败: {exc}") from exc
        return int(result.returncode), str(result.stdout or ""), str(result.stderr or "")

    # ---- 应用生命周期 ---------------------------------------------------
    def start_app(self, package_name: str):
        package_name = validate_package_name(package_name)
        logger.info("u2 启动应用: {}", package_name)
        activity = self._preferred_launcher_activity(package_name)
        if activity:
            logger.debug("u2 使用启动 Activity: {}", activity)
            started, output = self._am_start(package_name, activity)
            if started:
                return output
            # uiautomator2 的 app_start 在 am start 失败时不会抛异常，必须自己检查输出；
            # 缓存的 Activity 可能因应用更新等原因失效，清缓存后退回默认（monkey）启动。
            _LAUNCHER_ACTIVITY_CACHE.pop((self._serial, package_name), None)
            logger.warning("u2 指定 Activity 启动失败，已清除缓存并用默认方式重试: {}", output.strip()[:300])
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
            logger.warning("u2 查询启动 Activity 失败，使用默认启动方式: {}", exc)
            return ""

        # 先把 .MainActivity 这类相对名展开成全限定名再打分：否则相对名的
        # 命名空间得分恒为 0，会误选 com.example.lib.ShortcutActivity 之类的库组件。
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
        logger.info("u2 停止应用: {}", package_name)
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
            logger.warning("读取应用权限失败: {}", exc)
            return ""
        return result.stdout or ""

    def grant_app_permissions(self, package_name: str) -> dict:
        """授予目标应用已申请且尚未授予的运行期权限。

        缺陷检测关注功能而非授权流程，逐条权限弹窗会占用场景步数预算，因此
        在启动前一次性补齐。`pm grant` 对非运行期权限会直接失败，这里按条
        容错，返回 granted/missing/failed 三类清单，绝不因单条失败中断运行。
        """
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
                logger.warning("授予权限失败 {}: {}", permission, exc)
                failed.append(permission)
                continue
            if result.returncode == 0:
                granted.append(permission)
            elif "not a changeable permission" in (result.stdout or "") + (result.stderr or ""):
                # 普通权限/系统权限无法通过 pm grant 变更，属预期情况。
                missing.append(permission)
            else:
                failed.append(permission)
        return {
            "granted": granted,
            "missing": missing,
            "failed": failed,
            "unavailable": False,
        }

    # ---- 输入注入 -------------------------------------------------------
    # 部分厂商（荣耀/华为等）设备的安全策略可能丢弃 AccessibilityService
    # 注入的点击（UiAutomation.injectInputEvent），表现为 click 无任何效果而
    # 截图与坐标均正常；同一坐标改用 ADB shell 的 input 注入则稳定生效。
    # 因此点击/长按/返回统一走 shell 的 `input` 命令（经 adbutils 连接下发，
    # 仍是设备端 shell 执行 input，注入通道不变）。
    def _shell_input(self, *args) -> str:
        try:
            exit_code, stdout, stderr = self._adb_shell(["input", *args], ADB_INPUT_TIMEOUT)
        except RuntimeError as exc:
            raise RuntimeError(f"shell input 注入失败: {exc}") from exc
        if exit_code != 0:
            raise RuntimeError(
                f"shell input 注入失败 (exit={exit_code}): "
                f"{(stderr or stdout or '').strip()}"
            )
        return stdout

    def click(self, x, y):
        # 点击完全以图片坐标为准：点到哪就是哪，不读取控件树修正坐标。
        px, py = self._normalize_point(x, y)
        self._shell_input("tap", int(px), int(py))
        return {"x": int(px), "y": int(py)}

    def long_click(self, x, y):
        px, py = self._normalize_point(x, y)
        # 同点同坐标的 swipe 即长按，duration 单位为毫秒。
        self._shell_input("swipe", int(px), int(py), int(px), int(py), LONG_PRESS_DURATION_MS)
        return {"x": int(px), "y": int(py)}

    def swipe(self, x1, y1, x2, y2, duration=0.5):
        # 与 click/long_press 统一走 shell 通道：部分厂商（荣耀/华为等）的安全
        # 策略会丢弃 AccessibilityService 注入的手势，shell input swipe 更稳定，
        # 也让 scroll/swipe 与点击的注入通道保持一致。
        sx, sy = self._normalize_point(x1, y1)
        ex, ey = self._normalize_point(x2, y2)
        duration_ms = max(MIN_SWIPE_DURATION_MS, int(round(float(duration) * 1000)))
        return self._shell_input("swipe", int(sx), int(sy), int(ex), int(ey), duration_ms)

    def input(self, text, x=None, y=None, clear: bool = True):
        if x is not None and y is not None:
            logger.debug("u2 通过坐标输入: ({}, {})", x, y)
            self.click(x, y)
            self._wait_for_input_focus()
        logger.debug("u2 通过焦点控件输入(clear={}): {}", clear, text)
        return self._driver.send_keys(text, clear=bool(clear))

    def back(self):
        return self._shell_input("keyevent", "KEYCODE_BACK")

    # ---- 屏幕电源 -------------------------------------------------------
    def keep_awake(self) -> None:
        # USB 连接期间保持常亮，避免长时间运行中途息屏导致整段黑屏截图。
        try:
            exit_code, stdout, stderr = self._adb_shell(["svc", "power", "stayon", "usb"], ADB_POWER_TIMEOUT)
        except RuntimeError as exc:
            logger.warning("设置屏幕常亮失败（忽略继续）: {}", exc)
            return
        if exit_code != 0:
            logger.warning("设置屏幕常亮失败（忽略继续）: exit={}, {}", exit_code, (stderr or stdout).strip())

    def wake_screen(self) -> None:
        try:
            self._adb_shell(["input", "keyevent", "KEYCODE_WAKEUP"], ADB_POWER_TIMEOUT)
        except RuntimeError as exc:
            logger.warning("唤醒屏幕失败（忽略继续）: {}", exc)

    # ---- 前台应用 -------------------------------------------------------
    def current_app(self) -> tuple[str, str]:
        """一次查询同时返回前台 (package, activity)。

        adbutils 的 app_current 带 @retry(tries=3)，失败时会把三条 dumpsys 各跑三遍；
        这里只调用一次未包装的实现，失败后再用 `dumpsys activity top` 兜底一次。
        """
        try:
            package, activity = self._query_app_current()
            if package:
                return package, activity
        except Exception as exc:  # noqa: BLE001 - uiautomator2/adbutils 异常类型不稳定
            logger.warning("u2 获取前台应用失败，尝试 dumpsys activity top: {}", exc)
        try:
            exit_code, stdout, _stderr = self._adb_shell(["dumpsys", "activity", "top"], ADB_FOREGROUND_TIMEOUT)
            if exit_code == 0:
                return parse_activity_top(stdout)
        except RuntimeError as exc:
            logger.warning("u2 dumpsys activity top 获取前台应用失败: {}", exc)
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

    # ---- 截图与尺寸 -----------------------------------------------------
    def screenshot(self, path=""):
        image = self._adb_screenshot()
        if image is None:
            image = self._driver.screenshot(format="opencv")
        self._record_frame_size(image)
        if path:
            logger.debug("u2 保存截图: {}", path)
            cv2.imwrite(path, image)
        return image

    def _resolve_display_size(self):
        u2_error = None
        try:
            # uiautomator 的 displayWidth/Height 已按当前方向给出。
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
            return orient_size(wm_size, self._adb_rotation())

        logger.warning("adb wm size fallback 失败，尝试 adb 截图尺寸 fallback")
        image = self._adb_screenshot()
        if image is not None and len(image.shape) >= 2:
            self._record_frame_size(image)
            height, width = image.shape[:2]
            if width > 0 and height > 0:
                return int(width), int(height)

        raise RuntimeError("无法获取屏幕尺寸：uiautomator2、adb wm size 和截图尺寸 fallback 均失败") from u2_error

    def _adb_rotation(self) -> int | None:
        for command, pattern in (
            (["dumpsys", "input"], SURFACE_ORIENTATION_PATTERN),
            (["dumpsys", "display"], DISPLAY_ORIENTATION_PATTERN),
        ):
            try:
                exit_code, stdout, _stderr = self._adb_shell(command, ADB_WM_SIZE_TIMEOUT)
            except RuntimeError as exc:
                logger.debug("读取屏幕方向失败: {}", exc)
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
        """原始帧截图：设备端不做 PNG 编码，经 adbutils 连接传输后本地直接转 BGR。"""
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
        except Exception as exc:  # noqa: BLE001 - 任何失败都回退 PNG 截图
            logger.debug("raw 截图失败，回退 PNG: serial={}, error={}", self._serial, exc)
            return None
        image = parse_raw_screencap(bytes(data or b""))
        if image is None:
            logger.debug("raw 截图格式不匹配，回退 PNG: serial={}, bytes={}", self._serial, len(data or b""))
        return image

    @staticmethod
    def _read_shell_stream(device, args, timeout: float) -> bytes:
        """经 adbutils 连接读取 shell 命令的完整二进制输出，带整体超时。

        不用 device.shell(encoding=None)：其 read_until_close 以 4KB 分块做 bytes 拼接，
        对约 10MB 的原始帧是平方级拷贝；这里用 1MB 分块写入 bytearray。
        """
        deadline = time.monotonic() + timeout
        connection = device.open_shell(shlex.join(str(arg) for arg in args))
        try:
            sock = connection.conn
            buffer = bytearray()
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"读取 shell 输出超时（{timeout:g}s）")
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
        """返回自然方向（未旋转）的物理尺寸；调用方需按当前方向交换宽高。"""
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
