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

# HarmonyOS 按键码（hmdriver2.proto.KeyCode）：返回、退格、左 Ctrl、A、光标移到末尾。
KEYCODE_BACK = 2
KEYCODE_DEL = 2055
KEYCODE_CTRL_LEFT = 2072
KEYCODE_A = 2017
KEYCODE_MOVE_END = 2082

HDC_SCREENSHOT_TIMEOUT = max(1.0, float(os.getenv("HDC_SCREENSHOT_TIMEOUT", "15")))
HDC_APP_TIMEOUT = max(3.0, float(os.getenv("HDC_APP_TIMEOUT", "30")))
HDC_SHORT_TIMEOUT = max(1.0, float(os.getenv("HDC_SHORT_TIMEOUT", "10")))
# 按键清空兜底时额外发送的退格次数（部分输入法/控件不响应 Ctrl+A）；每次是一条
# hdc 命令，默认关闭，需要时通过环境变量开启。
H2_CLEAR_BACKSPACE_COUNT = max(0, int(os.getenv("H2_CLEAR_BACKSPACE_COUNT", "0")))
HDC_FAILURE_MARKERS = ("error:", "[fail]")

ABILITY_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")


def parse_foreground_mission(output: str) -> tuple[str, str]:
    """解析 `aa dump -l`，返回前台任务的 (bundle, ability)；与 hmdriver2 的解析口径一致。"""
    for block in re.findall(r"Mission ID #[\s\S]*?isKeepAlive: \w+\s*}", str(output or "")):
        if "state #FOREGROUND" not in block:
            continue
        bundle = re.search(r"bundle name \[(.*?)\]", block)
        ability = re.search(r"main name \[(.*?)\]", block)
        if bundle and ability:
            return bundle.group(1).strip(), ability.group(1).strip()
    return "", ""


def select_main_ability(bm_dump_output: str) -> str:
    """从 `bm dump -n <bundle>` 的 JSON 中选出启动 Ability，选择规则与 hmdriver2 的
    get_app_main_ability 一致：launcher Ability 优先，其次是模块主 Ability、主模块。"""
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
    """HarmonyOS 自动化后端。

    hmdriver2 的 hdc 层用 Popen(shell=True).communicate() 执行命令，既没有超时，
    又把包名拼进 shell 字符串；因此截图、启停应用、返回键、前台查询等设备命令
    全部改走项目自己的 HDC connector（列表参数 + 硬超时），hmdriver2 只保留
    UI 注入（click/swipe/input）所需的 RPC 通道。
    """

    def __init__(self, device):
        super().__init__(device)
        self._serial = device.serial
        self._driver = Driver(self._serial)
        self._display_size_cache = None
        self._last_frame_size = None
        logger.debug("hmdriver2 已连接: {}", self._serial)

    # ---- 内部工具 -------------------------------------------------------
    def _connector(self):
        connector = getattr(getattr(self, "device", None), "connector", None)
        if connector is None:
            raise RuntimeError("h2 缺少 HDC connector，无法执行设备命令")
        return connector

    def _hdc_shell(self, args, timeout: float) -> str:
        output = str(self._connector().shell([str(arg) for arg in args], timeout=timeout) or "")
        lowered = output.lower()
        if any(marker in lowered for marker in HDC_FAILURE_MARKERS):
            raise RuntimeError(f"hdc shell 执行失败: {' '.join(map(str, args))}: {output.strip()[:300]}")
        return output

    def _rpc(self, name: str, *args):
        """调用 hmdriver2 的 UI 方法并绕过其 @delay 固定 0.6s 休眠。

        工作流自己有 ACTION_DELAY，再叠加 0.6s 只会拖慢每一步；functools.wraps 保留了
        原函数的 __wrapped__，缺失时（库版本变化）退回带延迟的调用。
        """
        method = getattr(type(self._driver), name, None)
        raw = getattr(method, "__wrapped__", None)
        if callable(raw):
            return raw(self._driver, *args)
        return getattr(self._driver, name)(*args)

    def _press_key(self, key_code: int) -> None:
        self._hdc_shell(["uitest", "uiInput", "keyEvent", int(key_code)], HDC_SHORT_TIMEOUT)

    # ---- 应用生命周期 ---------------------------------------------------
    def start_app(self, package_name: str):
        package_name = validate_package_name(package_name)
        logger.info("h2 启动应用: {}", package_name)
        ability = self._main_ability(package_name)
        return self._hdc_shell(["aa", "start", "-a", ability, "-b", package_name], HDC_APP_TIMEOUT)

    def _main_ability(self, package_name: str) -> str:
        try:
            output = str(self._connector().shell(["bm", "dump", "-n", package_name], timeout=HDC_APP_TIMEOUT) or "")
            ability = select_main_ability(output)
        except Exception as exc:  # noqa: BLE001 - 解析失败时使用 hmdriver2 的默认值
            logger.warning("h2 查询主 Ability 失败，使用 MainAbility: {}", exc)
            ability = ""
        if not ABILITY_NAME_PATTERN.fullmatch(ability):
            ability = "MainAbility"
        return ability

    def stop_app(self, package_name: str):
        package_name = validate_package_name(package_name)
        logger.info("h2 停止应用: {}", package_name)
        return self._hdc_shell(["aa", "force-stop", package_name], HDC_APP_TIMEOUT)

    # ---- 输入注入（hmdriver2 RPC） --------------------------------------
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
            logger.debug("h2 通过坐标输入: ({}, {})", x, y)
            self.click(x, y)
            self._wait_for_input_focus()
        if clear:
            self._clear_focused_text()
        logger.debug("h2 通过焦点控件输入(clear={}): {}", clear, text)
        return self._rpc("input_text", text)

    def _clear_focused_text(self) -> None:
        """清空当前焦点输入框，使 input 与 Android 端（send_keys clear=True）语义一致。

        优先用 hmdriver2 的焦点控件 Component.clearText（一次 RPC）；找不到焦点控件
        时退回按键：Ctrl+A 全选后删除，可选再“移到末尾 + 连续退格”兜底。
        """
        try:
            focused = self._driver(focused=True)
            # find_component 默认每次重试前 sleep 1s；这里只查一次且不等待。
            if not focused.find_component(retries=1, wait_time=0):
                raise LookupError("未找到焦点控件")
            method = getattr(type(focused), "clear_text", None)
            raw = getattr(method, "__wrapped__", None)
            raw(focused) if callable(raw) else focused.clear_text()
            return
        except Exception as exc:  # noqa: BLE001 - 退回按键清空
            logger.debug("h2 焦点控件 clearText 失败，改用按键清空: {}", exc)
        try:
            self._hdc_shell(
                ["uitest", "uiInput", "keyEvent", KEYCODE_CTRL_LEFT, KEYCODE_A], HDC_SHORT_TIMEOUT
            )
            self._press_key(KEYCODE_DEL)
            if H2_CLEAR_BACKSPACE_COUNT:
                self._press_key(KEYCODE_MOVE_END)
                for _ in range(H2_CLEAR_BACKSPACE_COUNT):
                    self._press_key(KEYCODE_DEL)
        except Exception as exc:  # noqa: BLE001 - 清空失败仍继续输入，由执行器观察结果
            logger.warning("h2 清空输入框失败（继续输入）: {}", exc)

    def back(self):
        return self._press_key(KEYCODE_BACK)

    # ---- 屏幕电源 -------------------------------------------------------
    def keep_awake(self) -> None:
        self.wake_screen()

    def wake_screen(self) -> None:
        try:
            self._hdc_shell(["power-shell", "wakeup"], HDC_SHORT_TIMEOUT)
        except Exception as exc:  # noqa: BLE001 - 尽力而为
            logger.warning("h2 唤醒屏幕失败（忽略继续）: {}", exc)

    # ---- 前台应用 -------------------------------------------------------
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
                logger.warning("h2 获取当前前台应用失败: {}", exc)
                return "", ""
        return self._current_app_from_driver()

    def _current_app(self):
        return self.current_app()

    def _current_app_from_driver(self):
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

    # ---- 截图与尺寸 -----------------------------------------------------
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
                raise RuntimeError(f"h2 截图文件拉取失败: {output.strip()[:300]}")
            image = cv2.imread(local_path)
            if image is None:
                raise RuntimeError(f"h2 截图文件无法解码: {local_path}")
            self._record_frame_size(image)
            if path:
                logger.debug("h2 保存截图: {}", path)
                shutil.copyfile(local_path, path)
            return image
        finally:
            try:
                self._hdc_shell(["rm", "-f", remote_path], HDC_SHORT_TIMEOUT)
            except Exception as exc:  # noqa: BLE001 - 远端临时文件清理失败不影响本次截图
                logger.debug("h2 清理设备端截图临时文件失败: {}", exc)
            if os.path.isfile(local_path):
                os.remove(local_path)

    def _resolve_display_size(self):
        # hmdriver2 的 display_size 是 cached_property，旋转后不会更新；
        # 先使其失效再读取，保证拿到的是当前方向的尺寸。
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
            # 与 _normalize_point 一致：只有 [0, 1] 内的浮点数才是比例坐标，整数 0/1 是像素。
            return value * extent if isinstance(value, float) and 0 <= value <= 1 else value

        start_x, start_y = absolute(x1, width), absolute(y1, height)
        end_x, end_y = absolute(x2, width), absolute(y2, height)
        distance = ((end_x - start_x) ** 2 + (end_y - start_y) ** 2) ** 0.5

        speed = int(distance / duration)
        return min(max(speed, 200), 40000)
