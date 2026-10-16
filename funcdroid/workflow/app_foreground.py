"""目标应用前台状态查询、冷启动等待与重启的共享逻辑。

规划节点（scenario_plan）与执行节点（scenario_exec）都需要“确认目标应用在
前台、否则重启并等待它真正可见”。两处曾各自维护一套常量和轮询/重启循环，
口径漂移后容易重新引入“固定 sleep 截到桌面 → 模型误判重复 open_app →
force-stop 重启风暴”的问题，因此统一放在这里。

调用方的薄封装保留在原模块（测试会 monkeypatch 那些名字），这里只提供
可注入探测函数的通用实现。
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable

from workflow.system_dialogs import is_permission_dialog

# 冷启动等待：固定 sleep 改为轮询前台包名。微信等重型应用冷启动常超过
# APP_RESTART_DELAY（2s），固定等待会截到桌面/启动页截图，诱发执行模型
# 误判并重复 open_app（历史上形成 force-stop 重启风暴）。
APP_START_POLL_INTERVAL = max(0.1, float(os.getenv("APP_START_POLL_INTERVAL", "0.5")))
APP_START_MAX_WAIT = max(1.0, float(os.getenv("APP_START_MAX_WAIT", "12")))
APP_RESTART_SETTLE_DELAY = max(0.0, float(os.getenv("APP_RESTART_SETTLE_DELAY", "1.0")))
APP_RESTART_DELAY = float(os.getenv("APP_RESTART_DELAY", "2.0"))
APP_RESTART_ATTEMPTS = max(1, int(os.getenv("APP_RESTART_ATTEMPTS", "2")))
APP_RESTART_RETRY_DELAY = float(os.getenv("APP_RESTART_RETRY_DELAY", "1.0"))
FOREGROUND_CHECK_RETRIES = max(1, int(os.getenv("FOREGROUND_CHECK_RETRIES", "2")))
FOREGROUND_CHECK_RETRY_DELAY = float(os.getenv("FOREGROUND_CHECK_RETRY_DELAY", "0.2"))
# 一次完整重启（拉起 + 轮询前台 + 稳定等待）可能耗费的最长时间；
# 剩余时间预算不足该值时，调用方应直接结束而不是开始重启。
APP_RESTART_MAX_SECONDS = APP_START_MAX_WAIT + APP_RESTART_SETTLE_DELAY


def current_foreground_package(
    device,
    retries: int = FOREGROUND_CHECK_RETRIES,
    *,
    log_prefix: str = "[场景执行]",
) -> str | None:
    """读取前台包名；连续失败或为空时返回 None（无法确认）。"""
    last_error: Exception | None = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            current_package = str(device.current_package() or "").strip()
            if current_package:
                return current_package
        except Exception as exc:
            last_error = exc
            print(f"{log_prefix} 获取当前前台包名失败: {exc}")
        if attempt < retries:
            time.sleep(FOREGROUND_CHECK_RETRY_DELAY)
    if last_error is None:
        print(f"{log_prefix} 当前前台包名为空，无法确认目标应用状态。")
    return None


def target_app_foreground_state(device, package_name: str) -> tuple[str, str]:
    """返回 (状态, 前台包名)；状态为 foreground / background / unknown。"""
    package_name = str(package_name or "").strip()
    if not package_name:
        return "foreground", ""
    current_package = current_foreground_package(device)
    if current_package is None:
        return "unknown", ""
    if current_package == package_name:
        return "foreground", current_package
    return "background", current_package


def foreground_or_permission(state: str, current_package: str) -> bool:
    """目标应用在前台，或系统权限弹窗覆盖其上（需保留现场交给执行器处理）。"""
    return state == "foreground" or is_permission_dialog(current_package)


def poll_until_foreground(
    probe: Callable[[], bool],
    *,
    max_wait: float | None = None,
    poll_interval: float | None = None,
) -> bool:
    """轮询 probe 直到返回 True 或超时；代替固定时长的冷启动等待。"""
    wait = APP_START_MAX_WAIT if max_wait is None else max(0.0, float(max_wait))
    interval = APP_START_POLL_INTERVAL if poll_interval is None else max(0.0, float(poll_interval))
    deadline = time.monotonic() + wait
    while True:
        if probe():
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(interval)


def wait_target_app_foreground(
    device,
    package_name: str,
    *,
    max_wait: float | None = None,
    state_probe: Callable[[Any, str], tuple[str, str]] | None = None,
) -> bool:
    """轮询等待目标应用进入前台（或被系统权限弹窗覆盖）。"""
    package_name = str(package_name or "").strip()
    if not package_name:
        return True
    probe_state = state_probe or target_app_foreground_state
    return poll_until_foreground(
        lambda: foreground_or_permission(*probe_state(device, package_name)),
        max_wait=max_wait,
    )


def restart_until_foreground(
    device,
    package_name: str,
    *,
    wait_foreground: Callable[[Any, str], bool],
    on_ready: Callable[[], Any] | None = None,
    attempts: int = APP_RESTART_ATTEMPTS,
    log_prefix: str = "[场景执行]",
) -> tuple[bool, Any, list[Exception]]:
    """重启目标应用并轮询等待其进入前台，失败按 attempts 重试。

    返回 (是否成功, on_ready 的返回值, 各次尝试中捕获的异常)。on_ready 在应用
    进入前台并稳定后调用（如抓取当前页面），其抛出的异常视为本次尝试失败。
    """
    errors: list[Exception] = []
    attempts = max(1, int(attempts))
    for attempt in range(1, attempts + 1):
        try:
            print(f"{log_prefix} 第 {attempt}/{attempts} 次重启目标应用。")
            device.restart_app(package_name)
            if wait_foreground(device, package_name):
                time.sleep(APP_RESTART_SETTLE_DELAY)
                return True, (on_ready() if on_ready is not None else None), errors
            print(f"{log_prefix} 重启后目标应用仍未进入前台。")
        except Exception as exc:
            errors.append(exc)
            print(f"{log_prefix} 重启目标应用失败: {exc}")
        if attempt < attempts:
            time.sleep(APP_RESTART_RETRY_DELAY)
    return False, None, errors


def remaining_seconds(deadline: float | None) -> float | None:
    """距离阶段截止（time.time() 口径）的剩余秒数；无截止返回 None。"""
    if deadline is None:
        return None
    return float(deadline) - time.time()


def deadline_allows(deadline: float | None, seconds: float) -> bool:
    """剩余时间是否足够再执行一段耗时 seconds 的等待/重启。"""
    remaining = remaining_seconds(deadline)
    return remaining is None or remaining > max(0.0, float(seconds))
