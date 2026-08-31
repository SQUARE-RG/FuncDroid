"""“重启风暴”修复的回归测试。

背景（微信_11 实测第 14 轮）：open_app 在场景执行语义下被实现为
force-stop 重启；微信冷启动超过固定 2s 等待，截图仍是桌面，执行模型
误判后连续 4 次 open_app，每次都真实强杀重启应用，形成正反馈。
"""

from __future__ import annotations

import base64

import cv2
import numpy as np

from model.flow_graph import Page
from workflow.nodes import scenario_exec as scenario_exec_module
from workflow.nodes.scenario_exec import (
    OPEN_APP_STALL_REPEAT_WINDOW,
    _detect_stall,
    _looks_like_launcher_package,
    _recover_target_app_by_back,
    _settle_foreground_state,
    _wait_target_app_foreground,
)


def _screenshot(value: int) -> str:
    image = np.full((48, 32, 3), value, dtype=np.uint8)
    success, buffer = cv2.imencode(".png", image)
    assert success
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


class FakeDevice:
    def __init__(self, packages: list[str]):
        # current_package() 依次返回 packages，最后停留在末项。
        self._packages = list(packages)
        self.back_presses = 0
        self.restarts = 0

    def current_package(self):
        if len(self._packages) > 1:
            return self._packages.pop(0)
        return self._packages[0]

    def back(self):
        self.back_presses += 1


def test_wait_target_app_foreground_polls_until_app_visible():
    # 冷启动期间前 3 次查询是桌面，之后目标应用可见。
    device = FakeDevice([
        "com.hihonor.android.launcher",
        "com.hihonor.android.launcher",
        "com.hihonor.android.launcher",
        "com.tencent.mm",
    ])
    assert _wait_target_app_foreground(device, "com.tencent.mm", max_wait=5) is True


def test_wait_target_app_foreground_times_out():
    device = FakeDevice(["com.hihonor.android.launcher"])
    assert _wait_target_app_foreground(device, "com.tencent.mm", max_wait=0.3) is False


def test_looks_like_launcher_package():
    assert _looks_like_launcher_package("com.hihonor.android.launcher")
    assert _looks_like_launcher_package("com.android.launcher3")
    assert not _looks_like_launcher_package("com.tencent.mm")


def test_back_recovery_skipped_on_launcher(monkeypatch):
    monkeypatch.setattr(scenario_exec_module, "FAST_ACTION_DELAY", 0.0)
    device = FakeDevice(["com.hihonor.android.launcher"])

    assert _recover_target_app_by_back(device, "com.tencent.mm") is False
    # 桌面上按返回无效：一次都不按，直接交还给调用方走重启路径。
    assert device.back_presses == 0


def test_back_recovery_presses_when_app_backgrounded(monkeypatch):
    monkeypatch.setattr(scenario_exec_module, "FAST_ACTION_DELAY", 0.0)
    # 非桌面的其他应用：正常逐次返回。
    device = FakeDevice(["com.other.app"])

    assert _recover_target_app_by_back(device, "com.tencent.mm", attempts=2) is False
    assert device.back_presses == 2


def _stall_step(action_name: str, screenshot: str) -> dict:
    return {
        "action_name": action_name,
        "app_name": "微信",
        "current_page": Page(representative_screenshot=screenshot),
    }


def test_open_app_has_tighter_repeat_window_than_default():
    screenshot = _screenshot(30)
    steps = [
        _stall_step("open_app", screenshot),
        _stall_step("open_app", screenshot),
    ]
    reason = _detect_stall(steps)
    assert "open_app" in reason and "2 次" in reason
    assert OPEN_APP_STALL_REPEAT_WINDOW == 2


def test_single_open_app_is_not_stall():
    screenshot = _screenshot(30)
    steps = [
        _stall_step("open_app", screenshot),
        _stall_step("click", _screenshot(60)),
    ]
    assert _detect_stall(steps) == ""


def test_settle_foreground_state_recovers_transient_launcher(monkeypatch):
    """P5 回归：页面切换动画期 launcher 过渡态不得定性为外部跳转。

    微信实测：点击服务号卡片后 current_package 瞬时上报桌面，最终
    稳定在目标应用内的 webview Activity。
    """
    monkeypatch.setattr(scenario_exec_module, "EXTERNAL_RECHECK_DELAY", 0.01)
    device = FakeDevice([
        "com.hihonor.android.launcher",  # 初检（过渡态）
        "com.tencent.mm",                # 复检恢复
    ])
    state, current = _settle_foreground_state(device, "com.tencent.mm")
    assert state == "foreground"
    assert current == "com.tencent.mm"


def test_settle_foreground_state_confirms_real_exit(monkeypatch):
    monkeypatch.setattr(scenario_exec_module, "EXTERNAL_RECHECK_DELAY", 0.01)
    device = FakeDevice(["com.other.forever.app"])  # 复检仍非目标应用
    state, current = _settle_foreground_state(device, "com.tencent.mm")
    assert state == "background"
    assert current == "com.other.forever.app"
