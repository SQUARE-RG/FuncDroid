# Multilingual UI and model-response fixtures use Unicode escapes to preserve their exact text.
"""Restart-storm regressions: repeated open_app during slow cold start previously force-stopped and restarted the app before it could reach the foreground."""

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
        # current_package() consumes packages and then keeps returning the final entry.
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
    # During cold start, the first three queries report the launcher, then the target app.
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
    # Back is ineffective on the launcher; return immediately for caller-managed restart.
    assert device.back_presses == 0


def test_back_recovery_presses_when_app_backgrounded(monkeypatch):
    monkeypatch.setattr(scenario_exec_module, "FAST_ACTION_DELAY", 0.0)
    # For another non-launcher app, try back normally.
    device = FakeDevice(["com.other.app"])

    assert _recover_target_app_by_back(device, "com.tencent.mm", attempts=2) is False
    assert device.back_presses == 2


def _stall_step(action_name: str, screenshot: str) -> dict:
    return {
        "action_name": action_name,
        "app_name": "\u5fae\u4fe1",
        "current_page": Page(representative_screenshot=screenshot),
    }


def test_open_app_has_tighter_repeat_window_than_default():
    screenshot = _screenshot(30)
    steps = [
        _stall_step("open_app", screenshot),
        _stall_step("open_app", screenshot),
    ]
    reason = _detect_stall(steps)
    assert "open_app" in reason and "2 consecutive repeated actions" in reason
    assert OPEN_APP_STALL_REPEAT_WINDOW == 2


def test_single_open_app_is_not_stall():
    screenshot = _screenshot(30)
    steps = [
        _stall_step("open_app", screenshot),
        _stall_step("click", _screenshot(60)),
    ]
    assert _detect_stall(steps) == ""


def test_settle_foreground_state_recovers_transient_launcher(monkeypatch):
    """Foreground settling must recover transient launcher observations during page transitions before classifying an external jump."""
    monkeypatch.setattr(scenario_exec_module, "EXTERNAL_RECHECK_DELAY", 0.01)
    device = FakeDevice([
        "com.hihonor.android.launcher",  # Initial transient observation.
        "com.tencent.mm",                # Settling restores the target.
    ])
    state, current = _settle_foreground_state(device, "com.tencent.mm")
    assert state == "foreground"
    assert current == "com.tencent.mm"


def test_settle_foreground_state_confirms_real_exit(monkeypatch):
    monkeypatch.setattr(scenario_exec_module, "EXTERNAL_RECHECK_DELAY", 0.01)
    device = FakeDevice(["com.other.forever.app"])  # Settling still reports an external app.
    state, current = _settle_foreground_state(device, "com.tencent.mm")
    assert state == "background"
    assert current == "com.other.forever.app"
