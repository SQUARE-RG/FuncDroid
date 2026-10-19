# Multilingual UI and model-response fixtures use Unicode escapes to preserve their exact text.
from types import SimpleNamespace

import pytest

from model.flow_graph import Page
from workflow.nodes import scenario_exec, scenario_plan


@pytest.mark.parametrize("package", ["com.android.permissioncontroller", "com.google.android.permissioncontroller"])
def test_permission_overlay_is_captured_without_restart(monkeypatch, package):
    page = Page(activity="permission-dialog", representative_screenshot="image")
    device = SimpleNamespace(current_package=lambda: package,
                             restart_app=lambda *_: pytest.fail("permission dialog restarted"))
    monkeypatch.setattr(scenario_plan, "capture_current_page", lambda _: page)
    assert scenario_plan._ensure_target_app(device, "com.example") is page
    assert scenario_plan._wait_target_app_foreground(device, "com.example")
    assert scenario_exec._ensure_target_app(device, "com.example")
    assert scenario_exec._wait_target_app_foreground(device, "com.example")
    # Do not relabel system-owned evidence as the target app's own page.
    assert scenario_exec._target_app_foreground_state(device, "com.example") == ("background", package)


def test_unrelated_app_still_requires_recovery(monkeypatch):
    calls = []
    monkeypatch.setattr(scenario_exec, "_restart_target_app", lambda *args: calls.append(args) or False)
    device = SimpleNamespace(current_package=lambda: "com.unrelated.app")
    assert not scenario_exec._ensure_target_app(device, "com.example")
    assert len(calls) == 1


def test_expired_stage_never_starts_device_or_model(monkeypatch):
    monkeypatch.setattr(scenario_exec, "_ensure_target_app", lambda *_: pytest.fail("device touched"))
    result = scenario_exec._run_single_scenario(None, None, "App", "com.example", "\u6d4b\u8bd5", None,
                                                True, "test", deadline=1)
    assert result[0] == []
    assert result[2] == "incomplete"
