# Multilingual UI and model-response fixtures use Unicode escapes to preserve their exact text.
"""Device and batch-script regressions use fake objects only, without accessing a real device."""

import base64
import json
import struct
import subprocess
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import automator.base as base_module
import automator.u2 as u2_module
import device as device_module
from automator.base import parse_activity_top
from automator.h2 import H2
from automator.u2 import U2, orient_size, parse_raw_screencap
from connector.adb import parse_adb_devices
from connector.hdc import parse_hdc_targets
from device import Driver
from scripts import _batch_common, batch_compliance_run, batch_test_apks
from tools import actions
from tools.utils import screenshot_dimensions


# Shared fake objects.
class FakeAdbDevice:
    def __init__(self, outputs=None):
        self.commands = []
        self.outputs = outputs or {}

    def shell2(self, args, timeout=None):
        self.commands.append(list(args))
        key = " ".join(args[:2])
        returncode, output = self.outputs.get(key, (0, ""))
        return SimpleNamespace(returncode=returncode, output=output)


def make_u2(adb_device=None, **driver_attrs):
    automator = object.__new__(U2)
    automator._serial = "emulator-5554"
    automator._display_size_cache = None
    automator._last_frame_size = None
    automator._driver = SimpleNamespace(adb_device=adb_device or FakeAdbDevice(), **driver_attrs)
    return automator


class FakeConnector:
    def __init__(self, outputs=None):
        self.shell_calls = []
        self.run_calls = []
        self.outputs = outputs or {}

    def shell(self, args, timeout=None):
        assert isinstance(args, list), "H2 must call the connector with a list of arguments"
        assert timeout is not None, "H2 device commands must set a timeout"
        self.shell_calls.append(list(args))
        return self.outputs.get(args[0], "")

    def run_cmd(self, args, timeout=None):
        assert timeout is not None
        self.run_calls.append(list(args))
        if args[:2] == ["file", "recv"]:
            cv2.imwrite(args[3], np.full((40, 20, 3), 128, dtype=np.uint8))
        return ""


class FakeHmDriver:
    """Fake hmdriver2.Driver with delay-style wrappers that record whether dispatch bypasses them."""

    def __init__(self):
        self.calls = []
        self.delayed = []

    def _record(self, name, *args):
        self.calls.append((name, *args))

    def _make(name):  # noqa: N805 - Small factory inside the class body.
        import functools

        def raw(self, *args):
            self._record(name, *args)

        @functools.wraps(raw)
        def wrapper(self, *args):
            raw(self, *args)
            self.delayed.append(name)

        return wrapper

    click = _make("click")
    long_click = _make("long_click")
    swipe = _make("swipe")
    input_text = _make("input_text")
    del _make

    display_size = (1080, 2340)

    def __call__(self, **kwargs):
        raise LookupError("no focused element")

    def current_app(self):  # pragma: no cover - Not called when a connector is available.
        pytest.fail("\u6709 connector \u65f6\u4e0d\u5e94\u8d70 hmdriver2 \u7684\u65e0\u8d85\u65f6 hdc \u5c42")


def make_h2(connector=None, driver=None):
    automator = object.__new__(H2)
    automator._serial = "HM-1"
    automator._display_size_cache = None
    automator._last_frame_size = None
    automator._driver = driver or FakeHmDriver()
    automator.device = SimpleNamespace(connector=connector or FakeConnector())
    return automator


# 1. Screen size follows orientation changes.
def test_u2_display_size_follows_latest_frame_after_rotation(monkeypatch):
    automator = make_u2()
    monkeypatch.setattr(U2, "_resolve_display_size", lambda self: (1080, 2400))
    assert automator.display_size() == (1080, 2400)

    frames = iter([np.zeros((2400, 1080, 3), np.uint8), np.zeros((1080, 2400, 3), np.uint8)])
    monkeypatch.setattr(U2, "_adb_screenshot", lambda self: next(frames))
    automator.screenshot()
    assert automator.display_size() == (1080, 2400)
    automator.screenshot()  # Rotate to landscape.
    assert automator.display_size() == (2400, 1080)


def test_wm_size_fallback_is_swapped_to_current_orientation(monkeypatch):
    # Missing uiautomator2 dimensions fall back to wm size and swap for the current rotation.
    automator = make_u2(info={})
    monkeypatch.setattr(U2, "_adb_wm_size", lambda self: (1080, 2400))
    monkeypatch.setattr(U2, "_adb_rotation", lambda self: 1)
    assert automator._resolve_display_size() == (2400, 1080)
    assert orient_size((1080, 2400), 0) == (1080, 2400)
    assert orient_size((1080, 2400), 3) == (2400, 1080)
    assert orient_size((1080, 2400), None) == (1080, 2400)


def test_h2_display_size_uses_frame_and_invalidates_cached_property():
    driver = FakeHmDriver()
    invalidated = []
    driver._invalidate_cache = invalidated.append
    automator = make_h2(driver=driver)
    assert automator.display_size() == (1080, 2340)
    assert invalidated == ["display_size"]
    automator.screenshot()
    assert automator.display_size() == (20, 40)


def test_h2_duration_to_speed_treats_integer_zero_and_one_as_pixels():
    automator = make_h2()
    automator._last_frame_size = (1000, 2000)
    # Integer (1,1)->(1,201) spans 200 pixels; interpreting 1 as a ratio previously produced about 2000.
    assert automator._duration_to_speed(1, 1, 1, 201, 1.0) == 200
    assert automator._duration_to_speed(0.5, 0.1, 0.5, 0.6, 1.0) == 1000


# 3/8/15. H2 routing.
def test_h2_device_commands_go_through_connector_with_timeouts():
    connector = FakeConnector(outputs={
        "bm": json.dumps({"mainEntry": "entry", "hapModuleInfos": [{
            "mainAbility": "EntryAbility",
            "abilityInfos": [
                {"name": "OtherAbility", "moduleName": "entry", "skills": []},
                {"name": "EntryAbility", "moduleName": "entry", "skills": [{"actions": ["x"]}]},
            ]}]}),
        "aa": "Mission ID #1 {\n state #FOREGROUND\n bundle name [com.example.harmony]\n"
              " main name [EntryAbility]\n isKeepAlive: false\n }",
    })
    automator = make_h2(connector)

    automator.start_app("com.example.harmony")
    automator.stop_app("com.example.harmony")
    automator.back()
    assert automator.current_app() == ("com.example.harmony", "EntryAbility")
    image = automator.screenshot()

    calls = connector.shell_calls
    assert ["aa", "start", "-a", "EntryAbility", "-b", "com.example.harmony"] in calls
    assert ["aa", "force-stop", "com.example.harmony"] in calls
    assert ["uitest", "uiInput", "keyEvent", "2"] in calls
    assert ["aa", "dump", "-l"] in calls
    assert any(call[0] == "snapshot_display" for call in calls)
    assert any(call[:2] == ["rm", "-f"] for call in calls)
    assert connector.run_calls[0][:2] == ["file", "recv"]
    assert image.shape[:2] == (40, 20)


@pytest.mark.parametrize("bad", ["com.x;reboot", "com.x && rm", "x", "com.x$(id)"])
def test_h2_rejects_unsafe_package_names(bad):
    connector = FakeConnector()
    automator = make_h2(connector)
    with pytest.raises(ValueError):
        automator.start_app(bad)
    with pytest.raises(ValueError):
        automator.stop_app(bad)
    assert connector.shell_calls == []


def test_h2_ui_injection_bypasses_hmdriver2_delay_and_click_returns_point():
    driver = FakeHmDriver()
    automator = make_h2(driver=driver)
    assert automator.click(10, 20) == {"x": 10, "y": 20}
    assert automator.long_click(3, 4) == {"x": 3, "y": 4}
    automator.swipe(0, 0, 0, 500, 0.5)
    assert [call[0] for call in driver.calls] == ["click", "long_click", "swipe"]
    assert driver.delayed == []


def test_h2_rpc_falls_back_to_decorated_call_without_wrapped():
    class PlainDriver:
        def __init__(self):
            self.calls = []

        def click(self, x, y):
            self.calls.append((x, y))

    automator = make_h2(driver=PlainDriver())
    automator.click(5, 6)
    assert automator._driver.calls == [(5, 6)]


# 4. Input semantics.
def test_u2_input_clears_by_default_and_waits_for_focus(monkeypatch):
    sent = []
    sleeps = []
    automator = make_u2(send_keys=lambda text, clear=False: sent.append((text, clear)))
    monkeypatch.setattr(U2, "_shell_input", lambda self, *args: None)
    monkeypatch.setattr(base_module.time, "sleep", sleeps.append)
    monkeypatch.setattr(base_module, "INPUT_FOCUS_DELAY", 0.2)

    automator.input("abc", x=10, y=20)
    automator.input("def", clear=False)

    assert sent == [("abc", True), ("def", False)]
    assert sleeps == [0.2]


def test_h2_input_clears_focused_field_before_typing(monkeypatch):
    connector = FakeConnector()
    driver = FakeHmDriver()
    automator = make_h2(connector, driver)
    monkeypatch.setattr(base_module, "INPUT_FOCUS_DELAY", 0)

    automator.input("hello")
    # Without a focused control, clear with Ctrl+A and Backspace before typing.
    assert ["uitest", "uiInput", "keyEvent", "2072", "2017"] in connector.shell_calls
    assert ["uitest", "uiInput", "keyEvent", "2055"] in connector.shell_calls
    assert driver.calls[-1] == ("input_text", "hello")

    connector.shell_calls.clear()
    automator.input("more", clear=False)
    assert connector.shell_calls == []


def test_driver_input_forwards_clear_flag():
    received = []
    driver = object.__new__(Driver)
    driver._foreground_cache = None
    driver.automator = SimpleNamespace(input=lambda text, x=None, y=None, clear=True: received.append((text, x, y, clear)))
    driver.input("a")
    driver.input(1, 2, "b", clear=False)
    assert received == [("a", None, None, True), ("b", 1, 2, False)]


# 5. Keep awake and wake-up behavior.
def test_u2_keep_awake_and_wake_are_best_effort():
    adb_device = FakeAdbDevice(outputs={"svc power": (1, "denied")})
    automator = make_u2(adb_device)
    automator.keep_awake()  # Failures are logged only.
    automator.wake_screen()
    assert ["svc", "power", "stayon", "usb"] in adb_device.commands
    assert ["input", "keyevent", "KEYCODE_WAKEUP"] in adb_device.commands


def test_black_screenshot_triggers_wake_before_retry(monkeypatch):
    events = []
    frames = iter([np.zeros((50, 50, 3), np.uint8), np.full((50, 50, 3), 120, np.uint8)])
    fake = SimpleNamespace(
        screenshot=lambda: events.append("shot") or next(frames),
        wake_screen=lambda: events.append("wake"),
        current_activity=lambda: "Main",
    )
    monkeypatch.setattr(actions.time, "sleep", lambda _s: None)
    page, error = actions.capture_page_observation(fake)
    assert error == ""
    assert events == ["shot", "wake", "shot"]


# 6. Foreground query caching and parsing.
def _driver_with_counter(result=("com.a", ".Main")):
    calls = []
    driver = object.__new__(Driver)
    driver._foreground_cache = None
    driver.automator = SimpleNamespace(
        current_app=lambda: calls.append(1) or result,
        click=lambda x, y: {"x": x, "y": y},
        back=lambda: None,
    )
    return driver, calls


def test_current_app_ttl_cache_and_action_invalidation(monkeypatch):
    monkeypatch.setattr(device_module, "FOREGROUND_CACHE_TTL", 10.0)
    driver, calls = _driver_with_counter()
    assert driver.current_package() == "com.a"
    assert driver.current_activity() == ".Main"
    assert driver.current_app() == ("com.a", ".Main")
    assert len(calls) == 1
    driver.click(1, 2)
    driver.current_package()
    assert len(calls) == 2
    driver.back()
    driver.current_activity()
    assert len(calls) == 3


def test_current_app_cache_expires_and_failures_are_not_cached(monkeypatch):
    monkeypatch.setattr(device_module, "FOREGROUND_CACHE_TTL", 0.0)
    driver, calls = _driver_with_counter()
    driver.current_package()
    driver.current_package()
    assert len(calls) == 2
    monkeypatch.setattr(device_module, "FOREGROUND_CACHE_TTL", 10.0)
    driver, calls = _driver_with_counter(("", ""))
    driver.current_package()
    driver.current_package()
    assert len(calls) == 2


ACTIVITY_TOP = """TASK 10 id=5 userId=0
  ACTIVITY com.android.launcher3/.Launcher 1a2b pid=900
    Local Activity 1 State:
      mResumed=false mStopped=true
TASK 20 id=9 userId=0
  ACTIVITY com.example.app/.ui.Home 3c4d pid=1200
    Local Activity 2 State:
      mResumed=true mStopped=false
"""


def test_activity_top_parser_picks_top_resumed_activity():
    assert parse_activity_top(ACTIVITY_TOP) == ("com.example.app", ".ui.Home")
    # Without mResumed, select the last/topmost task, matching adbutils.
    plain = "ACTIVITY a.b/.First 1 pid=1\nACTIVITY c.d/.Second 2 pid=2\n"
    assert parse_activity_top(plain) == ("c.d", ".Second")
    assert parse_activity_top("") == ("", "")


def test_u2_current_app_single_query_then_activity_top_fallback():
    class Device(FakeAdbDevice):
        attempts = 0

        def app_current(self):  # adbutils-style retry wrapper with __wrapped__.
            raise AssertionError("must call the unwrapped single attempt")

    def single(self):
        Device.attempts += 1
        raise RuntimeError("Couldn't get focused app")

    Device.app_current.__wrapped__ = single
    adb_device = Device(outputs={"dumpsys activity": (0, ACTIVITY_TOP)})
    automator = make_u2(adb_device)
    assert automator.current_app() == ("com.example.app", ".ui.Home")
    assert Device.attempts == 1
    assert automator.current_package() == "com.example.app"


# ---------------------------------------------------------------- 7. raw screencap
def _raw_frame(width, height, header_size, pixel_format=1):
    rgba = np.zeros((height, width, 4), np.uint8)
    rgba[..., 0] = 200  # R
    rgba[..., 1] = 10   # G
    rgba[..., 2] = 50   # B
    rgba[..., 3] = 255
    header = struct.pack("<III", width, height, pixel_format)
    if header_size == 16:
        header += struct.pack("<I", 1)
    return header + rgba.tobytes()


@pytest.mark.parametrize("header_size", [12, 16])
def test_raw_screencap_header_variants(header_size):
    image = parse_raw_screencap(_raw_frame(4, 3, header_size))
    assert image.shape == (3, 4, 3)
    assert tuple(int(v) for v in image[0, 0]) == (50, 10, 200)  # BGR


@pytest.mark.parametrize("data", [
    b"",
    b"\x89PNG\r\n\x1a\n" + b"0" * 64,
    _raw_frame(4, 3, 16)[:-1],
    _raw_frame(4, 3, 16) + b"\n",
    _raw_frame(4, 3, 16, pixel_format=4),
])
def test_raw_screencap_mismatch_returns_none(data):
    assert parse_raw_screencap(data) is None


def test_screenshot_falls_back_to_png_when_raw_unparseable(monkeypatch):
    automator = make_u2()
    monkeypatch.setattr(U2, "_raw_screenshot", lambda self: None)
    png = np.full((7, 5, 3), 9, np.uint8)
    monkeypatch.setattr(U2, "_png_screenshot", lambda self: png)
    assert automator.screenshot() is png
    assert automator.display_size() == (5, 7)


def test_raw_screenshot_reads_stream_through_adb_connection():
    payload = _raw_frame(6, 2, 16)

    class Sock:
        def __init__(self):
            self.chunks = [payload[:10], payload[10:], b""]

        def settimeout(self, _t):
            pass

        def recv(self, _n):
            return self.chunks.pop(0)

    class Conn:
        conn = Sock()
        closed = False

        def close(self):
            Conn.closed = True

    opened = []
    adb_device = SimpleNamespace(open_shell=lambda cmd: opened.append(cmd) or Conn())
    automator = make_u2(adb_device)
    image = automator._raw_screenshot()
    assert opened == ["screencap"]
    assert image.shape == (2, 6, 3)
    assert Conn.closed


def test_shell_input_uses_adb_connection_and_reports_failures():
    adb_device = FakeAdbDevice(outputs={"input tap": (0, "")})
    automator = make_u2(adb_device)
    automator.click(10, 20)
    assert adb_device.commands == [["input", "tap", "10", "20"]]
    adb_device.outputs["input keyevent"] = (137, "Killed")
    with pytest.raises(RuntimeError, match="injection failed"):
        automator.back()


# 9. Image header dimensions.
def test_screenshot_dimensions_reads_png_header_without_decoding():
    ok, buffer = cv2.imencode(".png", np.zeros((11, 13, 3), np.uint8))
    encoded = base64.b64encode(buffer.tobytes()).decode()
    assert screenshot_dimensions(encoded) == (13, 11)
    assert screenshot_dimensions("data:image/png;base64," + encoded) == (13, 11)
    assert screenshot_dimensions(base64.b64encode(b"\xff\xd8\xff garbage").decode()) is None


# 10. Device listing and installation status.
def test_adb_devices_keeps_only_ready_devices():
    output = (
        "List of devices attached\n"
        "* daemon started successfully\n"
        "emulator-5554\tdevice\n"
        "R58M\toffline\n"
        "ABCD\tunauthorized\n"
        "192.168.0.2:5555 device product:x model:y\n"
    )
    assert parse_adb_devices(output) == ["emulator-5554", "192.168.0.2:5555"]
    assert parse_hdc_targets("[Empty]\n") == []
    assert parse_hdc_targets("HM-1\nHM-2\n") == ["HM-1", "HM-2"]


def test_incompatible_skip_survives_install_query_failure(tmp_path, monkeypatch):
    info = batch_test_apks.ApkInfo("x.apk", "abc", "com.example", "1", "1", "X", 99, ())
    monkeypatch.setattr(batch_test_apks, "find_aapt", lambda _: "aapt")
    monkeypatch.setattr(batch_test_apks, "build_manifest", lambda *_: ([info], []))
    monkeypatch.setattr(batch_test_apks, "device_capabilities", lambda *_: (30, {"arm64-v8a"}))
    monkeypatch.setenv("API_KEY", "k")
    monkeypatch.setenv("BASE_URL", "u")
    monkeypatch.setenv("MODEL", "m")
    monkeypatch.setattr(batch_test_apks.signal, "signal", lambda *a: None)

    def broken(*_a):
        raise RuntimeError("device offline")

    monkeypatch.setattr(batch_test_apks, "is_installed", broken)
    code = batch_test_apks.main(["--apk-root", str(tmp_path), "--device-id", "d",
                                 "--output-dir", str(tmp_path / "out")])
    assert code == 1
    record = json.loads((tmp_path / "out" / "batch-results.jsonl").read_text().splitlines()[0])
    assert record["test"]["reason"].startswith("min_sdk")
    assert record["cleanup"]["verified_absent"] is False


def test_is_installed_and_app_installed_share_query_semantics(monkeypatch):
    def fake(*_a, **_k):
        return subprocess.CompletedProcess([], 0, "package:com.example.extra\n", "")

    monkeypatch.setattr(batch_compliance_run, "_common_adb", fake)
    monkeypatch.setattr(batch_test_apks, "_adb", fake)
    assert batch_test_apks.is_installed("d", "com.example") is False
    assert batch_compliance_run.app_installed("d", "com.example") is False

    def timeout(*_a, **_k):
        raise subprocess.TimeoutExpired("adb", 1)

    monkeypatch.setattr(batch_compliance_run, "_common_adb", timeout)
    monkeypatch.setattr(batch_test_apks, "_adb", timeout)
    with pytest.raises(RuntimeError):
        batch_test_apks.is_installed("d", "com.example")
    assert batch_compliance_run.app_installed("d", "com.example") is False


# 11. Interruption propagation.
def test_interrupt_during_install_stops_queue(tmp_path, monkeypatch):
    info = batch_test_apks.ApkInfo("x.apk", "abc", "com.example", "1", "1", "X", 21, ())
    monkeypatch.setattr(batch_test_apks, "cleanup_package", lambda *_: {"verified_absent": True})
    monkeypatch.setattr(batch_test_apks, "collect_run_metrics", lambda *_: {})

    def interrupted(*_a):
        raise batch_test_apks.BatchInterrupted("received_signal_2")

    monkeypatch.setattr(batch_test_apks, "install_apk", interrupted)
    result = batch_test_apks.run_one(info, device_id="d", total_time_minutes=1, max_iterations=1,
                                     batch_output_dir=tmp_path, project_root=tmp_path)
    assert result["test"]["status"] == "interrupted"
    assert result["batch_interrupted"] is True


def test_cleanup_package_does_not_swallow_interrupt(monkeypatch):
    def interrupted(*_a, **_k):
        raise batch_test_apks.BatchInterrupted("received_signal_15")

    monkeypatch.setattr(batch_test_apks, "_adb", interrupted)
    with pytest.raises(batch_test_apks.BatchInterrupted):
        batch_test_apks.cleanup_package("d", "com.example")


# 12. Compliance batch timeout and cooldown.
def test_compliance_unlimited_budget_uses_finite_fallback_not_ten_minutes():
    assert batch_compliance_run._process_timeout(0) == batch_compliance_run.UNLIMITED_PROCESS_TIMEOUT_SECONDS
    assert batch_compliance_run._process_timeout(2) == 120 + batch_compliance_run.PROCESS_TIMEOUT_GRACE_SECONDS


@pytest.mark.parametrize("flag,value", [("--total-time", "-1"), ("--total-time", "nan"),
                                        ("--cooldown", "-5"), ("--limit", "-1")])
def test_compliance_rejects_negative_options(flag, value):
    with pytest.raises(SystemExit):
        batch_compliance_run._build_parser().parse_args(["--device-id", "d", "--app", "A:com.a", flag, value])


def test_cooldown_only_after_an_actually_run_app(tmp_path, monkeypatch):
    done = batch_compliance_run.AppEntry(1, "Done", "com.done")
    folder = tmp_path / done.dir_name
    folder.mkdir()
    (folder / "run_result.json").write_text('{"status":"success","mode":"compliance","package_name":"com.done"}')
    (folder / "artifact_manifest.json").write_text('{"status":"complete"}')
    (folder / "compliance_report.json").write_text('{"checks":[]}')

    stopped, slept, ran = [], [], []
    monkeypatch.setattr(batch_compliance_run, "wait_for_device", lambda *_: True)
    monkeypatch.setattr(batch_compliance_run, "app_installed", lambda *_: True)
    monkeypatch.setattr(batch_compliance_run, "force_stop_app", lambda _d, package: stopped.append(package))
    monkeypatch.setattr(batch_compliance_run.time, "sleep", slept.append)
    monkeypatch.setattr(batch_compliance_run.signal, "signal", lambda *a: None)
    monkeypatch.setattr(batch_compliance_run, "_run_one",
                        lambda entry, **_k: ran.append(entry.package) or {"ok": True})
    code = batch_compliance_run.main([
        "--device-id", "d", "--output-root", str(tmp_path), "--cooldown", "3",
        "--app", "Done:com.done", "--app", "B:com.b", "--app", "C:com.c",
    ])
    assert code == 0
    assert ran == ["com.b", "com.c"]
    # No cooldown before the first executed app and no force-stop for skipped com.done.
    assert stopped == ["com.b"]
    assert slept == [3.0]


# 13. Shared modules and legacy names.
def test_shared_helpers_remain_importable_from_both_scripts():
    for module in (batch_test_apks, batch_compliance_run):
        assert module.BatchInterrupted is _batch_common.BatchInterrupted
        assert callable(module._terminate_process_tree)
        assert callable(module._signal_process_tree)
        assert callable(module._adb)
        assert callable(module.run_artifacts_complete)
    assert batch_test_apks.append_jsonl is _batch_common.append_jsonl
    assert batch_test_apks._resolve_process_timeout(0, None) == batch_test_apks.DEFAULT_BATCH_PROCESS_TIMEOUT


# 14. Small script fixes.
def test_staged_stage_timeout_is_recorded_as_timeout(tmp_path, monkeypatch):
    from scripts import staged_apk_experiment as staged

    info = staged.batch.ApkInfo("example.apk", "hash", "com.example", "1", "1.0", "Example", 21, ())
    installed = iter([False, True])
    monkeypatch.setattr(staged.batch, "inspect_apk", lambda *_: info)
    monkeypatch.setattr(staged.batch, "is_installed", lambda *_: next(installed))
    monkeypatch.setattr(staged.batch, "device_capabilities", lambda *_: (33, set()))
    monkeypatch.setattr(staged, "network_state", lambda *_: {"status": "connected"})
    monkeypatch.setattr(staged, "snapshot_sources", lambda *_: {})
    monkeypatch.setattr(staged.batch, "install_apk", lambda *_: SimpleNamespace(returncode=0, stdout="", stderr=""))
    monkeypatch.setattr(staged.batch, "cleanup_package", lambda *_: {"verified_absent": True})
    monkeypatch.setattr(staged.batch, "collect_run_metrics", lambda *_: {})
    monkeypatch.setattr(staged.batch, "_terminate_process_tree", lambda *_: None)

    def wait(**kwargs):
        raise subprocess.TimeoutExpired("main.py", kwargs.get("timeout"))

    monkeypatch.setattr(staged.subprocess, "Popen", lambda *a, **k: SimpleNamespace(wait=wait))
    monkeypatch.setattr(staged.subprocess, "run", lambda command, **_: SimpleNamespace(
        stdout="revision\n" if command[1] == "rev-parse" else b""))
    args = SimpleNamespace(apk="example.apk", device_id="t", output_dir=str(tmp_path / "run"),
                           compliance_minutes=1, defect_minutes=1)
    assert staged.run_experiment(args) == 1
    stage = json.loads((tmp_path / "run/experiment.json").read_text())["stages"][0]
    assert stage["status"] == "timeout"
    assert stage["timeout_seconds"] == 360


def test_defect_report_overview_uses_deduped_counts(tmp_path):
    from scripts.collect_defect_report import load_batch, render

    bug = {"bug_id": "B1", "type": "logic_error", "evidence_category": "ui_error",
           "severity": "high", "confidence": "high", "description": "\u540c\u4e00\u95ee\u9898"}
    for label in ("a", "b"):
        app_dir = tmp_path / label / "App"
        app_dir.mkdir(parents=True)
        (app_dir / "run_result.json").write_text(json.dumps({
            "status": "success", "app_name": "App", "package_name": "com.app"}))
        (app_dir / "bugs.json").write_text(json.dumps([bug]))
    batches = [load_batch("A", str(tmp_path / "a")), load_batch("B", str(tmp_path / "b"))]
    markdown, payload = render(batches)
    overview = markdown.split("## 2.")[0]
    assert payload["total_defects"] == 1
    total_in_overview = sum(
        int(line.split("|")[3]) for line in overview.splitlines()
        if line.startswith("| A ") or line.startswith("| B ")
    )
    assert total_in_overview == payload["total_defects"]


def test_historical_report_glob_excludes_previous_archives(tmp_path, monkeypatch):
    from scripts import gen_experiment_report as report

    (tmp_path / "01_App").mkdir()
    (tmp_path / "01_App.previous-20260101-000000-000000").mkdir()
    monkeypatch.setattr(report, "COMPLIANCE_GLOB", str(tmp_path / "[0-9][0-9]_*"))
    assert [Path(d).name for d in report._compliance_dirs()] == ["01_App"]


def test_u2_shell_fallback_without_adb_connection_quotes_arguments(monkeypatch):
    captured = []
    monkeypatch.setattr(u2_module.subprocess, "run", lambda command, **kwargs: captured.append(command)
                        or SimpleNamespace(returncode=0, stdout="", stderr=""))
    automator = make_u2()
    automator._driver = None
    automator._adb_shell(["am", "start", "-n", "a.b/.C$D"], 5)
    assert captured[0][:4] == ["adb", "-s", "emulator-5554", "shell"]
    assert captured[0][4] == "am start -n 'a.b/.C$D'"
