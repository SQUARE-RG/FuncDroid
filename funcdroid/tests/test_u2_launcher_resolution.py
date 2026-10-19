# Multilingual UI and model-response fixtures use Unicode escapes to preserve their exact text.
from types import SimpleNamespace

import pytest

import automator.u2 as u2_module
from automator.u2 import U2


class _AdbDevice:
    """Fake adbutils connection that records shell2 commands and returns scripted output."""

    def __init__(self, am_output="Status: ok\nLaunchState: COLD\nComplete\n", returncode=0):
        self.commands = []
        self.am_output = am_output
        self.returncode = returncode

    def shell2(self, args, timeout=None):
        self.commands.append((list(args), timeout))
        return SimpleNamespace(returncode=self.returncode, output=self.am_output)


class _Driver:
    def __init__(self, adb_device=None):
        self.calls = []
        self.adb_device = adb_device or _AdbDevice()

    def app_start(self, package_name, activity=None):
        self.calls.append((package_name, activity))


def _u2_without_connection(adb_device=None):
    automator = object.__new__(U2)
    automator._serial = "emulator-5554"
    automator._driver = _Driver(adb_device)
    return automator


@pytest.fixture(autouse=True)
def _empty_launcher_cache(monkeypatch):
    monkeypatch.setattr(u2_module, "_LAUNCHER_ACTIVITY_CACHE", {})


def _query_output(monkeypatch, output):
    monkeypatch.setattr(
        "automator.u2.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout=output),
    )


def test_start_app_prefers_launcher_from_application_namespace(monkeypatch):
    _query_output(monkeypatch, """2 activities found:
      sample.app.foss/sample.app.MainActivity
      sample.app.foss/com.example.debug.DisplayActivity
    """)
    automator = _u2_without_connection()

    automator.start_app("sample.app.foss")

    command, timeout = automator._driver.adb_device.commands[0]
    assert command[:3] == ["am", "start", "-W"]
    assert command[-2:] == ["-n", "sample.app.foss/sample.app.MainActivity"]
    assert timeout == u2_module.ADB_START_TIMEOUT
    # Successful explicit Activity launch skips uiautomator2's default launch path.
    assert automator._driver.calls == []


def test_relative_launcher_name_is_expanded_before_scoring(monkeypatch):
    # The old lstrip handling scored .MainActivity as zero and selected a library ShortcutActivity.
    _query_output(monkeypatch, """2 activities found:
      com.example.app/.MainActivity
      com.example.app/com.example.lib.ShortcutActivity
    """)
    automator = _u2_without_connection()

    assert automator._preferred_launcher_activity("com.example.app") == "com.example.app.MainActivity"


def test_component_format_with_other_package_prefix_is_expanded():
    assert u2_module.expand_activity_name("com.a", "com.b/.Main") == "com.b.Main"
    assert u2_module.expand_activity_name("com.a", ".Main") == "com.a.Main"
    assert u2_module.expand_activity_name("com.a", "com.a.Main") == "com.a.Main"


@pytest.mark.parametrize("output", [
    "Error: Activity class {com.example.app/com.example.app.Gone} does not exist.",
    "Starting: Intent { ... }\njava.lang.SecurityException: Permission Denial",
])
def test_failed_am_start_clears_cache_and_falls_back_to_default_start(monkeypatch, output):
    _query_output(monkeypatch, "com.example.app/.Gone\n")
    automator = _u2_without_connection(_AdbDevice(am_output=output))

    automator.start_app("com.example.app")

    assert automator._driver.calls == [("com.example.app", None)]
    assert ("emulator-5554", "com.example.app") not in u2_module._LAUNCHER_ACTIVITY_CACHE


def test_am_start_transport_error_also_falls_back(monkeypatch):
    _query_output(monkeypatch, "com.example.app/.Main\n")

    class BrokenDevice(_AdbDevice):
        def shell2(self, args, timeout=None):
            raise TimeoutError("adb read timeout")

    automator = _u2_without_connection(BrokenDevice())
    automator.start_app("com.example.app")
    assert automator._driver.calls == [("com.example.app", None)]


def test_start_app_keeps_default_behavior_when_launcher_query_fails(monkeypatch):
    _query_output(monkeypatch, "No activities found")
    automator = _u2_without_connection()

    automator.start_app("sample.app")

    assert automator._driver.calls == [("sample.app", None)]


@pytest.mark.parametrize("bad", ["", "com.example; rm -rf /", "\u5355\u8bcd", "noDots", "com..x"])
def test_invalid_package_names_are_rejected_before_any_device_command(monkeypatch, bad):
    monkeypatch.setattr("automator.u2.subprocess.run", lambda *a, **k: pytest.fail("device accessed"))
    automator = _u2_without_connection()
    with pytest.raises(ValueError, match="package name"):
        automator.start_app(bad)
