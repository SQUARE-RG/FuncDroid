from types import SimpleNamespace

from automator.u2 import U2


class _Driver:
    def __init__(self):
        self.calls = []

    def app_start(self, package_name, activity=None):
        self.calls.append((package_name, activity))


def _u2_without_connection():
    automator = object.__new__(U2)
    automator._serial = "emulator-5554"
    automator._driver = _Driver()
    return automator


def test_start_app_prefers_launcher_from_application_namespace(monkeypatch):
    output = """2 activities found:
      sample.app.foss/sample.app.MainActivity
      sample.app.foss/com.example.debug.DisplayActivity
    """
    monkeypatch.setattr(
        "automator.u2.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout=output),
    )
    automator = _u2_without_connection()

    automator.start_app("sample.app.foss")

    assert automator._driver.calls == [("sample.app.foss", "sample.app.MainActivity")]


def test_start_app_keeps_default_behavior_when_launcher_query_fails(monkeypatch):
    monkeypatch.setattr(
        "automator.u2.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout="No activities found"),
    )
    automator = _u2_without_connection()

    automator.start_app("sample.app")

    assert automator._driver.calls == [("sample.app", None)]
