"""Parse runtime permissions from dumpsys and grant them with per-permission error tolerance."""

from types import SimpleNamespace

from automator.u2 import U2, parse_requested_permissions, parse_runtime_permissions

DUMPSYS_SAMPLE = """\
Packages:
  Package [com.example.app] (a1b2c3):
    userId=10123
    requested permissions:
      android.permission.INTERNET
      android.permission.CAMERA
      android.permission.ACCESS_FINE_LOCATION
    install permissions:
      android.permission.INTERNET: granted=true
    runtime permissions:
      android.permission.CAMERA: granted=false, flags=[ USER_SET|USER_FIXED]
      android.permission.ACCESS_FINE_LOCATION: granted=true
      android.permission.POST_NOTIFICATIONS: granted=false
"""

REQUESTED_ON_ONE_LINE_SAMPLE = """\
    declared permissions:
      com.example.app.permission.CUSTOM: granted=true
    requested permissions: [android.permission.INTERNET, android.permission.RECORD_AUDIO]
    install permissions:
"""


def test_runtime_permission_parsing_only_keeps_changeable_entries():
    # Parse only runtime entries with granted status, excluding ordinary and custom permissions.
    assert parse_runtime_permissions(DUMPSYS_SAMPLE) == [
        "android.permission.CAMERA",
        "android.permission.ACCESS_FINE_LOCATION",
        "android.permission.POST_NOTIFICATIONS",
    ]


def test_requested_permission_parsing_handles_single_line_list():
    assert parse_requested_permissions(REQUESTED_ON_ONE_LINE_SAMPLE) == [
        "android.permission.INTERNET",
        "android.permission.RECORD_AUDIO",
    ]


def test_runtime_permission_parsing_of_unrelated_output_is_empty():
    assert parse_runtime_permissions("Package [x] :\n  userId=1\n") == []
    assert parse_requested_permissions("no permission section here") == []


def test_runtime_section_ignores_bracket_lists_and_bare_subheaders():
    # Device output may also contain bracketed lists and enabledComponents subsections.
    # Neither represents a permission name.
    dump = (
        "  runtime permissions:\n"
        "    android.permission.CAMERA: granted=false\n"
        "    enabledComponents: [com.example/.Main]\n"
    )
    assert parse_runtime_permissions(dump) == ["android.permission.CAMERA"]


def test_runtime_section_stops_before_the_next_declaration():
    # Stop at the next peer declaration so later sections do not contribute permissions.
    dump = (
        "Package [com.example.app]:\n"
        "  runtime permissions:\n"
        "    android.permission.CAMERA: granted=false\n"
        "  install permissions:\n"
        "    android.permission.INTERNET: granted=true\n"
    )
    assert parse_runtime_permissions(dump) == ["android.permission.CAMERA"]


def _fake_device(run_results, calls):
    def fake_run(command, **_kwargs):
        calls.append(command)
        return run_results[len(calls) - 1]

    return fake_run


def test_granting_is_per_permission_and_never_fatal(monkeypatch):
    granted_process = SimpleNamespace(returncode=0, stdout="", stderr="")
    normal_permission = SimpleNamespace(
        returncode=1, stdout="Operation not allowed: not a changeable permission type", stderr=""
    )
    unexpected_failure = SimpleNamespace(returncode=1, stdout="", stderr="device offline")

    device = object.__new__(U2)
    device._serial = "device-1"
    monkeypatch.setattr(U2, "_package_dump", lambda self, package: DUMPSYS_SAMPLE)
    calls: list[list[str]] = []
    monkeypatch.setattr("automator.u2.subprocess.run", _fake_device(
        [granted_process, normal_permission, unexpected_failure], calls
    ))

    report = device.grant_app_permissions("com.example.app")

    assert report["granted"] == ["android.permission.CAMERA"]
    assert report["missing"] == ["android.permission.ACCESS_FINE_LOCATION"]
    assert report["failed"] == ["android.permission.POST_NOTIFICATIONS"]
    assert report["unavailable"] is False
    assert [call[4] for call in calls] == ["pm"] * 3


def test_missing_dumpsys_output_reports_unavailable_instead_of_guessing(monkeypatch):
    device = object.__new__(U2)
    monkeypatch.setattr(U2, "_package_dump", lambda self, package: "")

    report = device.grant_app_permissions("com.example.app")

    assert report["unavailable"] is True
    assert not report["granted"] and not report["failed"]
