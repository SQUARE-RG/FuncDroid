import subprocess

import pytest

import connector.base as connector_base


class _TestConnector(connector_base.Connector):
    def run_cmd(self, extra_args):
        return self._check_output(extra_args)

    def shell(self, extra_args):
        return self.run_cmd(extra_args)


def test_device_command_timeout_is_forwarded_and_normalized(monkeypatch):
    captured = {}

    def fake_check_output(args, *, timeout):
        captured["args"] = args
        captured["timeout"] = timeout
        raise subprocess.TimeoutExpired(args, timeout)

    monkeypatch.setattr(connector_base.subprocess, "check_output", fake_check_output)
    connector = _TestConnector(command_timeout=7)

    with pytest.raises(connector_base.DeviceCommandTimeoutError) as exc_info:
        connector.run_cmd(["adb", "shell", "getprop"])

    assert captured == {
        "args": ["adb", "shell", "getprop"],
        "timeout": 7.0,
    }
    assert "timed out after 7s" in str(exc_info.value)


def test_device_command_returns_decoded_output(monkeypatch):
    monkeypatch.setattr(
        connector_base.subprocess,
        "check_output",
        lambda _args, *, timeout: b"ready\n",
    )

    assert _TestConnector(command_timeout=3).run_cmd(["adb", "devices"]) == "ready"
