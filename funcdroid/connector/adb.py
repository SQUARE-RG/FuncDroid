import os
import subprocess
from loguru import logger

from .base import (
    Connector,
    DEFAULT_DEVICE_COMMAND_TIMEOUT,
    DeviceCommandTimeoutError,
)


ADB_COMMAND_TIMEOUT = max(
    1.0,
    float(os.getenv("ADB_COMMAND_TIMEOUT", str(DEFAULT_DEVICE_COMMAND_TIMEOUT))),
)


def parse_adb_devices(output: str) -> list[str]:
    """Parse adb devices and retain only authorized, online device serials.

    Exclude offline, unauthorized, and recovery entries so backend selection cannot treat them as usable devices."""
    devices = []
    for line in str(output or "").splitlines():
        line = line.strip()
        if not line or line.startswith("List of devices attached") or line.startswith("*"):
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "device":
            devices.append(parts[0])
    return devices


class ADB(Connector):
    def __init__(self, device=None):
        if device is None:
            raise ValueError("device is required")
        super().__init__(device, command_timeout=ADB_COMMAND_TIMEOUT)
        self.serial = device.serial
        self.cmd_prefix = ["adb", "-s", self.serial]

    def run_cmd(self, extra_args, timeout=None):
        args = self._normalize_args(extra_args)
        return self._check_output(self.cmd_prefix + args, timeout=timeout)

    def shell(self, extra_args, timeout=None):
        return self.run_cmd(self._shell_args(extra_args), timeout=timeout)

    @classmethod
    def devices(cls):
        try:
            result = subprocess.check_output(
                ["adb", "devices"],
                timeout=ADB_COMMAND_TIMEOUT,
            ).strip()
        except subprocess.TimeoutExpired as exc:
            raise DeviceCommandTimeoutError(
                f"adb devices timed out after {ADB_COMMAND_TIMEOUT:g}s"
            ) from exc
        if not isinstance(result, str):
            result = result.decode()
        devices = parse_adb_devices(result)
        logger.debug("adb devices: {}", devices)
        return devices

    @staticmethod
    def _normalize_args(extra_args):
        if isinstance(extra_args, str):
            return extra_args.split()
        if isinstance(extra_args, (list, tuple)):
            return [str(arg) for arg in extra_args]
        raise TypeError("command args must be str or list")
