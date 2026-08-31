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


class ADB(Connector):
    def __init__(self, device=None):
        if device is None:
            raise ValueError("device is required")
        super().__init__(device, command_timeout=ADB_COMMAND_TIMEOUT)
        self.serial = device.serial
        self.cmd_prefix = ["adb", "-s", self.serial]

    def run_cmd(self, extra_args):
        args = self._normalize_args(extra_args)
        return self._check_output(self.cmd_prefix + args)

    def shell(self, extra_args):
        if not isinstance(extra_args, str):
            raise TypeError("shell args must be str")
        return self.run_cmd(f"shell {extra_args}")

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
        devices = []
        for line in result.splitlines():
            line = line.strip()
            if not line or line.startswith("List of devices attached"):
                continue
            serial = line.split()[0]
            devices.append(serial)
        logger.debug("adb devices: {}", devices)
        return devices

    @staticmethod
    def _normalize_args(extra_args):
        if isinstance(extra_args, str):
            return extra_args.split()
        if isinstance(extra_args, list):
            return extra_args
        raise TypeError("command args must be str or list")
