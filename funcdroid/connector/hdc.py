import subprocess
from loguru import logger

from .base import (
    Connector,
    DEFAULT_DEVICE_COMMAND_TIMEOUT,
    DeviceCommandTimeoutError,
)


def parse_hdc_targets(output: str) -> list[str]:
    """解析 `hdc list targets` 输出；无设备时 hdc 打印 [Empty]，不能当成序列号。"""
    return [
        line.strip()
        for line in str(output or "").splitlines()
        if line.strip() and line.strip() != "[Empty]"
    ]


class HDC(Connector):
    def __init__(self, device=None):
        if device is None:
            raise ValueError("device is required")
        super().__init__(device)
        self.serial = device.serial
        self.cmd_prefix = ["hdc", "-t", self.serial]

    def run_cmd(self, extra_args, timeout=None):
        args = self._normalize_args(extra_args)
        return self._check_output(self.cmd_prefix + args, timeout=timeout)

    def shell(self, extra_args, timeout=None):
        return self.run_cmd(self._shell_args(extra_args), timeout=timeout)

    @classmethod
    def devices(cls):
        try:
            result = subprocess.check_output(
                ["hdc", "list", "targets"],
                timeout=DEFAULT_DEVICE_COMMAND_TIMEOUT,
            ).strip()
        except subprocess.TimeoutExpired as exc:
            raise DeviceCommandTimeoutError(
                f"hdc list targets timed out after {DEFAULT_DEVICE_COMMAND_TIMEOUT:g}s"
            ) from exc
        if not isinstance(result, str):
            result = result.decode()
        devices = parse_hdc_targets(result)
        logger.debug("hdc devices: {}", devices)
        return devices

    @staticmethod
    def _normalize_args(extra_args):
        if isinstance(extra_args, str):
            return extra_args.split()
        if isinstance(extra_args, (list, tuple)):
            return [str(arg) for arg in extra_args]
        raise TypeError("command args must be str or list")
