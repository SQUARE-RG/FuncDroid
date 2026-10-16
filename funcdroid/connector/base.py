import os
import subprocess
from abc import ABC, abstractmethod
from loguru import logger


DEFAULT_DEVICE_COMMAND_TIMEOUT = max(
    1.0,
    float(os.getenv("DEVICE_COMMAND_TIMEOUT", "30")),
)


class DeviceCommandTimeoutError(TimeoutError):
    """Raised when a device CLI command exceeds its hard timeout."""


class Connector(ABC):
    def __init__(self, device=None, command_timeout=None):
        self.device = device
        self.command_timeout = (
            DEFAULT_DEVICE_COMMAND_TIMEOUT
            if command_timeout is None
            else max(1.0, float(command_timeout))
        )

    @abstractmethod
    def run_cmd(self, extra_args, timeout=None):
        pass

    @abstractmethod
    def shell(self, extra_args, timeout=None):
        pass

    @staticmethod
    def _shell_args(extra_args) -> list:
        """shell 参数：字符串沿用旧语义按空白切分；列表原样传递（不经本地 shell 解释）。"""
        if isinstance(extra_args, str):
            return ["shell", *extra_args.split()]
        if isinstance(extra_args, (list, tuple)):
            return ["shell", *[str(arg) for arg in extra_args]]
        raise TypeError("shell args must be str or list")

    def _check_output(self, args, timeout=None):
        logger.debug("run command: {}", args)
        effective_timeout = self.command_timeout if timeout is None else max(0.1, float(timeout))
        try:
            result = subprocess.check_output(
                args,
                timeout=effective_timeout,
            ).strip()
        except subprocess.TimeoutExpired as exc:
            executable = str(args[0]) if args else "device command"
            logger.error(
                "device command timed out: executable={}, timeout={}s",
                executable,
                effective_timeout,
            )
            raise DeviceCommandTimeoutError(
                f"{executable} command timed out after {effective_timeout:g}s"
            ) from exc
        if not isinstance(result, str):
            result = result.decode(errors="replace")
        logger.debug("command output: {}", result)
        return result
