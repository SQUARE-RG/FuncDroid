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
    def run_cmd(self, extra_args):
        pass

    @abstractmethod
    def shell(self, extra_args):
        pass

    def _check_output(self, args):
        logger.debug("run command: {}", args)
        try:
            result = subprocess.check_output(
                args,
                timeout=self.command_timeout,
            ).strip()
        except subprocess.TimeoutExpired as exc:
            executable = str(args[0]) if args else "device command"
            logger.error(
                "device command timed out: executable={}, timeout={}s",
                executable,
                self.command_timeout,
            )
            raise DeviceCommandTimeoutError(
                f"{executable} command timed out after {self.command_timeout:g}s"
            ) from exc
        if not isinstance(result, str):
            result = result.decode()
        logger.debug("command output: {}", result)
        return result
