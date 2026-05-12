from abc import ABC, abstractmethod

class Connector(ABC):
    """
    this interface describes a connector (ADB or HDC)
    """
    @abstractmethod
    def __init__(self, device=None):
        pass

    @abstractmethod
    def run_cmd(self, extra_args):
        """
        Run a command and return the output

        Args:
            extra_args (str): arguments to run in adb or hdc

        Returns:
            str: output of command
        """
        pass

    @abstractmethod
    def shell(self, extra_args):
        """
        Run a shell-command and return the output

        Args:
            extra_args (str): arguments to run in adb or hdc

        Returns:
            str: output of command
        """
        pass

    @abstractmethod
    def shell_grep(self, extra_args, grep_args):
        """
        Run a shell-command with grep command and return the output

        Args:
            extra_args (str): arguments to run in adb or hdc
            grep_args (str): arguments to grep with shell command in adb or hdc

        Returns:
            str: output of command
        """
        pass

    @abstractmethod
    def page_info(self):
        """
        Run a shell-command and return the current ability (including its bundle name and app name).

        Returns:
            (str, str, str): output of the current (app name, bundle name, ability name).
        """
        pass

    @abstractmethod
    def get_resources(self, bundle):
        """
        Run a shell-command and return the resource status of the current app (including audio, camera).

        Returns:
            Resource(audio=AudioInfo, camera=CameraInfo),
            output of the resource status of the current app.
        """
        pass

    @abstractmethod
    def get_audio(self, bundle):
        """
        Run a shell-command and return the audio status of the current app.

        Returns:
            AudioInfo(type: AudioType, stat: Status),
            output of the audio info of the current app.
        """
        pass

    @abstractmethod
    def get_camera(self):
        """
        Run a shell-command and return the camera status of the current app.

        Returns:
            CameraInfo(type: CameraType, stat: Status),
            output of the camera info of the current app.
        """
        pass


