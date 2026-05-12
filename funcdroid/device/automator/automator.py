from abc import ABC, abstractmethod

class Automator(ABC):
    """
    The interface describes an automator (u2 or h2)
    """
    @abstractmethod
    def __init__(self, device):
        """
        Initialize an automator 

        Args:
            device (Device): The device to connect.
        """
        pass

    @abstractmethod
    def install_app(self, app):
        """
        Install an app.
        """
        pass

    @abstractmethod
    def uninstall_app(self, app):
        """
        Uninstall an app.
        """
        pass

    @abstractmethod
    def start_app(self, app):
        """
        Start an app.
        """
        pass

    @abstractmethod
    def stop_app(self, app):
        """
        Stop an app.
        """
        pass

    @abstractmethod
    def restart_app(self, app):
        """
        Restart an app.
        """
        pass

    @abstractmethod
    def click(self, x, y):
        """
        Click at (x, y)

        Args:
            x (float): The X coordinate.
            y (float): The Y coordinate.
        """
        pass

    @abstractmethod
    def long_click(self, x, y):
        """
        Long click at (x, y)

        Args:
            x (float): The X coordinate.
            y (float): The Y coordinate.
        """
        pass

    @abstractmethod
    def drag(self, x1, y1, x2, y2, speed=2000):
        """
        Drag from (x1, y1) to (x2, y2)

        Args:
            x1 (float): The start X coordinate.
            y1 (float): The start Y coordinate.
            x2 (float): The end X coordinate.
            y2 (float): The end Y coordinate.
            speed (int, optional): The drag speed in pixels per second. Default is 2000. Range: 200-40000,
            If not within the range, set to default value of 2000.
        """
        pass

    @abstractmethod
    def swipe(self, x1, y1, x2, y2, speed=2000):
        """
        Swipe from (x1, y1) to (x2, y2)

        Args:
            x1 (float): The start X coordinate.
            y1 (float): The start Y coordinate.
            x2 (float): The end X coordinate.
            y2 (float): The end Y coordinate.
            speed (int, optional): The drag speed in pixels per second. Default is 2000. Range: 200-40000,
            If not within the range, set to default value of 2000.
        """
        pass

    @abstractmethod
    def swipe_ext(self, direction, scale):
        """
        Swipe to direction.

        Args:
            direction (str): one of "left", "right", "up", "bottom" or SwipeDirection.LEFT
            scale (float): percent of swipe, range (0, 1.0]
        """
        pass

    @abstractmethod
    def input(self, node, text):
        pass

    @abstractmethod
    def dump_hierarchy(self, device):
        """
        Dump the UI hierarchy of the device screen.

        Returns:
            VHT: The dumped UI hierarchy as a object of VHT.
        """
        pass

    @abstractmethod
    def screenshot(self, path=''):
        """
        Take a screenshot of the device display.

        Args:
            path (str): The local path to save the screenshot.

        Returns:
            str: The path where the screenshot is saved.
        """
        pass

    # @abstractmethod
    # def display_rotation(self):
    #     """
    #     Get display rotation about the device.

    #     Returns:
    #         w, h (int, int): The width and height of the display.
    #     """
    #     pass

    @abstractmethod
    def display_info(self, refresh=True):
        """
        Get display info about the device.

        Returns:
            Dict: The display info as a dict.
        """
        pass

    @abstractmethod
    def home(self):
        """
        Click Home button.
        """
        pass

    @abstractmethod
    def back(self):
        """
        Click Back button.
        """
        pass
    
    @abstractmethod
    def recent(self):
        """
        Click Recent button.
        """
        pass

    @abstractmethod
    def hop(self, dst_device_name=None, app_name=None):
        """
        Hop a app to a device.
        """
        pass

    @abstractmethod
    def identify(self, node):
        pass