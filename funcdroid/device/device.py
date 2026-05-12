import sys
import time
from typing import Union
from loguru import logger
from ..utils.exception import*
from ..utils.proto import SwipeDirection
from ..utils.rfl.system_rfl import system_rfl
from ..model.page import Page

class Device(object):
    """
    The class describes a connected device
    """

    def __init__(self, device_serial, operating_system):
        """
        Initialize a device connection
        Args:
            device_serial (str): The serial of device.
            operating_system (str): The operating system of device.
        """
        self.serial = device_serial
        self.operating_system = operating_system
        try:
            connector_cls, automator_cls = system_rfl[self.operating_system]
            self.connector = connector_cls(self)
            self.automator = automator_cls(self)
        except OSKeyError:
            logger.error("%s is not supported" % operating_system)
            sys.exit(-1)
        self.page = None

    def __call__(self, **kwds):
        self.dump_page(refresh=True)
        return self.page(**kwds)

    def install_app(self, app):
        self.automator.install_app(app)

    def uninstall_app(self, app):
        self.automator.uninstall_app(app)

    def start_app(self, app):
        self.automator.start_app(app)

    def stop_app(self, app):
        self.automator.stop_app(app)

    def restart_app(self, app):
        self.automator.restart_app(app)
        
    def clear_app(self, app):
        self.stop_app(app)
        self.automator.clear_app(app)
        self.start_app(app)

    def restart_app_by_bundle(self, bundle):
        self.automator.restart_app_by_bundle(bundle)

    def click(self, x, y):
        return self.automator.click(x, y)

    def _click(self, node):
        (x, y) = node.attrib['center']
        return self.automator.click(x, y)

    def long_click(self, x, y):
        return self.automator.long_click(x, y)

    def _long_click(self, node):
        (x, y) = node.attrib['center']
        return self.automator.long_click(x, y)

    def drag(self, x1, y1, x2, y2, speed=2000):
        return self.automator.drag(x1, y1, x2, y2, speed)

    def swipe(self, x1, y1, x2, y2, speed=2000):
        return self.automator.swipe(x1, y1, x2, y2, speed)

    def swipe_ext(self, direction: Union[SwipeDirection, str]):
        return self.automator.swipe_ext(direction)
    
    def input(self, text):
        return self.automator.input(text)

    def dump_hierarchy(self, device=None):
        if device is None:
            device = self
        return self.automator.dump_hierarchy(device)

    def screenshot(self, path=''):
        return self.automator.screenshot(path)

    def home(self):
        self.automator.home()

    def back(self):
        self.automator.back()

    def recent(self):
        self.automator.recent()
    
    def page_info(self):
        return self.connector.page_info()
    
    def dump_page(self, device=None, refresh=False):
        if device is None:
            device = self
        if self.page == None or refresh:
            vht = self.dump_hierarchy(device=device)
            img = self.screenshot()
            info = self.page_info()
            rsc = self.resources()
            self.page = Page(vht=vht, img=img, rsc=rsc, info=info)
        return self.page

    def hop(self, dst_device_name=None, app_name=None):
        return self.automator.hop(dst_device_name, app_name)
    
    def execute(self, events):
        for event in events:
            event.execute()

    # def get_audio(self, bundle=None):
    #     return self.connector.get_audio(bundle)

    def resources(self, bundle=None):
        return self.connector.get_resources(bundle)