from .automator import Automator
from hmbot.model.vht import VHTParser
from hmbot.utils.proto import SwipeDirection, DisplayInfo
from hmbot.app.app import App
from hmdriver2.driver import Driver
from hmdriver2.proto import KeyCode
from loguru import logger
import uuid, os, shutil, logging
h2_logger = logging.getLogger('hmdriver2')
h2_logger.setLevel(logging.CRITICAL)

class H2(Automator):
    def __init__(self, device):
        self._serial = device.serial
        self._driver = Driver(self._serial)
        self._display_info = None
        logger.debug("hmdriver2 is connected to device:%s" %(self._serial))

    def install_app(self, app):
        if isinstance(app, App):
            self._driver.install_app(app.app_path)
        else:
            raise TypeError('expected an App, not %s' % type(app).__name__)

    def uninstall_app(self, app):
        if isinstance(app, App):
            self._driver.uninstall_app(app.package_name)
        else:
            raise TypeError('expected an App, not %s' % type(app).__name__)

    def start_app(self, app):
        if isinstance(app, App):
            self._driver.start_app(app.package_name)
        else:
            raise TypeError('expected an App, not %s' % type(app).__name__)

    def stop_app(self, app):
        if isinstance(app, App):
            self._driver.stop_app(app.package_name)
        else:
            raise TypeError('expected an App, not %s' % type(app).__name__)

    def restart_app(self, app):
        self.stop_app(app)
        self.start_app(app)
    
    def click(self, x, y):
        return self._driver.click(x, y)

    def long_click(self, x, y):
        return self._driver.long_click(x, y)

    def drag(self, x1, y1, x2, y2, speed):
        return self._driver.swipe(x1, y1, x2, y2, speed)

    def swipe(self, x1, y1, x2, y2, speed):
        return self._driver.swipe(x1, y1, x2, y2, speed)

    def swipe_ext(self, direction, scale=0.3):
        if direction == SwipeDirection.LEFT :
            self._driver.swipe(0.5, 0.5, 0.5-scale, 0.5, 500)
        elif direction == SwipeDirection.RIGHT :
            self._driver.swipe(0.5, 0.5, 0.5+scale, 0.5, 500)
        elif direction == SwipeDirection.UP :
            self._driver.swipe(0.5, 0.5, 0.5, 0.5-scale, 500)
        elif direction == SwipeDirection.DOWN :
            self._driver.swipe(0.5, 0.5, 0.5, 0.5+scale, 500)

    def input(self, node, text):
        id = node.attribute['id']
        if id:
            self._driver(id=id).input_text(text)

    def dump_hierarchy(self, device):
        return VHTParser._parse_hdc_json(self._driver.dump_hierarchy(), device)

    def screenshot(self, path=''):
        if isinstance(path, str):
            _uuid = uuid.uuid4().hex
            _tmp_path = f"_tmp_{_uuid}.jpeg"
            from hmbot.utils.cv import read
            img = read(self._driver.screenshot(_tmp_path))
            if path:
                shutil.copyfile(_tmp_path, path)
            os.remove(_tmp_path)
            return img
        else:
            raise TypeError('expected an str, not %s' % type(path).__name__)
    
    # def display_size(self):
    #     return self._driver.display_size

    # def display_rotation(self):
    #     return self._driver.display_rotation

    def display_info(self, refresh=True):
        if self._display_info is None or refresh:
            info = self._driver.device_info
            self._display_info = DisplayInfo(sdk=info.sdkVersion,
                                             width=info.displaySize[0],
                                             height=info.displaySize[1],
                                             rotation=info.displayRotation)
        return self._display_info

    def home(self):
        self._driver.go_home()

    def back(self):
        self._driver.go_back()

    def recent(self):
        self._driver.swipe(0.5, 2710/2720, 0.5, 2400/2720, 500)

    def hop(self, dst_device_name=None, app_name=None):
        pass

    def identify(self, node):
        pass