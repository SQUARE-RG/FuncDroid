from .automator import Automator
from hmbot.model.vht import VHTParser, VHT, VHTNode
from hmbot.utils.proto import SwipeDirection, DisplayInfo, DisplayRotation, SystemKey
from hmbot.app.app import App
from loguru import logger
import uuid, os, shutil
import uiautomator2
import time

class U2(Automator):
    def __init__(self, device):
        self._serial = device.serial
        self._driver = uiautomator2.connect(self._serial)
        self._display_info = None
        logger.debug("uiautomator2 is connected to device:%s" %(self._serial))

    def install_app(self, app):
        if isinstance(app, App):
            self._driver.app_install(app.app_path)
        else:
            raise TypeError('expected an App, not %s' % type(app).__name__)

    def uninstall_app(self, app):
        pass
        # if isinstance(app, App):
        #     self._driver.uninstall_app(app.package_name)
        # else:
        #     raise TypeError('expected an App, not %s' % type(app).__name__)

    def start_app(self, app):
        if isinstance(app, App):
            self._driver.app_start(app.package_name)
        else:
            raise TypeError('expected an App, not %s' % type(app).__name__)

    def stop_app(self, app):
        if isinstance(app, App):
            self._driver.app_stop(app.package_name)
        else:
            raise TypeError('expected an App, not %s' % type(app).__name__)

    def restart_app(self, app):
        self.stop_app(app)
        self.start_app(app)

    def clear_app(self, app):
        if isinstance(app, App):
            self._driver.app_clear(app.package_name)
        else:
            raise TypeError('expected an App, not %s' % type(app).__name__)

    def restart_app_by_bundle(self, bundle):
        self._driver.app_stop(bundle)
        self._driver.app_start(bundle)

    def click(self, x, y):
        self.display_info(refresh=True)
        width = self._display_info.width
        height = self._display_info.height
        if x < 1 and y < 1:
            x = x * width
            y = y * height
        return self._driver.click(x, y)

    def long_click(self, x, y):
        return self._driver.long_click(x, y, 1.5)

    def drag(self, x1, y1, x2, y2, duration=0.5):
        self.display_info(refresh=True)
        width = self._display_info.width
        height = self._display_info.height
        if x1 < 1 and y1 < 1:
            x1 = x1*width
            y1 = y1*height
        if x2 < 1 and y2 < 1:
            x2 = x2*width
            y2 = y2*height
        return self._driver.drag(x1, y1, x2, y2, duration)

    def swipe(self, x1, y1, x2, y2, duration=0.5):
        if x1 < 1 and y1 < 1 and x2 < 1 and y2 < 1:
            self.display_info(refresh=True)
            width = self._display_info.width
            height = self._display_info.height
            return self._driver.swipe(x1 * width, y1 * height, x2 * width, y2 * height, duration)
        else:
            return self._driver.swipe(x1, y1, x2, y2, duration)

    def swipe_ext(self, direction, scale=0.4):
        if direction == SwipeDirection.LEFT :
            self.swipe(0.5, 0.5, 0.5-scale, 0.5)
        elif direction == SwipeDirection.RIGHT :
            self.swipe(0.5, 0.5, 0.5+scale, 0.5)
        elif direction == SwipeDirection.UP :
            self.swipe(0.5, 0.5, 0.5, 0.5-scale)
        elif direction == SwipeDirection.DOWN :
            self.swipe(0.5, 0.5, 0.5, 0.5+scale)

    # def input(self, node, text):
    #     u2_nodes = []
    #     if len(node._compressed) == 0:
    #         u2_nodes.append(self.identify(node=node, type='android.widget.EditText', enabled="true", focused="true"))
    #     else:
    #         for child in node._compressed:
    #             u2_nodes.extend(self.identify(node=child, type='android.widget.EditText', enabled="true", focused="true"))
    #     for u2_node in u2_nodes:
    #         try:
    #             u2_node.set_text(text)
    #         except uiautomator2.UiObjectNotFoundError:
    #             self.identify(node=node, type='android.widget.AutoCompleteTextView', enabled="true", focused="true").set_text(text)

    def input(self, text):
        self._driver.send_keys(text, True)

    def dump_hierarchy(self, device):
        root = VHTParser._parse_adb_xml(self._driver.dump_hierarchy(compressed=False), device)._root
        # root_child = max(root._children, key=lambda child:
        #     (child.attribute['bounds'][1][0] - child.attribute['bounds'][0][0]) * (child.attribute['bounds'][1][1] - child.attribute['bounds'][0][1]))
        # root_child.attribute['type'] = 'root'
        # root_child.attribute['page'] = self._current()['activity']
        return VHT(root)

    def screenshot(self, path=''):
        img = self._driver.screenshot(format='opencv')
        if isinstance(path, str):
            if path:
                from ..cv import write
                write(path, img)
            return img
        else:
            raise TypeError('expected an str, not %s' % type(path).__name__)
    
    def display_info(self, refresh=True):
        if self._display_info is None or refresh:
            info = self._driver.info
            self._display_info = DisplayInfo(sdk=info['sdkInt'],
                                             width=info['displayWidth'],
                                             height=info['displayHeight'],
                                             rotation=info['displayRotation'])
        return self._display_info

    def home(self):
        self._driver.press(SystemKey.HOME)

    def back(self):
        self._driver.press(SystemKey.BACK)

    def recent(self):
        self._driver.press(SystemKey.RECENT)

    def _current(self):
        return self._driver.app_current()

    def hop(self, dst_device_name=None, app_name=None):
        if not dst_device_name:
            return False
        self.recent()
        time.sleep(1)
        self.swipe_ext('left')
        time.sleep(1)

        swipe_time = 0
        if app_name:
            while True:
                if swipe_time > 10:
                    return False
                vht = self.dump_hierarchy(device=self)
                cnode = vht(text=app_name)
                if len(cnode):
                    break
                self.swipe_ext('right')
                time.sleep(1)
                swipe_time += 1

        vht = self.dump_hierarchy(device=self)
        dnode = (vht(text=dst_device_name))[0]
        [dx, dy] = dnode.attribute.get('center')
        time.sleep(1)
        self.drag(0.5, 0.5, dx, dy, 1.0)
        print(f'hop: {app_name} to {dst_device_name}')
        return True

    def identify(self, node, **kwds):
        id = node.attribute['id']
        type = node.attribute['type']
        text = node.attribute['text']
        if 'type' in kwds:
            type = kwds['type']
        if 'enabled' in kwds:
            enabled = kwds['enabled']
        if 'focused' in kwds:
            focused = kwds['focused']
        return self._driver(resourceId=id, className=type, text=text, enabled=enabled, focused=focused)
