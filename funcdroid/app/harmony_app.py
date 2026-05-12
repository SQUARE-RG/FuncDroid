from .app import App
from ..device.device import Device
from loguru import logger

class HarmonyApp(App):
    def __init__(self, app_path='', device=None):
        if app_path == '' and isinstance(device, Device):
            infos = device.current_ability()
            self.app_path = ''
            self.name = infos['app']
            self.bundle = infos['bundle']
            self.entry = infos['ability']
            self.main_page = device.dump_hierarchy().roots()[0].attribute['page']
            return
        if app_path != '':
            self.app_path = app_path
            self.package_name = ''
            self.entry = ''
            self.main_page = ''
            return