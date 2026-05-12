from .app import App
from loguru import logger
from androguard.core.apk import APK

class AndroidApp(App):
    def __init__(self, app_path='', device=None):
        self.app_path = app_path
        apk = APK(self.app_path)
        self.package_name = apk.get_package()
        self.entry_ability = apk.get_main_activity()
        self.main_page = apk.get_main_activity()
        self.abilities = apk.get_activities()
        self.app_name = apk.get_app_name()

        # # 获取原始 XML 字符串
        # manifest_xml = apk.get_android_manifest_xml()

        # # 转成字符串
        # from lxml import etree
        # manifest_str = etree.tostring(manifest_xml, encoding="utf-8", pretty_print=True).decode("utf-8")

        # # 保存到文件
        # with open("AndroidManifest.xml", "w", encoding="utf-8") as f:
        #     f.write(manifest_str)

        # print(f"Manifest saved to AndroidManifest.xml")