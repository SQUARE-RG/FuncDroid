from hmbot.device.device import Device
from hmbot.utils.proto import OperatingSystem
from hmbot.utils.utils import *
from hmbot.app.android_app import AndroidApp
from hmbot.explorer.explorer import Explorer


s = get_android_available_devices()[0] 
device = Device(s, OperatingSystem.ANDROID)
app = AndroidApp(app_path=r"C:\Users\23314\Desktop\FuncDroid\experiment\geohashdroid\apk\geohashdroid-0.9.4-#73.apk")
explorer = Explorer(device, app)
# device.clear_app(app)

# explorer.explore(output_dir=r"C:\Users\23314\Desktop\FuncDroid\experiment\geohashdroid\output")
explorer.calculate_activity_coverage(r"C:\Users\23314\Desktop\FuncDroid\experiment\geohashdroid\output\ptg_report_20251212_164211.json")
# explorer.build_FDG(r"C:\Users\23314\Desktop\FuncDroid\experiment\geohashdroid\output\ptg_report_20251210_170500.json")

# explorer.build_FDG_with_dependency(r"C:\Users\23314\Desktop\FuncDroid\experiment\geohashdroid\output\ptg_report_20251204_120851.json", 
#                                         r"C:\Users\23314\Desktop\FuncDroid\experiment\geohashdroid\output\fdg.json")
# bug_explorer.single_function_test(r"C:\Users\23314\Desktop\FuncDroid\experiment\geohashdroid\output\ptg_report_20251109_105248.json", 
#                                  r"C:\Users\23314\Desktop\FuncDroid\experiment\geohashdroid\output\fdg_include_enhanced.json")
# bug_explorer.muti_function_test(r"C:\Users\23314\Desktop\FuncDroid\experiment\geohashdroid\output\ptg_report_20251109_105248.json", 
#                                  r"C:\Users\23314\Desktop\FuncDroid\experiment\geohashdroid\output\fdg_include_enhanced.json")
# bug_explorer.activity_coverage_exploration(r"C:\Users\23314\Desktop\FuncDroid\experiment\geohashdroid\output\ptg_report_20251109_105248.json")

