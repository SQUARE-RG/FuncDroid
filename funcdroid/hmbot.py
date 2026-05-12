import time
from .app.android_app import AndroidApp
from .app.harmony_app import HarmonyApp
from .device.device import Device
from .model.event import StartAppEvent
from .explorer.llm import LLM
from .utils.proto import OperatingSystem, ExploreGoal
from .model.page import Page
from .model.ptg import PTG
from loguru import logger
import os
import shutil


class HMBot(object):
    def __init__(self, os, serials, llm_config):
        self.os = os
        self.serials = serials
        self.devices = []
        self.llm_config = llm_config
        self.app = None
        for serial in serials:
            self.devices.append(Device(serial, os))

    def explore(self, args):
        if args.os == OperatingSystem.HARMONY:
            if not args.app_path.endswith('.hap'):
                logger.error("Harmony application path must end with .hap!")
                exit(1)
            else:
                self.app = HarmonyApp(args.app_path)
        elif args.os == OperatingSystem.ANDROID:
            if not args.app_path.endswith('.apk'):
                logger.error("Android application path must end with .apk!")
                exit(1)
            else:
                self.app = AndroidApp(app_path=args.app_path)

        for device in self.devices:
            llm = LLM(device=device, url=self.llm_config['base_url'], model=self.llm_config['model'],
                      api_key=self.llm_config['api_key'])
            device.install_app(self.app)
            time.sleep(5)
            device.start_app(self.app)
            time.sleep(10)

            output_dir = args.output
            if not output_dir.endswith('/'):
                output_dir = output_dir + '/'
            output_dir = output_dir + device.serial + '/'

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Explore
            if args.hardware:
                # If the exploration goal is hardware resources, the value parameter is the resource type
                for hardware in args.hardware:
                    hardware_dir = output_dir + hardware + '/'
                    if os.path.exists(hardware_dir):
                        for item in os.listdir(hardware_dir):
                            item_path = os.path.join(hardware_dir, item)
                            if os.path.isfile(item_path):
                                os.remove(item_path)
                            elif os.path.isdir(item_path):
                                shutil.rmtree(item_path)
                    else:
                        os.makedirs(hardware_dir)
                    llm.explore(key=ExploreGoal.HARDWARE, value=hardware, max_steps=args.max_steps,
                                output_dir=hardware_dir)
            elif args.testcase:
                # If the exploration goal is test-script, the value parameter is the script path
                for index, testcase in enumerate(args.testcase):
                    with open(testcase, 'r') as file:
                        script = file.read()
                        testcase_dir = output_dir + testcase + str(index) + '/'
                        if os.path.exists(testcase_dir):
                            for item in os.listdir(testcase_dir):
                                item_path = os.path.join(testcase_dir, item)
                                if os.path.isfile(item_path):
                                    os.remove(item_path)
                                elif os.path.isdir(item_path):
                                    shutil.rmtree(item_path)
                        else:
                            os.makedirs(testcase_dir)
                        llm.explore(key=ExploreGoal.TESTCASE, value=script, max_steps=args.max_steps,
                                    output_dir=testcase_dir)
