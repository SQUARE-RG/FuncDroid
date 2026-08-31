import subprocess
import sys
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

FUNCDROID_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FUNCDROID_DIR))

import main
from connector.adb import ADB
from workflow.account_safety import account_destructive_intent
from workflow.nodes.scenario_exec import _finished_declares_failure
from workflow.nodes.prompt import (
    BOOTSTRAP_FFG_SYSTEM_PROMPT,
    BUG_DETECT_SYSTEM_PROMPT,
    FFG_AUDIT_SYSTEM_PROMPT,
    FFG_UPDATE_SYSTEM_PROMPT,
    PAGE_AUDIT_SYSTEM_PROMPT,
    SCENARIO_EXEC_SYSTEM_PROMPT_TEMPLATE,
    UNIFIED_SCENARIO_PLAN_SYSTEM_PROMPT,
)
from workflow.nodes.scenario_plan import (
    _audit_scenario_is_executable,
    _parameterized_scenario_family,
    _quoted_action_control,
    _unquoted_audit_control,
)


class DeviceSelectionTests(unittest.TestCase):
    def test_single_device_is_selected_automatically(self):
        args = Namespace(device_id=None)
        with patch.object(main.ADB, "devices", return_value=["android-1"]), patch.object(
            main.HDC, "devices", return_value=[]
        ):
            main._ensure_device_id(args)
        self.assertEqual(args.device_id, "android-1")

    def test_multiple_devices_require_an_explicit_id(self):
        args = Namespace(device_id=None)
        with patch.object(main.ADB, "devices", return_value=["android-1", "android-2"]), patch.object(
            main.HDC, "devices", return_value=[]
        ):
            with self.assertRaisesRegex(ValueError, "Multiple devices detected"):
                main._ensure_device_id(args)


class AdbDeviceParsingTests(unittest.TestCase):
    def test_only_ready_devices_are_returned(self):
        output = (
            b"List of devices attached\n"
            b"ready-1\tdevice product:test\n"
            b"pending-1\tunauthorized\n"
            b"offline-1\toffline\n"
        )
        with patch.object(subprocess, "check_output", return_value=output):
            self.assertEqual(ADB.devices(), ["ready-1"])


class EnglishScenarioCompatibilityTests(unittest.TestCase):
    def test_every_model_role_requires_english_output(self):
        prompts = (
            UNIFIED_SCENARIO_PLAN_SYSTEM_PROMPT,
            BOOTSTRAP_FFG_SYSTEM_PROMPT,
            FFG_AUDIT_SYSTEM_PROMPT,
            SCENARIO_EXEC_SYSTEM_PROMPT_TEMPLATE,
            FFG_UPDATE_SYSTEM_PROMPT,
            BUG_DETECT_SYSTEM_PROMPT,
            PAGE_AUDIT_SYSTEM_PROMPT,
        )
        for prompt in prompts:
            self.assertIn("[Mandatory Output Language]", prompt)
            self.assertIn("Preserve visible UI labels", prompt)

    def test_english_audit_scenario_is_executable(self):
        self.assertTrue(_audit_scenario_is_executable('Click the "Settings" button'))

    def test_quoted_action_control_is_extracted(self):
        self.assertEqual(_quoted_action_control('Click the "Settings" button'), "settings")

    def test_unquoted_action_control_is_extracted(self):
        self.assertEqual(
            _unquoted_audit_control("Open the Settings menu item and observe the result"),
            "settings",
        )

    def test_parameterized_english_control_is_normalized(self):
        self.assertEqual(
            _parameterized_scenario_family('On the "Home" page, select "Recent" tab'),
            ("home", "select", "tab"),
        )

    def test_finished_status_distinguishes_success_and_failure(self):
        self.assertTrue(_finished_declares_failure("Goal unreachable: Settings not found"))
        self.assertFalse(_finished_declares_failure("Goal achieved"))

    def test_account_guard_is_package_independent(self):
        self.assertTrue(account_destructive_intent("\u9000\u51fa\u767b\u5f55"))
        self.assertTrue(account_destructive_intent("Delete the account"))
        self.assertFalse(account_destructive_intent("Open account settings"))


if __name__ == "__main__":
    unittest.main()
