# Multilingual UI and model-response fixtures use Unicode escapes to preserve their exact text.
"""Tests for the exploration-boundary rule engine."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workflow import exploration_boundary as boundary  # noqa: E402


class ExplorationBoundaryRuleLoadingTest(unittest.TestCase):
    def setUp(self):
        boundary.reset_cache()
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.rules_path = Path(self.tmpdir.name) / "boundary_rules.json"
        self.addCleanup(os.environ.pop, "BOUNDARY_RULES_FILE", None)
        os.environ["BOUNDARY_RULES_FILE"] = str(self.rules_path)

    def _write_rules(self, payload):
        self.rules_path.write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        boundary.reset_cache()

    def test_missing_file_degrades_to_disabled(self):
        self.assertFalse(boundary.rule_for("com.any.app").enabled)
        self.assertIsNone(boundary.check_boundary("com.any.app", "AppBrandUI"))
        self.assertEqual(boundary.enabled_packages(), [])

    def test_invalid_json_degrades_to_disabled(self):
        self.rules_path.write_text("{not json", encoding="utf-8")
        boundary.reset_cache()
        self.assertFalse(boundary.rule_for("com.tencent.mm").enabled)

    def test_rule_loading_and_activity_hit(self):
        self._write_rules({
            "com.tencent.mm": {
                "enabled": True,
                "label": "\u5c0f\u7a0b\u5e8f/\u7b2c\u4e09\u65b9\u670d\u52a1",
                "activity_markers": ["appbrand", "lite"],
                "process_markers": [":appbrand"],
                "exit_action": "press_back",
            }
        })
        rule = boundary.rule_for("com.tencent.mm")
        self.assertTrue(rule.enabled)
        self.assertEqual(rule.label, "\u5c0f\u7a0b\u5e8f/\u7b2c\u4e09\u65b9\u670d\u52a1")

        hit = boundary.check_boundary("com.tencent.mm", "com.tencent.mm.plugin.appbrand.ui.AppBrandUI00")
        self.assertIsNotNone(hit)
        self.assertEqual(hit.via, "activity")
        self.assertEqual(hit.matched, "appbrand")

    def test_unconfigured_package_never_hits(self):
        self._write_rules({
            "com.tencent.mm": {
                "enabled": True,
                "activity_markers": ["appbrand"],
            }
        })
        self.assertIsNone(boundary.check_boundary("com.other.app", "AppBrandUI"))
        self.assertIsNone(boundary.check_boundary("", "AppBrandUI"))

    def test_process_snapshot_fallback(self):
        self._write_rules({
            "com.tencent.mm": {
                "enabled": True,
                "activity_markers": ["appbrand"],
                "process_markers": [":appbrand"],
            }
        })
        # When Activity is readable, a resident :appbrand process does not establish a boundary crossing.
        self.assertIsNone(boundary.check_boundary(
            "com.tencent.mm",
            activity="com.tencent.mm.ui.LauncherUI",
            process_snapshot="u0_a12 1234 com.tencent.mm:appbrand0",
        ))
        # Use process snapshots only when Activity is unreadable.
        hit = boundary.check_boundary(
            "com.tencent.mm",
            activity="",
            process_snapshot="u0_a12 1234 com.tencent.mm:appbrand0",
        )
        self.assertIsNotNone(hit)
        self.assertEqual(hit.via, "process")
        self.assertEqual(hit.matched, ":appbrand")

    def test_enabled_false_is_equivalent_to_unconfigured(self):
        self._write_rules({
            "com.tencent.mm": {"enabled": False, "activity_markers": ["appbrand"]},
        })
        self.assertIsNone(boundary.check_boundary("com.tencent.mm", "AppBrandUI"))
        self.assertEqual(boundary.enabled_packages(), [])

    def test_prompt_hint_uses_label(self):
        self._write_rules({
            "com.demo.app": {
                "enabled": True,
                "label": "\u5185\u5d4c\u6e38\u620f\u4e0eH5\u9875\u9762",
                "activity_markers": ["minigame"],
            }
        })
        hint = boundary.rule_for("com.demo.app").to_prompt_hint()
        self.assertIn("\u5185\u5d4c\u6e38\u620f\u4e0eH5\u9875\u9762", hint)
        self.assertEqual(boundary.rule_for("com.none.app").to_prompt_hint(), "")

    def test_enabled_packages_list(self):
        self._write_rules({
            "default": {"enabled": False},
            "com.a.app": {"enabled": True, "activity_markers": ["x"]},
            "com.b.app": {"enabled": False},
        })
        self.assertEqual(boundary.enabled_packages(), ["com.a.app"])


if __name__ == "__main__":
    unittest.main()
