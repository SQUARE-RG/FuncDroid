"""Startup, scheduling, and multilingual account-session regressions."""

import contextlib
import io
import subprocess
import unittest
from unittest.mock import patch

import main
from connector.adb import ADB
from model.flow_graph import FunctionalFlowGraph, FunctionDependency, FunctionalityNode, Page
from workflow.account_safety import account_session_action_violation
from workflow.nodes.scenario_plan import (
    PLAN_BREADTH_MAX_SCENARIOS,
    _candidate_inter_functional_pairs,
    _intra_functionality_phase,
    _required_depth_scenario_count,
)


class DeviceSelectionTests(unittest.TestCase):
    def test_missing_device_id_is_rejected_before_runtime(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                main._build_parser().parse_args([])
        self.assertEqual(raised.exception.code, 2)

    def test_explicit_device_preserves_defect_app_data(self):
        args = main._build_parser().parse_args(["--device-id", "android-1"])
        self.assertEqual(args.device_id, "android-1")
        self.assertEqual(args.mode, "defect")
        self.assertFalse(args.reset_app_data)
        self.assertTrue(args.bug_detection)



    def test_resume_cannot_clear_app_data(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                main._build_parser().parse_args([
                    "--device-id", "android-1", "--package-name", "com.example.app",
                    "--resume-output", "previous-run", "--reset-app-data",
                ])
        self.assertEqual(raised.exception.code, 2)


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


class SchedulingTests(unittest.TestCase):
    def test_failed_depth_attempt_does_not_satisfy_depth_target(self):
        graph = FunctionalFlowGraph()
        node = FunctionalityNode(id="func_1", page_ids=["page_1"])
        graph.functions[node.id] = node
        graph.pages["page_1"] = Page(id="page_1", controls=["Save"])
        self.assertEqual(_intra_functionality_phase(graph, node), "breadth")
        node.intra_attempt_count = PLAN_BREADTH_MAX_SCENARIOS
        self.assertEqual(_intra_functionality_phase(graph, node), "depth")
        node.depth_scenario_count = _required_depth_scenario_count(graph, node)
        node.intra_attempt_count += node.depth_scenario_count
        self.assertEqual(_intra_functionality_phase(graph, node), "depth")
        node.successful_depth_scenario_count = node.depth_scenario_count
        self.assertEqual(_intra_functionality_phase(graph, node), "breadth")

    def test_shared_resources_keep_both_directions_without_explicit_dependency(self):
        graph = FunctionalFlowGraph()
        for node_id in ("func_1", "func_2"):
            graph.functions[node_id] = FunctionalityNode(id=node_id, resource_variables=["record"])
        pairs = {(src, dst) for src, dst, _ in _candidate_inter_functional_pairs(graph)}
        self.assertEqual(pairs, {("func_1", "func_2"), ("func_2", "func_1")})
        graph.function_dependencies.append(FunctionDependency("func_1", "func_2", "record"))
        pairs = {(src, dst) for src, dst, _ in _candidate_inter_functional_pairs(graph)}
        self.assertEqual(pairs, {("func_1", "func_2")})


class AccountSessionTests(unittest.TestCase):
    def test_session_guard_blocks_logout_but_not_observation(self):
        with patch.dict("os.environ", {"ACCOUNT_SESSION_GUARD": "1"}):
            self.assertTrue(account_session_action_violation("click", "\u70b9\u51fb\u9000\u51fa\u767b\u5f55"))
            self.assertTrue(account_session_action_violation("click", "Click Log out"))
            self.assertFalse(account_session_action_violation("click", "Open account settings"))
            self.assertFalse(account_session_action_violation("wait", "Log out button is visible"))


if __name__ == "__main__":
    unittest.main()
