# Multilingual UI and model-response fixtures use Unicode escapes to preserve their exact text.
"""External/recovery alignment regressions: accept ignore for either recovery or external context, repair missing/incorrect external alignment, avoid external Page creation, and skip only invalid restructure entries."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.flow_graph import FunctionalFlowGraph, FunctionalityNode, Page  # noqa: E402
from workflow.nodes.ffg_update import (  # noqa: E402
    _ensure_ignorable_alignments,
    _is_ignorable_observation_step,
    _unrepresented_trace_activities,
    validate_observation_alignment,
)


def _graph() -> FunctionalFlowGraph:
    graph = FunctionalFlowGraph()
    graph.add_node(FunctionalityNode(id="func_1", name="\u529f\u80fd\u4e00", description=""))
    graph.add_page(Page(id="page_1", name="\u4f1a\u8bdd\u9875", activity=".ui.LauncherUI", description=""), "func_1")
    return graph


def _trace() -> list[dict]:
    return [
        {
            "raw_step_index": 1,
            "action_name": "click",
            "current_page": {"id": "page_1", "name": "\u4f1a\u8bdd\u9875", "activity": ".ui.LauncherUI"},
        },
        # An action on a WeChat page transitions to the external launcher.
        {
            "raw_step_index": 2,
            "action_name": "press_back",
            "caused_external_transition": True,
            "external_package": "com.hihonor.android.launcher",
            "current_page": {"id": "page_1", "name": "\u4f1a\u8bdd\u9875", "activity": ".ui.LauncherUI"},
        },
        # The external launcher observation has only external_context.
        {
            "raw_step_index": 3,
            "action_name": "press_back",
            "external_context": True,
            "external_package": "com.hihonor.android.launcher",
            "current_page": {"id": "", "name": "\u684c\u9762", "activity": ".unihome.UniHomeLauncher"},
        },
        # The automatic recovery step has only is_interruption/foreground_recovery.
        {
            "raw_step_index": 4,
            "action_name": "press_back",
            "is_interruption": True,
            "foreground_recovery": True,
            "observation_after_recovery": True,
            "current_page": {"id": "page_1", "name": "\u4f1a\u8bdd\u9875", "activity": ".ui.LauncherUI"},
        },
    ]


class IgnorableObservationStepTest(unittest.TestCase):
    def test_recovery_step_is_ignorable_without_external_context(self):
        # Requiring both is_interruption and external_context incorrectly rejected recovery steps.
        step = {"is_interruption": True, "foreground_recovery": True}
        self.assertTrue(_is_ignorable_observation_step(step))

    def test_external_context_alone_is_ignorable(self):
        self.assertTrue(_is_ignorable_observation_step({"external_context": True}))

    def test_normal_step_not_ignorable(self):
        self.assertFalse(_is_ignorable_observation_step({"action_name": "click"}))
        self.assertFalse(_is_ignorable_observation_step({}))
        # The action causing an external transition still belongs to the target app and is not ignored.
        self.assertFalse(
            _is_ignorable_observation_step({"caused_external_transition": True})
        )

    def test_screenshot_health_error_is_ignorable(self):
        self.assertTrue(
            _is_ignorable_observation_step({"screenshot_health_error": "blur"})
        )


class ValidateAlignmentTest(unittest.TestCase):
    def test_ignore_on_recovery_and_external_steps_passes(self):
        graph = _graph()
        trace = _trace()
        candidate_map = {1: [{"page_id": "page_1"}], 2: [{"page_id": "page_1"}]}
        patch = {
            "observation_alignment": [
                {"source_step": 1, "page_decision": "reuse", "page_ref": "page_1"},
                {"source_step": 2, "page_decision": "reuse", "page_ref": "page_1"},
                {"source_step": 3, "page_decision": "ignore"},
                {"source_step": 4, "page_decision": "ignore"},
            ],
            "pages": [],
        }
        issues = validate_observation_alignment(graph, patch, trace, candidate_map)
        self.assertEqual(
            [i for i in issues if "ignorable" in i],
            [],
            f"\u6062\u590d/\u5916\u90e8\u6b65\u9aa4\u7684 ignore \u4e0d\u5e94\u518d\u62a5\u9519: {issues}",
        )


class EnsureIgnorableAlignmentsTest(unittest.TestCase):
    def test_auto_fills_missing_alignments_for_ignorable_steps(self):
        trace = _trace()
        patch = {
            "observation_alignment": [
                {"source_step": 1, "page_decision": "reuse", "page_ref": "page_1"},
                {"source_step": 2, "page_decision": "reuse", "page_ref": "page_1"},
            ],
        }
        fixed = _ensure_ignorable_alignments(patch, trace)
        self.assertEqual(fixed, 2)  # Fill alignments for steps 3 and 4.
        decisions = {a["source_step"]: a["page_decision"] for a in patch["observation_alignment"]}
        self.assertEqual(decisions[3], "ignore")
        self.assertEqual(decisions[4], "ignore")

    def test_auto_creates_alignment_list_when_missing(self):
        patch = {}
        fixed = _ensure_ignorable_alignments(patch, _trace())
        self.assertEqual(fixed, 2)
        self.assertEqual(len(patch["observation_alignment"]), 2)

    def test_corrects_wrong_decision_on_ignorable_step(self):
        patch = {
            "observation_alignment": [
                {"source_step": 3, "page_decision": "create", "page_ref": "new_page_9"},
                {"source_step": 4, "page_decision": "reuse", "page_ref": "page_1"},
            ],
            "pages": [
                {"page_ref": "new_page_9", "source_step": 3, "name": "\u7cfb\u7edf\u6743\u9650\u9875"},
            ],
            "page_ownership": [
                {"page_ref": "new_page_9", "add_function_refs": ["func_1"]},
            ],
        }
        fixed = _ensure_ignorable_alignments(patch, _trace())
        self.assertEqual(fixed, 2)
        for item in patch["observation_alignment"]:
            self.assertEqual(item["page_decision"], "ignore")
            self.assertNotIn("page_ref", item)
        self.assertEqual(patch["pages"], [])
        self.assertEqual(patch["page_ownership"], [])

    def test_normal_steps_untouched(self):
        patch = {
            "observation_alignment": [
                {"source_step": 1, "page_decision": "reuse", "page_ref": "page_1"},
            ],
        }
        fixed = _ensure_ignorable_alignments(patch, _trace())
        # Step 2 is an ordinary WeChat page and remains unchanged.
        self.assertEqual(fixed, 2)  # Only fill steps 3 and 4.
        self.assertEqual(
            patch["observation_alignment"][0],
            {"source_step": 1, "page_decision": "reuse", "page_ref": "page_1"},
        )


class UnrepresentedActivitiesTest(unittest.TestCase):
    def test_external_activity_not_required_as_page(self):
        graph = _graph()
        # The external launcher Activity at step 3 must not require a Page node.
        missing = _unrepresented_trace_activities(graph, _trace())
        self.assertNotIn(".unihome.UniHomeLauncher", missing)

    def test_new_target_app_activity_still_reported(self):
        graph = _graph()
        trace = [
            {
                "raw_step_index": 1,
                "action_name": "click",
                "current_page": {"id": "", "name": "\u6587\u7ae0\u9875", "activity": ".ui.article.ArticleUI"},
            },
        ]
        missing = _unrepresented_trace_activities(graph, trace)
        self.assertIn(".ui.article.ArticleUI", missing)


if __name__ == "__main__":
    unittest.main()
