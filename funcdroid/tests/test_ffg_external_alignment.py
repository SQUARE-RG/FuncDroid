"""FFG 更新外部/恢复步骤对齐修复（qwen-local-optimization）测试。

覆盖微信_10 实际发生的 refinement 死锁：
1. 恢复步骤只带 is_interruption（无 external_context）时 ignore 应合法；
2. 外部页面步骤只带 external_context 时 ignore 应合法；
3. 模型整段遗漏对齐时程序自动补齐 ignore；
4. 模型对外部步骤给出错误 decision 时程序矫正为 ignore;
5. _unrepresented_trace_activities 不再要求外部 Activity 建 Page；
6. restructures 非法 page_refs 降级为 advisory 不再整轮回滚。
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.flow_graph import FlowGraph, FunctionNode, Page  # noqa: E402
from workflow.nodes.ffg_update import (  # noqa: E402
    _ensure_ignorable_alignments,
    _is_ignorable_observation_step,
    _unrepresented_trace_activities,
    partition_alignment_issues,
    validate_observation_alignment,
)


def _graph() -> FlowGraph:
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="功能一", description=""))
    graph.add_page(Page(id="page_1", name="会话页", activity=".ui.LauncherUI", description=""), "func_1")
    return graph


def _trace() -> list[dict]:
    return [
        {
            "raw_step_index": 1,
            "action_name": "click",
            "current_page": {"id": "page_1", "name": "会话页", "activity": ".ui.LauncherUI"},
        },
        # 微信页面动作导致跳到外部（桌面）
        {
            "raw_step_index": 2,
            "action_name": "press_back",
            "caused_external_transition": True,
            "external_package": "com.hihonor.android.launcher",
            "current_page": {"id": "page_1", "name": "会话页", "activity": ".ui.LauncherUI"},
        },
        # 桌面观测步骤（受控外部路径记录）：只带 external_context
        {
            "raw_step_index": 3,
            "action_name": "press_back",
            "external_context": True,
            "external_package": "com.hihonor.android.launcher",
            "current_page": {"id": "", "name": "桌面", "activity": ".unihome.UniHomeLauncher"},
        },
        # 运行器自动恢复步骤：只带 is_interruption/foreground_recovery
        {
            "raw_step_index": 4,
            "action_name": "press_back",
            "is_interruption": True,
            "foreground_recovery": True,
            "observation_after_recovery": True,
            "current_page": {"id": "page_1", "name": "会话页", "activity": ".ui.LauncherUI"},
        },
    ]


class IgnorableObservationStepTest(unittest.TestCase):
    def test_recovery_step_is_ignorable_without_external_context(self):
        # 旧校验要求 is_interruption 且 external_context 同时成立，恢复步骤必然失败
        step = {"is_interruption": True, "foreground_recovery": True}
        self.assertTrue(_is_ignorable_observation_step(step))

    def test_external_context_alone_is_ignorable(self):
        self.assertTrue(_is_ignorable_observation_step({"external_context": True}))

    def test_normal_step_not_ignorable(self):
        self.assertFalse(_is_ignorable_observation_step({"action_name": "click"}))
        self.assertFalse(_is_ignorable_observation_step({}))
        # caused_external_transition 的动作步骤页面仍是目标应用的，不自动忽略
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
            f"恢复/外部步骤的 ignore 不应再报错: {issues}",
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
        self.assertEqual(fixed, 2)  # step 3、4 被补齐
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
        }
        fixed = _ensure_ignorable_alignments(patch, _trace())
        self.assertEqual(fixed, 2)
        for item in patch["observation_alignment"]:
            self.assertEqual(item["page_decision"], "ignore")
            self.assertNotIn("page_ref", item)

    def test_normal_steps_untouched(self):
        patch = {
            "observation_alignment": [
                {"source_step": 1, "page_decision": "reuse", "page_ref": "page_1"},
            ],
        }
        fixed = _ensure_ignorable_alignments(patch, _trace())
        # step 2 是普通微信页面步骤，程序不改动
        self.assertEqual(fixed, 2)  # 只补 3、4
        self.assertEqual(
            patch["observation_alignment"][0],
            {"source_step": 1, "page_decision": "reuse", "page_ref": "page_1"},
        )


class UnrepresentedActivitiesTest(unittest.TestCase):
    def test_external_activity_not_required_as_page(self):
        graph = _graph()
        # 步骤 3（桌面）activity 不应再要求建 Page
        missing = _unrepresented_trace_activities(graph, _trace(), "com.tencent.mm", "")
        self.assertNotIn(".unihome.UniHomeLauncher", missing)

    def test_new_target_app_activity_still_reported(self):
        graph = _graph()
        trace = [
            {
                "raw_step_index": 1,
                "action_name": "click",
                "current_page": {"id": "", "name": "文章页", "activity": ".ui.article.ArticleUI"},
            },
        ]
        missing = _unrepresented_trace_activities(graph, trace, "com.tencent.mm", "")
        self.assertIn(".ui.article.ArticleUI", missing)


class RestructureIssueAdvisoryTest(unittest.TestCase):
    def test_non_coverage_page_issue_is_advisory(self):
        blocking, advisory = partition_alignment_issues([
            "function_restructures[1].replacements[1] cannot move non-coverage page new_page_1 from func_1",
            "function_restructures[1].replacements[1] has no valid coverage pages",
            "missing observation_alignment for source_step 2",
        ])
        self.assertEqual(len(blocking), 1)
        self.assertIn("missing observation_alignment", blocking[0])
        self.assertEqual(len(advisory), 2)


if __name__ == "__main__":
    unittest.main()
