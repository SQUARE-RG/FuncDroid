"""页面控件后台审计（workflow/page_audit_worker.py + ffg_update 合并逻辑）测试。"""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.flow_graph import FlowGraph, FunctionNode, Page, control_family_key  # noqa: E402
from workflow import page_audit_worker as paw  # noqa: E402
from workflow.nodes import ffg_update  # noqa: E402
from workflow.page_audit_worker import page_audit_worker  # noqa: E402


def _graph_with_pages() -> FlowGraph:
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="功能一", description=""))
    graph.add_page(
        Page(
            id="page_1",
            name="主页",
            activity=".ui.Main",
            description="",
            representative_screenshot="aGVsbG8=",  # "hello" base64
        ),
        "func_1",
    )
    graph.get_page("page_1").add_control("顶部搜索按钮")
    return graph


class PageAuditWorkerStateTest(unittest.TestCase):
    def setUp(self):
        page_audit_worker.reset()

    def tearDown(self):
        page_audit_worker.reset()

    def test_filter_candidates_blocks_saturated_and_same_fingerprint(self):
        page_audit_worker.submit({"page_id": "page_1", "fingerprint": "f1", "messages": []})
        page_audit_worker.record_result("page_1", 0)
        page_audit_worker.record_result("page_1", 0)  # 连续 2 次零新增 → 饱和
        candidates = [
            {"page_id": "page_1", "fingerprint": "f2"},   # 已饱和，跳过
            {"page_id": "page_2", "fingerprint": "f2"},   # 新页面，允许
        ]
        allowed = page_audit_worker.filter_candidates(candidates)
        self.assertEqual([c["page_id"] for c in allowed], ["page_2"])

    def test_same_fingerprint_without_saturation_is_blocked(self):
        page_audit_worker.record_result("page_9", 3)  # 有新增，未饱和
        # 指纹记账发生在 _process_job；record_result 不更新指纹。
        # 手动模拟指纹已记录：
        with page_audit_worker._lock:
            page_audit_worker._audit_state["page_9"]["fingerprint"] = "fp"
        allowed = page_audit_worker.filter_candidates(
            [{"page_id": "page_9", "fingerprint": "fp"}]
        )
        self.assertEqual(allowed, [])
        # 指纹变化（新截图）则允许复审
        allowed = page_audit_worker.filter_candidates(
            [{"page_id": "page_9", "fingerprint": "fp2"}]
        )
        self.assertEqual(len(allowed), 1)

    def test_add_resets_zero_streak(self):
        page_audit_worker.record_result("page_1", 0)
        page_audit_worker.record_result("page_1", 2)  # 有新增，清零
        page_audit_worker.record_result("page_1", 0)
        state = page_audit_worker._audit_state["page_1"]
        self.assertEqual(state["zero_streak"], 1)
        self.assertFalse(state.get("saturated"))

    def test_parse_audit_payload(self):
        worker = paw.PageAuditWorker()
        missing, invalid = worker._parse_audit_payload(
            '{"missing_controls": ["顶部搜索按钮"], "invalid_operations": ["点击任意一个会话"]}'
        )
        self.assertEqual(missing, ["顶部搜索按钮"])
        self.assertEqual(invalid, ["点击任意一个会话"])
        self.assertEqual(
            worker._parse_audit_payload('废话 {"missing_controls": [], "invalid_operations": []} 结尾'),
            ([], []),
        )
        self.assertEqual(worker._parse_audit_payload("no json"), ([], []))
        self.assertEqual(
            worker._parse_audit_payload('{"missing_controls": "not a list"}'),
            ([], []),
        )
        self.assertEqual(
            worker._parse_audit_payload('{"invalid_operations": "not a list"}'),
            ([], []),
        )
        # 两个清单都做上限截断
        long_list = "[" + ", ".join(f'"顶部按钮{i}"' for i in range(10)) + "]"
        missing, invalid = worker._parse_audit_payload(
            '{"missing_controls": ' + long_list + ', "invalid_operations": ' + long_list + "}"
        )
        self.assertEqual(len(missing), paw.DEFAULT_PER_PAGE_LIMIT)
        self.assertEqual(len(invalid), paw.DEFAULT_PER_PAGE_LIMIT)


class ControlFamilyKeyTest(unittest.TestCase):
    def test_keeps_fixed_emoji_controls_distinct(self):
        self.assertEqual(control_family_key("表情搜索按钮"), "表情搜索")
        self.assertEqual(control_family_key("表情面板删除按钮"), "表情删除")
        self.assertEqual(control_family_key("表情退格图标"), "表情删除")
        self.assertEqual(control_family_key("表情按钮"), "表情")

    def test_normalizes_message_input_synonyms(self):
        self.assertEqual(control_family_key("文本输入框"), "文本输入框")
        self.assertEqual(control_family_key("文字输入框"), "文本输入框")
        self.assertEqual(control_family_key("消息输入框"), "文本输入框")


class MergePageAuditResultsTest(unittest.TestCase):
    def setUp(self):
        page_audit_worker.reset()

    def tearDown(self):
        page_audit_worker.reset()

    def _inject(self, page_id, missing, invalid=None, control_merges=None):
        with page_audit_worker._lock:
            page_audit_worker._results.append({
                "page_id": page_id,
                "fingerprint": "fp",
                "missing_controls": missing,
                "control_merges": control_merges or [],
                "invalid_operations": invalid or [],
            })

    def test_merges_valid_controls_and_reports_added(self):
        graph = _graph_with_pages()
        self._inject("page_1", ["底部点赞按钮", "右上角更多按钮"])
        ffg_update._merge_page_audit_results(graph)
        controls = graph.get_page("page_1").controls
        self.assertIn("底部点赞按钮", controls)
        self.assertIn("右上角更多按钮", controls)

    def test_keeps_controls_without_position_word(self):
        # 实测“从相册中选择”“拍一张”这类真实控件因无位置限定词被过滤，
        # 导致操作绑定无处可挂、覆盖率无法增长：不再做格式过滤。
        graph = _graph_with_pages()
        self._inject("page_1", ["点赞按钮", "从相册中选择"])
        ffg_update._merge_page_audit_results(graph)
        controls = graph.get_page("page_1").controls
        self.assertIn("点赞按钮", controls)
        self.assertIn("从相册中选择", controls)
        state = page_audit_worker._audit_state["page_1"]
        self.assertEqual(state["zero_streak"], 0)  # 有效新增

    def test_drops_semantically_duplicate_controls(self):
        graph = _graph_with_pages()
        self._inject("page_1", ["顶部搜索图标", "顶部搜索按钮"])
        ffg_update._merge_page_audit_results(graph)
        controls = graph.get_page("page_1").controls
        self.assertEqual(controls.count("顶部搜索按钮"), 1)
        self.assertNotIn("顶部搜索图标", controls)

    def test_drops_global_navigation_controls(self):
        # 返回/关闭类由系统返回键完成，不属于页面业务控件
        graph = _graph_with_pages()
        self._inject("page_1", ["左上角返回按钮", "右上角关闭按钮", "右上角更多按钮"])
        ffg_update._merge_page_audit_results(graph)
        controls = graph.get_page("page_1").controls
        self.assertNotIn("左上角返回按钮", controls)
        self.assertNotIn("右上角关闭按钮", controls)
        self.assertIn("右上角更多按钮", controls)

    def test_merges_repeated_resource_controls_and_preserves_bindings(self):
        graph = _graph_with_pages()
        page = graph.get_page("page_1")
        for control in ["背景图-纯色白", "背景图-蓝天", "背景图-爱心"]:
            page.add_control(control)
        page.add_operation("点击背景图-纯色白")
        page.add_operation_binding("点击背景图-纯色白", "背景图-纯色白")
        self._inject(
            "page_1",
            [],
            control_merges=[{
                "sources": ["背景图-纯色白", "背景图-蓝天", "背景图-爱心"],
                "target": "预设背景图项",
            }],
        )

        ffg_update._merge_page_audit_results(graph)

        self.assertNotIn("背景图-纯色白", page.controls)
        self.assertNotIn("背景图-蓝天", page.controls)
        self.assertNotIn("背景图-爱心", page.controls)
        self.assertIn("预设背景图项", page.controls)
        self.assertEqual(page.operation_bindings, [{
            "operation": "点击背景图-纯色白",
            "control": "预设背景图项",
        }])

    def test_rejects_control_merge_without_shared_business_prefix(self):
        graph = _graph_with_pages()
        page = graph.get_page("page_1")
        for control in ["搜索按钮", "添加按钮", "更多按钮"]:
            page.add_control(control)
        self._inject(
            "page_1",
            [],
            control_merges=[{
                "sources": ["搜索按钮", "添加按钮", "更多按钮"],
                "target": "顶部按钮",
            }],
        )

        ffg_update._merge_page_audit_results(graph)

        self.assertNotIn("顶部按钮", page.controls)
        self.assertIn("搜索按钮", page.controls)

    def test_missing_page_reports_zero(self):
        graph = _graph_with_pages()
        self._inject("page_gone", ["顶部按钮"])
        ffg_update._merge_page_audit_results(graph)  # 不应抛异常
        state = page_audit_worker._audit_state["page_gone"]
        self.assertEqual(state["zero_streak"], 1)

    def test_removes_point_named_invalid_operations(self):
        graph = _graph_with_pages()
        page = graph.get_page("page_1")
        page.add_operation("点击顶部搜索按钮")
        page.add_operation("点击任意一个会话")
        page.add_operation("点击张三的会话")
        self._inject(
            "page_1",
            [],
            ["点击任意一个会话", "点击张三的会话"],
        )
        ffg_update._merge_page_audit_results(graph)
        self.assertNotIn("点击任意一个会话", page.operations)
        self.assertNotIn("点击张三的会话", page.operations)
        self.assertIn("点击顶部搜索按钮", page.operations)
        # 有删除 = 有变化，不计零新增
        state = page_audit_worker._audit_state["page_1"]
        self.assertEqual(state["zero_streak"], 0)

    def test_drops_removals_not_matching_original_text(self):
        graph = _graph_with_pages()
        page = graph.get_page("page_1")
        page.add_operation("点击顶部搜索按钮")
        self._inject("page_1", [], ["点击搜索按钮", "幻觉操作名"])
        ffg_update._merge_page_audit_results(graph)
        # 原文不一致的待删项一律丢弃，清单不受影响
        self.assertEqual(page.operations, ["点击顶部搜索按钮"])

    def test_removal_accepts_display_only_unbound_marker(self):
        graph = _graph_with_pages()
        page = graph.get_page("page_1")
        page.add_operation("自动恢复：退出阻塞页面")
        self._inject("page_1", [], ["自动恢复：退出阻塞页面（未绑定控件）"])

        ffg_update._merge_page_audit_results(graph)

        self.assertNotIn("自动恢复：退出阻塞页面", page.operations)

    def test_removals_capped_per_merge(self):
        graph = _graph_with_pages()
        page = graph.get_page("page_1")
        for i in range(8):
            page.add_operation(f"点击顶部按钮{i}")
        self._inject("page_1", [], [f"点击顶部按钮{i}" for i in range(8)])
        ffg_update._merge_page_audit_results(graph)
        self.assertEqual(
            len(page.operations),
            8 - ffg_update.PAGE_AUDIT_MAX_REMOVALS,
        )


class SubmitBatchTest(unittest.TestCase):
    def setUp(self):
        page_audit_worker.reset()

    def tearDown(self):
        page_audit_worker.reset()

    def test_completed_owner_pages_prioritized(self):
        graph = _graph_with_pages()
        graph.add_node(FunctionNode(id="func_2", name="功能二", description=""))
        # page_2 属于已完成功能
        graph.add_page(
            Page(id="page_2", name="完成页", activity=".ui.Done",
                 description="", representative_screenshot="ZG9uZQ=="),
            "func_2",
        )
        graph.set_node_test_status("func_2", "completed")
        # page_3 无属主差异（都在候选中），控件更少应排前
        graph.add_page(
            Page(id="page_3", name="空页", activity=".ui.Empty",
                 description="", representative_screenshot="ZW1wdHk="),
            "func_1",
        )
        captured: list[dict] = []
        with patch.object(page_audit_worker, "filter_candidates", side_effect=lambda x: x), \
             patch.object(page_audit_worker, "submit", side_effect=lambda job: captured.append(job) or True):
            ffg_update._submit_page_audit_batch(graph, llm=object())
        submitted_ids = [job["page_id"] for job in captured]
        self.assertEqual(submitted_ids[0], "page_2")  # completed 属主优先
        self.assertEqual(submitted_ids[1], "page_3")  # controls 更少
        self.assertIn("page_1", submitted_ids)

    def test_pages_without_screenshot_skipped(self):
        graph = _graph_with_pages()
        graph.add_page(Page(id="page_ns", name="无图页", activity=".ui.NS", description=""), "func_1")
        captured: list[dict] = []
        with patch.object(page_audit_worker, "filter_candidates", side_effect=lambda x: x), \
             patch.object(page_audit_worker, "submit", side_effect=lambda job: captured.append(job) or True):
            ffg_update._submit_page_audit_batch(graph, llm=object())
        self.assertNotIn("page_ns", [job["page_id"] for job in captured])

    def test_audit_includes_every_screenshot_of_same_page(self):
        graph = _graph_with_pages()
        page = graph.get_page("page_1")
        for screenshot in ["Yg==", "Yw==", "ZA==", "ZQ=="]:
            page.add_evidence_screenshot(screenshot)
        captured: list[dict] = []
        with patch.object(page_audit_worker, "filter_candidates", side_effect=lambda x: x), \
             patch.object(page_audit_worker, "submit", side_effect=lambda job: captured.append(job) or True):
            ffg_update._submit_page_audit_batch(graph, llm=object())

        job = next(item for item in captured if item["page_id"] == "page_1")
        human_content = job["messages"][1].content
        image_items = [item for item in human_content if item.get("type") == "image_url"]
        state_labels = [
            item["text"] for item in human_content
            if item.get("type") == "text" and str(item.get("text", "")).startswith("[PAGE_STATE_IMAGE]")
        ]
        self.assertEqual(len(image_items), 5)
        self.assertEqual(state_labels, [
            "[PAGE_STATE_IMAGE] 1/5",
            "[PAGE_STATE_IMAGE] 2/5",
            "[PAGE_STATE_IMAGE] 3/5",
            "[PAGE_STATE_IMAGE] 4/5",
            "[PAGE_STATE_IMAGE] 5/5",
        ])

    def test_final_flush_excludes_fingerprint_already_submitted_once(self):
        graph = _graph_with_pages()
        attempted: set[tuple[str, str]] = set()
        captured: list[dict] = []
        with patch.object(
            page_audit_worker,
            "filter_candidates",
            side_effect=lambda candidates: candidates,
        ), patch.object(
            page_audit_worker,
            "submit",
            side_effect=lambda job: captured.append(job) or True,
        ):
            first = ffg_update._submit_page_audit_batch(
                graph,
                llm=object(),
                exclude_fingerprints=attempted,
            )
            second = ffg_update._submit_page_audit_batch(
                graph,
                llm=object(),
                exclude_fingerprints=attempted,
            )

        self.assertEqual(first, 1)
        self.assertEqual(second, 0)
        self.assertEqual(len(captured), 1)

    def test_disabled_by_env(self):
        graph = _graph_with_pages()
        with patch.object(ffg_update, "PAGE_AUDIT_ENABLED", False):
            with patch.object(page_audit_worker, "submit") as mock_submit:
                ffg_update._submit_page_audit_batch(graph, llm=object())
                mock_submit.assert_not_called()


class ProcessJobTest(unittest.TestCase):
    """覆盖 _process_job 真实路径（含 LLM 用量记账调用）。"""

    class _FakeLLM:
        def __init__(self, content='{"missing_controls": ["底部点赞按钮"]}'):
            self._content = content

        def invoke(self, messages):
            if isinstance(self._content, Exception):
                raise self._content
            return type("R", (), {"content": self._content})()

    def test_success_path_records_result_and_fingerprint(self):
        worker = paw.PageAuditWorker()
        worker.configure(self._FakeLLM())
        worker._process_job({
            "page_id": "page_1",
            "fingerprint": "f1",
            "messages": [("human", "x")],
        })
        results = worker.collect_results()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["page_id"], "page_1")
        self.assertEqual(results[0]["missing_controls"], ["底部点赞按钮"])
        self.assertEqual(results[0]["invalid_operations"], [])
        self.assertEqual(worker._audit_state["page_1"]["fingerprint"], "f1")

    def test_invoke_failure_remains_retryable_without_zero_add(self):
        worker = paw.PageAuditWorker()
        worker.configure(self._FakeLLM(RuntimeError("boom")))
        worker._process_job({
            "page_id": "page_2",
            "fingerprint": "f2",
            "messages": [("human", "x")],
        })
        self.assertEqual(worker.collect_results(), [])  # 无结果入队
        state = worker._audit_state["page_2"]
        self.assertEqual(state["fingerprint"], "")
        self.assertEqual(state.get("zero_streak", 0), 0)
        self.assertEqual(
            worker.filter_candidates([{"page_id": "page_2", "fingerprint": "f2"}]),
            [{"page_id": "page_2", "fingerprint": "f2"}],
        )

    def test_invalid_response_remains_retryable_without_zero_add(self):
        worker = paw.PageAuditWorker()
        worker.configure(self._FakeLLM("not json"))
        worker._process_job({
            "page_id": "page_4",
            "fingerprint": "f4",
            "messages": [("human", "x")],
        })
        self.assertEqual(worker.collect_results(), [])
        state = worker._audit_state["page_4"]
        self.assertEqual(state["fingerprint"], "")
        self.assertEqual(state.get("zero_streak", 0), 0)

    def test_usage_recording_calls_do_not_raise(self):
        # 回归：record_llm_request 需要 llm 位置参数（曾因缺参抛 TypeError）。
        # _USAGE 未初始化时记账为空操作，重点验证调用本身不抛异常。
        from workflow import llm_usage as lu

        worker = paw.PageAuditWorker()
        worker.configure(self._FakeLLM())
        try:
            lu._USAGE = {"total": {}}
            worker._process_job({
                "page_id": "page_3",
                "fingerprint": "f3",
                "messages": [("human", "x")],
            })
        finally:
            lu._USAGE = {}
        self.assertEqual(len(worker.collect_results()), 1)


class FingerprintTest(unittest.TestCase):
    def test_fingerprint_changes_with_screenshots(self):
        page = Page(id="p", name="p", activity="a", description="",
                    representative_screenshot="AAA")
        fp1 = ffg_update._page_audit_fingerprint(page)
        page.update_representative_screenshot("BBB")
        fp2 = ffg_update._page_audit_fingerprint(page)
        self.assertNotEqual(fp1, fp2)
        self.assertEqual(len(fp1), 16)

    def test_fingerprint_changes_with_operations(self):
        page = Page(id="p", name="p", activity="a", description="",
                    representative_screenshot="AAA")
        fp1 = ffg_update._page_audit_fingerprint(page)
        page.operations.append("点击顶部搜索按钮")
        fp2 = ffg_update._page_audit_fingerprint(page)
        self.assertNotEqual(fp1, fp2)


if __name__ == "__main__":
    unittest.main()
