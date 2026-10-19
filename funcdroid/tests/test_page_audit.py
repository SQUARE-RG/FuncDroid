# Multilingual UI and model-response fixtures use Unicode escapes to preserve their exact text.
"""Background page-control audit tests for the worker and Functional Flow Graph merge logic."""

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
    graph.add_node(FunctionNode(id="func_1", name="\u529f\u80fd\u4e00", description=""))
    graph.add_page(
        Page(
            id="page_1",
            name="\u4e3b\u9875",
            activity=".ui.Main",
            description="",
            representative_screenshot="aGVsbG8=",  # "hello" base64
        ),
        "func_1",
    )
    graph.get_page("page_1").add_control("\u9876\u90e8\u641c\u7d22\u6309\u94ae")
    return graph


class PageAuditWorkerStateTest(unittest.TestCase):
    def setUp(self):
        page_audit_worker.reset()

    def tearDown(self):
        page_audit_worker.reset()

    def test_submit_rejection_when_saturated_is_logged_and_counted(self):
        """Queue saturation must log and count rejected audits so omitted pages remain diagnosable."""
        with page_audit_worker._lock:
            page_audit_worker._pending = page_audit_worker._max_pending
        try:
            accepted = page_audit_worker.submit(
                {"page_id": "page_1", "fingerprint": "f1", "messages": []}
            )
            self.assertFalse(accepted)
            self.assertEqual(page_audit_worker.rejected_count(), 1)
        finally:
            with page_audit_worker._lock:
                page_audit_worker._pending = 0

    def test_rejected_page_audits_surface_as_manifest_warning(self):
        """Expose rejected-audit warnings in final artifacts even when they do not invalidate completeness."""
        import tempfile

        import main

        with page_audit_worker._lock:
            page_audit_worker._pending = page_audit_worker._max_pending
        try:
            page_audit_worker.submit({"page_id": "page_1", "fingerprint": "f1", "messages": []})
        finally:
            with page_audit_worker._lock:
                page_audit_worker._pending = 0
        state = {"bug_detection_enabled": False, "compliance_detection_enabled": False}
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest = main._validate_final_artifacts(state, tmp_dir, 0)
        self.assertTrue(any("Page audit queue saturation" in warning for warning in manifest["warnings"]))

    def test_filter_candidates_blocks_saturated_and_same_fingerprint(self):
        page_audit_worker.submit({"page_id": "page_1", "fingerprint": "f1", "messages": []})
        page_audit_worker.record_result("page_1", 0)
        page_audit_worker.record_result("page_1", 0)  # Two consecutive zero-addition results saturate the page.
        candidates = [
            {"page_id": "page_1", "fingerprint": "f2"},   # Skip an already saturated page.
            {"page_id": "page_2", "fingerprint": "f2"},   # Allow a new page.
        ]
        allowed = page_audit_worker.filter_candidates(candidates)
        self.assertEqual([c["page_id"] for c in allowed], ["page_2"])

    def test_same_fingerprint_without_saturation_is_blocked(self):
        page_audit_worker.record_result("page_9", 3)  # Additions keep the page unsaturated.
        # _process_job records the fingerprint; record_result does not update it.
        # Simulate an already recorded fingerprint.
        with page_audit_worker._lock:
            page_audit_worker._audit_state["page_9"]["fingerprint"] = "fp"
        allowed = page_audit_worker.filter_candidates(
            [{"page_id": "page_9", "fingerprint": "fp"}]
        )
        self.assertEqual(allowed, [])
        # A changed screenshot fingerprint permits re-audit.
        allowed = page_audit_worker.filter_candidates(
            [{"page_id": "page_9", "fingerprint": "fp2"}]
        )
        self.assertEqual(len(allowed), 1)

    def test_add_resets_zero_streak(self):
        page_audit_worker.record_result("page_1", 0)
        page_audit_worker.record_result("page_1", 2)  # Additions reset the zero-result counter.
        page_audit_worker.record_result("page_1", 0)
        state = page_audit_worker._audit_state["page_1"]
        self.assertEqual(state["zero_streak"], 1)
        self.assertFalse(state.get("saturated"))

    def test_parse_audit_payload(self):
        worker = paw.PageAuditWorker()
        missing, _merges, invalid, parsed = worker._parse_audit_payload_with_status(
            '{"missing_controls": ["\u9876\u90e8\u641c\u7d22\u6309\u94ae"], "invalid_operations": ["\u70b9\u51fb\u4efb\u610f\u4e00\u4e2a\u4f1a\u8bdd"]}'
        )
        self.assertTrue(parsed)
        self.assertEqual(missing, ["\u9876\u90e8\u641c\u7d22\u6309\u94ae"])
        self.assertEqual(invalid, ["\u70b9\u51fb\u4efb\u610f\u4e00\u4e2a\u4f1a\u8bdd"])
        self.assertEqual(
            worker._parse_audit_payload_with_status('\u5e9f\u8bdd {"missing_controls": [], "invalid_operations": []} \u7ed3\u5c3e'),
            ([], [], [], True),
        )
        self.assertEqual(worker._parse_audit_payload_with_status("no json"), ([], [], [], False))
        self.assertEqual(
            worker._parse_audit_payload_with_status('{"missing_controls": "not a list"}'),
            ([], [], [], True),
        )
        self.assertEqual(
            worker._parse_audit_payload_with_status('{"invalid_operations": "not a list"}'),
            ([], [], [], True),
        )
        # Apply size limits to both lists.
        long_list = "[" + ", ".join(f'"\u9876\u90e8\u6309\u94ae{i}"' for i in range(10)) + "]"
        missing, _merges, invalid, parsed = worker._parse_audit_payload_with_status(
            '{"missing_controls": ' + long_list + ', "invalid_operations": ' + long_list + "}"
        )
        self.assertEqual(len(missing), paw.DEFAULT_PER_PAGE_LIMIT)
        self.assertEqual(len(invalid), paw.DEFAULT_PER_PAGE_LIMIT)


class ControlFamilyKeyTest(unittest.TestCase):
    def test_keeps_fixed_emoji_controls_distinct(self):
        self.assertEqual(control_family_key("\u8868\u60c5\u641c\u7d22\u6309\u94ae"), "\u8868\u60c5\u641c\u7d22")
        self.assertEqual(control_family_key("\u8868\u60c5\u9762\u677f\u5220\u9664\u6309\u94ae"), "\u8868\u60c5\u5220\u9664")
        self.assertEqual(control_family_key("\u8868\u60c5\u9000\u683c\u56fe\u6807"), "\u8868\u60c5\u5220\u9664")
        self.assertEqual(control_family_key("\u8868\u60c5\u6309\u94ae"), "\u8868\u60c5")

    def test_normalizes_message_input_synonyms(self):
        self.assertEqual(control_family_key("\u6587\u672c\u8f93\u5165\u6846"), "\u6587\u672c\u8f93\u5165\u6846")
        self.assertEqual(control_family_key("\u6587\u5b57\u8f93\u5165\u6846"), "\u6587\u672c\u8f93\u5165\u6846")
        self.assertEqual(control_family_key("\u6d88\u606f\u8f93\u5165\u6846"), "\u6587\u672c\u8f93\u5165\u6846")


class MergePageAuditResultsTest(unittest.TestCase):
    def setUp(self):
        page_audit_worker.reset()

    def tearDown(self):
        page_audit_worker.reset()

    def _inject(self, page_id, missing, invalid=None, control_merges=None, fingerprint=""):
        # An empty fingerprint skips the check for page changes during the audit.
        with page_audit_worker._lock:
            page_audit_worker._results.append({
                "page_id": page_id,
                "fingerprint": fingerprint,
                "missing_controls": missing,
                "control_merges": control_merges or [],
                "invalid_operations": invalid or [],
            })

    def test_merges_valid_controls_and_reports_added(self):
        graph = _graph_with_pages()
        self._inject("page_1", ["\u5e95\u90e8\u70b9\u8d5e\u6309\u94ae", "\u53f3\u4e0a\u89d2\u66f4\u591a\u6309\u94ae"])
        ffg_update._merge_page_audit_results(graph)
        controls = graph.get_page("page_1").controls
        self.assertIn("\u5e95\u90e8\u70b9\u8d5e\u6309\u94ae", controls)
        self.assertIn("\u53f3\u4e0a\u89d2\u66f4\u591a\u6309\u94ae", controls)

    def test_keeps_controls_without_position_word(self):
        # Real controls such as choose-from-album or take-photo were filtered for lacking position qualifiers.
        # Remove that formatting filter so actions can bind and increase coverage.
        graph = _graph_with_pages()
        self._inject("page_1", ["\u70b9\u8d5e\u6309\u94ae", "\u4ece\u76f8\u518c\u4e2d\u9009\u62e9"])
        ffg_update._merge_page_audit_results(graph)
        controls = graph.get_page("page_1").controls
        self.assertIn("\u70b9\u8d5e\u6309\u94ae", controls)
        self.assertIn("\u4ece\u76f8\u518c\u4e2d\u9009\u62e9", controls)
        state = page_audit_worker._audit_state["page_1"]
        self.assertEqual(state["zero_streak"], 0)  # Valid addition.

    def test_drops_semantically_duplicate_controls(self):
        graph = _graph_with_pages()
        self._inject("page_1", ["\u9876\u90e8\u641c\u7d22\u56fe\u6807", "\u9876\u90e8\u641c\u7d22\u6309\u94ae"])
        ffg_update._merge_page_audit_results(graph)
        controls = graph.get_page("page_1").controls
        self.assertEqual(controls.count("\u9876\u90e8\u641c\u7d22\u6309\u94ae"), 1)
        self.assertNotIn("\u9876\u90e8\u641c\u7d22\u56fe\u6807", controls)

    def test_drops_global_navigation_controls(self):
        # Back/close controls use the system back action and are not page functionality controls.
        graph = _graph_with_pages()
        self._inject("page_1", ["\u5de6\u4e0a\u89d2\u8fd4\u56de\u6309\u94ae", "\u53f3\u4e0a\u89d2\u5173\u95ed\u6309\u94ae", "\u53f3\u4e0a\u89d2\u66f4\u591a\u6309\u94ae"])
        ffg_update._merge_page_audit_results(graph)
        controls = graph.get_page("page_1").controls
        self.assertNotIn("\u5de6\u4e0a\u89d2\u8fd4\u56de\u6309\u94ae", controls)
        self.assertNotIn("\u53f3\u4e0a\u89d2\u5173\u95ed\u6309\u94ae", controls)
        self.assertIn("\u53f3\u4e0a\u89d2\u66f4\u591a\u6309\u94ae", controls)

    def test_merges_repeated_resource_controls_and_preserves_bindings(self):
        graph = _graph_with_pages()
        page = graph.get_page("page_1")
        for control in ["\u80cc\u666f\u56fe-\u7eaf\u8272\u767d", "\u80cc\u666f\u56fe-\u84dd\u5929", "\u80cc\u666f\u56fe-\u7231\u5fc3"]:
            page.add_control(control)
        page.add_operation("\u70b9\u51fb\u80cc\u666f\u56fe-\u7eaf\u8272\u767d")
        page.add_operation_binding("\u70b9\u51fb\u80cc\u666f\u56fe-\u7eaf\u8272\u767d", "\u80cc\u666f\u56fe-\u7eaf\u8272\u767d")
        self._inject(
            "page_1",
            [],
            control_merges=[{
                "sources": ["\u80cc\u666f\u56fe-\u7eaf\u8272\u767d", "\u80cc\u666f\u56fe-\u84dd\u5929", "\u80cc\u666f\u56fe-\u7231\u5fc3"],
                "target": "\u9884\u8bbe\u80cc\u666f\u56fe\u9879",
            }],
        )

        ffg_update._merge_page_audit_results(graph)

        self.assertNotIn("\u80cc\u666f\u56fe-\u7eaf\u8272\u767d", page.controls)
        self.assertNotIn("\u80cc\u666f\u56fe-\u84dd\u5929", page.controls)
        self.assertNotIn("\u80cc\u666f\u56fe-\u7231\u5fc3", page.controls)
        self.assertIn("\u9884\u8bbe\u80cc\u666f\u56fe\u9879", page.controls)
        self.assertEqual(page.operation_bindings, [{
            "operation": "\u70b9\u51fb\u80cc\u666f\u56fe-\u7eaf\u8272\u767d",
            "control": "\u9884\u8bbe\u80cc\u666f\u56fe\u9879",
        }])

    def test_rejects_control_merge_without_shared_business_prefix(self):
        graph = _graph_with_pages()
        page = graph.get_page("page_1")
        for control in ["\u641c\u7d22\u6309\u94ae", "\u6dfb\u52a0\u6309\u94ae", "\u66f4\u591a\u6309\u94ae"]:
            page.add_control(control)
        self._inject(
            "page_1",
            [],
            control_merges=[{
                "sources": ["\u641c\u7d22\u6309\u94ae", "\u6dfb\u52a0\u6309\u94ae", "\u66f4\u591a\u6309\u94ae"],
                "target": "\u9876\u90e8\u6309\u94ae",
            }],
        )

        ffg_update._merge_page_audit_results(graph)

        self.assertNotIn("\u9876\u90e8\u6309\u94ae", page.controls)
        self.assertIn("\u641c\u7d22\u6309\u94ae", page.controls)

    def test_missing_page_reports_zero(self):
        graph = _graph_with_pages()
        self._inject("page_gone", ["\u9876\u90e8\u6309\u94ae"])
        ffg_update._merge_page_audit_results(graph)  # Must not raise.
        state = page_audit_worker._audit_state["page_gone"]
        self.assertEqual(state["zero_streak"], 1)

    def test_removes_point_named_invalid_operations(self):
        graph = _graph_with_pages()
        page = graph.get_page("page_1")
        page.add_operation("\u70b9\u51fb\u9876\u90e8\u641c\u7d22\u6309\u94ae")
        page.add_operation("\u70b9\u51fb\u4efb\u610f\u4e00\u4e2a\u4f1a\u8bdd")
        page.add_operation("\u70b9\u51fb\u5f20\u4e09\u7684\u4f1a\u8bdd")
        self._inject(
            "page_1",
            [],
            ["\u70b9\u51fb\u4efb\u610f\u4e00\u4e2a\u4f1a\u8bdd", "\u70b9\u51fb\u5f20\u4e09\u7684\u4f1a\u8bdd"],
        )
        ffg_update._merge_page_audit_results(graph)
        self.assertNotIn("\u70b9\u51fb\u4efb\u610f\u4e00\u4e2a\u4f1a\u8bdd", page.operations)
        self.assertNotIn("\u70b9\u51fb\u5f20\u4e09\u7684\u4f1a\u8bdd", page.operations)
        self.assertIn("\u70b9\u51fb\u9876\u90e8\u641c\u7d22\u6309\u94ae", page.operations)
        # Removals count as changes rather than zero-addition results.
        state = page_audit_worker._audit_state["page_1"]
        self.assertEqual(state["zero_streak"], 0)

    def test_stale_fingerprint_keeps_additions_but_skips_removals(self):
        graph = _graph_with_pages()
        page = graph.get_page("page_1")
        page.add_operation("\u70b9\u51fb\u5f20\u4e09\u7684\u4f1a\u8bdd")
        # A trace updated the page after audit submission, so the result fingerprint is stale.
        self._inject("page_1", ["\u5e95\u90e8\u70b9\u8d5e\u6309\u94ae"], ["\u70b9\u51fb\u5f20\u4e09\u7684\u4f1a\u8bdd"], fingerprint="stale")
        ffg_update._merge_page_audit_results(graph)
        self.assertIn("\u5e95\u90e8\u70b9\u8d5e\u6309\u94ae", page.controls)
        self.assertIn("\u70b9\u51fb\u5f20\u4e09\u7684\u4f1a\u8bdd", page.operations)

    def test_drops_removals_not_matching_original_text(self):
        graph = _graph_with_pages()
        page = graph.get_page("page_1")
        page.add_operation("\u70b9\u51fb\u9876\u90e8\u641c\u7d22\u6309\u94ae")
        self._inject("page_1", [], ["\u70b9\u51fb\u641c\u7d22\u6309\u94ae", "\u5e7b\u89c9\u64cd\u4f5c\u540d"])
        ffg_update._merge_page_audit_results(graph)
        # Discard removal entries whose original text differs without changing the list.
        self.assertEqual(page.operations, ["\u70b9\u51fb\u9876\u90e8\u641c\u7d22\u6309\u94ae"])

    def test_removal_accepts_display_only_unbound_marker(self):
        graph = _graph_with_pages()
        page = graph.get_page("page_1")
        page.add_operation("\u81ea\u52a8\u6062\u590d：\u9000\u51fa\u963b\u585e\u9875\u9762")
        self._inject("page_1", [], ["\u81ea\u52a8\u6062\u590d：\u9000\u51fa\u963b\u585e\u9875\u9762（\u672a\u7ed1\u5b9a\u63a7\u4ef6）"])

        ffg_update._merge_page_audit_results(graph)

        self.assertNotIn("\u81ea\u52a8\u6062\u590d：\u9000\u51fa\u963b\u585e\u9875\u9762", page.operations)

    def test_removals_capped_per_merge(self):
        graph = _graph_with_pages()
        page = graph.get_page("page_1")
        for i in range(8):
            page.add_operation(f"\u70b9\u51fb\u9876\u90e8\u6309\u94ae{i}")
        self._inject("page_1", [], [f"\u70b9\u51fb\u9876\u90e8\u6309\u94ae{i}" for i in range(8)])
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
        graph.add_node(FunctionNode(id="func_2", name="\u529f\u80fd\u4e8c", description=""))
        # page_2 belongs to a completed functionality.
        graph.add_page(
            Page(id="page_2", name="\u5b8c\u6210\u9875", activity=".ui.Done",
                 description="", representative_screenshot="ZG9uZQ=="),
            "func_2",
        )
        graph.set_node_test_status("func_2", "completed")
        # page_3 has no ownership distinction and fewer controls, so it should sort earlier.
        graph.add_page(
            Page(id="page_3", name="\u7a7a\u9875", activity=".ui.Empty",
                 description="", representative_screenshot="ZW1wdHk="),
            "func_1",
        )
        captured: list[dict] = []
        with patch.object(page_audit_worker, "filter_candidates", side_effect=lambda x: x), \
             patch.object(page_audit_worker, "submit", side_effect=lambda job: captured.append(job) or True):
            ffg_update._submit_page_audit_batch(graph, llm=object())
        submitted_ids = [job["page_id"] for job in captured]
        self.assertEqual(submitted_ids[0], "page_2")  # Completed owner first.
        self.assertEqual(submitted_ids[1], "page_3")  # Fewer controls next.
        self.assertIn("page_1", submitted_ids)

    def test_pages_without_screenshot_skipped(self):
        graph = _graph_with_pages()
        graph.add_page(Page(id="page_ns", name="\u65e0\u56fe\u9875", activity=".ui.NS", description=""), "func_1")
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
    """Exercise the actual _process_job path, including LLM usage bookkeeping."""

    class _FakeLLM:
        def __init__(self, content='{"missing_controls": ["\u5e95\u90e8\u70b9\u8d5e\u6309\u94ae"]}'):
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
        self.assertEqual(results[0]["missing_controls"], ["\u5e95\u90e8\u70b9\u8d5e\u6309\u94ae"])
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
        self.assertEqual(worker.collect_results(), [])  # No result enqueued.
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
        # Regression: record_llm_request requires the llm positional argument.
        # Uninitialized _USAGE makes bookkeeping a no-op; the call must still not raise.
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
        page.operations.append("\u70b9\u51fb\u9876\u90e8\u641c\u7d22\u6309\u94ae")
        fp2 = ffg_update._page_audit_fingerprint(page)
        self.assertNotEqual(fp1, fp2)


if __name__ == "__main__":
    unittest.main()
