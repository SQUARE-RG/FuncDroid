# Multilingual UI and model-response fixtures use Unicode escapes to preserve their exact text.
"""Regressions for best-effort Functional Flow Graph patches, bounded repair passes, and associated performance fixes without rollback."""

import base64
import os

import cv2
import numpy as np

from model import flow_graph as flow_graph_module
from model.flow_graph import FlowGraph, FunctionNode, Page, PageTransition
from workflow.nodes import ffg_update as updater
from workflow.nodes.ffg_update import (
    NAVIGATION_FAILURE_STATUSES,
    _normalize_observation_alignment,
    apply_ffg_patch,
    validate_ffg_invariants,
    validate_observation_alignment,
)
from workflow.page_audit_worker import PageAuditWorker


def _screenshot(value: int) -> str:
    image = np.full((48, 32, 3), value, dtype=np.uint8)
    success, buffer = cv2.imencode(".png", image)
    assert success
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def _graph() -> FlowGraph:
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="\u529f\u80fd\u4e00", test_status="in_progress"))
    graph.add_page(Page(id="page_1", name="\u9996\u9875", activity="HomeActivity"), "func_1")
    return graph


# Screenshot persistence cache: A->B->A rotation must not leave stale files.

def test_screenshot_write_cache_rewrites_after_content_rotation(tmp_path):
    first, second = _screenshot(10), _screenshot(200)
    screenshots_dir = str(tmp_path)
    for value in (first, second, first):
        flow_graph_module._write_screenshot(value, "page_1_evidence_1.jpg", screenshots_dir)
    with open(os.path.join(screenshots_dir, "page_1_evidence_1.jpg"), "rb") as handle:
        assert handle.read() == base64.b64decode(first)


# Transition-edge index and clone behavior.

def test_transition_index_survives_external_list_replacement():
    graph = _graph()
    graph.add_page(Page(id="page_2", name="\u8be6\u60c5"), "func_1")
    graph.add_page_transition(PageTransition("page_1", "page_2", "\u70b9\u51fb\u8be6\u60c5"))
    graph.page_transitions = []
    graph._page_transition_set = set()
    graph.add_page_transition(PageTransition("page_1", "page_2", "\u70b9\u51fb\u8be6\u60c5"))
    assert len(graph.page_transitions) == 1


def test_clone_is_independent_of_source():
    graph = _graph()
    graph.get_page("page_1").add_control("\u641c\u7d22\u6309\u94ae")
    copied = graph.clone()
    copied.get_page("page_1").add_control("\u66f4\u591a\u6309\u94ae")
    copied.get_node("func_1").scenarios.append("\u65b0\u573a\u666f")
    assert graph.get_page("page_1").controls == ["\u641c\u7d22\u6309\u94ae"]
    assert graph.get_node("func_1").scenarios == []
    assert copied.to_dict()["functions"][0]["id"] == "func_1"


# Alignment normalization.

def test_reuse_existing_page_with_pages_update_stays_reuse():
    graph = _graph()
    patch = {
        "observation_alignment": [{"source_step": 1, "page_decision": "reuse", "page_ref": "page_1"}],
        "pages": [{"page_ref": "page_1", "function_refs": ["func_1"], "source_step": 1,
                   "controls": ["\u65b0\u589e\u6309\u94ae"]}],
    }
    trace = [{"raw_step_index": 1, "current_page": Page(activity="HomeActivity")}]
    alignments = _normalize_observation_alignment(patch, graph, trace, {1: [{"page_id": "page_1"}]})
    assert alignments[0]["page_decision"] == "reuse"
    assert validate_observation_alignment(graph, patch, trace, {1: [{"page_id": "page_1"}]}) == []


def test_new_page_spanning_multiple_steps_has_a_valid_spelling():
    graph = _graph()
    trace = [
        {"raw_step_index": 1, "current_page": Page(activity="DetailActivity")},
        {"raw_step_index": 2, "current_page": Page(activity="DetailActivity")},
    ]
    patch = {
        "observation_alignment": [
            {"source_step": 1, "page_decision": "create", "page_ref": "new_page_1"},
            {"source_step": 2, "page_decision": "reuse", "page_ref": "new_page_1"},
        ],
        "pages": [{"page_ref": "new_page_1", "function_refs": ["func_1"], "source_step": 1,
                   "name": "\u8be6\u60c5\u9875"}],
    }
    _normalize_observation_alignment(patch, graph, trace, {})
    assert validate_observation_alignment(graph, patch, trace, {}) == []
    result = apply_ffg_patch(graph, patch, trace, {"target_function_id": "func_1"})
    new_page_id = result["page_refs"]["new_page_1"]
    assert result["trace_page_map"] == {1: new_page_id, 2: new_page_id}


# Best-effort application without rollback.

def test_invalid_restructure_does_not_discard_new_pages_of_same_patch():
    graph = _graph()
    trace = [{"raw_step_index": 1, "current_page": Page(activity="NewActivity")}]
    result = apply_ffg_patch(graph, {
        "pages": [{"page_ref": "new_page_1", "function_refs": ["func_1"], "source_step": 1,
                   "name": "\u65b0\u9875\u9762"}],
        "function_restructures": [{
            "source_function_ref": "func_1",
            "replacements": [{"function_ref": "func_1", "page_refs": ["missing"]}],
        }],
    }, trace, {"target_function_id": "func_1"})
    assert result["validation_issues"]
    assert result["page_refs"]["new_page_1"] in graph.pages
    assert validate_ffg_invariants(graph) == []


def test_unmapped_new_activity_is_created_by_program():
    graph = _graph()
    trace = [
        {"raw_step_index": 1, "action_name": "click", "action_description": "\u70b9\u51fb\u8bbe\u7f6e",
         "current_page": Page(activity="HomeActivity", representative_screenshot=_screenshot(10))},
        {"raw_step_index": 2, "action_name": "finished",
         "current_page": Page(activity="SettingsActivity", representative_screenshot=_screenshot(90))},
    ]
    graph.get_page("page_1").update_representative_screenshot(_screenshot(10))
    result = apply_ffg_patch(graph, {}, trace, {"target_function_id": "func_1"})
    assert len(result["auto_created_pages"]) == 1
    created = graph.get_page(result["auto_created_pages"][0])
    assert created.activity == "SettingsActivity"
    assert graph.get_page_owners(created.id) == ["func_1"]
    assert result["trace_page_map"] == {1: "page_1", 2: created.id}


def test_repair_assigns_orphan_page_to_planned_target():
    graph = _graph()
    graph.pages["page_9"] = Page(id="page_9", name="\u5b64\u7acb\u9875")
    issues = updater._repair_ffg_invariants(graph, {"target_function_id": "func_1"})
    assert any("page_9" in issue for issue in issues)
    assert graph.get_page_owners("page_9") == ["func_1"]
    assert validate_ffg_invariants(graph) == []


def test_node_keeps_partial_patch_and_stops_repair_on_identical_issues(monkeypatch):
    calls = []

    class Llm:
        def invoke(self, _messages):
            calls.append(1)
            # The pages patch is valid; observation_alignment is missing and restructure is invalid.
            return type("R", (), {"content": (
                '{"pages":[{"page_ref":"new_page_1","function_refs":["func_1"],'
                '"source_step":2,"name":"\u8be6\u60c5\u9875"}],'
                '"function_restructures":[{"source_function_ref":"func_404","replacements":[]}]}'
            )})()

    monkeypatch.setattr(updater, "_submit_page_audit_batch", lambda *a, **k: 0)
    graph = _graph()
    node = updater.create_ffg_update_node(Llm())
    result = node({
        "ffg": graph,
        "current_plan": {"scenario_type": "intra_test", "target_function_id": "func_1"},
        "current_test_scenario": "\u6253\u5f00\u8be6\u60c5",
        "navigation_execution_trace": [],
        "scenario_execution_trace": [
            {"action_name": "click", "action_description": "\u70b9\u51fb\u6761\u76ee",
             "current_page": Page(id="page_1", activity="HomeActivity")},
            {"action_name": "finished", "current_page": Page(activity="DetailActivity")},
        ],
        "execution_result": {"navigation_status": "skipped", "scenario_status": "completed"},
    })
    # Keep the valid first-pass page; stop after one repair repeats the invalid split.
    assert len(calls) == 2
    assert result["ffg_update_decision"] == "finish"
    assert [page.name for page in graph.pages.values()] == ["\u9996\u9875", "\u8be6\u60c5\u9875"]
    assert "\u70b9\u51fb\u6761\u76ee" in graph.get_page("page_1").operations
    assert "\u6253\u5f00\u8be6\u60c5" in graph.get_node("func_1").scenarios


def test_navigation_leaving_app_counts_as_navigation_failure():
    assert {"left_app", "app_crashed"} <= NAVIGATION_FAILURE_STATUSES
    graph = _graph()
    state = {"current_plan": {"scenario_type": "intra_test", "target_function_id": "func_1"}}
    for _ in range(updater.NAVIGATION_FAILURE_DEFER_LIMIT):
        updater._record_execution_outcome(graph, state, {
            "navigation_status": "left_app", "navigation_error": "\u8df3\u51fa\u5e94\u7528",
            "scenario_status": "skipped",
        })
    assert graph.get_node_test_status("func_1") == "deferred"


# Page-audit worker.

def test_inflight_audit_is_not_resubmitted():
    import threading

    worker = PageAuditWorker()
    release = threading.Event()
    worker._process_job = lambda _job: release.wait(5)  # Keep the task executing.
    try:
        assert worker.submit({"page_id": "page_1", "fingerprint": "fp", "messages": []})
        assert worker.filter_candidates([{"page_id": "page_1", "fingerprint": "fp"}]) == []
        assert worker.filter_candidates([{"page_id": "page_1", "fingerprint": "fp2"}]) == [
            {"page_id": "page_1", "fingerprint": "fp2"}
        ]
    finally:
        release.set()


def test_inflight_marker_cleared_even_when_job_finishes_before_submit_returns():
    worker = PageAuditWorker()  # Without an LLM, process the task immediately and record failure.
    assert worker.submit({"page_id": "page_1", "fingerprint": "fp", "messages": []})
    assert worker.flush(5)
    assert "inflight" not in worker._audit_state["page_1"]


def test_repeated_audit_failure_stops_retrying_same_fingerprint():
    worker = PageAuditWorker()
    for _ in range(3):
        worker._record_failure("page_1", "fp")
    assert worker.filter_candidates([{"page_id": "page_1", "fingerprint": "fp"}]) == []
    assert worker.filter_candidates([{"page_id": "page_1", "fingerprint": "fp2"}])


def test_page_with_unaudited_screenshots_is_not_saturated_and_gets_them_next():
    worker = PageAuditWorker()

    class Llm:
        def invoke(self, _messages):
            return type("R", (), {"content": '{"missing_controls": []}'})()

    worker.configure(Llm())
    worker._process_job({
        "page_id": "page_1", "fingerprint": "fp", "messages": [("human", "x")],
        "screenshot_digests": ["a", "b"], "pending_screenshots": 3,
    })
    worker.collect_results()
    worker.record_result("page_1", 0)
    worker.record_result("page_1", 0)
    assert not worker._audit_state["page_1"].get("saturated")
    assert worker.audited_screenshot_digests("page_1") == {"a", "b"}
    # An unchanged fingerprint can be resubmitted to audit screenshots not yet examined.
    assert worker.filter_candidates([{"page_id": "page_1", "fingerprint": "fp"}])


def test_audit_screenshot_selection_prefers_unaudited(monkeypatch):
    monkeypatch.setattr(updater, "PAGE_AUDIT_MAX_IMAGES", 3)
    shots = [_screenshot(10 + index * 20) for index in range(6)]
    audited = {updater.screenshot_digest(item) for item in shots[:4]}
    monkeypatch.setattr(
        updater.page_audit_worker, "audited_screenshot_digests", lambda _page_id: audited,
    )
    selected = updater._select_page_audit_screenshots("page_1", shots)
    assert len(selected) == 3
    assert shots[4] in selected and shots[5] in selected


def test_observed_page_resolution_reuses_thumbnail_cache(monkeypatch):
    decoded = []
    real = updater.screenshot_thumbnail

    def counting(screenshot, size):
        decoded.append(1)
        return real(screenshot, size)

    monkeypatch.setattr(updater, "screenshot_thumbnail", counting)
    cache = updater._ScreenshotThumbnailCache()
    graph = _graph()
    graph.get_page("page_1").update_representative_screenshot(_screenshot(10))
    step = {"current_page": Page(activity="HomeActivity", representative_screenshot=_screenshot(12))}
    for _ in range(5):
        updater._resolve_observed_page_id(graph, step, 1, None, cache)
    assert len(decoded) == 2


# Repair passes let the model fill omissions without rolling back valid changes.

def _two_step_state(graph):
    return {
        "ffg": graph,
        "current_plan": {"scenario_type": "intra_test", "target_function_id": "func_1"},
        "current_test_scenario": "\u6253\u5f00\u8be6\u60c5",
        "navigation_execution_trace": [],
        "scenario_execution_trace": [
            {"action_name": "click", "action_description": "\u70b9\u51fb\u6761\u76ee",
             "current_page": Page(id="page_1", activity="HomeActivity",
                                  representative_screenshot=_screenshot(10))},
            {"action_name": "finished",
             "current_page": Page(activity="DetailActivity", representative_screenshot=_screenshot(120))},
        ],
        "execution_result": {"navigation_status": "skipped", "scenario_status": "completed"},
    }


def test_repair_pass_lets_model_name_page_it_missed(monkeypatch):
    """Let the repair model supply a semantically named missing page rather than constructing a coarse fallback."""
    responses = [
        '{"pages": []}',
        '{"observation_alignment":[{"source_step":2,"page_decision":"create","page_ref":"new_page_1"}],'
        '"pages":[{"page_ref":"new_page_1","function_refs":["func_1"],"source_step":2,'
        '"name":"\u8ba2\u5355\u8be6\u60c5\u9875","controls":["\u53d6\u6d88\u8ba2\u5355\u6309\u94ae"]}]}',
    ]
    seen = []

    class Llm:
        def invoke(self, messages):
            seen.append(messages[1].content)
            return type("R", (), {"content": responses[min(len(seen) - 1, len(responses) - 1)]})()

    monkeypatch.setattr(updater, "_submit_page_audit_batch", lambda *a, **k: 0)
    graph = _graph()
    graph.get_page("page_1").update_representative_screenshot(_screenshot(10))
    result = updater.create_ffg_update_node(Llm())(_two_step_state(graph))

    assert len(seen) == 2
    repair_text = " ".join(block.get("text", "") for block in seen[1] if block.get("type") == "text")
    assert "missing observation_alignment for source_step 2" in repair_text
    names = [page.name for page in graph.pages.values()]
    assert names == ["\u9996\u9875", "\u8ba2\u5355\u8be6\u60c5\u9875"]
    assert "\u53d6\u6d88\u8ba2\u5355\u6309\u94ae" in graph.pages[[pid for pid in graph.pages if pid != "page_1"][0]].controls
    assert result["ffg_update_validation_issues"] == []


def test_program_fallback_only_after_repair_budget(monkeypatch):
    monkeypatch.setattr(updater, "FFG_UPDATE_MAX_STEPS", 2)
    calls = []

    class Llm:
        def invoke(self, _messages):
            calls.append(1)
            return type("R", (), {"content": '{"pages": []}'})()

    monkeypatch.setattr(updater, "_submit_page_audit_batch", lambda *a, **k: 0)
    graph = _graph()
    graph.get_page("page_1").update_representative_screenshot(_screenshot(10))
    updater.create_ffg_update_node(Llm())(_two_step_state(graph))

    assert len(calls) == 2
    created = [page for page_id, page in graph.pages.items() if page_id != "page_1"]
    assert [page.activity for page in created] == ["DetailActivity"]


def test_repair_pass_sends_only_focused_screenshots(monkeypatch):
    monkeypatch.setattr(updater, "FFG_UPDATE_MAX_REFINEMENT_IMAGES", 3)
    monkeypatch.setattr(updater, "_compact_screenshot_for_llm", lambda value: value)
    shots = [_screenshot(20 + index * 40) for index in range(5)]
    trace = [
        {"raw_step_index": index + 1, "action_name": "wait",
         "current_page": Page(activity=f"Novel{index}", representative_screenshot=shot)}
        for index, shot in enumerate(shots)
    ]
    content = updater._build_repair_message_content(
        [{"type": "text", "text": "ctx"}],
        trace,
        {index + 1: shot for index, shot in enumerate(shots)},
        ["missing observation_alignment for source_step 3"],
        FlowGraph(),
        {},
        2,
    )
    images = [block["image_url"]["url"].split(",", 1)[1] for block in content if block.get("type") == "image_url"]
    assert images == shots[1:4]
    assert all(block["image_url"]["url"].startswith("data:image/jpeg;base64,")
               for block in content if block.get("type") == "image_url")


def test_text_similarity_pruning_keeps_exact_ranking():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="\u529f\u80fd"))
    for index in range(30):
        graph.add_page(Page(id=f"page_{index}", name=f"\u9875\u9762{index}", activity="A",
                            controls=[f"\u6309\u94ae{index}"]), "func_1")
    graph.add_page(Page(id="page_target", name="\u8bbe\u7f6e\u4e2d\u5fc3", activity="A",
                        controls=["\u901a\u77e5\u5f00\u5173", "\u9690\u79c1\u8bbe\u7f6e"]), "func_1")
    step = {"current_page": Page(name="\u8bbe\u7f6e\u4e2d\u5fc3", activity="A", controls=["\u901a\u77e5\u5f00\u5173", "\u9690\u79c1\u8bbe\u7f6e"]),
            "action_description": "", "thought": ""}
    candidates = updater._retrieve_page_candidates(graph, step, {}, limit=3)
    assert candidates[0]["page_id"] == "page_target"


def test_exploration_boundary_block_does_not_retire_function():
    graph = _graph()
    state = {"current_plan": {"scenario_type": "intra_test", "target_function_id": "func_1"}}
    updater._record_execution_outcome(graph, state, {
        "navigation_status": "skipped",
        "scenario_status": "blocked",
        "scenario_error": "\u68c0\u6d4b\u5230\u8fdb\u5165\u5c0f\u7a0b\u5e8f（\u975e\u76ee\u6807\u5e94\u7528\u6838\u5fc3\u529f\u80fd），\u5df2\u81ea\u52a8\u8fd4\u56de\u5e76\u7ed3\u675f\u5f53\u524d\u573a\u666f",
    })
    assert graph.get_node_test_status("func_1") == "in_progress"
    updater._record_execution_outcome(graph, state, {
        "navigation_status": "skipped", "scenario_status": "blocked", "scenario_error": "\u51fa\u73b0\u9a8c\u8bc1\u7801",
    })
    assert graph.get_node_test_status("func_1") == "unavailable"
