# Multilingual UI and model-response fixtures use Unicode escapes to preserve their exact text.
from __future__ import annotations

import json

import pytest

from model.flow_graph import (
    FlowGraph,
    FunctionDependency,
    FunctionNode,
    Page,
    PageTransition,
    adjusted_control_coverage,
    control_family_key,
    operation_family_key,
)
from workflow.subgraphs import ffg_update as ffg_update_subgraph_module
from workflow.nodes import scenario_plan as scenario_plan_module
from workflow.page_audit_worker import page_audit_worker
from workflow.nodes.ffg_update import (
    _build_ffg_summary,
    _build_observation_candidate_map,
    _combined_execution_trace,
    _normalize_observation_alignment,
    _select_granularity_review_function,
    apply_ffg_patch,
    validate_observation_alignment,
    validate_ffg_invariants,
)


@pytest.fixture(autouse=True)
def _legacy_text_matching_enabled(monkeypatch):
    """Enable legacy text matching explicitly to test its false-coverage and mismatch guards. Explicit operation bindings remain authoritative and text fallback is disabled by default."""
    from model import flow_graph as _flow_graph

    monkeypatch.setattr(_flow_graph, "COVERAGE_LEGACY_TEXT_MATCHING", True)


def test_navigation_and_scenario_traces_are_both_used_for_ffg_updates():
    state = {
        "navigation_execution_trace": [{"action_name": "click"}],
        "scenario_execution_trace": [{"action_name": "finished"}],
    }

    trace = _combined_execution_trace(state)

    assert [step["trace_phase"] for step in trace] == ["navigation", "test"]


def test_update_summary_keeps_one_hop_boundary_page_and_transition():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="\u9996\u9875"))
    graph.add_node(FunctionNode(id="func_2", name="\u6536\u85cf"))
    graph.add_page(Page(id="page_1", name="\u9996\u9875", activity="HomeActivity"), "func_1")
    graph.add_page(Page(id="page_2", name="\u6536\u85cf\u5217\u8868", activity="CollectionActivity"), "func_2")
    graph.add_page_transition(PageTransition("page_1", "page_2", "\u70b9\u51fb\u6536\u85cf\u5165\u53e3"))
    trace = [{
        "raw_step_index": 1,
        "current_page": Page(id="page_1", name="\u9996\u9875", activity="HomeActivity"),
    }]

    summary = _build_ffg_summary(
        graph,
        {"target_function_id": "func_1"},
        trace,
    )
    facts = json.loads(summary.split("\n", 1)[1])

    assert facts["boundary_pages"] == [{
        "id": "page_2",
        "name": "\u6536\u85cf\u5217\u8868",
        "activity": "CollectionActivity",
        "owners": ["func_2"],
    }]
    assert facts["boundary_transitions"] == [{
        "src_page_id": "page_1",
        "dst_page_id": "page_2",
        "operation": "\u70b9\u51fb\u6536\u85cf\u5165\u53e3",
    }]


def test_observation_candidates_rank_matching_activity_and_semantics_first():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="\u9996\u9875"))
    graph.add_node(FunctionNode(id="func_2", name="\u7f16\u8f91"))
    graph.add_page(Page(id="page_1", name="\u9996\u9875", activity="HomeActivity"), "func_1")
    graph.add_page(Page(
        id="page_2",
        name="\u7f16\u8f91\u5185\u5bb9",
        description="\u7f16\u8f91\u6807\u9898\u548c\u6b63\u6587",
        controls=["\u6807\u9898\u8f93\u5165\u6846", "\u6b63\u6587\u8f93\u5165\u6846"],
        activity="EditActivity",
    ), "func_2")
    trace = [{
        "raw_step_index": 4,
        "action_name": "finished",
        "current_page": Page(
            name="\u7f16\u8f91\u5185\u5bb9",
            description="\u7f16\u8f91\u6807\u9898\u548c\u6b63\u6587",
            controls=["\u6807\u9898\u8f93\u5165\u6846"],
            activity="EditActivity",
        ),
    }]

    candidate_map, _step_similarity_cache = _build_observation_candidate_map(
        graph,
        {"target_function_id": "func_2"},
        trace,
    )

    assert candidate_map[4][0]["page_id"] == "page_2"
    assert candidate_map[4][0]["activity_match"] is True
    assert candidate_map[4][0]["scope_match"] is True


def test_observation_alignment_maps_reused_page_without_pages_patch():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="\u9996\u9875"))
    graph.add_page(Page(
        id="page_1",
        name="\u9996\u9875",
        representative_screenshot="image-stable",
    ), "func_1")
    trace = [{
        "raw_step_index": 2,
        "current_page": Page(
            name="\u9996\u9875",
            representative_screenshot="image-expanded-panel",
        ),
    }]
    patch = {
        "observation_alignment": [{
            "source_step": 2,
            "page_decision": "reuse",
            "page_ref": "page_1",
        }],
    }

    result = apply_ffg_patch(graph, patch, trace)

    assert result["trace_page_map"] == {2: "page_1"}
    assert result["persisted_trace_images"] == 1
    assert graph.get_page("page_1").representative_screenshot == "image-stable"
    assert graph.get_page("page_1").evidence_screenshots == ["image-expanded-panel"]
    assert validate_observation_alignment(
        graph,
        patch,
        trace,
        {2: [{"page_id": "page_1"}]},
    ) == []


def test_observation_alignment_reports_missing_observed_step():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="\u9996\u9875"))
    graph.add_page(Page(id="page_1", name="\u9996\u9875"), "func_1")
    trace = [{"raw_step_index": 1, "current_page": Page(name="\u672a\u77e5\u9875\u9762")}]

    issues = validate_observation_alignment(graph, {}, trace, {})

    assert issues == ["missing observation_alignment for source_step 1"]


def test_alignment_reuse_of_same_patch_page_is_normalized_to_create():
    patch = {
        "pages": [{"page_ref": "new_page_1", "source_step": 1}],
        "observation_alignment": [{
            "source_step": 1,
            "page_decision": "reuse",
            "page_ref": "new_page_1",
        }],
    }

    alignments = _normalize_observation_alignment(
        patch,
        FlowGraph(),
        [{"raw_step_index": 1, "current_page": Page(name="\u65b0\u9875\u9762")}],
        {},
    )

    assert alignments[0]["page_decision"] == "create"


def test_non_ignorable_observation_with_page_patch_is_normalized_to_create():
    patch = {
        "pages": [{"page_ref": "new_page_1", "source_step": 1}],
        "observation_alignment": [{
            "source_step": 1,
            "page_decision": "ignore",
            "page_ref": "",
        }],
    }

    alignments = _normalize_observation_alignment(
        patch,
        FlowGraph(),
        [{"raw_step_index": 1, "current_page": Page(activity="NovelActivity")}],
        {},
    )

    assert alignments[0] == {
        "source_step": 1,
        "page_decision": "create",
        "page_ref": "new_page_1",
    }


def test_external_interruption_ignore_is_not_forced_to_create():
    patch = {
        "observation_alignment": [{
            "source_step": 1,
            "page_decision": "ignore",
            "page_ref": "",
        }],
    }

    alignments = _normalize_observation_alignment(
        patch,
        FlowGraph(),
        [{
            "raw_step_index": 1,
            "current_page": Page(activity="ExternalActivity"),
            "is_interruption": True,
            "external_context": True,
        }],
        {},
    )

    assert alignments[0]["page_decision"] == "ignore"


def test_new_activity_ignore_is_reported_without_page_patch():
    graph = FlowGraph()
    trace = [{
        "raw_step_index": 1,
        "current_page": Page(activity="NovelActivity"),
    }]
    patch = {
        "observation_alignment": [{
            "source_step": 1,
            "page_decision": "ignore",
            "page_ref": "",
        }],
    }

    _normalize_observation_alignment(patch, graph, trace, {})

    assert validate_observation_alignment(graph, patch, trace, {}) == [
        "ignore alignment lacks an ignorable observation at source_step 1"
    ]


def test_ffg_subgraph_keeps_observed_page_when_alignment_is_missing(monkeypatch):
    """Keep observed pages and audit results even when the model omits alignment; do not roll back the iteration."""
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="\u9996\u9875"))
    graph.add_page(Page(id="page_1", name="\u9996\u9875", activity="HomeActivity"), "func_1")

    class MissingAlignmentLlm:
        class Response:
            content = (
                '{"update_control":{"decision":"finish","reason":"done"},'
                '"functions":[],"pages":[{"page_ref":"new_page_1",'
                '"function_refs":["func_1"],"source_step":1,'
                '"name":"\u8be6\u60c5\u9875","description":"\u8be6\u60c5\u5185\u5bb9"}],'
                '"page_ownership":[],"function_dependencies":[]}'
            )

        def invoke(self, _messages):
            return self.Response()

    page_audit_worker.reset()
    with page_audit_worker._lock:
        page_audit_worker._results.append({
            "page_id": "page_1",
            "fingerprint": "fp",
            "missing_controls": ["\u53f3\u4e0a\u89d2\u66f4\u591a\u6309\u94ae"],
            "control_merges": [],
            "invalid_operations": [],
        })
    monkeypatch.setattr(
        "workflow.nodes.ffg_update._submit_page_audit_batch",
        lambda *_args, **_kwargs: 0,
    )
    try:
        subgraph = ffg_update_subgraph_module.create_ffg_update_subgraph(MissingAlignmentLlm())
        result = subgraph.invoke({
            "ffg": graph,
            "current_page": Page(id="page_1", name="\u9996\u9875", activity="HomeActivity"),
            "current_plan": {
                "scenario_type": "intra_test",
                "target_function_id": "func_1",
            },
            "navigation_execution_trace": [],
            "scenario_execution_trace": [{
                "raw_step_index": 1,
                "action_name": "finished",
                "current_page": Page(name="\u8be6\u60c5\u9875", activity="DetailActivity"),
            }],
            "execution_result": {
                "navigation_status": "skipped",
                "scenario_status": "completed",
            },
            "current_testing_function_id": "func_1",
            "current_testing_function_status": "in_progress",
        })
    finally:
        page_audit_worker.reset()

    graph_after = result["ffg"]
    assert graph_after.get_page("page_1").controls == ["\u53f3\u4e0a\u89d2\u66f4\u591a\u6309\u94ae"]
    new_pages = [page for page_id, page in graph_after.pages.items() if page_id != "page_1"]
    assert [page.name for page in new_pages] == ["\u8be6\u60c5\u9875"]
    assert graph_after.get_page_owners(new_pages[0].id) == ["func_1"]
    assert result["ffg_update_decision"] == "finish"


def test_planner_refreshes_stale_page_id_from_live_visual(monkeypatch):
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="\u6d88\u606f\u5217\u8868"))
    graph.add_page(Page(
        id="page_1",
        name="\u6d88\u606f\u5217\u8868",
        activity="LauncherUI",
        representative_screenshot="home-image",
    ), "func_1")
    graph.add_page(Page(
        id="page_2",
        name="\u804a\u5929\u9875",
        activity="LauncherUI",
        representative_screenshot="chat-image",
    ), "func_1")
    monkeypatch.setattr(
        scenario_plan_module,
        "capture_current_page",
        lambda _device: Page(
            activity="LauncherUI",
            representative_screenshot="home-image",
        ),
    )

    refreshed = scenario_plan_module._refresh_current_page_observation(
        graph,
        object(),
        Page(id="page_2", activity="LauncherUI", representative_screenshot="chat-image"),
    )

    assert refreshed.id == "page_1"
    assert refreshed.name == "\u6d88\u606f\u5217\u8868"


def test_planner_does_not_force_unmatched_visual_to_only_activity_page(monkeypatch):
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="\u6d88\u606f\u5217\u8868"))
    graph.add_page(Page(
        id="page_1",
        name="\u6d88\u606f\u5217\u8868",
        activity="LauncherUI",
        representative_screenshot="home-image",
    ), "func_1")
    monkeypatch.setattr(
        scenario_plan_module,
        "capture_current_page",
        lambda _device: Page(
            activity="LauncherUI",
            representative_screenshot="unknown-chat-image",
        ),
    )

    refreshed = scenario_plan_module._refresh_current_page_observation(
        graph,
        object(),
        Page(id="page_1", activity="LauncherUI", representative_screenshot="home-image"),
    )

    assert refreshed.id is None


def test_existing_page_merges_changed_state_without_replacing_stable_image():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="\u4efb\u52a1\u5217\u8868"))
    graph.add_page(Page(
        id="page_1",
        name="\u4efb\u52a1\u5217\u8868",
        representative_screenshot="image-closed",
        controls=["\u4e00\u76f4\u53ef\u89c1\u7684‘\u6392\u5e8f’\u6309\u94ae"],
        operations=["\u70b9\u51fb\u6392\u5e8f\u6309\u94ae"],
    ), "func_1")
    trace = [
        {
            "raw_step_index": 1,
            "action_name": "click",
            "action_description": "\u70b9\u51fb\u6392\u5e8f\u6309\u94ae",
            "current_page": Page(representative_screenshot="image-closed"),
        },
        {
            "raw_step_index": 2,
            "action_name": "finished",
            "current_page": Page(representative_screenshot="image-dropdown"),
        },
    ]
    patch = {
        "pages": [{
            "page_ref": "page_1",
            "function_refs": ["func_1"],
            "source_step": 2,
            "name": "\u4efb\u52a1\u5217\u8868",
            "controls": ["\u5c55\u5f00\u6392\u5e8f\u4e0b\u62c9\u6846\u540e\u53ef\u89c1\u7684‘\u6309\u65f6\u95f4’\u9009\u9879"],
        }],
    }

    first = apply_ffg_patch(graph, patch, trace, {"target_function_id": "func_1"})
    second = apply_ffg_patch(graph, patch, trace, {"target_function_id": "func_1"})

    page = graph.get_page("page_1")
    assert page is not None
    assert page.representative_screenshot == "image-closed"
    assert page.evidence_screenshots == ["image-dropdown"]
    assert page.controls == [
        "\u4e00\u76f4\u53ef\u89c1\u7684‘\u6392\u5e8f’\u6309\u94ae",
        "\u5c55\u5f00\u6392\u5e8f\u4e0b\u62c9\u6846\u540e\u53ef\u89c1\u7684‘\u6309\u65f6\u95f4’\u9009\u9879",
    ]
    assert page.operations == ["\u70b9\u51fb\u6392\u5e8f\u6309\u94ae"]
    assert graph.page_transitions == []
    assert first["validation_issues"] == []
    assert second["validation_issues"] == []
    assert validate_ffg_invariants(graph) == []


def test_new_independent_function_entry_is_removed_from_parent_shell_owner():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="\u5fae\u4fe1\u4f1a\u8bdd"))
    graph.add_page(Page(id="page_1", name="\u804a\u5929\u9875"), "func_1")
    trace = [{
        "raw_step_index": 2,
        "current_page": Page(name="\u8868\u60c5\u5546\u5e97", activity="EmojiStoreActivity"),
    }]

    result = apply_ffg_patch(graph, {
        "functions": [{
            "function_ref": "new_func_1",
            "source_step": 2,
            "name": "\u8868\u60c5\u5305\u6d4f\u89c8\u4e0e\u7ba1\u7406",
            "description": "\u6d4f\u89c8、\u6dfb\u52a0\u548c\u9884\u89c8\u8868\u60c5\u5305",
            "coverage_percent": 10,
        }],
        "pages": [{
            "page_ref": "new_page_1",
            "function_refs": ["new_func_1", "func_1"],
            "source_step": 2,
            "name": "\u8868\u60c5\u5546\u5e97",
            "description": "\u6d4f\u89c8\u8868\u60c5\u5305",
        }],
    }, trace, {"target_function_id": "func_1"})

    new_function_id = result["function_refs"]["new_func_1"]
    new_page_id = result["page_refs"]["new_page_1"]
    assert graph.get_node(new_function_id).name == "\u8868\u60c5\u5305\u6d4f\u89c8\u4e0e\u7ba1\u7406"
    assert graph.get_page_owners(new_page_id) == [new_function_id]
    assert graph.get_node(new_function_id).page_ids == [new_page_id]
    assert graph.get_node(new_function_id).coverage_percent == 0
    assert result["validation_issues"] == []


def test_named_dynamic_instances_are_not_auto_merged_by_app_specific_vocabulary():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="\u5fae\u4fe1\u4f1a\u8bdd\u5217\u8868\u6d4f\u89c8\u4e0e\u8fdb\u5165"))
    graph.add_node(FunctionNode(
        id="func_2",
        name="\u6587\u4ef6\u4f20\u8f93\u52a9\u624b\u4f1a\u8bdd\u6d88\u606f\u6d4f\u89c8\u4e0e\u6536\u53d1",
        description="\u6d4f\u89c8\u4f1a\u8bdd\u6d88\u606f\u5e76\u53d1\u9001\u5185\u5bb9",
    ))
    graph.add_page(Page(id="page_1", name="\u5fae\u4fe1\u4f1a\u8bdd\u5217\u8868"), "func_1")
    graph.add_page(Page(
        id="page_2",
        name="\u6587\u4ef6\u4f20\u8f93\u52a9\u624b\u804a\u5929\u9875\u9762",
        activity=".ui.LauncherUI",
    ), "func_2")
    trace = [{
        "raw_step_index": 2,
        "current_page": Page(
            name="\u8054\u7cfb\u4eba\u804a\u5929\u9875\u9762",
            activity=".ui.LauncherUI",
        ),
    }]

    result = apply_ffg_patch(graph, {
        "functions": [{
            "function_ref": "new_func_1",
            "source_step": 2,
            "name": "\u8054\u7cfb\u4eba\u4f1a\u8bdd\u6d88\u606f\u6d4f\u89c8\u4e0e\u6536\u53d1",
            "description": "\u6d4f\u89c8\u8054\u7cfb\u4eba\u6d88\u606f\u5e76\u53d1\u9001\u5185\u5bb9",
        }],
        "pages": [{
            "page_ref": "new_page_1",
            "function_refs": ["func_1", "new_func_1"],
            "source_step": 2,
            "name": "\u8054\u7cfb\u4eba\u804a\u5929\u9875\u9762",
            "description": "\u8054\u7cfb\u4eba\u6d88\u606f\u6536\u53d1\u9875\u9762",
        }],
    }, trace, {"target_function_id": "func_1"})

    new_function_id = result["function_refs"]["new_func_1"]
    new_page_id = result["page_refs"]["new_page_1"]
    assert new_function_id != "func_2"
    assert new_page_id != "page_2"
    assert list(graph.functions) == ["func_1", "func_2", new_function_id]
    assert list(graph.pages) == ["page_1", "page_2", new_page_id]
    assert graph.get_page_owners("page_2") == ["func_2"]
    assert graph.get_page_owners(new_page_id) == [new_function_id]
    assert result["validation_issues"] == []


def test_dynamic_chat_pages_stay_separate_when_functions_are_already_distinct():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="\u6587\u4ef6\u4f20\u8f93\u52a9\u624b\u4f1a\u8bdd"))
    graph.add_node(FunctionNode(id="func_2", name="\u6d4b\u8bd5\u7fa4\u4f1a\u8bdd"))
    graph.add_page(Page(
        id="page_1",
        name="\u6587\u4ef6\u4f20\u8f93\u52a9\u624b\u804a\u5929\u9875\u9762",
        activity=".ui.LauncherUI",
    ), "func_1")
    trace = [{
        "raw_step_index": 1,
        "current_page": Page(name="\u8054\u7cfb\u4eba\u804a\u5929\u9875\u9762", activity=".ui.LauncherUI"),
    }]

    result = apply_ffg_patch(graph, {
        "pages": [{
            "page_ref": "new_page_1",
            "function_refs": ["func_2"],
            "source_step": 1,
            "name": "\u8054\u7cfb\u4eba\u804a\u5929\u9875\u9762",
        }],
    }, trace, {"target_function_id": "func_2"})

    new_page_id = result["page_refs"]["new_page_1"]
    assert new_page_id != "page_1"
    assert graph.get_page_owners("page_1") == ["func_1"]
    assert graph.get_page_owners(new_page_id) == ["func_2"]


def test_page_ownership_patch_can_split_page_from_coarse_function():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="\u5fae\u4fe1\u4f1a\u8bdd"))
    graph.add_page(Page(id="page_1", name="\u4f1a\u8bdd\u5217\u8868"), "func_1")
    graph.add_page(Page(id="page_2", name="\u804a\u5929\u8bb0\u5f55\u67e5\u627e"), "func_1")
    trace = [{
        "raw_step_index": 3,
        "current_page": Page(id="page_2", name="\u804a\u5929\u8bb0\u5f55\u67e5\u627e"),
    }]

    result = apply_ffg_patch(graph, {
        "functions": [{
            "function_ref": "new_func_1",
            "source_step": 3,
            "name": "\u804a\u5929\u8bb0\u5f55\u641c\u7d22",
            "description": "\u6309\u5173\u952e\u8bcd\u6216\u5185\u5bb9\u7c7b\u578b\u67e5\u627e\u804a\u5929\u8bb0\u5f55",
        }],
        "page_ownership": [{
            "page_ref": "page_2",
            "add_function_refs": ["new_func_1"],
            "remove_function_refs": ["func_1"],
            "source_step": 3,
            "reason": "\u76f4\u63a5\u89c2\u5bdf\u5230\u72ec\u7acb\u641c\u7d22\u95ed\u73af",
        }],
    }, trace, {"target_function_id": "func_1"})

    new_function_id = result["function_refs"]["new_func_1"]
    assert graph.get_page_owners("page_2") == [new_function_id]
    assert graph.get_node("func_1").page_ids == ["page_1"]
    assert graph.get_node(new_function_id).page_ids == ["page_2"]
    assert graph.get_node(new_function_id).entry_page_ids == []
    assert result["validation_issues"] == []


def test_oversized_function_can_be_atomically_replaced_from_persisted_pages():
    graph = FlowGraph()
    graph.add_node(FunctionNode(
        id="func_1",
        name="\u4e2a\u4eba\u4e2d\u5fc3\u4e0e\u8bbe\u7f6e",
        scenarios=[
            "\u70b9\u51fb\u8bbe\u7f6e\u8fdb\u5165\u8bbe\u7f6e\u9875",
            "\u70b9\u51fb\u8d26\u53f7\u5b89\u5168\u67e5\u770b\u8bbe\u5907",
            "\u70b9\u51fb\u6536\u85cf\u8fdb\u5165\u6536\u85cf\u9875",
            "\u70b9\u51fb\u5361\u5305\u67e5\u770b\u5361\u5238",
        ],
        test_status="in_progress",
    ))
    for index, name in enumerate((
        "\u4e2a\u4eba\u4e2d\u5fc3\u9875",
        "\u8bbe\u7f6e\u9875",
        "\u8d26\u53f7\u5b89\u5168\u9875",
        "\u6536\u85cf\u9875",
        "\u5361\u5305\u9875",
    ), start=1):
        graph.add_page(Page(
            id=f"page_{index}",
            name=name,
            activity=f"Activity{index}",
            operations=[f"\u70b9\u51fb{name}\u5165\u53e3"],
        ), "func_1")
    graph.add_node(FunctionNode(id="func_2", name="\u9996\u9875"))
    graph.add_page(Page(id="page_0", name="\u5fae\u4fe1\u9996\u9875"), "func_2")
    graph.add_entry_page(graph.get_page("page_0"), "func_1")

    result = apply_ffg_patch(graph, {
        "functions": [{
            "function_ref": "new_func_1",
            "name": "\u8d26\u53f7\u4e0e\u7cfb\u7edf\u8bbe\u7f6e",
            "description": "\u7ba1\u7406\u8d26\u53f7\u5b89\u5168\u548c\u7cfb\u7edf\u8bbe\u7f6e",
        }, {
            "function_ref": "new_func_2",
            "name": "\u4e2a\u4eba\u5185\u5bb9\u4e0e\u8d44\u4ea7",
            "description": "\u67e5\u770b\u6536\u85cf、\u5361\u5305\u7b49\u4e2a\u4eba\u5185\u5bb9",
        }],
        "function_restructures": [{
            "source_function_ref": "func_1",
            "replacements": [{
                "function_ref": "new_func_1",
                "page_refs": ["page_1", "page_2", "page_3"],
                "entry_page_refs": ["page_0"],
                "scenario_refs": ["scenario_1", "scenario_2"],
            }, {
                "function_ref": "new_func_2",
                "page_refs": ["page_4", "page_5"],
                "entry_page_refs": ["page_0"],
                "scenario_refs": ["scenario_3", "scenario_4"],
            }],
            "retire_source": True,
            "reason": "\u8bbe\u7f6e\u548c\u4e2a\u4eba\u8d44\u4ea7\u5df2\u5f62\u6210\u72ec\u7acb\u95ed\u73af",
        }],
    }, [{
        "raw_step_index": 1,
        "current_page": Page(id="page_1", name="\u4e2a\u4eba\u4e2d\u5fc3\u9875"),
    }], {"target_function_id": "func_1"})

    settings_id = result["function_refs"]["new_func_1"]
    assets_id = result["function_refs"]["new_func_2"]
    assert graph.get_node("func_1") is None
    assert graph.get_node(settings_id).coverage_page_ids == [
        "page_1", "page_2", "page_3",
    ]
    assert graph.get_node(assets_id).coverage_page_ids == ["page_4", "page_5"]
    assert graph.get_node(assets_id).entry_page_ids == ["page_0"]
    assert graph.get_node(settings_id).scenarios == [
        "\u70b9\u51fb\u8bbe\u7f6e\u8fdb\u5165\u8bbe\u7f6e\u9875",
        "\u70b9\u51fb\u8d26\u53f7\u5b89\u5168\u67e5\u770b\u8bbe\u5907",
    ]
    assert graph.get_node(assets_id).scenarios == [
        "\u70b9\u51fb\u6536\u85cf\u8fdb\u5165\u6536\u85cf\u9875",
        "\u70b9\u51fb\u5361\u5305\u67e5\u770b\u5361\u5238",
    ]
    assert result["function_replacements"] == {
        "func_1": [settings_id, assets_id],
    }
    assert result["validation_issues"] == []
    assert validate_ffg_invariants(graph) == []


def test_full_function_split_keeps_source_when_scenario_evidence_would_be_lost():
    graph = FlowGraph()
    graph.add_node(FunctionNode(
        id="func_1",
        name="\u7c97\u7c92\u5ea6\u529f\u80fd",
        scenarios=["\u573a\u666f\u4e00", "\u573a\u666f\u4e8c"],
    ))
    graph.add_page(Page(id="page_1", name="\u9875\u9762\u4e00"), "func_1")
    graph.add_page(Page(id="page_2", name="\u9875\u9762\u4e8c"), "func_1")

    result = apply_ffg_patch(graph, {
        "functions": [
            {"function_ref": "new_func_1", "name": "\u529f\u80fd\u4e00", "description": "\u9875\u9762\u4e00"},
            {"function_ref": "new_func_2", "name": "\u529f\u80fd\u4e8c", "description": "\u9875\u9762\u4e8c"},
        ],
        "function_restructures": [{
            "source_function_ref": "func_1",
            "replacements": [{
                "function_ref": "new_func_1",
                "page_refs": ["page_1"],
                "scenario_refs": ["scenario_1"],
            }, {
                "function_ref": "new_func_2",
                "page_refs": ["page_2"],
                "scenario_refs": [],
            }],
            "retire_source": True,
        }],
    }, [{"raw_step_index": 1, "current_page": Page(id="page_1")}])

    assert list(graph.functions) == ["func_1"]
    assert graph.get_node("func_1").page_ids == ["page_1", "page_2"]
    assert any("scenario_2" in issue for issue in result["validation_issues"])


def test_existing_functions_can_be_merged_with_pages_scenarios_and_dependencies():
    graph = FlowGraph()
    graph.add_node(FunctionNode(
        id="func_1",
        name="\u6536\u85cf\u5165\u53e3\u6d4f\u89c8",
        scenarios=["\u4ece\u4e2a\u4eba\u4e2d\u5fc3\u8fdb\u5165\u6536\u85cf"],
        resource_variables=["\u6536\u85cf\u5185\u5bb9"],
        test_status="completed",
    ))
    graph.add_node(FunctionNode(
        id="func_2",
        name="\u6536\u85cf\u5217\u8868\u67e5\u770b",
        scenarios=["\u67e5\u770b\u6536\u85cf\u5217\u8868"],
        resource_variables=["\u6536\u85cf\u5185\u5bb9"],
        test_status="in_progress",
    ))
    graph.add_node(FunctionNode(id="func_3", name="\u4e2a\u4eba\u4e2d\u5fc3"))
    graph.add_page(Page(id="page_1", name="\u4e2a\u4eba\u4e2d\u5fc3"), "func_3")
    graph.add_page(Page(
        id="page_2",
        name="\u6536\u85cf\u5165\u53e3\u9875",
        operations=["\u70b9\u51fb\u6536\u85cf\u5165\u53e3"],
    ), "func_1")
    graph.add_entry_page(graph.get_page("page_1"), "func_1")
    graph.add_page(Page(
        id="page_3",
        name="\u6536\u85cf\u5217\u8868\u9875",
        operations=["\u70b9\u51fb\u6536\u85cf\u6761\u76ee"],
    ), "func_2")
    graph.add_function_dependency(FunctionDependency(
        src_function_id="func_1",
        dst_function_id="func_2",
        resource_variable="\u6536\u85cf\u5185\u5bb9",
    ))

    result = apply_ffg_patch(graph, {
        "function_merges": [{
            "source_function_refs": ["func_1", "func_2"],
            "target_function_ref": "func_1",
            "reason": "\u4e24\u6bb5\u5df2\u786e\u8ba4\u5c5e\u4e8e\u540c\u4e00\u6536\u85cf\u76ee\u6807",
        }],
    }, [{"raw_step_index": 1, "current_page": Page(id="page_2")}])

    assert graph.get_node("func_2") is None
    merged = graph.get_node("func_1")
    assert merged is not None
    assert merged.page_ids == ["page_2", "page_1", "page_3"]
    assert merged.entry_page_ids == ["page_1"]
    assert merged.scenarios == ["\u4ece\u4e2a\u4eba\u4e2d\u5fc3\u8fdb\u5165\u6536\u85cf", "\u67e5\u770b\u6536\u85cf\u5217\u8868"]
    assert merged.test_status == "in_progress"
    assert graph.function_dependencies == []
    assert result["function_replacements"] == {"func_2": ["func_1"]}
    assert result["validation_issues"] == []
    assert validate_ffg_invariants(graph) == []


def test_granularity_review_is_spaced_as_function_grows(monkeypatch):
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="\u4e2a\u4eba\u4e2d\u5fc3\u4e0e\u8bbe\u7f6e"))
    for index in range(1, 13):
        graph.add_page(Page(
            id=f"page_{index}",
            name=f"\u5b50\u9875\u9762{index}",
            activity=f"Activity{index}",
        ), "func_1")

    monkeypatch.setattr(
        "workflow.nodes.ffg_update.FFG_UPDATE_GRANULARITY_MIN_PAGES", 6
    )
    first_id, first_metrics = _select_granularity_review_function(
        graph,
        {"target_function_id": "func_1"},
        {},
    )
    second_id, second_metrics = _select_granularity_review_function(
        graph,
        {"target_function_id": "func_1"},
        {"func_1": 1},
    )
    exhausted_id, _ = _select_granularity_review_function(
        graph,
        {"target_function_id": "func_1"},
        {"func_1": 2},
    )

    assert first_id == second_id == "func_1"
    assert first_metrics["page_threshold"] == 6
    assert second_metrics["page_threshold"] == 12
    assert exhausted_id == ""


def test_visible_business_entry_creates_function_with_shared_entry_page():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="\u4f1a\u8bdd\u6d88\u606f\u6d4f\u89c8\u4e0e\u6536\u53d1"))
    graph.add_page(Page(
        id="page_1",
        name="\u4f1a\u8bdd\u804a\u5929\u9875\u9762",
        controls=["\u76f8\u518c\u5165\u53e3（\u6269\u5c55\u529f\u80fd\u9762\u677f\u5c55\u5f00\u65f6\u53ef\u89c1）"],
    ), "func_1")
    trace = [{
        "raw_step_index": 2,
        "current_page": Page(
            id="page_1",
            name="\u4f1a\u8bdd\u804a\u5929\u9875\u9762",
            controls=["\u76f8\u518c\u5165\u53e3（\u6269\u5c55\u529f\u80fd\u9762\u677f\u5c55\u5f00\u65f6\u53ef\u89c1）"],
        ),
    }]

    result = apply_ffg_patch(graph, {
        "functions": [{
            "function_ref": "new_func_1",
            "source_step": 2,
            "name": "\u4ece\u76f8\u518c\u9009\u62e9\u5e76\u53d1\u9001\u5a92\u4f53",
            "description": "\u4ece\u804a\u5929\u6269\u5c55\u9762\u677f\u4e2d\u7684\u76f8\u518c\u5165\u53e3\u8fdb\u5165，\u5177\u4f53\u6d41\u7a0b\u5f85\u6267\u884c\u786e\u8ba4。",
            "coverage_percent": 0,
        }],
        "page_ownership": [{
            "page_ref": "page_1",
            "add_function_refs": ["new_func_1"],
            "remove_function_refs": [],
            "source_step": 2,
            "reason": "\u9875\u9762\u76f4\u63a5\u663e\u793a\u540d\u79f0\u660e\u786e\u7684\u76f8\u518c\u4e1a\u52a1\u5165\u53e3",
        }],
    }, trace, {"target_function_id": "func_1"})

    new_function_id = result["function_refs"]["new_func_1"]
    assert graph.get_page_owners("page_1") == ["func_1", new_function_id]
    assert graph.get_node(new_function_id).page_ids == ["page_1"]
    assert graph.get_node(new_function_id).entry_page_ids == ["page_1"]
    assert graph.get_node(new_function_id).coverage_page_ids == []
    assert graph.get_node(new_function_id).test_status == "not_started"
    assert graph.get_node(new_function_id).coverage_percent == 0
    assert list(graph.pages) == ["page_1"]
    assert graph.page_transitions == []
    assert result["validation_issues"] == []


def test_existing_function_name_and_description_update_require_direct_evidence():
    graph = FlowGraph()
    graph.add_node(FunctionNode(
        id="func_1",
        name="\u76f8\u518c\u76f8\u5173\u529f\u80fd",
        description="\u4ece\u76f8\u518c\u5165\u53e3\u53d1\u73b0，\u5177\u4f53\u6d41\u7a0b\u5f85\u786e\u8ba4。",
    ))
    graph.add_page(Page(id="page_1", name="\u804a\u5929\u9875\u9762"), "func_1")

    apply_ffg_patch(graph, {
        "functions": [{
            "function_ref": "func_1",
            "source_step": 3,
            "name": "\u4ece\u76f8\u518c\u9009\u62e9\u5e76\u53d1\u9001\u56fe\u7247\u6216\u89c6\u9891",
            "description": "\u9009\u62e9\u8bbe\u5907\u4e2d\u7684\u56fe\u7247\u6216\u89c6\u9891\u5e76\u53d1\u9001\u5230\u5f53\u524d\u4f1a\u8bdd。",
        }],
    }, [{
        "raw_step_index": 3,
        "current_page": Page(id="page_1", name="\u804a\u5929\u9875\u9762"),
    }])

    node = graph.get_node("func_1")
    assert node.name == "\u4ece\u76f8\u518c\u9009\u62e9\u5e76\u53d1\u9001\u56fe\u7247\u6216\u89c6\u9891"
    assert node.description == "\u9009\u62e9\u8bbe\u5907\u4e2d\u7684\u56fe\u7247\u6216\u89c6\u9891\u5e76\u53d1\u9001\u5230\u5f53\u524d\u4f1a\u8bdd。"

    apply_ffg_patch(graph, {
        "functions": [{
            "function_ref": "func_1",
            "source_step": 99,
            "name": "\u6ca1\u6709\u8bc1\u636e\u7684\u91cd\u547d\u540d",
            "description": "\u4e0d\u5e94\u5199\u5165。",
        }],
    }, [{
        "raw_step_index": 4,
        "current_page": Page(id="page_1", name="\u804a\u5929\u9875\u9762"),
    }])

    assert node.name == "\u4ece\u76f8\u518c\u9009\u62e9\u5e76\u53d1\u9001\u56fe\u7247\u6216\u89c6\u9891"
    assert node.description == "\u9009\u62e9\u8bbe\u5907\u4e2d\u7684\u56fe\u7247\u6216\u89c6\u9891\u5e76\u53d1\u9001\u5230\u5f53\u524d\u4f1a\u8bdd。"


def test_ffg_patch_cannot_write_function_terminal_status():
    graph = FlowGraph()
    graph.add_node(FunctionNode(
        id="func_1",
        name="\u4ece\u76f8\u518c\u9009\u62e9\u5e76\u53d1\u9001\u5a92\u4f53",
        test_status="in_progress",
    ))
    graph.add_page(Page(id="page_1", name="\u4f1a\u8bdd\u804a\u5929\u9875\u9762"), "func_1")

    apply_ffg_patch(graph, {
        "functions": [{
            "function_ref": "func_1",
            "source_step": 2,
            "test_status": "unavailable",
        }],
    }, [{
        "raw_step_index": 2,
        "current_page": Page(id="page_1", name="\u4f1a\u8bdd\u804a\u5929\u9875\u9762"),
    }])

    assert graph.get_node_test_status("func_1") == "in_progress"

    apply_ffg_patch(graph, {
        "functions": [{
            "function_ref": "func_1",
            "source_step": 99,
            "test_status": "blocked",
        }],
    }, [{
        "raw_step_index": 3,
        "current_page": Page(id="page_1", name="\u4f1a\u8bdd\u804a\u5929\u9875\u9762"),
    }])

    assert graph.get_node_test_status("func_1") == "in_progress"


def test_existing_function_resource_variables_are_merged_from_direct_evidence():
    graph = FlowGraph()
    graph.add_node(FunctionNode(
        id="func_1",
        name="\u6587\u7ae0\u9605\u8bfb",
        resource_variables=["\u9605\u8bfb\u8fdb\u5ea6"],
    ))
    graph.add_page(Page(id="page_1", name="\u6587\u7ae0\u8be6\u60c5"), "func_1")
    trace = [{
        "raw_step_index": 2,
        "current_page": Page(id="page_1", name="\u6587\u7ae0\u8be6\u60c5"),
    }]

    first = apply_ffg_patch(graph, {
        "functions": [{
            "function_ref": "func_1",
            "source_step": 2,
            "resource_variables": ["\u6536\u85cf\u5185\u5bb9", "\u6536\u85cf\u5185\u5bb9"],
        }],
    }, trace)
    second = apply_ffg_patch(graph, {
        "functions": [{
            "function_ref": "func_1",
            "source_step": 2,
            "resource_variables": [],
        }],
    }, trace)

    assert graph.get_node("func_1").resource_variables == ["\u9605\u8bfb\u8fdb\u5ea6", "\u6536\u85cf\u5185\u5bb9"]
    assert first["validation_issues"] == []
    assert second["validation_issues"] == []


def test_existing_function_resource_update_requires_direct_trace_step():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="\u6587\u7ae0\u9605\u8bfb"))

    apply_ffg_patch(graph, {
        "functions": [{
            "function_ref": "func_1",
            "source_step": 9,
            "resource_variables": ["\u6536\u85cf\u5185\u5bb9"],
        }],
    }, [{"raw_step_index": 1, "current_page": Page(name="\u6587\u7ae0\u8be6\u60c5")}])

    assert graph.get_node("func_1").resource_variables == []


def test_function_dependency_syncs_resource_to_both_functions():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="\u6587\u7ae0\u9605\u8bfb"))
    graph.add_node(FunctionNode(id="func_2", name="\u6536\u85cf\u5217\u8868"))
    graph.add_page(Page(id="page_1", name="\u6587\u7ae0\u8be6\u60c5"), "func_1")
    graph.add_page(Page(id="page_2", name="\u6536\u85cf\u5217\u8868"), "func_2")

    result = apply_ffg_patch(graph, {
        "function_dependencies": [{
            "src_function_ref": "func_1",
            "dst_function_ref": "func_2",
            "resource_variable": "\u6536\u85cf\u5185\u5bb9",
            "description": "\u6536\u85cf\u540e\u5728\u6536\u85cf\u5217\u8868\u53ef\u89c1",
        }],
    }, [{"raw_step_index": 1, "current_page": Page(name="\u6536\u85cf\u5217\u8868")}])

    assert graph.get_node("func_1").resource_variables == ["\u6536\u85cf\u5185\u5bb9"]
    assert graph.get_node("func_2").resource_variables == ["\u6536\u85cf\u5185\u5bb9"]
    assert len(graph.function_dependencies) == 1
    assert result["validation_issues"] == []

    graph.get_node("func_1").resource_variables.clear()
    graph.get_node("func_2").resource_variables.clear()
    graph.add_function_dependency(graph.function_dependencies[0])

    assert graph.get_node("func_1").resource_variables == ["\u6536\u85cf\u5185\u5bb9"]
    assert graph.get_node("func_2").resource_variables == ["\u6536\u85cf\u5185\u5bb9"]
    assert len(graph.function_dependencies) == 1


def test_page_control_union_preserves_similar_but_distinct_dropdown_options():
    page = Page(controls=[
        "\u5c55\u5f00\u6392\u5e8f\u4e0b\u62c9\u6846\u540e\u53ef\u89c1\u7684‘\u6309\u65f6\u95f4’\u9009\u9879",
        "\u5c55\u5f00\u6392\u5e8f\u4e0b\u62c9\u6846\u540e\u53ef\u89c1\u7684‘\u6309\u540d\u79f0’\u9009\u9879",
    ])

    assert page.controls == [
        "\u5c55\u5f00\u6392\u5e8f\u4e0b\u62c9\u6846\u540e\u53ef\u89c1\u7684‘\u6309\u65f6\u95f4’\u9009\u9879",
        "\u5c55\u5f00\u6392\u5e8f\u4e0b\u62c9\u6846\u540e\u53ef\u89c1\u7684‘\u6309\u540d\u79f0’\u9009\u9879",
    ]


def test_page_round_trip_has_only_screenshot_and_control_lists():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="\u4e3b\u9875"))
    graph.add_page(Page(
        id="page_1",
        representative_screenshot="image-a",
        evidence_screenshots=["image-b"],
        controls=["\u53ef\u70b9\u51fb\u7684‘\u65b0\u589e’\u6309\u94ae"],
    ), "func_1")

    data = graph.to_dict()
    page_data = data["pages"][0]

    assert page_data["representative_screenshot"] == "image-a"
    assert page_data["evidence_screenshots"] == ["image-b"]
    assert page_data["controls"] == ["\u53ef\u70b9\u51fb\u7684‘\u65b0\u589e’\u6309\u94ae"]
    assert "screenshot" not in page_data
    assert "status" not in page_data
    assert "alias_to" not in page_data
    assert "correction_log" not in data
    restored = FlowGraph.from_dict(data)
    assert restored.get_page("page_1").representative_screenshot == "image-a"
    assert restored.get_page("page_1").evidence_screenshots == ["image-b"]


def test_entry_page_role_survives_flow_graph_round_trip():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="\u4e3b\u9875"))
    graph.add_node(FunctionNode(id="func_2", name="\u76f8\u518c\u9009\u62e9"))
    page = Page(id="page_1", name="\u4e3b\u9875", controls=["\u76f8\u518c\u5165\u53e3"])
    graph.add_page(page, "func_1")
    graph.add_entry_page(page, "func_2")

    restored = FlowGraph.from_dict(graph.to_dict())

    assert restored.get_node("func_2").page_ids == ["page_1"]
    assert restored.get_node("func_2").entry_page_ids == ["page_1"]
    assert restored.get_node("func_2").coverage_page_ids == []


def test_page_owners_are_derived_and_global_transitions_can_cross_functions():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="\u5217\u8868"))
    graph.add_node(FunctionNode(id="func_2", name="\u8be6\u60c5"))
    shared = Page(id="page_shared", name="\u5171\u4eab\u5165\u53e3")
    graph.add_page(shared, "func_1")
    graph.add_page(shared, "func_2")
    graph.add_page(Page(id="page_detail", name="\u8be6\u60c5\u9875"), "func_2")
    graph.add_page_transition(PageTransition(
        src_page_id="page_shared",
        dst_page_id="page_detail",
        operation="\u6253\u5f00\u8be6\u60c5",
    ))

    assert graph.page_owners == {
        "page_shared": ["func_1", "func_2"],
        "page_detail": ["func_2"],
    }
    assert graph.get_node("func_2").page_ids == ["page_shared", "page_detail"]
    assert graph.page_transitions[0].dst_page_id == "page_detail"
    assert "page_owners" not in graph.to_dict()


def test_flow_graph_load_rejects_removed_legacy_structure(tmp_path):
    legacy = {
        "function_nodes": [{
            "id": "func_1",
            "name": "\u4e3b\u9875",
            "entry_page_id": "page_1",
            "test_depth": 7,
            "page_ids": ["page_1", "page_2"],
            "page_transitions": [{
                "src_page_id": "page_1",
                "dst_page_id": "page_2",
                "operation": "\u6253\u5f00\u8be6\u60c5",
            }],
            "test_scenarios": [{"description": "\u6d4f\u89c8\u8be6\u60c5"}],
            "attempted_scenarios": [{"description": "\u70b9\u51fb\u7b5b\u9009"}],
        }],
        "page_registry": [{
            "id": "page_1",
            "screenshots": ["representative", "evidence"],
        }, {
            "id": "page_2",
        }],
        "page_owners": {"page_1": ["func_1"], "page_2": ["func_1"]},
    }

    (tmp_path / "ffg.json").write_text(
        json.dumps(legacy, ensure_ascii=False),
        encoding="utf-8",
    )

    try:
        FlowGraph.load(str(tmp_path))
    except ValueError as exc:
        assert "functions" in str(exc)
    else:
        raise AssertionError("\u65e7 FFG \u7ed3\u6784\u4e0d\u5e94\u88ab\u9759\u9ed8\u52a0\u8f7d")


def test_function_coverage_percent_is_clamped_and_serialized():
    below = FunctionNode(id="func_1", coverage_percent=-1)
    above = FunctionNode(id="func_2", coverage_percent=120)

    assert below.coverage_percent == 0
    assert above.coverage_percent == 100
    assert below.to_dict()["coverage_percent"] == 0
    assert above.to_dict()["coverage_percent"] == 100


def test_llm_function_coverage_is_ignored_and_program_derived():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="\u641c\u7d22", coverage_percent=10))
    graph.add_page(Page(id="page_1", name="\u641c\u7d22\u9875"), "func_1")
    trace = [{
        "raw_step_index": 2,
        "current_page": Page(id="page_1", name="\u641c\u7d22\u9875"),
    }]

    apply_ffg_patch(graph, {
        "functions": [{
            "function_ref": "func_1",
            "source_step": 2,
            "coverage_percent": 35,
        }],
    }, trace)
    assert graph.get_node("func_1").coverage_percent == 0

    apply_ffg_patch(graph, {
        "functions": [{
            "function_ref": "func_1",
            "source_step": 99,
            "coverage_percent": 90,
        }],
    }, trace)
    assert graph.get_node("func_1").coverage_percent == 0


def test_controls_use_simple_exact_dedup_while_operations_remain_deduplicated():
    page = Page(
        controls=[
            "\u6587\u4ef6\u4f20\u8f93\u52a9\u624b\u4f1a\u8bdd\u6761\u76ee",
            "\u6d4b\u8bd5\u4f1a\u8bdd\u6761\u76ee",
            "\u641c\u7d22\u6309\u94ae",
            "\u516c\u4f17\u53f7\u5165\u53e3\u6761\u76ee",
            "\u670d\u52a1\u53f7\u5165\u53e3\u6761\u76ee",
        ],
        operations=[
            "\u70b9\u51fb\u5e95\u90e8\u901a\u8baf\u5f55\u6807\u7b7e",
            "\u70b9\u51fb\u5e95\u90e8\u5bfc\u822a\u680f\u7684\u901a\u8baf\u5f55\u6309\u94ae",
            "\u5c1d\u8bd5\u70b9\u51fb\u641c\u7d22\u6309\u94ae\u5931\u8d25：\u63a7\u4ef6\u65e0\u54cd\u5e94",
        ],
    )

    assert page.controls == [
        "\u6587\u4ef6\u4f20\u8f93\u52a9\u624b\u4f1a\u8bdd\u6761\u76ee",
        "\u6d4b\u8bd5\u4f1a\u8bdd\u6761\u76ee",
        "\u641c\u7d22\u6309\u94ae",
        "\u516c\u4f17\u53f7\u5165\u53e3\u6761\u76ee",
        "\u670d\u52a1\u53f7\u5165\u53e3\u6761\u76ee",
    ]
    assert page.operations == [
        "\u70b9\u51fb\u5e95\u90e8\u901a\u8baf\u5f55\u6807\u7b7e",
        "\u5c1d\u8bd5\u70b9\u51fb\u641c\u7d22\u6309\u94ae\u5931\u8d25：\u63a7\u4ef6\u65e0\u54cd\u5e94",
    ]
    assert operation_family_key(page.operations[0]) == "click:\u901a\u8baf\u5f55"


def test_picker_instances_and_visual_modifiers_share_operation_families():
    assert control_family_key("\u8868\u60c5\u5206\u7c7b\u4e0e\u8868\u60c5\u9879") == "\u8868\u60c5\u9009\u62e9\u9879"
    assert operation_family_key("\u70b9\u51fb\u6240\u6709\u8868\u60c5\u5206\u7c7b\u4e0b\u7b2c\u4e00\u4e2a\u5fae\u7b11\u8868\u60c5") == "click:\u8868\u60c5\u9009\u62e9\u9879"
    assert operation_family_key("\u70b9\u51fb\u5c0f\u9e21\u8868\u60c5\u5206\u7c7b\u4e2d\u7684“\u62dc\u6258”\u8868\u60c5") == "click:\u8868\u60c5\u9009\u62e9\u9879"
    assert operation_family_key("\u70b9\u51fb\u7eff\u8272\u7684\u53d1\u9001\u6309\u94ae") == "click:\u53d1\u9001"


def test_instance_conversation_operations_merge_into_one_family():
    """Merge instance-level conversation operations into their shared conversation-entry family."""
    assert control_family_key("\u5f20\u4e09\u7684\u4f1a\u8bdd") == "\u4f1a\u8bdd\u6761\u76ee"
    assert control_family_key("\u67d0\u4e2a\u4f1a\u8bdd") == "\u4f1a\u8bdd\u6761\u76ee"
    assert control_family_key("\u4f1a\u8bdd\u5217\u8868") == "\u4f1a\u8bdd\u6761\u76ee"
    # Non-instance descriptions remain unchanged.
    assert control_family_key("\u4f1a\u8bdd\u6807\u9898") != "\u4f1a\u8bdd\u6761\u76ee"
    assert (
        operation_family_key("\u70b9\u51fb\u5f20\u4e09\u7684\u4f1a\u8bdd")
        == operation_family_key("\u70b9\u51fb\u674e\u56db\u7684\u4f1a\u8bdd")
        == "click:\u4f1a\u8bdd\u6761\u76ee"
    )
    page = Page(id="p", controls=["\u4f1a\u8bdd\u6761\u76ee"])
    page.add_operation("\u70b9\u51fb\u5f20\u4e09\u7684\u4f1a\u8bdd")
    page.add_operation("\u70b9\u51fb\u674e\u56db\u7684\u4f1a\u8bdd")
    page.add_operation("\u70b9\u51fb\u67d0\u4e2a\u4f1a\u8bdd")
    assert page.operations == ["\u70b9\u51fb\u5f20\u4e09\u7684\u4f1a\u8bdd"]


def test_function_coverage_counts_all_page_controls_without_name_filtering():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="\u53d1\u9001\u8868\u60c5\u6d88\u606f"))
    graph.add_page(Page(
        id="page_1",
        controls=[
            "\u8fd4\u56de\u6309\u94ae",
            "\u4f1a\u8bdd\u6807\u9898",
            "\u6587\u672c\u6d88\u606f",
            "\u56fe\u7247\u6d88\u606f",
            "\u8868\u60c5\u6d88\u606f",
            "\u6d88\u606f\u6587\u672c\u8f93\u5165\u6846",
            "\u8868\u60c5\u6309\u94ae",
            "\u8868\u60c5\u9009\u62e9\u9762\u677f",
            "\u8868\u60c5\u5206\u7c7b\u4e0e\u8868\u60c5\u9879",
            "\u53d1\u9001\u6309\u94ae",
        ],
        operations=[
            "\u70b9\u51fb\u8868\u60c5\u6309\u94ae",
            "\u70b9\u51fb\u5c0f\u9e21\u8868\u60c5\u5206\u7c7b\u4e2d\u7684“\u62dc\u6258”\u8868\u60c5",
            "\u70b9\u51fb\u7eff\u8272\u7684\u53d1\u9001\u6309\u94ae",
        ],
    ), "func_1")

    details = graph.function_coverage_details("func_1")

    # The emoji button action matches that button; selecting an emoji matches the panel, category,
    # and item within the same family, and the send action matches the send button.
    # Although an emoji-message label contains the emoji substring, its family differs.
    # Do not count it as another covered emoji control.
    assert details["total_controls"] == 10
    assert details["executed_operations"] == 4
    assert details["coverage_percent"] == 45


def test_failed_operation_counts_as_executed_but_not_successful():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="\u641c\u7d22"))
    graph.add_page(Page(
        id="page_1",
        controls=["\u641c\u7d22\u6309\u94ae"],
        operations=["\u5c1d\u8bd5\u70b9\u51fb\u641c\u7d22\u6309\u94ae\u5931\u8d25：\u63a7\u4ef6\u65e0\u54cd\u5e94"],
    ), "func_1")
    graph.recompute_all_coverages()

    details = graph.function_coverage_details("func_1")
    assert details["executed_operations"] == 1
    assert details["coverage_percent"] == 100
    assert not graph.function_has_successful_business_operation("func_1")


def test_many_recorded_operations_do_not_drop_to_zero_from_function_name_filtering():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="\u5fae\u4fe1\u804a\u5929\u6d88\u606f\u6536\u53d1"))
    graph.add_page(Page(
        id="page_1",
        controls=[f"\u63a7\u4ef6{i}" for i in range(22)],
        operations=[f"\u70b9\u51fb\u63a7\u4ef6{i}" for i in range(12)],
    ), "func_1")

    details = graph.function_coverage_details("func_1")

    assert details["total_controls"] == 22
    assert details["executed_operations"] == 12
    assert details["coverage_percent"] == 70


def test_adjusted_control_coverage_requires_all_for_small_functions():
    assert adjusted_control_coverage(3, 2) == 67
    assert adjusted_control_coverage(3, 3) == 100
    assert adjusted_control_coverage(8, 6) == 84
    assert adjusted_control_coverage(8, 7) == 94
    assert adjusted_control_coverage(20, 14) == 90
    assert adjusted_control_coverage(20, 20) == 100
