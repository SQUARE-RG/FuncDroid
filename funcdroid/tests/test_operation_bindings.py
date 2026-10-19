# Multilingual UI and model-response fixtures use Unicode escapes to preserve their exact text.
"""Explicit operation/control binding regressions. Text-family wording differences once matched none of 13 controls across nine real operations; validated bindings provide authoritative coverage."""

from __future__ import annotations

import pytest

from model import flow_graph as flow_graph_module
from model.flow_graph import FunctionalFlowGraph, FunctionalityNode, Page
from workflow.nodes import ffg_update as ffg_update_module
from workflow.nodes.ffg_update import (
    _apply_operation_bindings,
    _auto_bind_unambiguous_trace_operations,
)


def _wechat_page() -> Page:
    """Reduced WeChat page fixture retaining observed control names and action wording."""
    return Page(
        id="page_1",
        name="\u5fae\u4fe1\u4e3b\u754c\u9762",
        controls=[
            "\u804a\u5929\u5217\u8868\u9879-\u5fae\u4fe1\u56e2\u961f",
            "\u804a\u5929\u5217\u8868\u9879-\u516c\u4f17\u53f7",
            "\u5e95\u90e8\u5bfc\u822a-\u901a\u8baf\u5f55Tab",
            "\u5e95\u90e8\u5bfc\u822a-\u53d1\u73b0Tab",
            "\u5e95\u90e8\u5bfc\u822a-\u6211Tab",
            "\u5de6\u4e0a\u89d2\u8fd4\u56de\u6309\u94ae",
        ],
        operations=[
            "\u70b9\u51fb\u804a\u5929\u5217\u8868\u4e2d\u7684“\u5fae\u4fe1\u56e2\u961f”\u6761\u76ee",
            "\u70b9\u51fb\u5e95\u90e8\u5bfc\u822a\u680f\u4e2d\u7684“\u901a\u8baf\u5f55”Tab",
            "\u5728\u804a\u5929\u5217\u8868\u4e2d\u7684“\u516c\u4f17\u53f7”\u6761\u76ee\u4e0a\u6267\u884c\u5de6\u6ed1\u64cd\u4f5c",
            "\u70b9\u51fb\u5e95\u90e8\u5bfc\u822a\u680f\u4e2d\u7684“\u53d1\u73b0”Tab",
        ],
    )


def _graph_with_wechat_page() -> tuple[FunctionalFlowGraph, Page]:
    graph = FunctionalFlowGraph()
    graph.add_node(FunctionalityNode(id="func_1", name="\u6d88\u606f\u5217\u8868\u6d4f\u89c8"))
    page = _wechat_page()
    graph.add_page(page, "func_1")
    return graph, page


def test_text_matching_alone_leaves_wechat_page_uncovered():
    """Baseline wording mismatch: without bindings these real operations do not match the corresponding controls."""
    graph, _page = _graph_with_wechat_page()
    handled = graph.function_handled_controls("func_1")
    # Instance chat-list and action wording produce different text families.
    # Bottom-navigation and action wording also strip noise asymmetrically.
    assert ("page_1", flow_graph_module._text_key("\u804a\u5929\u5217\u8868\u9879-\u5fae\u4fe1\u56e2\u961f")) not in handled
    assert ("page_1", flow_graph_module._text_key("\u5e95\u90e8\u5bfc\u822a-\u901a\u8baf\u5f55Tab")) not in handled


def test_bindings_cover_controls_that_text_matching_misses():
    graph, page = _graph_with_wechat_page()
    assert page.add_operation_binding(
        "\u70b9\u51fb\u804a\u5929\u5217\u8868\u4e2d\u7684“\u5fae\u4fe1\u56e2\u961f”\u6761\u76ee",
        "\u804a\u5929\u5217\u8868\u9879-\u5fae\u4fe1\u56e2\u961f",
    )
    assert page.add_operation_binding(
        "\u70b9\u51fb\u5e95\u90e8\u5bfc\u822a\u680f\u4e2d\u7684“\u901a\u8baf\u5f55”Tab",
        "\u5e95\u90e8\u5bfc\u822a-\u901a\u8baf\u5f55Tab",
    )
    assert page.add_operation_binding(
        "\u5728\u804a\u5929\u5217\u8868\u4e2d\u7684“\u516c\u4f17\u53f7”\u6761\u76ee\u4e0a\u6267\u884c\u5de6\u6ed1\u64cd\u4f5c",
        "\u804a\u5929\u5217\u8868\u9879-\u516c\u4f17\u53f7",
    )

    handled = graph.function_handled_controls("func_1")
    for control in ("\u804a\u5929\u5217\u8868\u9879-\u5fae\u4fe1\u56e2\u961f", "\u5e95\u90e8\u5bfc\u822a-\u901a\u8baf\u5f55Tab", "\u804a\u5929\u5217\u8868\u9879-\u516c\u4f17\u53f7"):
        assert ("page_1", flow_graph_module._text_key(control)) in handled

    details = graph.function_coverage_details("func_1")
    assert details["executed_operations"] == 3
    assert "\u804a\u5929\u5217\u8868\u9879-\u5fae\u4fe1\u56e2\u961f" not in details["uncovered_controls"]
    assert details["coverage_percent"] > 0


def test_binding_rejects_texts_not_in_page_lists():
    """Reject invented control/operation labels so hallucinated bindings cannot inflate coverage."""
    page = _wechat_page()
    assert page.add_operation_binding("\u70b9\u51fb\u4e0d\u5b58\u5728\u7684\u64cd\u4f5c", "\u804a\u5929\u5217\u8868\u9879-\u5fae\u4fe1\u56e2\u961f") is False
    assert page.add_operation_binding("\u70b9\u51fb\u804a\u5929\u5217\u8868\u4e2d\u7684“\u5fae\u4fe1\u56e2\u961f”\u6761\u76ee", "\u4e0d\u5b58\u5728\u7684\u63a7\u4ef6") is False
    assert page.operation_bindings == []


def test_binding_matches_survive_lenient_normalization():
    page = _wechat_page()
    # Accept equivalent normalized text despite full-width quotes or whitespace differences.
    assert page.add_operation_binding(
        "\u70b9\u51fb\u804a\u5929\u5217\u8868\u4e2d\u7684“\u5fae\u4fe1\u56e2\u961f”\u6761\u76ee ",  # Trailing whitespace.
        "\u804a\u5929\u5217\u8868\u9879-\u5fae\u4fe1\u56e2\u961f",
    )
    assert page.operation_bindings == [{
        "operation": "\u70b9\u51fb\u804a\u5929\u5217\u8868\u4e2d\u7684“\u5fae\u4fe1\u56e2\u961f”\u6761\u76ee",
        "control": "\u804a\u5929\u5217\u8868\u9879-\u5fae\u4fe1\u56e2\u961f",
    }]


def test_binding_deduplicates_repeated_declarations():
    page = _wechat_page()
    assert page.add_operation_binding("\u70b9\u51fb\u804a\u5929\u5217\u8868\u4e2d\u7684“\u5fae\u4fe1\u56e2\u961f”\u6761\u76ee", "\u804a\u5929\u5217\u8868\u9879-\u5fae\u4fe1\u56e2\u961f")
    assert page.add_operation_binding("\u70b9\u51fb\u804a\u5929\u5217\u8868\u4e2d\u7684“\u5fae\u4fe1\u56e2\u961f”\u6761\u76ee", "\u804a\u5929\u5217\u8868\u9879-\u5fae\u4fe1\u56e2\u961f") is False
    assert len(page.operation_bindings) == 1


def test_one_operation_cannot_cover_two_different_controls():
    page = Page(
        id="page_1",
        controls=["\u66f4\u591a\u6269\u5c55\u529f\u80fd\u6309\u94ae", "\u66f4\u591a\u4fe1\u606f\u6309\u94ae"],
        operations=["\u70b9\u51fb\u66f4\u591a\u6309\u94ae"],
    )

    assert page.add_operation_binding("\u70b9\u51fb\u66f4\u591a\u6309\u94ae", "\u66f4\u591a\u6269\u5c55\u529f\u80fd\u6309\u94ae")
    assert page.add_operation_binding("\u70b9\u51fb\u66f4\u591a\u6309\u94ae", "\u66f4\u591a\u4fe1\u606f\u6309\u94ae") is False
    assert page.operation_bindings == [{
        "operation": "\u70b9\u51fb\u66f4\u591a\u6309\u94ae",
        "control": "\u66f4\u591a\u6269\u5c55\u529f\u80fd\u6309\u94ae",
    }]


def test_recovery_and_failure_bindings_do_not_count(monkeypatch):
    graph, page = _graph_with_wechat_page()
    page.add_operation("\u81ea\u52a8\u6062\u590d：\u6253\u5f00\u5fae\u4fe1\u4e3b\u9875\u9762")
    page.add_operation("\u5c1d\u8bd5\u70b9\u51fb\u5e95\u90e8\u5bfc\u822a-\u53d1\u73b0Tab\u5931\u8d25：\u8d85\u65f6")
    assert page.add_operation_binding("\u81ea\u52a8\u6062\u590d：\u6253\u5f00\u5fae\u4fe1\u4e3b\u9875\u9762", "\u804a\u5929\u5217\u8868\u9879-\u5fae\u4fe1\u56e2\u961f")
    assert page.add_operation_binding("\u5c1d\u8bd5\u70b9\u51fb\u5e95\u90e8\u5bfc\u822a-\u53d1\u73b0Tab\u5931\u8d25：\u8d85\u65f6", "\u5e95\u90e8\u5bfc\u822a-\u53d1\u73b0Tab")
    # Disable text fallback to isolate exclusion of recovery/failed-operation bindings from coverage.
    monkeypatch.setattr(flow_graph_module, "COVERAGE_LEGACY_TEXT_MATCHING", False)
    assert graph.function_handled_controls("func_1") == set()


def test_legacy_text_matching_can_be_disabled(monkeypatch):
    """Disable legacy text fallback so coverage depends only on explicit bindings."""
    graph, page = _graph_with_wechat_page()
    page.add_operation_binding("\u70b9\u51fb\u5e95\u90e8\u5bfc\u822a\u680f\u4e2d\u7684“\u53d1\u73b0”Tab", "\u5e95\u90e8\u5bfc\u822a-\u53d1\u73b0Tab")
    monkeypatch.setattr(flow_graph_module, "COVERAGE_LEGACY_TEXT_MATCHING", False)

    handled = graph.function_handled_controls("func_1")
    assert handled == {("page_1", flow_graph_module._text_key("\u5e95\u90e8\u5bfc\u822a-\u53d1\u73b0Tab"))}


def test_legacy_text_matching_is_off_by_default(monkeypatch):
    """Default-off text fallback prevents generic family matches from marking unexecuted list instances covered."""
    monkeypatch.delenv("FFG_COVERAGE_LEGACY_MATCHING", raising=False)
    import importlib

    importlib.reload(flow_graph_module)
    assert flow_graph_module.COVERAGE_LEGACY_TEXT_MATCHING is False

    graph = FunctionalFlowGraph()
    graph.add_node(FunctionalityNode(id="func_1", name="\u6d88\u606f\u5217\u8868\u6d4f\u89c8"))
    page = Page(
        id="page_1",
        controls=["\u5217\u8868\u9879-\u5fae\u4fe1\u56e2\u961f", "\u5217\u8868\u9879-\u670d\u52a1\u901a\u77e5", "\u5217\u8868\u9879-\u6d4b\u8bd5"],
        operations=["\u70b9\u51fb\u670d\u52a1\u901a\u77e5\u5217\u8868\u9879"],
    )
    graph.add_page(page, "func_1")
    assert graph.function_handled_controls("func_1") == set()


def test_legacy_text_matching_can_be_reenabled(monkeypatch):
    """The explicit compatibility switch can reenable legacy text matching."""
    monkeypatch.setattr(flow_graph_module, "COVERAGE_LEGACY_TEXT_MATCHING", True)
    graph = FunctionalFlowGraph()
    graph.add_node(FunctionalityNode(id="func_1", name="\u6d88\u606f\u5217\u8868\u6d4f\u89c8"))
    page = Page(
        id="page_1",
        controls=["\u53d1\u9001\u6309\u94ae"],
        operations=["\u70b9\u51fb\u53d1\u9001\u6309\u94ae"],
    )
    graph.add_page(page, "func_1")
    assert graph.function_handled_controls("func_1") == {
        ("page_1", flow_graph_module._text_key("\u53d1\u9001\u6309\u94ae")),
    }


def test_remove_operation_drops_its_bindings():
    page = _wechat_page()
    page.add_operation_binding("\u70b9\u51fb\u5e95\u90e8\u5bfc\u822a\u680f\u4e2d\u7684“\u901a\u8baf\u5f55”Tab", "\u5e95\u90e8\u5bfc\u822a-\u901a\u8baf\u5f55Tab")
    assert page.remove_operation("\u70b9\u51fb\u5e95\u90e8\u5bfc\u822a\u680f\u4e2d\u7684“\u901a\u8baf\u5f55”Tab") is True
    assert page.operation_bindings == []


def test_operation_merge_keeps_bindings_consistent():
    """Migrate bindings when a family merge replaces the representative operation text."""
    page = Page(
        id="page_1",
        controls=["\u53d1\u9001\u6309\u94ae"],
        operations=["\u70b9\u51fb\u53d1\u9001\u6309\u94ae", "\u70b9\u51fb\u4f4d\u4e8e\u804a\u5929\u5217\u8868\u5e95\u90e8\u7684\u7eff\u8272\u7684\u5706\u5f62\u53d1\u9001\u6309\u94ae"],
        operation_bindings=[{"operation": "\u70b9\u51fb\u53d1\u9001\u6309\u94ae", "control": "\u53d1\u9001\u6309\u94ae"}],
    )
    # The richer wording exceeds the 12-character difference and wins the family merge.
    assert page.operations == ["\u70b9\u51fb\u4f4d\u4e8e\u804a\u5929\u5217\u8868\u5e95\u90e8\u7684\u7eff\u8272\u7684\u5706\u5f62\u53d1\u9001\u6309\u94ae"]
    assert page.operation_bindings == [{
        "operation": "\u70b9\u51fb\u4f4d\u4e8e\u804a\u5929\u5217\u8868\u5e95\u90e8\u7684\u7eff\u8272\u7684\u5706\u5f62\u53d1\u9001\u6309\u94ae",
        "control": "\u53d1\u9001\u6309\u94ae",
    }]


def test_bindings_roundtrip_through_serialization():
    graph, page = _graph_with_wechat_page()
    page.add_operation_binding("\u70b9\u51fb\u5e95\u90e8\u5bfc\u822a\u680f\u4e2d\u7684“\u901a\u8baf\u5f55”Tab", "\u5e95\u90e8\u5bfc\u822a-\u901a\u8baf\u5f55Tab")
    data = graph.to_dict()
    restored = FunctionalFlowGraph.from_dict(data)
    restored_page = restored.get_page("page_1")
    assert restored_page is not None
    assert restored_page.operation_bindings == [{
        "operation": "\u70b9\u51fb\u5e95\u90e8\u5bfc\u822a\u680f\u4e2d\u7684“\u901a\u8baf\u5f55”Tab",
        "control": "\u5e95\u90e8\u5bfc\u822a-\u901a\u8baf\u5f55Tab",
    }]
    assert restored.function_handled_controls("func_1") == {
        ("page_1", flow_graph_module._text_key("\u5e95\u90e8\u5bfc\u822a-\u901a\u8baf\u5f55Tab")),
    }


def test_invalid_bindings_dropped_on_deserialization():
    page = Page(
        id="page_1",
        controls=["\u4fdd\u5b58\u6309\u94ae"],
        operations=["\u70b9\u51fb\u4fdd\u5b58\u6309\u94ae"],
        operation_bindings=[
            {"operation": "\u70b9\u51fb\u4fdd\u5b58\u6309\u94ae", "control": "\u4fdd\u5b58\u6309\u94ae"},
            {"operation": "\u5e7b\u89c9\u64cd\u4f5c", "control": "\u4fdd\u5b58\u6309\u94ae"},
            {"operation": "\u70b9\u51fb\u4fdd\u5b58\u6309\u94ae", "control": "\u5e7b\u89c9\u63a7\u4ef6"},
        ],
    )
    assert page.operation_bindings == [
        {"operation": "\u70b9\u51fb\u4fdd\u5b58\u6309\u94ae", "control": "\u4fdd\u5b58\u6309\u94ae"},
    ]


def _trace_step(
    raw_step: int,
    action_name: str,
    action_description: str,
    **extra,
) -> dict:
    step = {
        "raw_step_index": raw_step,
        "action_name": action_name,
        "action_description": action_description,
    }
    step.update(extra)
    return step


def test_apply_operation_bindings_resolves_step_and_control():
    """Resolve source_step bindings to operation text derived consistently from the trace."""
    graph = FunctionalFlowGraph()
    graph.add_node(FunctionalityNode(id="func_1", name="\u6d88\u606f\u5217\u8868\u6d4f\u89c8"))
    page = _wechat_page()
    graph.add_page(page, "func_1")
    trace = [
        _trace_step(1, "click", "\u70b9\u51fb\u5e95\u90e8\u5bfc\u822a\u680f\u4e2d\u7684“\u901a\u8baf\u5f55”Tab"),
    ]
    trace_page_map = {1: "page_1"}

    applied = _apply_operation_bindings(
        graph,
        {"operation_bindings": [{"source_step": 1, "control": "\u5e95\u90e8\u5bfc\u822a-\u901a\u8baf\u5f55Tab"}]},
        trace,
        trace_page_map,
    )

    assert applied == 1
    assert page.operation_bindings == [{
        "operation": "\u70b9\u51fb\u5e95\u90e8\u5bfc\u822a\u680f\u4e2d\u7684“\u901a\u8baf\u5f55”Tab",
        "control": "\u5e95\u90e8\u5bfc\u822a-\u901a\u8baf\u5f55Tab",
    }]
    assert ("page_1", flow_graph_module._text_key("\u5e95\u90e8\u5bfc\u822a-\u901a\u8baf\u5f55Tab")) in (
        graph.function_handled_controls("func_1")
    )


def test_apply_operation_bindings_drops_invalid_entries():
    graph = FunctionalFlowGraph()
    graph.add_node(FunctionalityNode(id="func_1", name="\u6d88\u606f\u5217\u8868\u6d4f\u89c8"))
    page = _wechat_page()
    graph.add_page(page, "func_1")
    trace = [
        _trace_step(1, "click", "\u70b9\u51fb\u5e95\u90e8\u5bfc\u822a\u680f\u4e2d\u7684“\u901a\u8baf\u5f55”Tab"),
        _trace_step(2, "click", "\u70b9\u51fb\u804a\u5929\u5217\u8868\u4e2d\u7684“\u5fae\u4fe1\u56e2\u961f”\u6761\u76ee"),
    ]
    trace_page_map = {1: "page_1", 2: "page_1"}

    applied = _apply_operation_bindings(
        graph,
        {"operation_bindings": [
            {"source_step": 99, "control": "\u5e95\u90e8\u5bfc\u822a-\u901a\u8baf\u5f55Tab"},  # Missing step.
            {"source_step": 2},                                    # Missing control.
            {"operation": "\u5e7b\u89c9\u64cd\u4f5c\u6587\u672c", "control": "\u5e95\u90e8\u5bfc\u822a-\u901a\u8baf\u5f55Tab"},  # No source_step and the operation is not listed.
        ]},
        trace,
        trace_page_map,
    )

    assert applied == 0
    assert page.operation_bindings == []


def test_apply_operation_bindings_rejects_missing_control():
    """A valid operation must not implicitly register an invented control or one assigned to the wrong page."""
    graph = FunctionalFlowGraph()
    graph.add_node(FunctionalityNode(id="func_1", name="\u6d88\u606f\u5217\u8868\u6d4f\u89c8"))
    page = _wechat_page()
    graph.add_page(page, "func_1")
    # Match the real sequence: record trace operations before declaring bindings.
    page.add_operation("\u70b9\u51fb\u5206\u4eab\u9762\u677f\u4e2d\u7684“\u53d6\u6d88”\u6309\u94ae")
    trace = [_trace_step(1, "click", "\u70b9\u51fb\u5206\u4eab\u9762\u677f\u4e2d\u7684“\u53d6\u6d88”\u6309\u94ae")]
    trace_page_map = {1: "page_1"}

    applied = _apply_operation_bindings(
        graph,
        {"operation_bindings": [{"source_step": 1, "control": "\u53d6\u6d88\u6309\u94ae"}]},
        trace,
        trace_page_map,
    )

    assert applied == 0
    assert "\u53d6\u6d88\u6309\u94ae" not in page.controls
    assert page.operation_bindings == []


def test_unambiguous_existing_control_is_bound_without_model_declaration():
    graph, page = _graph_with_wechat_page()
    trace = [_trace_step(1, "click", "\u70b9\u51fb\u5e95\u90e8\u5bfc\u822a\u680f\u4e2d\u7684“\u901a\u8baf\u5f55”Tab")]

    applied = _auto_bind_unambiguous_trace_operations(
        graph,
        trace,
        {1: "page_1"},
    )

    assert applied == 1
    assert page.operation_bindings == [{
        "operation": "\u70b9\u51fb\u5e95\u90e8\u5bfc\u822a\u680f\u4e2d\u7684“\u901a\u8baf\u5f55”Tab",
        "control": "\u5e95\u90e8\u5bfc\u822a-\u901a\u8baf\u5f55Tab",
    }]


def test_scenario_credit_goes_to_planned_target_function():
    """Credit the planned target rather than the endpoint functionality; shared pages previously stole scenario counts despite 92% target coverage."""
    from workflow.nodes.ffg_update import _record_completed_intra_scenario

    graph = FunctionalFlowGraph()
    graph.add_node(FunctionalityNode(id="func_2", name="\u6267\u884c\u4f1a\u8bdd\u6d4f\u89c8"))
    graph.add_node(FunctionalityNode(id="func_10", name="\u6d4b\u8bd5\u4f1a\u8bdd\u6d4f\u89c8"))
    graph.get_node("func_10").test_status = "in_progress"

    owner = _record_completed_intra_scenario(
        graph,
        {"target_function_id": "func_10"},
        "\u64cd\u4f5c\u6d4b\u8bd5\u4f1a\u8bdd\u5217\u8868\u9879\u5e76\u67e5\u770b\u8be6\u60c5",
    )

    assert owner == "func_10"
    assert "\u64cd\u4f5c\u6d4b\u8bd5\u4f1a\u8bdd\u5217\u8868\u9879\u5e76\u67e5\u770b\u8be6\u60c5" in graph.get_node("func_10").scenarios
    assert graph.get_node("func_2").scenarios == []


def test_page_audit_removal_preserves_bound_operations():
    """Preserve bound operations during page audit so valid business actions and their coverage are not lost."""
    from workflow.nodes.ffg_update import _apply_page_audit_operation_removals

    page = _wechat_page()
    page.add_operation("\u70b9\u51fb\u8f93\u5165\u6cd5\u5f39\u7a97\u4e2d\u7684“\u5fae\u4fe1\u8f93\u5165\u6cd5”\u9009\u9879")
    page.add_operation_binding(
        "\u70b9\u51fb\u5e95\u90e8\u5bfc\u822a\u680f\u4e2d\u7684“\u901a\u8baf\u5f55”Tab", "\u5e95\u90e8\u5bfc\u822a-\u901a\u8baf\u5f55Tab"
    )

    removed = _apply_page_audit_operation_removals(
        page,
        [
            "\u70b9\u51fb\u5e95\u90e8\u5bfc\u822a\u680f\u4e2d\u7684“\u901a\u8baf\u5f55”Tab",   # Bound operation: reject removal.
            "\u70b9\u51fb\u8f93\u5165\u6cd5\u5f39\u7a97\u4e2d\u7684“\u5fae\u4fe1\u8f93\u5165\u6cd5”\u9009\u9879",  # Unbound operation: allow removal.
            "\u6e05\u5355\u91cc\u4e0d\u5b58\u5728\u7684\u64cd\u4f5c",                # Mismatched original text: discard.
        ],
    )

    assert removed == ["\u70b9\u51fb\u8f93\u5165\u6cd5\u5f39\u7a97\u4e2d\u7684“\u5fae\u4fe1\u8f93\u5165\u6cd5”\u9009\u9879"]
    assert "\u70b9\u51fb\u5e95\u90e8\u5bfc\u822a\u680f\u4e2d\u7684“\u901a\u8baf\u5f55”Tab" in page.operations
    assert page.operation_bindings  # Bindings remain intact.


def test_apply_operation_bindings_redeclaration_is_not_validation_failure():
    """Repeated declarations of existing bindings are valid during refinement and must not generate validation-failure warnings."""
    graph = FunctionalFlowGraph()
    graph.add_node(FunctionalityNode(id="func_1", name="\u6d88\u606f\u5217\u8868\u6d4f\u89c8"))
    page = _wechat_page()
    graph.add_page(page, "func_1")
    trace = [_trace_step(1, "click", "\u70b9\u51fb\u5e95\u90e8\u5bfc\u822a\u680f\u4e2d\u7684“\u901a\u8baf\u5f55”Tab")]
    trace_page_map = {1: "page_1"}
    patch_payload = {"operation_bindings": [{"source_step": 1, "control": "\u5e95\u90e8\u5bfc\u822a-\u901a\u8baf\u5f55Tab"}]}

    assert _apply_operation_bindings(graph, patch_payload, trace, trace_page_map) == 1
    # Redeclaration adds nothing but is not a validation failure or dropped-binding warning.
    assert _apply_operation_bindings(graph, patch_payload, trace, trace_page_map) == 0
    assert len(page.operation_bindings) == 1


def test_apply_operation_bindings_ignores_press_back_steps():
    graph = FunctionalFlowGraph()
    graph.add_node(FunctionalityNode(id="func_1", name="\u6d88\u606f\u5217\u8868\u6d4f\u89c8"))
    page = _wechat_page()
    graph.add_page(page, "func_1")
    trace = [_trace_step(1, "press_back", "\u8fd4\u56de\u4e0a\u4e00\u9875")]
    trace_page_map = {1: "page_1"}

    applied = _apply_operation_bindings(
        graph,
        {"operation_bindings": [{"source_step": 1, "control": "\u5e95\u90e8\u5bfc\u822a-\u901a\u8baf\u5f55Tab"}]},
        trace,
        trace_page_map,
    )

    assert applied == 0
    assert page.operation_bindings == []


def test_ffg_update_prompt_declares_operation_bindings():
    assert "operation_bindings" in ffg_update_module.FFG_UPDATE_SYSTEM_PROMPT
    assert "source_step" in ffg_update_module.FFG_UPDATE_SYSTEM_PROMPT


@pytest.mark.parametrize(
    "raw,expected_x,expected_y",
    [
        ('{"x": 226, 190}', 226, 190),
        ('{"x": 54, 69 }', 54, 69),
        ('{"x": 54,\n 69}', 54, 69),
        ('{"x": [682, 954]}', 682, 954),
    ],
)
def test_llm_json_repair_fixes_missing_coordinate_key(raw, expected_x, expected_y):
    import json

    from workflow.llm import _repair_common_json_typos

    repaired = _repair_common_json_typos(raw)
    payload = json.loads(repaired)
    assert payload["x"] == expected_x
    assert payload["y"] == expected_y


def test_llm_json_repair_leaves_valid_text_untouched():
    from workflow.llm import _repair_common_json_typos

    original = '{"x": 100, "y": 200, "action_name": "click"}'
    assert _repair_common_json_typos(original) == original
    # Do not repair negative coordinates; actual image dimensions validate upper bounds.
    assert _repair_common_json_typos('{"x": -2, 3000}') == '{"x": -2, 3000}'


def test_action_json_parser_repairs_missing_y_key():
    """The scenario parser uses shared typo repair for malformed coordinates with a missing y key."""
    from workflow.nodes.scenario_exec import _parse_action_json

    raw = (
        "```json\n"
        "{\n"
        '  "thought": "\u70b9\u51fb\u5e95\u90e8\u52a0\u53f7\u6309\u94ae",\n'
        '  "action_name": "click",\n'
        '  "is_interruption": false,\n'
        '  "x": 949, 537\n'
        "}\n"
        "```"
    )
    plan = _parse_action_json(raw)
    assert plan["x"] == 949
    assert plan["y"] == 537
    assert plan["action_name"] == "click"


def test_action_plan_accepts_action_type_alias():
    """Accept action_type as an alias for action_name in model responses."""
    from workflow.nodes.scenario_exec import _normalize_action_plan

    plan = _normalize_action_plan({
        "action_type": "press_back",
        "action_description": "\u70b9\u51fb\u5de6\u4e0a\u89d2\u8fd4\u56de\u7bad\u5934",
        "is_interruption": False,
    })
    assert plan["action_name"] == "press_back"


def test_back_button_and_return_home_button_are_bound_separately():
    """Keep back and return-home controls distinct; merging them previously lost bindings and stalled coverage at 67%."""
    graph = FunctionalFlowGraph()
    graph.add_node(FunctionalityNode(id="func_3", name="\u7535\u6c60\u6062\u590d"))
    page = Page(
        id="page_4",
        name="\u7535\u6c60\u4fee\u590d\u9875",
        controls=["\u8fd4\u56de\u6309\u94ae", "\u5f00\u59cb\u4fee\u590d\u6309\u94ae", "\u8fd4\u56de\u9996\u9875\u6309\u94ae"],
    )
    graph.add_page(page, "func_3")
    for operation in ("\u70b9\u51fb\u5f00\u59cb\u4fee\u590d\u6309\u94ae", "\u70b9\u51fb\u8fd4\u56de\u9996\u9875\u6309\u94ae", "\u70b9\u51fb\u5de6\u4e0a\u89d2\u8fd4\u56de\u6309\u94ae"):
        page.add_operation(operation)

    assert len(page.operations) == 3
    applied = _apply_operation_bindings(
        graph,
        {"operation_bindings": [
            {"page_ref": "page_4", "operation": "\u70b9\u51fb\u5f00\u59cb\u4fee\u590d\u6309\u94ae", "control": "\u5f00\u59cb\u4fee\u590d\u6309\u94ae"},
            {"page_ref": "page_4", "operation": "\u70b9\u51fb\u8fd4\u56de\u9996\u9875\u6309\u94ae", "control": "\u8fd4\u56de\u9996\u9875\u6309\u94ae"},
            {"page_ref": "page_4", "operation": "\u70b9\u51fb\u5de6\u4e0a\u89d2\u8fd4\u56de\u6309\u94ae", "control": "\u8fd4\u56de\u6309\u94ae"},
        ]},
        [],
        {},
    )

    assert applied == 3
    handled = graph.function_handled_controls("func_3")
    for control in page.controls:
        assert ("page_4", flow_graph_module._text_key(control)) in handled


def test_generic_back_wordings_stay_one_family():
    key = flow_graph_module.operation_family_key
    assert key("\u70b9\u51fb\u5de6\u4e0a\u89d2\u8fd4\u56de\u6309\u94ae") == key("\u70b9\u51fb\u8fd4\u56de\u6309\u94ae") == key("press_back")
    assert key("\u70b9\u51fb\u8fd4\u56de\u6309\u94ae\u8fd4\u56de\u9996\u9875") == key("\u70b9\u51fb\u8fd4\u56de\u6309\u94ae")
    assert key("\u70b9\u51fb\u8fd4\u56de\u9996\u9875\u6309\u94ae") != key("\u70b9\u51fb\u8fd4\u56de\u6309\u94ae")
    family = flow_graph_module.control_family_key
    assert family("\u8fd4\u56de\u9876\u90e8\u6309\u94ae") != family("\u9876\u90e8\u8fd4\u56de\u6309\u94ae") == family("\u5de6\u4e0a\u89d2\u8fd4\u56de\u7bad\u5934")
