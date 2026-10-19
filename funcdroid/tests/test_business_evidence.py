# Multilingual UI and model-response fixtures use Unicode escapes to preserve their exact text.
"""Evidence semantics and discovery-driven testing regressions."""
import pytest
import numpy as np
from tools.utils import encode_image_to_jpg_base64

from compliance.checkpoints import ComplianceChecklist
from compliance.report import build_compliance_report
from model.flow_graph import FunctionalFlowGraph, FunctionalityNode, Page
from workflow.nodes.ffg_update import _record_completed_intra_scenario
from workflow.nodes.scenario_plan import _cross_pair_ready
from workflow.nodes.scenario_exec import _normalize_action_plan
from workflow.trace_defect_detection import _dedupe_bugs, _bug_has_required_trace_evidence


def test_depth_attempt_is_counted_once_and_recorded_like_breadth():
    graph = FunctionalFlowGraph()
    graph.add_node(FunctionalityNode(id="func_1", name="\u6536\u85cf"))
    plan = dict(target_function_id="func_1", function_phase="depth")
    _record_completed_intra_scenario(graph, plan, "\u6536\u85cf\u540e\u79bb\u5f00\u518d\u8fd4\u56de\u67e5\u770b")
    _record_completed_intra_scenario(graph, plan, "\u6536\u85cf\u540e\u79bb\u5f00\u518d\u8fd4\u56de\u67e5\u770b")
    node = graph.get_node("func_1")
    assert node.scenarios == ["\u6536\u85cf\u540e\u79bb\u5f00\u518d\u8fd4\u56de\u67e5\u770b"]
    assert node.depth_scenario_count == 0


def test_ready_pair_does_not_depend_on_unrelated_unexplored_function():
    graph = FunctionalFlowGraph()
    for n in range(1, 4):
        node = FunctionalityNode(id=f"func_{n}", name=f"\u529f\u80fd{n}")
        graph.add_node(node)
        graph.add_page(Page(id=f"page_{n}", name="\u9875\u9762", controls=["\u6309\u94ae"]), node.id)
        if n < 3:
            node.add_scenario("\u5df2\u6267\u884c\u4e1a\u52a1\u52a8\u4f5c")
            page = graph.get_page(f"page_{n}")
            page.add_operation("\u70b9\u51fb\u6309\u94ae")
            page.add_operation_binding("\u70b9\u51fb\u6309\u94ae", "\u6309\u94ae")
    assert _cross_pair_ready(graph, "func_1", "func_2")
    assert not _cross_pair_ready(graph, "func_1", "func_3")


@pytest.mark.parametrize("verdict", ["compliant", "not_applicable"])
def test_new_scope_reopens_local_verdict_and_preserves_history(verdict):
    checklist = ComplianceChecklist()
    checklist.mark_executing(["P03"], "before")
    checklist.apply_results([dict(check_id="P03", verdict=verdict, description="\u65e7\u8def\u5f84", evidence=[])])
    assert checklist.reopen_deferred({"P03": "after-check"}) == []
    restored = ComplianceChecklist.from_state(checklist.to_state())
    assert restored.reopen_deferred({"P03": "after-check"}) == []
    assert restored.reopen_deferred({"P03": "new-related-entry"}) == ["P03"]
    restored.mark_executing(["P03"], "new-related-entry")
    restored.apply_results([dict(check_id="P03", verdict="violation", description="\u65b0\u8def\u5f84", evidence=[])])
    assert restored.findings[0]["history"][0]["description"] == "\u65e7\u8def\u5f84"
    assert restored.reopen_deferred({"P03": "another-entry"}) == []
    restored.apply_results([dict(check_id="P03", verdict="compliant", description="\u5c40\u90e8\u7b26\u5408")])
    assert restored.findings[0]["verdict"] == "violation"
    report = build_compliance_report(restored)
    row = next(row for row in report["checks"] if row["rule_id"] == "P03")
    assert row["history"] and row["scope"] == "observed_paths_only"


def test_network_words_do_not_erase_observed_app_inconsistency():
    bug = dict(type="logic_error", confidence="high", description="\u7f51\u7edc\u5f02\u5e38\u91cd\u8bd5\u5bfc\u81f4\u91cd\u590d\u521b\u5efa\u672c\u5730\u6761\u76ee",
               actual="\u4e24\u4e2a\u76f8\u540c\u6761\u76ee", expected="\u53ea\u521b\u5efa\u4e00\u4e2a", evidence_category="single_action_resource_mutation")
    assert len(_dedupe_bugs([bug])) == 1


@pytest.mark.parametrize("action", ["swipe", "click", "type", "press_back"])
def test_auto_persistence_does_not_require_save_word(action):
    bug = dict(evidence_category="persisted_result_error", evidence=[
        dict(global_step=1, role="action"), dict(global_step=2, role="after")])
    steps = [dict(action_name=action, trace_phase="test", action_description="\u81ea\u52a8\u66f4\u65b0"),
             dict(action_name="finished", trace_phase="test")]
    for step in steps:
        step["current_page"] = Page(representative_screenshot=encode_image_to_jpg_base64(np.full((32, 32, 3), 180, dtype=np.uint8)))
    assert _bug_has_required_trace_evidence(bug, steps)


def test_exec_prompt_has_no_verification_protocol():
    from workflow.nodes.prompt import SCENARIO_EXEC_SYSTEM_PROMPT_TEMPLATE
    assert "verification" not in _normalize_action_plan({"action_name": "finished", "verification": {"x": 1}})
    text = SCENARIO_EXEC_SYSTEM_PROMPT_TEMPLATE.format(mode_addendum="", app_name="Demo", package_name="com.demo",
        task_phase="depth", scenario="\u6536\u85cf", max_exec_steps=18)
    assert "readback_step" not in text and "reference path" in text


def test_planner_selects_ready_cross_before_intra(monkeypatch):
    from workflow.nodes import scenario_plan as planner
    graph = FunctionalFlowGraph()
    for n in range(1, 4):
        graph.add_node(FunctionalityNode(id=f"func_{n}", name=f"\u529f\u80fd{n}"))
        graph.add_page(Page(id=f"page_{n}", name=f"\u9875\u9762{n}", controls=["\u6309\u94ae"]), f"func_{n}")
    monkeypatch.setattr(planner, "_refresh_current_page_observation", lambda g, d, p: p)
    monkeypatch.setattr(planner, "_pending_cross_pairs", lambda g: [("func_1", "func_2", "\u5171\u4eab\u8d44\u6e90")])
    monkeypatch.setattr(planner, "_cross_pair_ready", lambda *args: True)
    monkeypatch.setattr(planner, "_navigation_to_function", lambda *args: ("", "", ""))
    def forbidden(*args):
        raise AssertionError("Ready cross pair was starved by intra selection")
    monkeypatch.setattr(planner, "_select_intra_target", forbidden)
    class Selected(Exception):
        pass
    def scope(g, page, kind, **kw):
        assert kind == "cross_test"
        assert kw["src"].id == "func_1" and kw["dst"].id == "func_2"
        raise Selected()
    monkeypatch.setattr(planner, "_build_scope_planning_summary", scope)
    with pytest.raises(Selected):
        planner.create_scenario_plan_node(object(), object())({"ffg": graph, "current_page": graph.get_page("page_1"), "run_phase": "all"})
