# Multilingual UI and model-response fixtures use Unicode escapes to preserve their exact text.
"""Compliance pipeline tests for rules, check states, scenario generation, trace detection, and reporting."""

from __future__ import annotations

import json

import pytest

from model.flow_graph import FlowGraph, FunctionNode, Page, PageTransition
from compliance.checkpoints import (
    CHECK_STATUS_DEFERRED,
    CHECK_STATUS_DONE,
    CHECK_STATUS_EXECUTING,
    ComplianceChecklist,
    build_compliance_scenarios,
)
from compliance.report import build_compliance_report, save_compliance_report
from compliance.rule_library import (
    CHECKPOINT_GROUP_ORDER,
    gui_rules,
    load_rules,
    pending_gui_checkpoints,
    rule_by_id,
)
from workflow.compliance_worker import (
    build_compliance_detection_job,
    compliance_worker,
    plan_check_ids,
)
from workflow import compliance_detection
from workflow.nodes import scenario_plan
from workflow.nodes import prompt as prompt_module


def test_compliance_result_parser_accepts_equivalent_model_shapes():
    list_shape = compliance_detection._extract_raw_check_results({
        "results": [{"rule_id": "R014", "verdict": "compliant"}],
    })
    assert list_shape[0]["rule_id"] == "R014"

    mapping_shape = compliance_detection._extract_raw_check_results({
        "check_results": {
            "R014": {"verdict": "compliant", "description": "\u6709\u5f39\u7a97"},
        },
    })
    assert mapping_shape == [{
        "check_id": "R014",
        "verdict": "compliant",
        "description": "\u6709\u5f39\u7a97",
    }]

    top_level_shape = compliance_detection._extract_raw_check_results({
        "R030": {"verdict": "violation"},
    })
    assert top_level_shape[0]["check_id"] == "R030"


def test_compliance_review_prompt_uses_balanced_gui_screening_scope():
    prompt = prompt_module.COMPLIANCE_DETECT_SYSTEM_PROMPT

    assert "dynamic GUI screening" in prompt
    assert "compliant applies only to observed applicable GUI paths" in prompt
    assert "may be compliant even when incidental backend collection, server deletion or response deadlines cannot be verified" in prompt
    assert "indicate missing coverage, even after targeted search" in prompt
    assert "explicit_scope_exclusion" in prompt
    assert "A reachable policy entry and readable text establish accessibility only" in prompt
    assert "An initial notice does not establish that no earlier background collection occurred" in prompt


# ----------------------------------------------------------------------
# Rule library.
# ----------------------------------------------------------------------
def test_rules_library_contains_only_gui_dynamic_rules():
    rules = load_rules()
    assert len(rules) == 20
    counts = {"gui_direct": 0, "gui_collect_llm": 0}
    for rule in rules:
        assert rule["rule_id"].startswith("P")
        assert rule["clause"].strip()
        counts[rule["detectability"]] += 1
    assert counts == {"gui_direct": 18, "gui_collect_llm": 2}
    gui = gui_rules()
    assert len(gui) == 20
    for rule in gui:
        assert rule["checkpoint"]["group"] in CHECKPOINT_GROUP_ORDER


def test_first_launch_rules_are_cold_start_and_prioritized():
    rules = load_rules()
    first_launch = [
        rule for rule in rules
        if rule.get("checkpoint", {}).get("group") == "first_launch_flow"
    ]
    assert first_launch
    assert all(rule["checkpoint"]["cold_start"] for rule in first_launch)
    pending = pending_gui_checkpoints({})
    assert pending[0]["checkpoint"]["group"] == "first_launch_flow"


def test_rule_by_id_lookup():
    rule = rule_by_id("R014")
    assert rule is not None
    assert "Article 14" in rule["clause"]
    assert rule["checkpoint"]["group"] == "first_launch_flow"
    assert rule_by_id("R999") is None


# ----------------------------------------------------------------------
# Check state machine.
# ----------------------------------------------------------------------
def test_checklist_marks_executing_and_filters_pending():
    checklist = ComplianceChecklist()
    first_group = checklist.next_group_rules()
    rule_ids = [rule["rule_id"] for rule in first_group]
    assert rule_ids

    checklist.mark_executing(rule_ids)
    statuses = {checklist.status[rule_id] for rule_id in rule_ids}
    assert statuses == {CHECK_STATUS_EXECUTING}

    # Executing checks are excluded from pending checks to prevent duplicate scheduling.
    assert checklist.next_group_rules()[0]["checkpoint"]["group"] != "first_launch_flow"


def test_checklist_deprioritizes_groups_with_more_attempts():
    rules = gui_rules()
    first_rule = next(
        rule for rule in rules
        if rule["checkpoint"]["group"] == "first_launch_flow"
    )
    later_rule = next(
        rule for rule in rules
        if rule["checkpoint"]["group"] == "permission_denial_flow"
    )
    selected_ids = {first_rule["rule_id"], later_rule["rule_id"]}
    checklist = ComplianceChecklist(
        status={
            rule["rule_id"]: (
                "pending" if rule["rule_id"] in selected_ids else CHECK_STATUS_DONE
            )
            for rule in rules
        },
        attempts={
            first_rule["rule_id"]: 3,
            later_rule["rule_id"]: 1,
        },
    )

    selected = checklist.next_group_rules()

    assert [rule["rule_id"] for rule in selected] == [later_rule["rule_id"]]


def test_compliance_execution_prompt_rotates_after_repeated_no_progress():
    prompt = prompt_module.COMPLIANCE_EXEC_ADDENDUM

    assert "After three consecutive attempts at one target without a new page" in prompt
    assert "try other uncovered targets" in prompt
    assert "Do not let one rule consume the entire round step budget" in prompt


def test_checklist_apply_results_is_idempotent_per_rule():
    checklist = ComplianceChecklist()
    checklist.apply_results([
        {"check_id": "R014", "verdict": "compliant", "description": "\u9996\u542f\u6709\u5f39\u7a97", "evidence": [{"global_step": 1, "role": "before"}]},
    ])
    checklist.apply_results([
        {"check_id": "R014", "verdict": "violation", "description": "\u590d\u6838\u540e\u6539\u5224", "evidence": []},
        {"check_id": "R013", "verdict": "bogus", "description": "\u975e\u6cd5 verdict \u5e94\u88ab\u4e22\u5f03"},
    ])
    findings = [f for f in checklist.findings if f["check_id"] == "R014"]
    assert len(findings) == 1
    assert findings[0]["verdict"] == "violation"
    assert all(f["check_id"] != "R013" for f in checklist.findings)
    assert checklist.status["R014"] == CHECK_STATUS_DONE
    assert "R013" not in checklist.status


def test_checklist_summary_counts():
    checklist = ComplianceChecklist()
    checklist.apply_results([
        {"check_id": "P03", "verdict": "compliant", "description": "", "evidence": []},
        {"check_id": "P04", "verdict": "violation", "description": "", "evidence": []},
        {"check_id": "P08", "verdict": "undetermined", "description": "", "evidence": []},
        {"check_id": "P16", "verdict": "not_applicable", "description": "\u660e\u786e\u4e0d\u63d0\u4f9b\u8ba2\u9605\u670d\u52a1", "evidence": []},
    ])
    summary = checklist.summary()
    assert summary["processed"] == 4
    assert summary["executed"] == 4  # Compatibility alias.
    assert summary["resolved"] == 3
    assert summary["compliant"] == 1
    assert summary["violation"] == 1
    assert summary["undetermined"] == 1
    assert summary["not_applicable"] == 1
    assert summary["deferred"] == 1
    assert summary["unresolved"] == 1
    assert summary["untested"] == summary["gui_total"] - 4


def test_undetermined_remains_unresolved_after_retry_budget_and_waits_for_discovery():
    checklist = ComplianceChecklist()
    checklist.mark_executing(["P03"], evidence_signature="ffg-v1")
    checklist.apply_results([{
        "check_id": "P03",
        "verdict": "undetermined",
        "description": "\u672a\u627e\u5230\u9690\u79c1\u5f39\u7a97",
        "evidence": [],
    }])

    assert checklist.status["P03"] == CHECK_STATUS_DEFERRED
    assert checklist.attempts["P03"] == 1
    assert checklist.reopen_deferred("ffg-v1") == []
    assert checklist.reopen_deferred("ffg-v2") == ["P03"]

    checklist.mark_executing(["P03"], evidence_signature="ffg-v2")
    checklist.apply_results([{
        "check_id": "P03",
        "verdict": "undetermined",
        "description": "\u7b2c\u4e8c\u6b21\u4ecd\u65e0\u5145\u5206\u8bc1\u636e",
        "evidence": [],
    }])

    assert checklist.status["P03"] == CHECK_STATUS_DEFERRED
    assert checklist.attempts["P03"] == 2
    assert checklist.summary()["undetermined"] == 1
    assert checklist.summary()["resolved"] == 0
    assert checklist.summary()["unresolved"] == 1

    # After the direct retry budget, first record the discovery baseline after compliance execution.
    # The same graph and force cannot trigger a mechanical third review.
    assert checklist.reopen_deferred("ffg-v2") == []
    assert checklist.reopen_deferred("ffg-v2", force=True) == []

    # Subsequent ordinary exploration can reopen the check when relevant evidence appears.
    assert checklist.reopen_deferred("ffg-v3") == ["P03"]


def test_legacy_done_undetermined_state_migrates_to_deferred():
    checklist = ComplianceChecklist.from_state({
        "status": {"R014": "done"},
        "findings": [{
            "check_id": "R014",
            "verdict": "undetermined",
            "description": "\u65e7\u683c\u5f0f\u8bc1\u636e\u4e0d\u8db3",
            "evidence": [],
        }],
    })

    assert checklist.status["R014"] == CHECK_STATUS_DEFERRED
    assert checklist.attempts["R014"] == 1

    exhausted = ComplianceChecklist.from_state({
        "status": {"R014": "done"},
        "findings": [{
            "check_id": "R014",
            "verdict": "undetermined",
            "description": "\u65e7\u683c\u5f0f\u7b2c\u4e8c\u6b21\u4ecd\u8bc1\u636e\u4e0d\u8db3",
            "evidence": [],
        }],
        "attempts": {"R014": 2},
        "evidence_signatures": {"R014": "ffg-v2"},
    })
    assert exhausted.status["R014"] == CHECK_STATUS_DEFERRED
    assert exhausted.discovery_baselines["R014"] == ""


def test_interrupted_executing_state_returns_to_pending_without_spending_attempt():
    checklist = ComplianceChecklist(
        status={"R014": CHECK_STATUS_EXECUTING},
        attempts={"R014": 1},
        evidence_signatures={"R014": "ffg-v1"},
    )

    assert checklist.recover_interrupted() == ["R014"]
    assert checklist.status["R014"] == "pending"
    assert checklist.attempts["R014"] == 0
    assert "R014" not in checklist.evidence_signatures


def test_build_compliance_scenarios_batches_same_group():
    group_rules = [
        rule for rule in gui_rules()
        if rule["checkpoint"]["group"] == "first_launch_flow"
    ]
    navigation_scenario, test_scenario, plan_extra = build_compliance_scenarios(group_rules)
    assert navigation_scenario == ""
    assert plan_extra["compliance_group"] == "first_launch_flow"
    assert plan_extra["compliance_cold_start"] is True
    assert "actually select Disagree/Deny once" in test_scenario
    assert "Seeing the button alone is insufficient" in test_scenario
    assert len(plan_extra["compliance_check_ids"]) == len(group_rules)
    for rule in group_rules:
        assert rule["rule_id"] in test_scenario
        assert rule["clause"][:10] in test_scenario
    assert "Compliance screening" in test_scenario


def test_ffg_change_signature_and_guidance_expose_new_compliance_route():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="\u9996\u9875"))
    graph.add_node(FunctionNode(id="func_2", name="\u8bbe\u7f6e"))
    home = Page(id="page_1", name="\u9996\u9875", controls=["\u6211\u7684\u6309\u94ae"])
    settings = Page(
        id="page_2",
        name="\u9690\u79c1\u8bbe\u7f6e",
        controls=["\u4e2a\u6027\u5316\u63a8\u8350\u5f00\u5173", "\u6743\u9650\u8c03\u7528\u8bb0\u5f55"],
    )
    graph.add_page(home, "func_1")
    graph.add_page(settings, "func_2")
    before = scenario_plan._compliance_evidence_signature(graph)
    graph.add_page_transition(PageTransition("page_1", "page_2", "\u70b9\u51fb\u6211\u7684，\u518d\u70b9\u51fb\u8bbe\u7f6e"))
    after = scenario_plan._compliance_evidence_signature(graph)

    navigation, guidance = scenario_plan._compliance_ffg_guidance(
        graph,
        home,
        [rule_by_id("R019")],
    )

    assert before != after
    assert "\u70b9\u51fb\u6211\u7684，\u518d\u70b9\u51fb\u8bbe\u7f6e" in navigation
    assert "\u9690\u79c1\u8bbe\u7f6e" in guidance
    assert "\u4e2a\u6027\u5316\u63a8\u8350\u5f00\u5173" in guidance


def test_deferred_compliance_terms_prioritize_related_unfinished_function():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="\u5185\u5bb9\u6d4f\u89c8"))
    graph.add_node(FunctionNode(id="func_2", name="\u9690\u79c1\u8bbe\u7f6e"))
    home = Page(id="page_1", name="\u5185\u5bb9\u9996\u9875", controls=["\u5185\u5bb9\u5217\u8868"])
    settings = Page(id="page_2", name="\u9690\u79c1\u8bbe\u7f6e", controls=["\u4e2a\u6027\u5316\u63a8\u8350\u5f00\u5173"])
    graph.add_page(home, "func_1")
    graph.add_page(settings, "func_2")
    graph.add_page_transition(PageTransition("page_1", "page_2", "\u70b9\u51fb\u6211\u7684\u548c\u8bbe\u7f6e"))

    target = scenario_plan._select_intra_target(
        graph,
        home,
        {},
        ("\u8bbe\u7f6e", "\u9690\u79c1", "\u63a8\u8350"),
    )

    assert target is not None
    assert target.id == "func_2"


def test_planner_reopens_deferred_group_after_ffg_progress(tmp_path):
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(json.dumps({
        "version": 1,
        "rules": [{
            "rule_id": "X001",
            "regulation": "\u6d4b\u8bd5\u89c4\u5219",
            "clause": "\u5e94\u63d0\u4f9b\u4e2a\u6027\u5316\u63a8\u8350\u5173\u95ed\u9009\u9879",
            "detectability": "gui_direct",
            "checkpoint": {
                "group": "settings_search",
                "objective": "\u8fdb\u5165\u8bbe\u7f6e\u5e76\u67e5\u627e\u4e2a\u6027\u5316\u63a8\u8350\u5173\u95ed\u9009\u9879。",
                "verdict_hints": ["\u5173\u95ed\u9009\u9879\u662f\u5426\u5b58\u5728"],
                "cold_start": False,
            },
        }],
    }, ensure_ascii=False), encoding="utf-8")
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="\u9996\u9875"))
    home = Page(id="page_1", name="\u9996\u9875", controls=["\u6211\u7684\u6309\u94ae"])
    graph.add_page(home, "func_1")
    custom_rule = gui_rules(str(rules_path))[0]
    old_signature = scenario_plan._compliance_evidence_signature(graph, [custom_rule])
    initial = ComplianceChecklist(
        status={"X001": CHECK_STATUS_DEFERRED},
        findings=[{
            "check_id": "X001",
            "verdict": "undetermined",
            "description": "\u672a\u627e\u5230\u8bbe\u7f6e\u5165\u53e3",
            "evidence": [],
        }],
        attempts={"X001": 1},
        evidence_signatures={"X001": old_signature},
    )
    compliance_worker.reset()
    try:
        compliance_worker.configure(object(), initial_checklist=initial.to_state())
        state = {
            "compliance_detection_enabled": True,
            "compliance_rules_path": str(rules_path),
        }
        assert scenario_plan._maybe_schedule_compliance_check(
            state, graph, home
        ) is None

        graph.add_node(FunctionNode(id="func_2", name="\u8bbe\u7f6e"))
        settings = Page(id="page_2", name="\u9690\u79c1\u8bbe\u7f6e", controls=["\u4e2a\u6027\u5316\u63a8\u8350\u5f00\u5173"])
        graph.add_page(settings, "func_2")
        graph.add_page_transition(PageTransition("page_1", "page_2", "\u70b9\u51fb\u6211\u7684\u548c\u8bbe\u7f6e"))

        update = scenario_plan._maybe_schedule_compliance_check(state, graph, home)

        assert update is not None
        assert update["current_plan"]["scenario_type"] == "compliance_check"
        assert update["current_plan"]["compliance_attempt"] == 2
        assert update["current_plan"]["compliance_check_ids"] == ["X001"]
        assert "\u70b9\u51fb\u6211\u7684\u548c\u8bbe\u7f6e" in update["current_navigation_scenario"]
        assert "attempt 2" in update["current_test_scenario"]
        assert "\u672a\u627e\u5230\u8bbe\u7f6e\u5165\u53e3" in update["current_test_scenario"]
        assert update["compliance_checklist"]["status"]["X001"] == CHECK_STATUS_EXECUTING
    finally:
        compliance_worker.reset()


def test_planner_after_retry_budget_waits_for_related_discovery(tmp_path):
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(json.dumps({
        "version": 1,
        "rules": [{
            "rule_id": "X001",
            "regulation": "\u6d4b\u8bd5\u89c4\u5219",
            "clause": "\u5e94\u63d0\u4f9b\u4e2a\u6027\u5316\u63a8\u8350\u5173\u95ed\u9009\u9879",
            "detectability": "gui_direct",
            "checkpoint": {
                "group": "settings_search",
                "objective": "\u8fdb\u5165\u8bbe\u7f6e\u5e76\u67e5\u627e\u4e2a\u6027\u5316\u63a8\u8350\u5173\u95ed\u9009\u9879。",
                "verdict_hints": ["\u5173\u95ed\u9009\u9879\u662f\u5426\u5b58\u5728"],
                "cold_start": False,
            },
        }],
    }, ensure_ascii=False), encoding="utf-8")
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="\u9996\u9875"))
    home = Page(id="page_1", name="\u9996\u9875", controls=["\u5546\u54c1\u5217\u8868"])
    graph.add_page(home, "func_1")
    initial = ComplianceChecklist(
        status={"X001": CHECK_STATUS_DEFERRED},
        findings=[{
            "check_id": "X001",
            "verdict": "undetermined",
            "description": "\u4e24\u6b21\u4ecd\u672a\u627e\u5230\u8bbe\u7f6e\u5165\u53e3",
            "evidence": [],
        }],
        attempts={"X001": 2},
        evidence_signatures={"X001": "pre-run-signature"},
        discovery_baselines={"X001": ""},
    )
    compliance_worker.reset()
    try:
        compliance_worker.configure(object(), initial_checklist=initial.to_state())
        state = {
            "compliance_detection_enabled": True,
            "compliance_rules_path": str(rules_path),
        }

        # First record the graph baseline after compliance retries without starting a third attempt.
        assert scenario_plan._maybe_schedule_compliance_check(state, graph, home) is None

        # An unrelated product discovery does not reopen a settings rule.
        graph.add_node(FunctionNode(id="func_2", name="\u5546\u54c1\u8be6\u60c5"))
        graph.add_page(Page(id="page_2", name="\u5546\u54c1\u8be6\u60c5"), "func_2")
        graph.add_page_transition(PageTransition("page_1", "page_2", "\u70b9\u51fb\u5546\u54c1"))
        assert scenario_plan._maybe_schedule_compliance_check(state, graph, home) is None

        # Discovering a relevant settings entry resumes compliance evaluation.
        graph.add_node(FunctionNode(id="func_3", name="\u9690\u79c1\u8bbe\u7f6e"))
        graph.add_page(
            Page(id="page_3", name="\u9690\u79c1\u8bbe\u7f6e", controls=["\u4e2a\u6027\u5316\u63a8\u8350\u5f00\u5173"]),
            "func_3",
        )
        graph.add_page_transition(PageTransition("page_1", "page_3", "\u70b9\u51fb\u8bbe\u7f6e"))
        update = scenario_plan._maybe_schedule_compliance_check(state, graph, home)

        assert update is not None
        assert update["current_plan"]["compliance_attempt"] == 3
        assert update["current_plan"]["compliance_check_ids"] == ["X001"]
        assert "\u70b9\u51fb\u8bbe\u7f6e" in update["current_navigation_scenario"]
    finally:
        compliance_worker.reset()


def test_related_compliance_signature_ignores_unrelated_product_exploration():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="\u9996\u9875"))
    graph.add_page(Page(id="page_1", name="\u9996\u9875", controls=["\u5546\u54c1\u5217\u8868"]), "func_1")
    settings_rule = rule_by_id("R019")
    before = scenario_plan._compliance_evidence_signature(graph, [settings_rule])

    graph.add_node(FunctionNode(id="func_2", name="\u5546\u54c1\u8be6\u60c5"))
    graph.add_page(
        Page(id="page_2", name="\u5546\u54c1\u8be6\u60c5", controls=["\u6536\u85cf", "\u5206\u4eab"]),
        "func_2",
    )
    graph.add_page_transition(PageTransition("page_1", "page_2", "\u70b9\u51fb\u5546\u54c1"))

    after = scenario_plan._compliance_evidence_signature(graph, [settings_rule])
    assert before == after


def test_related_compliance_signature_tracks_new_visual_evidence():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="\u9690\u79c1\u8bbe\u7f6e"))
    page = Page(
        id="page_1",
        name="\u9690\u79c1\u8bbe\u7f6e",
        controls=["\u4e2a\u6027\u5316\u63a8\u8350\u5f00\u5173"],
        representative_screenshot="shot-v1",
    )
    graph.add_page(page, "func_1")
    settings_rule = rule_by_id("R019")
    before = scenario_plan._compliance_evidence_signature(graph, [settings_rule])

    page.add_evidence_screenshot("shot-v2")
    after = scenario_plan._compliance_evidence_signature(graph, [settings_rule])

    assert before != after


# ----------------------------------------------------------------------
# Trace classification.
# ----------------------------------------------------------------------
def _screenshot_step(index: int, action: str = "click", **extra) -> dict:
    step = {
        "action_name": action,
        "current_page": {"screenshot": f"fake-b64-{index}"},
        "trace_phase": "test",
    }
    step.update(extra)
    return step


def test_collect_steps_keeps_permission_dialog_observations():
    trace = [
        _screenshot_step(1, "click"),
        {  # Exclude steps without screenshots.
            "action_name": "wait",
            "current_page": None,
            "trace_phase": "test",
        },
        _screenshot_step(3, "click", is_interruption=True, external_context=True),
    ]
    steps = compliance_detection._collect_steps(trace)
    assert [step["raw_step_index"] for step in steps] == [1, 3]


def test_compliance_messages_preserve_exact_action_and_result_relationship(monkeypatch):
    monkeypatch.setattr(compliance_detection, "_compact_screenshot", lambda shot: shot)
    steps = compliance_detection._collect_steps([
        _screenshot_step(
            1,
            "click",
            action_description="\u70b9\u51fb\u7981\u6b62\u6309\u94ae",
            thought="\u62d2\u7edd\u5b9a\u4f4d\u6743\u9650",
            x=600,
            y=2492,
        ),
        _screenshot_step(2, "finished", action_description="\u5b8c\u6210\u62d2\u7edd\u8def\u5f84"),
    ])

    messages = compliance_detection._build_messages(
        {"current_test_scenario": "\u62d2\u7edd\u5b9a\u4f4d\u6743\u9650\u540e\u89c2\u5bdf\u529f\u80fd"},
        steps,
        [rule_by_id("R030")],
    )
    content = messages[1].content
    text_parts = [item["text"] for item in content if item.get("type") == "text"]
    rendered = "\n".join(text_parts)

    assert "[ACTION_DESCRIPTION] \u70b9\u51fb\u7981\u6b62\u6309\u94ae" in rendered
    assert "x=600, y=2492" in rendered
    assert "This image shows the state after STEP 1 was executed." in rendered
    assert "must not be described as never denying permission" in rendered


def test_analyze_skips_plain_exploration_without_check_ids(monkeypatch):
    called = []
    monkeypatch.setattr(
        compliance_detection,
        "invoke_llm_json_with_retry",
        lambda *args, **kwargs: called.append(1) or {},
    )
    state = {
        "current_plan": {"scenario_type": "intra_test", "compliance_check_ids": []},
        "navigation_execution_trace": [],
        "scenario_execution_trace": [_screenshot_step(1)],
    }
    result = compliance_detection.analyze_execution_for_compliance(object(), state)
    assert result == {"check_results": []}
    assert not called


def test_analyze_fills_missing_verdicts_as_undetermined(monkeypatch):
    monkeypatch.setattr(
        compliance_detection,
        "invoke_llm_json_with_retry",
        lambda llm, messages, **kwargs: {
            "check_results": [
                {"check_id": "R014", "verdict": "violation", "description": "\u65e0\u9690\u79c1\u5f39\u7a97\u76f4\u63a5\u8fdb\u5165\u9996\u9875", "evidence": [{"global_step": 1, "role": "before"}]},
                {"check_id": "R013", "verdict": "garbage", "description": "\u975e\u6cd5\u7ed3\u8bba"},
            ]
        },
    )
    monkeypatch.setattr(
        compliance_detection, "_compact_screenshot", lambda shot: shot
    )
    state = {
        "current_plan": {
            "scenario_type": "compliance_check",
            "compliance_check_ids": ["R013", "R014", "R019"],
        },
        "current_test_scenario": "\u5408\u89c4\u68c0\u6d4b",
        "navigation_execution_trace": [],
        "scenario_execution_trace": [_screenshot_step(1), _screenshot_step(2)],
    }
    result = compliance_detection.analyze_execution_for_compliance(object(), state)
    verdicts = {item["check_id"]: item["verdict"] for item in result["check_results"]}
    assert verdicts == {"R014": "violation", "R013": "undetermined", "R019": "undetermined"}


def test_analyze_without_screenshots_marks_all_undetermined(monkeypatch):
    monkeypatch.setattr(
        compliance_detection,
        "invoke_llm_json_with_retry",
        lambda *args, **kwargs: pytest.fail("\u65e0\u622a\u56fe\u65f6\u4e0d\u5e94\u8c03\u7528 LLM"),
    )
    state = {
        "current_plan": {"compliance_check_ids": ["R014"]},
        "navigation_execution_trace": [],
        "scenario_execution_trace": [{"action_name": "click", "current_page": None}],
    }
    result = compliance_detection.analyze_execution_for_compliance(object(), state)
    assert result["check_results"] == [
        {
            "check_id": "R014",
            "verdict": "undetermined",
            "description": "This trace contains no screenshot-bearing steps; evidence is unavailable.",
            "evidence": [],
        }
    ]


def test_build_compliance_detection_job_extracts_plan_check_ids():
    job = build_compliance_detection_job({
        "navigation_execution_trace": [],
        "scenario_execution_trace": [],
        "current_plan": {
            "scenario_type": "compliance_check",
            "compliance_check_ids": ["R014", "", "R019"],
        },
    })
    assert plan_check_ids(job["current_plan"]) == ["R014", "R019"]


def test_custom_rules_path_is_used_by_job_analysis_and_summary(tmp_path, monkeypatch):
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(json.dumps({
        "version": 1,
        "rules": [{
            "rule_id": "X001",
            "regulation": "\u6d4b\u8bd5\u89c4\u5219",
            "clause": "\u9996\u542f\u5e94\u5c55\u793a\u544a\u77e5\u5f39\u7a97",
            "detectability": "gui_direct",
            "checkpoint": {
                "group": "first_launch_flow",
                "objective": "\u89c2\u5bdf\u9996\u542f\u5f39\u7a97",
                "verdict_hints": ["\u662f\u5426\u5c55\u793a"],
                "cold_start": True,
            },
        }],
    }, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(
        compliance_detection,
        "invoke_llm_json_with_retry",
        lambda *args, **kwargs: {"check_results": [{
            "check_id": "X001",
            "verdict": "compliant",
            "description": "\u9996\u5c4f\u5c55\u793a\u544a\u77e5\u5f39\u7a97",
            "evidence": [{"global_step": 1, "role": "before"}],
        }]},
    )
    monkeypatch.setattr(compliance_detection, "_compact_screenshot", lambda shot: shot)
    state = {
        "compliance_rules_path": str(rules_path),
        "current_plan": {"compliance_check_ids": ["X001"]},
        "scenario_execution_trace": [_screenshot_step(1)],
    }

    job = build_compliance_detection_job(state)
    result = compliance_detection.analyze_execution_for_compliance(object(), job)

    assert job["compliance_rules_path"] == str(rules_path)
    assert result["check_results"][0]["check_id"] == "X001"
    assert ComplianceChecklist().summary(str(rules_path))["gui_total"] == 1


def test_compliance_worker_processes_job_and_updates_checklist(monkeypatch):
    compliance_worker.reset()
    try:
        monkeypatch.setattr(
            compliance_detection,
            "invoke_llm_json_with_retry",
            lambda llm, messages, **kwargs: {
                "check_results": [
                    {"check_id": "R014", "verdict": "compliant", "description": "\u6709\u5f39\u7a97", "evidence": [{"global_step": 1, "role": "before"}]},
                ]
            },
        )
        monkeypatch.setattr(
            compliance_detection, "_compact_screenshot", lambda shot: shot
        )
        compliance_worker.configure(object())
        job = build_compliance_detection_job({
            "navigation_execution_trace": [],
            "scenario_execution_trace": [_screenshot_step(1)],
            "current_plan": {"compliance_check_ids": ["R014"]},
        })
        assert compliance_worker.submit(job) is True
        assert compliance_worker.flush(timeout=10) is True
        snapshot = compliance_worker.snapshot()
        assert snapshot["status"]["R014"] == CHECK_STATUS_DONE
        assert snapshot["findings"][0]["verdict"] == "compliant"
        # Checks with final conclusions are no longer pending.
        assert all(
            rule["rule_id"] != "R014" for rule in pending_gui_checkpoints(snapshot["status"])
        )
    finally:
        compliance_worker.reset()


# ----------------------------------------------------------------------
# Reporting.
# ----------------------------------------------------------------------
def test_build_compliance_report_covers_only_gui_rules(tmp_path):
    checklist = ComplianceChecklist()
    checklist.apply_results([
        {"check_id": "P03", "verdict": "violation", "description": "\u65e0\u9690\u79c1\u5f39\u7a97", "evidence": [{"global_step": 1, "role": "before"}]},
        {"check_id": "P06", "verdict": "compliant", "description": "\u6709\u6ce8\u9500\u5165\u53e3", "evidence": []},
    ])
    report = build_compliance_report(checklist, app_name="\u6d4b\u8bd5App", package_name="com.test")
    assert report["app_name"] == "\u6d4b\u8bd5App"
    assert "dynamic GUI screening" in report["scope_note"]
    assert len(report["checks"]) == 20
    by_id = {entry["rule_id"]: entry for entry in report["checks"]}
    assert by_id["P03"]["verdict"] == "violation"
    assert by_id["P03"]["evidence"] == [{"global_step": 1, "role": "before"}]
    assert by_id["P06"]["verdict"] == "compliant"
    assert "R001" not in by_id
    assert report["summary"]["total_rules"] == 20
    assert report["summary"]["violation"] == 1
    assert report["summary"]["processed"] == 2
    assert report["summary"]["resolved"] == 2

    save_compliance_report(checklist, str(tmp_path), app_name="\u6d4b\u8bd5App")
    data = json.loads((tmp_path / "compliance_report.json").read_text(encoding="utf-8"))
    assert len(data["checks"]) == 20
    markdown = (tmp_path / "compliance_report.md").read_text(encoding="utf-8")
    assert "Verdict scope" in markdown
    assert "\u65e0\u9690\u79c1\u5f39\u7a97" in markdown
    assert "Violation" in markdown
    assert "Processed checkpoints" in markdown
    assert "Resolved" in markdown
