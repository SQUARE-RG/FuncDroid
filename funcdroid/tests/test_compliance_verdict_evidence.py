# Multilingual UI and model-response fixtures use Unicode escapes to preserve their exact text.
"""Regressions from the first translation-app device pilot."""

import pytest

from workflow import compliance_detection as detection
from compliance.checkpoints import ComplianceChecklist
from compliance.report import build_compliance_report
from compliance.rule_library import LEGACY_RULES_PATH


@pytest.mark.parametrize("evidence", [[], [{"global_step": 99, "role": "before"}]])
def test_no_ad_in_partial_trace_does_not_close_rule(monkeypatch, evidence):
    monkeypatch.setattr(detection, "_compact_screenshot", lambda value: value)
    monkeypatch.setattr(detection, "invoke_llm_json_with_retry", lambda *args, **kwargs: {
        "check_results": [{"check_id": "R062", "verdict": "not_applicable",
                           "description": "\u672c\u8f6e\u4ec5\u770b\u5230\u9690\u79c1\u5f39\u7a97，\u672a\u51fa\u73b0\u5f00\u5c4f\u5e7f\u544a。", "evidence": evidence}]})
    result = detection.analyze_execution_for_compliance(None, {
        "current_plan": {"compliance_check_ids": ["R062"]},
        "scenario_execution_trace": [{"current_page": {"screenshot": "image"}, "action_name": "finish"}],
    })
    assert result["check_results"][0]["verdict"] == "undetermined"


def test_applicability_requires_positive_basis_even_with_screenshot(monkeypatch):
    monkeypatch.setattr(detection, "_compact_screenshot", lambda value: value)
    monkeypatch.setattr(detection, "invoke_llm_json_with_retry", lambda *args, **kwargs: {
        "check_results": [{"check_id": "R062", "verdict": "not_applicable",
                           "description": "\u5f53\u524d\u622a\u56fe\u6ca1\u6709\u5e7f\u544a。",
                           "evidence": [{"global_step": 1, "role": "before"}]}]})
    result = detection.analyze_execution_for_compliance(None, {
        "current_plan": {"compliance_check_ids": ["R062"]},
        "scenario_execution_trace": [{"current_page": {"screenshot": "image"}, "action_name": "finish"}],
    })
    assert result["check_results"][0]["verdict"] == "undetermined"


@pytest.mark.parametrize("step,expected", [(1, "not_applicable"), (99, "undetermined")])
def test_explicit_exclusion_needs_existing_screenshot(monkeypatch, step, expected):
    monkeypatch.setattr(detection, "_compact_screenshot", lambda value: value)
    monkeypatch.setattr(detection, "invoke_llm_json_with_retry", lambda *args, **kwargs: {
        "check_results": [{"check_id": "R040", "verdict": "not_applicable",
                           "description": "\u4e1a\u52a1\u8303\u56f4\u8bf4\u660e\u660e\u786e\u6392\u9664\u5730\u56fe\u5bfc\u822a\u670d\u52a1。",
                           "applicability": {"basis": "explicit_scope_exclusion",
                                             "explanation": "\u4e1a\u52a1\u8bf4\u660e\u5217\u660e\u4ec5\u63d0\u4f9b\u79bb\u7ebf\u5de5\u5177，\u4e0d\u63d0\u4f9b\u5730\u56fe\u5bfc\u822a\u670d\u52a1。"},
                           "evidence": [{"global_step": step, "role": "before"}]}]})
    result = detection.analyze_execution_for_compliance(None, {
        "current_plan": {"compliance_check_ids": ["R040"]},
        "scenario_execution_trace": [{"current_page": {"screenshot": "image"}, "action_name": "finish"}],
    })
    assert result["check_results"][0]["verdict"] == expected
    if expected == "not_applicable":
        checklist = ComplianceChecklist()
        checklist.apply_results(result["check_results"])
        report = build_compliance_report(checklist, rules_path=LEGACY_RULES_PATH)
        entry = next(item for item in report["checks"] if item["rule_id"] == "R040")
        assert entry["applicability"]["basis"] == "explicit_scope_exclusion"
