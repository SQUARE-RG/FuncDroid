# Multilingual UI and model-response fixtures use Unicode escapes to preserve their exact text.
"""Tests for new problem scheduling, legacy report compatibility, and official rule scope."""

import json

from compliance.checkpoints import ComplianceChecklist
from compliance.report import build_compliance_report
from compliance.rule_library import LEGACY_RULES_PATH, gui_rules, load_rules, rule_by_id
from scripts.batch_compliance_run import AppEntry, _official_rules_for, build_batch_report
from workflow.compliance_detection import _extract_raw_check_results


def test_problem_rules_schedule_once_and_preserve_all_legacy_bases():
    rules = gui_rules()
    assert {r["rule_id"] for r in rules} == {f"P{i:02d}" for i in range(1, 21)}
    assert {rid for r in rules for rid in r["source_rule_ids"]} == {
        r["rule_id"] for r in load_rules(LEGACY_RULES_PATH)
    }
    checklist = ComplianceChecklist()
    scheduled = []
    while group := checklist.next_group_rules():
        ids = [r["rule_id"] for r in group]
        scheduled.extend(ids)
        checklist.mark_executing(ids)
    assert len(scheduled) == len(set(scheduled)) == 20
    # Old IDs remain readable, but are never inserted into a new run.
    assert rule_by_id("R031") is not None
    report = build_compliance_report(checklist)
    assert report["summary"]["total_rules"] == 20
    assert all(c["rule_id"].startswith("P") for c in report["checks"])


def test_top_level_problem_verdict_is_not_dropped():
    result = _extract_raw_check_results({"P09": {"verdict": "violation"}})
    assert result == [{"check_id": "P09", "verdict": "violation"}]


def test_official_comparison_selects_rules_for_the_saved_ruleset():
    official = {"official_issues": ["APP \u5f3a\u5236、\u9891\u7e41、\u8fc7\u5ea6\u7d22\u53d6\u6743\u9650", "\u672a\u63d0\u4f9b\u6709\u6548\u8d26\u53f7\u6ce8\u9500\u529f\u80fd"]}
    legacy = {official["official_issues"][0]: ["R030"], official["official_issues"][1]: ["R031"]}
    _, current = _official_rules_for(official, legacy, ["P06", "P10"])
    assert current[official["official_issues"][0]] == ["P06", "P07"]
    assert current[official["official_issues"][1]] == ["P10"]
    _, saved = _official_rules_for(official, legacy, ["R030", "R031"])
    assert saved == legacy


def test_related_issue_counts_but_login_block_and_unrelated_rights_do_not(tmp_path):
    entry = AppEntry(1, "Mock", "com.mock")
    output = tmp_path / entry.dir_name
    output.mkdir()
    (output / "run_result.json").write_text(json.dumps({"package_name": entry.package, "status": "success"}))
    (output / "compliance_report.json").write_text(json.dumps({
        "summary": {"gui_total": 20, "processed": 20, "violation": 2},
        "checks": [
            {"rule_id": "P06", "verdict": "violation", "description": "\u62d2\u7edd\u540e\u53cd\u590d\u7d22\u53d6\u6743\u9650"},
            {"rule_id": "P10", "verdict": "undetermined", "description": "\u6ce8\u9500\u5165\u53e3\u53d7\u767b\u5f55\u9650\u5236"},
            {"rule_id": "P11", "verdict": "violation", "description": "\u64a4\u56de\u540c\u610f\u6e20\u9053\u7f3a\u5931"},
        ],
    }))
    official = tmp_path / "official.json"
    official.write_text(json.dumps({"local_apks": [{
        "app": "Mock", "version_relation": "same", "official_issues": [
            "\u8fdd\u89c4\u6536\u96c6\u4e2a\u4eba\u4fe1\u606f", "\u672a\u63d0\u4f9b\u6709\u6548\u8d26\u53f7\u6ce8\u9500\u529f\u80fd"
        ],
    }], "issue_to_rule_mapping": {
        "\u8fdd\u89c4\u6536\u96c6\u4e2a\u4eba\u4fe1\u606f": ["R035"], "\u672a\u63d0\u4f9b\u6709\u6548\u8d26\u53f7\u6ce8\u9500\u529f\u80fd": ["R031"]
    }}))
    build_batch_report(tmp_path, [entry], official_table=official)
    comparison = json.loads((tmp_path / "batch_report.json").read_text())[0]["official_comparison"]
    assert comparison["scope"] == "related_problem_match"
    assert [i["status"] for i in comparison["issues"]] == ["detected_related", "not_detected"]


def test_official_issue_normalization_supports_english_and_legacy_labels():
    from compliance.official import official_problem_rule_ids

    assert official_problem_rule_ids("Information window cannot be closed") == ["P14"]
    assert official_problem_rule_ids(" APP Information window cannot be closed ") == ["P14"]
    assert official_problem_rule_ids("\u4fe1\u606f\u7a97\u53e3\u65e0\u6cd5\u5173\u95ed") == ["P14"]
