# Multilingual UI and model-response fixtures use Unicode escapes to preserve their exact text.
"""Defect-report regression tests for artifact shapes, cross-batch deduplication, and cross-analysis."""

from __future__ import annotations

import json
import os
import sys


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.collect_defect_report import (  # noqa: E402
    load_batch,
    load_compliance_counts,
    render,
)


def _write_app(root, name, package, *, bugs, status="success",
               activities=("A", "B"), elapsed=3600, seed_pages=0):
    app_dir = os.path.join(root, name)
    os.makedirs(app_dir, exist_ok=True)
    result = {
        "status": status,
        "app_name": name,
        "package_name": package,
        "elapsed_seconds": elapsed,
        "high_confidence_bug_count": len(bugs),
        "activity_discovery": {"count": len(activities), "activities": list(activities)},
        "seed_provenance": {"inherited_pages": seed_pages},
        "llm_usage": {"requests": 100, "failures": 0},
    }
    json.dump(result, open(os.path.join(app_dir, "run_result.json"), "w"), ensure_ascii=False)
    json.dump(bugs, open(os.path.join(app_dir, "bugs.json"), "w"), ensure_ascii=False)
    return app_dir


def _bug(bug_id, category, description, confidence="high"):
    return {"bug_id": bug_id, "type": "logic_error", "evidence_category": category,
            "severity": "high", "confidence": confidence, "description": description,
            "expected": "e", "actual": "a"}


def test_hybrid_app_detected_by_single_activity(tmp_path):
    root = str(tmp_path)
    _write_app(root, "Hybrid", "com.hybrid", bugs=[], activities=("io.dcloud.PandoraEntryActivity",))
    batch = load_batch("B", root)
    assert batch["apps"][0]["hybrid"] is True


def test_native_app_not_flagged_as_hybrid(tmp_path):
    root = str(tmp_path)
    _write_app(root, "Native", "com.native", bugs=[], activities=("A", "B", "C"))
    batch = load_batch("B", root)
    assert batch["apps"][0]["hybrid"] is False


def test_only_high_confidence_bugs_are_counted(tmp_path):
    root = str(tmp_path)
    _write_app(root, "App", "com.app", bugs=[
        _bug("BUG-001", "logic_error", "\u786e\u5b9a\u7684\u95ee\u9898"),
        _bug("BUG-002", "logic_error", "\u4f4e\u7f6e\u4fe1\u95ee\u9898", confidence="low"),
    ])
    batch = load_batch("B", root)
    assert batch["apps"][0]["defect_count"] == 1


def test_cross_batch_dedupe_removes_duplicates(tmp_path):
    bug = _bug("BUG-001", "rendering_corruption", "\u540c\u4e00\u6bb5\u63cf\u8ff0" * 20)
    root_a = str(tmp_path / "a")
    root_b = str(tmp_path / "b")
    _write_app(root_a, "App", "com.app", bugs=[bug])
    _write_app(root_b, "App", "com.app", bugs=[dict(bug)])

    batches = [load_batch("A", root_a), load_batch("B", root_b)]
    _, payload = render(batches, compliance=None, dedupe=True)
    assert payload["total_defects"] == 1
    assert payload["deduped"] == 1


def test_dedupe_can_be_disabled(tmp_path):
    bug = _bug("BUG-001", "rendering_corruption", "\u540c\u4e00\u6bb5\u63cf\u8ff0" * 20)
    root_a = str(tmp_path / "a")
    root_b = str(tmp_path / "b")
    _write_app(root_a, "App", "com.app", bugs=[bug])
    _write_app(root_b, "App", "com.app", bugs=[dict(bug)])

    batches = [load_batch("A", root_a), load_batch("B", root_b)]
    _, payload = render(batches, compliance=None, dedupe=False)
    assert payload["total_defects"] == 2


def test_compliance_cross_analysis(tmp_path):
    defect_root = str(tmp_path / "defect")
    compliance_root = str(tmp_path / "compliance")
    _write_app(defect_root, "App", "com.app",
               bugs=[_bug("BUG-001", "logic_error", "\u95ee\u9898")])
    app_dir = os.path.join(compliance_root, "01_App")
    os.makedirs(app_dir, exist_ok=True)
    json.dump({"app_name": "App", "package_name": "com.app",
               "summary": {"violation": 9, "compliant": 5, "undetermined": 3}},
              open(os.path.join(app_dir, "compliance_report.json"), "w"), ensure_ascii=False)

    compliance = load_compliance_counts(compliance_root)
    markdown, payload = render([load_batch("B", defect_root)], compliance=compliance)
    assert payload["cross_analysis"][0]["compliance_violation"] == 9
    assert "Compliance and defect cross-analysis" in markdown


def test_missing_bugs_file_is_tolerated(tmp_path):
    root = str(tmp_path)
    app_dir = _write_app(root, "App", "com.app", bugs=[])
    os.remove(os.path.join(app_dir, "bugs.json"))
    batch = load_batch("B", root)
    assert batch["apps"][0]["defect_count"] == 0
