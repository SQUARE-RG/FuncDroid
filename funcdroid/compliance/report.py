"""Generate JSON and Markdown compliance reports."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from compliance.checkpoints import ComplianceChecklist
from compliance.rule_library import gui_rules

VERDICT_LABELS = {
    "compliant": "Compliant",
    "violation": "Violation",
    "undetermined": "Insufficient evidence",
    "not_applicable": "Not applicable",
    "": "Not executed",
}

STATUS_LABELS = {
    "pending": "Pending",
    "executing": "Executing",
    "deferred": "Deferred (awaiting new evidence)",
    "done": "Resolved",
    "": "Pending",
}

GUI_SCREENING_SCOPE_NOTE = (
    "This report covers dynamic GUI screening. Compliant means only that observed applicable interfaces and interaction paths met the rule. "
    "It is not comprehensive evidence about network traffic, background code, server-side processing, or long-term operation."
)


def build_compliance_report(
    checklist: ComplianceChecklist,
    app_name: str = "",
    package_name: str = "",
    rules_path: Any = None,
) -> dict[str, Any]:
    """Summarize GUI-screenable rules, verdicts, and supporting evidence."""
    all_rules = gui_rules(rules_path)

    entries: list[dict[str, Any]] = []
    for rule in all_rules:
        entry = {
            "rule_id": rule["rule_id"],
            "name": rule.get("name", rule["clause"]),
            "regulation": rule["regulation"],
            "clause": rule["clause"],
            "detectability": rule.get("detectability", "gui_direct"),
            "source_rule_ids": rule.get("source_rule_ids", []),
            "match_policy": rule.get("checkpoint", {}).get("match_policy", "observed_paths_only"),
        }
        finding = next(
            (
                item
                for item in checklist.findings
                if str(item.get("check_id", "")).strip() == rule["rule_id"]
            ),
            None,
        )
        entry["group"] = rule.get("checkpoint", {}).get("group", "")
        entry["verdict"] = str(finding.get("verdict", "")).strip() if finding else ""
        entry["description"] = str(finding.get("description", "")).strip() if finding else ""
        entry["evidence"] = finding.get("evidence", []) if finding else []
        entry["trace_path"] = finding.get("trace_path", "") if finding else ""
        entry["history"] = finding.get("history", []) if finding else []
        entry["scope"] = "observed_paths_only"
        if finding and isinstance(finding.get("functionality_evidence"), dict):
            entry["functionality_evidence"] = dict(finding["functionality_evidence"])
        if finding and isinstance(finding.get("applicability"), dict):
            entry["applicability"] = finding["applicability"]
        entry["attempts"] = int(checklist.attempts.get(rule["rule_id"], 0) or 0)
        entry["status"] = str(checklist.status.get(rule["rule_id"], "") or "pending")
        entries.append(entry)

    summary = checklist.summary(rules_path)
    return {
        "app_name": app_name,
        "package_name": package_name,
        "scope_note": GUI_SCREENING_SCOPE_NOTE,
        "ruleset_id": "funcdroid-compliance-20-v1" if {
            rule["rule_id"] for rule in all_rules
        } == {f"P{index:02d}" for index in range(1, 21)} else "legacy-or-custom",
        "summary": {
            **summary,
            "total_rules": len(all_rules),
        },
        "checks": entries,
    }


def _render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines: list[str] = [
        "# App Dynamic Compliance Screening Report",
        "",
        f"- Application: {report.get('app_name', '') or 'unknown'}（{report.get('package_name', '') or 'unknown package'}）",
        f"- GUI screening rules: {summary.get('total_rules', 0)}",
        f"- Processed checkpoints: {summary.get('processed', summary.get('executed', 0))}/{summary.get('gui_total', 0)}"
        f"｜Resolved {summary.get('resolved', 0)}"
        f"｜Compliant {summary.get('compliant', 0)}｜Violation {summary.get('violation', 0)}"
        f"｜Insufficient evidence {summary.get('undetermined', 0)}｜Not applicable {summary.get('not_applicable', 0)}"
        f"｜Awaiting new evidence {summary.get('unresolved', summary.get('deferred', 0))}"
        f"｜Not processed {summary.get('untested', summary.get('pending', 0))}",
        f"- Verdict scope: {report.get('scope_note', GUI_SCREENING_SCOPE_NOTE)}",
        "",
    ]
    by_regulation: dict[str, list[dict[str, Any]]] = {}
    for entry in report.get("checks", []):
        by_regulation.setdefault(entry.get("regulation", ""), []).append(entry)
    for regulation, entries in by_regulation.items():
        lines.append(f"## {regulation}")
        lines.append("")
        lines.append("| ID | Rule | Status | Verdict | Attempts | Description |")
        lines.append("| --- | --- | --- | --- | ---: | --- |")
        for entry in entries:
            verdict = VERDICT_LABELS.get(entry.get("verdict", ""), entry.get("verdict", ""))
            status = STATUS_LABELS.get(entry.get("status", ""), entry.get("status", ""))
            clause = str(entry.get("clause", "")).replace("|", "\\|").replace("\n", " ")
            description = str(entry.get("description", "")).replace("|", "\\|").replace("\n", " ")
            if entry.get("trace_path"):
                description += f" [Execution evidence]({entry['trace_path']})"
            lines.append(
                f"| {entry.get('rule_id', '')} | {clause} | {status} | {verdict} | "
                f"{entry.get('attempts', 0)} | {description} |"
            )
        lines.append("")
    return "\n".join(lines)


def save_compliance_report(
    checklist: ComplianceChecklist,
    output_dir: str,
    app_name: str = "",
    package_name: str = "",
    rules_path: Any = None,
) -> dict[str, Any]:
    report = build_compliance_report(checklist, app_name, package_name, rules_path)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(root / "compliance_report.json", json.dumps(report, ensure_ascii=False, indent=2))
    _atomic_write_text(root / "compliance_report.md", _render_markdown(report))
    return report


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
