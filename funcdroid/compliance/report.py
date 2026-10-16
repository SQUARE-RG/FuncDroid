"""合规检测报告生成（json + markdown）。"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from compliance.checkpoints import ComplianceChecklist
from compliance.rule_library import gui_rules

VERDICT_LABELS = {
    "compliant": "符合",
    "violation": "违规",
    "undetermined": "证据不足",
    "not_applicable": "不适用",
    "": "未执行",
}

STATUS_LABELS = {
    "pending": "待执行",
    "executing": "执行中",
    "deferred": "未决（等待新证据）",
    "done": "已形成结论",
    "": "待执行",
}

GUI_SCREENING_SCOPE_NOTE = (
    "本报告采用 GUI 动态筛查口径：‘符合’仅表示已观察到的适用界面与交互路径符合，"
    "不构成对网络请求、后台代码、服务端处理或长期运营行为的完整证明。"
)


def build_compliance_report(
    checklist: ComplianceChecklist,
    app_name: str = "",
    package_name: str = "",
    rules_path: Any = None,
) -> dict[str, Any]:
    """汇总规则库中的 GUI 动态检测条款及其结论与证据。"""
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
        "# App 合规性动态检测报告",
        "",
        f"- 应用：{report.get('app_name', '') or '未知'}（{report.get('package_name', '') or '未知包名'}）",
        f"- GUI 动态检测规则数：{summary.get('total_rules', 0)}",
        f"- 已处理检测点：{summary.get('processed', summary.get('executed', 0))}/{summary.get('gui_total', 0)}"
        f"｜已形成结论 {summary.get('resolved', 0)}"
        f"｜符合 {summary.get('compliant', 0)}｜违规 {summary.get('violation', 0)}"
        f"｜证据不足 {summary.get('undetermined', 0)}｜不适用 {summary.get('not_applicable', 0)}"
        f"｜未决待新证据 {summary.get('unresolved', summary.get('deferred', 0))}"
        f"｜尚未处理 {summary.get('untested', summary.get('pending', 0))}",
        f"- 结论范围：{report.get('scope_note', GUI_SCREENING_SCOPE_NOTE)}",
        "",
    ]
    by_regulation: dict[str, list[dict[str, Any]]] = {}
    for entry in report.get("checks", []):
        by_regulation.setdefault(entry.get("regulation", ""), []).append(entry)
    for regulation, entries in by_regulation.items():
        lines.append(f"## {regulation}")
        lines.append("")
        lines.append("| 编号 | 条款 | 状态 | 结论 | 尝试次数 | 说明 |")
        lines.append("| --- | --- | --- | --- | ---: | --- |")
        for entry in entries:
            verdict = VERDICT_LABELS.get(entry.get("verdict", ""), entry.get("verdict", ""))
            status = STATUS_LABELS.get(entry.get("status", ""), entry.get("status", ""))
            clause = str(entry.get("clause", "")).replace("|", "\\|").replace("\n", " ")
            description = str(entry.get("description", "")).replace("|", "\\|").replace("\n", " ")
            if entry.get("trace_path"):
                description += f" [执行证据]({entry['trace_path']})"
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
