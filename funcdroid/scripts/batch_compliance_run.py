#!/usr/bin/env python3
"""Run compliance checks sequentially on installed apps, with separate output directories.

Unlike batch_test_apks.py, this runner does not install or uninstall APKs. It calls
main.py for each installed app. Resume decisions use each app's run_result.json:
skip only successful runs with complete artifacts; rule coverage is counted separately. Aggregate compliance_report.json files into
batch_report.md。

Examples:
        python scripts/batch_compliance_run.py --device-id emulator-5554 --app Example:com.example
        python scripts/batch_compliance_run.py --device-id ... --limit 2     # Start with two apps
        python scripts/batch_compliance_run.py --device-id ... --report-only # Regenerate the summary only
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compliance.official import official_problem_rule_ids  # noqa: E402
from scripts._batch_common import (  # noqa: E402,F401 - retain legacy module exports
    BatchInterrupted,
    adb as _common_adb,
    append_jsonl,
    nonnegative_float as _nonnegative_float,
    nonnegative_int as _nonnegative_int,
    query_installed,
    resolve_process_timeout,
    run_artifacts_complete as _common_run_artifacts_complete,
    signal_handler as _signal_handler,
    signal_process_tree as _signal_process_tree,
    terminate_process_tree,
)

DEFAULT_OUTPUT_ROOT = "experiments/output_hegui"
DEFAULT_TOTAL_TIME_MINUTES = 60.0
DEFAULT_COOLDOWN_SECONDS = 60.0
DEFAULT_ADB_TIMEOUT = 30.0
DEVICE_WAIT_TIMEOUT_SECONDS = 300.0
# Per-app hard timeout = test budget + artifact/worker-flush grace period.
# With --total-time 0, use a finite fallback rather than terminating after ten minutes.
PROCESS_TIMEOUT_GRACE_SECONDS = 600.0
UNLIMITED_PROCESS_TIMEOUT_SECONDS = max(
    60.0,
    float(os.getenv("BATCH_PROCESS_TIMEOUT", "7200")),
)
PROCESS_TERMINATE_GRACE_SECONDS = 30.0

VERDICT_LABELS = {
    "compliant": "Compliant",
    "violation": "Violation",
    "undetermined": "Insufficient evidence",
    "not_applicable": "Not applicable",
}

# Report display limits; append an ellipsis when a clause or explanation is truncated.
CLAUSE_DISPLAY_LIMIT = 120
DESCRIPTION_DISPLAY_LIMIT = 600
# Legacy official-notice mapping filename; look in the batch output's parent by default.
OFFICIAL_TABLE_FILENAME = "\u5b98\u65b9\u901a\u62a5\u5bf9\u7167\u8868.json"


@dataclass(frozen=True)
class AppEntry:
    index: int
    name: str
    package: str

    @property
    def dir_name(self) -> str:
        safe_name = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", self.name).strip("_") or self.package
        return f"{self.index:02d}_{safe_name}"


def _adb(
    device_id: str,
    *args: str,
    timeout: float = DEFAULT_ADB_TIMEOUT,
) -> subprocess.CompletedProcess[str]:
    return _common_adb(device_id, *args, timeout=timeout)


def device_online(device_id: str) -> bool:
    try:
        result = _adb(device_id, "get-state")
    except (subprocess.TimeoutExpired, OSError):
        return False
    return result.returncode == 0 and "device" in result.stdout.strip().splitlines()[0:1]


def wait_for_device(device_id: str, timeout: float = DEVICE_WAIT_TIMEOUT_SECONDS) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if device_online(device_id):
            return True
        print(f"[Batch queue] Device offline; waiting for reconnection: {device_id}")
        time.sleep(15)
    return False


def app_installed(device_id: str, package: str) -> bool:
    """Lenient counterpart to batch_test_apks.is_installed, using exact pm list packages matches.

    This caller only decides whether to skip an app. Query errors return False and are logged.
    Use query_installed when the distinction between absent and unknown state matters.
    """
    try:
        return query_installed(_adb, device_id, package)
    except BatchInterrupted:
        raise
    except RuntimeError as exc:
        print(f"[Batch queue] Installation query failed; skipping as unavailable: {package} | {exc}")
        return False


def force_stop_app(device_id: str, package: str) -> None:
    try:
        _adb(device_id, "shell", "am", "force-stop", package)
    except (subprocess.TimeoutExpired, OSError) as exc:
        print(f"[Batch queue] force-stop failed; continuing: {package} | {exc}")


def run_result_status(app_output_dir: Path, package_name: str) -> str:
    """Read the app run status; return empty when the directory or result is absent."""
    result_path = app_output_dir / "run_result.json"
    if not result_path.is_file():
        return ""
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    if not isinstance(payload, dict) or payload.get("package_name") != package_name:
        return ""
    return str(payload.get("status", "") or "")


def run_artifacts_complete(app_output_dir: Path, package_name: str) -> bool:
    """Process success does not prove that the detector produced complete artifacts."""
    return _common_run_artifacts_complete(
        app_output_dir, "compliance", package_name, missing_mode_ok=True, require_checks=True,
    )


def _process_timeout(total_time_minutes: float) -> float:
    return resolve_process_timeout(
        total_time_minutes,
        None,
        grace_seconds=PROCESS_TIMEOUT_GRACE_SECONDS,
        unlimited_timeout_seconds=UNLIMITED_PROCESS_TIMEOUT_SECONDS,
    )


def _terminate_process_tree(
    process: subprocess.Popen,
    grace_seconds: float = PROCESS_TERMINATE_GRACE_SECONDS,
) -> None:
    terminate_process_tree(process, grace_seconds, log_prefix="[Batch queue]")


def _run_one(
    entry: AppEntry,
    *,
    device_id: str,
    total_time_minutes: float,
    batch_root: Path,
    project_root: Path,
) -> dict:
    app_output_dir = batch_root / entry.dir_name
    app_output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(project_root / "main.py"),
        "--mode", "compliance",
        "--device-id", device_id,
        "--package-name", entry.package,
        "--app-name", entry.name,
        "--total-time", str(total_time_minutes),
        "--output-dir", str(app_output_dir),
    ]
    hard_timeout = _process_timeout(total_time_minutes)
    record: dict = {
        "app_name": entry.name,
        "package_name": entry.package,
        "output_dir": str(app_output_dir),
        "command": command,
        "started_at": datetime.now().isoformat(),
    }
    print(
        f"\n[Batch queue] ({entry.index:02d}) Starting: {entry.name} ({entry.package}) | "
        f"budget {total_time_minutes:g} min -> {app_output_dir}"
    )
    started_monotonic = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=str(project_root),
        start_new_session=os.name == "posix",
    )
    try:
        returncode = process.wait(timeout=hard_timeout)
        record["returncode"] = returncode
    except subprocess.TimeoutExpired:
        print(f"[Batch queue] ({entry.index:02d}) Hard timeout exceeded; terminating process tree: {entry.name}")
        _terminate_process_tree(process)
        record["returncode"] = None
        record["timeout_seconds"] = hard_timeout
    except BatchInterrupted as exc:
        _terminate_process_tree(process)
        record["returncode"] = None
        record["interrupted"] = True
        record["reason"] = str(exc)
    record["finished_at"] = datetime.now().isoformat()
    record["elapsed_seconds"] = round(time.monotonic() - started_monotonic, 1)
    record["run_status"] = run_result_status(app_output_dir, entry.package) or "no_result"
    ok = record["returncode"] == 0 and run_artifacts_complete(app_output_dir, entry.package)
    print(
        f"[Batch queue] ({entry.index:02d}) Finished: {entry.name} | "
        f"status={record['run_status']} | exit={record['returncode']} | "
        f"elapsed {record['elapsed_seconds'] / 60:.1f} min"
    )
    record["ok"] = ok
    return record


def _append_batch_summary(batch_root: Path, record: dict) -> None:
    append_jsonl(batch_root / "batch_summary.jsonl", record)


def _clip(text: object, limit: int) -> str:
    """Truncate by character count and append an ellipsis instead of silently cutting text."""
    value = str(text or "")
    return value if len(value) <= limit else value[:limit] + "…"


def _evidence_steps(evidence: object) -> list[int]:
    """Collect screenshot STEP references for reviewing an individual verdict."""
    steps: set[int] = set()
    if isinstance(evidence, list):
        for item in evidence:
            if not isinstance(item, dict):
                continue
            try:
                steps.add(int(item.get("global_step")))
            except (TypeError, ValueError):
                continue
    return sorted(steps)


def _expand_rule_token(token: object) -> list[str]:
    """Expand ranges such as R040-R050 into rule IDs; preserve individual IDs."""
    text = str(token or "").strip()
    match = re.fullmatch(r"(R\d+)\s*[-~]\s*(R\d+)", text)
    if not match:
        return [text] if text else []
    start, end = int(match.group(1)[1:]), int(match.group(2)[1:])
    if start > end:
        start, end = end, start
    return [f"R{index:03d}" for index in range(start, end + 1)]


def _load_official_campaign(table_path: Path | None) -> dict | None:
    """Load official-notice mappings; return None if missing or invalid and omit that report section."""
    if table_path is None or not table_path.is_file():
        return None
    try:
        payload = json.loads(table_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    issue_to_rules: dict[str, list[str]] = {}
    for issue, tokens in (payload.get("issue_to_rule_mapping") or {}).items():
        rules: list[str] = []
        for token in tokens or []:
            rules.extend(_expand_rule_token(token))
        issue_to_rules[str(issue)] = rules
    official_by_no = {
        int(item["no"]): item
        for item in (payload.get("official_issues") or [])
        if isinstance(item, dict) and item.get("no") is not None
    }
    by_app: dict[str, dict] = {}
    for item in (payload.get("local_apks") or []):
        if not isinstance(item, dict) or not item.get("app"):
            continue
        merged = dict(item)
        try:
            source_entry = official_by_no.get(int(item.get("official_no")))
        except (TypeError, ValueError):
            source_entry = None
        if source_entry:
            merged.setdefault("developer", source_entry.get("developer"))
            merged.setdefault("source", source_entry.get("source"))
        by_app[str(item["app"])] = merged
    return {
        "path": str(table_path),
        "campaign": payload.get("campaign") or {},
        "by_app": by_app,
        "issue_to_rules": issue_to_rules,
        "not_downloaded": [
            item
            for item in (payload.get("not_downloaded_from_official_list") or [])
            if isinstance(item, dict)
        ],
    }


def _official_rules_for(
    official: dict, issue_to_rules: dict, detected_rule_ids: Sequence[str] | None = None,
) -> tuple[list[str], dict[str, list[str]]]:
    """Return all rules mapped to the app's official issues and each issue's rule list."""
    use_problem_rules = any(str(rule).startswith("P") for rule in (detected_rule_ids or []))
    per_issue = {}
    for issue in official.get("official_issues") or []:
        issue = str(issue)
        configured = list(issue_to_rules.get(issue, []))
        per_issue[issue] = (
            official_problem_rule_ids(issue) or [rule for rule in configured if str(rule).startswith("P")]
            if use_problem_rules else configured
        )
    rules: list[str] = []
    for issue_rules in per_issue.values():
        for rule in issue_rules:
            if rule not in rules:
                rules.append(rule)
    return rules, per_issue


def _candidate_issues(per_issue: dict[str, list[str]], verdicts: dict) -> list[str]:
    """Select related issues with violation evidence; the rule version determines matching scope."""
    return [
        issue
        for issue, rules in per_issue.items()
        if any(verdicts.get(rule) == "violation" for rule in rules)
    ]


def _verdict_distribution(rules: Sequence[str], verdicts: dict) -> str:
    """Summarize verdict counts for a rule group in a compact distribution string."""
    if not rules:
        return "—"
    counter: dict[str, int] = {}
    for rule in rules:
        label = VERDICT_LABELS.get(verdicts.get(rule), "Not covered")
        counter[label] = counter.get(label, 0) + 1
    order = ["Violation", "Compliant", "Insufficient evidence", "Not applicable", "Not covered"]
    return "｜".join(f"{label} {counter[label]}" for label in order if counter.get(label))


def _rule_list_text(rules: Sequence[str], limit: int = 6) -> str:
    """List only the first few rule IDs when a large set would overwhelm a table row."""
    if not rules:
        return "—"
    if len(rules) <= limit:
        return "、".join(rules)
    return f"{len(rules)} rules ({', '.join(rules[:limit])}, etc.)"


def _conclusion_text(view: dict) -> str:
    """Summarize official-issue candidate matches and additional findings for one app."""
    official = view["official"]
    violations = view["violations"]
    if not official:
        return f"Outside this official-notice batch; {len(violations)} violation findings"
    total = len(view["per_issue"])
    candidate_issues = len(view["candidate_issues"])
    extra = len(view["extra"])
    if any(str(rule).startswith("P") for rules in view["per_issue"].values() for rule in rules):
        version = "Earlier version" if official.get("version_relation") == "same" else "Reference version"
        return f"{version}: {candidate_issues}/{total} related official issues detected (candidate matches pending manual review); {extra} additional findings"
    if official.get("version_relation") == "same":
        head = (
            f"Same-version official-issue candidate matches: {candidate_issues}/{total}"
            if candidate_issues
            else "No same-version mapped rule matched; manual review required"
        )
    else:
        head = (
            f"Different-version official-issue candidate matches: {candidate_issues}/{total}"
            if candidate_issues
            else "No different-version mapped rule matched; remediation cannot be inferred"
        )
    if extra:
        return f"{head}; {extra} additional findings"
    return head


def _load_compliance_summary(app_output_dir: Path) -> dict:
    """Extract one app's compliance summary and violating rule details."""
    report_path = app_output_dir / "compliance_report.json"
    payload: dict = {"available": False}
    if not report_path.is_file():
        return payload
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        payload["error"] = str(exc)
        return payload
    summary = report.get("summary", {}) or {}
    payload.update(
        {
            "available": True,
            "gui_total": int(summary.get("gui_total", 0) or 0),
            "processed": int(summary.get("processed", summary.get("executed", 0)) or 0),
            "resolved": int(summary.get("resolved", 0) or 0),
            "compliant": int(summary.get("compliant", 0) or 0),
            "violation": int(summary.get("violation", 0) or 0),
            "undetermined": int(summary.get("undetermined", 0) or 0),
            "not_applicable": int(summary.get("not_applicable", 0) or 0),
            "verdicts": {
                str(item["rule_id"]): str(item.get("verdict", ""))
                for item in report.get("checks", [])
                if isinstance(item, dict) and item.get("rule_id")
            },
            "violations": [
                {
                    "rule_id": str(item.get("rule_id", "")),
                    "clause": _clip(item.get("clause", ""), CLAUSE_DISPLAY_LIMIT),
                    "description": _clip(item.get("description", ""), DESCRIPTION_DISPLAY_LIMIT),
                    "steps": _evidence_steps(item.get("evidence")),
                }
                for item in report.get("checks", [])
                if isinstance(item, dict) and item.get("verdict") == "violation"
            ],
        }
    )
    return payload


def build_batch_report(
    batch_root: Path,
    entries: Sequence[AppEntry],
    official_table: Path | None = None,
) -> Path:
    """Aggregate app verdicts into batch_report.md and batch_report.json.

    If official_table is omitted, look for the legacy mapping filename above the batch directory.
    If unavailable, report detection results without an official-notice comparison section.
    """
    if official_table is None:
        candidate = batch_root.parent / OFFICIAL_TABLE_FILENAME
        official_table = candidate if candidate.is_file() else None
    campaign = _load_official_campaign(official_table)
    by_app = campaign["by_app"] if campaign else {}
    issue_to_rules = campaign["issue_to_rules"] if campaign else {}
    rows: list[dict] = []
    for entry in entries:
        app_output_dir = batch_root / entry.dir_name
        run_status = run_result_status(app_output_dir, entry.package) or "not_run"
        run_result: dict = {}
        result_path = app_output_dir / "run_result.json"
        if result_path.is_file():
            try:
                run_result = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                run_result = {}
        identity_matches = isinstance(run_result, dict) and run_result.get("package_name") == entry.package
        if not identity_matches:
            run_result = {}
        rows.append(
            {
                "app_name": entry.name,
                "package_name": entry.package,
                "output_dir": entry.dir_name,
                "run_status": run_status,
                "elapsed_minutes": round(
                    float(run_result.get("elapsed_seconds", 0) or 0) / 60, 1
                ),
                "iterations": int(run_result.get("iteration_count", 0) or 0),
                "compliance": _load_compliance_summary(app_output_dir) if identity_matches else {"available": False},
                "official": by_app.get(entry.name),
            }
        )

    app_views: list[dict] = []
    for row in rows:
        comp = row["compliance"]
        available = bool(comp.get("available"))
        violations = comp.get("violations", []) if available else []
        verdicts = comp.get("verdicts", {}) if available else {}
        official = row.get("official")
        if official:
            official_rules, per_issue = _official_rules_for(official, issue_to_rules, list(comp.get("verdicts", {})))
        else:
            official_rules, per_issue = [], {}
        related_scope = any(str(rule).startswith("P") for rule in verdicts)
        candidates = _candidate_issues(per_issue, verdicts)
        row["official_comparison"] = {
            "scope": "related_problem_match" if related_scope else "automatic_rule_match_only",
            "issues": [
                {"issue": issue, "rule_ids": rules,
                 "status": (
                     "unmapped" if not rules else
                     ("detected_related" if related_scope else "pending_review") if issue in candidates else "not_detected"
                 )}
                for issue, rules in per_issue.items()
            ],
        }
        app_views.append(
            {
                "name": row["app_name"],
                "package": row["package_name"],
                "row": row,
                "comp": comp,
                "available": available,
                "violations": violations,
                "verdicts": verdicts,
                "official": official,
                "official_rules": official_rules,
                "per_issue": per_issue,
                "candidate_issues": candidates,
                "extra": [item for item in violations if item["rule_id"] not in official_rules],
            }
        )

    report_path = batch_root / "batch_report.json"
    report_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Aggregate only successful runs. Failed, timed-out, and unfinished runs may
    # contain partial findings; counting those would contradict the declared sample
    # population and silently distort violation and official-candidate totals.
    # Their individual statuses remain visible in the report.
    successful_views = [view for view in app_views if view["row"]["run_status"] == "success"]
    excluded_run_count = len(app_views) - len(successful_views)
    partial_coverage_views = [
        view for view in successful_views
        if view["available"] and view["comp"].get("processed", 0) < view["comp"].get("gui_total", 0)
    ]

    total_violations = sum(len(view["violations"]) for view in successful_views)
    violation_apps = sum(1 for view in successful_views if view["violations"])
    all_rules = {item["rule_id"] for view in successful_views for item in view["violations"]}
    extra_items = [item for view in successful_views for item in view["extra"]]
    extra_rules = {item["rule_id"] for item in extra_items}
    total_issues = sum(len(view["per_issue"]) for view in successful_views)
    total_candidates = sum(len(view["candidate_issues"]) for view in successful_views)
    candidate_apps = sum(1 for view in successful_views if view["candidate_issues"])
    same_views = [
        view for view in successful_views
        if (view["official"] or {}).get("version_relation") == "same"
    ]
    newer_views = [
        view for view in successful_views
        if view["official"] and view["official"].get("version_relation") != "same"
    ]
    newer_with_violation = sum(1 for view in newer_views if view["violations"])
    gui_total = max((view["comp"].get("gui_total", 0) or 0) for view in successful_views) if successful_views else 0
    not_downloaded = (campaign.get("not_downloaded") if campaign else []) or []

    related_report = bool(app_views) and all(
        view["row"]["official_comparison"]["scope"] == "related_problem_match"
        for view in app_views
    )
    match_label = "Related official issues detected (manual review pending)" if related_report else "Official-issue candidate matches"
    match_note = "same-type or related-issue scope" if related_report else "manual review pending"
    sections: list[tuple[str, list[str]]] = []

    overview: list[str] = ["| Item | Details |", "| --- | --- |"]
    overview.append(f"| Generated | {datetime.now().isoformat(timespec='seconds')} |")
    overview.append(
        f"| Sample | {len(app_views)} apps "
        f"({len(successful_views)} successful runs) |"
    )
    if excluded_run_count:
        overview.append(
            f"| Aggregation scope | Totals include only the {len(successful_views)} successful app runs; "
            f"{excluded_run_count} failed, timed-out, or unfinished runs "
            "are excluded from violation and official-issue candidate totals |"
        )
    if partial_coverage_views:
        overview.append(
            f"| Coverage note | {len(partial_coverage_views)} successful runs did not process every rule "
            "(the manifest verifies artifact completeness, not rule coverage; these runs remain in the totals): " +
            "、".join(
                f"{view['name']}({view['comp'].get('processed', 0)}/{view['comp'].get('gui_total', 0)})"
                for view in partial_coverage_views
            ) + " |"
        )
    if gui_total:
        overview.append(f"| Rules | {gui_total} GUI-checkable rules |")
    if campaign:
        campaign_info = campaign["campaign"]
        overview.append(
            f"| Official notice | {campaign_info.get('title', 'Official notice list')} "
            f"({campaign_info.get('batch', '')}, published {campaign_info.get('published', '')}, "
            f"{campaign_info.get('total_entries', len(by_app))} entries) |"
        )
        if not_downloaded:
            overview.append(
                f"| Comparison sample | APKs acquired for {len(by_app)} notice entries; "
                f"{len(not_downloaded)} entries excluded |"
            )
        overview.append(f"| Apps with violation findings | {violation_apps} / {len(app_views)} |")
        overview.append(
            f"| Total violation findings | {total_violations} findings across {len(all_rules)} distinct rules |"
        )
        overview.append(
            f"| {match_label} | {total_candidates} / {total_issues} issues "
            f"({candidate_apps} apps; {match_note}) |"
        )
        overview.append(
            f"| Additional findings absent from official notices | {len(extra_items)} findings across {len(extra_rules)} distinct rules |"
        )
        overview.append(
            f"| Different-version apps with violation candidates | {newer_with_violation} / {len(newer_views)} "
            f"(same-version comparisons: {len(same_views)}) |"
        )
    overview.append("| Method | Dynamic GUI screening with actual interactions and before-action-result evidence |")
    overview.append("| Evidence scope | Violations require direct interface or interaction evidence; network requests and background behavior are outside scope |")
    sections.append(("Detection overview", overview))

    data_rows: list[str] = [
        "| App | Package | Processed/total | Compliant | Violation | Insufficient evidence | Not applicable | Minutes |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for view in sorted(app_views, key=lambda item: -len(item["violations"])):
        comp = view["comp"]
        elapsed = view["row"].get("elapsed_minutes", 0)
        if view["available"]:
            data_rows.append(
                f"| {view['name']} | {view['package']} "
                f"| {comp['processed']}/{comp['gui_total']} | {comp['compliant']} | {comp['violation']} "
                f"| {comp['undetermined']} | {comp['not_applicable']} | {elapsed} |"
            )
        else:
            data_rows.append(
                f"| {view['name']} | {view['package']} | - | - | - | - | - | {elapsed} |"
            )
    sections.append(("App results (descending violation count)", data_rows))

    if campaign:
        conclusion_rows: list[str] = [
            "| App | Version | Official entry | Conclusion |",
            "| --- | --- | ---: | --- |",
        ]
        for view in app_views:
            official = view["official"]
            if not official:
                conclusion_rows.append(
                    f"| {view['name']} | - | - | Outside this official-notice batch; "
                    f"{len(view['violations'])} violation findings |"
                )
                continue
            relation = "Same version" if official.get("version_relation") == "same" else "Different or unknown version"
            conclusion_rows.append(
                f"| {view['name']} | {official.get('local_version', '—')}（{relation}）"
                f" | Entry {official.get('official_no', '-')} | {_conclusion_text(view)} |"
            )
        sections.append(("Per-app conclusions", conclusion_rows))

        issue_rows: list[str] = [
            "| App | Version relation | Official issue | Current verdict distribution | Assessment |",
            "| --- | --- | --- | --- | --- |",
        ]
        for view in app_views:
            official = view["official"]
            if not official:
                continue
            relation = "Same version" if official.get("version_relation") == "same" else "Different or unknown version"
            for issue, rules in view["per_issue"].items():
                hits = [rule for rule in rules if view["verdicts"].get(rule) == "violation"]
                related_scope = view["row"]["official_comparison"]["scope"] == "related_problem_match"
                hit_label = "Related finding; manual review pending" if related_scope else "Candidate match; manual review pending"
                judgement = f"{hit_label} ({', '.join(hits)})" if hits else "No related finding" if related_scope else "No mapped rule matched"
                issue_rows.append(
                    f"| {view['name']} | {relation} | {issue} "
                    f"| {_verdict_distribution(rules, view['verdicts'])} | {judgement} |"
                )
        sections.append(("Official-notice issue comparison", issue_rows))

        if issue_to_rules:
            mapping_rows: list[str] = ["| Official issue | Mapped rules |", "| --- | --- |"]
            displayed_rules = dict(issue_to_rules)
            for view in app_views:
                if view["row"]["official_comparison"]["scope"] == "related_problem_match":
                    displayed_rules.update(view["per_issue"])
            for issue, rules in displayed_rules.items():
                mapping_rows.append(f"| {issue} | {_rule_list_text(rules)} |")
            sections.append(("Official-issue to rule mapping", mapping_rows))

    if campaign:
        extra_rows: list[str] = [
            "| App | Rule ID | Clause | Evidence STEP |",
            "| --- | --- | --- | --- |",
        ]
        for view in app_views:
            for item in view["extra"]:
                clause = item["clause"].replace("|", "\\|")
                steps = ", ".join(str(step) for step in item.get("steps", [])) or "-"
                extra_rows.append(
                    f"| {view['name']} | {item['rule_id']} | {clause} | {steps} |"
                )
        if not extra_items:
            extra_rows.append("| - | - | No findings beyond the official notices in this run | - |")
        sections.append(("Additional findings absent from official notices", extra_rows))

    detail: list[str] = []
    for view in app_views:
        if not view["violations"]:
            continue
        detail.append(f"### {view['name']} ({view['package']}, {len(view['violations'])} findings)")
        detail.append("")
        detail.append("| Rule ID | Source | Clause | Evidence STEP | Explanation |")
        detail.append("| --- | --- | --- | --- | --- |")
        for item in view["violations"]:
            clause = item["clause"].replace("|", "\\|")
            desc = item["description"].replace("|", "\\|")
            if not view["official"]:
                source = "—"
            else:
                source = "Officially listed" if item["rule_id"] in view["official_rules"] else "Additional finding"
            steps = ", ".join(str(step) for step in item.get("steps", [])) or "-"
            detail.append(
                f"| {item['rule_id']} | {source} | {clause} | {steps} | {desc} |"
            )
        detail.append("")
    if not detail:
        detail.append("(No violating rules)")
    sections.append(("Full violation details", detail))

    if not_downloaded:
        pending_rows: list[str] = ["| Official entry | App |", "| ---: | --- |"]
        for item in not_downloaded:
            pending_rows.append(f"| Entry {item.get('official_no', '-')} | {item.get('app', '-')} |")
        sections.append(("Official entries excluded from comparison", pending_rows))

    sections.append(
        (
            "Interpretation notes",
            [
                "- This report describes dynamic GUI screening, not a full legal audit of network requests, background code, or server processing.",
                "- Officially listed means the rule maps to an issue type in the app's official notice; "
                "additional finding means this run identified an issue not covered by that notice.",
                "- Official issues map to multiple rules (see the mapping section). "
                "A candidate match means at least one mapped rule was marked as a violation. Confirm behavior, version, and evidence manually before claiming reproduction; do not calculate recall from these matches.",
                "- No mapped rule matched means the explored GUI paths supplied no violation evidence. "
                "It does not invalidate an official conclusion or independently prove remediation.",
                "- Evidence STEP IDs refer to screenshots under each app output's compliance_traces/ directory for individual review.",
            ],
        )
    )

    lines: list[str] = ["# Batch compliance screening report", ""]
    if campaign:
        lines.append(
            f"> Tested {len(app_views)} apps; {violation_apps} had violation findings "
            f"({total_violations} total). Official-issue candidate matches: "
            f"{total_candidates}/{total_issues}; {len(extra_items)} additional findings were absent from the official notices."
        )
    else:
        lines.append(
            f"> Tested {len(app_views)} apps; {violation_apps} had violation findings "
            f"({total_violations} total across {len(all_rules)} distinct rules)."
        )
    lines.append("")
    for index, (title, body) in enumerate(sections, start=1):
        lines.append(f"## {index}、{title}")
        lines.append("")
        lines.extend(body)
        lines.append("")


    markdown_path = batch_root / "batch_report.md"
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[Batch queue] Summary report generated: {markdown_path}")
    return markdown_path


def _parse_app_spec(specs: Sequence[str]) -> tuple[tuple[str, str], ...]:
    """Parse an --app override in name:package format."""
    apps: list[tuple[str, str]] = []
    for spec in specs:
        if ":" not in spec:
            raise ValueError(f"--app requires name:package; received: {spec}")
        name, package = spec.split(":", 1)
        apps.append((name.strip(), package.strip()))
    return tuple(apps)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sequential compliance screening for installed apps")
    parser.add_argument("--device-id", required=True)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT,
                        help=f"Batch output root (default: {DEFAULT_OUTPUT_ROOT})")
    parser.add_argument("--total-time", type=_nonnegative_float, default=DEFAULT_TOTAL_TIME_MINUTES,
                        help=f"Time budget per app in minutes (default: {DEFAULT_TOTAL_TIME_MINUTES:g}; 0 is unlimited)")
    parser.add_argument("--cooldown", type=_nonnegative_float, default=DEFAULT_COOLDOWN_SECONDS,
                        help=f"Cooldown between apps in seconds (default: {DEFAULT_COOLDOWN_SECONDS:g})")
    parser.add_argument("--app", action="append", required=True,
                        help="Explicit test queue entry in name:package format; repeat for multiple apps")
    parser.add_argument("--only", default=None,
                        help="Run only these app names or packages, separated by commas")
    parser.add_argument("--limit", type=_nonnegative_int, default=0,
                        help="Run only the first N queue entries in directory order; 0 means all")
    parser.add_argument("--rerun-failed", action="store_true",
                        help="Rerun failed apps, preserving their previous output directories")
    parser.add_argument("--force", action="store_true",
                        help="Ignore completion records and rerun all apps")
    parser.add_argument("--report-only", action="store_true",
                        help="Regenerate the summary report without running tests")
    parser.add_argument("--official-table", default=None,
                        help="Official-notice mapping JSON; by default, use the legacy mapping filename above the output directory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    batch_root = Path(args.output_root).expanduser().resolve()
    batch_root.mkdir(parents=True, exist_ok=True)

    apps = _parse_app_spec(args.app)
    entries = [
        AppEntry(index=i, name=name, package=package)
        for i, (name, package) in enumerate(apps, start=1)
    ]

    if args.only:
        wanted = {item.strip() for item in args.only.split(",") if item.strip()}
        entries = [e for e in entries if e.name in wanted or e.package in wanted]
        if not entries:
            print(f"[Batch queue] --only matched no apps: {args.only}")
            return 2
    if args.limit > 0:
        entries = entries[: args.limit]

    if args.report_only:
        official_table = Path(args.official_table).expanduser() if args.official_table else None
        build_batch_report(batch_root, entries, official_table=official_table)
        return 0

    if not wait_for_device(args.device_id):
        print("[Batch queue] Device wait timed out; exiting")
        return 2

    missing = [e for e in entries if not app_installed(args.device_id, e.package)]
    if missing:
        for entry in missing:
            print(f"[Batch queue] Skipping an app that is not installed: {entry.name} ({entry.package})")
        entries = [e for e in entries if e not in missing]

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    exit_code = 0
    # Only an executed app needs force-stop and cooldown before the next app; skipped entries do not.
    last_run_entry: AppEntry | None = None
    try:
        for entry in entries:
            app_output_dir = batch_root / entry.dir_name
            status = run_result_status(app_output_dir, entry.package)
            if run_artifacts_complete(app_output_dir, entry.package) and not args.force:
                print(f"[Batch queue] ({entry.index:02d}) Run and artifacts complete; skipping: {entry.name}")
                continue
            if status and not (args.rerun_failed or args.force):
                print(
                    f"[Batch queue] ({entry.index:02d}) Previous status: {status}; "
                    f"skipping (use --rerun-failed to retry): {entry.name}"
                )
                continue
            if app_output_dir.exists() and any(app_output_dir.iterdir()):
                archive = app_output_dir.with_name(app_output_dir.name + ".previous-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
                app_output_dir.rename(archive)
                print(f"[Batch queue] ({entry.index:02d}) Previous artifacts preserved: {archive}")
                app_output_dir.mkdir(parents=True, exist_ok=True)

            if last_run_entry is not None:
                force_stop_app(args.device_id, last_run_entry.package)
                if args.cooldown > 0:
                    print(f"[Batch queue] Cooling down for {args.cooldown:g}s before continuing...")
                    time.sleep(args.cooldown)
                if not wait_for_device(args.device_id):
                    print("[Batch queue] Device disconnected after cooldown and reconnection failed; stopping")
                    exit_code = 2
                    break

            last_run_entry = entry
            record = _run_one(
                entry,
                device_id=args.device_id,
                total_time_minutes=args.total_time,
                batch_root=batch_root,
                project_root=project_root,
            )
            _append_batch_summary(batch_root, record)
            if not record.get("ok"):
                exit_code = 1
            if record.get("interrupted"):
                print("[Batch queue] Interrupt received; stopping safely")
                exit_code = 130
                break
    except BatchInterrupted as exc:
        print(f"[Batch queue] Queue interrupted: {exc}")
        exit_code = 130
    finally:
        # Generate a progress summary after success, failure, or interruption.
        try:
            official_table = Path(args.official_table).expanduser() if args.official_table else None
            build_batch_report(batch_root, entries, official_table=official_table)
        except Exception as exc:  # noqa: BLE001
            print(f"[Batch queue] Summary report generation failed: {exc}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
