#!/usr/bin/env python3
"""Aggregate functional defect findings, optionally merging multiple batches.

Usage:
  .venv/bin/python scripts/collect_defect_report.py \
      --batch NoticeBatch58:experiments/defect_batch \
      --batch NoticeApps30:experiments/defect_batch_v2_<TS> \
      --output experiments/defect_batch_v2_<TS>

Scan app subdirectories in each batch for run_result.json and bugs.json.
Show unfinished runs with their status; distinguish zero defects from an app not tested.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter
from datetime import datetime

SEVERITY_LABELS = {"high": "High", "medium": "Medium", "low": "Low"}
CATEGORY_LABELS = {
    "persisted_result_error": "Result not persisted or state not restored",
    "simultaneous_state_contradiction": "Contradictory data on one screen",
    "single_action_resource_mutation": "Unexpected resource change from one action",
    "ui_error": "UI error",
    "logic_error": "Business logic error",
}


def _load_json(path: str):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_batch(label: str, root: str) -> dict:
    apps = []
    for path in sorted(glob.glob(os.path.join(root, "*/", "run_result.json"))):
        app_dir = os.path.dirname(path)
        try:
            result = _load_json(path)
        except (json.JSONDecodeError, OSError):
            continue
        bugs = []
        bugs_path = os.path.join(app_dir, "bugs.json")
        if os.path.exists(bugs_path):
            try:
                bugs = _load_json(bugs_path) or []
            except (json.JSONDecodeError, OSError):
                bugs = []
        if not isinstance(bugs, list):
            bugs = []
        high = [b for b in bugs if str(b.get("confidence", "")).lower() == "high"]
        usage = result.get("llm_usage") or {}
        discovery = result.get("activity_discovery") or {}
        activities = discovery.get("activities") or []
        activity_count = discovery.get("count") or len(activities)
        hybrid = activity_count <= 1 or any(
            "dcloud" in str(a).lower() or "pandoraentry" in str(a).lower()
            for a in activities
        )
        apps.append({
            "batch": label,
            "app_name": result.get("app_name") or result.get("package_name"),
            "package_name": result.get("package_name"),
            "status": result.get("status"),
            "elapsed_minutes": round((result.get("elapsed_seconds") or 0) / 60, 1),
            "defects": high,
            "defect_count": len(high),
            "all_bug_count": len(bugs),
            "seed_inherited": int((result.get("seed_provenance") or {}).get("inherited_pages") or 0),
            "activity_count": activity_count,
            "hybrid": hybrid,
            "requests": usage.get("requests"),
            "failures": usage.get("failures"),
            "fatal": (result.get("fatal_error") or {}).get("message") or "",
        })
    return {"label": label, "root": root, "apps": apps}


def md_table(headers: list, rows: list) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


def load_compliance_counts(root: str) -> dict:
    """Read per-app violation/compliant/undetermined counts for cross-analysis."""
    counts: dict[str, dict] = {}
    if not root:
        return counts
    for path in sorted(glob.glob(os.path.join(root, "*/", "compliance_report.json"))):
        try:
            report = _load_json(path)
        except (json.JSONDecodeError, OSError):
            continue
        package = report.get("package_name")
        if not package:
            continue
        summary = report.get("summary", {})
        counts[package] = {
            "app_name": report.get("app_name"),
            "violation": summary.get("violation", 0),
            "compliant": summary.get("compliant", 0),
            "undetermined": summary.get("undetermined", 0),
        }
    return counts


def dedupe_defects(apps: list) -> tuple[list, int]:
    """Deduplicate same-app/category findings across batches using the normalized first 60 description characters."""
    seen = set()
    removed = 0
    for app in apps:
        kept = []
        for defect in app["defects"]:
            key = (app["package_name"], defect.get("evidence_category", ""),
                   (defect.get("description") or "")[:60])
            if key in seen:
                removed += 1
                continue
            seen.add(key)
            kept.append(defect)
        app["defects"] = kept
        app["defect_count"] = len(kept)
    return apps, removed


def dedupe_apps(apps: list) -> tuple[list, int]:
    """Keep one run per app, preferring success, then more findings, then longer elapsed time.

    A resumed app may be interrupted in one batch and successful in the next.
    Prefer the successful result rather than presenting the interrupted attempt as final.
    """
    best: dict[str, dict] = {}
    removed = 0
    for app in apps:
        key = app.get("package_name") or app.get("app_name")
        current = best.get(key)
        if current is None:
            best[key] = app
            continue
        removed += 1

        def rank(item):
            return (1 if item.get("status") == "success" else 0,
                    item.get("defect_count") or 0,
                    item.get("elapsed_minutes") or 0)

        if rank(app) > rank(current):
            best[key] = app
    return list(best.values()), removed


def render(batches: list, compliance: dict | None = None, dedupe: bool = True) -> tuple[str, dict]:
    now = datetime.now().isoformat(timespec="seconds")
    lines = ["# Functional defect detection summary", "", f"> Generated: {now}",
             "> Scope: only `confidence=high` findings; evidence consists of action-result screenshots.", ""]

    all_apps = [a for b in batches for a in b["apps"]]
    removed = 0
    if dedupe:
        all_apps, app_removed = dedupe_apps(all_apps)
        all_apps, defect_removed = dedupe_defects(all_apps)
        removed = app_removed + defect_removed
    total_defects = sum(a["defect_count"] for a in all_apps)

    lines += ["## 1. Batch overview", ""]
    rows = []
    for b in batches:
        # Attribute deduplicated apps/findings to batches consistently with total_defects.
        batch_apps = [a for a in all_apps if a["batch"] == b["label"]]
        n_apps = len(batch_apps)
        n_def = sum(a["defect_count"] for a in batch_apps)
        n_with = sum(1 for a in batch_apps if a["defect_count"] > 0)
        rows.append([b["label"], n_apps, n_def, f"{n_with}/{n_apps}" if n_apps else "-"])
    lines += [md_table(["Batch", "Apps", "High-confidence defects", "Apps with defects"], rows), ""]

    lines += ["## 2. Per-app results", ""]
    rows = []
    for a in sorted(all_apps, key=lambda x: (x["batch"], -x["defect_count"])):
        note = "✅" if a["status"] == "success" else f"⚠️ {a['status']}"
        rows.append([a["batch"], a["app_name"], note,
                     "Hybrid (WebView)" if a.get("hybrid") else "Native",
                     a.get("activity_count", "-"),
                     a["defect_count"], a["seed_inherited"] or "—",
                     a["elapsed_minutes"], f"{a['requests']}/{a['failures']}"])
    lines += [md_table(["Batch", "App", "Status", "App type", "Activity", "Defects",
                        "Seed pages", "Minutes", "Requests/failures"], rows), ""]
    hybrid_apps = [a["app_name"] for a in all_apps if a.get("hybrid")]
    if hybrid_apps:
        lines += ["> **App-type note**: the detector models native pages. Hybrid apps such as uni-app/DCloud "
                  "with one Activity and a WebView can have systematically undercounted pages. Their seed-page/coverage "
                  "figures are not directly comparable with native apps and do not imply shallower testing. Hybrid apps in this batch: "
                  + "、".join(hybrid_apps) + "。", ""]

    # Defect type distribution.
    counter_type, counter_cat, counter_sev = Counter(), Counter(), Counter()
    for a in all_apps:
        for d in a["defects"]:
            counter_type[d.get("type", "?")] += 1
            counter_cat[d.get("evidence_category", "?")] += 1
            counter_sev[d.get("severity", "?")] += 1
    lines += ["## 3. Defect distribution", ""]
    if counter_type:
        lines += [md_table(["Dimension", "Value", "Count"],
                           [["Type", k, v] for k, v in counter_type.most_common()] +
                           [["Evidence category", CATEGORY_LABELS.get(k, k), v] for k, v in counter_cat.most_common()] +
                           [["Severity", SEVERITY_LABELS.get(k, k), v] for k, v in counter_sev.most_common()]), ""]
    else:
        lines += ["(No high-confidence findings in this run)", ""]

    # Defect details.
    lines += ["## 4. Defect details", ""]
    for a in sorted(all_apps, key=lambda x: -x["defect_count"]):
        if not a["defects"]:
            continue
        lines += [f"### {a['app_name']} ({a['package_name']}, {a['defect_count']} findings)", ""]
        for d in a["defects"]:
            lines += [
                f"- **{d.get('bug_id', '?')}｜{d.get('type', '?')} / "
                f"{CATEGORY_LABELS.get(d.get('evidence_category'), d.get('evidence_category', '?'))}"
                f" | Severity: {SEVERITY_LABELS.get(d.get('severity'), d.get('severity'))}**",
                f"  - Issue: {d.get('description', '')}",
                f"  - Expected: {d.get('expected', '')}",
                f"  - Actual: {d.get('actual', '')}",
                "",
            ]

    if removed:
        lines += [f"> Cross-batch deduplication removed {removed} findings with the same app, evidence category, and description.", ""]

    if compliance:
        lines += ["## 5. Compliance and defect cross-analysis", "",
                  "> Compliance data come from the supplied batch directory. Present both risk dimensions per app to distinguish compliance exposure from functional reliability.", ""]
        rows, cross = [], []
        for a in sorted(all_apps, key=lambda x: -x["defect_count"]):
            c = compliance.get(a["package_name"])
            if not c:
                continue
            rows.append([a["app_name"], c["violation"], c["compliant"],
                         c["undetermined"], a["defect_count"]])
            cross.append({"package_name": a["package_name"], "app_name": a["app_name"],
                          "compliance_violation": c["violation"], "defects": a["defect_count"]})
        if rows:
            lines += [md_table(["App", "Violations", "Compliant", "Insufficient evidence", "High-confidence defects"], rows), ""]
        payload_cross = cross
    else:
        payload_cross = []

    payload = {"generated_at": now, "total_defects": total_defects,
               "deduped": removed, "cross_analysis": payload_cross,
               "batches": [{"label": b["label"],
                            "apps": [a for a in all_apps if a["batch"] == b["label"]]}
                           for b in batches]}
    return "\n".join(lines), payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch", action="append", required=True,
                        help="Label:directory; repeat for multiple batches, e.g. Batch58:experiments/defect_batch")
    parser.add_argument("--output", required=True)
    parser.add_argument("--compliance-root", default="",
                        help="Optional compliance batch directory for cross-analysis with defects")
    parser.add_argument("--no-dedupe", action="store_true", help="Disable cross-batch defect deduplication")
    args = parser.parse_args()

    batches = []
    for item in args.batch:
        label, _, root = item.partition(":")
        if not root:
            label, root = os.path.basename(item), item
        batches.append(load_batch(label, root))

    compliance = load_compliance_counts(args.compliance_root)
    markdown, payload = render(batches, compliance, dedupe=not args.no_dedupe)
    os.makedirs(args.output, exist_ok=True)
    md_path = os.path.join(args.output, "defect_report.md")
    json_path = os.path.join(args.output, "defect_report.json")
    with open(md_path, "w", encoding="utf-8") as h:
        h.write(markdown)
    with open(json_path, "w", encoding="utf-8") as h:
        json.dump(payload, h, ensure_ascii=False, indent=2)

    for b in batches:
        print(f"[Batch] {b['label']}: {len(b['apps'])} apps, "
              f"{sum(a['defect_count'] for a in b['apps'])} high-confidence defects")
    print(f"[Done] Report: {md_path}")
    print(f"[Done] Data: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
