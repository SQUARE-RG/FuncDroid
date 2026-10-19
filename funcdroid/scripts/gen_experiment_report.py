"""Generate the historical 2026-09-18 experiment report (legacy 57-rule library).

This fixed-batch report is not the current rule library's reporting entry point.
For current batches use batch_compliance_run.py and compliance/report.py.

Read the fixed historical compliance and defect batches beneath EXPERIMENT,
plus the official-notice mapping at OFFICIAL, and write one Markdown report.
Legacy filenames and official issue keys are preserved exactly as escaped literals.
"""
from __future__ import annotations

import glob
import json
import os
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Preserve legacy dataset paths and labels; do not rename archived evidence.
EXPERIMENT = ROOT / "experiments" / "\u5408\u89c4\u6027\u5b9e\u9a8c"
COMPLIANCE_GLOB = str(EXPERIMENT / "output" / "[0-9][0-9]_*")
DEFECT_GLOB = str(EXPERIMENT / "output_defect_batch" / "*")
OFFICIAL = EXPERIMENT / "\u5b98\u65b9\u901a\u62a5\u5bf9\u7167\u8868.json"
OUTPUT = EXPERIMENT / "\u7efc\u5408\u68c0\u6d4b\u62a5\u544a_20260918.md"


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _compliance_dirs():
    """List app directories, excluding archived NN_xxx.previous-<timestamp> attempts."""
    return [
        d for d in sorted(glob.glob(COMPLIANCE_GLOB))
        if ".previous-" not in os.path.basename(d.rstrip("/"))
    ]


def load_compliance():
    apps = []
    for d in _compliance_dirs():
        report = load_json(os.path.join(d, "compliance_report.json"))
        checks = report.get("checks", [])
        if len(checks) != 57 or any(not str(c.get("rule_id", "")).startswith("R") for c in checks):
            raise ValueError(f"This historical report requires the 2026-09-18 dataset with 57 R-rules; use the current batch reporter instead: {d}")
        run = load_json(os.path.join(d, "run_result.json"))
        apps.append(
            {
                "dir": os.path.basename(d),
                "app": report["app_name"],
                "pkg": report["package_name"],
                "summary": report["summary"],
                "checks": report["checks"],
                "status": run.get("status"),
                "iterations": run.get("iteration_count"),
                "elapsed_min": round((run.get("elapsed_seconds") or 0) / 60, 1),
                "llm_tokens": (run.get("llm_usage") or {}).get("total_tokens"),
            }
        )
    return apps


def load_defects():
    apps = []
    for d in sorted(glob.glob(DEFECT_GLOB)):
        if not os.path.isdir(d):
            continue
        bugs_path = os.path.join(d, "bugs.json")
        if not os.path.exists(bugs_path):
            continue
        bugs = load_json(bugs_path)
        run_path = os.path.join(d, "run_result.json")
        run = load_json(run_path) if os.path.exists(run_path) else {}
        apps.append(
            {
                "dir": os.path.basename(d.rstrip("/")),
                "app": run.get("app_name"),
                "pkg": run.get("package_name"),
                "status": run.get("status"),
                "iterations": run.get("iteration_count"),
                "elapsed_min": round((run.get("elapsed_seconds") or 0) / 60, 1),
                "llm_tokens": (run.get("llm_usage") or {}).get("total_tokens"),
                "bugs": bugs,
            }
        )
    return apps


# Exact original notice labels are lookup keys, not translated report prose.
OFFICIAL_RULE_MAP = {
    "\u4fe1\u606f\u7a97\u53e3\u65e0\u6cd5\u5173\u95ed": ["R062", "R070"],
    "\u4fe1\u606f\u7a97\u53e3\u70b9\u51fb\u4e71\u8df3\u8f6c": ["R062", "R070"],
    "\u5f3a\u5236\u7528\u6237\u4f7f\u7528\u5b9a\u5411\u63a8\u9001\u529f\u80fd": ["R054", "R019"],
    "\u8fdd\u89c4\u6536\u96c6\u4e2a\u4eba\u4fe1\u606f": [
        "R013", "R029", "R033", "R034", "R035",
        *["R%03d" % i for i in range(40, 51)],
        "R083", "R091", "R094",
    ],
    "\u8fdd\u89c4\u4f7f\u7528\u4e2a\u4eba\u4fe1\u606f": ["R035", "R036"],
    "APP\u5f3a\u5236、\u9891\u7e41、\u8fc7\u5ea6\u7d22\u53d6\u6743\u9650": ["R030", "R036", "R039", "R055", "R076"],
    "APP\u9891\u7e41\u81ea\u542f\u52a8\u548c\u5173\u8054\u542f\u52a8": ["R081"],
    "\u8fdd\u89c4\u63d0\u4f9b\u81ea\u52a8\u7eed\u8d39\u670d\u52a1": ["R071", "R073"],
    "\u6b3a\u9a97\u8bef\u5bfc\u7528\u6237\u4e0b\u8f7d APP": ["R035", "R058"],
    "\u8d85\u8303\u56f4\u6536\u96c6\u4e2a\u4eba\u4fe1\u606f": ["R013", "R029", "R094"],
}


def build_official_records(compliance_apps, official_doc):
    from scripts.batch_compliance_run import _expand_rule_token, _official_rules_for

    by_pkg = {a["pkg"]: a for a in compliance_apps}
    mapping = official_doc.get("issue_to_rule_mapping", OFFICIAL_RULE_MAP)
    records = []
    for item in official_doc["local_apks"]:
        app = by_pkg.get(item["package"])
        if not app:
            continue
        verdicts = {c["rule_id"]: c["verdict"] for c in app["checks"]}
        expanded = {issue: [rule for token in tokens for rule in _expand_rule_token(token)]
                    for issue, tokens in mapping.items()}
        _, per_issue = _official_rules_for(item, expanded, list(verdicts))
        related_scope = any(str(rule).startswith("P") for rule in verdicts)
        for issue, rules in per_issue.items():
            dist = Counter()
            for rule in rules:
                dist[verdicts.get(rule, "missing")] += 1
            records.append(
                {
                    "app": item["app"],
                    "relation": item["version_relation"],
                    "local_version": item["local_version"],
                    "official_version": item["official_version"],
                    "issue": issue,
                    "candidate_match": dist.get("violation", 0) > 0,
                    "comparison_status": (
                        "unmapped" if not rules else
                        ("detected_related" if related_scope else "pending_review") if dist.get("violation", 0) > 0 else "not_detected"
                    ),
                    "dist": dict(dist),
                }
            )
    return records


def rule_clause_map(compliance_apps):
    clauses = {}
    for a in compliance_apps:
        for c in a["checks"]:
            clauses[c["rule_id"]] = c["clause"]
    return clauses


def fmt_dist(dist):
    order = ["violation", "compliant", "undetermined", "not_applicable", "missing"]
    labels = {"violation": "Violation", "compliant": "Compliant", "undetermined": "Insufficient evidence",
              "not_applicable": "Not applicable", "missing": "Not reviewed"}
    return " | ".join("%s %s" % (labels[k], dist[k]) for k in order if dist.get(k))


def main():
    compliance = load_compliance()
    defects = load_defects()
    official_doc = load_json(OFFICIAL)
    # Normalize defect app labels to the compliance/report names (some APKs carry
    # a different device label, e.g. com.privacy.album -> "Hidden Photo Vault Calculator").
    name_by_pkg = {a["pkg"]: a["app"] for a in compliance}
    for d in defects:
        d["app"] = name_by_pkg.get(d["pkg"], d["app"])
    official_records = build_official_records(compliance, official_doc)
    clauses = rule_clause_map(compliance)

    total_viol = sum(a["summary"]["violation"] for a in compliance)
    viol_counter = Counter()
    viol_apps = defaultdict(list)
    for a in compliance:
        for c in a["checks"]:
            if c["verdict"] == "violation":
                viol_counter[c["rule_id"]] += 1
                viol_apps[c["rule_id"]].append(a["app"])

    total_bugs = sum(len(d["bugs"]) for d in defects)
    bug_apps = sum(1 for d in defects if d["bugs"])
    repro = sum(1 for r in official_records if r["candidate_match"])
    total_official_issue = len(official_records)

    lines = []
    add = lines.append

    add("# Compliance and functional defect report for 12 apps")
    add("")
    add("> Generated: %s  " % datetime.now().strftime("%Y-%m-%d %H:%M"))
    add("> Sample: 12 acquired APKs from the Ministry of Industry and Information Technology's notice on apps/SDKs infringing user rights "
        "(2026 batch 5, overall batch 58, published 2026-08-25).  ")
    add("> Tool: FuncDroid (dynamic GUI exploration and multimodal LLM review).  ")
    add("> Evidence scope: compliance conclusions use interface/interaction evidence; network requests and background behavior are excluded. "
        "Defect findings retain only high-confidence cases with screenshot evidence chains.")
    add("")
    add("---")
    add("")

    # ---------------- 1. Overview ----------------
    add("## 1. Experiment overview")
    add("")
    add("| Item | Compliance screening | Functional defect detection |")
    add("| --- | ---: | ---: |")
    add("| Sample size | %d | %d |" % (len(compliance), len(defects)))
    add("| Successful runs | %d / %d | %d / %d |"
        % (sum(1 for a in compliance if a["status"] == "success"), len(compliance),
           sum(1 for d in defects if d["status"] == "success"), len(defects)))
    add("| Rules | 57 GUI-checkable rules | Open-ended business-logic exploration |")
    add("| Per-app budget | 45 minutes | 60 minutes |")
    add("| Findings | **%d violations** across %d distinct rules | **%d high-confidence defects** across %d apps |"
        % (total_viol, len(viol_counter), total_bugs, bug_apps))
    add("| Official-notice candidate matches | **%d / %d** official issue instances | Not applicable: different evaluation scope |"
        % (repro, total_official_issue))
    add("")
    add("**Key findings**")
    add("")
    add("1. All 12 apps had at least one type of compliance finding, totaling %d violations. " % total_viol)
    add("   The most frequent categories were bundled mandatory consent (R028, 8 occurrences),")
    add("   inaccessible/unannounced privacy policy (R023 and R033, 7 each), and non-voluntary consent (R014, 6).")
    add("2. Among %d officially reported issue instances, **%d were candidate matches pending manual review** "
        "(same-version %d/%d; newer-version %d/%d). "
        % (total_official_issue, repro,
           sum(1 for r in official_records if r["relation"] == "same" and r["candidate_match"]),
           sum(1 for r in official_records if r["relation"] == "same"),
           sum(1 for r in official_records if r["relation"] == "newer" and r["candidate_match"]),
           sum(1 for r in official_records if r["relation"] == "newer")))
    add("   Matches primarily concern improper personal-information collection/use and forced, frequent, or excessive permission requests. "
        "Rule overlap does not prove identity with an official issue or establish remediation.")
    add("3. Among the 12 apps, **%d had %d high-confidence functional defects**. "
        % (bug_apps, total_bugs))
    add("   All were `logic_error` findings with action-result screenshot evidence.")
    add("4. Compliance exposure and functional quality do not fully overlap: the app with the most defects need not have the most violations.")
    add("   The two detection modes provide complementary perspectives.")
    add("")

    # ---------------- 2. Method ----------------
    add("## 2. Detection method")
    add("")
    add("### 2.1 Compliance screening (compliance mode)")
    add("")
    add("- **Scenario source**: evidence-collection scenarios are assembled programmatically from 57 GUI-checkable rules, "
        "rather than generated by an LLM, giving a deterministic rule schedule.")
    add("- **Execution**: an LLM controls the real-device interface step by step, capturing a screenshot before each action; "
        "critical first-launch consent and permission-denial paths exercise the refusal branch.")
    add("- **Review**: a multimodal LLM assesses each rule from before-action-result evidence using "
        "`compliant / violation / undetermined / not_applicable` verdicts.")
    add("- **Reassessment**: rule groups with insufficient evidence receive the preceding evidence gaps and a second evidence-collection attempt.")
    add("")
    add("### 2.2 Functional defect detection (defect mode)")
    add("")
    add("- **Scenario source**: a unified planning LLM generates breadth exploration and depth business scenarios.")
    add("- **Execution**: the same GUI executor explores each scenario and preserves its full screenshot trace.")
    add("- **Review**: `trace_defect_detection` independently submits the complete trace, with a pre-action screenshot per step, "
        "to the review model and retains only high-confidence findings that satisfy evidence-structure requirements.")
    add("- **Evidence gate**: `type` must be `ui_error` or `logic_error`, `confidence=high`, "
        "with action-plus-after evidence or a single contradictory screenshot. Low-confidence findings are discarded.")
    add("- **Reuse in this experiment**: defect runs use the same app's compliance-run Functional Flow Graph (FFG) as a seed "
        "to avoid repeating exploration and focus on business logic.")
    add("")

    # ---------------- 3. Compliance results ----------------
    add("## 3. Compliance screening results")
    add("")
    add("### 3.1 App overview (descending violation count)")
    add("")
    add("| App | Package | Version | Processed/total | Compliant | Violation | Insufficient evidence | Not applicable | Iterations | Minutes |")
    add("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    # version lookup
    version_by_pkg = {x["package"]: x["local_version"] for x in official_doc["local_apks"]}
    for a in sorted(compliance, key=lambda x: -x["summary"]["violation"]):
        s = a["summary"]
        add("| %s | %s | %s | %d/57 | %d | **%d** | %d | %d | %d | %.1f |"
            % (a["app"], a["pkg"], version_by_pkg.get(a["pkg"], "-"),
               s["processed"], s["compliant"], s["violation"], s["undetermined"],
               s["not_applicable"], a["iterations"], a["elapsed_min"]))
    add("")
    add("> Note: 10 of the 57 rules do not apply to non-SDK or other non-target app forms, so not-applicable counts are commonly 10.")
    add("")

    add("### 3.2 Violation frequency")
    add("")
    add("| Rank | Rule | Clause summary | Apps | Affected apps |")
    add("| ---: | --- | --- | ---: | --- |")
    for i, (rule, n) in enumerate(viol_counter.most_common(), 1):
        clause = clauses.get(rule, "")
        clause_short = clause.split("：", 1)[-1]
        add("| %d | %s | %s | %d | %s |"
            % (i, rule, clause_short, n, "、".join(viol_apps[rule])))
    add("")
    add("**Frequent findings**")
    add("")
    add("- **Mandatory bundled consent (R028, 8/12)**: rejection exits the app or repeatedly blocks access, "
        "without a path that consents only to necessary information. This was the most frequent pattern.")
    add("- **Missing first-launch privacy notice (R023/R033, 7/12 each)**: the app opens directly into its main screen, "
        "or the policy entry does not respond, leaving the policy inaccessible or unreadable.")
    add("- **Non-voluntary consent (R014, 6/12)**: agree/disagree buttons exist, but refusal blocks use, "
        "so consent is not informed, voluntary, and explicit.")
    add("- **Missing autostart/identifier controls (R081, 5/12)**: settings commonly lack entries to disable autostart "
        "or reset the device identifier.")
    add("")

    # ---------------- 4. Official-notice comparison ----------------
    add("## 4. Official-issue candidate comparison")
    add("")
    add("Each official issue maps to multiple rules. A candidate match means at least one mapped rule was marked as a violation. "
        "Behavior, version, and evidence require manual review before claiming reproduction; automatic matches are not recall.")
    add("")
    add("| App | Version relation | Local/official version | Official issue | Rule verdict distribution | Conclusion |")
    add("| --- | --- | --- | --- | --- | --- |")
    for r in official_records:
        add("| %s | %s | %s / %s | %s | %s | %s |"
            % (r["app"], r["relation"], r["local_version"], r["official_version"],
               r["issue"], fmt_dist(r["dist"]),
               "**Candidate match; manual review pending**" if r["candidate_match"] else "No mapped rule matched"))
    add("")
    add("### 4.1 Candidate-match distribution")
    add("")
    add("| Version relation | Official issue instances | Candidate matches | Notes |")
    add("| --- | ---: | ---: | --- |")
    same = [r for r in official_records if r["relation"] == "same"]
    newer = [r for r in official_records if r["relation"] == "newer"]
    add("| Same version (manual review pending) | %d | %d | %s |"
        % (len(same), sum(1 for r in same if r["candidate_match"]),
           ", " .join(r["app"] + "(" + r["issue"] + ")" for r in same)))
    add("| Newer version after notice (remediation cannot be inferred) | %d | %d | %s |"
        % (len(newer), sum(1 for r in newer if r["candidate_match"]),
           "Manual review pending: " + ", ".join(
               r["app"] + "(" + r["issue"] + ")" for r in newer if r["candidate_match"])))
    add("")
    add("> Candidate matches indicate rule-level overlap only. Version differences, environmental blockers, and unvisited paths must be recorded separately; they are not direct evidence of missed detection or successful remediation.")
    add("")

    add("### 4.2 Official entries excluded from comparison")
    add("")
    add("The notice contains 26 entries. All 12 acquired APKs were tested; %d entries had no acquired APK "
        "(App Store apps, WeChat mini-programs, SDKs, or unavailable downloads) and were excluded:" % len(official_doc["not_downloaded_from_official_list"]))
    add("")
    for item in official_doc["not_downloaded_from_official_list"]:
        add("- Entry %s: %s" % (item["official_no"], item["app"]))
    add("")

    # ---------------- 5. Defect results ----------------
    add("## 5. Functional defect detection results")
    add("")
    add("### 5.1 Per-app overview")
    add("")
    add("| App | Package | High-confidence defects | Iterations | Minutes | LLM tokens |")
    add("| --- | --- | ---: | ---: | ---: | ---: |")
    for d in sorted(defects, key=lambda x: -len(x["bugs"])):
        add("| %s | %s | **%d** | %d | %.1f | %s |"
            % (d["app"], d["pkg"], len(d["bugs"]), d["iterations"], d["elapsed_min"],
               "{:,}".format(d["llm_tokens"] or 0)))
    add("")
    add("> Four apps (\u56fd\u7434\u7f51, \u9a7e\u7167\u6361\u5206\u5b9d, \u79d8\u5bc6\u7a7a\u95f4, \u6d0b\u8471\u5b66\u56ed) had no high-confidence defect findings in this run. "
        "This means explored paths provided no direct evidence; **it does not mean the apps are defect-free**, "
        "given GUI path-coverage limits and possible false negatives.")
    add("")

    add("### 5.2 Defect details (%d total)" % total_bugs)
    add("")
    for d in sorted(defects, key=lambda x: -len(x["bugs"])):
        if not d["bugs"]:
            continue
        add("#### %s (%s, %d findings)" % (d["app"], d["pkg"], len(d["bugs"])))
        add("")
        for i, bug in enumerate(d["bugs"], 1):
            steps = "、".join(
                "STEP %s(%s)" % (e.get("global_step"), e.get("role"))
                for e in bug.get("evidence", [])
            )
            add("**BUG-%d｜%s**" % (i, bug.get("evidence_category", "")))
            add("")
            add("- **Type**: %s | **Confidence**: %s | **Severity**: %s"
                % (bug.get("type"), bug.get("confidence"), bug.get("severity")))
            add("- **Issue**: %s" % bug.get("description", "").strip())
            add("- **Expected**: %s" % bug.get("expected", "").strip())
            add("- **Actual**: %s" % bug.get("actual", "").strip())
            add("- **Evidence**: %s" % steps)
            add("")
        add("")

    add("### 5.3 Defect distribution")
    add("")
    type_counter = Counter()
    cat_counter = Counter()
    sev_counter = Counter()
    for d in defects:
        for b in d["bugs"]:
            type_counter[b.get("type")] += 1
            cat_counter[b.get("evidence_category")] += 1
            sev_counter[b.get("severity")] += 1
    add("| Dimension | Distribution |")
    add("| --- | --- |")
    add("| Defect type | %s |" % ", ".join("%s %d" % kv for kv in type_counter.items()))
    add("| Evidence category | %s |" % ", ".join("%s %d" % kv for kv in cat_counter.items()))
    add("| Severity | %s |" % ", ".join("%s %d" % kv for kv in sev_counter.items()))
    add("")
    add("> `persisted_result_error` (result not persisted or state not restored) and "
        "`simultaneous_state_contradiction` (contradictory data on one screen) were the two main categories, "
        "both typical business-logic/state-management defects.")
    add("")

    # ---------------- 6. Cross-analysis ----------------
    add("## 6. Compliance and defect cross-analysis")
    add("")
    add("| App | Violation findings | High-confidence defects | Profile |")
    add("| --- | ---: | ---: | --- |")
    comp_by_pkg = {a["pkg"]: a for a in compliance}
    for d in sorted(defects, key=lambda x: -(x["bugs"] and len(x["bugs"]) or 0)):
        a = comp_by_pkg.get(d["pkg"])
        if not a:
            continue
        v = a["summary"]["violation"]
        b = len(d["bugs"])
        if v >= 7 and b >= 2:
            profile = "High compliance and functional risk"
        elif v >= 7:
            profile = "High compliance risk; relatively stable functionality"
        elif b >= 2:
            profile = "Prominent functional defects; relatively limited compliance findings"
        else:
            profile = "Low findings in both dimensions"
        add("| %s | %d | %d | %s |" % (d["app"], v, b, profile))
    add("")
    add("**Observation**: compliance and defect findings are weakly correlated. For example, \u53e3\u888b\u4e94\u7ebf\u8c31 had 17 violation findings "
        "and only 2 defects, while \u8db3\u7403\u6bd4\u5206 had 2 violations and 2 logic defects. "
        "These modes target different risks and are recommended as parallel release checks.")
    add("")

    # ---------------- 7. Conclusions ----------------
    add("## 7. Conclusions and recommendations")
    add("")
    add("### 7.1 Conclusions")
    add("")
    add("1. Both detection modes **completed successfully for all 12 apps**, "
        "with no failures or incomplete artifacts in this fixed batch.")
    add("2. Compliance screening reported %d apps with %d violation candidates; specific issues and remediation still require manual review." % (len(compliance), total_viol))
    add("3. Functional testing found %d apps with %d high-confidence business-logic defects, "
        "mainly involving persistence, on-screen consistency, and resource/filter-state management."
        % (bug_apps, total_bugs))
    add("4. Same-version candidate matches require manual review before effectiveness evaluation. Because GUI path coverage is limited, "
        "non-reproduction does not prove that an official issue is fixed.")
    add("")
    add("### 7.2 Recommendations")
    add("")
    add("- **For developers**: prioritize first-launch privacy notices and usable necessary-information-only consent paths "
        "(R023/R028/R033/R014), plus autostart and device-identifier controls (R081).")
    add("- **For defect fixes**: review persistence and filter/selection-state rollback; "
        "both major defect categories point to state management.")
    add("- **For evaluation**: refine official-issue-to-rule mappings to individual rules "
        "and run targeted depth scenarios for unreproduced official issues to reduce false negatives.")
    add("- **Release checks**: run compliance and functional defect checks side by side and block release when either fails.")
    add("")

    # ---------------- 8. Appendix ----------------
    add("## 8. Appendix")
    add("")
    add("### 8.1 Artifact locations")
    add("")
    add("| Content | Path |")
    add("| --- | --- |")
    add(f"| Compliance artifacts | `{EXPERIMENT.relative_to(ROOT)}/output/01_* ~ 12_*` |")
    add(f"| Compliance batch report | `{EXPERIMENT.relative_to(ROOT)}/output/batch_report.md` |")
    add(f"| Defect artifacts | `{EXPERIMENT.relative_to(ROOT)}/output_defect_batch/<pkg>__<ver>__<sha10>/` |")
    add(f"| Defect batch results | `{EXPERIMENT.relative_to(ROOT)}/output_defect_batch/batch-results.jsonl` |")
    add(f"| Official-notice mapping | `{OFFICIAL.relative_to(ROOT)}` |")
    add(f"| Original official notice | `{EXPERIMENT.relative_to(ROOT)}/\u8fdd\u89c4\u5217\u8868.pdf` |")
    add(f"| Run logs | `{EXPERIMENT.relative_to(ROOT)}/runs/` |")
    add("")
    add("### 8.2 Interpretation scope")
    add("")
    add("- These are dynamic GUI screening results, not a complete legal audit of "
        "network requests, background code, and server processing.")
    add("- Compliant refers only to observed paths; insufficient evidence means no adequate verdict evidence was collected; "
        "not applicable means the rule does not target this app form.")
    add("- High-confidence defects must also satisfy the evidence-structure gate; low-confidence findings are not persisted.")
    add("- App evidence screenshots are in `compliance_traces/` and `bugs/BUG-*/screenshots/`.")
    add("")
    add("---")
    add("")
    add("*Report generated automatically from FuncDroid experiment artifacts.*")
    add("")

    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", OUTPUT, len(lines), "lines")


if __name__ == "__main__":
    main()
