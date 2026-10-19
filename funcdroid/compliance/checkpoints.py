"""Compliance checkpoint lifecycle and deterministic scenario generation.

Pending checks become executing, then done for a supported terminal verdict or deferred for insufficient evidence. Retry limits prevent repeated review of identical evidence; new relevant FFG evidence can reopen deferred checks. Reverification of previously resolved checks has a separate bounded counter. Insufficient evidence never overwrites a resolved verdict. Rules in one group share one GUI scenario."""

from __future__ import annotations

import os
from typing import Any

from compliance.rule_library import (
    CHECKPOINT_GROUP_ORDER,
    GROUP_LABELS,
    gui_rules,
    rule_by_id,
)

CHECK_STATUS_PENDING = "pending"
CHECK_STATUS_EXECUTING = "executing"
CHECK_STATUS_DEFERRED = "deferred"
CHECK_STATUS_DONE = "done"

MAX_COMPLIANCE_ATTEMPTS = max(
    1,
    int(os.getenv("COMPLIANCE_MAX_ATTEMPTS", "2") or 2),
)
# Bound rechecks of resolved rules separately from attempts; zero disables reverification.
MAX_COMPLIANCE_REVERIFY = max(
    0,
    int(os.getenv("COMPLIANCE_MAX_REVERIFY", "1") or 0),
)

VERDICT_COMPLIANT = "compliant"
VERDICT_VIOLATION = "violation"
VERDICT_UNDETERMINED = "undetermined"
VERDICT_NOT_APPLICABLE = "not_applicable"

VALID_VERDICTS = {
    VERDICT_COMPLIANT,
    VERDICT_VIOLATION,
    VERDICT_UNDETERMINED,
    VERDICT_NOT_APPLICABLE,
}
TERMINAL_VERDICTS = {VERDICT_COMPLIANT, VERDICT_VIOLATION, VERDICT_NOT_APPLICABLE}

GROUP_EXECUTION_REQUIREMENTS = {
    "first_launch_flow": (
        "[Required branch] Read accessible agreement/policy content, then actually select "
        "Disagree/Deny once and observe the resulting interface and application usability. Seeing the button alone is insufficient. "
        "If denial exits the app or returns to launch, reopen it; after capturing denial evidence, consent may be selected "
        "to observe subsequent first-launch screens."
    ),
    "permission_denial_flow": (
        "[Required branch] Actually select Deny/Do not allow in a genuine system permission dialog. "
        "Retry the functionality and observe restrictions or repeated prompts. Also execute a basic functionality independent of the denied permission and capture its result. Returning home or seeing an entry does not prove usability; seeing the permission dialog alone is insufficient."
    ),
}


def _copy_value(value: Any) -> Any:
    if isinstance(value, list):
        return [dict(item) if isinstance(item, dict) else item for item in value]
    if isinstance(value, dict):
        return dict(value)
    return value


def _copy_finding(finding: dict[str, Any]) -> dict[str, Any]:
    """Deep-copy a finding, including evidence/history, so snapshots share no mutable state."""
    copied = {key: _copy_value(value) for key, value in finding.items() if key != "history"}
    if "history" in finding:
        history = finding.get("history")
        copied["history"] = [
            {key: _copy_value(value) for key, value in entry.items()} if isinstance(entry, dict) else entry
            for entry in (history if isinstance(history, list) else [])
        ]
    return copied


def _int_map(values: dict[str, Any] | None) -> dict[str, int]:
    result: dict[str, int] = {}
    for rule_id, count in (values or {}).items():
        try:
            result[str(rule_id)] = max(0, int(count or 0))
        except (TypeError, ValueError):
            continue
    return result


class ComplianceChecklist:
    """In-memory compliance checklist and lifecycle transitions."""

    def __init__(
        self,
        status: dict[str, str] | None = None,
        findings: list[dict[str, Any]] | None = None,
        attempts: dict[str, int] | None = None,
        evidence_signatures: dict[str, str] | None = None,
        discovery_baselines: dict[str, str] | None = None,
        reverify_counts: dict[str, int] | None = None,
    ):
        self.status: dict[str, str] = dict(status or {})
        self.findings: list[dict[str, Any]] = [
            _copy_finding(finding) for finding in (findings or []) if isinstance(finding, dict)
        ]
        self.attempts: dict[str, int] = _int_map(attempts)
        # Rechecks after new FFG scope use a separate counter.
        self.reverify_counts: dict[str, int] = _int_map(reverify_counts)
        self.evidence_signatures: dict[str, str] = {
            str(rule_id): str(signature or "")
            for rule_id, signature in (evidence_signatures or {}).items()
        }
        # After direct retries, record the post-scenario baseline and reopen only after later exploration changes it.
        self.discovery_baselines: dict[str, str] = {
            str(rule_id): str(signature or "")
            for rule_id, signature in (discovery_baselines or {}).items()
        }

    # ------------------------------------------------------------------
    # State and report serialization.
    # ------------------------------------------------------------------
    def to_state(self) -> dict[str, Any]:
        # Copy nested findings/history because the main thread may serialize while the worker updates.
        return {
            "status": dict(self.status),
            "findings": [_copy_finding(finding) for finding in self.findings],
            "attempts": dict(self.attempts),
            "evidence_signatures": dict(self.evidence_signatures),
            "discovery_baselines": dict(self.discovery_baselines),
            "reverify_counts": dict(self.reverify_counts),
        }

    @classmethod
    def from_state(cls, state: dict[str, Any] | None) -> "ComplianceChecklist":
        state = state if isinstance(state, dict) else {}
        status = state.get("status") if isinstance(state.get("status"), dict) else {}
        findings = state.get("findings") if isinstance(state.get("findings"), list) else []
        attempts = state.get("attempts") if isinstance(state.get("attempts"), dict) else {}
        evidence_signatures = (
            state.get("evidence_signatures")
            if isinstance(state.get("evidence_signatures"), dict)
            else {}
        )
        discovery_baselines = (
            state.get("discovery_baselines")
            if isinstance(state.get("discovery_baselines"), dict)
            else {}
        )
        # Older snapshots default to zero reverifications.
        reverify_counts = (
            state.get("reverify_counts")
            if isinstance(state.get("reverify_counts"), dict)
            else {}
        )
        checklist = cls(
            status=status,
            findings=findings,
            attempts=attempts,
            evidence_signatures=evidence_signatures,
            discovery_baselines=discovery_baselines,
            reverify_counts=reverify_counts,
        )
        # Migrate legacy undetermined/done entries to deferred rather than treating missing evidence as completion.
        for finding in checklist.findings:
            rule_id = str(finding.get("check_id", "") or "").strip()
            if (
                rule_id
                and str(finding.get("verdict", "") or "").strip() == VERDICT_UNDETERMINED
                and checklist.status.get(rule_id) == CHECK_STATUS_DONE
            ):
                checklist.attempts[rule_id] = max(
                    1,
                    int(checklist.attempts.get(rule_id, 0) or 0),
                )
                checklist.status[rule_id] = CHECK_STATUS_DEFERRED
                if checklist.attempts[rule_id] >= MAX_COMPLIANCE_ATTEMPTS:
                    checklist.discovery_baselines.setdefault(rule_id, "")
        return checklist

    # ------------------------------------------------------------------
    # Status transitions.
    # ------------------------------------------------------------------
    @staticmethod
    def _signature_for(
        rule_id: str,
        evidence_signature: str | dict[str, str],
    ) -> str:
        if isinstance(evidence_signature, dict):
            return str(evidence_signature.get(rule_id, "") or "")
        return str(evidence_signature or "")

    def mark_executing(
        self,
        rule_ids: list[str],
        evidence_signature: str | dict[str, str] = "",
    ) -> None:
        for rule_id in rule_ids:
            if str(self.status.get(rule_id, "")).strip() != CHECK_STATUS_DONE:
                self.attempts[rule_id] = int(self.attempts.get(rule_id, 0) or 0) + 1
                self.evidence_signatures[rule_id] = self._signature_for(
                    rule_id,
                    evidence_signature,
                )
                self.discovery_baselines.pop(rule_id, None)
                self.status[rule_id] = CHECK_STATUS_EXECUTING

    def apply_results(self, results: list[dict[str, Any]]) -> None:
        """Merge worker verdicts idempotently by rule_id while retaining historical evidence."""
        for item in results if isinstance(results, list) else []:
            if not isinstance(item, dict):
                continue
            rule_id = str(item.get("check_id", "") or item.get("rule_id", "")).strip()
            verdict = str(item.get("verdict", "")).strip().lower()
            if not rule_id or verdict not in VALID_VERDICTS:
                continue
            attempt = max(1, int(self.attempts.get(rule_id, 0) or 0))
            self.attempts[rule_id] = attempt
            previous = next((finding for finding in self.findings if str(finding.get("check_id", "")) == rule_id), None)
            previous_verdict = str(previous.get("verdict", "") or "") if previous else ""
            if previous_verdict in TERMINAL_VERDICTS and (
                verdict == VERDICT_UNDETERMINED
                or (previous_verdict == VERDICT_VIOLATION and verdict != VERDICT_VIOLATION)
            ):
                # Insufficient recheck evidence cannot overturn a terminal verdict; a passing path cannot erase an observed violation. Append history through a fresh dictionary.
                entry = {
                    key: _copy_value(value)
                    for key, value in item.items()
                    if key != "history"
                }
                entry.update({"check_id": rule_id, "verdict": verdict, "attempt": attempt})
                kept = _copy_finding(previous)
                kept["history"] = list(kept.get("history", [])) + [entry]
                self.findings = [kept if finding is previous else finding for finding in self.findings]
                self.status[rule_id] = CHECK_STATUS_DONE
                self.discovery_baselines.pop(rule_id, None)
                continue
            if verdict == VERDICT_UNDETERMINED:
                self.status[rule_id] = CHECK_STATUS_DEFERRED
                if attempt >= MAX_COMPLIANCE_ATTEMPTS:
                    # An empty signature means the post-scenario baseline has not yet been recorded; recording it does not trigger another review.
                    self.discovery_baselines[rule_id] = ""
            else:
                self.status[rule_id] = CHECK_STATUS_DONE
                self.discovery_baselines.pop(rule_id, None)
            history = list(_copy_finding(previous).get("history", [])) if previous else []
            if previous:
                history.append({key: _copy_value(value) for key, value in previous.items() if key != "history"})
            self.findings = [
                finding
                for finding in self.findings
                if str(finding.get("check_id", "")).strip() != rule_id
            ]
            self.findings.append(
                {
                    "check_id": rule_id,
                    "verdict": verdict,
                    "description": str(item.get("description", "")).strip(),
                    "evidence": _copy_value(item.get("evidence")) if isinstance(item.get("evidence"), list) else [],
                    "attempt": attempt,
                    "history": history,
                    "scope": "observed_paths_only",
                    **({"applicability": dict(item["applicability"])}
                       if isinstance(item.get("applicability"), dict) else {}),
                    **({"trace_path": item["trace_path"]} if item.get("trace_path") else {}),
                    **({"functionality_evidence": dict(item["functionality_evidence"])}
                       if isinstance(item.get("functionality_evidence"), dict) else {}),
                }
            )

    def reopen_deferred(
        self,
        evidence_signature: str | dict[str, str],
        *,
        force: bool = False,
        rules_path: Any = None,
    ) -> list[str]:
        """Return retryable checks to pending when new relevant FFG evidence appears."""
        reopened: list[str] = []
        valid_ids = {rule["rule_id"] for rule in gui_rules(rules_path)}
        for rule_id, status in list(self.status.items()):
            if rule_id not in valid_ids:
                continue
            if status == CHECK_STATUS_DONE:
                finding = next((item for item in self.findings if item.get("check_id") == rule_id), {})
                if finding.get("verdict") not in {VERDICT_COMPLIANT, VERDICT_NOT_APPLICABLE}:
                    continue
                # Bound reverification even when graph fingerprints change on every round.
                if int(self.reverify_counts.get(rule_id, 0) or 0) >= MAX_COMPLIANCE_REVERIFY:
                    continue
                signature = self._signature_for(rule_id, evidence_signature)
                if not signature:
                    continue
                baseline = self.discovery_baselines.get(rule_id)
                if not baseline:
                    self.discovery_baselines[rule_id] = signature
                    continue
                if baseline != signature:
                    self.status[rule_id] = CHECK_STATUS_PENDING
                    self.discovery_baselines.pop(rule_id, None)
                    self.reverify_counts[rule_id] = int(self.reverify_counts.get(rule_id, 0) or 0) + 1
                    reopened.append(rule_id)
                continue
            if status != CHECK_STATUS_DEFERRED:
                continue
            current_signature = self._signature_for(rule_id, evidence_signature)
            attempt = int(self.attempts.get(rule_id, 0) or 0)
            discovery_changed = False
            if attempt >= MAX_COMPLIANCE_ATTEMPTS:
                baseline = self.discovery_baselines.get(rule_id)
                if baseline is None or baseline == "":
                    # First observe the post-scenario graph as a baseline without reopening.
                    self.discovery_baselines[rule_id] = current_signature
                    continue
                if baseline == current_signature:
                    continue
                # Only a subsequent relevant graph change establishes newly discovered evidence.
                self.discovery_baselines.pop(rule_id, None)
                discovery_changed = True
            else:
                previous_signature = str(self.evidence_signatures.get(rule_id, "") or "")
                if not force and previous_signature == current_signature:
                    continue
            if force and attempt >= MAX_COMPLIANCE_ATTEMPTS and not discovery_changed:
                # Forced scheduling cannot bypass the evidence-discovery requirement after retries are exhausted.
                continue
            self.status[rule_id] = CHECK_STATUS_PENDING
            reopened.append(rule_id)
        return reopened

    def recover_interrupted(self) -> list[str]:
        """Return executing checks without surviving worker jobs to pending after restart."""
        recovered: list[str] = []
        for rule_id, status in list(self.status.items()):
            if status != CHECK_STATUS_EXECUTING:
                continue
            self.status[rule_id] = CHECK_STATUS_PENDING
            self.attempts[rule_id] = max(
                0,
                int(self.attempts.get(rule_id, 0) or 0) - 1,
            )
            self.evidence_signatures.pop(rule_id, None)
            self.discovery_baselines.pop(rule_id, None)
            recovered.append(rule_id)
        return recovered

    def release_scheduled(self, rule_ids: list[str]) -> list[str]:
        """Release checks that produced no verdict because execution or queueing failed.

        Return them to pending and refund the attempt charged by mark_executing. Infrastructure failure must not consume evidence-review retry budget."""
        released: list[str] = []
        for rule_id in rule_ids:
            if self.status.get(rule_id) != CHECK_STATUS_EXECUTING:
                continue
            self.status[rule_id] = CHECK_STATUS_PENDING
            self.attempts[rule_id] = max(
                0,
                int(self.attempts.get(rule_id, 0) or 0) - 1,
            )
            self.evidence_signatures.pop(rule_id, None)
            self.discovery_baselines.pop(rule_id, None)
            released.append(rule_id)
        return released

    def deferred_rules(self, rules_path: Any = None) -> list[dict[str, Any]]:
        return [
            rule
            for rule in gui_rules(rules_path)
            if self.status.get(rule["rule_id"]) == CHECK_STATUS_DEFERRED
        ]

    def pending_rules(self, rules_path: Any = None) -> list[dict[str, Any]]:
        from compliance.rule_library import pending_gui_checkpoints

        return pending_gui_checkpoints(self.status, rules_path)

    def next_group_rules(self, rules_path: Any = None) -> list[dict[str, Any]]:
        """Return all pending rules in the next group.

        Prefer fewer attempts, then fixed group priority, so repeatedly inconclusive groups do not starve untested ones."""
        pending = self.pending_rules(rules_path)
        if not pending:
            return []
        grouped: dict[str, list[dict[str, Any]]] = {}
        for rule in pending:
            group = str(rule.get("checkpoint", {}).get("group", "") or "")
            # Custom rules without a group are scheduled separately by rule_id.
            key = group or f"__rule__:{rule['rule_id']}"
            grouped.setdefault(key, []).append(rule)

        group_rank = {
            group: rank for rank, group in enumerate(CHECKPOINT_GROUP_ORDER)
        }

        def scheduling_key(item: tuple[str, list[dict[str, Any]]]) -> tuple[int, int, str]:
            key, rules = item
            # A group with low-attempt rules retains priority; repeated failures across the whole group naturally lower it.
            retry_level = min(
                int(self.attempts.get(str(rule.get("rule_id", "")), 0) or 0)
                for rule in rules
            )
            group = str(rules[0].get("checkpoint", {}).get("group", "") or "")
            return retry_level, group_rank.get(group, 99), key

        _selected_key, selected_rules = min(grouped.items(), key=scheduling_key)
        return selected_rules

    def summary(self, rules_path: Any = None) -> dict[str, int]:
        total = resolved = processed = violation = compliant = undetermined = 0
        not_applicable = deferred = executing = 0
        findings_by_id = {
            str(item.get("check_id", "") or "").strip(): item
            for item in self.findings
            if isinstance(item, dict)
        }
        for rule in gui_rules(rules_path):
            total += 1
            rule_id = rule["rule_id"]
            status = str(self.status.get(rule_id, "")).strip()
            if status == CHECK_STATUS_DEFERRED:
                deferred += 1
            elif status == CHECK_STATUS_EXECUTING:
                executing += 1
            if status == CHECK_STATUS_DONE:
                resolved += 1
            finding = findings_by_id.get(rule_id)
            verdict = str(finding.get("verdict", "") or "").strip() if finding else ""
            if verdict not in VALID_VERDICTS:
                continue
            processed += 1
            if verdict == VERDICT_VIOLATION:
                violation += 1
            elif verdict == VERDICT_COMPLIANT:
                compliant += 1
            elif verdict == VERDICT_NOT_APPLICABLE:
                not_applicable += 1
            else:
                undetermined += 1
        return {
            "gui_total": total,
            # Keep executed as a serialized compatibility alias; interfaces use processed.
            "executed": processed,
            "processed": processed,
            "resolved": resolved,
            "compliant": compliant,
            "violation": violation,
            "undetermined": undetermined,
            "not_applicable": not_applicable,
            "deferred": deferred,
            "unresolved": undetermined,
            "executing": executing,
            "untested": max(0, total - processed),
            "pending": max(0, total - processed),
        }


# ----------------------------------------------------------------------
# Deterministic scenario generation; the model performs execution and review.
# ----------------------------------------------------------------------
def build_compliance_scenarios(rules: list[dict[str, Any]]) -> tuple[str, str, dict[str, Any]]:
    """Build navigation, test guidance, and plan metadata for a group of checkpoints.

    Metadata supplies rule IDs and whether the test phase should restart the process. Application data reset is handled only by the run entrypoint."""
    valid_rules = [rule for rule in rules if rule.get("checkpoint")]
    if not valid_rules:
        raise ValueError("build_compliance_scenarios requires at least one rule with a checkpoint")
    checkpoints = [rule["checkpoint"] for rule in valid_rules]
    group = str(checkpoints[0].get("group", "") or "compliance")
    label = GROUP_LABELS.get(group, group)
    rule_ids = [rule["rule_id"] for rule in valid_rules]

    clause_lines = []
    for rule in valid_rules:
        hints = "；".join(rule["checkpoint"].get("verdict_hints") or [])
        clause_lines.append(f"- {rule['rule_id']}（{rule['regulation']}）：{rule['clause']}" + (f" | Focus: {hints}" if hints else ""))
    clause_text = "\n".join(clause_lines)

    cold_start = all(bool(cp.get("cold_start")) for cp in checkpoints)

    # Start from the current screen. Cold start restarts only the process; reset application data only once at new-run initialization. Empty navigation guidance skips a separate navigation stage.
    navigation_scenario = ""
    objectives = list(
        dict.fromkeys(
            str(checkpoint.get("objective", "") or "").strip()
            for checkpoint in checkpoints
            if str(checkpoint.get("objective", "") or "").strip()
        )
    )
    objective_text = "\n".join(f"- {objective}" for objective in objectives)
    group_requirement = GROUP_EXECUTION_REQUIREMENTS.get(group, "")
    test_scenario = (
        f"[Compliance screening | {label}] Evidence objectives:\n{objective_text}\n"
        f"Rules to review:\n{clause_text}\n"
        + (f"{group_requirement}\n" if group_requirement else "")
        +
        "Execution requirements: collect evidence for each objective, showing the complete current screen before every action. "
        "For system permission or in-app consent dialogs, first capture original text and button layout, "
        "then select the required deny/necessary-only/consent action. Test only actual, safely reachable application functionality. "
        "Do not invent entries for inapplicable rules or perform irreversible account deletion, data deletion, or payments. "
        "Do not log out or switch accounts, including to inspect login/registration forms: "
        "the runner cannot restore credentials, and losing the session invalidates later authenticated evidence collection. "
        "End the scenario as unreachable when login/registration forms cannot be reached. "
        "Finish after reachable objectives are complete or inapplicability is established."
    )
    plan_extra = {
        "compliance_check_ids": rule_ids,
        "compliance_group": group,
        "compliance_cold_start": bool(cold_start),
    }
    return navigation_scenario, test_scenario, plan_extra


def check_rules_by_ids(rule_ids: list[str], rules_path: Any = None) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    for rule_id in rule_ids or []:
        rule = rule_by_id(str(rule_id), rules_path)
        if rule is not None:
            rules.append(rule)
    return rules
