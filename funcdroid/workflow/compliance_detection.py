"""Review execution traces against associated compliance checkpoints.

Return a verdict per rule using original execution JPEGs. Reusing image dimensions preserves coordinate context and small text legibility."""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from compliance.checkpoints import (
    VALID_VERDICTS,
    VERDICT_COMPLIANT,
    VERDICT_NOT_APPLICABLE,
    VERDICT_UNDETERMINED,
    VERDICT_VIOLATION,
    check_rules_by_ids,
)
from workflow.llm import invoke_llm_json_with_retry, llm_json_result_failed
from workflow.nodes._shared import JSON_BLOCK_PATTERN
from workflow.nodes.prompt import COMPLIANCE_DETECT_SYSTEM_PROMPT
from workflow.trace_messages import (
    build_step_text,
    coerce_step,
    image_block,
    image_ref_text,
    normalize_evidence_items,
    raw_step_index,
    step_screenshot,
)
from tools.actions import _screenshot_health_error
from tools.utils import decode_screenshot


class ComplianceReviewError(RuntimeError):
    """Model invocation or parsing still failed after retries, distinct from an omitted rule verdict."""


TERMINAL_EVIDENCE_VERDICTS = {VERDICT_COMPLIANT, VERDICT_VIOLATION, VERDICT_NOT_APPLICABLE}


def _collect_steps(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep all observed steps, including dialogs and recovery, when page evidence exists."""
    steps: list[dict[str, Any]] = []
    for raw_index, step in enumerate(trace, start=1):
        if not isinstance(step, dict):
            continue
        if not step_screenshot(step):
            continue
        copied = dict(step)
        copied["raw_step_index"] = raw_index
        steps.append(copied)
    return steps


def _compact_screenshot(screenshot: str) -> str:
    """Reuse the execution JPEG, removing only its data-URL prefix.

    Do not resize review evidence or change the coordinate reference dimensions."""
    screenshot = str(screenshot or "").strip()
    if screenshot.startswith("data:image") and "," in screenshot:
        screenshot = screenshot.split(",", 1)[1]
    return screenshot


def _downgrade_without_evidence(normalized: dict[str, Any], reason: str) -> dict[str, Any]:
    """Downgrade unsupported terminal verdicts to undetermined in place."""
    if normalized["verdict"] in TERMINAL_EVIDENCE_VERDICTS and not normalized["evidence"]:
        normalized["verdict"] = VERDICT_UNDETERMINED
        normalized["description"] = reason + normalized["description"]
    return normalized


def _normalize_result(item: dict[str, Any]) -> dict[str, Any] | None:
    check_id = str(item.get("check_id", "") or item.get("rule_id", "")).strip()
    verdict = str(item.get("verdict", "")).strip().lower()
    if not check_id or verdict not in VALID_VERDICTS:
        return None
    evidence = normalize_evidence_items(item.get("evidence"))
    description = str(item.get("description", "")).strip()
    applicability = item.get("applicability")
    applicability = applicability if isinstance(applicability, dict) else {}
    if verdict == VERDICT_NOT_APPLICABLE and (
        applicability.get("basis") != "explicit_scope_exclusion"
        or not str(applicability.get("explanation", "") or "").strip()
    ):
        verdict = VERDICT_UNDETERMINED
        description = "No positive evidence excludes applicability; not observed does not mean not applicable. " + description
    return _downgrade_without_evidence(
        {
            "check_id": check_id,
            "verdict": verdict,
            "description": description,
            "evidence": evidence,
            **({"applicability": applicability} if applicability else {}),
        },
        "The reviewer did not provide valid screenshot evidence. ",
    )


def _validate_functionality_evidence(item, steps):
    """Require ordered, executed actions and a later observed result."""
    chain = item.get("functionality_evidence")
    if not isinstance(chain, dict):
        return False
    # Accept positive integer steps, including numeric strings; reject booleans and nonpositive values.
    indices = [
        coerce_step(chain.get(key))
        for key in ("denial_step", "functional_action_step", "result_observation_step")
    ]
    if any(value is None for value in indices):
        return False
    denial, action, result = indices
    by_index = {step["raw_step_index"]: step for step in steps}
    if not (denial < action < result) or any(value not in by_index for value in indices):
        return False
    for value in indices:
        step = by_index[value]
        screenshot = step_screenshot(step)
        if step.get("screenshot_health_error") or _screenshot_health_error(decode_screenshot(screenshot)):
            return False
    for value in (denial, action):
        step = by_index[value]
        if step.get("action_failed") or step.get("action_name") not in {"click", "long_press", "type", "swipe", "scroll"}:
            return False
    return bool(str(chain.get("observed_result", "")).strip())


def _extract_raw_check_results(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Accept the small set of equivalent JSON shapes used by review models."""
    if not isinstance(result, dict):
        return []
    payload: Any = None
    for key in ("check_results", "results", "checks"):
        candidate = result.get(key)
        if isinstance(candidate, (list, dict)):
            payload = candidate
            break
    if payload is None and any(re.fullmatch(r"(?:R\d{3}|P\d{2})", str(key).strip()) for key in result):
        payload = result
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    extracted: list[dict[str, Any]] = []
    for check_id, value in payload.items():
        if not isinstance(value, dict):
            continue
        item = dict(value)
        item.setdefault("check_id", str(check_id))
        extracted.append(item)
    return extracted


def _build_messages(
    state: dict[str, Any],
    steps: list[dict[str, Any]],
    rules: list[dict[str, Any]],
) -> list:
    check_lines = []
    for rule in rules:
        checkpoint = rule.get("checkpoint", {}) or {}
        hints = "；".join(checkpoint.get("verdict_hints") or [])
        check_lines.append(
            f"- {rule['rule_id']}（{rule['regulation']}）：{rule['clause']}"
            + (f" | Review hints: {hints}" if hints else "")
        )
    header_text = (
        "[Compliance trace review]\n"
        "The complete evidence-collection trace follows; each step includes its pre-action screenshot.\n\n"
        "[Rules under review]\n" + "\n".join(check_lines) + "\n\n"
        "[Scenario objective]\n"
        f"{str(state.get('current_test_scenario', ''))[:800]}\n\n"
        "- ACTION_DESCRIPTION, parameters, and screenshots jointly identify the executed action; clicking a Deny button "
        "must not be described as never denying permission.\n"
        "- STEP N also observes the result of STEP N-1; the final step has no later result screenshot.\n"
        "- Permission dialogs, consent dialogs, and settings controls are core evidence; inspect their text and buttons.\n"
        "- For rules requiring basic functionality after denial, compliant must include functionality_evidence: "
        "{denial_step: actual denial step, functional_action_step: basic functionality action after denial, "
        "result_observation_step: later result screenshot step, observed_result: observed successful outcome}. "
        "A visible entry, returning home, or agent reasoning does not prove usability; without execution, return undetermined. "
        "- Return a verdict for every rule; use undetermined when the trace does not cover it."
    )
    message_content: list[dict[str, Any]] = [{"type": "text", "text": header_text}]
    previous_raw_index: int | None = None
    for index, step in enumerate(steps, start=1):
        raw_index = raw_step_index(step, index)
        compact = _compact_screenshot(step_screenshot(step))
        text = build_step_text(
            step,
            raw_index,
            coordinate_note="Distinguish image and screen coordinates; executor prose alone cannot prove a tap hit its target.",
        )
        # A screenshot observes the previous retained action only when their original step indices are consecutive.
        if compact and previous_raw_index is not None and previous_raw_index == raw_index - 1:
            text += (
                f"\n[OBSERVES_PREVIOUS_RESULT] This image shows the state after STEP {previous_raw_index} "
                "was executed."
            )
        text += image_ref_text(raw_index, has_image=bool(compact), image_note="state before the action.")
        message_content.append({"type": "text", "text": text})
        if compact:
            message_content.append(image_block(compact))
        previous_raw_index = raw_index
    return [
        SystemMessage(content=COMPLIANCE_DETECT_SYSTEM_PROMPT),
        HumanMessage(content=message_content),
    ]


def analyze_execution_for_compliance(
    llm,
    state: dict[str, Any],
) -> dict:
    """Review one evidence trace and return check_results."""
    check_ids = [
        str(item).strip()
        for item in (state.get("current_plan", {}) or {}).get("compliance_check_ids", []) or []
        if str(item).strip()
    ]
    rules_path = str(state.get("compliance_rules_path", "") or "").strip() or None
    rules = check_rules_by_ids(check_ids, rules_path)
    if not rules:
        return {"check_results": []}

    navigation_trace = state.get("navigation_execution_trace", [])
    scenario_trace = state.get("scenario_execution_trace", [])
    navigation_trace = list(navigation_trace) if isinstance(navigation_trace, list) else []
    scenario_trace = list(scenario_trace) if isinstance(scenario_trace, list) else []
    steps = _collect_steps(navigation_trace + scenario_trace)

    if not steps:
        # Without screenshot-bearing steps, mark every rule undetermined and finish this review attempt.
        return {
            "check_results": [
                {
                    "check_id": rule["rule_id"],
                    "verdict": VERDICT_UNDETERMINED,
                    "description": "This trace contains no screenshot-bearing steps; evidence is unavailable.",
                    "evidence": [],
                }
                for rule in rules
            ]
        }

    print(
        f"[Compliance review] Input state: check_ids={len(rules)}, "
        f"recordable_steps={len(steps)}"
    )
    messages = _build_messages(state, steps, rules)
    result = invoke_llm_json_with_retry(
        llm,
        messages,
        log_prefix="[Compliance review]",
        json_block_pattern=JSON_BLOCK_PATTERN,
    )
    if llm_json_result_failed(result):
        # Exhausted model/parse retries are infrastructure failures, distinct from an omitted verdict; raise so the worker records artifact incompleteness.
        raise ComplianceReviewError(getattr(result, "failure_reason", "Model review failed"))
    raw_results = _extract_raw_check_results(result)
    print(
        "[Compliance review] Model response structure: "
        f"top_keys={list(result)[:8] if isinstance(result, dict) else []}, "
        f"raw_results={len(raw_results)}, "
        f"ids={[str(item.get('check_id', '') or item.get('rule_id', '')) for item in raw_results[:12]]}"
    )
    normalized_by_id: dict[str, dict[str, Any]] = {}
    observed_indices = {step["raw_step_index"] for step in steps}
    rules_by_id = {rule["rule_id"]: rule for rule in rules}
    for item in raw_results:
        normalized = _normalize_result(item if isinstance(item, dict) else {})
        if normalized is not None:
            normalized["evidence"] = [
                evidence for evidence in normalized["evidence"]
                if evidence["global_step"] in observed_indices
            ]
            _downgrade_without_evidence(normalized, "Referenced steps do not have valid screenshot evidence. ")
            rule = rules_by_id.get(normalized["check_id"], {})
            requires_functionality = rule.get("checkpoint", {}).get("requires_functionality_evidence", False)
            if normalized["verdict"] == VERDICT_COMPLIANT and requires_functionality:
                if not _validate_functionality_evidence(item, steps):
                    normalized["verdict"] = VERDICT_UNDETERMINED
                    normalized["description"] = "Missing a complete chain of denial, actual basic-functionality execution, and a later result screenshot. " + normalized["description"]
                else:
                    normalized["functionality_evidence"] = dict(item["functionality_evidence"])
            normalized_by_id[normalized["check_id"]] = normalized
    # Fill omitted rules with undetermined so every check_id receives a result.
    check_results = [
        normalized_by_id.get(
            rule["rule_id"],
            {
                "check_id": rule["rule_id"],
                "verdict": VERDICT_UNDETERMINED,
                "description": "The reviewer omitted this rule verdict.",
                "evidence": [],
            },
        )
        for rule in rules
    ]
    verdict_counts: dict[str, int] = {}
    for item in check_results:
        verdict_counts[item["verdict"]] = verdict_counts.get(item["verdict"], 0) + 1
    print(f"[Compliance review] Round verdicts: {json.dumps(verdict_counts, ensure_ascii=False)}")
    return {"check_results": check_results}
