"""Execution-trace bug analysis."""

# Unicode escapes preserve literal UI labels and legacy evidence matching.
import json
import os
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from model.flow_graph import Page
from tools.utils import screenshot_dimensions
from workflow.bug_dedupe import bug_identity_key, dedupe_bugs, is_similar_bug_key
from workflow.image_compact import compact_screenshot
from workflow.llm import invoke_llm_json_with_retry, llm_json_result_failed
from workflow.logcat_crash import NON_CRASH_LOGCAT_PATTERNS, _logcat_has_target_process_crash  # noqa: F401 - Keep compatibility imports available.
from workflow.nodes._shared import JSON_BLOCK_PATTERN
from workflow.nodes.prompt import BUG_DETECT_SYSTEM_PROMPT
from workflow.state import AgentState
from workflow.trace_messages import (
    build_step_text,
    image_block,
    image_ref_text,
    normalize_evidence_items,
    page_screenshot,
    raw_step_index,
)


class DefectReviewError(RuntimeError):
    """The review failed to produce a valid bugs array; [] is a valid verdict."""


FUNCTIONAL_BUG_TYPES = {"ui_error", "logic_error"}
DIRECT_EVIDENCE_CATEGORIES = {
    "rendering_corruption",
    "persisted_result_error",
    "simultaneous_state_contradiction",
    "single_action_resource_mutation",
}
SEMANTIC_ANALYSIS_STATUSES = {
    "completed",
    "stalled",
    "failed_goal",
    "left_app",
}
SPECULATIVE_NAVIGATION_SCENARIO_PATTERN = re.compile(
    r"\u89c2\u5bdf[^\n。]{0,80}\u662f\u5426(?:\u8df3\u8f6c|\u51fa\u73b0|\u6253\u5f00|\u8fdb\u5165|\u5207\u6362)"
)
SPECULATIVE_NAVIGATION_BUG_PATTERN = re.compile(
    r"\u672a[^\n。]{0,40}(?:\u9884\u671f|\u8df3\u8f6c)|\u53cd\u800c|\u4e1a\u52a1\u8bed\u4e49|\u5e94\u8df3\u8f6c|\u5e94\u8fdb\u5165"
)
SPATIAL_ASSOCIATION_PATTERN = re.compile(
    r"\u65c1\u8fb9|\u65c1\u7684|\u5bf9\u5e94\u4f4d\u7f6e|\u5173\u8054\u5230|\u9519\u8bef\u5173\u8054|\u800c\u975e\u76ee\u6807|\u65e0\u5173"
)
BOUNDARY_INPUT_SCENARIO_PATTERN = re.compile(
    r"\u4e0d\u8f93\u5165\u4efb\u4f55|\u7a7a\u6807\u9898|\u7a7a\u503c|\u7279\u6b8a\u5b57\u7b26|\u8d85\u957f|\u8fb9\u754c|truncation|!@#",
    re.IGNORECASE,
)
UNSUPPORTED_INPUT_RULE_PATTERN = re.compile(
    r"\u5fc5\u586b|\u4e0d\u80fd\u4e3a\u7a7a|\u5e94\u5f53|\u4e0d\u5e94|\u6821\u9a8c|\u5408\u6cd5|\u4e1a\u52a1.*\u903b\u8f91"
)
INPUT_WAS_NOT_APPLIED_PATTERN = re.compile(
    r"\u8f93\u5165\u6846(?:\u4ecd|\u5c1a\u672a|\u4e3a)?\u7a7a|\u672a\u8f93\u5165\u4efb\u4f55|\u6ca1\u6709\u8f93\u5165|\u8f93\u5165\u672a\u751f\u6548"
)
RESOURCE_MUTATION_PATTERN = re.compile(
    r"\u91cd\u590d|\u4e22\u5931|\u5220\u9664|\u65b0\u589e|\u65b0\u5efa|\u521b\u5efa|\u751f\u6210|\u5173\u8054|\u5199\u5165|\u8986\u76d6|\u9519\u4f4d"
)
# A one-shot task may restart or reset its result view on reentry. Do not infer lost persistence unless the evidence concerns saved settings or data.
ONE_SHOT_TASK_PATTERN = re.compile(
    r"(?:\u4e00\u952e|\u7acb\u5373|\u5f00\u59cb)[\u4e00-\u9fa5]{1,4}"
    r"|(?:\u4efb\u52a1|\u6267\u884c|\u8fd0\u884c|\u5904\u7406|\u68c0\u6d4b|\u626b\u63cf)(?:\u6210\u529f|\u5b8c\u6210|\u7ed3\u675f)"
)
REENTRY_RESET_PATTERN = re.compile(
    r"(?:\u518d\u6b21|\u91cd\u65b0|\u4e8c\u6b21)\u8fdb\u5165|\u6062\u590d(?:\u4e3a|\u5230|\u6210)?(?:\u521d\u59cb|\u9ed8\u8ba4|\u539f\u6765|\u6267\u884c\u524d|\u64cd\u4f5c\u524d)"
    r"|\u91cd\u7f6e|\u53c8\u56de\u5230|\u56de\u5230[^\n。]{0,8}(?:\u521d\u59cb|\u4e4b\u524d|\u524d\u7684)|\u91cd\u65b0\u51fa\u73b0[^\n。]{0,10}(?:\u5f00\u59cb|\u4e00\u952e|\u7acb\u5373)"
)
PERSISTENT_STATE_PATTERN = re.compile(
    r"\u5f00\u5173|\u8bbe\u7f6e|\u6a21\u5f0f|\u9009\u9879|\u52fe\u9009|\u8f93\u5165|\u7f16\u8f91|\u4fdd\u5b58|\u63d0\u4ea4|\u6536\u85cf|\u6dfb\u52a0|\u5220\u9664|\u4fee\u6539|\u8d26\u53f7|\u767b\u5f55"
)
# Transient progress graphics need not match a numerical indicator frame by frame. Exempt only concurrent progress, animation, and numerical indicators; contradictory text remains reportable.
IN_PROGRESS_STATE_PATTERN = re.compile(
    r"\u6b63\u5728[^\n。，,]{0,8}"
    r"|[\u4e00-\u9fa5]{2}\u4e2d\s*(?:…|\.{2,}|⋯)"
    r"|(?:\u52a0\u8f7d|\u5904\u7406|\u4e0b\u8f7d|\u4e0a\u4f20|\u540c\u6b65|\u5b89\u88c5|\u66f4\u65b0|\u68c0\u6d4b|\u626b\u63cf|\u6267\u884c|\u8fd0\u884c)\u4e2d"
)
ANIMATION_GRAPHIC_PATTERN = re.compile(
    r"\u52a8\u753b|\u56fe\u5f62|\u56fe\u793a|\u56fe\u6848|\u586b\u5145|\u8fdb\u5ea6\u6761|\u8fdb\u5ea6\u73af|\u5706\u73af|\u6ce2\u7eb9|\u6307\u9488"
)
PROGRESS_VALUE_PATTERN = re.compile(r"\d{1,3}\s*[%％]|\u8fdb\u5ea6")
PLACEHOLDER_OR_EMPTY_ITEM_PATTERN = re.compile(
    r"placeholder|new item|\u7a7a\u767d\u9879|\u7a7a\u9879|\u5360\u4f4d|\u672a\u8f93\u5165",
    re.IGNORECASE,
)
# Match FFG compaction defaults so reviewers share cached images; quality 100 can increase image size.
BUG_DETECT_IMAGE_MAX_SIDE = max(
    160,
    int(os.getenv("BUG_DETECT_IMAGE_MAX_SIDE", os.getenv("FFG_UPDATE_IMAGE_MAX_SIDE", "960"))),
)
BUG_DETECT_IMAGE_QUALITY = max(
    30,
    min(
        int(os.getenv("BUG_DETECT_IMAGE_QUALITY", os.getenv("FFG_UPDATE_IMAGE_QUALITY", "80"))),
        95,
    ),
)
BUG_CONTEXT_FIELDS = (
    "scenario_type",
    "target_function_id",
    "src_function_id",
    "dst_function_id",
)
ENVIRONMENT_CRASH_LOGCAT_PATTERNS = (
    "EGL_BAD_ALLOC",
    "EGL_NOT_INITIALIZED",
    "Failed to create EGLSurface",
    "Failed to choose config",
    "DeadSystemException",
)
def _extract_page(step: dict[str, Any]) -> Page | dict[str, Any] | None:
    page = step.get("current_page")
    if isinstance(page, (Page, dict)):
        return page
    return None


# Review prose is English; retain legacy UI/report vocabulary as additional patterns.
def _with_english(pattern: re.Pattern, english: str) -> re.Pattern:
    return re.compile(f"(?:{pattern.pattern})|(?:{english})", pattern.flags | re.IGNORECASE)


SPECULATIVE_NAVIGATION_SCENARIO_PATTERN = _with_english(SPECULATIVE_NAVIGATION_SCENARIO_PATTERN, r"observe.{0,80}(?:whether|if).{0,30}(?:open|navigate|appear|enter|switch)")
SPECULATIVE_NAVIGATION_BUG_PATTERN = _with_english(SPECULATIVE_NAVIGATION_BUG_PATTERN, r"should (?:open|navigate|enter)|instead|business semantics|did not.{0,30}(?:expect|navigate)")
SPATIAL_ASSOCIATION_PATTERN = _with_english(SPATIAL_ASSOCIATION_PATTERN, r"beside|adjacent|next to|wrong association|unrelated|rather than the target")
BOUNDARY_INPUT_SCENARIO_PATTERN = _with_english(BOUNDARY_INPUT_SCENARIO_PATTERN, r"empty (?:title|value|input)|no input|special characters|overlong|boundary")
UNSUPPORTED_INPUT_RULE_PATTERN = _with_english(UNSUPPORTED_INPUT_RULE_PATTERN, r"required|must not be empty|should|validation|valid input|business logic")
INPUT_WAS_NOT_APPLIED_PATTERN = _with_english(INPUT_WAS_NOT_APPLIED_PATTERN, r"(?:input|field).{0,20}(?:still|remains?).{0,10}empty|nothing (?:was )?entered|input (?:did not|was not) (?:apply|applied)")
RESOURCE_MUTATION_PATTERN = _with_english(RESOURCE_MUTATION_PATTERN, r"duplicate|lost|delet|add|creat|generat|associat|writ|overwrit|misalign")
ONE_SHOT_TASK_PATTERN = _with_english(ONE_SHOT_TASK_PATTERN, r"one[- ]click|start (?:task|scan|process)|(?:task|execution|scan|process).{0,15}(?:complete|finished|succeeded)")
REENTRY_RESET_PATTERN = _with_english(REENTRY_RESET_PATTERN, r"re[- ]?enter|enter again|reset|return.{0,15}(?:initial|default|previous)")
PERSISTENT_STATE_PATTERN = _with_english(PERSISTENT_STATE_PATTERN, r"switch|setting|mode|option|check|input|edit|save|submit|favorite|add|delete|modify|account|login")
IN_PROGRESS_STATE_PATTERN = _with_english(IN_PROGRESS_STATE_PATTERN, r"in progress|loading|processing|downloading|uploading|syncing|installing|updating|scanning|running")
ANIMATION_GRAPHIC_PATTERN = _with_english(ANIMATION_GRAPHIC_PATTERN, r"animation|graphic|diagram|pattern|fill|progress bar|progress ring|ripple|pointer")
PROGRESS_VALUE_PATTERN = _with_english(PROGRESS_VALUE_PATTERN, r"progress")


def _page_screenshot(page: Page | dict[str, Any] | None) -> str:
    return page_screenshot(page)


def _is_recordable_step(step: dict[str, Any]) -> bool:
    return (
        not bool(step.get("is_interruption"))
        and not bool(step.get("action_failed"))
        and not bool(step.get("external_context"))
    )


def _filter_trace(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for raw_index, step in enumerate(trace, start=1):
        if not _is_recordable_step(step):
            continue
        copied = dict(step)
        copied["raw_step_index"] = raw_index
        filtered.append(copied)
    return filtered


def _compact_screenshot_for_llm(
    screenshot: str,
    *,
    max_side: int = BUG_DETECT_IMAGE_MAX_SIDE,
    quality: int = BUG_DETECT_IMAGE_QUALITY,
) -> str:
    """Compact review images through the process-wide shared cache."""
    return compact_screenshot(
        screenshot,
        max_side=max_side,
        quality=quality,
        source_label="Bug detection",
    )


def _bug_context_from_plan(
    current_plan: dict[str, Any],
    execution_result: dict[str, Any] | None = None,
    state: AgentState | None = None,
) -> dict[str, str]:
    execution_result = execution_result or {}
    state = state or {}
    scenario_type = str(
        current_plan.get("scenario_type")
        or execution_result.get("scenario_type")
        or ""
    ).strip()
    target_function_id = str(current_plan.get("target_function_id", "")).strip()
    if scenario_type == "intra_test" and not target_function_id:
        target_function_id = str(state.get("current_testing_function_id", "")).strip()
    return {
        "scenario_type": scenario_type,
        "target_function_id": target_function_id,
        "src_function_id": str(current_plan.get("src_function_id", "")).strip(),
        "dst_function_id": str(current_plan.get("dst_function_id", "")).strip(),
    }


def _apply_bug_context(bug: dict[str, Any], bug_context: dict[str, str]) -> dict[str, Any]:
    for field in BUG_CONTEXT_FIELDS:
        bug[field] = str(bug_context.get(field, "")).strip()
    return bug


def _normalize_bug(
    bug: dict[str, Any],
    scenario: str,
    bug_context: dict[str, str] | None = None,
) -> dict[str, Any]:
    evidence = normalize_evidence_items(bug.get("evidence"))
    normalized = {
        "type": str(bug.get("type", "")).strip().lower(),
        "evidence_category": str(bug.get("evidence_category", "")).strip().lower(),
        "severity": str(bug.get("severity", "")).strip(),
        "confidence": str(bug.get("confidence", "")).strip().lower(),
        "description": str(bug.get("description", "")).strip(),
        "expected": str(bug.get("expected", "")).strip(),
        "actual": str(bug.get("actual", "")).strip(),
        "evidence": evidence,
        "scenario": scenario,
        "source": "trace_analyzer",
    }
    return _apply_bug_context(normalized, bug_context or {})


def _runtime_crash_bug(
    execution_result: dict[str, Any],
    execution_trace: list[dict[str, Any]],
    scenario: str,
    package_name: str,
    bug_context: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    navigation_status = str(execution_result.get("navigation_status", "")).strip()
    scenario_status = str(execution_result.get("scenario_status", "")).strip()
    crashed = (
        navigation_status == "app_crashed"
        or scenario_status == "app_crashed"
        or bool(execution_result.get("crash_detected"))
    )
    # Retain execution_trace for compatibility; crash acceptance depends on runner status and target logcat evidence.
    if not crashed:
        return None

    crash_logcat = str(execution_result.get("crash_logcat", "")).strip()
    escaped_package = re.escape(str(package_name or "").strip())
    has_explicit_app_runtime_crash = bool(
        escaped_package
        and (
            re.search(rf"\bANR in\s+{escaped_package}\b", crash_logcat)
            or (
                "FATAL EXCEPTION" in crash_logcat
                and re.search(rf"\bProcess:\s*{escaped_package}\b", crash_logcat)
            )
        )
    )
    if (
        not has_explicit_app_runtime_crash
        and any(pattern in crash_logcat for pattern in ENVIRONMENT_CRASH_LOGCAT_PATTERNS)
    ):
        print("[Bug detection] Crash evidence belongs to emulator graphics/system services; do not report a target-app bug.")
        return None
    if not _logcat_has_target_process_crash(crash_logcat, package_name):
        if crash_logcat:
            print("[Bug detection] No confirmed target-process crash stack; do not report a crash bug.")
        return None

    crash_phase = "navigation" if navigation_status == "app_crashed" else "test"
    error_key = "navigation_error" if crash_phase == "navigation" else "scenario_error"
    actual = (
        str(execution_result.get(error_key, "")).strip()
        or f"Target application crashed during the {crash_phase} scenario."
    )
    if crash_logcat:
        actual = f"{actual}\n\nlogcat excerpt:\n{crash_logcat[:4000]}"
    recovered = bool(execution_result.get("recovered_to_target_app"))
    recovery_text = "the runner restarted the target" if recovered else "the runner could not restore the target"
    bug = {
        "type": "crash",
        "severity": "high",
        "confidence": "high",
        "description": f"The target crashed during the {crash_phase} scenario; {recovery_text} and ended the scenario.",
        "expected": f"The target should remain stable and complete the {crash_phase} scenario.",
        "actual": actual,
        "scenario": scenario,
        "source": "runtime_monitor",
    }
    return _apply_bug_context(bug, bug_context or {})


def _dedupe_bugs(bugs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for bug in bugs:
        if not isinstance(bug, dict):
            continue
        if str(bug.get("confidence", "")).strip().lower() != "high":
            print(
                "[Bug detection] Skip finding below high confidence: "
                f"{bug.get('description', '')}"
            )
            continue
        filtered.append(bug)
    deduped = dedupe_bugs(filtered)
    if len(deduped) != len(filtered):
        print(f"[Bug detection] Deduplicated findings: {len(filtered) - len(deduped)}")
    return deduped


def _bug_has_required_trace_evidence(
    bug: dict[str, Any],
    trace: list[dict[str, Any]],
) -> bool:
    evidence = bug.get("evidence", [])
    if not isinstance(evidence, list) or not evidence:
        return False
    step_by_index = {
        int(step.get("raw_step_index", index) or index): step
        for index, step in enumerate(trace, start=1)
        if isinstance(step, dict) and _is_recordable_step(step)
    }
    referenced: list[tuple[int, str, dict[str, Any]]] = []
    for item in evidence:
        if not isinstance(item, dict):
            return False
        index = item.get("global_step")
        role = str(item.get("role", "") or "").strip().lower()
        if not isinstance(index, int) or index not in step_by_index or role not in {"before", "action", "after"}:
            return False
        if screenshot_dimensions(_page_screenshot(step_by_index[index].get("current_page"))) is None:
            return False
        # Exclude unhealthy screenshots such as secure-content black frames from defect evidence.
        if str(step_by_index[index].get("screenshot_health_error", "") or "").strip():
            return False
        referenced.append((index, role, step_by_index[index]))
    if not any(str(step.get("trace_phase", "")).strip() == "test" for _, _, step in referenced):
        return False

    category = bug.get("evidence_category")
    if category in {"persisted_result_error", "single_action_resource_mutation"}:
        action_refs = [item for item in referenced if item[1] == "action"]
        after_refs = [item for item in referenced if item[1] == "after"]
        if len(action_refs) != 1 or not after_refs:
            return False
        action_index, _, action_step = action_refs[0]
        if str(action_step.get("trace_phase", "")).strip() != "test":
            return False
        if not any(index > action_index for index, _, _ in after_refs):
            return False
        # Verify that the referenced action actually executed, not how it was named.
        if action_step.get("action_name") not in {"click", "long_press", "type", "swipe", "scroll", "press_back"}:
            return False

    bug_text = " ".join(str(bug.get(key, "")) for key in ("description", "expected", "actual"))
    if PLACEHOLDER_OR_EMPTY_ITEM_PATTERN.search(bug_text):
        action_indices = [index for index, role, _ in referenced if role == "action"]
        cutoff = min(action_indices) if action_indices else 10**9
        has_nonempty_input = any(
            int(step.get("raw_step_index", index) or index) < cutoff
            and str(step.get("trace_phase", "")).strip() == "test"
            and str(step.get("action_name", "")).strip() == "type"
            and bool(str(step.get("content", "")).strip())
            for index, step in enumerate(trace, start=1)
            if isinstance(step, dict)
        )
        if not has_nonempty_input:
            return False
    return True


def _bug_uses_speculative_or_automation_expectation(
    bug: dict[str, Any],
    scenario: str,
) -> bool:
    bug_text = " ".join(
        str(bug.get(key, "")).strip()
        for key in ("description", "expected", "actual")
    )
    scenario = str(scenario or "").strip()
    if (
        bug.get("evidence_category") == "single_action_resource_mutation"
        and not RESOURCE_MUTATION_PATTERN.search(bug_text)
    ):
        return True
    if (
        bug.get("evidence_category") == "persisted_result_error"
        and ONE_SHOT_TASK_PATTERN.search(bug_text)
        and REENTRY_RESET_PATTERN.search(bug_text)
        and not PERSISTENT_STATE_PATTERN.search(bug_text)
    ):
        return True
    if (
        bug.get("evidence_category") == "simultaneous_state_contradiction"
        and IN_PROGRESS_STATE_PATTERN.search(bug_text)
        and ANIMATION_GRAPHIC_PATTERN.search(bug_text)
        and PROGRESS_VALUE_PATTERN.search(bug_text)
    ):
        return True
    if (
        SPECULATIVE_NAVIGATION_SCENARIO_PATTERN.search(scenario)
        and SPECULATIVE_NAVIGATION_BUG_PATTERN.search(bug_text)
    ):
        return True
    if re.search(r"\u65c1\u8fb9|beside|next to|adjacent", scenario, re.IGNORECASE) and SPATIAL_ASSOCIATION_PATTERN.search(bug_text):
        return True
    if (
        BOUNDARY_INPUT_SCENARIO_PATTERN.search(scenario)
        and UNSUPPORTED_INPUT_RULE_PATTERN.search(bug_text)
    ):
        return True
    if (
        re.search(r"\u8f93\u5165(?:\u6587\u672c)?\s*[\"'“‘][^\"'”’]+[\"'”’]", scenario)
        and INPUT_WAS_NOT_APPLIED_PATTERN.search(bug_text)
    ):
        return True
    return False


def copy_bug_record(bug: dict[str, Any]) -> dict[str, Any]:
    """Copy one finding, including one level of nested lists/dictionaries, for thread-safe snapshots."""
    return {
        key: (
            [dict(item) if isinstance(item, dict) else item for item in value]
            if isinstance(value, list)
            else dict(value) if isinstance(value, dict) else value
        )
        for key, value in bug.items()
    }


def _ensure_bug_ids(bugs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return new finding dictionaries with context and bug_id.

    Do not mutate records that the main thread may be serializing concurrently."""
    used_ids = {
        str(bug.get("bug_id", "")).strip()
        for bug in bugs
        if isinstance(bug, dict) and str(bug.get("bug_id", "")).strip()
    }
    next_index = 1
    result: list[dict[str, Any]] = []
    for bug in bugs:
        if not isinstance(bug, dict):
            result.append(bug)
            continue
        bug = {**{field: "" for field in BUG_CONTEXT_FIELDS}, **bug}
        if not str(bug.get("bug_id", "")).strip():
            while f"BUG-{next_index:03d}" in used_ids:
                next_index += 1
            bug["bug_id"] = f"BUG-{next_index:03d}"
            used_ids.add(bug["bug_id"])
            next_index += 1
        result.append(bug)
    return result


def _bug_trace_quality(trace: list[dict[str, Any]]) -> tuple[int, int, int]:
    if not isinstance(trace, list):
        return (0, 0, 0)
    step_count = 0
    screenshot_count = 0
    non_empty_page_count = 0
    for step in trace:
        if not isinstance(step, dict):
            continue
        step_count += 1
        page = step.get("current_page")
        if isinstance(page, Page):
            if page.representative_screenshot:
                screenshot_count += 1
            if page.activity or page.description or page.operations or page.representative_screenshot:
                non_empty_page_count += 1
        elif isinstance(page, dict):
            screenshot = str(page.get("screenshot") or page.get("screenshot_base64") or "").strip()
            if screenshot:
                screenshot_count += 1
            if screenshot or str(page.get("activity", "")).strip() or str(page.get("description", "")).strip():
                non_empty_page_count += 1
    return (step_count, screenshot_count, non_empty_page_count)


def _is_better_bug_trace(candidate: list[dict[str, Any]], existing: list[dict[str, Any]] | None) -> bool:
    if not isinstance(existing, list) or not existing:
        return bool(candidate)
    return _bug_trace_quality(candidate) > _bug_trace_quality(existing)


def _current_bug_trace(state: AgentState) -> list[dict[str, Any]]:
    execution_result = state.get("execution_result", {}) or {}
    navigation_trace = state.get("navigation_execution_trace", [])
    scenario_trace = state.get("scenario_execution_trace", [])
    navigation_trace = list(navigation_trace) if isinstance(navigation_trace, list) else []
    scenario_trace = list(scenario_trace) if isinstance(scenario_trace, list) else []
    if str(execution_result.get("navigation_status", "")).strip() == "app_crashed":
        return navigation_trace
    return navigation_trace + scenario_trace


def _update_bug_trace_records(
    state: AgentState,
    detected_bugs: list[dict[str, Any]],
    current_bugs: list[dict[str, Any]],
    trace: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    existing_records = state.get("bug_trace_records", {})
    records = dict(existing_records) if isinstance(existing_records, dict) else {}
    if not current_bugs or not trace:
        return records
    for index, bug in enumerate(detected_bugs):
        if not isinstance(bug, dict):
            continue
        bug_id = str(bug.get("bug_id", "")).strip()
        if not bug_id:
            continue
        bug_key = bug_identity_key(bug)
        candidate = next((current for current in current_bugs if isinstance(current, dict) and (
            str(current.get("bug_id", "")).strip() == bug_id
            or is_similar_bug_key(bug_key, bug_identity_key(current))
        )), None)
        if candidate is not None and _is_better_bug_trace(trace, records.get(bug_id)):
            # Replace verdict, evidence and trace together; never mutate the worker's old snapshot.
            detected_bugs[index] = {**candidate, "bug_id": bug_id}
            records[bug_id] = list(trace)
    return records


def _build_full_trace_messages(
    state: AgentState,
    trace: list[dict[str, Any]],
) -> list:
    """Build detection messages from the complete recordable trace.

    Every recordable step's screenshot is sent independently (no reuse of the
    action-agent conversation, no sampling): this function is the sole
    message-building path so that async/background detection only needs a
    serializable state snapshot, not a live LLM conversation object.
    """
    execution_result = dict(state.get("execution_result", {}) or {})
    execution_result.pop("crash_logcat", None)
    current_plan = dict(state.get("current_plan", {}) or {})
    existing_bug_summaries = [
        {
            field: str(bug.get(field, "")).strip()[:limit]
            for field, limit in (
                ("bug_id", 80),
                ("type", 40),
                ("evidence_category", 80),
                ("target_function_id", 80),
                ("description", 320),
                ("expected", 240),
                ("actual", 320),
            )
            if str(bug.get(field, "")).strip()
        }
        for bug in state.get("detected_bugs", []) or []
        if isinstance(bug, dict)
        and any(
            str(bug.get(field, "")).strip()
            for field in ("description", "expected", "actual")
        )
    ]
    review_context = {
        "navigation_scenario": str(state.get("current_navigation_scenario", "")).strip(),
        "test_scenario": str(state.get("current_test_scenario", "")).strip(),
        "current_plan": current_plan,
        "execution_result": execution_result,
        "existing_bug_summaries": existing_bug_summaries,
    }
    # Provide the phase, not an expected-result narrative: test intent does not establish a product promise.
    function_phase = str(current_plan.get("function_phase", "") or "").strip()
    if function_phase:
        review_context["function_phase"] = function_phase
    header_text = (
        "[Complete execution-trace bug review]\n"
        "The complete navigation/test action sequence follows, with the screenshot before each action. "
        "Judge only the supplied screenshots and step metadata; do not infer unseen image content.\n\n"
        "[Authoritative context]\n"
        f"{json.dumps(review_context, ensure_ascii=False, default=str)}\n\n"
        "- Steps marked is_interruption/action_failed/external_context/foreground_recovery in [STEP_FLAGS] "
        "are automation or external-context events by default, not direct evidence of an application bug.\n"
        "- [AGENT_OBSERVATION] is executor commentary used to locate suspicious states; it is not evidence. "
        "Ground conclusions in screenshots.\n"
        "- function_phase=depth means the scenario follows a functionality business-operation chain. "
        "Compare before/after states and results after leaving and returning; do not treat test intent as a product promise.\n"
        "- Scenario text expresses test intent. Support expected behavior with screenshots, explicit save/submit results, or a contradiction within the same screen.\n"
        "- existing_bug_summaries contains previously confirmed findings for deduplication by root cause and visible consequence. "
        "It is not current evidence; do not emit a duplicate finding."
    )
    message_content: list[dict[str, Any]] = [{"type": "text", "text": header_text}]
    # Image references must point to actual attached images to avoid shifting the perceived step state.
    screenshots = [
        _compact_screenshot_for_llm(_page_screenshot(_extract_page(step)))
        for step in trace
    ]
    for index, step in enumerate(trace, start=1):
        raw_index = raw_step_index(step, index)
        screenshot = screenshots[index - 1]
        text = build_step_text(
            step,
            raw_index,
            coordinate_note="Review images may be resized; do not interpret screen pixels as positions in this image.",
            include_flags=True,
        )
        # Use raw_step_index for both images and evidence references, including when intermediate steps were filtered.
        text += image_ref_text(
            raw_index,
            has_image=bool(screenshot),
            image_note="page state before the action, not its result.",
        )
        if index < len(trace) and screenshots[index]:
            next_raw_index = raw_step_index(trace[index], index + 1)
            text += f" The result of STEP {raw_index} is normally visible in BEFORE_STEP_{next_raw_index}."
        message_content.append({"type": "text", "text": text})
        if screenshot:
            message_content.append(image_block(screenshot))
    return [
        SystemMessage(content=BUG_DETECT_SYSTEM_PROMPT),
        HumanMessage(content=message_content),
    ]


def analyze_execution_for_bugs(
    llm,
    state: AgentState,
) -> dict:
    """Analyze a completed navigation/test execution and update bug evidence."""
    navigation_trace = state.get("navigation_execution_trace", [])
    scenario_trace = state.get("scenario_execution_trace", [])
    navigation_trace = list(navigation_trace) if isinstance(navigation_trace, list) else []
    scenario_trace = list(scenario_trace) if isinstance(scenario_trace, list) else []
    full_trace = navigation_trace + scenario_trace
    trace = _filter_trace(full_trace)
    execution_result = state.get("execution_result", {}) or {}
    current_plan = state.get("current_plan", {}) or {}
    test_scenario = str(state.get("current_test_scenario", "")).strip()
    navigation_scenario = str(state.get("current_navigation_scenario", "")).strip()
    scenario_for_bug = (
        test_scenario
        or navigation_scenario
    )
    package_name = str(state.get("package_name", "")).strip()
    existing_bugs = list(state.get("detected_bugs", []) or [])
    bug_context = _bug_context_from_plan(current_plan, execution_result, state)

    runtime_bugs = [
        bug
        for bug in [
            _runtime_crash_bug(
                execution_result,
                full_trace,
                scenario_for_bug,
                package_name,
                bug_context,
            )
        ]
        if bug is not None
    ]

    print(
        "[Bug detection] Input state: "
        f"scenario_status={execution_result.get('scenario_status', '')}, "
        f"navigation_steps={len(navigation_trace)}, test_steps={len(scenario_trace)}, "
        f"recordable_steps={len(trace)}, "
        f"existing_bugs={len(existing_bugs)}"
    )

    scenario_status = str(execution_result.get("scenario_status", "")).strip()
    last_step = full_trace[-1] if full_trace else {}
    observed_at_step_limit = (
        scenario_status == "incomplete"
        and bool(last_step.get("terminal_observation"))
        and bool(last_step.get("observation_only"))
        and not last_step.get("screenshot_health_error")
        and _is_recordable_step(last_step)
        and bool(_page_screenshot(_extract_page(last_step)))
    )
    skip_reason = ""
    if not full_trace:
        skip_reason = "Execution trace is empty"
    elif not trace:
        skip_reason = "Execution trace is empty after filtering invalid steps"
    elif scenario_status not in SEMANTIC_ANALYSIS_STATUSES and not observed_at_step_limit:
        skip_reason = f"Execution status {scenario_status or 'unknown'} lacks a reliable final observation"
    elif current_plan.get("audit_exploration") is True:
        skip_reason = "Entry exploration establishes page/functionality evidence only; skip model bug review without a business result"

    if skip_reason:
        print(f"[Bug detection] {skip_reason}; skipping model bug detection.")
        detected_bugs = _ensure_bug_ids(_dedupe_bugs(existing_bugs + runtime_bugs))
        bug_trace_records = _update_bug_trace_records(
            state,
            detected_bugs,
            runtime_bugs,
            _current_bug_trace(state),
        )
        return {
            "detected_bugs": detected_bugs,
            "bug_trace_records": bug_trace_records,
        }

    print(
        "[Bug detection] Review the complete trace independently: "
        f"recordable_steps={len(trace)} (each step includes its pre-action screenshot; executor conversation is not reused)"
    )
    detection_messages = _build_full_trace_messages(state, trace)

    result = invoke_llm_json_with_retry(
        llm,
        detection_messages,
        log_prefix="[Bug detection]",
        json_block_pattern=JSON_BLOCK_PATTERN,
    )

    if llm_json_result_failed(result):
        raise DefectReviewError(getattr(result, "failure_reason", "Model review failed"))
    bugs_patch = result.get("bugs")
    if not isinstance(bugs_patch, list) or any(not isinstance(bug, dict) for bug in bugs_patch):
        raise DefectReviewError("Bug-review response lacks a valid bugs array")
    new_bugs = []
    for candidate in bugs_patch:
        normalized = _normalize_bug(candidate, scenario_for_bug, bug_context)
        reason = ""
        if not normalized["description"] or normalized["type"] not in FUNCTIONAL_BUG_TYPES:
            reason = "invalid_description_or_type"
        elif normalized["evidence_category"] not in DIRECT_EVIDENCE_CATEGORIES:
            reason = "unsupported_evidence_category"
        elif normalized["confidence"] != "high":
            reason = "insufficient_confidence"
        elif not _bug_has_required_trace_evidence(normalized, trace):
            reason = "incomplete_or_unordered_evidence"
        elif _bug_uses_speculative_or_automation_expectation(normalized, scenario_for_bug):
            reason = "unsupported_expectation"
        if reason:
            print("[Finding filter] " + json.dumps(
                {"reason": reason, "candidate": normalized}, ensure_ascii=False))
        else:
            new_bugs.append(normalized)
    if len(new_bugs) != len(bugs_patch):
        print(
            "[Bug detection] Filtered invalid finding candidates: "
            f"accepted={len(new_bugs)}, candidates={len(bugs_patch)}"
        )

    current_bugs = runtime_bugs + new_bugs
    detected_bugs = _ensure_bug_ids(_dedupe_bugs(existing_bugs + current_bugs))
    bug_trace_records = _update_bug_trace_records(
        state,
        detected_bugs,
        current_bugs,
        _current_bug_trace(state),
    )
    print(
        "[Bug detection] Round result: "
        f"runtime={len(runtime_bugs)}, llm={len(new_bugs)}, total={len(detected_bugs)}"
    )
    return {
        "detected_bugs": detected_bugs,
        "bug_trace_records": bug_trace_records,
    }
