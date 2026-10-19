"""Unified scenario-planning node for the Functional Flow Graph (FFG)."""

import hashlib
import json
import os
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from model.flow_graph import (
    FunctionalFlowGraph,
    FunctionalityNode,
    Page,
    PageTransition,
    TERMINAL_FUNCTION_STATUSES,
)
from tools.actions import ScreenshotUnavailableError, capture_current_page
from compliance.checkpoints import ComplianceChecklist, build_compliance_scenarios
from compliance.rule_library import GROUP_LABELS, gui_rules
from workflow.llm import invoke_llm_json_with_retry, llm_json_result_failed
from workflow.nodes._shared import (
    JSON_BLOCK_PATTERN,
    has_exhaustive_control_family_intent,
    screenshot_similarity,
)
from workflow.app_foreground import (
    APP_RESTART_RETRY_DELAY,
    APP_RESTART_SETTLE_DELAY,  # noqa: F401 - retain module exports for callers and tests
    APP_START_MAX_WAIT,
    APP_START_POLL_INTERVAL,  # noqa: F401 - retain module exports for callers and tests
    restart_until_foreground,
    wait_target_app_foreground,
)
from workflow.exploration_boundary import rule_for
from workflow.nodes.ffg_update import apply_ffg_patch, is_black_screen_page_name
from workflow.nodes.prompt import (
    BOOTSTRAP_FFG_SYSTEM_PROMPT,
    EXPLORATION_MODE_INSTRUCTIONS,
    FFG_AUDIT_SYSTEM_PROMPT,
    UNIFIED_SCENARIO_PLAN_SYSTEM_PROMPT,
)
from workflow.state import AgentState
from workflow.system_dialogs import is_permission_dialog
from workflow.wechat_safety import (
    wechat_action_policy_violation,
    wechat_read_only_violation,
)
from workflow.account_safety import account_session_action_violation

FUNCTION_ID_PATTERN = re.compile(r"\bfunc_\d+\b")
PAGE_ID_PATTERN = re.compile(r"\bpage_\d+\b")
PAGE_ID_ANNOTATION_PATTERN = re.compile(r"\bpage_\d+（[^）]*）")
TEXT_NORMALIZE_PATTERN = re.compile(r"[\"'“”‘’（）()【】\[\]{}<>《》、，,。.：:；;!！?？\-_/\s]+")
MAX_AUDIT_IMAGES_PER_FUNCTION = int(os.getenv("PLAN_MAX_AUDIT_IMAGES_PER_FUNCTION", "4"))
MAX_COMPLETION_AUDITS_PER_FUNCTION = max(
    1,
    int(os.getenv("PLAN_MAX_COMPLETION_AUDITS_PER_FUNCTION", "20")),
)
# Limit completed-functionality audits per planning round; each audit may issue
# several multimodal requests. Leave remaining functionalities to later rounds.
PLAN_MAX_AUDITS_PER_ROUND = max(1, int(os.getenv("PLAN_MAX_AUDITS_PER_ROUND", "2")))
MAX_CROSS_TEST_PAIRS = int(os.getenv("PLAN_MAX_CROSS_TEST_PAIRS", "20"))
# Semantic retries multiply the transport/parsing retries inside
# invoke_llm_json_with_retry, so the outer default is only two attempts.
PLAN_VALIDATION_RETRY_ATTEMPTS = max(1, int(os.getenv("PLAN_VALIDATION_RETRY_ATTEMPTS", "2")))
# Show only the latest K tested scenarios per functionality in scope summaries.
# Summarize older scenarios by count to keep planning prompts bounded.
PLAN_SCOPE_RECENT_SCENARIOS = max(1, int(os.getenv("PLAN_SCOPE_RECENT_SCENARIOS", "20")))
# Limit related page transitions in the summary, preferring transitions within scope.
PLAN_SCOPE_MAX_TRANSITIONS = max(1, int(os.getenv("PLAN_SCOPE_MAX_TRANSITIONS", "40")))
# Number of widget families per supplementary scenario. Planning, FFG updates,
# screenshots, and logcat have fixed per-round costs; batching related widgets
# avoids spending that overhead on only one or two useful actions.
PLAN_SCENARIO_CONTROL_BATCH = max(
    1,
    int(os.getenv("PLAN_SCENARIO_CONTROL_BATCH", "4")),
)
# Program-derived widget coverage required for completion and coverage-gap auditing.
PLAN_FUNCTION_COVERAGE_TARGET = max(
    1,
    min(100, int(os.getenv("PLAN_FUNCTION_COVERAGE_TARGET", "80"))),
)
# Intra-functional exploration first covers entries, then tests complete business scenarios.
# Bound breadth attempts so a large functionality cannot postpone depth indefinitely.
PLAN_BREADTH_MAX_SCENARIOS = max(
    1,
    int(os.getenv("PLAN_BREADTH_MAX_SCENARIOS", "3")),
)
# Require one depth scenario per PLAN_DEPTH_CONTROLS_PER_SCENARIO widgets,
# bounded by PLAN_MIN_DEPTH_SCENARIOS and PLAN_MAX_DEPTH_SCENARIOS.
PLAN_DEPTH_CONTROLS_PER_SCENARIO = max(
    1,
    int(os.getenv("PLAN_DEPTH_CONTROLS_PER_SCENARIO", "5")),
)
PLAN_MIN_DEPTH_SCENARIOS = max(0, int(os.getenv("PLAN_MIN_DEPTH_SCENARIOS", "1")))
PLAN_MAX_DEPTH_SCENARIOS = max(
    PLAN_MIN_DEPTH_SCENARIOS,
    int(os.getenv("PLAN_MAX_DEPTH_SCENARIOS", "6")),
)
# Scenario cap = breadth budget + depth target + margin. At the cap, complete
# only if the evidence gate passes; otherwise defer to avoid monopolizing execution.
PLAN_SCENARIO_CAP_MARGIN = max(0, int(os.getenv("PLAN_SCENARIO_CAP_MARGIN", "2")))
# Maximum attempts per retryable inter-functional candidate (navigation/screenshot failures).
PLAN_MAX_CROSS_ATTEMPTS = max(1, int(os.getenv("PLAN_MAX_CROSS_ATTEMPTS", "2")))
# Defer only after repeated empty plans from both the model and fallback templates.
PLAN_UNPLANNABLE_ROUNDS_BEFORE_DEFER = max(
    1, int(os.getenv("PLAN_UNPLANNABLE_ROUNDS_BEFORE_DEFER", "2"))
)
FUNCTIONALITY_PHASE_BREADTH = "breadth"
FUNCTIONALITY_PHASE_DEPTH = "depth"
# Navigation levels describe reference paths; they do not gate scheduling.
# Empty: already in scope; A: recorded forward edges; B: reverse edges as back;
# C: restart and use A/B; D: no recorded path, so the executor explores from
# the target description and defers after a navigation failure.
NAV_LEVEL_HERE = ""
NAV_LEVEL_FORWARD = "A"
NAV_LEVEL_BACKTRACK = "B"
NAV_LEVEL_RESTART = "C"
NAV_LEVEL_EXPLORE = "D"
KNOWN_NAV_LEVELS = {NAV_LEVEL_HERE, NAV_LEVEL_FORWARD, NAV_LEVEL_BACKTRACK, NAV_LEVEL_RESTART}
# Only the core exploration scope is supported by EXPLORATION_MODE_INSTRUCTIONS.
PLAN_EXPLORATION_MODE = "core"
# Match English output and legacy Chinese entry labels without changing UI data.
ENTRY_FUNCTION_NAME_PATTERN = re.compile(r"(?:\u5165\u53e3|\bentry)\s*$", re.IGNORECASE)

# Multilingual input vocabulary: escaped Chinese labels preserve app matching.
COMPLIANCE_GROUP_DISCOVERY_TERMS: dict[str, tuple[str, ...]] = {
    "first_launch_flow": ("\u9690\u79c1", "\u534f\u8bae", "\u540c\u610f", "\u62d2\u7edd", "\u5f00\u5c4f", "privacy", "agreement", "consent", "agree", "reject", "splash"),
    "permission_denial_flow": ("\u6743\u9650", "\u76f8\u673a", "\u5b9a\u4f4d", "\u4f4d\u7f6e", "\u901a\u8baf\u5f55", "\u76f8\u518c", "\u5b58\u50a8", "\u9ea6\u514b\u98ce", "permission", "camera", "location", "contacts", "album", "storage", "microphone"),
    "consent_observation": ("\u654f\u611f", "\u4eba\u8138", "\u4f4d\u7f6e", "\u901a\u8baf\u5f55", "\u5065\u5eb7", "\u652f\u4ed8", "\u4f1a\u5458", "\u8ba2\u9605", "\u7eed\u8d39", "\u9752\u5c11\u5e74", "sensitive", "face", "location", "contacts", "health", "payment", "membership", "subscription", "renewal", "teen"),
    "registration_form": ("\u767b\u5f55", "\u6ce8\u518c", "\u8d26\u53f7", "\u624b\u673a\u53f7", "\u8d44\u6599", "\u5b9e\u540d\u8ba4\u8bc1", "\u4e2a\u4eba\u4fe1\u606f", "login", "register", "account", "phone", "profile", "identity", "personal information"),
    "settings_search": ("\u8bbe\u7f6e", "\u9690\u79c1", "\u63a8\u8350", "\u63a8\u9001", "\u6743\u9650", "\u81ea\u542f\u52a8", "\u8bbe\u5907\u6807\u8bc6", "\u8d44\u8d39", "settings", "privacy", "recommendation", "notification", "permission", "autostart", "device identifier", "fee"),
    "rights_entry_search": ("\u8bbe\u7f6e", "\u8d26\u53f7", "\u6ce8\u9500", "\u5220\u9664", "\u66f4\u6b63", "\u64a4\u56de", "\u5ba2\u670d", "\u6295\u8bc9", "\u53cd\u9988", "\u5e2e\u52a9", "settings", "account", "deletion", "delete", "correction", "withdraw", "service", "complaint", "feedback", "help"),
    "policy_text_review": ("\u9690\u79c1\u653f\u7b56", "\u9690\u79c1", "\u7528\u6237\u534f\u8bae", "\u4e2a\u4eba\u4fe1\u606f", "\u7b2c\u4e09\u65b9", "\u5171\u4eab\u6e05\u5355", "privacy policy", "privacy", "user agreement", "personal information", "third party", "sharing"),
}
COMPLIANCE_CLOSE_LOOP_WAIT_SECONDS = max(
    0.0,
    float(os.getenv("COMPLIANCE_CLOSE_LOOP_WAIT_SECONDS", "10") or 10),
)
PLAN_APP_RESTART_ATTEMPTS = max(
    1,
    int(os.getenv("PLAN_APP_RESTART_ATTEMPTS", os.getenv("APP_RESTART_ATTEMPTS", "2"))),
)
PLAN_SCREENSHOT_BACK_ATTEMPTS = max(
    1,
    int(os.getenv("PLAN_SCREENSHOT_BACK_ATTEMPTS", os.getenv("APP_RECOVERY_BACK_ATTEMPTS", "3"))),
)
# Share APP_RESTART_RETRY_DELAY, APP_START_*, and APP_RESTART_SETTLE_DELAY
# with the executor via workflow/app_foreground.py. Poll the foreground package
# after restarting so slow app startup cannot be mistaken for the launcher.
MAX_CONSECUTIVE_SCREENSHOT_UNAVAILABLE = max(1, int(os.getenv("MAX_CONSECUTIVE_SCREENSHOT_UNAVAILABLE", "2")))
MAX_CONSECUTIVE_TARGET_APP_UNAVAILABLE = max(
    1,
    int(os.getenv("MAX_CONSECUTIVE_TARGET_APP_UNAVAILABLE", "3")),
)
STOP_AFTER_SCREENSHOT_UNAVAILABLE = os.getenv(
    "STOP_AFTER_SCREENSHOT_UNAVAILABLE", "0"
).strip().lower() in {"1", "true", "yes", "on"}
STOP_AFTER_TARGET_APP_UNAVAILABLE = os.getenv(
    "STOP_AFTER_TARGET_APP_UNAVAILABLE", "0"
).strip().lower() in {"1", "true", "yes", "on"}
CURRENT_PAGE_MATCH_SIMILARITY = float(os.getenv("PLAN_CURRENT_PAGE_MATCH_SIMILARITY", "0.90"))
PLAN_SUMMARY_FIELD_MAX_CHARS = max(120, int(os.getenv("PLAN_SUMMARY_FIELD_MAX_CHARS", "600")))
PLAN_REQUIRED_FIELDS = {"test_scenario"}
AUDIT_REQUIRED_FIELDS = {"entry_page_ref", "entry_name"}


def _planner_foreground_probe(device, package_name: str) -> tuple[str, str]:
    """Probe the foreground once; treat a failed read as background without retries or logs."""
    try:
        current = str(device.current_package() or "").strip()
    except Exception:
        current = ""
    return ("foreground" if current == package_name else "background"), current


def _wait_target_app_foreground(device, package_name: str) -> bool:
    """Wait at most APP_START_MAX_WAIT seconds for the target app to enter the foreground."""
    return wait_target_app_foreground(
        device,
        package_name,
        max_wait=APP_START_MAX_WAIT,
        state_probe=_planner_foreground_probe,
    )


def _ensure_target_app(device, package_name: str, current_page: Page | None = None) -> Page | None:
    package_name = package_name.strip()
    if not package_name:
        return current_page

    try:
        current_package = str(device.current_package() or "").strip()
    except Exception as exc:
        print(f"[Scenario planning] Failed to read the foreground package: {exc}")
        current_package = ""

    if is_permission_dialog(current_package):
        print("[Scenario planning] A permission dialog covers the target app; preserve the screenshot for the executor.")
        return capture_current_page(device)

    needs_reset = False
    if current_package and current_package != package_name:
        print(
            f"[Scenario planning] Foreground app is {current_package}; target is {package_name}. "
            "Restarting the target app."
        )
        needs_reset = True
    elif current_page is None and current_package == package_name:
        print("[Scenario planning] No current page, but the target app is foreground; capturing directly.")
        try:
            return capture_current_page(device)
        except ScreenshotUnavailableError as exc:
            print(f"[Scenario planning] Foreground screenshot unavailable; trying back navigation first: {exc}")
            for attempt in range(1, PLAN_SCREENSHOT_BACK_ATTEMPTS + 1):
                try:
                    print(
                        "[Scenario planning] Black-screen recovery: "
                        f"back attempt {attempt}/{PLAN_SCREENSHOT_BACK_ATTEMPTS}."
                    )
                    device.back()
                    time.sleep(APP_RESTART_RETRY_DELAY)
                    recovered_package = str(device.current_package() or "").strip()
                    if recovered_package != package_name:
                        print(
                            "[Scenario planning] Target app left the foreground after back; switching to restart recovery: "
                            f"current={recovered_package or 'unknown'}"
                        )
                        break
                    return capture_current_page(device)
                except ScreenshotUnavailableError as back_exc:
                    print(f"[Scenario planning] Screenshot still unavailable after back: {back_exc}")
                except Exception as back_exc:
                    print(f"[Scenario planning] Back-navigation recovery failed: {back_exc}")
            needs_reset = True
    elif current_page is None:
        print("[Scenario planning] Target foreground state is unconfirmed; restarting the target app.")
        needs_reset = True

    if not needs_reset:
        return current_page

    restarted, page, errors = restart_until_foreground(
        device,
        package_name,
        wait_foreground=lambda dev, pkg: _wait_target_app_foreground(dev, pkg),
        on_ready=lambda: capture_current_page(device),
        attempts=PLAN_APP_RESTART_ATTEMPTS,
        log_prefix="[Scenario planning]",
    )
    if restarted:
        return page
    screenshot_errors = [exc for exc in errors if isinstance(exc, ScreenshotUnavailableError)]
    if screenshot_errors:
        raise screenshot_errors[-1]
    return None


def _safe_capture_current_page(device) -> Page | None:
    try:
        return capture_current_page(device)
    except ScreenshotUnavailableError:
        raise
    except Exception as exc:
        print(f"[Scenario planning] Failed to capture the current page: {exc}")
        return None


def _clip_summary_text(value: object, limit: int = PLAN_SUMMARY_FIELD_MAX_CHARS) -> str:
    text = _clean_natural_text(value)
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 1)] + "…"


def _normalize_run_phase(value: object) -> str:
    text = str(value or "all").strip().lower()
    if text == "intra":
        return "intra"
    return "all"


def _count_ffg_stats(graph: FunctionalFlowGraph) -> dict[str, int]:
    active_pages = graph.active_pages()
    page_count = len(active_pages)
    operation_count = sum(len(page.operations) for page in active_pages)
    intra_scenario_count = sum(_node_attempt_count(node) for node in graph.functions.values())
    return {
        "functions": len(graph.functions),
        "pages": page_count,
        "operations": operation_count,
        "page_transitions": len(graph.page_transitions),
        "function_dependencies": len(graph.function_dependencies),
        "intra_scenarios": intra_scenario_count,
        "cross_scenarios": sum(
            1
            for node in graph.functions.values()
            for scenario in node.scenarios
            if scenario.startswith("[cross:")
        ) // 2,
    }


def _node_is_terminal(node: FunctionalityNode | None) -> bool:
    """True for any terminal status (completed/blocked/deferred/unavailable).

    Terminal functions leave the active scheduling queue and never get planned
    again in this run; only `completed` is additionally eligible for the
    coverage-gap audit.
    """
    if node is None:
        return True
    return (node.test_status or "not_started") in TERMINAL_FUNCTION_STATUSES


def _node_attempt_count(node: FunctionalityNode | None) -> int:
    if node is None:
        return 0
    return len(node.scenarios)


def _node_cross_testable(node: FunctionalityNode | None) -> bool:
    """Accept inter-functional endpoints only when they are not blocked, unavailable, or deferred."""
    if node is None:
        return False
    return (node.test_status or "not_started") not in {
        "blocked",
        "deferred",
        "unavailable",
    }


def _node_intra_attempt_count(node: FunctionalityNode | None) -> int:
    """Actual test attempts; deduplicated scenario count is a lower bound for legacy snapshots."""
    return node.intra_attempts() if node is not None else 0


def _node_depth_count(node: FunctionalityNode | None) -> int:
    """Depth scenarios that entered the test phase, including failed test attempts."""
    return int(node.depth_scenario_count or 0) if node is not None else 0


def _node_breadth_count(node: FunctionalityNode | None) -> int:
    return max(0, _node_intra_attempt_count(node) - _node_depth_count(node))


def _required_depth_scenario_count(graph: FunctionalFlowGraph, node: FunctionalityNode | None) -> int:
    """Require one depth scenario per PLAN_DEPTH_CONTROLS_PER_SCENARIO widgets."""
    if node is None:
        return PLAN_MIN_DEPTH_SCENARIOS
    controls = len(graph.function_controls(node.id))
    required = -(-controls // PLAN_DEPTH_CONTROLS_PER_SCENARIO)
    return max(PLAN_MIN_DEPTH_SCENARIOS, min(PLAN_MAX_DEPTH_SCENARIOS, required))


def _function_scenario_cap(graph: FunctionalFlowGraph, node: FunctionalityNode | None) -> int:
    """Scenario cap = breadth budget + depth target + margin; it grows with widget count."""
    return (
        PLAN_BREADTH_MAX_SCENARIOS
        + _required_depth_scenario_count(graph, node)
        + PLAN_SCENARIO_CAP_MARGIN
    )


def _intra_functionality_phase(graph: FunctionalFlowGraph, node: FunctionalityNode | None) -> str:
    """Current intra-functional phase: entry breadth or complete business-scenario depth.

    Use depth while its target is unmet and either breadth is exhausted or every recorded widget is covered.
    Otherwise use breadth; after sufficient depth, return to any remaining widget-coverage gaps.
    """
    if node is None:
        return FUNCTIONALITY_PHASE_BREADTH
    if node.successful_depth_scenario_count >= _required_depth_scenario_count(graph, node):
        return FUNCTIONALITY_PHASE_BREADTH
    if _node_breadth_count(node) >= PLAN_BREADTH_MAX_SCENARIOS:
        # The attempt budget bounds breadth even when widget coverage keeps growing.
        return FUNCTIONALITY_PHASE_DEPTH
    details = graph.function_coverage_details(node.id)
    if int(details["total_controls"] or 0) > 0 and not details["uncovered_controls"]:
        return FUNCTIONALITY_PHASE_DEPTH
    return FUNCTIONALITY_PHASE_BREADTH


def _describe_intra_functionality_phase(graph: FunctionalFlowGraph, node: FunctionalityNode | None) -> str:
    """Phase and budget summary for both the planning model and runtime diagnostics."""
    if node is None:
        return "No target functionality"
    phase = _intra_functionality_phase(graph, node)
    uncovered = len(graph.function_coverage_details(node.id)["uncovered_controls"])
    return (
        f"{phase} | breadth {_node_breadth_count(node)}/{PLAN_BREADTH_MAX_SCENARIOS}"
        f" | depth {_node_depth_count(node)}/{_required_depth_scenario_count(graph, node)}"
        f" (observed completions {node.successful_depth_scenario_count})"
        f" | uncovered widgets {uncovered}"
        f" | scenarios {_node_intra_attempt_count(node)}/{_function_scenario_cap(graph, node)}"
    )


def _cross_pair_ready(graph: FunctionalFlowGraph, src_id: str, dst_id: str) -> bool:
    """Both endpoints need their own attempts and successful business operations before flow testing."""
    return all(
        graph.get_node(node_id) is not None
        and graph.function_has_successful_business_operation(node_id)
        and _node_intra_attempt_count(graph.get_node(node_id)) > 0
        for node_id in (src_id, dst_id)
    )


def _function_completion_met(graph: FunctionalFlowGraph, node: FunctionalityNode) -> bool:
    """Require sufficient widget coverage, a business operation, and successful depth scenarios.

    Shared pages may contribute coverage to sibling functionalities. Depth completions,
    however, are counted only for the functionality explicitly targeted by the depth scenario.
    """
    return (
        node.coverage_percent >= PLAN_FUNCTION_COVERAGE_TARGET
        and graph.function_has_successful_business_operation(node.id)
        and node.successful_depth_scenario_count >= _required_depth_scenario_count(graph, node)
    )


def _update_function_statuses(graph: FunctionalFlowGraph) -> None:
    """Apply FFG-derived status transitions once per round before scheduling."""
    reopened = _normalize_and_reopen_undercovered_functions(graph)
    if reopened:
        print("[Scenario planning] Reopened completed nodes for derived coverage/depth targets: " + ", ".join(reopened))
    graph.recompute_all_coverages()
    completed: list[str] = []
    capped: list[str] = []
    for node in list(graph.functions.values()):
        status = node.test_status or "not_started"
        # Deferred means the completion gate is unmet; later evidence may still complete it.
        if status in TERMINAL_FUNCTION_STATUSES and status != "deferred":
            continue
        if _function_completion_met(graph, node):
            graph.set_node_test_status(node.id, "completed", reason="completion_met")
            completed.append(node.id)
        elif (
            status != "deferred"
            and _node_intra_attempt_count(node) >= _function_scenario_cap(graph, node)
        ):
            graph.set_node_test_status(node.id, "deferred", reason="scenario_cap")
            capped.append(node.id)
    if completed:
        print(
            "[Scenario planning] Completion gate passed; marked completed, pending coverage-gap audit: "
            + ", ".join(f"{fid} | {_function_label(graph.get_node(fid))}" for fid in completed)
        )
    if capped:
        print(
            "[Scenario planning] Scenario cap reached without completion evidence; marked deferred: "
            + ", ".join(f"{fid} | {_function_label(graph.get_node(fid))}" for fid in capped)
        )


def _normalize_and_reopen_undercovered_functions(graph: FunctionalFlowGraph) -> list[str]:
    """Normalize entry names and reopen completed nodes below derived coverage."""
    reopened: list[str] = []
    for node in graph.functions.values():
        name = str(node.name or "").strip()
        if ENTRY_FUNCTION_NAME_PATTERN.search(name):
            business_name = ENTRY_FUNCTION_NAME_PATTERN.sub("", name).strip(" -_/：:")
            if business_name:
                node.name = business_name
        graph.recompute_function_coverage(node.id)
        coverage_details = graph.function_coverage_details(node.id)
        has_uncovered = bool(coverage_details["uncovered_controls"])
        # A planned functionality with no widget or operation records has no auditable
        # coverage basis and must be revisited. Exclude unplanned nodes so truly new
        # functionalities retain their place in breadth exploration.
        never_observed = (
            bool(node.scenarios)
            and int(coverage_details["total_controls"] or 0) == 0
            and not graph.function_executed_operation_keys(node.id)
        )
        # Reopening must follow derived coverage: unhandled widgets or no recorded
        # observations. Unrelated heuristics such as entry naming and page count can
        # oscillate against completion and waste planning rounds.
        needs_reopen = (
            (has_uncovered or never_observed)
            and node.coverage_percent < PLAN_FUNCTION_COVERAGE_TARGET
        )
        # Legacy completed snapshots also need observed depth completions, not just attempts.
        needs_more_depth = node.successful_depth_scenario_count < _required_depth_scenario_count(graph, node)
        # At the scenario cap, defer remaining entries to the coverage-gap audit.
        if (
            (node.test_status or "not_started") != "completed"
            or not (needs_reopen or needs_more_depth)
        ):
            continue
        capped = _node_intra_attempt_count(node) >= _function_scenario_cap(graph, node)
        graph.set_node_test_status(
            node.id, "deferred" if capped else "in_progress",
            reason="scenario_cap" if capped else "coverage_reopened",
        )
        reopened.append(node.id)
    return reopened


def _unfinished_function_nodes(graph: FunctionalFlowGraph) -> list[FunctionalityNode]:
    return [node for node in graph.functions.values() if not _node_is_terminal(node)]


def _function_sort_index(node: FunctionalityNode) -> int:
    match = re.search(r"\d+", str(node.id or ""))
    return int(match.group(0)) if match else 10**9


def _intra_sort_key(
    graph: FunctionalFlowGraph,
    node: FunctionalityNode,
    current_owner_ids: list[str],
    failure_counts: dict[str, int],
    navigation_level: str = NAV_LEVEL_HERE,
) -> tuple:
    """Deterministic ordering for intra-functional targets.

    Prefer no recent failure, then depth, current-page ownership, known navigation,
    no previous attempts, fewer attempts, and finally functionality ID.
    """
    return (
        0 if int(failure_counts.get(node.id, 0) or 0) == 0 else 1,
        0 if _intra_functionality_phase(graph, node) == FUNCTIONALITY_PHASE_DEPTH else 1,
        0 if node.id in current_owner_ids else 1,
        0 if navigation_level in KNOWN_NAV_LEVELS else 1,
        0 if not _node_intra_attempt_count(node) else 1,
        _node_intra_attempt_count(node),
        _function_sort_index(node),
    )


def _cross_pair_key(src_function_id: str, dst_function_id: str) -> tuple[str, str]:
    return (str(src_function_id or "").strip(), str(dst_function_id or "").strip())


def _cross_state_variable(reason: str) -> str:
    return str(reason or "").partition(": ")[2] or str(reason or "").strip()


def _add_cross_candidate(
    candidates: list[tuple[str, str, str]],
    seen: set[tuple[str, str, str]],
    graph: FunctionalFlowGraph,
    src_function_id: str,
    dst_function_id: str,
    reason: str,
):
    src_function_id, dst_function_id = _cross_pair_key(src_function_id, dst_function_id)
    if not src_function_id or not dst_function_id or src_function_id == dst_function_id:
        return
    if graph.get_node(src_function_id) is None or graph.get_node(dst_function_id) is None:
        return
    key = graph.cross_test_key(src_function_id, dst_function_id, _cross_state_variable(reason))
    if key in seen:
        return
    seen.add(key)
    candidates.append((src_function_id, dst_function_id, reason))


def _candidate_inter_functional_pairs(graph: FunctionalFlowGraph) -> list[tuple[str, str, str]]:
    candidates: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    for dependency in graph.function_dependencies:
        dependency_state_variable = _clean_natural_text(dependency.resource_variable) or "Functional dependency"
        _add_cross_candidate(
            candidates,
            seen,
            graph,
            dependency.src_function_id,
            dependency.dst_function_id,
            f"Functional dependency: {dependency_state_variable}",
        )

    # Without an explicit resource direction, either endpoint may produce the data.
    # Generate both directions, but retain only explicit directions when available.
    state_variable_owners: dict[str, list[tuple[FunctionalityNode, str]]] = {}
    for functionality in sorted(graph.functions.values(), key=_function_sort_index):
        seen_state_variables: set[str] = set()
        for raw_state_variable in functionality.resource_variables:
            state_variable = _clean_natural_text(raw_state_variable)
            state_variable_key = _normalize_similarity_key(state_variable)
            if not state_variable_key or state_variable_key in seen_state_variables:
                continue
            seen_state_variables.add(state_variable_key)
            state_variable_owners.setdefault(state_variable_key, []).append((functionality, state_variable))

    for functionality_owners in state_variable_owners.values():
        if len(functionality_owners) < 2:
            continue
        for src_index, (src_node, state_variable) in enumerate(functionality_owners[:-1]):
            for dst_node, _ in functionality_owners[src_index + 1:]:
                # Explicit producer/consumer directions take precedence for this
                # resource only. Do not guess a reverse direction for shared data.
                if (
                    graph.cross_test_key(src_node.id, dst_node.id, state_variable) in seen
                    or graph.cross_test_key(dst_node.id, src_node.id, state_variable) in seen
                ):
                    continue
                for first, second in ((src_node, dst_node), (dst_node, src_node)):
                    _add_cross_candidate(
                        candidates,
                        seen,
                        graph,
                        first.id,
                        second.id,
                        f"Shared state variable: {state_variable}",
                    )

    return candidates


def _cross_pair_exhausted_keys(graph: FunctionalFlowGraph) -> set[tuple[str, str, str]]:
    """Candidates exhausted by completion, non-retryable failure, or the retry budget."""
    exhausted: set[tuple[str, str, str]] = set()
    for item in graph.cross_test_results:
        key = graph.cross_test_key(item["src_function_id"], item["dst_function_id"], item["resource_variable"])
        if (
            item["status"] == "done"
            or not item.get("retryable", False)
            or int(item.get("attempts", 1) or 1) >= PLAN_MAX_CROSS_ATTEMPTS
        ):
            exhausted.add(key)
    return exhausted


def _pending_cross_pairs(graph: FunctionalFlowGraph) -> list[tuple[str, str, str]]:
    # Remove completed and terminal failures; bound retries for navigation/screenshot failures.
    processed = _cross_pair_exhausted_keys(graph)
    return [
        (src_id, dst_id, reason)
        for src_id, dst_id, reason in _candidate_inter_functional_pairs(graph)
        if graph.cross_test_key(src_id, dst_id, _cross_state_variable(reason)) not in processed
        and _node_cross_testable(graph.get_node(src_id))
        and _node_cross_testable(graph.get_node(dst_id))
    ][:MAX_CROSS_TEST_PAIRS]


def _function_label(node) -> str:
    if node is None:
        return "Unknown functionality"
    return _clean_natural_text(node.name or node.description) or "Unnamed functionality"


def _page_display_name(page: Page | None) -> str:
    """Human-readable page name for planning: name, then description, with internal IDs removed."""
    if page is None:
        return "Unknown page"
    return _clean_natural_text(page.name or page.description) or "Unnamed page"


def _page_node_label(page: Page | None) -> str:
    if page is None:
        return "Unknown page"
    return _clean_natural_text(page.name) or str(page.id or "").strip() or "Unnamed page"


def _is_black_screen_page(
    graph: FunctionalFlowGraph,
    current_page: Page | None,
    current_page_id: str | None = None,
) -> bool:
    page_id = (
        _resolve_current_page_id(graph, current_page)
        if current_page_id is None else current_page_id
    )
    page = graph.get_page(page_id) if page_id else None
    return bool(page and is_black_screen_page_name(page.name))


def _matches_black_screen_path(graph: FunctionalFlowGraph, scenario: str) -> bool:
    scenario_key = _normalize_similarity_key(scenario)
    if not scenario_key:
        return False
    black_page_ids = {
        str(page.id or "")
        for page in graph.active_pages()
        if is_black_screen_page_name(page.name)
    }
    for transition in graph.page_transitions:
        if transition.dst_page_id not in black_page_ids:
            continue
        failed_action = re.split(
            r"[；;]?\s*(?:\u7ed3\u679c|[Rr]esult)[:：]",
            str(transition.operation or ""),
            maxsplit=1,
        )[0].strip()
        action_key = _normalize_similarity_key(failed_action)
        if action_key and action_key in scenario_key:
            return True
    return False


def _resolve_current_page_id(graph: FunctionalFlowGraph, current_page: Page | None) -> str:
    if not isinstance(current_page, Page):
        return ""
    page_id = str(current_page.id or "").strip()
    if page_id and graph.get_page(page_id) is not None:
        return graph.resolve_page_id(page_id)
    if current_page.representative_screenshot:
        for page in graph.active_pages():
            if current_page.representative_screenshot in [page.representative_screenshot, *page.evidence_screenshots]:
                return str(page.id or "")
    activity = str(current_page.activity or "").strip()
    candidates = [
        page
        for page in graph.active_pages()
        if not activity or str(page.activity or "").strip() == activity
    ]
    if current_page.representative_screenshot and candidates:
        scored: list[tuple[float, str]] = []
        for page in candidates:
            similarities = [
                similarity
                for stored in [page.representative_screenshot, *page.evidence_screenshots]
                if (similarity := screenshot_similarity(current_page.representative_screenshot, stored)) is not None
            ]
            if similarities and page.id:
                scored.append((max(similarities), str(page.id)))
        if scored:
            similarity, matched_page_id = max(scored)
            if similarity >= CURRENT_PAGE_MATCH_SIMILARITY:
                return matched_page_id
        # A screenshot is stronger evidence than an Activity name.  A single
        # Activity often hosts many logical pages; do not force an unmatched
        # live frame onto the only Page discovered so far.
        return ""
    if len(candidates) == 1:
        return str(candidates[0].id or "")
    return ""


def _attach_current_page_identity_with_id(
    graph: FunctionalFlowGraph,
    current_page: Page | None,
) -> tuple[Page | None, str]:
    """Align a runtime observation with a recorded Page and return its ID, or an empty ID."""
    if not isinstance(current_page, Page):
        return current_page, ""
    page_id = _resolve_current_page_id(graph, current_page)
    recorded_page = graph.get_page(page_id)
    if recorded_page is None:
        return current_page, ""
    current_page.id = recorded_page.id
    current_page.name = recorded_page.name
    current_page.description = recorded_page.description
    current_page.controls = list(recorded_page.controls)
    return current_page, str(recorded_page.id or "")


def _attach_current_page_identity(graph: FunctionalFlowGraph, current_page: Page | None) -> Page | None:
    return _attach_current_page_identity_with_id(graph, current_page)[0]


def _refresh_current_page_observation(
    graph: FunctionalFlowGraph,
    device,
    current_page: Page | None,
) -> Page | None:
    """Refresh persisted planner state from the device before scheduling.

    A Page id is an assertion from the previous round, not proof that the UI is
    still there.  Capture without that id so visual/activity matching can detect
    app-internal returns, delayed navigation, and other state changes.
    """
    try:
        observed = capture_current_page(device)
    except Exception as exc:
        # Refresh is a consistency check, not the primary screenshot-health
        # gate. Keep the last usable state when a transient capture fails.
        print(f"[Scenario planning] Pre-execution page refresh failed; retaining previous state: {exc}")
        return current_page
    if not isinstance(observed, Page):
        return current_page
    previous_id = str(current_page.id or "").strip() if isinstance(current_page, Page) else ""
    observed = _attach_current_page_identity(graph, observed)
    observed_id = str(observed.id or "").strip()
    if previous_id and observed_id != previous_id:
        print(
            "[Scenario planning] Pre-execution refresh detected a changed page: "
            f"state={previous_id}, observed={observed_id or 'unmatched new page'}"
        )
    return observed


def _build_current_location_summary(
    graph: FunctionalFlowGraph,
    current_page: Page | None,
    current_page_id: str | None = None,
) -> str:
    page_id = (
        _resolve_current_page_id(graph, current_page)
        if current_page_id is None else current_page_id
    )
    page = graph.get_page(page_id) if page_id else None
    owner_ids = graph.get_page_owners(page_id) if page_id else []

    lines = ["[Current position]"]
    if owner_ids:
        owner_labels = [
            f"{owner_id} | {_function_label(graph.get_node(owner_id))}"
            for owner_id in owner_ids
        ]
        lines.append(f"- Candidate current functionality owners: {'; '.join(owner_labels)}")
    else:
        lines.append("- Candidate current functionality owners: no recorded functionality matched")

    if page is not None:
        lines.append(f"- Current page node: {page.id} | {_page_node_label(page)}")
        lines.append(f"- Current page owners: {owner_ids}")
    else:
        lines.append("- Current page node: no recorded page matched")
        if isinstance(current_page, Page) and current_page.activity:
            lines.append(f"- Current Activity: {_clean_natural_text(current_page.activity)}")

    return "\n".join(lines)


def _strip_internal_ids_from_text(text: str) -> str:
    if not text:
        return text
    text = PAGE_ID_ANNOTATION_PATTERN.sub("", text)
    text = PAGE_ID_PATTERN.sub("", text)
    text = FUNCTION_ID_PATTERN.sub("", text)
    text = re.sub(r"（\s*）", "", text)
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def _clean_natural_text(value: object) -> str:
    text = str(value or "").strip()
    if text.lower() == "none":
        return ""
    return _strip_internal_ids_from_text(text)


def _normalize_similarity_key(value: object) -> str:
    return TEXT_NORMALIZE_PATTERN.sub("", str(value or "").strip().lower())


def _normalize_optional_id(value: object) -> str:
    text = str(value or "").strip()
    if text.lower() in {"none", "null"}:
        return ""
    return text


def _sanitize_planned_scenario(text: str) -> str:
    text = _strip_internal_ids_from_text(text)
    text = re.sub(r"\u8fdb\u5165\s*\u5bf9\u5e94\u7684\s*\u9875\u9762", "\u8fdb\u5165\u76ee\u6807\u9875\u9762", text)
    text = re.sub(r"\u5728\u5f53\u524d\s*\u9875\u9762", "\u5728\u5f53\u524d\u9875\u9762", text)
    text = re.sub(r"\s+，", "，", text)
    text = re.sub(r"\s+。", "。", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def _invoke_llm(llm, system_prompt: str, message_content: list[dict]) -> dict:
    return invoke_llm_json_with_retry(
        llm,
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=message_content),
        ],
        log_prefix="[Scenario planning]",
        json_block_pattern=JSON_BLOCK_PATTERN,
    )


def _validate_plan_result(
    result: object,
    scenario_type: str = "intra_test",
    package_name: str = "",
) -> list[str]:
    if not isinstance(result, dict) or not result:
        return ["empty_or_non_object_result"]

    errors: list[str] = []
    missing_fields = sorted(PLAN_REQUIRED_FIELDS - set(result))
    if missing_fields:
        errors.append(f"missing_fields={missing_fields}")
    test_scenario = _normalize_optional_id(result.get("test_scenario"))
    risky_intent = wechat_read_only_violation(package_name, test_scenario)
    if risky_intent:
        errors.append(f"wechat_read_only_policy={risky_intent!r}")
    if scenario_type == "cross_test" and not test_scenario:
        errors.append("cross_test_requires_nonempty_test_scenario")
    if has_exhaustive_control_family_intent(test_scenario):
        errors.append("test_scenario_exhaustive_control_family")
    return errors


def _invoke_valid_plan(
    llm,
    system_prompt: str,
    message_content: list[dict],
    scenario_type: str = "intra_test",
    package_name: str = "",
) -> tuple[dict, list[str]]:
    last_errors: list[str] = []
    current_message_content = list(message_content)
    for attempt in range(1, PLAN_VALIDATION_RETRY_ATTEMPTS + 1):
        result = _invoke_llm(llm, system_prompt, current_message_content)
        if llm_json_result_failed(result):
            last_errors = ["llm_request_or_parse_failed"]
            print("[Scenario planning] Underlying request retries exhausted; using a fallback scenario.")
            break
        errors = _validate_plan_result(
            result,
            scenario_type=scenario_type,
            package_name=package_name,
        )
        if not errors:
            return result, []
        last_errors = errors
        print(
            "[Scenario planning] Invalid fields returned by the LLM"
            f" (attempt {attempt}/{PLAN_VALIDATION_RETRY_ATTEMPTS}): {errors}"
        )
        current_message_content = list(message_content)
        current_message_content.append({
            "type": "text",
            "text": (
                'The previous output was invalid. Return {"test_scenario": "..."} again. '
                "For duplicate widgets in one functionality with the same outcome, select one representative. "
                "You may name 2-3 parallel widgets with distinct outcomes precisely. "
                "Do not request all, every, or exhaustive traversal of a widget family. "
                f"Field errors: {errors}"
            ),
        })
    return {}, last_errors


def _skip_response(
    graph: FunctionalFlowGraph,
    current_page: Page | None,
    target_function_id: str = "",
    *,
    testing_function_id: str = "",
    testing_function_status: str = "not_started",
    workflow_complete: bool = False,
    **extra,
) -> dict:
    """Common response for rounds without execution: initialization, status updates, waits, or finalization."""
    return {
        "ffg": graph,
        "current_page": current_page,
        "current_navigation_scenario": "",
        "current_test_scenario": "",
        "current_testing_function_id": testing_function_id,
        "current_testing_function_status": testing_function_status,
        "current_plan": {
            "scenario_type": "skip",
            "target_function_id": target_function_id,
            "src_function_id": "",
            "dst_function_id": "",
        },
        "workflow_complete": workflow_complete,
        **extra,
    }


def _normalize_failure_counts(value: object) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    normalized: dict[str, int] = {}
    for key, count in value.items():
        try:
            normalized[str(key)] = max(0, int(count or 0))
        except (TypeError, ValueError):
            continue
    return normalized


def _node_representative_page(graph: FunctionalFlowGraph, node: FunctionalityNode | None) -> Page | None:
    if node is None:
        return None
    page_ids = node.coverage_page_ids or node.page_ids
    for page_id in page_ids:
        page = graph.get_page(page_id)
        if page is not None:
            return page
    return None


def _node_uncovered_control_hints(
    graph: FunctionalFlowGraph,
    node: FunctionalityNode | None,
    limit: int = 4,
) -> list[str]:
    """Return visible controls with no directly matching success/failure operation."""
    if node is None:
        return []
    controls = graph.function_controls(node.id)
    handled = graph.function_handled_controls(node.id)
    allowed_page_ids = set(node.coverage_page_ids)
    hints: list[str] = []
    for control_ref, label in controls.items():
        page_id, _control_key = control_ref
        if page_id not in allowed_page_ids or control_ref in handled:
            continue
        text = _clean_natural_text(label)
        if text and text not in hints:
            hints.append(text)
        if len(hints) >= limit:
            break
    return hints


def _fallback_intra_functionality_depth_scenario(graph: FunctionalFlowGraph, node: FunctionalityNode) -> str:
    """Generate a generic business-scenario fallback for the depth phase.

    Describe completion of a business operation without hard-coding which operation.
    The executor selects it from the observed interface; text-based business rule
    tables generalize poorly and can select inappropriate targets in other apps.
    """
    # No functionality page means no execution anchor. Return empty so the existing
    # repeated-unplannable path can defer the functionality instead of trapping it in depth.
    if not node.coverage_page_ids:
        return ""
    label = _function_label(node)
    representative_page = _node_representative_page(graph, node)
    entry_label = (
        _page_display_name(representative_page) if representative_page is not None else "the recorded functionality page"
    )
    hints = _node_uncovered_control_hints(graph, node, limit=3)
    hint_text = (
        "Prefer relevant uncovered visible widgets: " + ", ".join(f'"{hint}"' for hint in hints) + ". "
        if hints
        else ""
    )
    return (
        f'Execute one complete business operation for "{label}": first try to enter "{entry_label}" and capture the visible state. '
        "Then choose and complete an operation chain on the recorded widgets of this functionality with an observable result "
        "(for example, create and save, edit an existing item, remove an item, or change a toggle or filter). "
        "After the operation, leave the page and return to inspect the result. "
        f"{hint_text}"
        "Do not repeat scenarios already executed for this functionality. Wait for a stable screen before each screenshot; use only recorded pages and widgets. "
        "Operate only on self-created data identified by a FuncDroid_ suffix. Do not delete accounts, log out, switch accounts, pay, order, or perform identity verification. "
        "If the entry is unreachable or the operation cannot be performed, record the reason and stop."
    )


def _forced_intra_scenario(
    graph: FunctionalFlowGraph,
    node: FunctionalityNode,
    hint_index: int = 0,
    package_name: str = "",
) -> str:
    if _intra_functionality_phase(graph, node) == FUNCTIONALITY_PHASE_DEPTH:
        # Prefer a depth scenario. An empty result would release the functionality as
        # unplannable, so fall back to breadth below when the depth template cannot run.
        depth_scenario = _fallback_intra_functionality_depth_scenario(graph, node)
        if depth_scenario:
            print(
                "[Scenario planning] phase=depth; generated a business scenario: "
                f"{node.id} | {_function_label(node)}"
            )
            return depth_scenario
    label = _function_label(node)
    representative_page = _node_representative_page(graph, node)
    entry_label = _page_display_name(representative_page) if representative_page is not None else "the recorded functionality page"
    hints = _node_uncovered_control_hints(
        graph,
        node,
    )
    if str(package_name or "").strip() == "com.tencent.mm":
        safe_hints: list[str] = []
        for hint in hints:
            violation = wechat_action_policy_violation(
                package_name,
                "click",
                f"Click {hint}",
            )
            if violation:
                print(
                    "[Scenario planning] Account-session protection skipped a logout/account-deletion widget: "
                    f"{hint} | {violation}"
                )
                continue
            safe_hints.append(hint)
        hints = safe_hints
    if not hints:
        if not node.coverage_page_ids:
            return ""
        # Observe recorded pages with no widget records before concluding that nothing is testable.
        return (
            f'Observe the interactive content of "{label}": try to enter "{entry_label}" from the current page, '
            "wait for the interface to stabilize, and capture visible buttons, inputs, list items, toggles, and tabs. "
            "If scrollable, scroll down one screen and capture it. Select a low-risk widget such as a filter, tab, or "
            "details expander, click once, and observe its direct result; return if a new screen opens. "
            "Do not delete, pay, log out, or perform other irreversible operations. If unreachable, record the reason and stop."
        )
    # Testing one widget per round leaves only one or two useful actions while
    # planning, multimodal FFG updates, screenshots, and logcat retain fixed costs.
    # Batch related widget families to cover testable entries on the same page.
    batch: list[str] = []
    for offset in range(min(PLAN_SCENARIO_CONTROL_BATCH, len(hints))):
        candidate = hints[(hint_index + offset) % len(hints)]
        if candidate not in batch:
            batch.append(candidate)
    hint_text = ", ".join(f'"{hint}"' for hint in batch)
    if len(batch) == 1:
        coverage_clause = (
            f"Explore this visible widget with no execution record ({hint_text}) and observe its direct result. "
        )
    else:
        coverage_clause = (
            f"Explore these {len(batch)} visible widgets with no execution records in order ({hint_text}). "
            "After each click observe its direct result; if a new screen opens, return before the next widget. "
        )
    return (
        f'Perform a low-risk supplementary exploration of "{label}": try to enter "{entry_label}" from the current page. '
        + coverage_clause
        + "Do not repeat widgets already covered by operation records. "
        "If the entry is unreachable, no testable content remains, or only historical paths can be repeated, record the reason and stop."
    )


def _fallback_inter_functional_scenario(graph: FunctionalFlowGraph, src_id: str, dst_id: str, reason: str) -> str:
    src_label = _function_label(graph.get_node(src_id))
    dst_label = _function_label(graph.get_node(dst_id))
    reason = _clean_natural_text(reason) or "an existing functional dependency"
    data_values = [item["value"] for item in graph.function_test_data(src_id)][-3:]
    data_clause = (
        "Test data previously created by the source functionality: " + ", ".join(f'"{value}"' for value in data_values)
        + ". Prefer finding these data in the target functionality. "
        if data_values
        else f"If the source functionality has no usable data, create an item with the unique name FuncDroid_{src_id}. "
    )
    return (
        f'Test the inter-functional interaction from "{src_label}" to "{dst_label}": first confirm the recorded resource, '
        f'state, or result associated with "{reason}" in the source. {data_clause}Then enter the target and check that it reads, displays, or consumes that resource, state, or result correctly. '
        "If the path is unreachable or preconditions are unmet, record the reason and stop."
    )


def _test_data_instruction(
    graph: FunctionalFlowGraph,
    scenario_type: str,
    target: FunctionalityNode | None,
    src: FunctionalityNode | None,
) -> str:
    """Stable per-functionality data suffixes (FuncDroid_<func_id>) support reuse across rounds and flows."""
    owner = src if scenario_type == "cross_test" else target
    owner_id = owner.id if owner is not None else "data"
    text = f"[Test data for this round] Use the unique suffix FuncDroid_{owner_id} when creating or naming resources. "
    existing = [item["value"] for item in graph.function_test_data(owner_id)][-5:]
    if existing:
        text += "Previously created test data: " + ", ".join(existing) + ". Prefer reusing these data in depth scenarios and inter-functional reads."
    return text


def _release_unplannable_function(
    graph: FunctionalFlowGraph,
    current_page: Page | None,
    node: FunctionalityNode,
    reason: str,
    unplannable_counts: dict[str, int] | None = None,
) -> dict:
    """Defer only after repeated rounds where neither model nor templates yield a scenario.

    An isolated empty plan may be a transient model failure or an incomplete page record.
    Count it and skip the round; defer at PLAN_UNPLANNABLE_ROUNDS_BEFORE_DEFER.
    Deferred functionalities retain a final retry and may complete once sufficient evidence arrives.
    """
    counts = dict(unplannable_counts or {})
    counts[node.id] = int(counts.get(node.id, 0) or 0) + 1
    if counts[node.id] < PLAN_UNPLANNABLE_ROUNDS_BEFORE_DEFER:
        print(
            f"[Scenario planning] {reason}; skipping this round ({counts[node.id]}/"
            f"{PLAN_UNPLANNABLE_ROUNDS_BEFORE_DEFER}): {node.id} | {_function_label(node)}"
        )
        return {**_skip_response(graph, current_page, node.id), "unplannable_counts": counts}
    print(f"[Scenario planning] {reason}; repeatedly unplannable, deferring to release scheduling: {node.id} | {_function_label(node)}")
    if not _node_is_terminal(node):
        graph.set_node_test_status(node.id, "deferred", reason="unplannable")
    counts.pop(node.id, None)
    return {**_skip_response(graph, current_page, node.id), "unplannable_counts": counts}


def _bootstrap_initial_ffg(
    llm,
    graph: FunctionalFlowGraph,
    current_page: Page,
    app_name: str = "",
    package_name: str = "",
) -> dict:
    """Build an initial FFG from the first screenshot."""
    message_content: list[dict] = [{
        "type": "text",
        "text": f"Target app: {app_name or package_name or 'unknown'} | package={package_name or 'unknown'}",
    }]
    if current_page.representative_screenshot:
        message_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{current_page.representative_screenshot}"},
        })
    else:
        message_content.append({
            "type": "text",
            "text": 'No page screenshot is available; return {"functions": [], "pages": []}.',
        })

    result = _invoke_llm(llm, BOOTSTRAP_FFG_SYSTEM_PROMPT, message_content)
    # Initialization has a single screenshot; assign evidence step 1 programmatically.
    bootstrap_patch = {
        key: [
            {**item, "source_step": 1}
            for item in (result.get(key, []) if isinstance(result, dict) else [])
            if isinstance(item, dict)
        ]
        for key in ("functions", "pages")
    }
    trace = [{"current_page": current_page, "raw_step_index": 1}]
    patch_result = apply_ffg_patch(graph, bootstrap_patch, trace, current_plan={})
    if not graph.functions:
        print("[Scenario planning] Initialization patch produced no functionality; applying a minimal current-page fallback.")
        fallback_patch = {
            "functions": [
                {
                    "function_ref": "new_func_1",
                    "source_step": 1,
                    "name": _page_display_name(current_page) or "Current-page navigation",
                    "description": current_page.description or "Current page and visible entries observed during initialization.",
                }
            ],
            "pages": [
                {
                    "page_ref": "new_page_1",
                    "function_refs": ["new_func_1"],
                    "source_step": 1,
                    "name": _page_display_name(current_page) or "Current page",
                    "description": current_page.description or "Current page captured in the initialization screenshot.",
                }
            ],
        }
        patch_result = apply_ffg_patch(graph, fallback_patch, trace, current_plan={})

    for page in graph.active_pages():
        owner_ids = graph.get_page_owners(str(page.id or ""))
        for owner_id in owner_ids[1:]:
            graph.add_entry_page(page, owner_id)
    graph.recompute_all_coverages()

    first_page = next(iter(graph.active_pages()), current_page)
    print(
        "[Scenario planning] FFG initialization complete: "
        f"functions={len(graph.functions)}, page={_page_display_name(first_page)}, "
        f"operations={sum(len(page.operations) for page in graph.active_pages())}, "
        f"patch_functions={len(patch_result.get('functions_patch', []))}, "
        f"patch_pages={len(patch_result.get('pages_patch', []))}"
    )
    return patch_result


def _build_function_audit_summary(
    graph: FunctionalFlowGraph,
    node: FunctionalityNode,
    exploration_mode: str,
    app_name: str = "",
    package_name: str = "",
) -> str:
    boundary_hint = rule_for(package_name).to_prompt_hint()
    lines: list[str] = [
        "[Coverage-gap audit]",
        f"Target app: {app_name or package_name or 'unknown'} | package={package_name or 'unknown'}",
        EXPLORATION_MODE_INSTRUCTIONS[PLAN_EXPLORATION_MODE],
        boundary_hint,
        "",
        f"Functionality: {node.id} | {_function_label(node)}",
    ]
    if node.description:
        lines.append(f"Description: {_clean_natural_text(node.description)}")
    lines.append(f"Test status: {node.test_status or 'not_started'}")
    lines.append(f"Coverage score: {node.coverage_percent}%")
    coverage_details = graph.function_coverage_details(node.id)
    lines.append(
        "Widgets/operations: "
        f"{coverage_details['executed_operations']}/{coverage_details['total_controls']}"
    )
    if node.resource_variables:
        lines.append(f"State variables: {', '.join(_clean_natural_text(item) for item in node.resource_variables)}")
    lines.append("")
    lines.append("Recorded pages, widgets, and operations:")
    for page_id in node.coverage_page_ids:
        page = graph.get_page(page_id)
        if page is None:
            continue
        lines.append(f"- Page: {page.id} | {_page_display_name(page)}")
        if page.description:
            lines.append(f"  Description: {_clean_natural_text(page.description)}")
        if page.controls:
            lines.append("  All recorded widgets across screenshot states:")
            for control in page.controls:
                lines.append(f"    * {_clean_natural_text(control)}")
        if page.operations:
            lines.append("  Executed/known operations:")
            for operation in page.operations:
                lines.append(f"    * {_clean_natural_text(operation)}")
        else:
            lines.append("  Executed/known operations: none")
    transitions = graph.get_function_page_transitions(node.id)
    if transitions:
        lines.append("")
        lines.append("Page transitions:")
        for transition in transitions:
            src_page = graph.get_page(transition.src_page_id)
            dst_page = graph.get_page(transition.dst_page_id)
            lines.append(
                f'- From "{_page_display_name(src_page)}" via "{_clean_natural_text(transition.operation)}" to "{_page_display_name(dst_page)}"'
            )
    if node.scenarios:
        lines.append("")
        lines.append("Tested scenarios:")
        for scenario in node.scenarios:
            lines.append(f"- {_clean_natural_text(scenario)}")
    else:
        lines.append("")
        lines.append("Tested scenarios: none")
    return "\n".join(lines)


def _audit_functionality_completeness(
    llm,
    graph: FunctionalFlowGraph,
    node: FunctionalityNode,
    exploration_mode: str,
    validation_feedback: str = "",
    app_name: str = "",
    package_name: str = "",
) -> dict:
    pages_with_images = [
        page
        for page_id in node.coverage_page_ids
        if (page := graph.get_page(page_id)) is not None
        and page.representative_screenshot
    ]
    batch_size = max(1, MAX_AUDIT_IMAGES_PER_FUNCTION)
    page_batches = [
        pages_with_images[index:index + batch_size]
        for index in range(0, len(pages_with_images), batch_size)
    ] or [[]]
    last_result: dict = {}
    for batch_index, page_batch in enumerate(page_batches, start=1):
        message_content: list[dict] = [
            {
                "type": "text",
                "text": _build_function_audit_summary(
                    graph,
                    node,
                    exploration_mode,
                    app_name=app_name,
                    package_name=package_name,
                ),
            },
            {
                "type": "text",
                "text": (
                    f"[Batched visual review] Batch {batch_index}/{len(page_batches)}. "
                    "Look for visible omissions only in the attached screenshots. Return false if this batch has none; "
                    "do not infer anything about pages not pictured. Remaining batches will be checked separately."
                ),
            },
        ]
        if validation_feedback:
            message_content.append({"type": "text", "text": validation_feedback})
        for page in page_batch:
            message_content.append({
                "type": "text",
                "text": f"Screenshot page: {page.id} | {_page_display_name(page)}",
            })
            message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{page.representative_screenshot}"},
            })

        print(
            "[Scenario planning] Coverage-gap audit: "
            f"{_function_label(node)} | batch={batch_index}/{len(page_batches)}, "
            f"pages={len(node.coverage_page_ids)}, screenshots={len(page_batch)}"
        )
        result = _invoke_llm(llm, FFG_AUDIT_SYSTEM_PROMPT, message_content)
        last_result = result if isinstance(result, dict) else {}
        # Any gap must be executed before later batches. Invalid output is
        # returned to the protocol-retry path instead of being hidden.
        if _validate_audit_result(last_result) or _audit_has_gap(last_result):
            return last_result
    return last_result


def _audit_has_gap(result: dict) -> bool:
    return bool(
        _normalize_optional_id(result.get("entry_page_ref"))
        or _clean_natural_text(result.get("entry_name"))
    )


def _validate_audit_result(result: object) -> list[str]:
    """Both fields must be empty (no gap) or nonempty (one executable entry)."""
    if not isinstance(result, dict) or not result:
        return ["empty_or_non_object_result"]
    errors: list[str] = []
    missing_fields = sorted(AUDIT_REQUIRED_FIELDS - set(result))
    if missing_fields:
        errors.append(f"missing_fields={missing_fields}")
    entry_page_ref = _normalize_optional_id(result.get("entry_page_ref"))
    entry_name = _clean_natural_text(result.get("entry_name"))
    if bool(entry_page_ref) != bool(entry_name):
        errors.append("entry_page_ref_and_entry_name_must_both_be_set_or_empty")
    return errors


def _audit_entry_scenario(page: Page | None, entry_name: str) -> str:
    return (
        f'On "{_page_display_name(page)}", click "{entry_name}", '
        "observe and record the resulting page, dialog, or state change, then stop."
    )


def _function_audit_signature(graph: FunctionalFlowGraph, node: FunctionalityNode) -> str:
    """Structural coverage signature; only page/widget/transition changes trigger a new audit.

    Exclude scenario and operation counts because an audit execution adds both.
    Including them would change the signature after every audit scenario and cause
    repeated audits up to MAX_COMPLETION_AUDITS_PER_FUNCTION even when no new
    pages or widgets were found.
    """
    coverage_page_ids = set(node.coverage_page_ids)
    page_ids = ",".join(sorted(str(page_id) for page_id in coverage_page_ids))
    transition_count = sum(
        1
        for transition in graph.page_transitions
        if transition.src_page_id in coverage_page_ids
        and transition.dst_page_id in coverage_page_ids
    )
    control_count = 0
    for page_id in coverage_page_ids:
        page = graph.get_page(page_id)
        if page is not None:
            control_count += len(page.controls)
    return (
        f"status={node.test_status or 'not_started'};"
        f"pages={page_ids};transitions={transition_count};"
        f"controls={control_count}"
    )


def _run_completion_audit(
    llm,
    graph: FunctionalFlowGraph,
    exploration_mode: str,
    previous_signatures: dict[str, str] | None = None,
    audit_attempts: dict[str, int] | None = None,
    app_name: str = "",
    package_name: str = "",
    deadline: float | None = None,
    max_audits: int | None = None,
) -> tuple[dict | None, bool, dict[str, str]]:
    """Audit completed functionalities and return (entry, all_done, audited_signatures).

    Record invalid model output or unknown pages as no executable gap to avoid
    blocking finalization. Stop auditing when the run deadline is reached.
    Audit at most max_audits functionalities per round (PLAN_MAX_AUDITS_PER_ROUND
    by default), leaving remaining audits to later rounds with all_done=False.
    """
    previous_signatures = previous_signatures or {}
    audit_attempts = audit_attempts or {}
    audited_signatures: dict[str, str] = {}
    audit_budget = PLAN_MAX_AUDITS_PER_ROUND if max_audits is None else max(1, int(max_audits))
    audits_this_round = 0
    for node in graph.functions.values():
        if (node.test_status or "not_started") != "completed":
            continue
        if deadline is not None and time.time() >= deadline:
            print("[Scenario planning] Run deadline reached; stopping the coverage-gap audit.")
            return None, False, audited_signatures
        signature = _function_audit_signature(graph, node)
        if previous_signatures.get(node.id) == signature:
            audited_signatures[node.id] = signature
            print(f"[Scenario planning] Coverage-gap audit skipped unchanged functionality: {_function_label(node)}")
            continue
        if int(audit_attempts.get(node.id, 0) or 0) >= MAX_COMPLETION_AUDITS_PER_FUNCTION:
            audited_signatures[node.id] = signature
            print(
                "[Scenario planning] Per-functionality audit limit reached; closing this signature: "
                f"{_function_label(node)} | {MAX_COMPLETION_AUDITS_PER_FUNCTION}"
            )
            continue
        if audits_this_round >= audit_budget:
            print(
                "[Scenario planning] Per-round audit limit reached; deferring remaining functionalities: "
                f"{audits_this_round}/{audit_budget}"
            )
            return None, False, audited_signatures
        audits_this_round += 1
        validation_feedback = ""
        for _ in range(2):
            result = _audit_functionality_completeness(
                llm,
                graph,
                node,
                exploration_mode,
                validation_feedback=validation_feedback,
                app_name=app_name,
                package_name=package_name,
            )
            protocol_errors = _validate_audit_result(result)
            result_dict = result if isinstance(result, dict) else {}
            entry_page_ref = _normalize_optional_id(result_dict.get("entry_page_ref"))
            entry_name = _clean_natural_text(result_dict.get("entry_name"))
            has_gap = bool(entry_page_ref and entry_name)
            valid_entry_page = (
                entry_page_ref in node.page_ids
                and graph.get_page(entry_page_ref) is not None
            )
            if not (protocol_errors or (has_gap and not valid_entry_page)):
                break
            validation_feedback = (
                "The previous output violated the coverage-audit protocol. With no gap, return empty entry_page_ref and entry_name. "
                "With a gap, both must be nonempty: entry_page_ref must be an existing page_x of this functionality in the summary, "
                "and entry_name must identify an unexecuted entry visibly present in that page screenshot or description. "
                f"Field errors: {protocol_errors or ['entry_page_ref_not_in_function']}"
            )
        if protocol_errors or (has_gap and not valid_entry_page):
            print(
                "[Scenario planning] Audit repeatedly returned non-executable results; recording no executable gap for this signature: "
                f"{_function_label(node)} | "
                f"{protocol_errors or f'page={entry_page_ref} | entry={entry_name}'}"
            )
            audited_signatures[node.id] = signature
            continue
        action_policy_violation = ""
        if has_gap:
            # Protect account sessions regardless of package: ignore logout/account-switch
            # candidates so visiting a login form cannot invalidate later authenticated evidence.
            action_policy_violation = wechat_action_policy_violation(
                package_name, "click", f"Click {entry_name}"
            ) or account_session_action_violation("click", f"Click {entry_name}")
        if action_policy_violation:
            print(
                "[Scenario planning] Ignored a coverage-gap candidate involving logout or account deletion: "
                f"{_function_label(node)} | entry={entry_name} | "
                f"reason={action_policy_violation}"
            )
        elif has_gap:
            test_scenario = _audit_entry_scenario(graph.get_page(entry_page_ref), entry_name)
            print(
                "[Scenario planning] Audit found an entry to verify; executing it before updating the FFG: "
                f"{node.id} | page={entry_page_ref} | scenario={test_scenario}"
            )
            return {
                "owner_function_id": node.id,
                "entry_page_id": entry_page_ref,
                "test_scenario": test_scenario,
            }, False, audited_signatures
        else:
            print(f"[Scenario planning] Coverage-gap audit passed: {_function_label(node)}")
        audited_signatures[node.id] = signature
    return None, True, audited_signatures


def _page_path_to_entry(
    graph: FunctionalFlowGraph,
    node: FunctionalityNode,
    start_page_id: str,
    entry_page_id: str,
    allow_backtrack: bool = False,
) -> list[tuple[PageTransition, bool]]:
    """Find a path to the entry page using only pages owned by this functionality."""
    return _leveled_page_path(
        graph,
        start_page_id,
        [entry_page_id],
        allow_backtrack=allow_backtrack,
        allowed_page_ids=set(node.page_ids),
    )


def _build_audit_navigation_scenario(
    graph: FunctionalFlowGraph,
    current_page: Page | None,
    owner_function_id: str,
    entry_page_id: str,
) -> str:
    node = graph.get_node(owner_function_id)
    entry_page_id = graph.resolve_page_id(entry_page_id) or entry_page_id
    entry_page = graph.get_page(entry_page_id)
    if node is None or entry_page is None:
        return ""

    current_page_id = ""
    if current_page is not None and current_page.id:
        current_page_id = graph.resolve_page_id(str(current_page.id)) or str(current_page.id)
    if current_page_id == entry_page_id:
        return ""

    # Level A uses recorded forward edges; level B also permits reverse edges as back.
    start_page_id = current_page_id
    path = _page_path_to_entry(graph, node, start_page_id, entry_page_id)
    if not path:
        path = _page_path_to_entry(graph, node, start_page_id, entry_page_id, allow_backtrack=True)
    starts_from_current = bool(path)
    if not starts_from_current:
        first_page_id = node.page_ids[0] if node.page_ids else ""
        start_page_id = graph.resolve_page_id(first_page_id) or first_page_id
        if not start_page_id or graph.get_page(start_page_id) is None:
            start_page_id = entry_page_id
        path = _page_path_to_entry(graph, node, start_page_id, entry_page_id)

    instructions: list[str] = []
    if not starts_from_current:
        start_page = graph.get_page(start_page_id)
        current_is_owned_child = bool(
            current_page_id
            and owner_function_id in graph.get_page_owners(current_page_id)
        )
        if current_is_owned_child:
            instructions.append(
                "The current page belongs to the target functionality but has no recorded forward path. If it is clearly a "
                f'child page, selector, or overlay opened from the entry, try back once and capture whether it returns to "{_page_display_name(entry_page)}". '
                "If the entry is not reached or the target app is left, use open_app once to open its main page, "
                "then follow the recorded path below"
            )
        else:
            instructions.append(
                "The current position is outside known navigation paths to the target. First use open_app once to open the app main page. "
                "If still not there, close temporary menus or dialogs that were already open when navigation began, and safely go back to "
                f'the recorded page "{_page_display_name(start_page)}". If back leaves the app, use open_app again'
            )
    instructions.extend(_path_step_instructions(graph, path))
    if not path and start_page_id != entry_page_id:
        instructions.append(f'Continue safe back navigation until reaching the recorded page "{_page_display_name(entry_page)}"')
    instructions.append(
        f'Capture confirmation that "{_page_display_name(entry_page)}" is reached, then end navigation. '
        "Do not click the entry under test during navigation. Menus opened deliberately to reach the target are part of the path; do not close them again"
    )
    return "; ".join(instructions) + "."


def _audit_exploration_response(
    graph: FunctionalFlowGraph,
    current_page: Page | None,
    audit_result: dict,
    audit_signatures: dict[str, str],
    audit_attempts: dict[str, int] | None = None,
) -> dict:
    owner_function_id = str(audit_result.get("owner_function_id", "")).strip()
    entry_page_id = str(audit_result.get("entry_page_id", "")).strip()
    audit_scenario = str(audit_result.get("test_scenario", "")).strip()
    navigation_scenario = _build_audit_navigation_scenario(
        graph,
        current_page,
        owner_function_id,
        entry_page_id,
    )
    next_attempts = dict(audit_attempts or {})
    next_attempts[owner_function_id] = int(next_attempts.get(owner_function_id, 0) or 0) + 1
    print(
        "[Scenario planning] Coverage-gap audit scheduled evidence-driven execution: "
        f"owner={owner_function_id}, scenario={audit_scenario}"
    )
    return {
        "ffg": graph,
        "current_page": current_page,
        "current_navigation_scenario": navigation_scenario,
        "current_test_scenario": audit_scenario,
        "current_testing_function_id": owner_function_id,
        # An audit is evidence collection for an already-completed owner.  Keep
        # the persisted status completed so an interruption/resume cannot turn
        # it back into an ordinary unfinished function.
        "current_testing_function_status": "completed",
        "completion_audit_signatures": audit_signatures,
        "completion_audit_attempts": next_attempts,
        "current_plan": {
            "scenario_type": "intra_test",
            "target_function_id": owner_function_id,
            "src_function_id": "",
            "dst_function_id": "",
            "audit_exploration": True,
        },
        "workflow_complete": False,
    }


def _select_intra_target(
    graph: FunctionalFlowGraph,
    current_page: Page | None,
    failure_counts: dict[str, int] | None = None,
    compliance_terms: tuple[str, ...] = (),
    current_page_id: str | None = None,
) -> FunctionalityNode | None:
    """Select the first unfinished functionality under the deterministic ordering.

    Navigation is a reference, not a candidate gate. Without a known path, use
    level-D exploratory navigation, ranked after candidates with known paths.
    """
    failure_counts = failure_counts or {}
    # Resolve current-page identity and build adjacency only once; all candidate
    # navigation-level calculations reuse them instead of scanning screenshots again.
    if current_page_id is None:
        current_page_id = _resolve_current_page_id(graph, current_page)
    current_owner_ids = graph.get_page_owners(current_page_id)
    candidates = _unfinished_function_nodes(graph)
    adjacency = _page_adjacency(graph)
    levels = {
        node.id: _navigation_to_function(
            graph, current_page, node, current_page_id, adjacency,
        )[2]
        for node in candidates
    }
    candidates.sort(
        key=lambda node: _intra_sort_key(
            graph, node, current_owner_ids, failure_counts, levels[node.id],
        )
    )
    # Pending compliance targets prioritize functionalities with relevant pages or
    # widgets so ordinary exploration supplies new paths and evidence for reassessment.
    if compliance_terms:
        relevant = [
            node for node in candidates
            if _function_compliance_score(graph, node, compliance_terms) > 0
        ]
        # Recently failed relevant targets must not outrank healthy unrelated ones.
        # max preserves the existing deterministic ordering when relevance scores tie.
        healthy = [node for node in relevant if not failure_counts.get(node.id)]
        if healthy:
            return max(
                healthy,
                key=lambda node: _function_compliance_score(graph, node, compliance_terms),
            )
    return candidates[0] if candidates else None


def _run_deadline(state: AgentState) -> float | None:
    """Absolute run deadline, or None when no total time limit is configured."""
    limit = float(state.get("total_time_limit", 0) or 0)
    start = float(state.get("start_time", 0) or 0)
    return start + limit if limit > 0 and start > 0 else None


def _select_deferred_retry(
    graph: FunctionalFlowGraph,
    current_page: Page | None,
    retried_ids: list[str],
    current_page_id: str | None = None,
) -> FunctionalityNode | None:
    """Give each deferred functionality one final retry, restricted to known A/B/C paths.

    Level-D navigation already failed; retrying without a new path repeats the same exploration.
    """
    if current_page_id is None:
        current_page_id = _resolve_current_page_id(graph, current_page)
    adjacency = _page_adjacency(graph)
    for node in sorted(graph.functions.values(), key=_function_sort_index):
        if (node.test_status or "") != "deferred" or node.id in retried_ids:
            continue
        level = _navigation_to_function(graph, current_page, node, current_page_id, adjacency)[2]
        if level in KNOWN_NAV_LEVELS:
            return node
    return None


@dataclass
class _PageAdjacency:
    """Forward/reverse page adjacency, built once per planning round for repeated BFS calls."""

    forward: dict[str, list[PageTransition]] = field(default_factory=dict)
    reverse: dict[str, list[PageTransition]] = field(default_factory=dict)


def _page_adjacency(graph: FunctionalFlowGraph) -> _PageAdjacency:
    adjacency = _PageAdjacency()
    # Preserve page_transitions order so BFS edge traversal matches the previous linear scan.
    for transition in graph.page_transitions:
        adjacency.forward.setdefault(transition.src_page_id, []).append(transition)
        adjacency.reverse.setdefault(transition.dst_page_id, []).append(transition)
    return adjacency


def _leveled_page_path(
    graph: FunctionalFlowGraph,
    start_page_id: str,
    target_page_ids: list[str],
    allow_backtrack: bool = False,
    allowed_page_ids: set[str] | None = None,
    adjacency: _PageAdjacency | None = None,
) -> list[tuple[PageTransition, bool]]:
    """Shortest page path by BFS, with each step represented as (transition, reverse).

    With allow_backtrack, a recorded src->dst edge also proposes dst->src as back.
    This is only a reference; the executor must confirm the actual result by screenshot.
    Reuse caller-supplied adjacency, otherwise build it once within this call.
    """
    start_page_id = graph.resolve_page_id(start_page_id)
    targets = {
        resolved
        for page_id in target_page_ids
        if (resolved := graph.resolve_page_id(page_id))
    }
    if not start_page_id or not targets or start_page_id in targets:
        return []
    if adjacency is None:
        adjacency = _page_adjacency(graph)
    queue: deque[tuple[str, list[tuple[PageTransition, bool]]]] = deque([(start_page_id, [])])
    visited = {start_page_id}
    while queue:
        page_id, path = queue.popleft()
        edges = [
            (transition, transition.dst_page_id, False)
            for transition in adjacency.forward.get(page_id, ())
        ]
        if allow_backtrack:
            edges.extend(
                (transition, transition.src_page_id, True)
                for transition in adjacency.reverse.get(page_id, ())
            )
        for transition, next_page_id, reversed_edge in edges:
            if next_page_id in visited:
                continue
            if allowed_page_ids is not None and next_page_id not in allowed_page_ids:
                continue
            next_path = [*path, (transition, reversed_edge)]
            if next_page_id in targets:
                return next_path
            visited.add(next_page_id)
            queue.append((next_page_id, next_path))
    return []


def _shortest_page_path(
    graph: FunctionalFlowGraph,
    start_page_id: str,
    target_page_ids: list[str],
    adjacency: _PageAdjacency | None = None,
) -> list:
    return [
        transition
        for transition, _ in _leveled_page_path(
            graph, start_page_id, target_page_ids, adjacency=adjacency,
        )
    ]


def _root_page(graph: FunctionalFlowGraph) -> Page | None:
    """App entry page: the first functionality's representative page, expected after open_app."""
    return _node_representative_page(graph, next(iter(graph.functions.values()), None))


def _route_via_restart(
    graph: FunctionalFlowGraph,
    current_page_id: str,
    target_page_ids: list[str],
    adjacency: _PageAdjacency,
    allow_backtrack: bool,
) -> tuple[list[tuple[PageTransition, bool]], bool]:
    """Level-C fallback: reopen the app, then follow recorded paths from its entry page.

    Return (path, needs_restart). If the entry is the target, restart alone is sufficient.
    """
    root_page = _root_page(graph)
    root_page_id = str(root_page.id or "") if root_page is not None else ""
    if root_page_id in target_page_ids:
        return [], current_page_id != root_page_id
    if not root_page_id:
        return [], False
    path = _leveled_page_path(graph, root_page_id, target_page_ids, adjacency=adjacency)
    if not path and allow_backtrack:
        path = _leveled_page_path(
            graph, root_page_id, target_page_ids, allow_backtrack=True, adjacency=adjacency,
        )
    return path, bool(path)


# Level-C explicitly requests reopening, as permitted by the executor's open_app rule.
RESTART_NAVIGATION_INSTRUCTION = "Execute open_app to reopen the target app and wait for a stable screen"


def _path_step_instructions(
    graph: FunctionalFlowGraph,
    path: list[tuple[PageTransition, bool]],
) -> list[str]:
    instructions: list[str] = []
    for transition, reversed_edge in path:
        src_page = graph.get_page(transition.src_page_id)
        dst_page = graph.get_page(transition.dst_page_id)
        if reversed_edge:
            instructions.append(
                f'On "{_page_display_name(dst_page)}", try back once and capture whether "{_page_display_name(src_page)}" is reached'
            )
            continue
        operation = _clean_natural_text(transition.operation) or "perform the recorded page operation"
        instructions.append(
            f'On "{_page_display_name(src_page)}", perform "{operation}" to enter "{_page_display_name(dst_page)}"'
        )
    return instructions


def _exploratory_navigation_text(graph: FunctionalFlowGraph, target: FunctionalityNode) -> str:
    """Level D: without a recorded path, describe the target and let the executor find an entry."""
    target_pages = [
        page for page_id in target.coverage_page_ids
        if (page := graph.get_page(page_id)) is not None
    ][:3]
    entry_pages = [
        page for page_id in target.entry_page_ids
        if (page := graph.get_page(page_id)) is not None
    ][:3]
    parts = [f'No path to "{_function_label(target)}" is recorded in the FFG; use exploratory navigation']
    if target.description:
        parts.append(f"Functionality: {_clip_summary_text(target.description, 160)}")
    if target_pages:
        parts.append(
            "Target pages: "
            + ", ".join(
                f'"{_page_display_name(page)}"'
                + (f" ({_clip_summary_text(page.description, 80)})" if page.description else "")
                for page in target_pages
            )
        )
    if entry_pages:
        parts.append("Known entry pages: " + ", ".join(f'"{_page_display_name(page)}"' for page in entry_pages))
    parts.append(
        "Find an entry from the current page (you may first use open_app to return to the app main page), confirming each step by screenshot. "
        "Stop navigation when the target page is visible; if it cannot be found, explain why and stop"
    )
    return "; ".join(parts) + "."


def _navigation_to_function(
    graph: FunctionalFlowGraph,
    current_page: Page | None,
    target: FunctionalityNode,
    current_page_id: str | None = None,
    adjacency: _PageAdjacency | None = None,
) -> tuple[str, str, str]:
    """Build reference navigation and return (instructions, reason, level).

    Reuse current_page_id and adjacency supplied by the planning round to avoid
    repeating screenshot-similarity matching and adjacency scans for each candidate.

    NAV_LEVEL_*: empty=already at target, A=forward, B=backtrack, C=restart plus
    recorded path, D=exploratory. Navigation does not determine scheduling eligibility.
    """
    # entry_page_ids are navigation-only anchors, not pages implementing the target
    # functionality. Stopping there could test similarly named widgets in the wrong module.
    target_page_ids = list(target.coverage_page_ids)
    if current_page_id is None:
        current_page_id = _resolve_current_page_id(graph, current_page)
    if target_page_ids and current_page_id in target_page_ids:
        return "", "Already on a page of the target functionality", NAV_LEVEL_HERE
    if adjacency is None:
        adjacency = _page_adjacency(graph)

    level = NAV_LEVEL_FORWARD
    restart_first = False
    path: list[tuple[PageTransition, bool]] = []
    if target_page_ids:
        path = _leveled_page_path(graph, current_page_id, target_page_ids, adjacency=adjacency)
        if not path:
            level = NAV_LEVEL_BACKTRACK
            path = _leveled_page_path(
                graph, current_page_id, target_page_ids, allow_backtrack=True, adjacency=adjacency,
            )
        if not path:
            level = NAV_LEVEL_RESTART
            path, restart_first = _route_via_restart(
                graph, current_page_id, target_page_ids, adjacency, allow_backtrack=True,
            )

    if not path and not restart_first:
        return (
            _exploratory_navigation_text(graph, target),
            "No recorded path from the current position or app entry; using exploratory navigation",
            NAV_LEVEL_EXPLORE,
        )

    instructions: list[str] = []
    if restart_first:
        instructions.append(RESTART_NAVIGATION_INSTRUCTION)
        root_page = _root_page(graph)
        if root_page is not None:
            instructions.append(
                f'If reopening does not reach "{_page_display_name(root_page)}", '
                "perform safe back actions with screenshot confirmation until this recorded entry page is reached"
            )
    instructions.extend(_path_step_instructions(graph, path))
    if path:
        last_transition, reversed_edge = path[-1]
        target_page = graph.get_page(
            last_transition.src_page_id if reversed_edge else last_transition.dst_page_id
        )
    else:
        target_page = _node_representative_page(graph, target)
    instructions.append(f'Confirm arrival at "{_page_display_name(target_page)}", then end navigation')
    reason = {
        NAV_LEVEL_FORWARD: "Use the shortest recorded page path in the FFG",
        NAV_LEVEL_BACKTRACK: "The recorded path requires returning to a source page first",
        NAV_LEVEL_RESTART: "Reopen the app before following the recorded path",
    }[level]
    return "; ".join(instructions) + ".", reason, level


def _build_scope_planning_summary(
    graph: FunctionalFlowGraph,
    current_page: Page | None,
    scenario_type: str,
    target: FunctionalityNode | None = None,
    src: FunctionalityNode | None = None,
    dst: FunctionalityNode | None = None,
    dependency_reason: str = "",
    current_page_id: str | None = None,
) -> tuple[str, str, str]:
    """Build the scope summary and return (summary, reference navigation, navigation level)."""
    nodes = [node for node in (target, src, dst) if node is not None]
    navigation_target = src if scenario_type == "cross_test" else target
    navigation_scenario = ""
    navigation_reason = "No target functionality"
    navigation_level = NAV_LEVEL_HERE
    if current_page_id is None:
        current_page_id = _resolve_current_page_id(graph, current_page)
    adjacency = _page_adjacency(graph)
    if navigation_target is not None:
        navigation_scenario, navigation_reason, navigation_level = _navigation_to_function(
            graph,
            current_page,
            navigation_target,
            current_page_id,
            adjacency,
        )

    lines = [
        "[Current position]",
        _build_current_location_summary(graph, current_page, current_page_id),
        "",
        "[Reference navigation] (executed first; verify against actual observations)",
        navigation_scenario or navigation_reason,
    ]
    if scenario_type == "cross_test":
        lines.extend([
            "",
            "[Inter-functional dependency for this round]",
            f"{src.id if src else ''} -> {dst.id if dst else ''} | {_clean_natural_text(dependency_reason)}",
        ])
        cross_path: list = []
        if src is not None and dst is not None:
            for start_page_id in src.page_ids:
                candidate = _shortest_page_path(
                    graph,
                    start_page_id,
                    list(dst.page_ids),
                    adjacency=adjacency,
                )
                if candidate and (not cross_path or len(candidate) < len(cross_path)):
                    cross_path = candidate
        lines.append("[Known path from source to target functionality]")
        shared_pages = set(src.page_ids if src else []) & set(dst.page_ids if dst else [])
        if shared_pages:
            lines.append(f"- Shared pages: {sorted(shared_pages)}")
        elif cross_path:
            for transition in cross_path:
                lines.append(
                    f"- {transition.src_page_id} ({_page_display_name(graph.get_page(transition.src_page_id))})"
                    f" --[{_clean_natural_text(transition.operation)}]--> "
                    f"{transition.dst_page_id} ({_page_display_name(graph.get_page(transition.dst_page_id))})"
                )
        else:
            lines.append("- No recorded path; after the source operation, use the interface to find the target entry")

    scope_page_ids = {page_id for node in nodes for page_id in node.page_ids}
    for node in nodes:
        title = "Current functionality" if scenario_type == "intra_test" else (
            "Source functionality" if src is not None and node.id == src.id else "Target functionality"
        )
        lines.extend([
            "",
            f"[{title}]",
            f"- {node.id} | {_function_label(node)}",
            f"- Description: {_clip_summary_text(node.description)}",
            f"- Status: {node.test_status} | Coverage score: {node.coverage_percent}%",
            f"- Phase: {_describe_intra_functionality_phase(graph, node)}",
            f"- Associated pages: {node.page_ids or 'none'}",
            f"- Navigation-only entry pages: {node.entry_page_ids or 'none'}",
            f"- State variables: {node.resource_variables or 'none'}",
            "- Tested scenarios:",
        ])
        if node.scenarios:
            # Show the latest K scenarios to avoid repeating tested business operations.
            # Summarize older scenarios by count so long runs do not grow prompts without bound.
            older_count = max(0, len(node.scenarios) - PLAN_SCOPE_RECENT_SCENARIOS)
            if older_count:
                lines.append(
                    f"  - ({older_count} earlier tested scenarios are omitted; "
                    "do not repeat them either. The following are the most recent scenarios.)"
                )
            lines.extend(
                f"  - {_clip_summary_text(item, 360)}"
                for item in node.scenarios[older_count:]
            )
        else:
            lines.append("  - none")
        lines.append("- Pages:")
        for page_id in node.page_ids:
            page = graph.get_page(page_id)
            if page is None:
                continue
            page_role = "navigation-only entry" if page_id in node.entry_page_ids else "functionality page"
            lines.append(
                f"  - {page.id} | {_page_display_name(page)} | role={page_role} | "
                f"activity={page.activity or 'unknown'}"
            )
            if page.description:
                lines.append(f"    Description: {_clip_summary_text(page.description)}")
            if page.controls:
                lines.append("    Widgets: " + "; ".join(_clip_summary_text(value, 180) for value in page.controls[-30:]))
            if page.operations:
                lines.append("    Executed: " + "; ".join(_clip_summary_text(value, 180) for value in page.operations[-12:]))

    # Include only transitions adjacent to scope pages: internal paths first, then
    # boundary paths, capped so large graphs cannot fill the entire prompt.
    internal_transitions = [
        transition
        for transition in graph.page_transitions
        if transition.src_page_id in scope_page_ids and transition.dst_page_id in scope_page_ids
    ]
    boundary_transitions = [
        transition
        for transition in graph.page_transitions
        if (transition.src_page_id in scope_page_ids) != (transition.dst_page_id in scope_page_ids)
    ]
    relevant_transitions = (internal_transitions + boundary_transitions)[:PLAN_SCOPE_MAX_TRANSITIONS]
    omitted_transitions = len(internal_transitions) + len(boundary_transitions) - len(relevant_transitions)
    lines.extend(["", "[Related page paths]"])
    if relevant_transitions:
        for transition in relevant_transitions:
            lines.append(
                f"- {transition.src_page_id} ({_page_display_name(graph.get_page(transition.src_page_id))})"
                f" --[{_clean_natural_text(transition.operation)}]--> "
                f"{transition.dst_page_id} ({_page_display_name(graph.get_page(transition.dst_page_id))})"
            )
        if omitted_transitions > 0:
            lines.append(f"- ({omitted_transitions} additional scope-boundary transitions omitted)")
    else:
        lines.append("- none")
    return "\n".join(lines), navigation_scenario, navigation_level


def _compliance_evidence_signature(
    graph: FunctionalFlowGraph,
    rules: list[dict[str, Any]] | None = None,
) -> str:
    """Sign FFG facts relevant to GUI navigation/evidence, excluding large screenshots.

    With rules, include only functionalities, pages, and adjacent paths matching that
    group's keywords. Unrelated browsing must not reopen settings/account targets;
    new relevant entries should. Without rules, retain a whole-graph diagnostic signature.
    """
    selected_function_ids: set[str] | None = None
    selected_page_ids: set[str] | None = None
    if rules:
        terms = _compliance_terms(rules)
        selected_page_ids = {
            str(page.id or "")
            for page in graph.active_pages()
            if _page_compliance_score(page, terms) > 0
        }
        selected_function_ids = {
            str(node.id or "")
            for node in graph.functions.values()
            if _function_compliance_score(graph, node, terms) > 0
        }
        for function_id in selected_function_ids:
            node = graph.get_node(function_id)
            if node is not None:
                selected_page_ids.update(str(page_id or "") for page_id in node.page_ids)
                selected_page_ids.update(str(page_id or "") for page_id in node.entry_page_ids)
        for page_id in tuple(selected_page_ids):
            selected_function_ids.update(graph.get_page_owners(page_id))

    functions = [
        node
        for node in sorted(graph.functions.values(), key=lambda item: str(item.id or ""))
        if selected_function_ids is None or str(node.id or "") in selected_function_ids
    ]
    pages = [
        page
        for page in sorted(graph.active_pages(), key=lambda item: str(item.id or ""))
        if selected_page_ids is None or str(page.id or "") in selected_page_ids
    ]
    payload = {
        "functions": [
            {
                "id": node.id,
                "name": node.name,
                "description": node.description,
                "pages": sorted(node.page_ids),
                "entries": sorted(node.entry_page_ids),
            }
            for node in functions
        ],
        "pages": [
            {
                "id": page.id,
                "name": page.name,
                "description": page.description,
                "activity": page.activity,
                "controls": sorted(page.controls),
                "operations": sorted(page.operations),
                # Store a digest rather than the screenshot payload. New stable GUI states
                # observed on relevant pages may justify reassessing deferred checks.
                "visual_evidence": sorted(
                    hashlib.sha256(str(screenshot).encode("utf-8")).hexdigest()[:16]
                    for screenshot in [
                        page.representative_screenshot,
                        *page.evidence_screenshots,
                    ]
                    if str(screenshot or "").strip()
                ),
            }
            for page in pages
        ],
        "transitions": sorted(
            (
                str(transition.src_page_id or ""),
                str(transition.dst_page_id or ""),
                str(transition.operation or ""),
            )
            for transition in graph.page_transitions
            if selected_page_ids is None
            or str(transition.src_page_id or "") in selected_page_ids
            or str(transition.dst_page_id or "") in selected_page_ids
        ),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _compliance_evidence_signatures(
    graph: FunctionalFlowGraph,
    rules: list[dict[str, Any]],
) -> dict[str, str]:
    """Rules in the same group share the relevant FFG signature; return rule_id -> signature."""
    signatures: dict[str, str] = {}
    for group in _compliance_groups(rules):
        group_rules = [
            rule
            for rule in rules
            if str(rule.get("checkpoint", {}).get("group", "") or "").strip() == group
        ]
        signature = _compliance_evidence_signature(graph, group_rules)
        for rule in group_rules:
            signatures[str(rule.get("rule_id", "") or "")] = signature
    for rule in rules:
        rule_id = str(rule.get("rule_id", "") or "")
        signatures.setdefault(rule_id, _compliance_evidence_signature(graph, [rule]))
    return signatures


def _compliance_groups(rules: list[dict[str, Any]]) -> list[str]:
    return list(dict.fromkeys(
        str(rule.get("checkpoint", {}).get("group", "") or "").strip()
        for rule in rules
        if str(rule.get("checkpoint", {}).get("group", "") or "").strip()
    ))


def _compliance_terms(rules: list[dict[str, Any]]) -> tuple[str, ...]:
    terms: list[str] = []
    for group in _compliance_groups(rules):
        terms.extend(COMPLIANCE_GROUP_DISCOVERY_TERMS.get(group, ()))
    return tuple(dict.fromkeys(terms))


def _page_compliance_score(page: Page, terms: tuple[str, ...]) -> int:
    text = " ".join([
        str(page.name or ""),
        str(page.description or ""),
        str(page.activity or ""),
        *[str(item or "") for item in page.controls],
        *[str(item or "") for item in page.operations],
    ]).lower()
    return sum(1 for term in terms if term.lower() in text)


def _function_compliance_score(
    graph: FunctionalFlowGraph,
    node: FunctionalityNode,
    terms: tuple[str, ...],
) -> int:
    text = f"{node.name} {node.description}".lower()
    score = sum(1 for term in terms if term.lower() in text)
    for page_id in node.page_ids:
        page = graph.get_page(page_id)
        if page is not None:
            score += _page_compliance_score(page, terms)
    return score


def _compliance_ffg_guidance(
    graph: FunctionalFlowGraph,
    current_page: Page | None,
    rules: list[dict[str, Any]],
    current_page_id: str | None = None,
) -> tuple[str, str]:
    """Return reference navigation and candidate-page guidance from the recorded FFG."""
    terms = _compliance_terms(rules)
    if not terms:
        return "", ""
    ranked_pages = sorted(
        (
            (_page_compliance_score(page, terms), page)
            for page in graph.active_pages()
        ),
        key=lambda item: (-item[0], str(item[1].id or "")),
    )
    candidates = [page for score, page in ranked_pages if score > 0][:5]
    if not candidates:
        return "", "[FFG compliance evidence] No clearly relevant page is recorded for this group. Prefer candidate settings, profile, login, or help entries."

    candidate_lines = []
    for page in candidates:
        matched_controls = [
            control
            for control in page.controls
            if any(term.lower() in str(control).lower() for term in terms)
        ][:4]
        candidate_lines.append(
            f"- {_page_display_name(page)}"
            + (f" | Relevant widgets: {', '.join(matched_controls)}" if matched_controls else "")
        )
    guidance = "[Recorded FFG compliance evidence]\n" + "\n".join(candidate_lines)

    if current_page_id is None:
        current_page_id = _resolve_current_page_id(graph, current_page)
    adjacency = _page_adjacency(graph)
    for candidate in candidates:
        if current_page_id == candidate.id:
            return "", guidance + f'\nAlready on candidate page "{_page_display_name(candidate)}"; collect evidence here.'
        # Use recorded forward edges; share level-C restart fallback with functionality navigation.
        path = _leveled_page_path(graph, current_page_id, [candidate.id], adjacency=adjacency)
        restart_first = False
        if not path:
            path, restart_first = _route_via_restart(
                graph, current_page_id, [str(candidate.id or "")], adjacency, allow_backtrack=False,
            )
        if not path and not restart_first:
            continue
        instructions = [RESTART_NAVIGATION_INSTRUCTION] if restart_first else []
        instructions.extend(_path_step_instructions(graph, path))
        instructions.append(f'Confirm arrival at "{_page_display_name(candidate)}", then end navigation')
        return "; ".join(instructions) + ".", guidance
    return "", guidance + "\nNo recorded path reaches these candidates; prioritize discovering their entry paths during functionality exploration."


def _deferred_compliance_guidance(rules_path: str | None) -> tuple[str, tuple[str, ...]]:
    from workflow.compliance_worker import compliance_worker

    checklist = ComplianceChecklist.from_state(compliance_worker.snapshot())
    rules = checklist.deferred_rules(rules_path)
    if not rules:
        return "", ()
    groups = _compliance_groups(rules)
    lines = [
        "[Unresolved compliance targets] These groups lacked evidence in the previous round. If this round's fixed functionality has relevant entries, "
        "prefer different paths or widgets that add GUI evidence; do not simply repeat the previous actions:"
    ]
    for group in groups:
        group_rules = [
            rule for rule in rules
            if str(rule.get("checkpoint", {}).get("group", "") or "") == group
        ]
        objectives = list(dict.fromkeys(
            str(rule.get("checkpoint", {}).get("objective", "") or "").strip()
            for rule in group_rules
            if str(rule.get("checkpoint", {}).get("objective", "") or "").strip()
        ))
        lines.append(
            f"- {GROUP_LABELS.get(group, group)}: "
            + "; ".join(objectives[:2])
        )
    return "\n".join(lines), _compliance_terms(rules)


def _maybe_schedule_compliance_check(
    state: AgentState,
    graph: FunctionalFlowGraph,
    current_page: Page | None,
    *,
    force_deferred: bool = False,
    current_page_id: str | None = None,
) -> dict | None:
    """In compliance mode, schedule unfinished check groups before functionality exploration.

    The compliance worker owns the authoritative checklist and updates verdicts
    asynchronously. Read a fresh worker snapshot each round instead of stale state.
    """
    if not bool(state.get("compliance_detection_enabled", False)):
        return None
    from workflow.compliance_worker import compliance_worker

    rules_path = str(state.get("compliance_rules_path", "") or "").strip() or None
    checklist = ComplianceChecklist.from_state(compliance_worker.snapshot())
    group_rules = checklist.next_group_rules(rules_path)
    if not group_rules:
        deferred_rules = gui_rules(rules_path)
        reopened, snapshot = compliance_worker.reopen_deferred(
            _compliance_evidence_signatures(graph, deferred_rules),
            force=force_deferred,
            rules_path=rules_path,
        )
        if not reopened:
            return None
        checklist = ComplianceChecklist.from_state(snapshot)
        group_rules = checklist.next_group_rules(rules_path)
        if not group_rules:
            return None
        print(
            "[Scenario planning] Reopened compliance checks with insufficient evidence: "
            f"checks={','.join(reopened)}, force={force_deferred}"
        )
    navigation_scenario, test_scenario, plan_extra = build_compliance_scenarios(group_rules)
    rule_ids = plan_extra["compliance_check_ids"]
    known_navigation, ffg_guidance = _compliance_ffg_guidance(
        graph,
        current_page,
        group_rules,
        current_page_id,
    )
    if not bool(plan_extra.get("compliance_cold_start")):
        navigation_scenario = known_navigation
    if ffg_guidance:
        test_scenario = f"{test_scenario}\n{ffg_guidance}"
    next_attempt = max(
        (int(checklist.attempts.get(rule_id, 0) or 0) + 1 for rule_id in rule_ids),
        default=1,
    )
    if next_attempt > 1:
        previous_gaps = [
            str(finding.get("description", "") or "").strip()
            for finding in checklist.findings
            if str(finding.get("check_id", "") or "") in rule_ids
            and str(finding.get("description", "") or "").strip()
        ]
        test_scenario += (
            f"\n[Evidence reassessment | attempt {next_attempt}] Previous evidence gaps: "
            + ("; ".join(previous_gaps[:6]) if previous_gaps else "no specific gap recorded")
            + ". Prefer new FFG paths or different entries above; do not mechanically repeat the previous actions."
        )
    plan_extra["compliance_attempt"] = next_attempt
    compliance_worker.mark_executing(
        rule_ids,
        _compliance_evidence_signatures(graph, group_rules),
    )
    checklist = ComplianceChecklist.from_state(compliance_worker.snapshot())
    print(
        "[Scenario planning] Scheduled compliance checks: "
        f"group={plan_extra['compliance_group']}, checks={','.join(rule_ids)}, "
        f"attempt={next_attempt}, cold_start={plan_extra['compliance_cold_start']}"
    )
    return {
        "ffg": graph,
        "current_page": current_page,
        "current_navigation_scenario": navigation_scenario,
        "current_test_scenario": test_scenario,
        "current_testing_function_id": "",
        "current_testing_function_status": "not_started",
        "current_plan": {
            "scenario_type": "compliance_check",
            "target_function_id": "",
            "src_function_id": "",
            "dst_function_id": "",
            **plan_extra,
        },
        "compliance_checklist": checklist.to_state(),
        "workflow_complete": False,
    }


def _targeted_compliance_stop(state: AgentState) -> dict | None:
    """Wait for targeted asynchronous verdicts without exploring unrelated functionalities."""
    if state.get("run_mode") != "compliance" or not state.get("compliance_targeted"):
        return None
    from workflow.compliance_worker import compliance_worker

    update = {
        "current_navigation_scenario": "", "current_test_scenario": "",
        "current_plan": {"scenario_type": "skip"},
    }
    if compliance_worker.pending_count() and not compliance_worker.flush(timeout=1.0):
        return {**update, "workflow_complete": False}
    snapshot = compliance_worker.snapshot()
    summary = ComplianceChecklist.from_state(snapshot).summary(state.get("compliance_rules_path") or None)
    if summary["gui_total"] > 0 and summary["processed"] == summary["gui_total"]:
        print("[Targeted compliance] Requested rules reviewed; stopping targeted testing while preserving insufficient-evidence verdicts.")
        return {**update, "workflow_complete": True, "compliance_checklist": snapshot}
    return None


def _planning_skip_unavailable(
    state: AgentState,
    graph: FunctionalFlowGraph,
    *,
    screenshot_unavailable: bool,
) -> dict:
    """Skip unavailable screenshots or unconfirmed app pages and track consecutive failures."""
    if screenshot_unavailable:
        unavailable_count = int(state.get("screenshot_unavailable_count", 0) or 0) + 1
        workflow_complete = (
            STOP_AFTER_SCREENSHOT_UNAVAILABLE
            and unavailable_count >= MAX_CONSECUTIVE_SCREENSHOT_UNAVAILABLE
        )
        if workflow_complete:
            print(
                "[Scenario planning] Repeated unavailable/black screenshots; stopping this app run: "
                f"{unavailable_count}/{MAX_CONSECUTIVE_SCREENSHOT_UNAVAILABLE}"
            )
        else:
            print(
                "[Scenario planning] Screenshot unavailable or black; skipping this round and retrying after recovery: "
                f"{unavailable_count}/{MAX_CONSECUTIVE_SCREENSHOT_UNAVAILABLE}"
            )
        return _skip_response(
            graph,
            None,
            testing_function_id=_normalize_optional_id(
                state.get("current_testing_function_id", "")
            ),
            testing_function_status=str(
                state.get("current_testing_function_status", "not_started")
            ).strip() or "not_started",
            workflow_complete=workflow_complete,
            screenshot_unavailable_count=unavailable_count,
        )
    unavailable_count = int(state.get("target_app_unavailable_count", 0) or 0) + 1
    workflow_complete = (
        STOP_AFTER_TARGET_APP_UNAVAILABLE
        and unavailable_count >= MAX_CONSECUTIVE_TARGET_APP_UNAVAILABLE
    )
    if workflow_complete:
        print(
            "[Scenario planning] Repeated target-app recovery failures; stopping to avoid idle looping: "
            f"{unavailable_count}/{MAX_CONSECUTIVE_TARGET_APP_UNAVAILABLE}"
        )
    else:
        print(
            "[Scenario planning] Target-app page unconfirmed; skipping this round and retrying after restart: "
            f"{unavailable_count}/{MAX_CONSECUTIVE_TARGET_APP_UNAVAILABLE}"
        )
    return _skip_response(
        graph,
        None,
        workflow_complete=workflow_complete,
        target_app_unavailable_count=unavailable_count,
    )


def _observe_planning_page(
    state: AgentState,
    graph: FunctionalFlowGraph,
    device,
    cold_start: bool,
) -> tuple[Page | None, dict | None]:
    """Confirm the target app and refresh its page; return (page, optional early skip response)."""
    current_page = state.get("current_page")
    if not isinstance(current_page, Page):
        current_page = None
    package_name = str(state.get("package_name", ""))
    try:
        current_page = _ensure_target_app(
            device,
            package_name,
            current_page=current_page,
        )
        if not isinstance(current_page, Page) and not package_name.strip():
            current_page = _safe_capture_current_page(device)
        elif not cold_start and isinstance(current_page, Page):
            current_page = _refresh_current_page_observation(
                graph,
                device,
                current_page,
            )
    except ScreenshotUnavailableError:
        return None, _planning_skip_unavailable(state, graph, screenshot_unavailable=True)
    if not isinstance(current_page, Page):
        return None, _planning_skip_unavailable(state, graph, screenshot_unavailable=False)
    return current_page, None


def _cold_start_response(llm, state: AgentState, graph: FunctionalFlowGraph, current_page: Page) -> dict:
    """Initialize an empty FFG from the current screenshot without executing a scenario."""
    print("[Scenario planning] Empty FFG; initializing from the current screenshot.")
    _bootstrap_initial_ffg(
        llm,
        graph,
        current_page,
        app_name=str(state.get("app_name", "") or ""),
        package_name=str(state.get("package_name", "") or ""),
    )
    current_page = next(iter(graph.active_pages()), current_page)
    stats = _count_ffg_stats(graph)
    print("[Scenario planning] Input context")
    print(
        f"  Exploration scope: core functionalities | Current page: {_page_display_name(current_page)} | Screenshot: yes"
    )
    print(
        "  FFG statistics: "
        f"functionalities={stats['functions']}, pages={stats['pages']}, operations={stats['operations']}, "
        f"page_transitions={stats['page_transitions']}, "
        f"functional_dependencies={stats['function_dependencies']}, intra_scenarios={stats['intra_scenarios']}, "
        f"inter_functional_scenarios={stats['cross_scenarios']}"
    )
    print("[Scenario planning] Initialization only this round; plan scenarios from the FFG summary next round.")
    return _skip_response(graph, current_page)


@dataclass
class _RoundSelection:
    """Scope selected for one planning round."""

    scenario_type: str = ""
    target_node: FunctionalityNode | None = None
    src_node: FunctionalityNode | None = None
    dst_node: FunctionalityNode | None = None
    dependency_reason: str = ""
    extra_updates: dict = field(default_factory=dict)


@dataclass
class _PlanningRound:
    """Context shared within a planning round; resolve current_page_id only once."""

    llm: Any
    state: AgentState
    graph: FunctionalFlowGraph
    current_page: Page | None
    current_page_id: str
    current_testing_function_id: str
    current_testing_function_status: str
    compliance_guidance: str
    compliance_terms: tuple[str, ...]

    def skip_round(self, reason: str, **extra) -> dict:
        print(f"[Scenario planning] {reason}; skipping execution this round.")
        return _skip_response(
            self.graph,
            self.current_page,
            testing_function_id=self.current_testing_function_id,
            testing_function_status=self.current_testing_function_status,
            **extra,
        )


def _select_round_scope(round_ctx: _PlanningRound) -> tuple[_RoundSelection, dict | None]:
    """Select inter-functional, intra-functional, then finalization work in that order.

    Return (selection, early_response); a nonempty early response suppresses scenario generation.
    """
    state, graph, current_page = round_ctx.state, round_ctx.graph, round_ctx.current_page
    selection = _RoundSelection()

    # 1) Schedule inter-functional candidates once both endpoints have attempts and
    # successful business operations. Unrelated work must not starve them; navigation is advisory.
    if _normalize_run_phase(state.get("run_phase", "all")) == "all":
        for src_id, dst_id, candidate_reason in _pending_cross_pairs(graph):
            if not _cross_pair_ready(graph, src_id, dst_id):
                continue
            selection.src_node, selection.dst_node = graph.get_node(src_id), graph.get_node(dst_id)
            selection.dependency_reason = candidate_reason
            selection.scenario_type = "cross_test"
            return selection, None

    # 2) Select the first unfinished intra-functional target under the deterministic order.
    selection.target_node = _select_intra_target(
        graph,
        current_page,
        _normalize_failure_counts(state.get("function_execution_failure_counts")),
        round_ctx.compliance_terms,
        round_ctx.current_page_id,
    )
    if selection.target_node is not None:
        selection.scenario_type = "intra_test"
        return selection, None

    # 3) No ordinary executable targets remain; enter finalization.
    if bool(state.get("compliance_detection_enabled", False)):
        from workflow.compliance_worker import compliance_worker

        if compliance_worker.pending_count() > 0:
            print(
                "[Scenario planning] No ordinary exploration targets remain; waiting for asynchronous compliance review."
            )
            if not compliance_worker.flush(
                timeout=COMPLIANCE_CLOSE_LOOP_WAIT_SECONDS
            ):
                return selection, round_ctx.skip_round("Asynchronous compliance reviews are pending; workflow remains open")
    # Give deferred check groups one final chance if their direct retry budget remains.
    # Exhausted rules stay unresolved; without new relevant observations they must
    # not prevent this app run from finalizing.
    compliance_update = _maybe_schedule_compliance_check(
        state,
        graph,
        current_page,
        force_deferred=True,
        current_page_id=round_ctx.current_page_id,
    )
    if compliance_update is not None:
        return selection, compliance_update

    # 3a) Give each deferred functionality one final retry when a known path exists.
    #     The next round's status transition determines its outcome.
    retried_ids = [str(item) for item in (state.get("deferred_retry_ids") or [])]
    selection.target_node = _select_deferred_retry(
        graph, current_page, retried_ids, round_ctx.current_page_id,
    )
    if selection.target_node is not None:
        selection.scenario_type = "intra_test"
        selection.extra_updates["deferred_retry_ids"] = [*retried_ids, selection.target_node.id]
        print(
            "[Scenario planning] Finalization: one final retry for deferred functionality: "
            f"{selection.target_node.id} | {_function_label(selection.target_node)}"
        )
        return selection, None

    # 3b) Audit coverage gaps in completed functionalities without changing their status.
    return selection, _completion_audit_round(round_ctx)


def _completion_audit_round(round_ctx: _PlanningRound) -> dict:
    """Execute audit gaps, skip while audits remain, or finish once all audits are processed."""
    state, graph, current_page = round_ctx.state, round_ctx.graph, round_ctx.current_page
    audit_attempts = state.get("completion_audit_attempts", {}) or {}
    previous_signatures = state.get("completion_audit_signatures", {}) or {}
    print("[Scenario planning] No functionality or flow targets remain; auditing coverage gaps.")
    audit_result, audit_complete, audited_signatures = _run_completion_audit(
        round_ctx.llm,
        graph,
        PLAN_EXPLORATION_MODE,
        previous_signatures if isinstance(previous_signatures, dict) else {},
        audit_attempts if isinstance(audit_attempts, dict) else {},
        app_name=str(state.get("app_name", "") or ""),
        package_name=str(state.get("package_name", "") or ""),
        deadline=_run_deadline(state),
    )
    merged_signatures = dict(previous_signatures) if isinstance(previous_signatures, dict) else {}
    merged_signatures.update(audited_signatures)
    if audit_result is not None:
        return _audit_exploration_response(
            graph,
            current_page,
            audit_result,
            merged_signatures,
            audit_attempts if isinstance(audit_attempts, dict) else {},
        )
    if not audit_complete:
        return round_ctx.skip_round(
            "Coverage-gap auditing is unfinished; workflow remains open",
            completion_audit_signatures=merged_signatures,
        )
    print("[Scenario planning] All functionalities, inter-functional dependencies, and coverage gaps have been processed.")
    return _skip_response(
        graph,
        current_page,
        workflow_complete=True,
        completion_audit_signatures=merged_signatures,
    )


def _scope_instruction(
    graph: FunctionalFlowGraph,
    selection: _RoundSelection,
) -> tuple[str, str]:
    """Return (functionality phase, scope instructions for this round)."""
    if selection.scenario_type == "cross_test":
        return FUNCTIONALITY_PHASE_DEPTH, (
            "[Round scope] cross_test: "
            f'source functionality "{_function_label(selection.src_node)}" -> target functionality "{_function_label(selection.dst_node)}". '
            "Generate one resource-linked scenario: produce/modify in the source, then read/display/consume in the target."
        )
    target_node = selection.target_node
    function_phase = _intra_functionality_phase(graph, target_node)
    if function_phase == FUNCTIONALITY_PHASE_DEPTH:
        return function_phase, (
            f'[Round scope] intra_test: functionality "{_function_label(target_node)}". '
            "[Phase=depth: complete business scenario] Entries have been explored; write one complete scenario following this functionality's business logic. "
            "Choose a suitable direction using recorded pages, widgets, and operations; examples are guidance rather than mandatory categories: "
            "persistence after leaving and returning; a create-edit-delete lifecycle; "
            "boundary or invalid input (empty, long, special characters); linked states or mutually exclusive options; cancellation or back-navigation state. "
            "Avoid business scenarios already present in [Tested scenarios]. "
            "If no new business scenario is possible, leave test_scenario empty."
        )
    return function_phase, (
        f'[Round scope] intra_test: functionality "{_function_label(target_node)}". '
        "[Phase=breadth: explore entries] Return a nonempty test_scenario if a distinct business direction remains. "
        "If coverage is sufficient or only tested scenarios can be repeated, leave test_scenario empty."
    )


def _build_plan_message_content(
    round_ctx: _PlanningRound,
    selection: _RoundSelection,
    scope_summary: str,
    scope_instruction: str,
) -> list[dict]:
    state, graph, current_page = round_ctx.state, round_ctx.graph, round_ctx.current_page
    package_name = str(state.get("package_name", "") or "")
    boundary_hint = rule_for(package_name).to_prompt_hint()
    message_content: list[dict] = [
        {
            "type": "text",
            "text": (
                f"[Target app] name={state.get('app_name', '') or 'unknown'}, "
                f"package={package_name or 'unknown'}"
            ),
        },
        {
            "type": "text",
            "text": (
                "[Exploration scope] "
                + EXPLORATION_MODE_INSTRUCTIONS[PLAN_EXPLORATION_MODE]
                + (boundary_hint if boundary_hint else "")
            ),
        },
        {"type": "text", "text": scope_summary},
        {"type": "text", "text": scope_instruction},
        {
            "type": "text",
            "text": _test_data_instruction(
                graph, selection.scenario_type, selection.target_node, selection.src_node,
            ),
        },
    ]
    if round_ctx.compliance_guidance:
        message_content.append({"type": "text", "text": round_ctx.compliance_guidance})
    # A screenshot helps only when already on this round's functionality page; navigation may change it.
    scope_node = selection.src_node if selection.scenario_type == "cross_test" else selection.target_node
    if (
        current_page is not None
        and current_page.representative_screenshot
        and round_ctx.current_page_id in scope_node.coverage_page_ids
    ):
        message_content.extend([
            {
                "type": "text",
                "text": "[Current-page screenshot] Use only to confirm position; do not use widgets absent from the FFG.",
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{current_page.representative_screenshot}",
                },
            },
        ])
    return message_content


def _plan_test_scenario(
    round_ctx: _PlanningRound,
    selection: _RoundSelection,
    message_content: list[dict],
    known_navigation: str,
) -> tuple[str, list[str]]:
    """Ask the planning model for test_scenario, falling back to a template (possibly empty)."""
    graph = round_ctx.graph
    package_name = str(round_ctx.state.get("package_name", "") or "")
    result, validation_errors = _invoke_valid_plan(
        round_ctx.llm,
        UNIFIED_SCENARIO_PLAN_SYSTEM_PROMPT,
        message_content,
        scenario_type=selection.scenario_type,
        package_name=package_name,
    )
    test_scenario = _sanitize_planned_scenario(
        _normalize_optional_id(result.get("test_scenario"))
    ) if result else ""
    if test_scenario and _matches_black_screen_path(
        graph, f"{known_navigation}; {test_scenario}"
    ):
        print("[Scenario planning] Plan repeats a path known to cause a black/unavailable page; using a fallback template.")
        test_scenario = ""
    if test_scenario:
        return test_scenario, validation_errors
    # Keep the round executable with a fallback template when the model has no usable scenario.
    if selection.scenario_type == "cross_test":
        test_scenario = _fallback_inter_functional_scenario(
            graph, selection.src_node.id, selection.dst_node.id, selection.dependency_reason
        )
    else:
        test_scenario = _forced_intra_scenario(
            graph, selection.target_node, package_name=package_name
        )
        if test_scenario and _matches_black_screen_path(graph, test_scenario):
            test_scenario = ""
    if test_scenario:
        print(
            "[Scenario planning] Model produced no usable scenario"
            f" ({validation_errors or 'empty scenario'}); using a fallback template."
        )
    return test_scenario, validation_errors


def create_scenario_plan_node(llm, device):
    """Create the unified scenario-planning node."""

    def scenario_plan_node(state: AgentState) -> dict:
        targeted_stop = _targeted_compliance_stop(state)
        if targeted_stop is not None:
            return targeted_stop
        graph = state.get("ffg")
        if not isinstance(graph, FunctionalFlowGraph):
            graph = FunctionalFlowGraph()
        cold_start = not bool(graph.functions)

        # Stage 1: confirm the foreground app and refresh the current screenshot.
        current_page, early_response = _observe_planning_page(state, graph, device, cold_start)
        if early_response is not None:
            return early_response
        if cold_start:
            return _cold_start_response(llm, state, graph, current_page)
        # Resolve page identity once per round; target selection, navigation, and summaries
        # reuse it instead of repeating potentially expensive screenshot comparisons.
        current_page, current_page_id = _attach_current_page_identity_with_id(graph, current_page)

        # Stage 2: shared-page evidence can change coverage between rounds. Apply all
        # FFG-derived status transitions now; later target selection reads those states.
        _update_function_statuses(graph)

        rules_path = str(state.get("compliance_rules_path", "") or "").strip() or None
        compliance_guidance, compliance_terms = (
            _deferred_compliance_guidance(rules_path)
            if bool(state.get("compliance_detection_enabled", False))
            else ("", ())
        )

        # In compliance mode, run cold-start and permission-denial checks before ordinary
        # exploration establishes a session or makes state-changing checks more disruptive.
        compliance_update = _maybe_schedule_compliance_check(
            state, graph, current_page, current_page_id=current_page_id,
        )
        if compliance_update is not None:
            return compliance_update

        if state.get("run_mode") == "compliance":
            from workflow.compliance_worker import compliance_worker

            summary = ComplianceChecklist.from_state(compliance_worker.snapshot()).summary(rules_path)
            if summary["resolved"] == summary["gui_total"] and not _unfinished_function_nodes(graph):
                print("[Compliance review] All applicable checks have verdicts; ending this run.")
                return {"workflow_complete": True, "current_plan": {},
                        "current_navigation_scenario": "", "current_test_scenario": ""}

        round_ctx = _PlanningRound(
            llm=llm,
            state=state,
            graph=graph,
            current_page=current_page,
            current_page_id=current_page_id,
            current_testing_function_id=_normalize_optional_id(
                state.get("current_testing_function_id", "")
            ),
            current_testing_function_status=str(
                state.get("current_testing_function_status", "not_started")
            ).strip() or "not_started",
            compliance_guidance=compliance_guidance,
            compliance_terms=compliance_terms,
        )

        # Stage 3: select scope (inter-functional, intra-functional, then finalization).
        selection, early_response = _select_round_scope(round_ctx)
        if early_response is not None:
            return early_response
        scenario_type = selection.scenario_type
        target_node, src_node, dst_node = selection.target_node, selection.src_node, selection.dst_node
        extra_updates = selection.extra_updates

        # Stage 4: build model context and generate the scenario.
        scope_summary, known_navigation, navigation_level = _build_scope_planning_summary(
            graph,
            current_page,
            scenario_type,
            target=target_node,
            src=src_node,
            dst=dst_node,
            dependency_reason=selection.dependency_reason,
            current_page_id=current_page_id,
        )
        function_phase, scope_instruction = _scope_instruction(graph, selection)
        message_content = _build_plan_message_content(
            round_ctx, selection, scope_summary, scope_instruction,
        )
        print(
            "[Scenario planning] Fixed scope: "
            + (
                f"cross {src_node.id}->{dst_node.id} | {selection.dependency_reason}"
                if scenario_type == "cross_test"
                else f"intra {target_node.id} | {_function_label(target_node)} | "
                     f"phase={_describe_intra_functionality_phase(graph, target_node)}"
            )
            + f" | navigation_level={navigation_level or 'already at target'}"
        )
        test_scenario, validation_errors = _plan_test_scenario(
            round_ctx, selection, message_content, known_navigation,
        )
        if not test_scenario:
            return {
                **_release_unplannable_function(
                    graph,
                    current_page,
                    target_node,
                    f"Neither model nor template produced an executable scenario ({validation_errors or 'empty scenario'})",
                    state.get("unplannable_counts") or {},
                ),
                **extra_updates,
            }

        # Stage 5: assemble navigation and scheduling state.
        navigation_scenario = known_navigation
        if _is_black_screen_page(graph, current_page, current_page_id):
            navigation_scenario = (
                "Currently on a recorded black-screen page: press_back once and confirm by screenshot; "
                "if outside the target app, use open_app. Then "
                + (known_navigation or "confirm arrival at the target functionality page and stop navigation.")
            )

        if scenario_type == "intra_test":
            # Final retries retain deferred status; only ordinarily scheduled targets become in_progress.
            if (target_node.test_status or "not_started") != "deferred":
                graph.set_node_test_status(target_node.id, "in_progress", reason="scheduled")
            unplannable_counts = dict(state.get("unplannable_counts") or {})
            if unplannable_counts.pop(target_node.id, None) is not None:
                extra_updates["unplannable_counts"] = unplannable_counts
            testing_function_id = target_node.id
            testing_function_status = target_node.test_status or "in_progress"
        else:
            testing_function_id = round_ctx.current_testing_function_id
            testing_function_status = round_ctx.current_testing_function_status

        return {
            "ffg": graph,
            "current_page": current_page,
            "current_navigation_scenario": navigation_scenario,
            "current_test_scenario": test_scenario,
            "current_testing_function_id": testing_function_id,
            "current_testing_function_status": testing_function_status,
            "current_plan": {
                "scenario_type": scenario_type,
                "function_phase": function_phase,
                "navigation_level": navigation_level,
                "target_function_id": target_node.id if target_node is not None else "",
                "src_function_id": src_node.id if src_node is not None else "",
                "dst_function_id": dst_node.id if dst_node is not None else "",
                "resource_variable": _cross_state_variable(selection.dependency_reason) if scenario_type == "cross_test" else "",
            },
            "workflow_complete": False,
            **extra_updates,
        }

    return scenario_plan_node
