"""Unified scenario planning node."""

import os
import re
import time
from difflib import SequenceMatcher

from langchain_core.messages import HumanMessage, SystemMessage

from model.flow_graph import (
    FlowGraph,
    FunctionNode,
    Page,
    TERMINAL_FUNCTION_STATUSES,
    operation_family_key,
)
from tools.actions import ScreenshotUnavailableError, capture_current_page
from workflow.llm import invoke_llm_json_with_retry, llm_json_result_failed
from workflow.nodes._shared import (
    JSON_BLOCK_PATTERN,
    has_exhaustive_control_family_intent,
    screenshot_similarity,
)
from workflow.exploration_boundary import rule_for
from workflow.nodes.ffg_update import apply_ffg_patch
from workflow.nodes.prompt import (
    BOOTSTRAP_FFG_SYSTEM_PROMPT,
    EXPLORATION_MODE_INSTRUCTIONS,
    FFG_AUDIT_SYSTEM_PROMPT,
    UNIFIED_SCENARIO_PLAN_SYSTEM_PROMPT,
)
from workflow.state import AgentState
from workflow.account_safety import (
    account_action_policy_violation,
    account_destructive_intent,
)

FUNCTION_ID_PATTERN = re.compile(r"\bfunc_\d+\b")
PAGE_ID_PATTERN = re.compile(r"\bpage_\d+\b")
PAGE_ID_ANNOTATION_PATTERN = re.compile(r"\bpage_\d+(?:\([^)]*\)|\uff08[^\uff09]*\uff09)")
TEXT_NORMALIZE_PATTERN = re.compile(
    r"[\"'\u201c\u201d\u2018\u2019\uff08\uff09()\u3010\u3011\[\]{}<>"
    r"\u300a\u300b\u3001\uff0c,\u3002.\uff1a:\uff1b;!\uff01?\uff1f\-_/\s]+"
)
DUPLICATE_SCENARIO_ACTION_MARKERS = (
    "pull-down refresh",
    "refresh gesture",
    "pulltorefresh",
)
MAX_AUDIT_IMAGES_PER_FUNCTION = int(os.getenv("PLAN_MAX_AUDIT_IMAGES_PER_FUNCTION", "4"))
MAX_COMPLETION_AUDITS_PER_FUNCTION = max(
    1,
    int(os.getenv("PLAN_MAX_COMPLETION_AUDITS_PER_FUNCTION", "20")),
)
MAX_CROSS_TEST_PAIRS = int(os.getenv("PLAN_MAX_CROSS_TEST_PAIRS", "20"))
MAX_CONSECUTIVE_PLANNER_NOOPS = max(1, int(os.getenv("PLAN_MAX_CONSECUTIVE_NOOP_STATUS_UPDATES", "3")))
PLAN_VALIDATION_RETRY_ATTEMPTS = max(1, int(os.getenv("PLAN_VALIDATION_RETRY_ATTEMPTS", "3")))
# Hard budget for intra-function scenarios of a single function: forced completion once reached, handed over to coverage audit to review remaining entry points by screenshot,
# avoiding a healthy but massive function occupying execution rights for a long time (breadth-first).
PLAN_MAX_SCENARIOS_PER_FUNCTION = max(
    1,
    int(os.getenv("PLAN_MAX_SCENARIOS_PER_FUNCTION", "6")),
)
# Number of control families covered at once by a single supplementary test scenario. The fixed overhead per round (planning + FFG update with image
# LLM call + screenshot + logcat) is independent of scenario length; probing only one control at a time amortizes the fixed overhead
# over just 1-2 effective actions.
PLAN_SCENARIO_CONTROL_BATCH = max(
    1,
    int(os.getenv("PLAN_SCENARIO_CONTROL_BATCH", "4")),
)
# Enters completion and omission audit once the programmatically derived control/operation coverage reaches this value.
PLAN_FUNCTION_COVERAGE_TARGET = max(
    1,
    min(100, int(os.getenv("PLAN_FUNCTION_COVERAGE_TARGET", "80"))),
)
# Coverage may be "inherited" from operations of sibling functions sharing the page, so before completion this function
# itself is required to have run at least this many intra-function scenarios (unless there are no uncovered controls left).
PLAN_MIN_SCENARIOS_BEFORE_COMPLETE = max(
    1,
    int(os.getenv("PLAN_MIN_SCENARIOS_BEFORE_COMPLETE", "2")),
)
ENTRY_FUNCTION_NAME_PATTERN = re.compile(r"\u5165\u53e3\s*$")
CROSS_SCENARIO_MARKER_PATTERN = re.compile(r"^\[cross:([^\]>]+)->([^\]]+)\]")
# Functions continuously unable to produce executable scenarios are written to the terminal state 'unavailable' after reaching this count,
# preventing them from both lagging at the end of scheduling and permanently blocking overall convergence determination (long idling).
PLAN_UNPRODUCTIVE_RELEASE_LIMIT = max(
    1,
    int(os.getenv("PLAN_UNPRODUCTIVE_RELEASE_LIMIT", "3")),
)
APP_RESTART_DELAY = float(os.getenv("APP_RESTART_DELAY", "2.0"))
PLAN_APP_RESTART_ATTEMPTS = max(
    1,
    int(os.getenv("PLAN_APP_RESTART_ATTEMPTS", os.getenv("APP_RESTART_ATTEMPTS", "2"))),
)
PLAN_SCREENSHOT_BACK_ATTEMPTS = max(
    1,
    int(os.getenv("PLAN_SCREENSHOT_BACK_ATTEMPTS", os.getenv("APP_RECOVERY_BACK_ATTEMPTS", "3"))),
)
APP_RESTART_RETRY_DELAY = float(os.getenv("APP_RESTART_RETRY_DELAY", "1.0"))
# Cold start polling parameters aligned with the execution node: poll foreground package name after restart until the target app is visible,
# avoiding a fixed sleep capturing desktop screenshots during cold start of heavy apps.
APP_START_POLL_INTERVAL = max(0.1, float(os.getenv("APP_START_POLL_INTERVAL", "0.5")))
APP_START_MAX_WAIT = max(1.0, float(os.getenv("APP_START_MAX_WAIT", "12")))
APP_RESTART_SETTLE_DELAY = max(0.0, float(os.getenv("APP_RESTART_SETTLE_DELAY", "1.0")))
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
PLAN_MAX_GLOBAL_SUMMARY_FUNCTIONS = max(
    10, int(os.getenv("PLAN_MAX_GLOBAL_SUMMARY_FUNCTIONS", "60"))
)
PLAN_REQUIRED_FIELDS = {
    "scenario_type",
    "target_function",
    "src_function",
    "dst_function",
    "navigation_scenario",
    "test_scenario",
}
AUDIT_REQUIRED_FIELDS = {
    "has_exploration_gap",
    "entry_page_ref",
    "entry_name",
    "test_scenario",
}
VALID_SCENARIO_TYPES = {"intra_test", "cross_test"}
SCENARIO_ACTION_PATTERN = re.compile(
    r"\b(?:click|long\s+press|type|enter|swipe|scroll|back|open|launch|switch|select|"
    r"check|send|save|submit|delete|clear)\b|"
    r"\u70b9\u51fb|\u957f\u6309|\u8f93\u5165|\u6ed1\u52a8|\u6eda\u52a8|\u8fd4\u56de|\u6253\u5f00|\u542f\u52a8|\u5207\u6362|\u9009\u62e9|\u52fe\u9009|\u53d1\u9001|\u4fdd\u5b58|\u63d0\u4ea4|\u5220\u9664|\u6e05\u9664",
    re.IGNORECASE,
)
PARAMETERIZED_CONTROL_PATTERN = re.compile(
    r"(click|select|switch\s+to|check|\u70b9\u51fb|\u9009\u62e9|\u5207\u6362\u5230|\u52fe\u9009)\s*"
    r"[\"'\u201c\u2018]([^\"'\u201d\u2019]{1,40})[\"'\u201d\u2019]\s*"
    r"(activity\s+button|activity\s+tab|filter\s+option|sort\s+option|category\s+option|radio\s+button|checkbox|tab|option|"
    r"\u6d3b\u52a8\u6309\u94ae|\u6d3b\u52a8\u6807\u7b7e|\u7b5b\u9009\u9879|\u6392\u5e8f\u9879|\u5206\u7c7b\u9879|\u5355\u9009\u9879|\u590d\u9009\u9879|\u6807\u7b7e\u9875|\u9009\u9879)",
    re.IGNORECASE,
)
NAMED_MENU_CONTROL_PATTERN = re.compile(
    r"(click|open|select|\u70b9\u51fb|\u6253\u5f00|\u9009\u62e9)\s*"
    r"[\"'\u201c\u2018]([^\"'\u201d\u2019]{1,40})[\"'\u201d\u2019]\s*"
    r"(menu\s+item|navigation\s+item|feature\s+item|entry|\u83dc\u5355\u9879|\u5bfc\u822a\u9879|\u529f\u80fd\u9879|\u5165\u53e3)",
    re.IGNORECASE,
)
UNQUOTED_AUDIT_CONTROL_PATTERN = re.compile(
    r"(?:click/open/select|click|open|select|long\s+press|switch\s+to|"
    r"\u70b9\u51fb/\u6253\u5f00/\u9009\u62e9|\u70b9\u51fb|\u6253\u5f00|\u9009\u62e9|\u957f\u6309|\u5207\u6362\u5230)\s*"
    r"(?:the\s+)?([^\u2192\uff0c,.\u3002\uff1b;]{1,60}?)"
    r"(?=\s*(?:\u2192|->|[\uff0c,.\u3002\uff1b;]|and\s+(?:observe|record|verify|check)|\u5e76\u89c2\u5bdf|\u5e76\u8bb0\u5f55|$))",
    re.IGNORECASE,
)
QUOTED_ACTION_CONTROL_PATTERN = re.compile(
    r"(?:click/open|click|open|select|long\s+press|\u70b9\u51fb/\u6253\u5f00|\u70b9\u51fb|\u6253\u5f00|\u9009\u62e9)\s*"
    r"(?:the\s+)?"
    r"(?:\u622a\u56fe(?:\u548c\u9875\u9762\u63cf\u8ff0)?\u4e2d\s*)?(?:\u771f\u5b9e)?(?:\u53ef\u89c1\u7684?)?\s*"
    r"[\"'\u201c\u2018]\s*([^\"'\u201d\u2019]{1,40}?)\s*[\"'\u201d\u2019]",
    re.IGNORECASE,
)
NAMED_ICON_CONTROL_PATTERN = re.compile(
    r"([A-Za-z][A-Za-z0-9_+\-/]{0,39})\s*(icon|\u56fe\u6807)",
    re.IGNORECASE,
)
QUOTED_ICON_CONTROL_PATTERN = re.compile(
    r"[\"'\u201c\u2018]\s*([^\"'\u201d\u2019]{1,20})\s*[\"'\u201d\u2019]\s*\u56fe\u6807"
)
PARENTHESIZED_ICON_CONTROL_PATTERN = re.compile(
    r"[\uff08(]\s*([^\uff08\uff09()]{1,20}?)\s*\u56fe\u6807\s*[\uff09)]"
)
CHINESE_ICON_CONTROL_PATTERN = re.compile(
    r"(?:\u70b9\u51fb|\u9009\u62e9|\u627e\u5230)\s*"
    r"(?:(?:\u9876\u90e8\u5de5\u5177\u680f|\u5de6\u4e0a\u89d2|\u53f3\u4e0a\u89d2|\u5de6\u4e0b\u89d2|\u53f3\u4e0b\u89d2|\u5e95\u90e8\u5bfc\u822a\u680f)(?:\u4e2d|\u4e0a|\u7684)?\s*)?"
    r"(?:\u771f\u5b9e)?(?:\u53ef\u89c1\u7684?)?\s*([\u4e00-\u9fffA-Za-z0-9_+\-/]{1,12})\s*\u56fe\u6807",
    re.IGNORECASE,
)
SCENARIO_PAGE_PATTERN = re.compile(
    r"(?:(?:\b(?:on|in)\s+(?:the\s+)?)|\u5728)[\"'\u201c\u2018](.{1,60}?)[\"'\u201d\u2019]"
    r"(?:\s+(?:page|screen)|\u9875\u9762|\u9875|\u4e2d)",
    re.IGNORECASE,
)
SCENARIO_INPUT_VALUE_PATTERN = re.compile(
    r"((?:type|enter)(?:\s+text)?\s*|\u8f93\u5165(?:\u6587\u672c)?\s*)"
    r"[\"'\u201c\u2018][^\"'\u201d\u2019]{1,100}[\"'\u201d\u2019]",
    re.IGNORECASE,
)
SCENARIO_EMPTY_INPUT_PATTERN = re.compile(
    r"(?:(?:enter|type)\s+(?:no\s+text|nothing)\s+(?:in|into)\s+(?:the\s+)?|"
    r"\u4e0d\u8f93\u5165\u4efb\u4f55\u6587\u672c\u5230\s*)"
    r"([\"'\u201c\u2018][^\"'\u201d\u2019]{1,60}[\"'\u201d\u2019]\s*(?:input|field|text\s+box|\u8f93\u5165\u6846))",
    re.IGNORECASE,
)
SCENARIO_QUOTED_TOKEN_PATTERN = re.compile(
    r"[\"'\u201c\u2018]([^\"'\u201d\u2019]{1,60})[\"'\u201d\u2019]"
)


def _normalize_exploration_mode(value: str | None) -> str:
    return "core"


def _wait_target_app_foreground(device, package_name: str) -> bool:
    """Poll and wait for target app to enter foreground; max APP_START_MAX_WAIT seconds."""
    package_name = package_name.strip()
    if not package_name:
        return True
    deadline = time.monotonic() + APP_START_MAX_WAIT
    while True:
        try:
            current = str(device.current_package() or "").strip()
        except Exception:
            current = ""
        if current == package_name:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(APP_START_POLL_INTERVAL)


def _ensure_target_app(device, package_name: str, current_page: Page | None = None) -> Page | None:
    package_name = package_name.strip()
    if not package_name:
        return current_page

    try:
        current_package = str(device.current_package() or "").strip()
    except Exception as exc:
        print(f"[Scenario Planning] Failed to get current foreground package name: {exc}")
        current_package = ""

    needs_reset = False
    if current_package and current_package != package_name:
        print(
            f"[Scenario Planning] Current foreground app is {current_package}, target app is {package_name}, "
            "preparing to restart target app."
        )
        needs_reset = True
    elif current_page is None and current_package == package_name:
        print("[Scenario Planning] Current page is empty and target app is already in foreground; capturing page directly.")
        try:
            return capture_current_page(device)
        except ScreenshotUnavailableError as exc:
            print(f"[Scenario Planning] Foreground screenshot is unavailable, recovering via back key first: {exc}")
            for attempt in range(1, PLAN_SCREENSHOT_BACK_ATTEMPTS + 1):
                try:
                    print(
                        "[Scenario Planning] Black screen recovery: "
                        f"Pressing back key for attempt {attempt}/{PLAN_SCREENSHOT_BACK_ATTEMPTS}."
                    )
                    device.back()
                    time.sleep(APP_RESTART_RETRY_DELAY)
                    recovered_package = str(device.current_package() or "").strip()
                    if recovered_package != package_name:
                        print(
                            "[Scenario Planning] Target app left foreground after back press, switching to restart recovery: "
                            f"current={recovered_package or 'unknown'}"
                        )
                        break
                    return capture_current_page(device)
                except ScreenshotUnavailableError as back_exc:
                    print(f"[Scenario Planning] Screenshot still unavailable after back press: {back_exc}")
                except Exception as back_exc:
                    print(f"[Scenario Planning] Back recovery failed: {back_exc}")
            needs_reset = True
    elif current_page is None:
        print("[Scenario Planning] Unable to confirm target app is in foreground, preparing to restart target app.")
        needs_reset = True

    if not needs_reset:
        return current_page

    last_screenshot_error: ScreenshotUnavailableError | None = None
    for attempt in range(1, PLAN_APP_RESTART_ATTEMPTS + 1):
        try:
            print(f"[Scenario Planning] Restarting target app (attempt {attempt}/{PLAN_APP_RESTART_ATTEMPTS}). ")
            device.restart_app(package_name)
            if _wait_target_app_foreground(device, package_name):
                time.sleep(APP_RESTART_SETTLE_DELAY)
                return capture_current_page(device)
            print(
                "[Scenario Planning] Target app still did not enter foreground after restart."
            )
        except ScreenshotUnavailableError as exc:
            last_screenshot_error = exc
            print(f"[Scenario Plan] Screenshot is still unavailable after restart: {exc}")
        except Exception as exc:
            print(f"[Scenario Plan] Failed to launch target app: {exc}")
        if attempt < PLAN_APP_RESTART_ATTEMPTS:
            time.sleep(APP_RESTART_RETRY_DELAY)
    if last_screenshot_error is not None:
        raise last_screenshot_error
    return None


def _safe_capture_current_page(device) -> Page | None:
    try:
        return capture_current_page(device)
    except ScreenshotUnavailableError:
        raise
    except Exception as exc:
        print(f"[Scenario Plan] Failed to capture current page: {exc}")
        return None


def _clip_summary_text(value: object, limit: int = PLAN_SUMMARY_FIELD_MAX_CHARS) -> str:
    text = _clean_natural_text(value)
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 3)] + "..."


def _normalize_run_phase(value: object) -> str:
    text = str(value or "all").strip().lower()
    if text == "intra":
        return "intra"
    return "all"


def _count_ffg_stats(graph: FlowGraph) -> dict[str, int]:
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


def _node_test_finished(node: FunctionNode | None) -> bool:
    """True only for genuinely completed (successful + covered) functions.

    This is the completion-audit gate; blocked/deferred/unavailable are NOT
    audited even though they are terminal.
    """
    if node is None:
        return False
    return (node.test_status or "not_started") == "completed"


def _node_is_terminal(node: FunctionNode | None) -> bool:
    """True for any terminal status (completed/blocked/deferred/unavailable).

    Terminal functions leave the active scheduling queue and never get planned
    again in this run; only `completed` is additionally eligible for the
    coverage-gap audit.
    """
    if node is None:
        return True
    return (node.test_status or "not_started") in TERMINAL_FUNCTION_STATUSES


def _node_attempt_count(node: FunctionNode | None) -> int:
    if node is None:
        return 0
    return len(node.scenarios)


def _node_cross_testable(node: FunctionNode | None) -> bool:
    """Cross-function testing only accepts functions that are not in a failed/crashed/yielded-closed state."""
    if node is None:
        return False
    return (node.test_status or "not_started") not in {
        "blocked",
        "deferred",
        "unavailable",
    }


def _node_intra_attempt_count(node: FunctionNode | None) -> int:
    """Number of intra-function scenarios, excluding cross-function scenarios, used for single-function budget."""
    if node is None:
        return 0
    return sum(
        1
        for scenario in node.scenarios
        if not str(scenario or "").startswith("[cross:")
    )


def _function_completion_met(graph: FlowGraph, node: FunctionNode) -> bool:
    """Completion criteria: coverage reached + successful business operations exist + sufficient evidence for the function itself.

    The same page can be shared by multiple functions, and operations on the page are credited to all owners.
    Therefore, a function might "inherit" qualifying coverage before it has been tested even once.
    Before concluding, evidence from this function itself is required: no uncovered controls remain, or it has
    executed at least PLAN_MIN_SCENARIOS_BEFORE_COMPLETE intra-function scenarios (or reached scenario budget).
    """
    intra_attempts = _node_intra_attempt_count(node)
    budget_reached = intra_attempts >= PLAN_MAX_SCENARIOS_PER_FUNCTION
    uncovered_controls = graph.function_coverage_details(node.id)[
        "uncovered_controls"
    ]
    own_evidence = (
        not uncovered_controls
        or intra_attempts >= PLAN_MIN_SCENARIOS_BEFORE_COMPLETE
    )
    return (
        node.coverage_percent >= PLAN_FUNCTION_COVERAGE_TARGET
        and graph.function_has_successful_business_operation(node.id)
        and (own_evidence or budget_reached)
    )


def _normalize_and_reopen_undercovered_functions(graph: FlowGraph) -> list[str]:
    """Normalize entry names and reopen completed nodes below derived coverage."""
    reopened: list[str] = []
    for node in graph.functions.values():
        name = str(node.name or "").strip()
        if ENTRY_FUNCTION_NAME_PATTERN.search(name):
            business_name = ENTRY_FUNCTION_NAME_PATTERN.sub("", name).strip(" -_/\uff1a:")
            if business_name:
                node.name = business_name
        graph.recompute_function_coverage(node.id)
        coverage_details = graph.function_coverage_details(node.id)
        has_uncovered = bool(coverage_details["uncovered_controls"])
        # Planned but left no control and operation records: its completed state has no
        # verifiable coverage basis and requires re-auditing. Unplanned nodes are excluded to avoid preempting
        # breadth-first scheduling priority for truly unstarted functions.
        never_observed = (
            bool(node.scenarios)
            and int(coverage_details["total_controls"] or 0) == 0
            and not graph.function_executed_operation_keys(node.id)
        )
        # Reopening must be driven by "program-derived coverage facts": uncovered controls still remain, or
        # no controls have ever been recorded under this function (indicating it was never truly observed). Non-coverage-related heuristics
        # (such as entry naming + page count) will flip back and forth with completion criteria, wasting scheduling rounds.
        needs_reopen = (
            (has_uncovered or never_observed)
            and node.coverage_percent < PLAN_FUNCTION_COVERAGE_TARGET
            and (node.test_status or "not_started") == "completed"
        )
        if not needs_reopen:
            continue
        attempts = _node_intra_attempt_count(node)
        if attempts >= PLAN_MAX_SCENARIOS_PER_FUNCTION:
            print(
                "[Scenario Plan] Completed node has insufficient programmatic coverage, re-auditing despite reaching old scenario budget: "
                f"{node.id} | coverage={node.coverage_percent}%"
            )
        node.test_status = "in_progress"
        reopened.append(node.id)
    return reopened


def _unfinished_function_nodes(graph: FlowGraph) -> list[FunctionNode]:
    return [node for node in graph.functions.values() if not _node_is_terminal(node)]


def _function_sort_index(node: FunctionNode) -> int:
    match = re.search(r"\d+", str(node.id or ""))
    return int(match.group(0)) if match else 10**9


def _choose_function_match(matches: list[FunctionNode]) -> str:
    if not matches:
        return ""
    unfinished = [node for node in matches if not _node_is_terminal(node)]
    candidates = unfinished or matches
    return sorted(
        candidates,
        key=lambda node: (
            0 if not _node_attempt_count(node) else 1,
            _node_attempt_count(node),
            _function_sort_index(node),
        ),
    )[0].id


def _priority_unfinished_nodes(
    graph: FlowGraph,
    failure_counts: dict[str, int] | None = None,
) -> list[FunctionNode]:
    """Sort incomplete functions by "no failure first, unstarted first, fewer attempts first".

    Functions with recent execution failures are placed later, letting the scheduler move to other incomplete functions first,
    avoiding repeated retries stuck on a single failing function.
    """
    failure_counts = failure_counts or {}
    unfinished = _unfinished_function_nodes(graph)
    return sorted(
        unfinished,
        key=lambda node: (
            0 if int(failure_counts.get(node.id, 0) or 0) == 0 else 1,
            0 if not _node_attempt_count(node) else 1,
            0 if (node.test_status or "not_started") == "not_started" else 1,
            _node_attempt_count(node),
            0 if node.page_ids else 1,
            _function_sort_index(node),
        ),
    )


def _cross_pair_key(src_function_id: str, dst_function_id: str) -> tuple[str, str]:
    return (str(src_function_id or "").strip(), str(dst_function_id or "").strip())


def _tested_cross_pair_keys(graph: FlowGraph) -> set[frozenset[str]]:
    """Undirected set of tested cross-function combinations: tested in either direction of the same relation counts as covered.

    Scan the ``[cross:a->b]`` tag on scenarios directly, instead of traversing function_dependencies.
    ``add_cross_scenario`` does not create dependency edges, dependency traversal would keep all candidates derived from shared resources
    pending forever, causing the same combination to be repeatedly planned each round.
    """
    tested: set[frozenset[str]] = set()
    for node in graph.functions.values():
        for scenario in node.scenarios:
            match = CROSS_SCENARIO_MARKER_PATTERN.match(str(scenario or ""))
            if not match:
                continue
            src_id = match.group(1).strip()
            dst_id = match.group(2).strip()
            if src_id and dst_id:
                tested.add(frozenset((src_id, dst_id)))
    return tested


def _add_cross_candidate(
    candidates: list[tuple[str, str, str]],
    seen: set[tuple[str, str]],
    graph: FlowGraph,
    src_function_id: str,
    dst_function_id: str,
    reason: str,
):
    src_function_id, dst_function_id = _cross_pair_key(src_function_id, dst_function_id)
    if not src_function_id or not dst_function_id or src_function_id == dst_function_id:
        return
    if graph.get_node(src_function_id) is None or graph.get_node(dst_function_id) is None:
        return
    key = frozenset((src_function_id, dst_function_id))
    if key in seen:
        return
    seen.add(key)
    candidates.append((src_function_id, dst_function_id, reason))


def _candidate_cross_pairs(graph: FlowGraph) -> list[tuple[str, str, str]]:
    candidates: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()

    for dep in graph.function_dependencies:
        reason = _clean_natural_text(dep.resource_variable) or "Function dependency"
        _add_cross_candidate(
            candidates,
            seen,
            graph,
            dep.src_function_id,
            dep.dst_function_id,
            f"Function dependency: {reason}",
        )

    # When shared resource variables have no extra direction, supplement candidates in stable function order; explicit dependencies always take precedence.
    resource_owners: dict[str, list[tuple[FunctionNode, str]]] = {}
    for node in sorted(graph.functions.values(), key=_function_sort_index):
        seen_resources: set[str] = set()
        for raw_resource in node.resource_variables:
            resource = _clean_natural_text(raw_resource)
            resource_key = _normalize_similarity_key(resource)
            if not resource_key or resource_key in seen_resources:
                continue
            seen_resources.add(resource_key)
            resource_owners.setdefault(resource_key, []).append((node, resource))

    paired_functions = {frozenset(pair) for pair in seen}
    for owners in resource_owners.values():
        if len(owners) < 2:
            continue
        for src_index, (src_node, resource) in enumerate(owners[:-1]):
            for dst_node, _ in owners[src_index + 1:]:
                pair = frozenset((src_node.id, dst_node.id))
                if pair in paired_functions:
                    continue
                _add_cross_candidate(
                    candidates,
                    seen,
                    graph,
                    src_node.id,
                    dst_node.id,
                    f"Shared resource variable: {resource}",
                )
                paired_functions.add(pair)

    return candidates


def _pending_cross_pairs(graph: FlowGraph) -> list[tuple[str, str, str]]:
    tested = _tested_cross_pair_keys(graph)
    return [
        (src_id, dst_id, reason)
        for src_id, dst_id, reason in _candidate_cross_pairs(graph)
        if frozenset((src_id, dst_id)) not in tested
        and _node_cross_testable(graph.get_node(src_id))
        and _node_cross_testable(graph.get_node(dst_id))
    ][:MAX_CROSS_TEST_PAIRS]


def _coverage_ready_for_audit(graph: FlowGraph, run_phase: str) -> tuple[bool, str]:
    unfinished = _unfinished_function_nodes(graph)
    if unfinished:
        labels = ", ".join(_function_label(node) for node in unfinished[:8])
        return False, f"Incomplete function nodes still remain: {labels}"
    if _normalize_run_phase(run_phase) == "all":
        pending_cross = _pending_cross_pairs(graph)
        if pending_cross:
            labels = ", ".join(
                f"{_function_label(graph.get_node(src_id))}->{_function_label(graph.get_node(dst_id))}"
                for src_id, dst_id, _ in pending_cross[:6]
            )
            return False, f"Key cross-function combinations still untested: {labels}"
    return True, "Both intra-function and key cross-function coverage are completed"


def _function_label(node) -> str:
    if node is None:
        return "Unknown function"
    return _clean_natural_text(node.name or node.description) or "Unnamed function"


def _page_label(page: Page | None) -> str:
    if page is None:
        return "Unknown page"
    return _clean_natural_text(page.name or page.description) or "Unnamed page"


def _page_node_label(page: Page | None) -> str:
    if page is None:
        return "Unknown page"
    return _clean_natural_text(page.name) or str(page.id or "").strip() or "Unnamed page"


def _is_black_screen_page(graph: FlowGraph, current_page: Page | None) -> bool:
    page_id = _resolve_current_page_id(graph, current_page)
    page = graph.get_page(page_id) if page_id else None
    return bool(page and _normalize_similarity_key(page.name) == _normalize_similarity_key("\u9ed1\u5c4f\u6216\u9875\u9762\u4e0d\u53ef\u7528"))


def _matches_black_screen_path(graph: FlowGraph, scenario: str) -> bool:
    scenario_key = _normalize_similarity_key(scenario)
    if not scenario_key:
        return False
    black_page_ids = {
        str(page.id or "")
        for page in graph.active_pages()
        if _normalize_similarity_key(page.name) == _normalize_similarity_key("\u9ed1\u5c4f\u6216\u9875\u9762\u4e0d\u53ef\u7528")
    }
    for transition in graph.page_transitions:
        if transition.dst_page_id not in black_page_ids:
            continue
        failed_action = re.split(
            r"[\uff1b;]?\s*\u7ed3\u679c[:\uff1a]",
            str(transition.operation or ""),
            maxsplit=1,
        )[0].strip()
        action_key = _normalize_similarity_key(failed_action)
        if action_key and (
            action_key in scenario_key
            or _is_similar_scenario(failed_action, scenario)
        ):
            return True
    return False


def _resolve_current_page_id(graph: FlowGraph, current_page: Page | None) -> str:
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


def _attach_current_page_identity(graph: FlowGraph, current_page: Page | None) -> Page | None:
    if not isinstance(current_page, Page):
        return current_page
    page_id = _resolve_current_page_id(graph, current_page)
    recorded_page = graph.get_page(page_id)
    if recorded_page is None:
        return current_page
    current_page.id = recorded_page.id
    current_page.name = recorded_page.name
    current_page.description = recorded_page.description
    current_page.controls = list(recorded_page.controls)
    return current_page


def _refresh_current_page_observation(
    graph: FlowGraph,
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
        print(f"[Scenario Plan] Page refresh before execution failed, continuing with previous round state: {exc}")
        return current_page
    if not isinstance(observed, Page):
        return current_page
    previous_id = str(current_page.id or "").strip() if isinstance(current_page, Page) else ""
    observed = _attach_current_page_identity(graph, observed)
    observed_id = str(observed.id or "").strip()
    if previous_id and observed_id != previous_id:
        print(
            "[Scenario Plan] Pre-execution refresh found current page has changed: "
            f"state={previous_id}, observed={observed_id or 'Unmatched new page'}"
        )
    return observed


def _build_current_location_summary(graph: FlowGraph, current_page: Page | None) -> str:
    page_id = _resolve_current_page_id(graph, current_page)
    page = graph.get_page(page_id) if page_id else None
    owner_ids = graph.get_page_owners(page_id) if page_id else []

    lines = ["[Current Location]"]
    if owner_ids:
        owner_labels = [
            f"{owner_id} | {_function_label(graph.get_node(owner_id))}"
            for owner_id in owner_ids
        ]
        lines.append(f"- Current page candidate functions: {'; '.join(owner_labels)}")
    else:
        lines.append("- Current page candidate functions: No recorded functions matched")

    if page is not None:
        lines.append(f"- Current page node: {page.id} | {_page_node_label(page)}")
        lines.append(f"- Current page owners: {owner_ids}")
    else:
        lines.append("- Current page node: No recorded page matched")
        if isinstance(current_page, Page) and current_page.activity:
            lines.append(f"- Current Activity: {_clean_natural_text(current_page.activity)}")

    return "\n".join(lines)


def _strip_internal_ids_from_text(text: str) -> str:
    if not text:
        return text
    text = PAGE_ID_ANNOTATION_PATTERN.sub("", text)
    text = PAGE_ID_PATTERN.sub("", text)
    text = FUNCTION_ID_PATTERN.sub("", text)
    text = re.sub(r"\uff08\s*\uff09", "", text)
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def _clean_natural_text(value: object) -> str:
    text = str(value or "").strip()
    if text.lower() == "none":
        return ""
    return _strip_internal_ids_from_text(text)


def _resolve_function_ref(graph: FlowGraph, value: object) -> str:
    text = _normalize_optional_id(value)
    if not text:
        return ""
    if text in graph.functions:
        return text

    normalized = _normalize_for_match(text)
    if not normalized:
        return ""

    matches: list[FunctionNode] = []
    for node in graph.functions.values():
        candidates = (
            node.name,
            node.description,
            f"{node.name} {node.description}",
        )
        if any(_normalize_for_match(candidate) == normalized for candidate in candidates):
            matches.append(node)
    if matches:
        return _choose_function_match(matches)

    matches = []
    for node in graph.functions.values():
        candidates = (
            node.name,
            node.description,
            f"{node.name} {node.description}",
        )
        if any(normalized in _normalize_for_match(candidate) for candidate in candidates):
            matches.append(node)
    if matches:
        return _choose_function_match(matches)
    return ""


def _normalize_for_match(value: object) -> str:
    return re.sub(r"\s+", "", str(value or "").strip().lower())


def _normalize_similarity_key(value: object) -> str:
    return TEXT_NORMALIZE_PATTERN.sub("", str(value or "").strip().lower())


def _is_similar_scenario(left: str, right: str) -> bool:
    left_key = _normalize_similarity_key(left)
    right_key = _normalize_similarity_key(right)
    if not left_key or not right_key:
        return False
    if left_key == right_key:
        return True
    left_operations = _scenario_operation_family_keys(left)
    right_operations = _scenario_operation_family_keys(right)
    if left_operations and right_operations and left_operations == right_operations:
        # Treat as duplicate only when the operation family sets are completely identical. Partial overlap (e.g. sharing 2 out of 3 steps) indicates
        # there is an untested different control, and deduplication would directly discard the coverage opportunity for that control.
        return True
    left_action_chain = _scenario_action_chain_key(left)
    right_action_chain = _scenario_action_chain_key(right)
    if (
        left_action_chain
        and right_action_chain
        and left_action_chain == right_action_chain
    ):
        return True
    left_family = _parameterized_scenario_family(left)
    right_family = _parameterized_scenario_family(right)
    if left_family and left_family == right_family:
        return True
    left_quoted_control = _quoted_action_control(left)
    right_quoted_control = _quoted_action_control(right)
    if (
        left_quoted_control
        and right_quoted_control
        and left_quoted_control != right_quoted_control
    ):
        return False
    left_menu_control = _named_menu_control(left)
    right_menu_control = _named_menu_control(right)
    if (
        left_menu_control
        and right_menu_control
        and left_menu_control[1] == right_menu_control[1]
        and left_menu_control[0] != right_menu_control[0]
    ):
        return False
    left_audit_control = _unquoted_audit_control(left)
    right_audit_control = _unquoted_audit_control(right)
    if (
        left_audit_control
        and right_audit_control
        and left_audit_control != right_audit_control
    ):
        return False
    left_icon_control = _named_icon_control(left)
    right_icon_control = _named_icon_control(right)
    if left_icon_control and right_icon_control and left_icon_control != right_icon_control:
        return False
    left_input_key = _normalize_input_scenario_key(left)
    right_input_key = _normalize_input_scenario_key(right)
    if (
        left_input_key
        and right_input_key
        and _input_scenario_control_tokens(left) == _input_scenario_control_tokens(right)
        and SequenceMatcher(None, left_input_key, right_input_key).ratio() >= 0.70
    ):
        return True
    for marker in DUPLICATE_SCENARIO_ACTION_MARKERS:
        if marker in left_key and marker in right_key:
            return True
    shorter, longer = sorted((left_key, right_key), key=len)
    if len(shorter) >= 24 and shorter in longer and len(shorter) / max(len(longer), 1) >= 0.78:
        return True
    return SequenceMatcher(None, left_key, right_key).ratio() >= 0.84


def _scenario_action_chain_key(value: str) -> str:
    """Canonicalize what is executed, excluding paraphrased result clauses."""
    action_chain = re.split(
        r"(?:\u2192|->)?\s*(?:\u786e\u8ba4|\u89c2\u5bdf|\u9a8c\u8bc1|\u68c0\u67e5|\u8bb0\u5f55)(?:\u5b9e\u9645)?",
        str(value or "").strip(),
        maxsplit=1,
    )[0]
    key = _normalize_similarity_key(action_chain)
    return key if SCENARIO_ACTION_PATTERN.search(action_chain) else ""


def _scenario_operation_family_keys(value: str) -> frozenset[str]:
    """Canonical device-action families, excluding result observations."""
    action_text = re.split(
        r"(?:\u2192|->)?\s*(?:\u786e\u8ba4|\u89c2\u5bdf|\u9a8c\u8bc1|\u68c0\u67e5|\u8bb0\u5f55|\u770b\u5230)(?:\u5b9e\u9645)?",
        str(value or "").strip(),
        maxsplit=1,
    )[0]
    clauses = re.split(
        r"(?:\u2192|->|\uff0c|,|\uff1b|;|\u3002|(?:\u7136\u540e|\u968f\u540e|\u4e4b\u540e|\u540e\u518d)|\u5e76(?=\u70b9\u51fb|\u9009\u62e9|\u5207\u6362|\u52fe\u9009|\u8f93\u5165|\u8fd4\u56de|\u6253\u5f00))",
        action_text,
    )
    return frozenset(
        family
        for clause in clauses
        if SCENARIO_ACTION_PATTERN.search(clause)
        and (family := operation_family_key(clause.strip()))
    )


def _named_menu_control(value: str) -> tuple[str, str] | None:
    match = NAMED_MENU_CONTROL_PATTERN.search(str(value or ""))
    if match is None:
        return None
    return (
        _normalize_similarity_key(match.group(2)),
        _normalize_similarity_key(match.group(3)),
    )


def _unquoted_audit_control(value: str) -> str:
    match = UNQUOTED_AUDIT_CONTROL_PATTERN.search(str(value or ""))
    if match is None:
        return ""
    label = re.sub(
        r"\s+(?:button|menu\s+item|navigation\s+item|feature\s+item|entry|tab|option|control)\s*$",
        "",
        match.group(1),
        flags=re.IGNORECASE,
    )
    return _normalize_similarity_key(label)


def _quoted_action_control(value: str) -> str:
    match = QUOTED_ACTION_CONTROL_PATTERN.search(str(value or ""))
    return _normalize_similarity_key(match.group(1)) if match else ""


def _named_icon_control(value: str) -> str:
    text = str(value or "")
    for pattern in (
        QUOTED_ICON_CONTROL_PATTERN,
        PARENTHESIZED_ICON_CONTROL_PATTERN,
        NAMED_ICON_CONTROL_PATTERN,
        CHINESE_ICON_CONTROL_PATTERN,
    ):
        match = pattern.search(text)
        if match:
            return _normalize_similarity_key(match.group(1))
    return ""


def _normalize_input_scenario_key(value: str) -> str:
    text = str(value or "").strip()
    if not re.search(r"\u8f93\u5165\u6846|\b(?:input|field|text\s+box)\b", text, re.IGNORECASE):
        return ""
    text = SCENARIO_EMPTY_INPUT_PATTERN.sub(r"enter text '<value>' into \1", text)
    text = SCENARIO_INPUT_VALUE_PATTERN.sub(r"\1<value>", text)
    return _normalize_similarity_key(text)


def _input_scenario_control_tokens(value: str) -> tuple[str, ...]:
    text = str(value or "").strip()
    text = SCENARIO_EMPTY_INPUT_PATTERN.sub(r"enter text '<value>' into \1", text)
    text = SCENARIO_INPUT_VALUE_PATTERN.sub(r"\1<value>", text)
    action_text = re.split(
        r"\b(?:observe|verify|confirm|check)\b|\u89c2\u5bdf|\u9a8c\u8bc1|\u786e\u8ba4",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    return tuple(sorted({
        token
        for token in (
            _normalize_similarity_key(item)
            for item in SCENARIO_QUOTED_TOKEN_PATTERN.findall(action_text)
        )
        if token and token != "value"
    }))


def _parameterized_scenario_family(value: str) -> tuple[str, str, str] | None:
    text = str(value or "").strip()
    control_match = PARAMETERIZED_CONTROL_PATTERN.search(text)
    if control_match is None:
        return None
    page_match = SCENARIO_PAGE_PATTERN.search(text)
    page_key = _normalize_similarity_key(page_match.group(1)) if page_match else ""
    action = control_match.group(1).lower()
    action_family = "select" if action in {
        "click", "select", "switch to", "check",
        "\u70b9\u51fb", "\u9009\u62e9", "\u5207\u6362\u5230", "\u52fe\u9009",
    } else action
    control_type = control_match.group(3)
    control_family = _normalize_similarity_key(
        control_match.group(2) if control_type == "\u9009\u9879" else control_type
    )
    return page_key, action_family, control_family


def _normalize_optional_id(value: object) -> str:
    text = str(value or "").strip()
    if text.lower() in {"none", "null"}:
        return ""
    return text


def _sanitize_planned_scenario(text: str) -> str:
    text = _strip_internal_ids_from_text(text)
    text = re.sub(r"\u8fdb\u5165\s*\u5bf9\u5e94\u7684\s*\u9875\u9762", "\u8fdb\u5165\u76ee\u6807\u9875\u9762", text)
    text = re.sub(r"\u5728\u5f53\u524d\s*\u9875\u9762", "\u5728\u5f53\u524d\u9875\u9762", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"\s+([\uff0c\u3002\uff01\uff1f\uff1b\uff1a])", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def _sanitize_navigation_scenario(text: str, graph: FlowGraph, current_page: Page | None) -> str:
    text = _sanitize_planned_scenario(text)
    if not text:
        return ""
    page_id = _resolve_current_page_id(graph, current_page)
    page = graph.get_page(page_id) if page_id else None
    current_label = _page_node_label(page) if page is not None else _page_label(current_page)
    current_key = _normalize_similarity_key(current_label)
    starts_from_home = re.match(r"^(\u4ece|\u5148\u4ece)?(\u9996\u9875|\u4e3b\u9875|\u4e3b\u9875\u9762|\u542f\u52a8\u9875|\u5e94\u7528\u9996\u9875)", text)
    current_is_home = bool(re.search(r"(\u9996\u9875|\u4e3b\u9875|\u4e3b\u9875\u9762|home|main)", current_key, re.IGNORECASE))
    if starts_from_home and not current_is_home:
        text = re.sub(r"^(\u4ece|\u5148\u4ece)?(\u9996\u9875|\u4e3b\u9875|\u4e3b\u9875\u9762|\u542f\u52a8\u9875|\u5e94\u7528\u9996\u9875)", "\u4ece\u5f53\u524d\u4f4d\u7f6e", text, count=1)
        print(
            "[Scenario Planning] Navigation scenario suspected of starting from home by default; changed to start from current position: "
            f"current={current_label}"
        )
    return text


def _invoke_llm(llm, system_prompt: str, message_content: list[dict]) -> dict:
    return invoke_llm_json_with_retry(
        llm,
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=message_content),
        ],
        log_prefix="[Scenario Planning]",
        json_block_pattern=JSON_BLOCK_PATTERN,
    )


def _validate_plan_result(
    result: object,
    package_name: str = "",
) -> list[str]:
    if not isinstance(result, dict) or not result:
        return ["empty_or_non_object_result"]

    errors: list[str] = []
    missing_fields = sorted(PLAN_REQUIRED_FIELDS - set(result))
    if missing_fields:
        errors.append(f"missing_fields={missing_fields}")

    scenario_type = str(result.get("scenario_type", "")).strip()
    if scenario_type and scenario_type not in VALID_SCENARIO_TYPES:
        errors.append(f"invalid_scenario_type={scenario_type!r}")
    target_function = _normalize_optional_id(result.get("target_function"))
    src_function = _normalize_optional_id(result.get("src_function"))
    dst_function = _normalize_optional_id(result.get("dst_function"))
    if scenario_type == "intra_test" and not FUNCTION_ID_PATTERN.fullmatch(target_function):
        errors.append(f"intra_test_requires_func_id={target_function!r}")
    if scenario_type == "cross_test":
        if not FUNCTION_ID_PATTERN.fullmatch(src_function):
            errors.append(f"cross_test_requires_src_func_id={src_function!r}")
        if not FUNCTION_ID_PATTERN.fullmatch(dst_function):
            errors.append(f"cross_test_requires_dst_func_id={dst_function!r}")

    test_scenario = _normalize_optional_id(result.get("test_scenario"))
    risky_intent = account_destructive_intent(
        result.get("navigation_scenario", ""),
        test_scenario,
    )
    if risky_intent:
        errors.append(f"account_safety_policy={risky_intent!r}")
    if scenario_type == "cross_test" and not test_scenario:
        errors.append("cross_test_requires_nonempty_test_scenario")
    if has_exhaustive_control_family_intent(test_scenario):
        errors.append("test_scenario_exhaustive_control_family")

    return errors


def _invoke_valid_plan(
    llm,
    system_prompt: str,
    message_content: list[dict],
    package_name: str = "",
) -> tuple[dict, list[str]]:
    last_errors: list[str] = []
    current_message_content = list(message_content)
    for attempt in range(1, PLAN_VALIDATION_RETRY_ATTEMPTS + 1):
        result = _invoke_llm(llm, system_prompt, current_message_content)
        if llm_json_result_failed(result):
            last_errors = ["llm_request_or_parse_failed"]
            print(
                "[Scenario Planning] Underlying call exhausted retries; stopping field-level duplicate requests; "
                "deferring to subsequent scheduler rounds to continue."
            )
            break
        errors = _validate_plan_result(
            result,
            package_name=package_name,
        )
        if not errors:
            return result, []
        last_errors = errors
        print(
            "[Scenario Planning] LLM returned invalid fields"
            f" (attempt {attempt}/{PLAN_VALIDATION_RETRY_ATTEMPTS}): {errors}"
        )
        current_message_content = list(message_content)
        current_message_content.append({
            "type": "text",
            "text": (
                "The previous output fields were invalid. Please re-output complete and valid JSON without omitting any fields."
                "For repeated controls with the same function that lead to the same result, select only one representative item;"
                "For parallel controls with different results, 2-3 items can be listed by exact name."
                "Do not require all, every, each, one by one, or traversing a certain category of controls."
                "Do not click, open, edit, or delete dynamic objects that are not recorded in the current scope and not created earlier in this scenario."
                "When downstream pages, results, or entries are not recorded, a conditional open ending is permitted: if clear and unexecuted results/entries appear,"
                "select one representative item and verify the direct result; if none appear, or if empty or unreachable, end."
                f"Field issues: {errors}"
            ),
        })
    return {}, last_errors


def _reset_planner_noop_fields() -> dict[str, object]:
    return {
        "planner_noop_count": 0,
        "planner_feedback": "",
        "planner_rejected_scenario": "",
    }


def _planner_noop_count(state: AgentState | None) -> int:
    if state is None:
        return 0
    try:
        return int(state.get("planner_noop_count", 0) or 0)
    except (TypeError, ValueError):
        return 0


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


def _node_representative_page(graph: FlowGraph, node: FunctionNode | None) -> Page | None:
    if node is None:
        return None
    page_ids = node.coverage_page_ids or node.page_ids
    for page_id in page_ids:
        page = graph.get_page(page_id)
        if page is not None:
            return page
    return None


def _node_uncovered_control_hints(
    graph: FlowGraph,
    node: FunctionNode | None,
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


def _forced_intra_scenario(
    graph: FlowGraph,
    node: FunctionNode,
    hint_index: int = 0,
    package_name: str = "",
) -> str:
    label = _function_label(node)
    representative_page = _node_representative_page(graph, node)
    entry_label = _page_label(representative_page) if representative_page is not None else "recorded page for this function"
    hints = _node_uncovered_control_hints(
        graph,
        node,
    )
    safe_hints: list[str] = []
    for hint in hints:
        violation = account_action_policy_violation(f"Click {hint}")
        if violation:
            print(
                "[Scenario Planning] Account safety policy skipped a logout or account-deletion control: "
                f"{hint} | {violation}"
            )
            continue
        safe_hints.append(hint)
    hints = safe_hints
    if not hints:
        return ""
    # Exploring only one control per round leaves each scenario with only 1-2 effective actions, while the fixed overhead of each round
    # (1 planning + 1 or more LLM calls with images for FFG update + screenshot + logcat) remains unchanged,
    # leading to very low diluted throughput. Here we provide multiple control families at once so a single scenario can cover testable points on the same page.
    batch: list[str] = []
    for offset in range(min(PLAN_SCENARIO_CONTROL_BATCH, len(hints))):
        candidate = hints[(hint_index + offset) % len(hints)]
        if candidate not in batch:
            batch.append(candidate)
    hint_text = ", ".join(f'"{hint}"' for hint in batch)
    if len(batch) == 1:
        coverage_clause = (
            f"In this round, explore the following visible control with no execution record ({hint_text}), observing its direct result;"
        )
    else:
        coverage_clause = (
            f"In this round, sequentially explore the following {len(batch)} visible controls with no execution records ({hint_text}), "
            "after clicking each control, observe its direct result; if entering a new interface, return to the previous level before continuing to the next;"
        )
    return (
        f"Perform a low-risk supplementary exploration around \"{label}\": try entering \"{entry_label}\" from the current page, "
        + coverage_clause
        + "do not re-execute controls that already have operation records;"
        "if the entry is unreachable, the target has no testable content, or only historical paths can be repeated, record the reason and end."
    )


def _forced_cross_scenario(graph: FlowGraph, src_id: str, dst_id: str, reason: str) -> str:
    src_label = _function_label(graph.get_node(src_id))
    dst_label = _function_label(graph.get_node(dst_id))
    reason = _clean_natural_text(reason) or "existing function dependency"
    return (
        f"Test cross-function relationship from \"{src_label}\" to \"{dst_label}\": first confirm source-function recorded resources related to \"{reason}\", "
        f"states, or results, then enter target function to observe whether that resource, state, or result is correctly read, displayed, or used;"
        "if the path is unreachable or conditions are insufficient, record the unreachable reason and end."
    )


def _force_progress_after_planner_noop(
    graph: FlowGraph,
    current_page: Page | None,
    exploration_mode: str,
    current_testing_function_id: str,
    current_testing_function_status: str,
    state: AgentState,
    reason: str,
    noop_count: int,
) -> dict | None:
    run_phase = _normalize_run_phase(state.get("run_phase", "all"))
    pending_cross = _pending_cross_pairs(graph) if run_phase == "all" else []
    current_node = graph.get_node(_normalize_optional_id(current_testing_function_id))
    reached_threshold = noop_count >= MAX_CONSECUTIVE_PLANNER_NOOPS

    def progress_log(first_attempt: str, threshold_attempt: str) -> str:
        return threshold_attempt if reached_threshold else first_attempt

    def release_unproductive_function(node: FunctionNode, message: str) -> dict:
        if (node.test_status or "not_started") == "not_started":
            graph.set_node_test_status(node.id, "in_progress")
        failure_counts = _normalize_failure_counts(
            state.get("function_execution_failure_counts")
        )
        release_count = int(failure_counts.get(node.id, 0) or 0) + 1
        failure_counts[node.id] = release_count
        print(f"{message}: {node.id} | {_function_label(node)}")
        if release_count >= PLAN_UNPRODUCTIVE_RELEASE_LIMIT and not _node_is_terminal(node):
            # Consecutive rounds have neither supplementary controls nor reachable paths: write terminal state,
            # otherwise it will stay in unfinished, blocking convergence and triggering prolonged idling.
            graph.set_node_test_status(node.id, "unavailable")
            failure_counts.pop(node.id, None)
            print(
                "[Scenario Planning] Function consecutively lacks executable scenarios; marking unavailable to release scheduling: "
                f"{node.id} | {_function_label(node)} ({release_count}/{PLAN_UNPRODUCTIVE_RELEASE_LIMIT})"
            )
        return {
            "ffg": graph,
            "current_page": current_page,
            "current_navigation_scenario": "",
            "current_test_scenario": "",
            "current_testing_function_id": "",
            "current_testing_function_status": "not_started",
            "current_plan": {
                "scenario_type": "status_update",
                "target_function_id": node.id,
                "src_function_id": "",
                "dst_function_id": "",
            },
            "function_execution_failure_counts": failure_counts,
            "workflow_complete": False,
            **_reset_planner_noop_fields(),
        }
    if pending_cross and (current_node is None or _node_is_terminal(current_node)):
        src_id, dst_id, cross_reason = pending_cross[0]
        scenario = _forced_cross_scenario(graph, src_id, dst_id, cross_reason)
        print(
            progress_log(
                "[Scenario Planning] Coverage insufficient; directly generating representative cross-functional supplementary scenario: ",
                "[Scenario Planning] Consecutive empty plans reached threshold; forcing progress on pending cross-functional combination: ",
            )
            +
            f"{_function_label(graph.get_node(src_id))} -> {_function_label(graph.get_node(dst_id))} "
            f"({noop_count}/{MAX_CONSECUTIVE_PLANNER_NOOPS})"
        )
        return {
            "ffg": graph,
            "current_page": current_page,
            "current_navigation_scenario": "",
            "current_test_scenario": scenario,
            "current_testing_function_id": current_testing_function_id,
            "current_testing_function_status": current_testing_function_status,
            "current_plan": {
                "scenario_type": "cross_test",
                "target_function_id": "",
                "src_function_id": src_id,
                "dst_function_id": dst_id,
            },
            "workflow_complete": False,
            **_reset_planner_noop_fields(),
        }
    if current_node is not None and "\u76f8\u4f3c\u5386\u53f2\u573a\u666f" in reason:
        return release_unproductive_function(
            current_node,
            "[Scenario Planning] Similar duplicate candidates consecutively reached threshold; releasing current function and lowering scheduling priority"
            f" ({noop_count}/{MAX_CONSECUTIVE_PLANNER_NOOPS})",
        )
    if current_node is not None and not _node_is_terminal(current_node):
        scenario = _forced_intra_scenario(
            graph,
            current_node,
            package_name=str(state.get("package_name", "") or ""),
        )
        if not scenario:
            return release_unproductive_function(
                current_node,
                "[Scenario Planning] No supplementary executable controls found after consecutive empty plans; releasing current function and rescheduling",
            )
        if (current_node.test_status or "not_started") == "not_started":
            graph.set_node_test_status(current_node.id, "in_progress")
        print(
            progress_log(
                "[Scenario Planning] Coverage insufficient; directly generating representative supplementary scenario: ",
                "[Scenario Planning] Consecutive empty plans reached threshold; forcing closure of previous round function: ",
            )
            +
            f"{current_node.id} | {_function_label(current_node)} "
            f"({noop_count}/{MAX_CONSECUTIVE_PLANNER_NOOPS})"
        )
        return {
            "ffg": graph,
            "current_page": current_page,
            "current_navigation_scenario": "",
            "current_test_scenario": scenario,
            "current_testing_function_id": current_node.id,
            "current_testing_function_status": "in_progress",
            "current_plan": {
                "scenario_type": "intra_test",
                "target_function_id": current_node.id,
                "src_function_id": "",
                "dst_function_id": "",
            },
            "workflow_complete": False,
            **_reset_planner_noop_fields(),
        }

    unfinished = _priority_unfinished_nodes(
        graph,
        _normalize_failure_counts(state.get("function_execution_failure_counts")),
    )
    if unfinished:
        node = unfinished[0]
        scenario = _forced_intra_scenario(
            graph,
            node,
            package_name=str(state.get("package_name", "") or ""),
        )
        if not scenario:
            return release_unproductive_function(
                node,
                "[Scenario Planning] Unfinished function has no supplementary executable controls; lowering its scheduling priority and replanning",
            )
        if (node.test_status or "not_started") == "not_started":
            graph.set_node_test_status(node.id, "in_progress")
        print(
            progress_log(
                "[Scenario Planning] Coverage insufficient; directly advancing representative supplementary testing for unfinished function: ",
                "[Scenario Planning] Consecutive empty plans reached threshold; forcing progress on unfinished function: ",
            )
            +
            f"{node.id} | {_function_label(node)} ({noop_count}/{MAX_CONSECUTIVE_PLANNER_NOOPS})"
        )
        return {
            "ffg": graph,
            "current_page": current_page,
            "current_navigation_scenario": "",
            "current_test_scenario": scenario,
            "current_testing_function_id": node.id,
            "current_testing_function_status": "in_progress",
            "current_plan": {
                "scenario_type": "intra_test",
                "target_function_id": node.id,
                "src_function_id": "",
                "dst_function_id": "",
            },
            "workflow_complete": False,
            **_reset_planner_noop_fields(),
        }

    ready_to_complete, complete_reason = _coverage_ready_for_audit(graph, run_phase)
    if ready_to_complete:
        print(
            "[Scenario Planning] Consecutive empty plans reached threshold and no remaining functions/cross-functions; ending planning directly: "
            f"{complete_reason}"
        )
        return {
            "ffg": graph,
            "current_page": current_page,
            "current_navigation_scenario": "",
            "current_test_scenario": "",
            "current_testing_function_id": "",
            "current_testing_function_status": "not_started",
            "current_plan": {
                "scenario_type": "status_update",
                "target_function_id": "",
                "src_function_id": "",
                "dst_function_id": "",
            },
            "workflow_complete": True,
            **_reset_planner_noop_fields(),
        }

    return None


def _planner_status_update_response(
    graph: FlowGraph,
    current_page: Page | None,
    exploration_mode: str,
    current_testing_function_id: str,
    current_testing_function_status: str,
    reason: str,
    state: AgentState | None = None,
    rejected_scenario: str = "",
) -> dict:
    noop_count = _planner_noop_count(state) + 1
    should_force_now = (
        "\u7a0b\u5e8f\u6d3e\u751f\u8986\u76d6\u7387\u4e0d\u8db3\u4e14\u4ecd\u6709\u672a\u5904\u7406\u63a7\u4ef6" in reason
        or "\u5165\u53e3\u4e3b\u8def\u5f84\u5df2\u7ecf\u9a8c\u8bc1\uff0c\u5fc5\u987b\u6539\u9009\u76ee\u6807\u9875\u9762\u672a\u6267\u884c\u7684\u4e0d\u540c\u529f\u80fd\u63a7\u4ef6" in reason
    )
    if state is not None and (
        should_force_now or noop_count >= MAX_CONSECUTIVE_PLANNER_NOOPS
    ):
        forced = _force_progress_after_planner_noop(
            graph,
            current_page,
            exploration_mode,
            current_testing_function_id,
            current_testing_function_status,
            state,
            reason,
            noop_count,
        )
        if forced is not None:
            if should_force_now:
                print(
                    "[Scenario Planning] Coverage insufficient but unhandled controls remain; "
                    "directly generating representative supplementary scenario without producing pure-state planning round."
                )
            return forced
    print(
        f"[Scenario Planning] {reason}, skipping execution this round and proceeding to the next round of planning."
        f" Empty plan count={noop_count}/{MAX_CONSECUTIVE_PLANNER_NOOPS}"
    )
    return {
        "ffg": graph,
        "current_page": current_page,
        "current_navigation_scenario": "",
        "current_test_scenario": "",
        "current_testing_function_id": current_testing_function_id,
        "current_testing_function_status": current_testing_function_status,
        "current_plan": {
            "scenario_type": "status_update",
            "target_function_id": "",
            "src_function_id": "",
            "dst_function_id": "",
        },
        "workflow_complete": False,
        "planner_noop_count": noop_count,
        "planner_feedback": reason,
        "planner_rejected_scenario": _clean_natural_text(rejected_scenario),
    }


def _bootstrap_initial_ffg(
    llm,
    graph: FlowGraph,
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
            "text": 'Currently no page screenshot; please return {"functions": [], "pages": []}.',
        })

    result = _invoke_llm(llm, BOOTSTRAP_FFG_SYSTEM_PROMPT, message_content)
    bootstrap_patch = {
        "functions": result.get("functions", []) if isinstance(result, dict) else [],
        "pages": result.get("pages", []) if isinstance(result, dict) else [],
    }
    trace = [{"current_page": current_page, "raw_step_index": 1}]
    patch_result = apply_ffg_patch(graph, bootstrap_patch, trace, current_plan={})
    if not graph.functions:
        print("[Scenario Planning] Cold start patch produced no Function nodes; falling back to minimal home page patch.")
        fallback_patch = {
            "functions": [
                {
                    "function_ref": "new_func_1",
                    "source_step": 1,
                    "name": _page_label(current_page) or "Current Page Navigation",
                    "description": current_page.description or "Current Page and visible entries identified during cold start.",
                    "resource_variables": [],
                }
            ],
            "pages": [
                {
                    "page_ref": "new_page_1",
                    "function_refs": ["new_func_1"],
                    "source_step": 1,
                    "name": _page_label(current_page) or "Current Page",
                    "description": current_page.description or "Current Page corresponding to the cold start screenshot.",
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
        "[Scenario Planning] Cold start graph construction completed: "
        f"functions={len(graph.functions)}, page={_page_label(first_page)}, "
        f"operations={sum(len(page.operations) for page in graph.active_pages())}, "
        f"patch_functions={len(patch_result.get('functions_patch', []))}, "
        f"patch_pages={len(patch_result.get('pages_patch', []))}"
    )
    return patch_result


def _build_function_audit_summary(
    graph: FlowGraph,
    node: FunctionNode,
    exploration_mode: str,
    app_name: str = "",
    package_name: str = "",
) -> str:
    boundary_hint = rule_for(package_name).to_prompt_hint()
    lines: list[str] = [
        "[Coverage Gap Check]",
        f"Target app: {app_name or package_name or 'unknown'} | package={package_name or 'unknown'}",
        EXPLORATION_MODE_INSTRUCTIONS[_normalize_exploration_mode(exploration_mode)],
        boundary_hint,
        "",
        f"Function: {node.id} | {_function_label(node)}",
    ]
    if node.description:
        lines.append(f"Description: {_clean_natural_text(node.description)}")
    lines.append(f"Test status: {node.test_status or 'not_started'}")
    lines.append(f"Coverage: {node.coverage_percent}%")
    coverage_details = graph.function_coverage_details(node.id)
    lines.append(
        "Controls/operations: "
        f"{coverage_details['executed_operations']}/{coverage_details['total_controls']}"
    )
    if node.resource_variables:
        lines.append(f"Resource variables: {', '.join(_clean_natural_text(item) for item in node.resource_variables)}")
    lines.append("")
    lines.append("Recorded Pages, controls, and operations:")
    for page_id in node.coverage_page_ids:
        page = graph.get_page(page_id)
        if page is None:
            continue
        lines.append(f"- Page: {page.id} | {_page_label(page)}")
        if page.description:
            lines.append(f"  Description: {_clean_natural_text(page.description)}")
        if page.controls:
            lines.append("  Complete control set (across screenshot states):")
            for control in page.controls:
                lines.append(f"    * {_clean_natural_text(control)}")
        if page.operations:
            lines.append("  Executed/known operations:")
            for operation in page.operations:
                lines.append(f"    * {_clean_natural_text(operation)}")
        else:
            lines.append("  Executed/known operations: None")
    transitions = graph.get_function_page_transitions(node.id)
    if transitions:
        lines.append("")
        lines.append("Page transitions:")
        for transition in transitions:
            src_page = graph.get_page(transition.src_page_id)
            dst_page = graph.get_page(transition.dst_page_id)
            lines.append(
                f"- From '{_page_label(src_page)}' via '{_clean_natural_text(transition.operation)}' to '{_page_label(dst_page)}'"
            )
    if node.scenarios:
        lines.append("")
        lines.append("Tested scenarios:")
        for scenario in node.scenarios:
            lines.append(f"- {_clean_natural_text(scenario)}")
    else:
        lines.append("")
        lines.append("Tested scenarios: None")
    return "\n".join(lines)


def _audit_function_for_gap(
    llm,
    graph: FlowGraph,
    node: FunctionNode,
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
                    f"[Batched Visual Review] Currently batch {batch_index}/{len(page_batches)}."
                    "Only identify visible omissions based on the screenshots attached in this batch; return false if this batch has no omissions,"
                    "do not infer for pages without attached screenshots. The program will continue checking remaining batches."
                ),
            },
        ]
        if validation_feedback:
            message_content.append({"type": "text", "text": validation_feedback})
        for page in page_batch:
            message_content.append({
                "type": "text",
                "text": f"Screenshot corresponds to Page: {page.id} | {_page_label(page)}",
            })
            message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{page.representative_screenshot}"},
            })

        print(
            "[Scenario Planning] Coverage gap check: "
            f"{_function_label(node)} | batch={batch_index}/{len(page_batches)}, "
            f"pages={len(node.coverage_page_ids)}, screenshots={len(page_batch)}"
        )
        result = _normalize_audit_result(
            _invoke_llm(llm, FFG_AUDIT_SYSTEM_PROMPT, message_content)
        )
        last_result = result if isinstance(result, dict) else {}
        # Any gap must be executed before later batches. Invalid output is
        # returned to the existing protocol-retry path instead of being hidden.
        if _validate_audit_result(last_result) or last_result.get("has_exploration_gap") is True:
            return last_result
    return last_result


def _audit_scenario_is_executable(test_scenario: str) -> bool:
    text = str(test_scenario or "").strip()
    return len(text) >= 12 and SCENARIO_ACTION_PATTERN.search(text) is not None


def _audit_entry_is_visible(page: Page | None, entry_name: str) -> bool:
    if page is None or not entry_name:
        return False
    entry_key = _normalize_similarity_key(entry_name)
    page_key = _normalize_similarity_key(
        " ".join((page.name, page.description, *page.controls, *page.operations))
    )
    return bool(entry_key and page_key and (entry_key in page_key or page_key in entry_key))


def _validate_audit_result(result: object) -> list[str]:
    """Validate the audit protocol before interpreting an answer as no gap."""
    if not isinstance(result, dict) or not result:
        return ["empty_or_non_object_result"]
    errors: list[str] = []
    missing_fields = sorted(AUDIT_REQUIRED_FIELDS - set(result))
    if missing_fields:
        errors.append(f"missing_fields={missing_fields}")
    has_gap = result.get("has_exploration_gap")
    if not isinstance(has_gap, bool):
        errors.append("has_exploration_gap_must_be_boolean")
        return errors

    entry_page_ref = _normalize_optional_id(result.get("entry_page_ref"))
    entry_name = _clean_natural_text(result.get("entry_name"))
    test_scenario = _sanitize_planned_scenario(
        _normalize_optional_id(result.get("test_scenario"))
    )
    if has_gap:
        if not entry_page_ref:
            errors.append("gap_requires_entry_page_ref")
        if not entry_name:
            errors.append("gap_requires_entry_name")
        if not test_scenario:
            errors.append("gap_requires_test_scenario")
    elif entry_page_ref or entry_name or test_scenario:
        errors.append("no_gap_requires_empty_detail_fields")
    return errors


def _normalize_audit_result(result: object) -> dict:
    """Tolerate a missing entry_name when the executable scenario names it."""
    if not isinstance(result, dict):
        return {}
    normalized = dict(result)
    if normalized.get("has_exploration_gap") is not True:
        return normalized
    if _clean_natural_text(normalized.get("entry_name")):
        return normalized
    test_scenario = _sanitize_planned_scenario(
        _normalize_optional_id(normalized.get("test_scenario"))
    )
    derived_entry = _quoted_action_control(test_scenario) or _unquoted_audit_control(
        test_scenario
    )
    if derived_entry:
        normalized["entry_name"] = derived_entry
        print(f"[Scenario Planning] Coverage audit missing entry_name; extracted from execution scenario: {derived_entry}")
    return normalized


def _function_audit_signature(graph: FlowGraph, node: FunctionNode) -> str:
    """Structural coverage signature: Only re-triggers audit when pages/controls/operations/transitions change.

    Deliberately excludes scenario count: audit exploration itself invokes add_scenario; if scenario count were included in the signature,
    every exploration would change the signature, causing the next round to perform a full re-audit on the same completed function,
    even if the exploration produced no new coverage facts.
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
    operation_count = 0
    for page_id in coverage_page_ids:
        page = graph.get_page(page_id)
        if page is not None:
            control_count += len(page.controls)
            operation_count += len(page.operations)
    return (
        f"status={node.test_status or 'not_started'};"
        f"pages={page_ids};transitions={transition_count};"
        f"controls={control_count};operations={operation_count}"
    )


def _run_completion_audit(
    llm,
    graph: FlowGraph,
    exploration_mode: str,
    previous_signatures: dict[str, str] | None = None,
    audit_attempts: dict[str, int] | None = None,
    app_name: str = "",
    package_name: str = "",
) -> tuple[dict | None, bool, dict[str, str]]:
    previous_signatures = previous_signatures or {}
    audit_attempts = audit_attempts or {}
    audited_signatures: dict[str, str] = {}
    for node in graph.functions.values():
        if not _node_test_finished(node):
            continue
        signature = _function_audit_signature(graph, node)
        if previous_signatures.get(node.id) == signature:
            audited_signatures[node.id] = signature
            print(f"[Scenario Planning] Coverage gap check skipped unchanged Function: {_function_label(node)}")
            continue
        if int(audit_attempts.get(node.id, 0) or 0) >= MAX_COMPLETION_AUDITS_PER_FUNCTION:
            audited_signatures[node.id] = signature
            print(
                "[Scenario Planning] Single Function coverage audit reached safety limit; ending audit for this signature: "
                f"{_function_label(node)} | {MAX_COMPLETION_AUDITS_PER_FUNCTION}"
            )
            continue
        result = _normalize_audit_result(_audit_function_for_gap(
            llm,
            graph,
            node,
            exploration_mode,
            app_name=app_name,
            package_name=package_name,
        ))
        protocol_errors = _validate_audit_result(result)
        result_dict = result if isinstance(result, dict) else {}
        has_gap = result_dict.get("has_exploration_gap") is True
        entry_page_ref = _normalize_optional_id(result_dict.get("entry_page_ref"))
        entry_name = _clean_natural_text(result_dict.get("entry_name"))
        test_scenario = _sanitize_planned_scenario(
            _normalize_optional_id(result_dict.get("test_scenario"))
        )
        valid_entry_page = (
            bool(entry_page_ref)
            and entry_page_ref in node.page_ids
            and graph.get_page(entry_page_ref) is not None
        )
        valid_scenario = _audit_scenario_is_executable(test_scenario)
        action_policy_violation = account_action_policy_violation(
            f"Click {entry_name}"
        ) if has_gap else ""
        if protocol_errors or (has_gap and (not valid_entry_page or not valid_scenario)):
            result = _normalize_audit_result(_audit_function_for_gap(
                llm,
                graph,
                node,
                exploration_mode,
                validation_feedback=(
                    "The previous output did not comply with the coverage audit protocol. has_exploration_gap must be a JSON boolean;"
                    "when false, all other fields must be empty strings; when true, entry_page_ref, entry_name, and test_scenario must all be non-empty."
                    "entry_page_ref must be an existing page_x in the summary; entry_name must be in that page's screenshot or description"
                    "a visible and unexecuted entry name; test_scenario must explicitly specify clicking entry_name on this page and observing the actual result."
                    f"Field issue: {protocol_errors}"
                ),
                app_name=app_name,
                package_name=package_name,
            ))
            protocol_errors = _validate_audit_result(result)
            result_dict = result if isinstance(result, dict) else {}
            has_gap = result_dict.get("has_exploration_gap") is True
            entry_page_ref = _normalize_optional_id(result_dict.get("entry_page_ref"))
            entry_name = _clean_natural_text(result_dict.get("entry_name"))
            test_scenario = _sanitize_planned_scenario(
                _normalize_optional_id(result_dict.get("test_scenario"))
            )
            valid_entry_page = (
                bool(entry_page_ref)
                and entry_page_ref in node.page_ids
                and graph.get_page(entry_page_ref) is not None
            )
            valid_scenario = _audit_scenario_is_executable(test_scenario)
            action_policy_violation = account_action_policy_violation(
                f"Click {entry_name}"
            ) if has_gap else ""
        if protocol_errors:
            print(
                "[Scenario Planning] Missing coverage check continuously returned invalid structures, not determining completion based on this in this round: "
                f"{_function_label(node)} | {protocol_errors}"
            )
            return None, False, audited_signatures
        if (
            has_gap
            and valid_entry_page
            and not valid_scenario
            and _audit_entry_is_visible(graph.get_page(entry_page_ref), entry_name)
        ):
            page = graph.get_page(entry_page_ref)
            test_scenario = (
                f"On page '{_page_label(page)}', click '{entry_name}', "
                "observe and record the actual appearing page, popup, or state change."
            )
            valid_scenario = True
        if has_gap and (not valid_entry_page or not valid_scenario):
            print(
                "[Scenario Planning] Missing coverage check insisted there are omissions, but did not provide a verifiable entry; "
                "not determining completion based on this in this round: "
                f"{_function_label(node)} | page={entry_page_ref or 'none'} | "
                f"scenario={test_scenario or 'none'}"
            )
            return None, False, audited_signatures
        if action_policy_violation:
            print(
                "[Scenario Planning] Missing coverage candidate involves logout or account deletion, ignored: "
                f"{_function_label(node)} | entry={entry_name or 'none'} | "
                f"reason={action_policy_violation}"
            )
        if has_gap and valid_entry_page and valid_scenario and not action_policy_violation:
            print(
                "[Scenario Planning] Missing coverage check found an entry to verify, switching to real execution instead of direct graph construction: "
                f"{node.id} | page={entry_page_ref} | scenario={test_scenario}"
            )
            return {
                "owner_function_id": node.id,
                "entry_page_id": entry_page_ref,
                "test_scenario": test_scenario,
            }, False, audited_signatures
        if has_gap:
            print(
                "[Scenario Planning] Missing coverage check candidate lacks valid existing page or execution scenario, ignored: "
                f"{_function_label(node)} | entry_page_ref={entry_page_ref or 'none'} | "
                f"entry_name={entry_name or 'none'} | scenario={test_scenario or 'none'}"
            )
        else:
            print(f"[Scenario Planning] Missing coverage check passed: {_function_label(node)}")
        audited_signatures[node.id] = signature
    return None, True, audited_signatures


def _page_path_to_entry(
    graph: FlowGraph,
    node: FunctionNode,
    start_page_id: str,
    entry_page_id: str,
) -> list:
    if not start_page_id or not entry_page_id:
        return []
    if start_page_id == entry_page_id:
        return []
    queue: list[tuple[str, list]] = [(start_page_id, [])]
    visited = {start_page_id}
    while queue:
        page_id, path = queue.pop(0)
        for transition in graph.get_transitions_from(page_id):
            if transition.dst_page_id not in node.page_ids:
                continue
            if transition.dst_page_id in visited:
                continue
            next_path = [*path, transition]
            if transition.dst_page_id == entry_page_id:
                return next_path
            visited.add(transition.dst_page_id)
            queue.append((transition.dst_page_id, next_path))
    return []


def _build_audit_navigation_scenario(
    graph: FlowGraph,
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

    start_page_id = current_page_id
    path = _page_path_to_entry(graph, node, start_page_id, entry_page_id)
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
                "Current page still belongs to the target Function but has no recorded forward navigation path: if it is clearly a "
                f"subpage, picker page, or popup opened from this round's entry page, first execute back once, take a screenshot to confirm whether it directly returns to '{_page_label(entry_page)}'; "
                "if it did not return to the target entry page or has left the target app, execute open_app once to open the target app main page, "
                "and navigate according to subsequent recorded paths"
            )
        else:
            instructions.append(
                "Currently not on the known navigation path of the target Function: first execute open_app once to directly open the target app main page; "
                "if still not on the main page, then close temporary menus or popups existing when navigation started, and use page back to return to the recorded "
                f"'{_page_label(start_page)}' page; if leaving the target app when returning, then open_app again"
            )
    for transition in path:
        src_page = graph.get_page(transition.src_page_id)
        dst_page = graph.get_page(transition.dst_page_id)
        operation = _clean_natural_text(transition.operation) or "Execute recorded page operation"
        instructions.append(
            f"Execute '{operation}' on '{_page_label(src_page)}', entering '{_page_label(dst_page)}'"
        )
    if not path and start_page_id != entry_page_id:
        instructions.append(f"Continue safe back until entering the recorded '{_page_label(entry_page)}' page")
    instructions.append(
        f"Take a screenshot to confirm arrival at '{_page_label(entry_page)}' before ending navigation, "
        "do not click this round's entry to verify during navigation; menus opened to reach the target page belong to navigation path, do not close them again"
    )
    return "; ".join(instructions) + "."


def _audit_exploration_response(
    graph: FlowGraph,
    current_page: Page | None,
    audit_result: dict,
    audit_updates: int,
    audit_signatures: dict[str, str],
    audit_attempts: dict[str, int] | None = None,
) -> dict:
    owner_function_id = str(audit_result.get("owner_function_id", "")).strip()
    entry_page_id = str(audit_result.get("entry_page_id", "")).strip()
    audit_scenario = str(audit_result.get("test_scenario", "")).strip()
    internal_refs = re.findall(r"(?<![A-Za-z0-9_])(?:page|func)_\d+(?![A-Za-z0-9_])", audit_scenario)
    for ref in dict.fromkeys(internal_refs):
        if ref.startswith("page_"):
            entity = graph.get_page(ref)
            label = _page_label(entity) if entity is not None else ""
        else:
            entity = graph.get_node(ref)
            label = _function_label(entity) if entity is not None else ""
        if label and label != ref:
            audit_scenario = re.sub(
                rf"(?<![A-Za-z0-9_]){re.escape(ref)}(?![A-Za-z0-9_])",
                f'"{label}"',
                audit_scenario,
            )
    navigation_scenario = _build_audit_navigation_scenario(
        graph,
        current_page,
        owner_function_id,
        entry_page_id,
    )
    next_attempts = dict(audit_attempts or {})
    next_attempts[owner_function_id] = int(next_attempts.get(owner_function_id, 0) or 0) + 1
    print(
        "[Scenario Planning] Missing coverage check switched to evidence-driven exploration execution: "
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
        "completion_audit_updates": audit_updates + 1,
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
        **_reset_planner_noop_fields(),
    }


def _select_intra_target(
    graph: FlowGraph,
    current_page: Page | None,
    current_testing_function_id: str,
    failure_counts: dict[str, int] | None = None,
) -> FunctionNode | None:
    failure_counts = failure_counts or {}
    current_page_id = _resolve_current_page_id(graph, current_page)
    current_owner_ids = graph.get_page_owners(current_page_id)

    # The same entry page may belong to both current parent Function and multiple new Functions just discovered from explicit business entries.
    # Give unstarted shared owners an entry verification opportunity first, avoiding the case where parent Function exhausts scenario budget
    # before starting to identify independent capabilities like gallery, files, location, etc.; otherwise maintain continuity of the current Function.
    sticky_ids: list[str] = []
    preferred_id = graph.resolve_function_id(current_testing_function_id)
    shared_unstarted_ids = sorted(
        (
            function_id
            for function_id in current_owner_ids
            if (candidate := graph.get_node(function_id)) is not None
            and not _node_is_terminal(candidate)
            and (candidate.test_status or "not_started") == "not_started"
            and _node_intra_attempt_count(candidate) == 0
        ),
        key=lambda function_id: _function_sort_index(graph.get_node(function_id)),
    )
    sticky_ids.extend(shared_unstarted_ids)
    if preferred_id and preferred_id in current_owner_ids and preferred_id not in sticky_ids:
        sticky_ids.append(preferred_id)
    for function_id in current_owner_ids:
        if function_id not in sticky_ids:
            sticky_ids.append(function_id)
    if preferred_id and preferred_id not in sticky_ids:
        sticky_ids.append(preferred_id)

    deferred_sticky: list[FunctionNode] = []
    for function_id in sticky_ids:
        candidate = graph.get_node(function_id)
        if candidate is None or _node_is_terminal(candidate):
            continue
        if not _navigation_to_function(graph, current_page, candidate)[2]:
            continue
        if preferred_id and candidate.id == preferred_id:
            return candidate
        if int(failure_counts.get(candidate.id, 0) or 0) > 0:
            deferred_sticky.append(candidate)
        else:
            return candidate

    candidates = _priority_unfinished_nodes(graph, failure_counts)
    for candidate in candidates:
        if candidate.id in sticky_ids:
            continue
        if _navigation_to_function(graph, current_page, candidate)[2]:
            return candidate
    if deferred_sticky:
        return deferred_sticky[0]
    return None


def _shortest_page_path(
    graph: FlowGraph,
    start_page_id: str,
    target_page_ids: list[str],
) -> list:
    start_page_id = graph.resolve_page_id(start_page_id)
    targets = {
        resolved
        for page_id in target_page_ids
        if (resolved := graph.resolve_page_id(page_id))
    }
    if not start_page_id or not targets or start_page_id in targets:
        return []
    queue: list[tuple[str, list]] = [(start_page_id, [])]
    visited = {start_page_id}
    while queue:
        page_id, path = queue.pop(0)
        for transition in graph.get_transitions_from(page_id):
            if transition.dst_page_id in visited:
                continue
            next_path = [*path, transition]
            if transition.dst_page_id in targets:
                return next_path
            visited.add(transition.dst_page_id)
            queue.append((transition.dst_page_id, next_path))
    return []


def _navigation_to_function(
    graph: FlowGraph,
    current_page: Page | None,
    target: FunctionNode,
) -> tuple[str, str, bool]:
    # entry_page_ids only provides entries to enter the Function and does not host the target Function itself. Treating the entry page
    # as the navigation destination will cause the executor to stop at the app home page and search for same-name controls in other modules.
    target_page_ids = list(target.coverage_page_ids)
    if not target_page_ids:
        return "", "Target Function has no recorded Function hosting page yet", False

    current_page_id = _resolve_current_page_id(graph, current_page)
    if current_page_id in target_page_ids:
        return "", "Current location already belongs to target Function page", True

    path = _shortest_page_path(graph, current_page_id, target_page_ids)
    restart_first = False
    if not path:
        first_node = next(iter(graph.functions.values()), None)
        root_page = _node_representative_page(graph, first_node)
        root_page_id = str(root_page.id or "") if root_page is not None else ""
        if root_page_id in target_page_ids:
            restart_first = current_page_id != root_page_id
        else:
            path = _shortest_page_path(graph, root_page_id, target_page_ids)
            restart_first = bool(path)

    if not path and not restart_first:
        return "", "Neither current location nor app entry has a recorded page path to target Function", False

    instructions: list[str] = []
    if restart_first:
        instructions.append("Execute open_app to reopen target app, and wait for UI stabilization")
        root_page = _node_representative_page(graph, next(iter(graph.functions.values()), None))
        if root_page is not None:
            instructions.append(
                f"If not on '{_page_label(root_page)}' after opening, "
                "execute safe back step by step with screenshot confirmation each time until returning to this recorded entry page"
            )
    for transition in path:
        src_page = graph.get_page(transition.src_page_id)
        dst_page = graph.get_page(transition.dst_page_id)
        instructions.append(
            f"Execute '{_clean_natural_text(transition.operation)}' on '{_page_label(src_page)}', "
            f"entering '{_page_label(dst_page)}'"
        )
    target_page = graph.get_page(path[-1].dst_page_id) if path else _node_representative_page(graph, target)
    instructions.append(f"Confirm arrival at '{_page_label(target_page)}' before ending navigation")
    return "; ".join(instructions) + ".", "Use shortest recorded page path in FFG", True


def _build_global_planning_summary(
    graph: FlowGraph,
    run_phase: str,
) -> str:
    lines = [
        "[Global FFG Summary]",
        f"Running Mode: {_normalize_run_phase(run_phase)}",
        "Function Status:",
    ]
    sorted_functions = sorted(graph.functions.values(), key=_function_sort_index)
    for node in sorted_functions[:PLAN_MAX_GLOBAL_SUMMARY_FUNCTIONS]:
        lines.append(
            f"- {node.id} | {_function_label(node)} | status={node.test_status}"
            f" | coverage={node.coverage_percent}% | pages={len(node.page_ids)}"
        )
    omitted = len(sorted_functions) - PLAN_MAX_GLOBAL_SUMMARY_FUNCTIONS
    if omitted > 0:
        lines.append(f"- ... [Omitted {omitted} Functions, see current round scope summary for details]")
    lines.append("Function Dependencies:")
    if graph.function_dependencies:
        for dependency in graph.function_dependencies:
            lines.append(
                f"- {dependency.src_function_id} -> {dependency.dst_function_id}"
                f" | resource={_clip_summary_text(dependency.resource_variable, 240)}"
            )
    else:
        lines.append("- None")
    pending = _pending_cross_pairs(graph) if _normalize_run_phase(run_phase) == "all" else []
    lines.append("Cross-Function Combinations to Test:")
    if pending:
        for src_id, dst_id, reason in pending:
            lines.append(f"- {src_id} -> {dst_id} | {_clip_summary_text(reason, 240)}")
    else:
        lines.append("- None")
    return "\n".join(lines)


def _build_scope_planning_summary(
    graph: FlowGraph,
    current_page: Page | None,
    scenario_type: str,
    target: FunctionNode | None = None,
    src: FunctionNode | None = None,
    dst: FunctionNode | None = None,
    dependency_reason: str = "",
) -> tuple[str, str, bool]:
    nodes = [node for node in (target, src, dst) if node is not None]
    navigation_target = src if scenario_type == "cross_test" else target
    navigation_scenario = ""
    navigation_reason = "No target Function"
    navigation_reachable = False
    if navigation_target is not None:
        navigation_scenario, navigation_reason, navigation_reachable = _navigation_to_function(
            graph,
            current_page,
            navigation_target,
        )

    lines = [
        "[Current Location]",
        _build_current_location_summary(graph, current_page),
        "",
        "[Known Navigation]",
        navigation_scenario or navigation_reason,
    ]
    if scenario_type == "cross_test":
        lines.extend([
            "",
            "[Current Round Cross-Function Dependencies]",
            f"{src.id if src else ''} -> {dst.id if dst else ''} | {_clean_natural_text(dependency_reason)}",
        ])
        cross_path: list = []
        if src is not None and dst is not None:
            for start_page_id in src.page_ids:
                candidate = _shortest_page_path(
                    graph,
                    start_page_id,
                    list(dst.page_ids),
                )
                if candidate and (not cross_path or len(candidate) < len(cross_path)):
                    cross_path = candidate
        lines.append("[Known Paths from Source Function to Target Function]")
        shared_pages = set(src.page_ids if src else []) & set(dst.page_ids if dst else [])
        if shared_pages:
            lines.append(f"- The two Functions share pages: {sorted(shared_pages)}")
        elif cross_path:
            for transition in cross_path:
                lines.append(
                    f"- {transition.src_page_id} ({_page_label(graph.get_page(transition.src_page_id))})"
                    f" --[{_clean_natural_text(transition.operation)}]--> "
                    f"{transition.dst_page_id} ({_page_label(graph.get_page(transition.dst_page_id))})"
                )
        else:
            lines.append("- No recorded paths; cross-Function navigation must not be guessed")
            navigation_reachable = False

    scope_page_ids = {page_id for node in nodes for page_id in node.page_ids}
    for node in nodes:
        title = "Current Function" if scenario_type == "intra_test" else (
            "Source Function" if src is not None and node.id == src.id else "Target Function"
        )
        lines.extend([
            "",
            f"[{title}]",
            f"- {node.id} | {_function_label(node)}",
            f"- Description: {_clip_summary_text(node.description)}",
            f"- Status: {node.test_status} | Coverage: {node.coverage_percent}%",
            f"- Associated Pages: {node.page_ids or 'None'}",
            f"- Navigation-Only Entry Pages: {node.entry_page_ids or 'None'}",
            f"- Resource Variables: {node.resource_variables or 'None'}",
            "- Tested Scenarios:",
        ])
        if node.scenarios:
            lines.extend(f"  - {_clip_summary_text(item, 360)}" for item in node.scenarios[-8:])
        else:
            lines.append("  - None")
        lines.append("- Pages:")
        for page_id in node.page_ids:
            page = graph.get_page(page_id)
            if page is None:
                continue
            page_role = "Navigation Entry Only" if page_id in node.entry_page_ids else "Function Page"
            lines.append(
                f"  - {page.id} | {_page_label(page)} | role={page_role} | "
                f"activity={page.activity or 'Unknown'}"
            )
            if page.description:
                lines.append(f"    Description: {_clip_summary_text(page.description)}")
            if page.controls:
                lines.append("    Controls: " + "; ".join(_clip_summary_text(value, 180) for value in page.controls[-30:]))
            if page.operations:
                lines.append("    Executed: " + "; ".join(_clip_summary_text(value, 180) for value in page.operations[-12:]))

    relevant_transitions = [
        transition
        for transition in graph.page_transitions
        if transition.src_page_id in scope_page_ids or transition.dst_page_id in scope_page_ids
    ]
    lines.extend(["", "[Related Page Paths]"])
    if relevant_transitions:
        for transition in relevant_transitions:
            lines.append(
                f"- {transition.src_page_id} ({_page_label(graph.get_page(transition.src_page_id))})"
                f" --[{_clean_natural_text(transition.operation)}]--> "
                f"{transition.dst_page_id} ({_page_label(graph.get_page(transition.dst_page_id))})"
            )
    else:
        lines.append("- None")
    return "\n".join(lines), navigation_scenario, navigation_reachable


def create_scenario_plan_node(llm, device):
    """Create unified scenario planning node."""

    def scenario_plan_node(state: AgentState) -> dict:
        graph = state.get("ffg")
        if not isinstance(graph, FlowGraph):
            graph = FlowGraph()
        cold_start = not bool(graph.functions)

        current_page = state.get("current_page")
        if not isinstance(current_page, Page):
            current_page = None

        try:
            current_page = _ensure_target_app(
                device,
                str(state.get("package_name", "")),
                current_page=current_page,
            )
            if (
                not isinstance(current_page, Page)
                and not str(state.get("package_name", "")).strip()
            ):
                current_page = _safe_capture_current_page(device)
            elif not cold_start and isinstance(current_page, Page):
                current_page = _refresh_current_page_observation(
                    graph,
                    device,
                    current_page,
                )
        except ScreenshotUnavailableError:
            previous_count = int(state.get("screenshot_unavailable_count", 0) or 0)
            unavailable_count = previous_count + 1
            workflow_complete = (
                STOP_AFTER_SCREENSHOT_UNAVAILABLE
                and unavailable_count >= MAX_CONSECUTIVE_SCREENSHOT_UNAVAILABLE
            )
            if workflow_complete:
                print(
                    "[Scenario Plan] Current app screenshot is continuously unavailable or black screen, stopping this app run: "
                    f"{unavailable_count}/{MAX_CONSECUTIVE_SCREENSHOT_UNAVAILABLE}"
                )
            else:
                print(
                    "[Scenario Plan] Current screenshot is unavailable or black screen, skipping this round and retrying after recovery: "
                    f"{unavailable_count}/{MAX_CONSECUTIVE_SCREENSHOT_UNAVAILABLE}"
                )
            return {
                "ffg": graph,
                "current_page": None,
                "current_navigation_scenario": "",
                "current_test_scenario": "",
                "current_testing_function_id": _normalize_optional_id(
                    state.get("current_testing_function_id", "")
                ),
                "current_testing_function_status": str(
                    state.get("current_testing_function_status", "not_started")
                ).strip() or "not_started",
                "current_plan": {
                    "scenario_type": "status_update",
                    "target_function_id": "",
                    "src_function_id": "",
                    "dst_function_id": "",
                },
                "screenshot_unavailable_count": unavailable_count,
                "workflow_complete": workflow_complete,
            }
        if not isinstance(current_page, Page):
            previous_count = int(state.get("target_app_unavailable_count", 0) or 0)
            unavailable_count = previous_count + 1
            workflow_complete = (
                STOP_AFTER_TARGET_APP_UNAVAILABLE
                and unavailable_count >= MAX_CONSECUTIVE_TARGET_APP_UNAVAILABLE
            )
            if workflow_complete:
                print(
                    "[Scenario Plan] Continuously unable to recover target app, ending run to avoid idling: "
                    f"{unavailable_count}/{MAX_CONSECUTIVE_TARGET_APP_UNAVAILABLE}"
                )
            else:
                print(
                    "[Scenario Plan] Target app page not confirmed, skipping this round and retrying after restart: "
                    f"{unavailable_count}/{MAX_CONSECUTIVE_TARGET_APP_UNAVAILABLE}"
                )
            return {
                "ffg": graph,
                "current_page": None,
                "current_navigation_scenario": "",
                "current_test_scenario": "",
                "current_testing_function_id": "",
                "current_testing_function_status": "not_started",
                "current_plan": {
                    "scenario_type": "status_update",
                    "target_function_id": "",
                    "src_function_id": "",
                    "dst_function_id": "",
                },
                "target_app_unavailable_count": unavailable_count,
                "workflow_complete": workflow_complete,
            }

        if not cold_start:
            current_page = _attach_current_page_identity(graph, current_page)

        exploration_mode = _normalize_exploration_mode(state.get("exploration_mode", "core"))
        run_phase = _normalize_run_phase(state.get("run_phase", "all"))

        if cold_start:
            print("[Scenario Plan] FFG is empty; entering cold-start screenshot graph construction.")
            _bootstrap_initial_ffg(
                llm,
                graph,
                current_page,
                app_name=str(state.get("app_name", "") or ""),
                package_name=str(state.get("package_name", "") or ""),
            )
            current_page = next(iter(graph.active_pages()), current_page)
            stats = _count_ffg_stats(graph)
            print("[Scenario Plan] Input context")
            print(
                f"  Exploration Scope: Core Functions | Current Page: {_page_label(current_page)} | Input Screenshot: yes"
            )
            print(
                "  FFG Statistics: "
                f"functions={stats['functions']}, pages={stats['pages']}, operations={stats['operations']}, "
                f"page_transitions={stats['page_transitions']}, "
                f"function_dependencies={stats['function_dependencies']}, intra_scenarios={stats['intra_scenarios']}, "
                f"cross_scenarios={stats['cross_scenarios']}"
            )
            print("[Scenario Plan] Cold start only builds graph, generating no execution scenarios this round; next round will continue planning based on FFG summary.")
            return {
                "ffg": graph,
                "current_page": current_page,
                "current_navigation_scenario": "",
                "current_test_scenario": "",
                "current_testing_function_id": "",
                "current_testing_function_status": "not_started",
                "current_plan": {
                    "scenario_type": "bootstrap",
                    "target_function_id": "",
                    "src_function_id": "",
                    "dst_function_id": "",
                },
                "workflow_complete": False,
            }

        reopened_functions = _normalize_and_reopen_undercovered_functions(graph)
        if reopened_functions:
            print(
                "[Scenario Plan] Completed node restored to pending test based on program-derived coverage: "
                + ", ".join(reopened_functions)
            )

        # Coverage is dynamic (testing shared pages in other functions keeps increasing it),
        # and completion determination cannot run only once when a function becomes the current round target.
        # Batch recheck all in_progress functions before each round begins, finalizing those that qualify immediately.
        graph.recompute_all_coverages()
        auto_completed_ids: list[str] = []
        for sweep_node in list(graph.functions.values()):
            if sweep_node.test_status != "in_progress":
                continue
            if _function_completion_met(graph, sweep_node):
                graph.set_node_test_status(sweep_node.id, "completed")
                auto_completed_ids.append(sweep_node.id)
        if auto_completed_ids:
            print(
                "[Scenario Plan] Batch completion determination: coverage reached threshold and own evidence is sufficient, marked completed: "
                + ", ".join(
                    f"{fid} | {_function_label(graph.get_node(fid))}"
                    for fid in auto_completed_ids
                )
            )

        current_testing_function_id = _normalize_optional_id(
            state.get("current_testing_function_id", "")
        )
        current_testing_function_status = str(
            state.get("current_testing_function_status", "not_started")
        ).strip() or "not_started"

        pending_cross_pairs = _pending_cross_pairs(graph) if run_phase == "all" else []
        scenario_type = ""
        target_node: FunctionNode | None = None
        src_node: FunctionNode | None = None
        dst_node: FunctionNode | None = None
        dependency_reason = ""

        # Test the function currently located in first; only schedule cross-function scenarios when no intra-function targets remain.
        target_node = _select_intra_target(
            graph,
            current_page,
            current_testing_function_id,
            _normalize_failure_counts(state.get("function_execution_failure_counts")),
        )
        if target_node is not None:
            scenario_type = "intra_test"
            intra_attempts = _node_intra_attempt_count(target_node)
            budget_reached = intra_attempts >= PLAN_MAX_SCENARIOS_PER_FUNCTION
            uncovered_controls = graph.function_coverage_details(target_node.id)[
                "uncovered_controls"
            ]
            own_evidence = (
                not uncovered_controls
                or intra_attempts >= PLAN_MIN_SCENARIOS_BEFORE_COMPLETE
            )
            coverage_reached = _function_completion_met(graph, target_node)
            if not coverage_reached and not own_evidence:
                print(
                    "[Scenario Plan] Coverage comes from shared pages rather than this function's own tests, not finalizing yet: "
                    f"{target_node.id} | {_function_label(target_node)} | "
                    f"coverage={target_node.coverage_percent}% | "
                    f"own_scenarios={intra_attempts}/{PLAN_MIN_SCENARIOS_BEFORE_COMPLETE} | "
                    f"uncovered_controls={len(uncovered_controls)}"
                )
            if coverage_reached:
                graph.set_node_test_status(target_node.id, "completed")
                print(
                    "[Scenario Plan] Program-derived control/operation coverage reached threshold, marked completed and waiting for omission audit: "
                    f"{target_node.id} | {_function_label(target_node)}"
                )
                return {
                    "ffg": graph,
                    "current_page": current_page,
                    "current_navigation_scenario": "",
                    "current_test_scenario": "",
                    "current_testing_function_id": "",
                    "current_testing_function_status": "not_started",
                    "current_plan": {
                        "scenario_type": "status_update",
                        "target_function_id": target_node.id,
                        "src_function_id": "",
                        "dst_function_id": "",
                    },
                    "workflow_complete": False,
                    **_reset_planner_noop_fields(),
                }
            if budget_reached:
                print(
                    "[Scenario Plan] Intra-function scenarios reached standard budget but program coverage is insufficient, "
                    "continuing audit based on uncovered controls, not completing directly: "
                    f"{target_node.id} | coverage={target_node.coverage_percent}%"
                )

        if not scenario_type and pending_cross_pairs:
            for src_id, dst_id, candidate_reason in pending_cross_pairs:
                candidate_src = graph.get_node(src_id)
                candidate_dst = graph.get_node(dst_id)
                if (
                    candidate_src is not None
                    and candidate_dst is not None
                    and _navigation_to_function(graph, current_page, candidate_src)[2]
                ):
                    src_node = candidate_src
                    dst_node = candidate_dst
                    dependency_reason = candidate_reason
                    scenario_type = "cross_test"
                    break

        if not scenario_type and (pending_cross_pairs or _unfinished_function_nodes(graph)):
            return _planner_status_update_response(
                graph,
                current_page,
                exploration_mode,
                current_testing_function_id,
                current_testing_function_status,
                "No recorded page path to remaining functions to be tested in FFG",
                state,
            )

        if not scenario_type:
            audit_updates = int(state.get("completion_audit_updates", 0) or 0)
            audit_attempts = state.get("completion_audit_attempts", {}) or {}
            previous_signatures = state.get("completion_audit_signatures", {}) or {}
            print("[Scenario Plan] No pending functions or cross-function dependencies remaining; starting coverage omission check.")
            audit_result, audit_complete, audited_signatures = _run_completion_audit(
                llm,
                graph,
                exploration_mode,
                previous_signatures if isinstance(previous_signatures, dict) else {},
                audit_attempts if isinstance(audit_attempts, dict) else {},
                app_name=str(state.get("app_name", "") or ""),
                package_name=str(state.get("package_name", "") or ""),
            )
            merged_signatures = dict(previous_signatures) if isinstance(previous_signatures, dict) else {}
            merged_signatures.update(audited_signatures)
            if audit_result is not None:
                return _audit_exploration_response(
                    graph,
                    current_page,
                    audit_result,
                    audit_updates,
                    merged_signatures,
                    audit_attempts if isinstance(audit_attempts, dict) else {},
                )
            if not audit_complete:
                response = _planner_status_update_response(
                    graph,
                    current_page,
                    exploration_mode,
                    current_testing_function_id,
                    current_testing_function_status,
                    "Coverage omission check did not return verifiable complete JSON; workflow completion cannot be determined based on this",
                    state,
                )
                response["completion_audit_signatures"] = merged_signatures
                return response
            print("[Scenario Planning] All Functions, cross-Function dependencies, and coverage omissions have been processed.")
            return {
                "ffg": graph,
                "current_page": current_page,
                "current_navigation_scenario": "",
                "current_test_scenario": "",
                "current_testing_function_id": "",
                "current_testing_function_status": "not_started",
                "completion_audit_signatures": merged_signatures,
                "current_plan": {
                    "scenario_type": "status_update",
                    "target_function_id": "",
                    "src_function_id": "",
                    "dst_function_id": "",
                },
                "workflow_complete": True,
                **_reset_planner_noop_fields(),
            }

        global_summary = _build_global_planning_summary(graph, run_phase)
        scope_summary, known_navigation, navigation_reachable = _build_scope_planning_summary(
            graph,
            current_page,
            scenario_type,
            target=target_node,
            src=src_node,
            dst=dst_node,
            dependency_reason=dependency_reason,
        )
        if not navigation_reachable:
            return _planner_status_update_response(
                graph,
                current_page,
                exploration_mode,
                current_testing_function_id,
                current_testing_function_status,
                "There is no recorded Page path in the FFG from the current location or app entry point to the target Function",
                state,
            )
        if scenario_type == "cross_test":
            scope_instruction = (
                "[Fixed scope for this round] Must return cross_test, "
                f"src_function={src_node.id}, dst_function={dst_node.id}. "
                "only generate a single closed loop of source Function producing/modifying and target Function reading/displaying/consuming around the given resource dependency."
            )
        else:
            scope_instruction = (
                "[Fixed scope for this round] Must return intra_test, "
                f"target_function={target_node.id}. "
                "if there are still distinct explorable business directions, return a non-empty test_scenario; "
                "if already fully covered or can only repeat tested scenarios, leave navigation_scenario and test_scenario empty."
            )
        if _is_black_screen_page(graph, current_page):
            scope_instruction += (
                " The current device stays on a recorded black screen abnormal Page; navigation_scenario must first perform a back recovery, "
                "and after recovery enter the fixed Function for this round along a known path. Do not re-execute any historical operation leading to a black screen Page."
            )

        package_name = str(state.get("package_name", "") or "")
        boundary_hint = rule_for(package_name).to_prompt_hint()
        message_content: list[dict] = [
            {
                "type": "text",
                "text": (
                    f"[Target App] name={state.get('app_name', '') or 'Unknown'}, "
                    f"package={package_name or 'Unknown'}"
                ),
            },
            {
                "type": "text",
                "text": (
                    "[Exploration Scope]"
                    + EXPLORATION_MODE_INSTRUCTIONS[_normalize_exploration_mode(exploration_mode)]
                    + (boundary_hint if boundary_hint else "")
                ),
            },
            {"type": "text", "text": global_summary},
            {"type": "text", "text": scope_summary},
            {"type": "text", "text": scope_instruction},
            {
                "type": "text",
                "text": (
                    "[Test Data for This Round] Use a unique suffix when creating or naming resources "
                    f"FuncDroid_{int(state.get('iteration_count', 0) or 0)}."
                ),
            },
        ]
        planner_feedback = _clean_natural_text(state.get("planner_feedback", ""))
        rejected_scenario = _clean_natural_text(state.get("planner_rejected_scenario", ""))
        if planner_feedback:
            message_content.append({
                "type": "text",
                "text": (
                    f"[Previous Round Validation Feedback] {planner_feedback}. "
                    + (f"Rejected scenario: {rejected_scenario}. " if rejected_scenario else "")
                    + (
                        "Other values of the same parameterized entry are also considered representative coverage; "
                        "you must switch to controls of a different Function, or leave both scenario fields empty."
                        if rejected_scenario else ""
                    )
                ),
            })
        if current_page and current_page.representative_screenshot:
            message_content.extend([
                {
                    "type": "text",
                    "text": "[Current Page Screenshot] Only used to check current location, do not use controls not recorded in FFG.",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{current_page.representative_screenshot}",
                    },
                },
            ])

        print(
            "[Scenario Planning] Fixed scope: "
            + (
                f"cross {src_node.id}->{dst_node.id} | {dependency_reason}"
                if scenario_type == "cross_test"
                else f"intra {target_node.id} | {_function_label(target_node)}"
            )
        )
        result, validation_errors = _invoke_valid_plan(
            llm,
            UNIFIED_SCENARIO_PLAN_SYSTEM_PROMPT,
            message_content,
            package_name=str(state.get("package_name", "") or ""),
        )
        if not result:
            return _planner_status_update_response(
                graph,
                current_page,
                exploration_mode,
                current_testing_function_id,
                current_testing_function_status,
                f"Planner continuously returned invalid fields: {validation_errors}",
                state,
            )

        planned_type = str(result.get("scenario_type", "")).strip()
        target_function_id = _resolve_function_ref(graph, result.get("target_function"))
        src_function_id = _resolve_function_ref(graph, result.get("src_function"))
        dst_function_id = _resolve_function_ref(graph, result.get("dst_function"))
        if scenario_type == "intra_test" and (
            planned_type != "intra_test" or target_function_id != target_node.id
        ):
            return _planner_status_update_response(
                graph,
                current_page,
                exploration_mode,
                current_testing_function_id,
                current_testing_function_status,
                f"Planner deviated from fixed Function {target_node.id}",
                state,
            )
        if scenario_type == "cross_test" and (
            planned_type != "cross_test"
            or src_function_id != src_node.id
            or dst_function_id != dst_node.id
        ):
            return _planner_status_update_response(
                graph,
                current_page,
                exploration_mode,
                current_testing_function_id,
                current_testing_function_status,
                f"Planner deviated from fixed cross-Function dependency {src_node.id}->{dst_node.id}",
                state,
            )

        test_scenario = _sanitize_planned_scenario(
            _normalize_optional_id(result.get("test_scenario"))
        )
        planned_navigation = _sanitize_navigation_scenario(
            _normalize_optional_id(result.get("navigation_scenario")),
            graph,
            current_page,
        )
        navigation_scenario = known_navigation or planned_navigation
        recovery_navigation = bool(re.search(
            r"\u8fd4\u56de|press_back|\u5173\u95ed|\u53d6\u6d88|open_app|\u91cd\u65b0\u6253\u5f00|\u91cd\u542f",
            navigation_scenario,
            re.IGNORECASE,
        ))

        if _is_black_screen_page(graph, current_page) and not recovery_navigation:
            return _planner_status_update_response(
                graph,
                current_page,
                exploration_mode,
                current_testing_function_id,
                current_testing_function_status,
                "Currently on a black screen abnormal Page, but the planning result does not have recovery navigation to go back or reopen the App first",
                state,
                rejected_scenario=test_scenario,
            )
        if _matches_black_screen_path(
            graph,
            f"{navigation_scenario}; {test_scenario}",
        ):
            return _planner_status_update_response(
                graph,
                current_page,
                exploration_mode,
                current_testing_function_id,
                current_testing_function_status,
                "Planning result repeated an operation path known to cause a black screen or make the Page unavailable",
                state,
                rejected_scenario=test_scenario,
            )

        if scenario_type == "intra_test" and not test_scenario:
            coverage_details = graph.function_coverage_details(target_node.id)
            if (
                target_node.coverage_percent < PLAN_FUNCTION_COVERAGE_TARGET
                and coverage_details["uncovered_controls"]
            ):
                graph.set_node_test_status(target_node.id, "in_progress")
                return _planner_status_update_response(
                    graph,
                    current_page,
                    exploration_mode,
                    target_node.id,
                    "in_progress",
                    "Program-derived coverage is insufficient and unprocessed controls remain; cannot finish directly",
                    state,
                )
            graph.set_node_test_status(target_node.id, "completed")
            print(f"[Scenario Planning] Function planning completed: {_function_label(target_node)}")
            return {
                "ffg": graph,
                "current_page": current_page,
                "current_navigation_scenario": "",
                "current_test_scenario": "",
                "current_testing_function_id": "",
                "current_testing_function_status": "not_started",
                "current_plan": {
                    "scenario_type": "status_update",
                    "target_function_id": target_node.id,
                    "src_function_id": "",
                    "dst_function_id": "",
                },
                "workflow_complete": False,
                **_reset_planner_noop_fields(),
            }

        if scenario_type == "intra_test":
            graph.set_node_test_status(target_node.id, "in_progress")
            testing_function_id = target_node.id
            testing_function_status = "in_progress"
        else:
            testing_function_id = current_testing_function_id
            testing_function_status = current_testing_function_status

        return {
            "ffg": graph,
            "current_page": current_page,
            "current_navigation_scenario": navigation_scenario,
            "current_test_scenario": test_scenario,
            "current_testing_function_id": testing_function_id,
            "current_testing_function_status": testing_function_status,
            "current_plan": {
                "scenario_type": scenario_type,
                "target_function_id": target_node.id if target_node is not None else "",
                "src_function_id": src_node.id if src_node is not None else "",
                "dst_function_id": dst_node.id if dst_node is not None else "",
            },
            "workflow_complete": False,
            **_reset_planner_noop_fields(),
        }

    return scenario_plan_node
