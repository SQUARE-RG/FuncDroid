"""Scenario execution node."""

import json
import math
import os
import re
import subprocess
import time
from difflib import SequenceMatcher
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from model.flow_graph import Page
from tools.actions import (
    ScreenshotUnavailableError,
    _perform_action,
    capture_current_page,
    capture_runtime_page,
)
from workflow.exploration_boundary import BoundaryHit, check_boundary, rule_for
from workflow.llm import _repair_common_json_typos, compute_retry_delay_seconds
from workflow.llm_usage import record_llm_failure, record_llm_request, record_llm_response
from workflow.nodes.prompt import SCENARIO_EXEC_SYSTEM_PROMPT_TEMPLATE
from workflow.nodes._shared import (
    normalize_llm_text,
    representative_control_family_scenario,
    screenshot_similarity,
)
from workflow.state import AgentState
from workflow.trace_artifacts import save_trace_screenshot
from workflow.account_safety import (
    account_action_policy_violation,
    account_destructive_intent,
)

JSON_BLOCK_PATTERN = re.compile(r"^```(?:json)?\s*|\s*```$")


def _compact_execution_log_text(value: object, limit: int = 200) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[:limit - 3] + "..."


KEEP_IMAGE_STEPS = int(os.getenv("EXEC_KEEP_IMAGE_STEPS", "3"))
# Upper limit of steps per round in scenario execution (test phase); navigation phase uses an independent MAX_NAVIGATION_STEPS.
MAX_EXEC_STEPS = max(1, int(os.getenv("MAX_EXEC_STEPS", "15")))
MAX_NAVIGATION_STEPS = max(1, int(os.getenv("MAX_NAVIGATION_STEPS", "10")))
MAX_CONSECUTIVE_ERRORS = 3
# Total allowed model errors (including timeouts) in a single-round scenario. Only limits cumulative count, does not change consecutive failure judgment.
MAX_TOTAL_LLM_ERRORS = max(
    MAX_CONSECUTIVE_ERRORS,
    int(os.getenv("EXEC_MAX_TOTAL_LLM_ERRORS", "4")),
)
MAX_ACTION_FAILURE_RECOVERIES = max(
    0,
    int(os.getenv("EXEC_ACTION_FAILURE_RECOVERIES", "1")),
)
MAX_CONSECUTIVE_SCREENSHOT_UNAVAILABLE = max(1, int(os.getenv("MAX_CONSECUTIVE_SCREENSHOT_UNAVAILABLE", "2")))
STOP_AFTER_SCREENSHOT_UNAVAILABLE = os.getenv(
    "STOP_AFTER_SCREENSHOT_UNAVAILABLE", "0"
).strip().lower() in {"1", "true", "yes", "on"}
ACTION_DELAY = float(os.getenv("ACTION_DELAY", "0.7"))
FAST_ACTION_DELAY = float(os.getenv("FAST_ACTION_DELAY", "0.15"))
FAST_ACTIONS = {"press_back"}
COORDINATE_FIELDS = ("x", "y")
APP_RESTART_DELAY = float(os.getenv("APP_RESTART_DELAY", "2.0"))
STALL_REPEAT_WINDOW = 3
WAIT_STALL_REPEAT_WINDOW = max(
    2,
    min(int(os.getenv("EXEC_WAIT_STALL_REPEAT_WINDOW", "2")), MAX_EXEC_STEPS),
)
CLICK_STALL_REPEAT_WINDOW = 3
SEMANTIC_CLICK_STALL_REPEAT_WINDOW = 2
SCROLL_STALL_REPEAT_WINDOW = 5
NAVIGATION_SCROLL_INTENT_REPEAT_WINDOW = max(
    2,
    min(int(os.getenv("EXEC_NAV_SCROLL_INTENT_REPEAT_WINDOW", "2")), MAX_NAVIGATION_STEPS),
)
TEST_SCROLL_INTENT_REPEAT_WINDOW = max(
    2,
    min(int(os.getenv("EXEC_TEST_SCROLL_INTENT_REPEAT_WINDOW", "5")), MAX_EXEC_STEPS),
)
# Hard budget for consecutive scrolls in the same direction during test phase: after reaching this count, exploration must switch to other directions,
# continuing to scroll in the same direction will be intercepted in advance by _detect_stall.
TEST_MAX_CONSECUTIVE_SAME_DIRECTION_SCROLLS = max(
    1,
    min(
        int(os.getenv("EXEC_TEST_MAX_CONSECUTIVE_SCROLLS", "3")),
        MAX_EXEC_STEPS,
    ),
)
PAGE_NAVIGATION_CYCLE_WINDOW = 6
REPEAT_COORD_TOLERANCE = int(os.getenv("EXEC_REPEAT_COORD_TOLERANCE", "80"))
STALL_SCREENSHOT_SIMILARITY_THRESHOLD = float(os.getenv("EXEC_STALL_SCREENSHOT_SIMILARITY", "0.9995"))
SCROLL_INTENT_TEXT_SIMILARITY_THRESHOLD = float(os.getenv("EXEC_SCROLL_INTENT_TEXT_SIMILARITY", "0.72"))
PAGE_CYCLE_ACTIONS = {"click", "press_back", "wait"}
# Direction-independent, action-independent fallback stall criterion: repeatedly probing within the same Activity while constantly
# circulating between a few visual states finishes early, avoiding maxing out MAX_EXEC_STEPS.
NO_PROGRESS_STALL_WINDOW = max(
    3,
    min(int(os.getenv("EXEC_NO_PROGRESS_STALL_WINDOW", "6")), MAX_EXEC_STEPS),
)
NO_PROGRESS_MAX_VISUAL_STATES = max(
    1,
    int(os.getenv("EXEC_NO_PROGRESS_MAX_VISUAL_STATES", "2")),
)
# Use a looser threshold than duplicate action judgment when clustering the same visual state: what needs to be answered here is
# "whether the interface has changed substantially", rather than "whether two screenshots are pixel-by-pixel identical".
NO_PROGRESS_SIMILARITY_THRESHOLD = float(
    os.getenv("EXEC_NO_PROGRESS_SIMILARITY", "0.985")
)
NO_PROGRESS_STALL_ACTIONS = {"click", "long_press", "scroll", "press_back", "wait"}
APP_RECOVERY_BACK_ATTEMPTS = max(1, int(os.getenv("APP_RECOVERY_BACK_ATTEMPTS", "3")))
APP_RESTART_ATTEMPTS = max(1, int(os.getenv("APP_RESTART_ATTEMPTS", "2")))
APP_RESTART_RETRY_DELAY = float(os.getenv("APP_RESTART_RETRY_DELAY", "1.0"))
# Cold start wait: replaced fixed sleep with polling the foreground package name. Cold starts of heavy apps often exceed
# APP_RESTART_DELAY (2s); fixed waiting captures desktop/splash screenshots, leading execution models to
# misjudge and repeat open_app (historically causing force-stop restart storms).
APP_START_POLL_INTERVAL = max(0.1, float(os.getenv("APP_START_POLL_INTERVAL", "0.5")))
APP_START_MAX_WAIT = max(1.0, float(os.getenv("APP_START_MAX_WAIT", "12")))
APP_RESTART_SETTLE_DELAY = max(0.0, float(os.getenv("APP_RESTART_SETTLE_DELAY", "1.0")))
# open_app repeat window: every open_app may trigger a real restart, costing more than normal actions,
# intercepted early when occurring 2 consecutive times.
OPEN_APP_STALL_REPEAT_WINDOW = 2
# Pressing back key is ineffective when foreground is launcher, recovery flow jumps directly to restart.
LAUNCHER_PACKAGE_PATTERN = re.compile(r"launcher|home", re.IGNORECASE)
TARGET_APP_RESTARTED_END = "The target app left the foreground and could not be recovered by pressing back; it was restarted and the current round of scenario ended"
TARGET_APP_CRASHED_END = "Detected target app crash; restarted the target app and ended the current scenario"
TARGET_APP_CRASH_RECOVERY_FAILED_END = "Detected target app crash, but still failed to recover to target app after restart; ended the current scenario"
TARGET_APP_RECOVERY_FAILED_END = "The target app left the foreground; neither automatic back nor restart succeeded, ended the current round of scenario"
EXTERNAL_TRANSITION_BLOCKED_END = "The target app transitioned to an uncontrolled external app; restored target app and ended the current scenario"
# Exploration boundary (core mode) end message template: {label} comes from boundary rule config, e.g. "Mini-program / third-party service".
EXPLORATION_BOUNDARY_BLOCKED_END = "Detected entry into {label} (non-core function of target app); automatically backed out and ended the current scenario"
ACCOUNT_SAFETY_BLOCKED_END = "Account safety policy intercepted a logout or account-deletion operation"
VERIFICATION_BLOCKED_END = "Captcha or human verification cannot be completed by unattended testing; exited current blocking page"
FOREGROUND_CHECK_RETRIES = max(1, int(os.getenv("FOREGROUND_CHECK_RETRIES", "2")))
FOREGROUND_CHECK_RETRY_DELAY = float(os.getenv("FOREGROUND_CHECK_RETRY_DELAY", "0.2"))
# Foreground status recheck: during page transition animation, current_package may momentarily report desktop
# launcher (app's own transient state); a single immediate check would misjudge transient state as external transition
# (external_transition_blocked). Non-foreground conclusions must be confirmed by rechecking after a brief wait.
EXTERNAL_RECHECK_ATTEMPTS = max(1, int(os.getenv("EXTERNAL_RECHECK_ATTEMPTS", "3")))
EXTERNAL_RECHECK_DELAY = max(0.1, float(os.getenv("EXTERNAL_RECHECK_DELAY", "0.5")))
CRASH_LOGCAT_TAIL_LINES = max(50, min(int(os.getenv("CRASH_LOGCAT_TAIL_LINES", "400")), 2000))
CRASH_LOGCAT_KEEP_LINES = max(20, min(int(os.getenv("CRASH_LOGCAT_KEEP_LINES", "120")), 300))
CRASH_LOGCAT_PATTERNS = (
    "FATAL EXCEPTION",
    "AndroidRuntime",
    "ANR in",
    "Fatal signal",
    "Force finishing",
    "has died",
    "Process crashed",
    "EGL_BAD_ALLOC",
    "EGL_NOT_INITIALIZED",
    "Failed to create EGLSurface",
    "Failed to choose config",
)
NON_CRASH_LOGCAT_PATTERNS = (
    "BufferLayer",
    "Timestamp",
    "seems implausible",
    "expectedPresent",
)
FINISHED_FAILURE_PATTERNS = (
    "\u76ee\u6807\u4e0d\u53ef\u8fbe",
    "\u5f53\u524d\u4e0d\u53ef\u8fbe",
    "\u4efb\u52a1\u65e0\u6cd5",
    "\u65e0\u6cd5\u5b8c\u6210",
    "\u65e0\u6cd5\u7ee7\u7eed",
    "\u65e0\u6cd5\u5207\u6362",
    "\u65e0\u6cd5\u627e\u5230",
    "\u65e0\u6cd5\u8fdb\u5165",
    "\u672a\u627e\u5230",
    "\u672a\u6210\u529f",
    "\u4e0d\u80fd\u7ee7\u7eed",
    "\u5747\u672a\u6210\u529f",
    "\u5747\u65e0\u6cd5",
    "not able",
    "unable",
    "cannot",
    "can't",
    "unreachable",
    "not found",
    "failed to",
    "could not complete",
    "unable to continue",
    "goal unreachable",
    "target unreachable",
)
FINISHED_OVERRIDE_FAILURE_PATTERNS = (
    r"(\u66ff\u4ee3|\u66f4\u6362).{0,12}(\u5165\u53e3|\u8def\u5f84|\u63a7\u4ef6|\u6309\u94ae)",
    r"(\u539f|\u6307\u5b9a).{0,20}(\u672a\u627e\u5230|\u672a\u6210\u529f|\u672a\u5b8c\u6210|\u65e0\u6cd5|\u4e0d\u80fd|\u4e0d\u7b26\u5408)",
    r"(instead|alternate|alternative).{0,20}(original|specified|requested)",
)
FINISHED_SUCCESS_PATTERNS = (
    r"\u76ee\u6807\u5df2\u8fbe\u6210",
    r"\u4efb\u52a1\u76ee\u6807\u5df2\u8fbe\u6210",
    r"(\u6240\u6709|\u5168\u90e8).{0,8}(\u4efb\u52a1)?\u76ee\u6807.{0,4}(\u6ee1\u8db3|\u8fbe\u6210|\u5b8c\u6210)",
    r"\u5f53\u524d\u4efb\u52a1\u5df2\u5b8c\u6210",
    r"(\u6240\u6709|\u5168\u90e8).{0,16}(\u9a8c\u8bc1\u9879|\u68c0\u67e5\u9879).{0,8}(\u5747\u5df2\u786e\u8ba4|\u5df2\u786e\u8ba4|\u5b8c\u6210)",
    r"\u5df2\u5b8c\u6210.{0,12}(\u6240\u6709|\u5168\u90e8)",
    r"\u6d4b\u8bd5\u4efb\u52a1\u5df2\u5b8c\u6210",
    r"\u9a8c\u8bc1\u4e86.{0,40}\u6b63\u5e38",
    r"(\u6240\u6709|\u5168\u90e8).{0,12}\u9a8c\u8bc1(\u70b9|\u8981\u6c42|\u76ee\u6807).{0,12}(\u6ee1\u8db3|\u8fbe\u6210|\u5b8c\u6210)",
    r"\u529f\u80fd\u6b63\u5e38",
    r"(\u4e3b\u8981|\u6838\u5fc3).{0,16}(\u9a8c\u8bc1\u70b9|\u76ee\u6807).{0,20}(\u8986\u76d6|\u8fbe\u6210|\u6ee1\u8db3|\u5b8c\u6210|\u5df2\u9a8c\u8bc1)",
    r"\bgoal\b.{0,30}\b(achieved|complete|completed)\b",
    r"\btask\b.{0,30}\b(complete|completed)\b",
    r"\bgoal\s+achieved\b",
    r"\btask\s+completed\b",
)
MIXED_FINISH_FAILURE_PATTERN = re.compile(
    r"(?:\u76ee\u6807\u5df2\u8fbe\u6210|\u4efb\u52a1\u5df2\u5b8c\u6210|\u5df2\u5b8c\u6210\u53ef\u6267\u884c\u90e8\u5206).{0,120}"
    r"(?:\u4e0d\u53ef\u8fbe|\u672a\u5b8c\u6210|\u672a\u9a8c\u8bc1|\u672a\u627e\u5230|\u672a\u6267\u884c|\u65e0\u6cd5\u8fdb\u5165)"
    r"|(?:\u4e0d\u53ef\u8fbe|\u672a\u5b8c\u6210|\u672a\u9a8c\u8bc1|\u672a\u627e\u5230|\u672a\u6267\u884c|\u65e0\u6cd5\u8fdb\u5165).{0,120}"
    r"(?:\u76ee\u6807\u5df2\u8fbe\u6210|\u4efb\u52a1\u5df2\u5b8c\u6210|\u5df2\u5b8c\u6210\u53ef\u6267\u884c\u90e8\u5206)"
    r"|(?:goal\s+achieved|task\s+completed).{0,120}(?:unreachable|not\s+completed|not\s+verified|not\s+found|unable|failed)"
    r"|(?:unreachable|not\s+completed|not\s+verified|not\s+found|unable|failed).{0,120}(?:goal\s+achieved|task\s+completed)",
    re.IGNORECASE,
)
VERIFICATION_CHALLENGE_PATTERN = re.compile(
    r"\u9a8c\u8bc1\u7801|\u56fe\u5f62\u9a8c\u8bc1|\u4eba\u673a\u9a8c\u8bc1|\u5b89\u5168\u9a8c\u8bc1|\u6ed1\u5757|\u62fc\u56fe|captcha|verification\s+challenge",
    re.IGNORECASE,
)
VERIFICATION_ATTEMPT_PATTERN = re.compile(
    r"\u8bc6\u522b|\u62d6\u52a8|\u6ed1\u52a8|\u5bf9\u9f50|\u62fc\u5408|\u63d0\u4ea4|\u5b8c\u6210|\u901a\u8fc7|solve|drag|swipe|submit|verify",
    re.IGNORECASE,
)
VERIFICATION_EXIT_PATTERN = re.compile(
    r"\u5173\u95ed|\u53d6\u6d88|\u8fd4\u56de|\u9000\u51fa|close|cancel|back|exit",
    re.IGNORECASE,
)
SCROLL_INTENT_STOPWORDS = {
    "android",
    "app",
    "button",
    "buttons",
    "changelog",
    "current",
    "page",
    "screen",
    "scroll",
    "target",
}
SCROLL_INTENT_SEARCH_MARKERS = (
    "\u5bfb\u627e",
    "\u627e\u5230",
    "\u672a\u663e\u793a",
    "\u672a\u770b\u5230",
    "\u6ca1\u770b\u5230",
    "\u76ee\u6807",
    "\u9732\u51fa",
    "\u6309\u94ae",
    "\u63a7\u4ef6",
    "\u5165\u53e3",
    "visible",
    "find",
    "target",
    "not found",
    "search",
    "locate",
    "not visible",
)
EXTERNAL_APP_SCENARIO_PATTERNS = (
    r"(\u70b9\u51fb|\u9009\u62e9).{0,20}(\u5e94\u7528\u9879|\u5e94\u7528\u5217\u8868\u9879|app item)",
    # "Target app" is the app under test itself, often appearing in navigation recovery text, and cannot be used to permit external apps.
    r"(\u542f\u52a8|\u6253\u5f00|\u8df3\u8f6c\u5230|\u8fdb\u5165).{0,30}(\u5916\u90e8\u5e94\u7528|\u5176\u4ed6\u5e94\u7528|\u53e6\u4e00\u4e2a\u5e94\u7528|\u5bf9\u5e94\u5e94\u7528|\u5e94\u7528\u7a0b\u5e8f)",
    r"(\u542f\u52a8|\u6253\u5f00).{0,30}(Calculator|Calendar|Camera|Clock|Contacts|Files|Gmail|Maps|Messages|Phone|Photos)",
    r"(\u8fd4\u56de|\u56de\u5230).{0,30}(launcher|\u542f\u52a8\u5668|\u5e94\u7528\u5217\u8868|TextLauncher)",
    r"(\u7cfb\u7edf)?(\u6587\u4ef6|\u6587\u6863).{0,20}(\u9009\u62e9\u5668|\u7ba1\u7406\u5668|\u4fdd\u5b58|\u6253\u5f00|\u5bfc\u5165|\u521b\u5efa)",
    r"(\u6253\u5f00|\u8fdb\u5165|\u524d\u5f80).{0,24}(\u7cfb\u7edf\u8bbe\u7f6e|\u7cfb\u7edf\u6743\u9650\u9875|\u6743\u9650\u7ba1\u7406\u9875|\u76f8\u673a|\u76f8\u518c)",
    r"(?:\u5728|\u4f7f\u7528|\u901a\u8fc7)?\s*(?:\u7cfb\u7edf|\u5916\u90e8|\u9ed8\u8ba4)?\s*(?:\u6d4f\u89c8\u5668|browser).{0,16}(?:\u4e2d)?\s*(?:\u6253\u5f00|\u67e5\u770b)",
    r"(?:\u6253\u5f00|\u8fdb\u5165|\u8df3\u8f6c\u5230).{0,24}(?:\u6d4f\u89c8\u5668|browser)",
    r"(\u7cfb\u7edf\u6743\u9650|\u5b9a\u4f4d\u6743\u9650|\u76f8\u673a\u6743\u9650|\u76f8\u518c\u6743\u9650).{0,30}(\u8bbe\u7f6e|\u6388\u6743|\u5141\u8bb8|\u7ba1\u7406|\u9009\u62e9)",
    # Standalone "path" will also hit generic navigation prompts like "the opened menu belongs to the navigation path".
    # Files and directories already cover real system picker scenarios, no need for this overly broad keyword.
    r"(\u9009\u62e9|\u5bfc\u5165|\u4fdd\u5b58|\u521b\u5efa|\u6253\u5f00).{0,30}(\u6587\u4ef6|\u6587\u6863|\u76ee\u5f55|TiddlyWiki)",
    r"(DocumentsUI|document picker|file picker|file chooser|system file)",
    r"\b(open|launch)\b.{0,30}\b(app|application)\b",
)


def _is_non_crash_logcat_noise(line: str) -> bool:
    return all(pattern in line for pattern in NON_CRASH_LOGCAT_PATTERNS)
SAFE_EXTERNAL_PACKAGES = {
    package.strip()
    for package in os.getenv(
        "EXEC_SAFE_EXTERNAL_PACKAGES",
        ",".join((
            "com.android.permissioncontroller",
            "com.google.android.permissioncontroller",
            "com.android.packageinstaller",
            "com.google.android.packageinstaller",
        )),
    ).split(",")
    if package.strip()
}
TARGET_EXIT_SCENARIO_PATTERNS = (
    r"(\u9000\u51fa|\u5173\u95ed).{0,20}(\u5e94\u7528|app|App|\u7a0b\u5e8f)",
    r"(\u8fd4\u56de|\u56de\u5230|\u9000\u56de).{0,30}(\u7cfb\u7edf\u684c\u9762|\u684c\u9762|launcher|\u542f\u52a8\u5668|\u524d\u4e00\u4e2a\u5e94\u7528)",
    r"(\u9a8c\u8bc1|\u6d4b\u8bd5).{0,30}(\u9000\u51fa\u903b\u8f91|\u9000\u51fa\u529f\u80fd|\u5173\u95ed\u903b\u8f91)",
    r"\b(exit|quit|close)\b.{0,20}\b(app|application)\b",
)


def _parse_coordinate_value(value, key: str) -> float | None:
    """Parse an LLM coordinate value while tolerating JSON numeric strings."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{key} coordinate cannot be boolean: {value!r}")
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            number = float(stripped)
        except ValueError as exc:
            raise ValueError(f"{key} coordinate is not a number: {value!r}") from exc
    else:
        raise ValueError(f"{key} coordinate type is not supported: {type(value).__name__}")

    if not math.isfinite(number):
        raise ValueError(f"{key} coordinate is not a finite number: {value!r}")
    return number


def _scale_plan_coordinates(plan: dict, img_w: int, img_h: int) -> None:
    """Convert relative 0-1000 LLM coordinates in-place into screen pixels."""
    for key in ("x",):
        number = _parse_coordinate_value(plan.get(key), key)
        if number is None:
            plan.pop(key, None)
            continue
        plan[key] = round(number / 1000 * max(0, img_w - 1))
    for key in ("y",):
        number = _parse_coordinate_value(plan.get(key), key)
        if number is None:
            plan.pop(key, None)
            continue
        plan[key] = round(number / 1000 * max(0, img_h - 1))


def _normalize_action_plan(plan: dict) -> dict:
    """Drop fields that are irrelevant for the selected action."""
    plan = dict(plan)
    if not str(plan.get("action_name", "")).strip():
        for alias in ("action", "next_action", "action_type"):
            if plan.get(alias) is not None:
                plan["action_name"] = plan.get(alias)
                break
    if not str(plan.get("action_description", "")).strip() and plan.get("target"):
        plan["action_description"] = f"{plan.get('action_name', '')} {plan.get('target')}".strip()
    action_name = str(plan.get("action_name", "")).strip().lower()
    plan["action_name"] = action_name
    if "direction" in plan:
        plan["direction"] = str(plan.get("direction", "")).strip().lower()
    interruption = plan.get("is_interruption", False)
    if isinstance(interruption, str):
        normalized_interruption = interruption.strip().lower()
        if normalized_interruption in {"true", "1", "yes", "on"}:
            plan["is_interruption"] = True
        elif normalized_interruption in {"false", "0", "no", "off", ""}:
            plan["is_interruption"] = False
    if action_name in {"click", "long_press"}:
        for key in COORDINATE_FIELDS:
            value = plan.get(key)
            if isinstance(value, (list, tuple)) and len(value) >= 2:
                try:
                    start = float(value[0])
                    end = float(value[-1])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(start) and math.isfinite(end):
                    plan[key] = (start + end) / 2
    common_fields = {"action_name", "thought", "action_description", "is_interruption"}
    allowed_fields_by_action = {
        "click": common_fields | {"x", "y"},
        "long_press": common_fields | {"x", "y"},
        "type": common_fields | {"content"},
        "scroll": common_fields | {"direction"},
        "open_app": common_fields | {"app_name"},
        "press_back": common_fields,
        "finished": common_fields,
        "wait": common_fields,
    }
    allowed_fields = allowed_fields_by_action.get(action_name)
    if not allowed_fields:
        return plan
    return {
        key: value
        for key, value in plan.items()
        if key in allowed_fields and value is not None
    }


def _validate_action_plan(plan: dict) -> None:
    """Reject incomplete model actions so they are retried instead of touching the device."""
    action_name = str(plan.get("action_name", "")).strip()
    valid_actions = {"click", "long_press", "type", "scroll", "open_app", "press_back", "finished", "wait"}
    if action_name not in valid_actions:
        raise ValueError(f"Unsupported or missing action type: {action_name!r}")
    if action_name in {"click", "long_press"} and (
        plan.get("x") is None or plan.get("y") is None
    ):
        raise ValueError(f"{action_name} is missing x/y coordinates")
    if action_name in {"click", "long_press"}:
        for key in COORDINATE_FIELDS:
            number = _parse_coordinate_value(plan.get(key), key)
            if number is None or not 0 <= number <= 1000:
                raise ValueError(f"{key} coordinate must be in the range 0-1000: {plan.get(key)!r}")
    if action_name == "type" and not str(plan.get("content", "")):
        raise ValueError("type missing content")
    if action_name == "scroll":
        direction = str(plan.get("direction", "")).strip().lower()
        if direction not in {"up", "down", "left", "right"}:
            raise ValueError(f"Illegal scroll direction: {direction!r}")
    if action_name == "open_app" and not str(plan.get("app_name", "")).strip():
        raise ValueError("open_app missing app_name")
    if not isinstance(plan.get("is_interruption", False), bool):
        raise ValueError("is_interruption must be a boolean")


_ACTION_CLAUSE_MARKERS = re.compile(
    r"\b(?:next|need\s+to|prepare\s+to|about\s+to|immediately|click|long\s+press|type|enter|swipe|scroll|open|select|send|switch|launch|close|delete|mark|pin|back)\b|"
    r"\u4e0b\u4e00\u6b65|\u63a5\u4e0b\u6765|\u9700\u8981|\u51c6\u5907|\u5c06\u8981|\u7acb\u5373|\u70b9\u51fb|\u957f\u6309|\u8f93\u5165|\u6ed1\u52a8|\u6253\u5f00|\u8fdb\u5165|\u9009\u62e9|\u53d1\u9001|\u5207\u6362|\u5f00\u542f|\u5173\u95ed|\u5220\u9664|\u6807\u8bb0|\u7f6e\u9876|\u8fd4\u56de",
    re.IGNORECASE,
)


def _proposed_action_text(plan: dict) -> str:
    """Return only the intended operation, excluding unrelated visible-page prose."""
    description = str(plan.get("action_description", "")).strip()
    if description:
        return description
    content = str(plan.get("content", "")).strip()
    thought = str(plan.get("thought", "")).strip()
    clauses = [
        item.strip()
        for item in re.split(r"[.;\n\u3002\uff1b]+", thought)
        if item.strip()
    ]
    actionable = [item for item in clauses if _ACTION_CLAUSE_MARKERS.search(item)]
    intent = actionable[-1] if actionable else ""
    return " ".join(item for item in (intent, content) if item)


def _finished_declares_failure(thought: str) -> bool:
    text = str(thought or "").strip().lower()
    if not text:
        return False
    if MIXED_FINISH_FAILURE_PATTERN.search(text):
        return True
    if any(re.search(pattern, text, re.IGNORECASE) for pattern in FINISHED_OVERRIDE_FAILURE_PATTERNS):
        return True
    if any(re.search(pattern, text, re.IGNORECASE) for pattern in FINISHED_SUCCESS_PATTERNS):
        return False
    return any(pattern in text for pattern in FINISHED_FAILURE_PATTERNS)


def _verification_attempt_reason(action_name: str, *texts: object) -> str:
    if str(action_name or "").strip() in {"", "finished", "press_back", "wait"}:
        return ""
    combined = " ".join(str(text or "") for text in texts if text is not None)
    if not VERIFICATION_CHALLENGE_PATTERN.search(combined):
        return ""
    if VERIFICATION_EXIT_PATTERN.search(combined):
        return ""
    if not VERIFICATION_ATTEMPT_PATTERN.search(combined):
        return ""
    return VERIFICATION_BLOCKED_END


def _verification_finish_reason(thought: str) -> str:
    text = str(thought or "").strip()
    if not VERIFICATION_CHALLENGE_PATTERN.search(text):
        return ""
    if re.search(r"\u9a8c\u8bc1\u6210\u529f|\u5df2\u901a\u8fc7\u9a8c\u8bc1|\u6210\u529f\u901a\u8fc7|verification\s+succeeded", text, re.IGNORECASE):
        return ""
    return VERIFICATION_BLOCKED_END


def _scenario_allows_external_app(scenario: str) -> bool:
    text = str(scenario or "").strip()
    if not text:
        return False
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in EXTERNAL_APP_SCENARIO_PATTERNS)


def _scenario_allows_target_exit(scenario: str) -> bool:
    text = str(scenario or "").strip()
    if not text:
        return False
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in TARGET_EXIT_SCENARIO_PATTERNS)


def _is_safe_external_package(package_name: str) -> bool:
    package_name = str(package_name or "").strip()
    return bool(package_name and package_name in SAFE_EXTERNAL_PACKAGES)


def _uses_adb_backend(device) -> bool:
    operating_system = str(getattr(device, "operating_system", "adb") or "adb").strip().lower()
    return operating_system in {"adb", "android"}


def _current_package(device, retries: int = FOREGROUND_CHECK_RETRIES) -> str | None:
    last_error: Exception | None = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            current_package = str(device.current_package() or "").strip()
            if current_package:
                return current_package
        except Exception as exc:
            last_error = exc
            print(f"[Scenario Exec] Failed to get current foreground package name: {exc}")
        if attempt < retries:
            time.sleep(FOREGROUND_CHECK_RETRY_DELAY)
    if last_error is None:
        print("[Scenario Exec] Current foreground package name is empty, unable to confirm target app state.")
    return None


def _target_process_pid(device, package_name: str) -> str:
    package_name = str(package_name or "").strip()
    if not package_name or not hasattr(device, "connector"):
        return ""
    try:
        output = str(device.connector.shell(f"pidof {package_name}") or "").strip()
    except Exception:
        return ""
    return output.split()[0] if output else ""


def _target_app_foreground_state(device, package_name: str) -> tuple[str, str]:
    package_name = package_name.strip()
    if not package_name:
        return "foreground", ""

    current_package = _current_package(device)
    if current_package is None:
        return "unknown", ""
    if current_package == package_name:
        return "foreground", current_package
    return "background", current_package


def _is_target_app_foreground(device, package_name: str) -> bool:
    foreground_state, _ = _target_app_foreground_state(device, package_name)
    return foreground_state == "foreground"


def _looks_like_launcher_package(package_name: str) -> bool:
    return bool(LAUNCHER_PACKAGE_PATTERN.search(str(package_name or "").strip()))


def _settle_foreground_state(device, package_name: str):
    """Recheck foreground status after a brief wait to filter false positives of transient states during page transition animations.

    During some in-app transitions, current_package momentarily reports the desktop
    launcher, eventually stabilizing in a new Activity within the target app (webview, etc.); a single
    immediate check would misjudge the transient state as an external transition (external_transition_blocked),
    terminating the scenario unnecessarily and logging a failure for the function.
    """
    state, current = _target_app_foreground_state(device, package_name)
    for _ in range(EXTERNAL_RECHECK_ATTEMPTS):
        if state == "foreground":
            return state, current
        time.sleep(EXTERNAL_RECHECK_DELAY)
        state, current = _target_app_foreground_state(device, package_name)
    return state, current


def _wait_target_app_foreground(
    device,
    package_name: str,
    *,
    max_wait: float | None = None,
) -> bool:
    """Poll waiting for target app to enter the foreground, replacing fixed-duration cold start wait."""
    package_name = package_name.strip()
    if not package_name:
        return True
    wait = APP_START_MAX_WAIT if max_wait is None else max(0.0, float(max_wait))
    deadline = time.monotonic() + wait
    while True:
        if _is_target_app_foreground(device, package_name):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(APP_START_POLL_INTERVAL)


def _restart_target_app(device, package_name: str) -> bool:
    package_name = package_name.strip()
    if not package_name:
        return True

    for attempt in range(1, APP_RESTART_ATTEMPTS + 1):
        try:
            print(f"[Scenario Exec] Restarting target app, attempt {attempt}/{APP_RESTART_ATTEMPTS}.")
            device.restart_app(package_name)
            if _wait_target_app_foreground(device, package_name):
                time.sleep(APP_RESTART_SETTLE_DELAY)
                return True
            print("[Scenario Exec] Target app still not in foreground after restart.")
        except Exception as exc:
            print(f"[Scenario Exec] Failed to restart target app: {exc}")
        if attempt < APP_RESTART_ATTEMPTS:
            time.sleep(APP_RESTART_RETRY_DELAY)
    return False


def _ensure_target_app(device, package_name: str) -> bool:
    foreground_state, current_package = _target_app_foreground_state(device, package_name)
    if foreground_state == "foreground":
        return True

    if foreground_state == "unknown":
        print(f"[Scenario Exec] Cannot confirm current foreground app, target app is {package_name}, preparing to restart target app.")
    else:
        print(
            f"[Scenario Exec] Detected current foreground app is {current_package}, target app is {package_name}, "
            "preparing to restart target app."
        )
    return _restart_target_app(device, package_name)


def _recover_target_app_by_back(
    device,
    package_name: str,
    attempts: int = APP_RECOVERY_BACK_ATTEMPTS,
    *,
    continue_message: bool = True,
) -> bool:
    foreground_state, _ = _target_app_foreground_state(device, package_name)
    if foreground_state == "foreground":
        return True

    current_package = str(
        _target_app_foreground_state(device, package_name)[1] or ""
    ).strip()
    if _looks_like_launcher_package(current_package):
        # Pressing back on the launcher neither returns to the target app nor avoids triggering system gestures;
        # abandon back recovery directly and let caller take the restart path.
        print(
            "[Scenario Exec] Current foreground is launcher, back key cannot recover target app, "
            "skipping back recovery and trying restart directly."
        )
        return False

    for attempt in range(1, attempts + 1):
        if foreground_state == "unknown":
            print(f"[Scenario Exec] Cannot confirm if target app is in foreground, rechecking after wait, attempt {attempt}/{attempts}.")
        else:
            try:
                print(f"[Scenario Exec] Target app left foreground, trying back recovery attempt {attempt}/{attempts}.")
                _perform_action(device=device, action_name="press_back")
            except Exception as exc:
                print(f"[Scenario Exec] Back recovery action failed: {exc}")
        time.sleep(FAST_ACTION_DELAY)
        foreground_state, _ = _target_app_foreground_state(device, package_name)
        if foreground_state == "foreground":
            message = "Continuing current scenario execution." if continue_message else "Preparing to end current scenario."
            print(f"[Scenario Exec] Recovered to target app via back, {message}")
            return True
    return False


def _append_target_app_recovery_step(
    trace: list[dict],
    device,
    package_name: str,
    previous_package: str,
    trace_phase: str,
    recovery_method: str = "press_back",
    final_recovery: bool = False,
    reason: str = "",
) -> None:
    """Record automatic recovery from an external foreground app as non-recordable trace context."""
    try:
        page = capture_current_page(device)
    except Exception as exc:
        print(f"[Scenario Exec] Failed to capture page after back recovery: {exc}")
        page = trace[-1].get("current_page") if trace else None

    action_name = "press_back" if recovery_method == "press_back" else "restart_app"
    recovery_text = "back key" if recovery_method == "press_back" else "restart target app"
    scene_text = "When scenario ended" if final_recovery else "During execution"
    thought = (
        f"{scene_text}, target app was detected leaving foreground; runner automatically recovered to target app via {recovery_text}."
        "This is a test environment recovery step and does not represent a valid operation within the user scenario."
    )
    trace.append({
        "action_name": action_name,
        "thought": thought,
        "action_description": "Automatic recovery after target app left foreground",
        "current_page": page,
        "is_interruption": True,
        "trace_phase": trace_phase,
        "foreground_recovery": True,
        "recovery_method": recovery_method,
        "final_recovery": final_recovery,
        "observation_after_recovery": True,
        "recovered_to_target_app": True,
        "previous_package": previous_package,
        "target_package": package_name,
        "action_error": (
            reason
            or (
                f"Target app left foreground: current={previous_package or 'unknown'}, "
                f"target={package_name or 'unknown'}; recovered via {recovery_text}."
            )
        ),
    })


def _append_external_observation_step(
    trace: list[dict],
    device,
    package_name: str,
    external_package: str,
    trace_phase: str,
) -> None:
    """Preserve the external foreground frame before automatic recovery."""
    try:
        page = capture_current_page(device)
    except Exception as exc:
        print(f"[Scenario Exec] Failed to capture external page before automatic recovery: {exc}")
        return
    trace.append({
        "action_name": "capture_screenshot",
        "thought": "Runner preserved external foreground page before automatic back for execution trace auditing.",
        "action_description": "External page observation before automatic recovery",
        "current_page": page,
        "observation_only": True,
        "is_interruption": True,
        "external_context": True,
        "external_package": external_package,
        "target_package": package_name,
        "trace_phase": trace_phase,
    })
    screenshot_ref = save_trace_screenshot(
        page,
        phase=trace_phase,
        step_index=len(trace),
        moment="external_before_recovery",
        metadata={
            "foreground_package": external_package,
            "target_package": package_name,
            "observation_only": True,
            "is_interruption": True,
        },
    )
    if screenshot_ref:
        trace[-1]["trace_screenshot_before"] = screenshot_ref
    print(
        "[Scenario Exec] Saved external page observation before automatic recovery: "
        f"current={external_package or 'unknown'}, target={package_name or 'unknown'}"
    )


def _recover_from_exploration_boundary(
    device,
    package_name: str,
    hit: BoundaryHit,
) -> bool:
    """Automatically exit pages crossing exploration boundaries (such as mini-programs) and return whether left the boundary.

    Different from external app recovery: boundary pages still run in target app package (foreground package name unchanged),
    so recheck whether boundary is exited by Activity name rather than package name.
    """
    attempts = max(1, APP_RECOVERY_BACK_ATTEMPTS)
    for attempt in range(1, attempts + 1):
        try:
            print(
                f"[Exploration Boundary] Attempting {attempt}/{attempts} {hit.rule.exit_action} "
                f"to exit {hit.rule.label} (via={hit.via}, matched={hit.matched})."
            )
            _perform_action(device=device, action_name="press_back")
        except Exception as exc:
            print(f"[Exploration Boundary] Exit action failed: {exc}")
        time.sleep(FAST_ACTION_DELAY)
        try:
            current_activity = str(device.current_activity() or "")
        except Exception as exc:
            print(f"[Exploration Boundary] Failed to read current Activity: {exc}")
            current_activity = ""
        if not current_activity or check_boundary(package_name, current_activity) is None:
            return True
    return False


def _recover_target_app_at_scene_end(
    trace: list[dict],
    device,
    package_name: str,
    trace_phase: str,
) -> tuple[bool, str, str]:
    foreground_state, current_package = _target_app_foreground_state(device, package_name)
    if foreground_state == "foreground":
        return False, "", ""

    previous_package = current_package or "unknown"
    print(
        "[Scenario Exec] Still not in target app when scenario ended, preparing recovery: "
        f"status={foreground_state}, current={previous_package}, target={package_name or 'unknown'}"
    )
    crash_logcat = _read_crash_logcat(device, package_name)
    if _logcat_has_target_process_crash(crash_logcat, package_name):
        print("[Scenario Exec] Detected target app crash when scenario ended; restarted target app for subsequent scenarios to continue.")
        if _restart_target_app(device, package_name):
            _append_target_app_recovery_step(
                trace,
                device,
                package_name,
                previous_package,
                trace_phase,
                recovery_method="restart_app",
                final_recovery=True,
            )
            return True, "restart_app", crash_logcat
        return False, "", crash_logcat

    if _recover_target_app_by_back(device, package_name, continue_message=False):
        _append_target_app_recovery_step(
            trace,
            device,
            package_name,
            previous_package,
            trace_phase,
            recovery_method="press_back",
            final_recovery=True,
        )
        return True, "press_back", ""

    if _restart_target_app(device, package_name):
        _append_target_app_recovery_step(
            trace,
            device,
            package_name,
            previous_package,
            trace_phase,
            recovery_method="restart_app",
            final_recovery=True,
        )
        return True, "restart_app", crash_logcat
    return False, "", crash_logcat


def _clear_logcat(device):
    if not _uses_adb_backend(device):
        return
    device_serial = device.serial if hasattr(device, "serial") else ""
    if not device_serial:
        print("[Scenario Exec] Failed to obtain device serial number, skipping logcat clear.")
        return
    try:
        subprocess.run(
            ["adb", "-s", device_serial, "logcat", "-c"],
            timeout=5,
            capture_output=True,
        )
        print(f"[Scenario Exec] Cleared logcat: device={device_serial}")
    except Exception as exc:
        print(f"[Scenario Exec] Failed to clear logcat: {exc}")


def _read_crash_logcat(device, package_name: str = "") -> str:
    if not _uses_adb_backend(device):
        return ""
    device_serial = device.serial if hasattr(device, "serial") else ""
    if not device_serial:
        return ""
    try:
        result = subprocess.run(
            [
                "adb",
                "-s",
                device_serial,
                "logcat",
                "-d",
                "-v",
                "time",
                "-t",
                str(CRASH_LOGCAT_TAIL_LINES),
            ],
            timeout=8,
            capture_output=True,
            text=True,
            errors="replace",
        )
    except Exception as exc:
        print(f"[Scenario Exec] Failed to read crash logcat: {exc}")
        return ""

    output = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
    lines = [line.rstrip() for line in output.splitlines() if line.strip()]
    if not lines:
        return ""

    package_name = package_name.strip()
    has_crash_signal = any(any(pattern in line for pattern in CRASH_LOGCAT_PATTERNS) for line in lines)
    if not has_crash_signal:
        return ""

    filtered = [
        line
        for line in lines
        if not _is_non_crash_logcat_noise(line)
        and (
            any(pattern in line for pattern in CRASH_LOGCAT_PATTERNS)
            or (package_name and package_name in line)
        )
    ]
    selected = filtered[-CRASH_LOGCAT_KEEP_LINES:] if filtered else []
    return "\n".join(selected)


def _logcat_has_target_process_crash(crash_logcat: str, package_name: str) -> bool:
    text = str(crash_logcat or "")
    package_name = str(package_name or "").strip()
    if not text or not package_name:
        return False
    non_empty_lines = [line for line in text.splitlines() if line.strip()]
    if non_empty_lines and all(_is_non_crash_logcat_noise(line) for line in non_empty_lines):
        return False

    escaped_package = re.escape(package_name)
    if re.search(rf"\bANR in\s+{escaped_package}\b", text):
        return True
    if re.search(rf"\bProcess:\s*{escaped_package}\b", text) and "FATAL EXCEPTION" in text:
        return True
    if "Fatal signal" in text and (
        f">>> {package_name} <<<" in text
        or re.search(rf"\bCmdline:\s*{escaped_package}\b", text)
    ):
        return True
    strong_crash_signal = any(
        signal in text
        for signal in ("FATAL EXCEPTION", "AndroidRuntime", "Fatal signal", "App crash", "ANR in")
    )
    if strong_crash_signal and re.search(rf"\bForce finishing activity\s+{escaped_package}/", text):
        return True
    if strong_crash_signal and re.search(rf"\bProcess\s+{escaped_package}\s+\(pid\s+\d+\)\s+has died\b", text):
        return True
    return False


def _sleep_for_llm_retry(consecutive_errors: int):
    time.sleep(compute_retry_delay_seconds(consecutive_errors))


def _page_label(page: Page | None) -> str:
    if not isinstance(page, Page):
        return "Unknown Page"
    return page.name or page.description or page.activity or "Unknown Page"


def _format_execution_history(trace: list[dict]) -> str:
    if not trace:
        return "No history steps yet."

    lines = []
    for index, step in enumerate(trace, start=1):
        page = step.get("current_page")
        action = str(step.get("action_name", "")).strip() or "unknown action"
        thought = str(step.get("thought", "")).strip()
        content = str(step.get("content", "")).strip()
        direction = str(step.get("direction", "")).strip()
        detail_parts = []
        if step.get("external_context"):
            external_package = str(step.get("external_package", "")).strip() or "external app"
            detail_parts.append(f"External page: {external_package}")
        if step.get("caused_external_transition"):
            external_package = str(step.get("external_package", "")).strip() or "external app"
            detail_parts.append(f"Triggered external transition: {external_package}")
        if step.get("is_interruption"):
            detail_parts.append("Interruption/recovery step")
        if step.get("foreground_recovery"):
            previous_package = str(step.get("previous_package", "")).strip() or "unknown"
            target_package = str(step.get("target_package", "")).strip() or "target app"
            detail_parts.append(f"Auto back recovery: {previous_package} -> {target_package}")
        if content:
            detail_parts.append(f"Input/Text: {content}")
        if direction:
            detail_parts.append(f"Direction: {direction}")
        if step.get("action_failed"):
            action_error = str(step.get("action_error", "")).strip() or "Device action execution failed"
            detail_parts.append(f"Action failed: {action_error[:200]}")
        detail = f" | {'; '.join(detail_parts)}" if detail_parts else ""
        thought_text = f" | Thought: {thought}" if thought else ""
        lines.append(f"{index}. Page: {_page_label(page)} | Action: {action}{detail}{thought_text}")
    return "\n".join(lines)


def _foreground_state_note(foreground_package: str, package_name: str) -> str:
    """Give the model package-authoritative foreground state information."""
    foreground_package = str(foreground_package or "").strip()
    package_name = str(package_name or "").strip()
    if not foreground_package or not package_name:
        return ""
    if foreground_package == package_name:
        return f"[Foreground State] Foreground app: {foreground_package} (target app, screenshot is its current interface; Activity name containing Launcher/Home cannot be used to judge it as mobile launcher).\n"
    if _looks_like_launcher_package(foreground_package):
        return f"[Foreground State] Foreground app: {foreground_package} (mobile launcher, target app {package_name} is not in foreground, may still be launching).\n"
    return f"[Foreground State] Foreground app: {foreground_package} (non-target app {package_name}).\n"


def _build_step_input(
    trace: list[dict],
    current_screenshot: str,
    step_index: int,
    max_steps: int,
    screenshot_health_error: str = "",
    foreground_note: str = "",
) -> list[dict]:
    remaining_steps = max(0, max_steps - step_index + 1)
    scroll_budget_note = ""
    if trace and str(trace[-1].get("trace_phase", "")).strip() != "navigation":
        scroll_direction, consecutive_scrolls = (
            _consecutive_same_direction_scroll(trace)
        )
        if consecutive_scrolls >= TEST_MAX_CONSECUTIVE_SAME_DIRECTION_SCROLLS:
            scroll_budget_note = (
                f"[Scroll Budget] Scrolled {_format_scroll_direction(scroll_direction)} continuously "
                f"{consecutive_scrolls} times, reaching the upper limit for same-direction scrolling "
                f"({TEST_MAX_CONSECUTIVE_SAME_DIRECTION_SCROLLS} times). "
                "Continuing to scroll in this direction is prohibited; please switch to clicking list items, going back, or exploring other directions.\n"
            )
        elif consecutive_scrolls >= TEST_MAX_CONSECUTIVE_SAME_DIRECTION_SCROLLS - 1:
            scroll_budget_note = (
                f"[Scroll Budget] Scrolled {_format_scroll_direction(scroll_direction)} continuously "
                f"{consecutive_scrolls} times, up to "
                f"{TEST_MAX_CONSECUTIVE_SAME_DIRECTION_SCROLLS} consecutive scrolls in the same direction are allowed. "
                "Next step please prepare to switch to clicking list items, going back, or other exploration.\n"
            )
    content: list[dict] = [
        {
            "type": "text",
            "text": (
                f"[Step {step_index}] Please decide the next action based on the test task, complete execution history, and current screenshot.\n"
                f"At most {max_steps} steps in this phase; currently at step {step_index}, with {remaining_steps} steps remaining including this one.\n"
                "If remaining steps are insufficient for valuable exploration, prioritize recording current verifiable results and choose finished.\n"
                "Only operate on one representative control among duplicate controls having the same function and reaching the same result; "
                "intermediate result pages are not the default endpoint; finished only when the exploration direction is complete or there are no clearly relevant unexecuted controls.\n"
                "Check before outputting: whether the same action was recently performed on the same control on the same page without changes; "
                "if so, do not repeat the original action, change strategy or finished.\n"
                f"{scroll_budget_note}"
                "[Complete Execution History]\n"
                f"{_format_execution_history(trace)}\n\n"
                f"[Last {KEEP_IMAGE_STEPS} Historical Screenshots]"
            ),
        }
    ]

    recent_steps = trace[-KEEP_IMAGE_STEPS:]
    for offset, step in enumerate(recent_steps, start=1):
        page = step.get("current_page")
        screenshot = page.representative_screenshot if isinstance(page, Page) else ""
        if not screenshot:
            continue
        absolute_index = len(trace) - len(recent_steps) + offset
        content.append({
            "type": "text",
            "text": f"Historical screenshot: Page before executing step {absolute_index}."
        })
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{screenshot}"},
        })

    content.append({
        "type": "text",
        "text": (
            f"{foreground_note}"
            "[Current Screenshot] Please output the next JSON action based on this screenshot."
            + (
                "\nScreenshot health check indicates the current interface may be a black screen or unavailable state. "
                "If it can be exited via back key, prioritize executing a press_back; do not repeatedly enter the entry that caused this state."
                if screenshot_health_error
                else ""
            )
        ),
    })
    content.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{current_screenshot}"},
    })
    return content


def _parse_action_json(raw_content: str) -> dict[str, Any]:
    """Parse the last valid JSON object, tolerating a model's corrected draft."""
    text = JSON_BLOCK_PATTERN.sub("", str(raw_content or "").strip()).strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except (TypeError, ValueError, json.JSONDecodeError):
        pass

    # Unified repair for known high-frequency typos (e.g. "x": 226, 190 missing "y": key):
    # Shared with llm.invoke_json repair layer to avoid rule drift across parsing paths.
    repaired_text = _repair_common_json_typos(text)
    if repaired_text != text:
        try:
            parsed = json.loads(repaired_text)
            if isinstance(parsed, dict):
                return parsed
        except (TypeError, ValueError, json.JSONDecodeError):
            pass

    # Some lightweight vision models occasionally emit an anonymous second
    # coordinate between x and y: `"x": 52, 70, "y": 70`.  It cannot be a
    # legitimate JSON field, and the explicit y value remains authoritative.
    # Repair only this narrow shape so unrelated malformed payloads still retry.
    repaired_text = re.sub(
        r'("x"\s*:\s*-?\d+(?:\.\d+)?)\s*,\s*-?\d+(?:\.\d+)?\s*,(?=\s*"y"\s*:)',
        r"\1,",
        repaired_text,
    )
    if repaired_text != text:
        try:
            parsed = json.loads(repaired_text)
            if isinstance(parsed, dict):
                return parsed
        except (TypeError, ValueError, json.JSONDecodeError):
            pass

    decoder = json.JSONDecoder()
    parsed_objects: list[dict[str, Any]] = []
    for match in re.finditer(r"\{", text):
        try:
            candidate, _end = decoder.raw_decode(text[match.start():])
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(candidate, dict):
            parsed_objects.append(candidate)
    if parsed_objects:
        return parsed_objects[-1]
    raise ValueError("No valid JSON object found in model output")


def _recent_screenshot_count(trace: list[dict]) -> int:
    count = 0
    for step in trace[-KEEP_IMAGE_STEPS:]:
        page = step.get("current_page")
        if isinstance(page, Page) and page.representative_screenshot:
            count += 1
    return count


def _step_signature(step: dict) -> tuple:
    page = step.get("current_page")
    page_activity = page.activity if isinstance(page, Page) else ""
    page_name = page.name if isinstance(page, Page) else ""
    return (
        str(step.get("action_name", "")).strip(),
        page_activity,
        page_name,
        int(step.get("x", 0) or 0) // 20,
        int(step.get("y", 0) or 0) // 20,
        str(step.get("content", "")).strip(),
        str(step.get("direction", "")).strip(),
    )


def _page_state_key(step: dict) -> tuple[str, str, str]:
    page = step.get("current_page")
    if not isinstance(page, Page):
        return "", "", ""
    return (
        str(page.id or "").strip(),
        str(page.activity or "").strip(),
        str(page.name or "").strip(),
    )


def _page_screenshot(step: dict) -> str:
    page = step.get("current_page")
    if isinstance(page, Page):
        return str(page.representative_screenshot or "")
    return ""


def _screenshots_stable(steps: list[dict]) -> bool:
    screenshots = [_page_screenshot(step) for step in steps]
    if any(not screenshot for screenshot in screenshots):
        return False

    baseline = screenshots[0]
    similarities: list[float] = []
    for screenshot in screenshots[1:]:
        similarity = screenshot_similarity(baseline, screenshot)
        if similarity is None:
            return False
        similarities.append(similarity)

    return bool(similarities) and min(similarities) >= STALL_SCREENSHOT_SIMILARITY_THRESHOLD


def _is_consecutive_repeated_action(steps: list[dict]) -> bool:
    action_names = [str(step.get("action_name", "")).strip() for step in steps]
    if not action_names or len(set(action_names)) != 1:
        return False

    action_name = action_names[0]
    if action_name == "finished":
        return False

    if action_name in {"click", "long_press"}:
        xs = [int(step.get("x", 0) or 0) for step in steps]
        ys = [int(step.get("y", 0) or 0) for step in steps]
        same_target = (
            max(xs) - min(xs) <= REPEAT_COORD_TOLERANCE
            and max(ys) - min(ys) <= REPEAT_COORD_TOLERANCE
        )
        return same_target and _screenshots_stable(steps)
    if action_name == "type":
        same_content = len({str(step.get("content", "")).strip() for step in steps}) <= 1
        return same_content and _screenshots_stable(steps)
    if action_name == "scroll":
        same_direction = len({str(step.get("direction", "")).strip() for step in steps}) <= 1
        return same_direction and _screenshots_stable(steps)
    if action_name == "open_app":
        same_app = len({str(step.get("app_name", "")).strip() for step in steps}) <= 1
        return same_app and _screenshots_stable(steps)
    return action_name in {"press_back", "wait"} and _screenshots_stable(steps)


def _semantic_action_intent(step: dict) -> str:
    text = " ".join(
        str(step.get(field, "") or "").strip()
        for field in ("action_description", "thought")
    ).lower()
    text = re.sub(
        r"\u5f53\u524d|\u5df2\u7ecf|\u5df2\u6210\u529f|\u672c\u6b21|\u6b64\u524d|\u518d\u6b21|\u51c6\u786e|\u5c1d\u8bd5|\u627e\u5230|\u9875\u9762|\u6309\u94ae|\u5165\u53e3|\u8be5|\u8fdb\u884c|"
        r"\u4efb\u52a1\u76ee\u6807|\u8fdb\u5165\u5bf9\u5e94|\u8fdb\u5165|\u70b9\u51fb|\u64cd\u4f5c|\u67e5\u770b|\u7ed3\u679c|\u5b8c\u6210|\u8fbe\u6210|\u540e\u7ed3\u675f|\s+|[\uff0c\u3002\u3001\u201c\u201d'\"]",
        "",
        text,
    )
    return text


def _is_semantic_click_stall(steps: list[dict]) -> bool:
    if len(steps) < SEMANTIC_CLICK_STALL_REPEAT_WINDOW:
        return False
    recent = steps[-SEMANTIC_CLICK_STALL_REPEAT_WINDOW:]
    if any(str(step.get("action_name", "")).strip() not in {"click", "long_press"} for step in recent):
        return False
    if not _screenshots_stable(recent):
        return False
    intents = [_semantic_action_intent(step) for step in recent]
    if any(len(intent) < 4 for intent in intents):
        return False
    return SequenceMatcher(None, intents[0], intents[1]).ratio() >= 0.72


def _repeated_scroll_stall_direction(steps: list[dict]) -> str:
    if len(steps) < SCROLL_STALL_REPEAT_WINDOW:
        return ""
    recent = steps[-SCROLL_STALL_REPEAT_WINDOW:]
    if any(str(step.get("action_name", "")).strip() != "scroll" for step in recent):
        return ""
    directions = [str(step.get("direction", "")).strip() for step in recent]
    if not directions or len(set(directions)) != 1:
        return ""
    if not _screenshots_stable(recent):
        return ""
    return directions[0] or "unknown direction"


def _format_scroll_direction(direction: str) -> str:
    return {
        "up": "up",
        "down": "down",
        "left": "left",
        "right": "right",
    }.get(str(direction or "").strip(), str(direction or "").strip() or "unknown direction")


def _consecutive_same_direction_scroll(trace: list[dict]) -> tuple[str, int]:
    """Count consecutive scroll steps in the same direction backwards from the tail.

    Only counts consecutive segments where action_name == "scroll"; inserting other actions or changing direction
    will break the count. Returns 0 when the tail action is not scroll (or lacks direction).
    """
    if not trace:
        return "", 0
    latest = trace[-1]
    if str(latest.get("action_name", "")).strip() != "scroll":
        return "", 0
    latest_direction = str(latest.get("direction", "")).strip()
    if not latest_direction:
        return "", 0
    count = 0
    for step in reversed(trace):
        if str(step.get("action_name", "")).strip() != "scroll":
            break
        if str(step.get("direction", "")).strip() != latest_direction:
            break
        count += 1
    return latest_direction, count


def _scroll_intent_repeat_window(trace: list[dict]) -> int:
    phase = str(trace[-1].get("trace_phase", "")).strip() if trace else ""
    if phase == "navigation":
        return NAVIGATION_SCROLL_INTENT_REPEAT_WINDOW
    return TEST_SCROLL_INTENT_REPEAT_WINDOW


def _normalize_scroll_intent_text(text: str) -> str:
    text = str(text or "").strip().lower()
    text = re.sub(r"[0-9\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u591a]+\u6b21", "", text)
    text = re.sub(r"(\u7ecf\u8fc7|\u5df2\u7ecf|\u4ecd\u7136|\u5f53\u524d|\u9700\u8981|\u7ee7\u7eed|\u9875\u9762|\u5e95\u90e8|\u533a\u57df|\u5b8c\u5168|\u5c1a\u672a|\u76ee\u6807)", "", text)
    text = re.sub(r"[\"'\u201c\u201d\u2018\u2019\uff08\uff09()\u3010\u3011\[\]{}<>\u300a\u300b\u3001\uff0c,\u3002.\uff1a:\uff1b;!\uff01?\uff1f\-_/\s]+", "", text)
    return text


def _scroll_intent_tokens(step: dict) -> set[str]:
    text = " ".join((
        str(step.get("thought", "")).strip(),
        str(step.get("action_description", "")).strip(),
        str(step.get("content", "")).strip(),
    ))
    tokens: set[str] = set()
    for match in re.findall(r"[\"'\u201c\u201d\u2018\u2019\u300c\u300d\u300e\u300f`]{1}([^\"'\u201c\u201d\u2018\u2019\u300c\u300d\u300e\u300f`]{1,32})[\"'\u201c\u201d\u2018\u2019\u300c\u300d\u300e\u300f`]{1}", text):
        token = _normalize_scroll_intent_text(match)
        if len(token) >= 2:
            tokens.add(token)
    for match in re.findall(r"(?<![A-Za-z0-9_])[A-Za-z][A-Za-z0-9_-]{1,30}(?![A-Za-z0-9_])", text):
        token = match.lower()
        if token not in SCROLL_INTENT_STOPWORDS:
            tokens.add(token)
    for match in re.findall(r"(?:\u76ee\u6807|\u627e\u5230|\u5bfb\u627e|\u663e\u793a|\u9732\u51fa|\u70b9\u51fb)?([\u4e00-\u9fffA-Za-z0-9_-]{1,16})(?:\u6309\u94ae|\u63a7\u4ef6|\u6587\u672c|\u5165\u53e3)", text):
        token = _normalize_scroll_intent_text(match)
        if len(token) >= 2 and token not in {"\u76ee\u6807", "\u6309\u94ae", "\u63a7\u4ef6", "\u6587\u672c", "\u5165\u53e3"}:
            tokens.add(token)
    return tokens


def _scroll_intent_is_search_text(step: dict) -> bool:
    text = " ".join((
        str(step.get("thought", "")).strip(),
        str(step.get("action_description", "")).strip(),
    )).lower()
    return any(marker in text for marker in SCROLL_INTENT_SEARCH_MARKERS)


def _repeated_scroll_intent_reason(trace: list[dict]) -> str:
    repeat_window = _scroll_intent_repeat_window(trace)
    if len(trace) < repeat_window:
        return ""
    recent = trace[-repeat_window:]
    if any(str(step.get("action_name", "")).strip() != "scroll" for step in recent):
        return ""

    directions = [str(step.get("direction", "")).strip() for step in recent]
    if not directions or len(set(directions)) != 1:
        return ""

    token_sets = [_scroll_intent_tokens(step) for step in recent]
    common_tokens = set.intersection(*token_sets) if all(token_sets) else set()
    direction = _format_scroll_direction(directions[0])
    if common_tokens:
        target = sorted(common_tokens, key=lambda item: (-len(item), item))[0]
        return f"Scrolled {direction} {repeat_window} consecutive times searching for the same target \"{target}\""

    if not all(_scroll_intent_is_search_text(step) for step in recent):
        return ""

    normalized_texts = [
        _normalize_scroll_intent_text(
            " ".join((
                str(step.get("thought", "")).strip(),
                str(step.get("action_description", "")).strip(),
            ))
        )
        for step in recent
    ]
    if any(len(text) < 12 for text in normalized_texts):
        return ""
    baseline = normalized_texts[0]
    if all(
        SequenceMatcher(None, baseline, text).ratio() >= SCROLL_INTENT_TEXT_SIMILARITY_THRESHOLD
        for text in normalized_texts[1:]
    ):
        return f"Scrolled {direction} {repeat_window} consecutive times executing highly similar search actions"

    return ""


def _detect_page_navigation_cycle(trace: list[dict]) -> str:
    if len(trace) < PAGE_NAVIGATION_CYCLE_WINDOW:
        return ""

    recent = trace[-PAGE_NAVIGATION_CYCLE_WINDOW:]
    action_names = [str(step.get("action_name", "")).strip() for step in recent]
    if any(action not in PAGE_CYCLE_ACTIONS for action in action_names):
        return ""

    page_keys = [_page_state_key(step) for step in recent]
    if any(not any(key) for key in page_keys):
        return ""

    unique_pages = set(page_keys)
    if not (2 <= len(unique_pages) <= 3):
        return ""

    if (
        len(unique_pages) == 2
        and page_keys[-6] == page_keys[-4] == page_keys[-2]
        and page_keys[-5] == page_keys[-3] == page_keys[-1]
        and page_keys[-1] != page_keys[-2]
    ):
        return "Detected page navigation loop"

    if page_keys[-6:-3] == page_keys[-3:]:
        return "Detected page navigation loop"

    return ""


def _detect_visual_action_cycle(trace: list[dict]) -> str:
    """Detect an A->B->A->B UI loop even when both states share one Activity."""
    if len(trace) < 4:
        return ""
    recent = trace[-4:]
    if any(
        str(step.get("action_name", "")).strip() not in PAGE_CYCLE_ACTIONS
        for step in recent
    ):
        return ""
    screenshots = [_page_screenshot(step) for step in recent]
    if any(not screenshot for screenshot in screenshots):
        return ""
    same_a = screenshot_similarity(screenshots[0], screenshots[2])
    same_b = screenshot_similarity(screenshots[1], screenshots[3])
    different_states = screenshot_similarity(screenshots[0], screenshots[1])
    if None in {same_a, same_b, different_states}:
        return ""
    if (
        same_a >= STALL_SCREENSHOT_SIMILARITY_THRESHOLD
        and same_b >= STALL_SCREENSHOT_SIMILARITY_THRESHOLD
        and different_states < STALL_SCREENSHOT_SIMILARITY_THRESHOLD
    ):
        return "Detected UI state round-trip loop within the same Activity"
    return ""


def _count_distinct_visual_states(steps: list[dict]) -> int | None:
    """Cluster steps into distinct visual states; None when screenshots unusable."""
    screenshots = [_page_screenshot(step) for step in steps]
    if any(not screenshot for screenshot in screenshots):
        return None

    representatives: list[str] = []
    for screenshot in screenshots:
        for representative in representatives:
            similarity = screenshot_similarity(representative, screenshot)
            if similarity is None:
                return None
            if similarity >= NO_PROGRESS_SIMILARITY_THRESHOLD:
                break
        else:
            representatives.append(screenshot)
    return len(representatives)


def _detect_no_progress_stall(trace: list[dict]) -> str:
    """Catch direction-agnostic blind probing that existing detectors miss.

    Repeated actions, same-direction scrolling, and A->B->A->B round-trips all have dedicated criteria, but blind trial across mixed directions
    (e.g., alternating scroll up/left/right and click) bypasses all criteria, maxing out the steps.
    Here we only rely on Activity and screenshots, not on UI hierarchy, nor requiring identical actions or directions.
    """
    if len(trace) < NO_PROGRESS_STALL_WINDOW:
        return ""

    recent = trace[-NO_PROGRESS_STALL_WINDOW:]
    if any(
        str(step.get("action_name", "")).strip() not in NO_PROGRESS_STALL_ACTIONS
        for step in recent
    ):
        return ""
    # Interruption recovery itself switches screens and is handled by dedicated criteria; do not intervene here.
    if any(bool(step.get("is_interruption")) for step in recent):
        return ""

    activities = {str(_page_state_key(step)[1]).strip() for step in recent}
    if len(activities) != 1:
        return ""
    activity = next(iter(activities))
    if not activity:
        return ""

    visual_states = _count_distinct_visual_states(recent)
    if visual_states is None or visual_states > NO_PROGRESS_MAX_VISUAL_STATES:
        return ""

    return (
        f"Repeatedly probing in the same interface for {NO_PROGRESS_STALL_WINDOW} consecutive steps, "
        f"with only {visual_states} visual states; determined current goal is unreachable"
    )


def _detect_stall(trace: list[dict]) -> str:
    if len(trace) < min(WAIT_STALL_REPEAT_WINDOW, CLICK_STALL_REPEAT_WINDOW):
        return ""

    page_cycle_reason = _detect_page_navigation_cycle(trace)
    if page_cycle_reason:
        return page_cycle_reason
    visual_cycle_reason = _detect_visual_action_cycle(trace)
    if visual_cycle_reason:
        return visual_cycle_reason
    no_progress_reason = _detect_no_progress_stall(trace)
    if no_progress_reason:
        return no_progress_reason

    if len(trace) >= 4:
        recent_cycle = trace[-4:]
        cycle_signatures = [_step_signature(step) for step in recent_cycle]
        if (
            any(bool(step.get("is_interruption")) for step in recent_cycle)
            and cycle_signatures[:2] == cycle_signatures[2:]
        ):
            return "Detected repeated loop between target action and interruption recovery"

    latest_action = str(trace[-1].get("action_name", "")).strip()
    if latest_action in {"click", "long_press"} and _is_semantic_click_stall(trace):
        return "Page still has no noticeable change after retrying once with same click intent"
    if latest_action == "scroll":
        scroll_phase = str(trace[-1].get("trace_phase", "")).strip()
        if scroll_phase != "navigation":
            scroll_direction, consecutive_scrolls = (
                _consecutive_same_direction_scroll(trace)
            )
            if (
                consecutive_scrolls
                > TEST_MAX_CONSECUTIVE_SAME_DIRECTION_SCROLLS
            ):
                return (
                    f"Scrolled {_format_scroll_direction(scroll_direction)} continuously "
                    f"{consecutive_scrolls} times, exceeding scroll budget for the same direction "
                    f"({TEST_MAX_CONSECUTIVE_SAME_DIRECTION_SCROLLS} times), "
                    "should switch to clicking list items, returning, or exploring other directions"
                )
        repeated_intent_reason = _repeated_scroll_intent_reason(trace)
        if repeated_intent_reason:
            return repeated_intent_reason
        scroll_direction = _repeated_scroll_stall_direction(trace)
        if scroll_direction:
            return f"Scrolled {_format_scroll_direction(scroll_direction)} {SCROLL_STALL_REPEAT_WINDOW} consecutive times with no noticeable change on the page"

    if latest_action in {"click", "long_press"}:
        repeat_window = CLICK_STALL_REPEAT_WINDOW
    elif latest_action == "scroll":
        repeat_window = SCROLL_STALL_REPEAT_WINDOW
    elif latest_action == "wait":
        repeat_window = WAIT_STALL_REPEAT_WINDOW
    elif latest_action == "open_app":
        repeat_window = OPEN_APP_STALL_REPEAT_WINDOW
    else:
        repeat_window = STALL_REPEAT_WINDOW
    if len(trace) < repeat_window:
        return ""

    recent = trace[-repeat_window:]
    if _is_consecutive_repeated_action(recent):
        action_name = str(recent[-1].get("action_name", "")).strip()
        if action_name == "type":
            return f"Detected {repeat_window} consecutive repeated inputs"
        if action_name in {"click", "long_press"}:
            return f"Detected {repeat_window} consecutive repeated clicks on the same area"
        return f"Detected {repeat_window} consecutive repeated actions: {action_name}"

    if len(trace) >= 8:
        latest = [_step_signature(step) for step in trace[-4:]]
        previous = [_step_signature(step) for step in trace[-8:-4]]
        if latest == previous and len(set(latest)) <= 2:
            return "Detected repeated pattern loop"

    return ""


def _trace_external_app_used(trace: list[dict]) -> bool:
    return any(
        bool(step.get("external_context"))
        or bool(step.get("caused_external_transition"))
        or bool(step.get("external_app_used"))
        for step in trace
    )


def _trace_recovered_to_target_app(trace: list[dict]) -> bool:
    return any(bool(step.get("recovered_to_target_app")) for step in trace)


def _trace_final_recovery_method(trace: list[dict]) -> str:
    for step in reversed(trace):
        method = str(step.get("final_recovery_method") or step.get("recovery_method") or "").strip()
        if method and (step.get("final_recovery") or step.get("recovered_to_target_app")):
            return method
    return ""


def _screenshot_unavailable_count(state: AgentState) -> int:
    try:
        return int(state.get("screenshot_unavailable_count", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _append_unavailable_observation(
    trace: list[dict],
    page: Page,
    health_error: str,
    trace_phase: str,
) -> None:
    trace.append({
        "action_name": "capture_screenshot",
        "thought": "Black screen or unavailable interface observed at runtime, ending current scenario and leaving recovery to next round.",
        "action_description": "Observed black screen or unavailable page",
        "current_page": page,
        "trace_phase": trace_phase,
        "observation_only": True,
        "screenshot_health_error": health_error,
    })


def _append_terminal_observation(
    trace: list[dict],
    device,
    trace_phase: str,
) -> tuple[Page | None, str]:
    """Capture one final observation when the last real action has no result frame."""
    if not trace:
        return None, ""
    last_step = trace[-1]
    action_name = str(last_step.get("action_name", "") or "").strip()
    if (
        action_name in {"", "finished", "capture_screenshot", "restart_app"}
        or bool(last_step.get("observation_only"))
        or bool(last_step.get("action_failed"))
        or bool(last_step.get("observation_after_recovery"))
    ):
        return None, ""
    try:
        page, health_error = capture_runtime_page(device)
    except ScreenshotUnavailableError as exc:
        print(f"[Scenario Exec] Final observation before termination is unavailable: {exc}")
        return None, str(exc)
    except Exception as exc:
        print(f"[Scenario Exec] Final observation before termination failed: {exc}")
        return None, ""
    trace.append({
        "action_name": "capture_screenshot",
        "thought": "Supplementing result observation of the last side-effect action before scenario end.",
        "action_description": "Final page observation before scenario termination",
        "current_page": page,
        "trace_phase": trace_phase,
        "observation_only": True,
        "terminal_observation": True,
        **({"screenshot_health_error": health_error} if health_error else {}),
    })
    print(
        "[Scenario Exec] Supplemented final observation before termination: "
        f"phase={trace_phase}, page={page.id or 'unknown'}, "
        f"health={'ok' if not health_error else health_error}"
    )
    return page, health_error


def _persist_unwritten_trace_screenshots(trace: list[dict], trace_phase: str) -> int:
    """Persist helper/recovery observations not written by the main action loop."""
    saved = 0
    for index, step in enumerate(trace, start=1):
        if str(step.get("trace_screenshot_before", "") or "").strip():
            continue
        page = step.get("current_page")
        screenshot_ref = save_trace_screenshot(
            page,
            phase=str(step.get("trace_phase") or trace_phase),
            step_index=index,
            moment=(
                "terminal_observation"
                if step.get("terminal_observation")
                else "recovery_observation"
                if step.get("observation_after_recovery")
                else "trace_observation"
            ),
            metadata={
                "action_name": str(step.get("action_name", "") or ""),
                "action_description": str(step.get("action_description", "") or ""),
                "observation_only": bool(step.get("observation_only")),
                "is_interruption": bool(step.get("is_interruption")),
                "screenshot_health_error": str(step.get("screenshot_health_error", "") or ""),
            },
        )
        if screenshot_ref:
            step["trace_screenshot_before"] = screenshot_ref
            saved += 1
    return saved


def _restart_after_repeated_black_screen(
    trace: list[dict],
    device,
    package_name: str,
    trace_phase: str,
) -> tuple[Page | None, str]:
    print("[Scenario Exec] Consecutive black screens reached threshold, restarting target App and waiting for UI to stabilize.")
    if not _restart_target_app(device, package_name):
        return None, "Failed to restart target App after consecutive black screens"
    try:
        page = capture_current_page(device)
    except ScreenshotUnavailableError as exc:
        return None, f"Interface still unavailable after restarting from consecutive black screens: {exc}"
    except Exception as exc:
        return None, f"Failed to capture page after restarting from consecutive black screens: {exc}"
    trace.append({
        "action_name": "restart_app",
        "thought": "Consecutive black screens reached threshold, runner restarted target App to recover testable state.",
        "action_description": "Restart target App after consecutive black screens",
        "current_page": page,
        "trace_phase": trace_phase,
        "is_interruption": True,
        "foreground_recovery": True,
        "recovery_method": "restart_app",
        "recovered_to_target_app": True,
        "observation_after_recovery": True,
    })
    screenshot_ref = save_trace_screenshot(
        page,
        phase=trace_phase,
        step_index=len(trace),
        moment="black_screen_restart_observation",
        metadata={
            "action_name": "restart_app",
            "recovery_reason": "repeated_black_screen",
        },
    )
    if screenshot_ref:
        trace[-1]["trace_screenshot_before"] = screenshot_ref
    return page, ""


def _exit_blocked_page(
    trace: list[dict],
    device,
    trace_phase: str,
) -> Page | None:
    last_executed_action = next(
        (
            str(step.get("action_name", "") or "").strip()
            for step in reversed(trace)
            if not bool(step.get("action_failed"))
            and str(step.get("action_name", "") or "").strip()
            not in {"", "finished", "capture_screenshot", "wait"}
        ),
        "",
    )
    if last_executed_action == "press_back":
        print("[Scenario Exec] Blocking page already exited by executor, skipping duplicate back action.")
        return None
    try:
        print("[Scenario Exec] Current page cannot continue, automatically pressing back once to exit blocking page.")
        _perform_action(device=device, action_name="press_back")
        time.sleep(FAST_ACTION_DELAY)
        page = capture_current_page(device)
    except Exception as exc:
        print(f"[Scenario Exec] Failed to exit blocking page, ending current Function and switching to another Function: {exc}")
        return None
    trace.append({
        "action_name": "press_back",
        "thought": "Runner exited captcha or confirmed uncontinuable blocking page.",
        "action_description": "Exit uncontinuable blocking page",
        "current_page": page,
        "trace_phase": trace_phase,
        "is_interruption": True,
        "blocked_page_exit": True,
        "observation_after_recovery": True,
    })
    return page


def _run_single_scenario(
    llm,
    device,
    app_name: str,
    package_name: str,
    scenario: str,
    initial_page: Page | None,
    clear_logcat: bool,
    trace_phase: str,
    exploration_mode: str = "core",
) -> tuple[list[dict], Page | None, str, str, str, dict[str, Any] | None]:
    phase_label = {"navigation": "Navigation", "test": "Scenario"}.get(
        trace_phase, trace_phase
    )
    step_limit = (
        MAX_NAVIGATION_STEPS if trace_phase == "navigation" else MAX_EXEC_STEPS
    )
    if not scenario:
        print(f"[Scenario Exec] {phase_label} phase skipped: no scenario configured.")
        return [], initial_page, "skipped", "", "", None

    representative_scenario = representative_control_family_scenario(scenario)
    if representative_scenario != scenario:
        print("[Scenario Exec] Detected exhaustive scenario with same-function controls, forcing conversion to a single representative item.")
        scenario = representative_scenario

    risky_scenario_intent = account_destructive_intent(scenario)
    if risky_scenario_intent:
        reason = f"{ACCOUNT_SAFETY_BLOCKED_END}: {risky_scenario_intent}"
        print(f"[Scenario Exec] {reason}")
        return [], initial_page, "blocked_by_safety_policy", reason, "", None

    if clear_logcat:
        _clear_logcat(device)
        print(f"[Scenario Exec] Set scenario logcat start point: phase={trace_phase}")

    scene_started_at = time.time()
    if not _ensure_target_app(device, package_name):
        return [], initial_page, "error", "Unable to launch target app", "", None
    target_pid = _target_process_pid(device, package_name)
    print(
        "[Scenario Exec] Scenario start point: "
        f"phase={trace_phase}, target={package_name or 'unknown'}, "
        f"pid={target_pid or 'unknown'}, started_at={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(scene_started_at))}"
    )

    system_prompt = SCENARIO_EXEC_SYSTEM_PROMPT_TEMPLATE.format(
        app_name=app_name,
        package_name=package_name,
        scenario=scenario,
        max_exec_steps=step_limit,
    )
    system_msg = SystemMessage(content=system_prompt)
    allows_external_app = _scenario_allows_external_app(scenario)
    if allows_external_app:
        print("[Scenario Exec] Current scenario allows controlled external app transitions; target app will be restored when scenario ends.")
    allows_target_exit = _scenario_allows_target_exit(scenario)
    if allows_target_exit:
        print("[Scenario Exec] Current scenario allows target app to leave foreground; target app will be restored when scenario ends.")

    trace: list[dict] = []
    consecutive_errors = 0
    # Number of app restarts triggered by screenshot failures in a single scenario round. Ends scenario directly once upper limit is reached,
    # preventing continuous device failures (such as adb disconnection, uiautomator2 hanging)
    # from running into an infinite "screenshot failure - restart - failure again" loop that occupies the entire execution budget.
    screenshot_restart_count = 0
    # Cumulative model errors in a single round. consecutive_errors is cleared whenever one intermediate call succeeds,
    # failing to prevent alternating "timeout - success - timeout" from accumulating minute-level idle waiting within one round.
    total_llm_errors = 0
    action_failure_recoveries = 0
    status = "completed"
    error_message = ""
    initial_page_id = initial_page.id if isinstance(initial_page, Page) else None
    final_page_override: Page | None = None
    crash_logcat = ""
    external_app_used = False
    recovered_to_target_app = False
    final_recovery_method = ""
    last_execution_context: dict[str, Any] | None = None

    print(f"[Scenario Exec] Starting {phase_label} phase: {scenario}")
    if isinstance(initial_page, Page):
        print(f"[Scenario Exec] Starting page: {initial_page.id or 'unknown'}")

    while True:
        if len(trace) >= step_limit:
            print(f"[Warning] Reached maximum step limit ({step_limit}), forcing termination.")
            status = "max_steps_reached"
            error_message = f"Reached maximum step limit {step_limit}"
            break

        page_health_error = ""
        try:
            foreground_state, current_package = _target_app_foreground_state(device, package_name)
            if foreground_state != "foreground":
                # Launcher may be momentarily reported during page transition animations (app's own transient state),
                # recheck and confirm before characterizing as external transition/crash to avoid terminating scenario mistakenly.
                foreground_state, current_package = _settle_foreground_state(
                    device, package_name
                )
            page_external_context = False
            page_external_package = ""
            if foreground_state != "foreground":
                scene_crash_logcat = _read_crash_logcat(device, package_name)
                if _logcat_has_target_process_crash(scene_crash_logcat, package_name):
                    crash_logcat = scene_crash_logcat
                    if trace:
                        trace[-1]["caused_target_crash"] = True
                    print(
                        "[Scenario Exec] Detected target app crash, ending current scenario: "
                        f"current={current_package or 'unknown'}, target={package_name or 'unknown'}"
                    )
                    restarted = _restart_target_app(device, package_name)
                    if restarted:
                        recovered_to_target_app = True
                        final_recovery_method = "restart_app"
                        _append_target_app_recovery_step(
                            trace,
                            device,
                            package_name,
                            current_package or "unknown",
                            trace_phase,
                            recovery_method="restart_app",
                        )
                        try:
                            final_page_override = capture_current_page(device)
                            save_trace_screenshot(
                                final_page_override,
                                phase=trace_phase,
                                step_index=len(trace) + 1,
                                moment="crash_restart_settled",
                                metadata={
                                    "recovery_method": "restart_app",
                                    "previous_package": current_package or "",
                                },
                            )
                        except Exception as capture_err:
                            print(f"[Scenario Exec] Failed to capture page after crash restart: {capture_err}")
                    status = "target_app_crashed"
                    error_message = (
                        TARGET_APP_CRASHED_END
                        if restarted
                        else TARGET_APP_CRASH_RECOVERY_FAILED_END
                    )
                    break

                controlled_external = (
                    foreground_state == "background"
                    and (
                        allows_external_app
                        or allows_target_exit
                        or _is_safe_external_package(current_package)
                    )
                )
                if controlled_external:
                    external_app_used = True
                    page_external_context = True
                    page_external_package = current_package or ""
                    if trace and not trace[-1].get("external_context"):
                        trace[-1]["caused_external_transition"] = True
                        trace[-1]["external_package"] = page_external_package
                    print(
                        f"[Scenario Exec] Detected controlled external app context, continuing current scenario execution: "
                        f"current={current_package or 'unknown'}, target={package_name or 'unknown'}"
                    )
                else:
                    if foreground_state == "background":
                        external_app_used = True
                        if trace and not trace[-1].get("external_context"):
                            trace[-1]["caused_external_transition"] = True
                            trace[-1]["external_package"] = current_package or ""
                    print(
                        f"[Scenario Exec] Detected uncontrolled external app state during execution: status={foreground_state}, "
                        f"current={current_package or 'unknown'}, target={package_name or 'unknown'}"
                    )
                    _append_external_observation_step(
                        trace,
                        device,
                        package_name,
                        current_package or "unknown",
                        trace_phase,
                    )
                    if _recover_target_app_by_back(device, package_name, continue_message=False):
                        _append_target_app_recovery_step(
                            trace,
                            device,
                            package_name,
                            current_package,
                            trace_phase,
                            recovery_method="press_back",
                        )
                        recovered_to_target_app = True
                        final_recovery_method = "press_back"
                        status = "external_transition_blocked"
                        error_message = EXTERNAL_TRANSITION_BLOCKED_END
                        break
                    print("[Scenario Exec] Back recovery failed, restarting target app and ending this round of scenario.")
                    crash_logcat = _read_crash_logcat(device, package_name)
                    if _restart_target_app(device, package_name):
                        recovered_to_target_app = True
                        final_recovery_method = "restart_app"
                        _append_target_app_recovery_step(
                            trace,
                            device,
                            package_name,
                            current_package or "unknown",
                            trace_phase,
                            recovery_method="restart_app",
                        )
                        try:
                            final_page_override = capture_current_page(device)
                            save_trace_screenshot(
                                final_page_override,
                                phase=trace_phase,
                                step_index=len(trace) + 1,
                                moment="external_restart_settled",
                                metadata={
                                    "recovery_method": "restart_app",
                                    "previous_package": current_package or "",
                                },
                            )
                        except Exception as capture_err:
                            print(f"[Scenario Exec] Failed to capture page after restart: {capture_err}")
                        status = "target_app_restarted"
                        error_message = TARGET_APP_RESTARTED_END
                    else:
                        status = "target_app_recovery_failed"
                        error_message = TARGET_APP_RECOVERY_FAILED_END
                    break
            page, page_health_error = capture_runtime_page(
                device,
                page_id=initial_page_id if not trace else None,
            )
            if page_health_error:
                print(f"[Scenario Exec] Observed abnormal screenshot during runtime: {page_health_error}")
                should_end_for_next_plan = bool(trace)
                if should_end_for_next_plan:
                    _append_unavailable_observation(
                        trace,
                        page,
                        page_health_error,
                        trace_phase,
                    )
                    status = "screenshot_unavailable"
                    error_message = (
                        "Observed black screen or unavailable page, preserved current interface and ended scenario: "
                        f"{page_health_error}"
                    )
                    final_page_override = page
                    break
        except ScreenshotUnavailableError as e:
            print(f"[Scenario Exec] Completely unable to acquire runtime screenshot, stopping current scenario immediately: {e}")
            status = "screenshot_unavailable"
            error_message = f"Unable to acquire runtime screenshot, current scenario stopped: {e}"
            break
        except Exception as e:
            print(f"[Error] Screenshot failed: {e}")
            consecutive_errors += 1
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                if screenshot_restart_count >= APP_RESTART_ATTEMPTS:
                    status = "error"
                    error_message = (
                        f"Consecutive screenshot failures and app restarts reached {APP_RESTART_ATTEMPTS} times, "
                        f"ending current scenario: {e}"
                    )
                    print(f"[Scenario Exec] {error_message}")
                    break
                try:
                    device.restart_app(package_name)
                    time.sleep(APP_RESTART_DELAY)
                    consecutive_errors = 0
                    screenshot_restart_count += 1
                    continue
                except Exception as restart_err:
                    status = "error"
                    error_message = f"Consecutive screenshot failures and restart failed: {restart_err}"
                    break
            time.sleep(2)
            continue

        # Exploration boundary detection (core mode): automatically exit and end current scenario when entering third-party service content
        # (such as mini-programs, embedded web pages, etc.) inside the target app. Rules are loaded from config by package name; apps without
        # config do not enable it, behaving completely identically to when this detection is absent.
        boundary_hit = None
        if exploration_mode == "core":
            boundary_rule = rule_for(package_name)
            if boundary_rule.enabled:
                boundary_hit = check_boundary(package_name, page.activity)
                if boundary_hit is None and boundary_rule.process_markers:
                    try:
                        snapshot = str(device.connector.shell("ps -A") or "")
                        boundary_hit = check_boundary(
                            package_name, activity=None, process_snapshot=snapshot
                        )
                    except Exception as exc:
                        print(f"[Exploration Boundary] Process fallback detection failed (ignored): {exc}")
        if boundary_hit is not None:
            print(
                f"[Exploration Boundary] Detected entry into {boundary_hit.rule.label}"
                f" (via={boundary_hit.via}, matched={boundary_hit.matched}), "
                "automatically exiting and ending current scenario."
            )
            if _recover_from_exploration_boundary(device, package_name, boundary_hit):
                recovered_to_target_app = True
                final_recovery_method = "press_back"
                _append_target_app_recovery_step(
                    trace,
                    device,
                    package_name,
                    package_name,
                    trace_phase,
                    recovery_method="press_back",
                    reason=(
                        f"Entered {boundary_hit.rule.label} crossing exploration boundary; "
                        f" (via={boundary_hit.via}, matched={boundary_hit.matched}); "
                        "exited via back key."
                    ),
                )
                status = "exploration_boundary_blocked"
                error_message = EXPLORATION_BOUNDARY_BLOCKED_END.format(
                    label=boundary_hit.rule.label
                )
                break
            print("[Exploration Boundary] Back exit failed, restarting target app and ending this round of scenario.")
            crash_logcat = _read_crash_logcat(device, package_name)
            if _restart_target_app(device, package_name):
                recovered_to_target_app = True
                final_recovery_method = "restart_app"
                _append_target_app_recovery_step(
                    trace,
                    device,
                    package_name,
                    package_name,
                    trace_phase,
                    recovery_method="restart_app",
                    reason=(
                        f"Entered {boundary_hit.rule.label} crossing exploration boundary and back exit failed; "
                        "restarted target app."
                    ),
                )
                try:
                    final_page_override = capture_current_page(device)
                    save_trace_screenshot(
                        final_page_override,
                        phase=trace_phase,
                        step_index=len(trace) + 1,
                        moment="boundary_restart_settled",
                        metadata={
                            "recovery_method": "restart_app",
                            "boundary_label": boundary_hit.rule.label,
                        },
                    )
                except Exception as capture_err:
                    print(f"[Scenario Exec] Failed to capture page after restart: {capture_err}")
                status = "target_app_restarted"
                error_message = TARGET_APP_RESTARTED_END
            else:
                status = "target_app_recovery_failed"
                error_message = TARGET_APP_RECOVERY_FAILED_END
            break

        try:
            img_w, img_h = device.display_size()
        except Exception as e:
            print(f"[Error] Failed to get screen dimensions: {e}")
            consecutive_errors += 1
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                status = "error"
                error_message = f"Consecutive failures to get screen dimensions: {e}"
                break
            time.sleep(2)
            continue
        b64_image = page.representative_screenshot

        messages = [system_msg]
        step_index = len(trace) + 1
        try:
            foreground_package = str(device.current_package() or "").strip()
        except Exception:
            foreground_package = ""
        previous_step = trace[-1] if trace else {}
        trace_screenshot_before = save_trace_screenshot(
            page,
            phase=trace_phase,
            step_index=step_index,
            moment="before_action",
            metadata={
                "foreground_package": foreground_package,
                "target_package": package_name,
                "observes_result_of_step": len(trace) if trace else None,
                "previous_action_name": str(previous_step.get("action_name", "") or ""),
                "previous_action_description": str(
                    previous_step.get("action_description", "") or ""
                ),
                "screenshot_health_error": page_health_error,
            },
        )
        current_human_msg = HumanMessage(content=_build_step_input(
            trace,
            b64_image,
            step_index,
            step_limit,
            page_health_error,
            foreground_note=_foreground_state_note(
                foreground_package,
                package_name,
            ),
        ))
        messages.append(current_human_msg)
        print(
            "[Scenario Exec] LLM input summary: "
            f"phase={trace_phase}, step={step_index}, history_steps={len(trace)}, "
            f"recent_screenshots={_recent_screenshot_count(trace)}, "
            f"current_page={_page_label(page)}, activity={page.activity or 'unknown'}, "
            f"foreground={foreground_package or 'unknown'}, "
            f"current_screenshot={'yes' if b64_image else 'no'}"
        )

        raw_content = "unknown"
        try:
            record_llm_request(llm, operation="[Scenario Exec]")
            response = llm.invoke(messages)
            record_llm_response(llm, response, operation="[Scenario Exec]")
            raw_content = normalize_llm_text(getattr(response, "content", ""))
            plan = _parse_action_json(raw_content)
            plan = _normalize_action_plan(plan)
            _validate_action_plan(plan)
            last_execution_context = {
                "messages": list(messages),
                "assistant_content": raw_content,
                "trace_phase": trace_phase,
                "step_index": step_index,
                "retained_trace_step_indices": [
                    *range(max(1, len(trace) - KEEP_IMAGE_STEPS + 1), len(trace) + 1),
                    step_index,
                ],
            }
            log_plan = dict(plan)
            log_plan.pop("action_description", None)
            print(f"[Scenario Exec] LLM output JSON: {log_plan}")
        except Exception as e:
            record_llm_failure(llm, operation="[Scenario Exec]")
            print(f"[Error] LLM call or parsing failed: {e}\n[Raw Content]: {raw_content}")
            consecutive_errors += 1
            total_llm_errors += 1
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                status = "error"
                error_message = f"Consecutive LLM parsing failures: {e}"
                break
            if total_llm_errors >= MAX_TOTAL_LLM_ERRORS:
                status = "error"
                error_message = (
                    f"Cumulative model errors reached {total_llm_errors} in this round, ending scenario early: {e}"
                )
                print(f"[Scenario Exec] {error_message}")
                break
            _sleep_for_llm_retry(consecutive_errors)
            continue

        action_name = plan.get("action_name", "")
        thought = plan.get("thought", "")
        action_description = str(plan.get("action_description", "")).strip()
        is_interruption = bool(plan.get("is_interruption", False))

        risky_action_intent = ""
        proposed_action = _proposed_action_text(plan)
        whitelist_violation = account_action_policy_violation(proposed_action)
        if whitelist_violation:
            risky_action_intent = whitelist_violation
        if risky_action_intent:
            status = "blocked_by_safety_policy"
            error_message = f"{ACCOUNT_SAFETY_BLOCKED_END}: {risky_action_intent}"
            print(f"[Scenario Exec] Intercepted before action execution: {error_message}")
            break

        print(
            f"\n[Step {step_index}]{' [Interruption]' if is_interruption else ''} "
            f"Page={page.id or 'unknown'} Action={action_name} | Thought: {thought}"
        )

        try:
            _scale_plan_coordinates(plan, img_w, img_h)
        except ValueError as e:
            print(f"[Error] LLM coordinate parsing failed: {e}\n[Raw Content]: {raw_content}")
            consecutive_errors += 1
            total_llm_errors += 1
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                status = "error"
                error_message = f"Consecutive LLM coordinate parsing failures: {e}"
                break
            if total_llm_errors >= MAX_TOTAL_LLM_ERRORS:
                status = "error"
                error_message = (
                    f"Cumulative model errors reached {total_llm_errors} in this round, ending scenario early: {e}"
                )
                print(f"[Scenario Exec] {error_message}")
                break
            _sleep_for_llm_retry(consecutive_errors)
            continue

        consecutive_errors = 0
        step_info = {
            "action_name": action_name,
            "thought": thought,
            "action_description": action_description,
            "current_page": page,
            "is_interruption": is_interruption,
            "trace_phase": trace_phase,
        }
        if trace_screenshot_before:
            step_info["trace_screenshot_before"] = trace_screenshot_before
        if page_health_error:
            step_info["screenshot_health_error"] = page_health_error
        if page_external_context:
            step_info["external_context"] = True
            step_info["external_package"] = page_external_package
        for field in (*COORDINATE_FIELDS, "content", "direction", "app_name"):
            if plan.get(field) is not None:
                step_info[field] = plan[field]
        trace.append(step_info)

        verification_attempt_reason = _verification_attempt_reason(
            action_name,
            thought,
            proposed_action,
        )
        if verification_attempt_reason:
            step_info["action_failed"] = True
            step_info["action_error"] = verification_attempt_reason
            status = "blocked_by_verification"
            error_message = verification_attempt_reason
            print(f"[Scenario Exec] Intercepted before action execution: {verification_attempt_reason}")
            break

        stall_reason = _detect_stall(trace)
        if stall_reason:
            print(f"[Warning] {stall_reason}, ending current scenario early.")
            step_info["action_failed"] = True
            step_info["action_error"] = stall_reason
            status = "stalled"
            error_message = stall_reason
            break

        if action_name == "finished":
            if page_health_error:
                status = "screenshot_unavailable"
                error_message = "Executor ended on black screen or unavailable interface, waiting for next round recovery"
                step_info["action_failed"] = True
                step_info["action_error"] = error_message
                final_page_override = page
                break
            verification_finish_reason = _verification_finish_reason(thought)
            if verification_finish_reason:
                status = "blocked_by_verification"
                error_message = verification_finish_reason
                step_info["action_failed"] = True
                step_info["action_error"] = error_message
                print(f"[Scenario Exec] Verification challenge blocked current Function: {thought[:160]}")
            elif _finished_declares_failure(thought):
                status = "failed_goal"
                error_message = f"Executor determined current goal is unreachable: {thought[:240]}"
                step_info["action_failed"] = True
                step_info["action_error"] = error_message
                print(f"[Scenario Exec] finished contains failure semantics, demoting to failed_goal: {thought[:160]}")
            break

        if action_name != "wait":
            action_kwargs = {
                k: plan[k]
                for k in (*COORDINATE_FIELDS, "content", "direction", "app_name")
                if plan.get(k) is not None
            }
            if action_name == "open_app" and package_name:
                action_kwargs["package_name"] = package_name
            if action_kwargs:
                print(f"[Scenario Exec] Action parameters: {action_kwargs}")
            try:
                _perform_action(device=device, action_name=action_name, **action_kwargs)
            except Exception as e:
                print(f"[Error] Action execution failed: {e}")
                step_info["is_interruption"] = True
                step_info["action_failed"] = True
                step_info["action_error"] = str(e)
                if action_failure_recoveries < MAX_ACTION_FAILURE_RECOVERIES:
                    action_failure_recoveries += 1
                    print(
                        "[Scenario Exec] Passing action failure and reason to execution model, allowing one alternative action to continue current scenario."
                    )
                    continue
                status = "action_failed"
                error_message = f"Action execution failed and alternative strategy also failed; ended current scenario: {e}"
                break

        if action_name == "wait":
            time.sleep(ACTION_DELAY)
        elif action_name == "open_app" and package_name:
            # Cold start polling replacing fixed delay: avoids capturing screenshots while app is still starting,
            # preventing execution model from seeing launcher/splash screen and erroneously repeating open_app.
            _wait_target_app_foreground(
                device,
                package_name,
                max_wait=min(APP_START_MAX_WAIT, 8.0),
            )
            time.sleep(APP_RESTART_SETTLE_DELAY)
        elif action_name in FAST_ACTIONS:
            time.sleep(FAST_ACTION_DELAY)
        else:
            time.sleep(ACTION_DELAY)

    if status in {"blocked_by_verification", "stalled"}:
        blocked_exit_page = _exit_blocked_page(trace, device, trace_phase)
        if blocked_exit_page is not None:
            final_page_override = blocked_exit_page

    if status not in {
        "screenshot_unavailable",
        "target_app_restarted",
        "target_app_crashed",
        "target_app_recovery_failed",
        "external_transition_blocked",
    }:
        terminal_page, terminal_health_error = _append_terminal_observation(
            trace,
            device,
            trace_phase,
        )
        if terminal_page is not None:
            final_page_override = terminal_page
        if terminal_health_error:
            status = "screenshot_unavailable"
            error_message = (
                "Page unavailable after the last side-effect action, ended current scenario: "
                f"{terminal_health_error}"
            )

    if package_name and status not in {
        "screenshot_unavailable",
        "target_app_restarted",
        "target_app_crashed",
        "target_app_recovery_failed",
    }:
        did_recover, recovery_method, recovery_crash_logcat = _recover_target_app_at_scene_end(
            trace,
            device,
            package_name,
            trace_phase,
        )
        if did_recover:
            recovered_to_target_app = True
            final_recovery_method = recovery_method
        if recovery_crash_logcat and not crash_logcat:
            crash_logcat = recovery_crash_logcat
        scene_end_crash = (
            recovery_crash_logcat
            and _logcat_has_target_process_crash(recovery_crash_logcat, package_name)
        )
        if scene_end_crash:
            status = "target_app_crashed"
            error_message = (
                TARGET_APP_CRASHED_END
                if did_recover
                else TARGET_APP_CRASH_RECOVERY_FAILED_END
            )
        elif not did_recover and not _is_target_app_foreground(device, package_name):
            status = "target_app_recovery_failed"
            error_message = TARGET_APP_RECOVERY_FAILED_END

    for step in trace:
        if external_app_used:
            step.setdefault("external_app_used", True)
        if recovered_to_target_app:
            step.setdefault("recovered_to_target_app", True)
        if final_recovery_method:
            step.setdefault("final_recovery_method", final_recovery_method)

    supplemental_screenshots = _persist_unwritten_trace_screenshots(trace, trace_phase)
    print(
        "[Trace Screenshot] Scenario screenshot indexing completed: "
        f"phase={trace_phase}, trace_steps={len(trace)}, "
        f"supplemental_saved={supplemental_screenshots}"
    )

    final_page = final_page_override or (trace[-1]["current_page"] if trace else initial_page)
    interruption_count = sum(1 for s in trace if s.get("is_interruption"))
    final_page_id = (
        str(final_page.id or "unknown")
        if isinstance(final_page, Page)
        else "unknown"
    )
    reason = f" | reason={_compact_execution_log_text(error_message)}" if error_message else ""
    print(
        f"[Scenario Exec] {phase_label} phase completed | status={status} | "
        f"steps={len(trace)} | interruptions={interruption_count} | "
        f"page={final_page_id}{reason}"
    )
    return trace, final_page, status, error_message, crash_logcat, last_execution_context


def create_scenario_execution_pass(execution_llm, device):
    """Create the navigation + test execution pass used by the exec subgraph."""

    def execute_scenarios(state: AgentState) -> tuple[dict, dict[str, Any] | None]:
        app_name = state.get("app_name", "")
        package_name = state.get("package_name", "")
        exploration_mode = state.get("exploration_mode", "core")
        navigation_scenario = state.get("current_navigation_scenario", "")
        test_scenario = state.get("current_test_scenario", "")
        current_page = state.get("current_page")
        navigation_failure_count = int(state.get("navigation_failure_count", 0) or 0)
        if package_name and not _ensure_target_app(device, package_name):
            return {
                "navigation_execution_trace": [],
                "scenario_execution_trace": [],
                "navigation_failure_count": navigation_failure_count + 1,
                "scenario_exec_error_count": 0,
                "execution_result": {
                    "navigation_status": "error",
                    "navigation_error": "Unable to launch target app",
                    "navigation_trace_length": 0,
                    "scenario_status": "skipped_due_target_app_error",
                    "scenario_error": "",
                    "scenario_trace_length": 0,
                    "current_scenario": "",
                    "scenario_type": state.get("current_plan", {}).get("scenario_type", ""),
                    "navigation_failure_count": navigation_failure_count + 1,
                },
            }, None

        (
            navigation_trace,
            nav_final_page,
            navigation_status,
            navigation_error,
            navigation_crash_logcat,
            navigation_execution_context,
        ) = _run_single_scenario(
            llm=execution_llm,
            device=device,
            app_name=app_name,
            package_name=package_name,
            scenario=navigation_scenario,
            initial_page=current_page if isinstance(current_page, Page) else None,
            clear_logcat=True,
            trace_phase="navigation",
            exploration_mode=exploration_mode,
        )
        if navigation_status == "screenshot_unavailable":
            unavailable_count = _screenshot_unavailable_count(state) + 1
            final_page = nav_final_page or current_page
            restart_error = ""
            if unavailable_count >= MAX_CONSECUTIVE_SCREENSHOT_UNAVAILABLE:
                reset_page, restart_error = _restart_after_repeated_black_screen(
                    navigation_trace,
                    device,
                    package_name,
                    "navigation",
                )
                if reset_page is not None:
                    final_page = reset_page
                    unavailable_count = 0
            workflow_complete = (
                STOP_AFTER_SCREENSHOT_UNAVAILABLE
                and bool(restart_error)
                and unavailable_count >= MAX_CONSECUTIVE_SCREENSHOT_UNAVAILABLE
            )
            if workflow_complete:
                print(
                    "[Scenario Exec] Navigation phase screenshots continuously unavailable, ending current app run: "
                    f"{unavailable_count}/{MAX_CONSECUTIVE_SCREENSHOT_UNAVAILABLE}"
                )
            else:
                print(
                    "[Scenario Exec] Navigation phase screenshots unavailable, current scenario ended; exploring other Functions next: "
                    f"{unavailable_count}/{MAX_CONSECUTIVE_SCREENSHOT_UNAVAILABLE}"
                )
            return {
                "current_page": final_page,
                "current_test_scenario": "",
                "navigation_execution_trace": navigation_trace,
                "scenario_execution_trace": [],
                "navigation_failure_count": 0,
                "scenario_exec_error_count": 0,
                "workflow_complete": workflow_complete,
                "screenshot_unavailable_count": unavailable_count,
                "execution_result": {
                    "navigation_status": navigation_status,
                    "navigation_error": restart_error or navigation_error,
                    "navigation_trace_length": len(navigation_trace),
                    "scenario_status": "skipped_due_screenshot_unavailable",
                    "scenario_error": restart_error or navigation_error,
                    "scenario_trace_length": 0,
                    "current_scenario": navigation_scenario,
                    "scenario_type": state.get("current_plan", {}).get("scenario_type", ""),
                    "crash_logcat": navigation_crash_logcat,
                    "screenshot_unavailable_count": unavailable_count,
                    "workflow_complete_after_screenshot_unavailable": workflow_complete,
                    "external_app_used": _trace_external_app_used(navigation_trace),
                    "recovered_to_target_app": _trace_recovered_to_target_app(navigation_trace),
                    "final_recovery_method": _trace_final_recovery_method(navigation_trace),
                },
            }, navigation_execution_context

        if navigation_status in {
            "target_app_restarted",
            "target_app_crashed",
            "target_app_recovery_failed",
        } or navigation_error in {
            TARGET_APP_RESTARTED_END,
            TARGET_APP_CRASHED_END,
            TARGET_APP_CRASH_RECOVERY_FAILED_END,
            TARGET_APP_RECOVERY_FAILED_END,
        }:
            final_page = nav_final_page or current_page
            is_crash = navigation_status == "target_app_crashed" or navigation_error == TARGET_APP_CRASHED_END
            recovery_failed = (
                navigation_status == "target_app_recovery_failed"
                or navigation_error == TARGET_APP_RECOVERY_FAILED_END
            )
            if is_crash:
                end_message = navigation_error or TARGET_APP_CRASHED_END
                skipped_status = "skipped_due_target_app_crashed"
            elif recovery_failed:
                end_message = TARGET_APP_RECOVERY_FAILED_END
                skipped_status = "skipped_due_target_app_recovery_failed"
            else:
                end_message = TARGET_APP_RESTARTED_END
                skipped_status = "skipped_due_target_app_restarted"
            print("[Scenario Exec] Left target app during navigation phase, will not continue executing test scenario in this round.")
            return {
                "current_page": final_page,
                "current_test_scenario": "",
                "navigation_execution_trace": navigation_trace,
                "scenario_execution_trace": [],
                "navigation_failure_count": 0,
                "scenario_exec_error_count": 0,
                "execution_result": {
                    "navigation_status": navigation_status,
                    "navigation_error": navigation_error,
                    "navigation_trace_length": len(navigation_trace),
                    "scenario_status": skipped_status,
                    "scenario_error": end_message,
                    "scenario_trace_length": 0,
                    "current_scenario": navigation_scenario,
                    "scenario_type": state.get("current_plan", {}).get("scenario_type", ""),
                    "crash_logcat": navigation_crash_logcat,
                    "crash_detected": is_crash,
                    "external_app_used": _trace_external_app_used(navigation_trace),
                    "recovered_to_target_app": _trace_recovered_to_target_app(navigation_trace),
                    "final_recovery_method": _trace_final_recovery_method(navigation_trace),
                },
            }, navigation_execution_context

        if navigation_scenario and navigation_status != "completed":
            navigation_failure_count += 1
            observed_navigation_failure_count = navigation_failure_count
            navigation_restarted_after_failures = False
            print(f"[Scenario Exec] Navigation failed, skipping scenario execution for this round. Consecutive failures: {navigation_failure_count}")
            reset_page = nav_final_page or current_page
            if navigation_failure_count >= 2:
                if package_name:
                    navigation_restarted_after_failures = True
                    print("[Scenario Exec] Navigation failed twice consecutively, restarting app to return to initial page.")
                    if _restart_target_app(device, package_name):
                        try:
                            reset_page = capture_current_page(device)
                            save_trace_screenshot(
                                reset_page,
                                phase="navigation",
                                step_index=len(navigation_trace) + 1,
                                moment="navigation_failure_restart_settled",
                                metadata={
                                    "recovery_method": "restart_app",
                                    "navigation_failure_count": observed_navigation_failure_count,
                                },
                            )
                        except Exception as e:
                            print(f"[Scenario Exec] Failed to capture page after restart: {e}")
                            reset_page = nav_final_page or current_page
                    else:
                        print("[Scenario Exec] Still unable to recover target app after consecutive navigation failures.")
                else:
                    print("[Scenario Exec] Navigation failed twice consecutively, but target package name is empty; resetting failure count.")
                navigation_failure_count = 0

            return {
                "current_page": reset_page,
                "current_test_scenario": "",
                "navigation_execution_trace": navigation_trace,
                "scenario_execution_trace": [],
                "navigation_failure_count": navigation_failure_count,
                "scenario_exec_error_count": 0,
                "execution_result": {
                    "navigation_status": navigation_status,
                    "navigation_error": navigation_error,
                    "navigation_trace_length": len(navigation_trace),
                    "scenario_status": "skipped_due_navigation_failure",
                    "scenario_error": "",
                    "scenario_trace_length": 0,
                    "current_scenario": "",
                    "scenario_type": state.get("current_plan", {}).get("scenario_type", ""),
                    "navigation_failure_count": observed_navigation_failure_count,
                    "navigation_restarted_after_failures": navigation_restarted_after_failures,
                    "crash_logcat": navigation_crash_logcat,
                    "external_app_used": _trace_external_app_used(navigation_trace),
                    "recovered_to_target_app": _trace_recovered_to_target_app(navigation_trace),
                    "final_recovery_method": _trace_final_recovery_method(navigation_trace),
                },
            }, navigation_execution_context

        test_initial_page = nav_final_page if isinstance(nav_final_page, Page) else current_page
        (
            test_trace,
            test_final_page,
            scenario_status,
            scenario_error,
            test_crash_logcat,
            test_execution_context,
        ) = _run_single_scenario(
            llm=execution_llm,
            device=device,
            app_name=app_name,
            package_name=package_name,
            scenario=test_scenario,
            initial_page=test_initial_page if isinstance(test_initial_page, Page) else None,
            clear_logcat=True,
            trace_phase="test",
            exploration_mode=exploration_mode,
        )
        final_page = test_final_page or nav_final_page or current_page
        unavailable_count = _screenshot_unavailable_count(state)
        workflow_complete = False
        if scenario_status == "screenshot_unavailable":
            unavailable_count += 1
            if unavailable_count >= MAX_CONSECUTIVE_SCREENSHOT_UNAVAILABLE:
                reset_page, restart_error = _restart_after_repeated_black_screen(
                    test_trace,
                    device,
                    package_name,
                    "test",
                )
                if reset_page is not None:
                    final_page = reset_page
                    unavailable_count = 0
                elif restart_error:
                    scenario_error = restart_error or scenario_error
            workflow_complete = (
                STOP_AFTER_SCREENSHOT_UNAVAILABLE
                and unavailable_count >= MAX_CONSECUTIVE_SCREENSHOT_UNAVAILABLE
            )
            if workflow_complete:
                print(
                    "[Scenario Exec] Scenario phase screenshots continuously unavailable, ending current app run: "
                    f"{unavailable_count}/{MAX_CONSECUTIVE_SCREENSHOT_UNAVAILABLE}"
                )
            else:
                print(
                    "[Scenario Exec] Scenario phase screenshots unavailable, current scenario ended; exploring other Functions next: "
                    f"{unavailable_count}/{MAX_CONSECUTIVE_SCREENSHOT_UNAVAILABLE}"
                )
        if scenario_status != "screenshot_unavailable":
            unavailable_count = 0
        return {
            "current_page": final_page,
            "navigation_execution_trace": navigation_trace,
            "scenario_execution_trace": test_trace,
            "navigation_failure_count": 0,
            "scenario_exec_error_count": 0,
            "workflow_complete": workflow_complete,
            "screenshot_unavailable_count": unavailable_count,
            "execution_result": {
                "navigation_status": navigation_status,
                "navigation_error": navigation_error,
                "navigation_trace_length": len(navigation_trace),
                "scenario_status": scenario_status,
                "scenario_error": scenario_error,
                "scenario_trace_length": len(test_trace),
                "current_scenario": test_scenario,
                "scenario_type": state.get("current_plan", {}).get("scenario_type", ""),
                "crash_logcat": test_crash_logcat,
                "crash_detected": scenario_status == "target_app_crashed",
                "screenshot_unavailable_count": unavailable_count,
                "workflow_complete_after_screenshot_unavailable": workflow_complete,
                "external_app_used": _trace_external_app_used(navigation_trace + test_trace),
                "recovered_to_target_app": _trace_recovered_to_target_app(navigation_trace + test_trace),
                "final_recovery_method": _trace_final_recovery_method(navigation_trace + test_trace),
            },
        }, test_execution_context or navigation_execution_context

    return execute_scenarios
