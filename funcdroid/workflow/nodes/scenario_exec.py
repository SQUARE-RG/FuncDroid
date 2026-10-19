"""Scenario-driven GUI execution."""

import json
import math
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

import cv2
from langchain_core.messages import HumanMessage, SystemMessage

from model.flow_graph import Page
from tools.utils import decode_screenshot, encode_image_to_jpg_base64, screenshot_dimensions
from tools.actions import (
    ScreenshotUnavailableError,
    _perform_action,
    capture_current_page,
    capture_runtime_page,
)
from workflow.logcat_crash import NON_CRASH_LOGCAT_PATTERNS, _logcat_has_target_process_crash
from workflow.app_foreground import (
    APP_RESTART_ATTEMPTS,
    APP_RESTART_DELAY,  # noqa: F401 - retained for callers and tests
    APP_RESTART_MAX_SECONDS,
    APP_RESTART_RETRY_DELAY,  # noqa: F401 - retained for callers and tests
    APP_RESTART_SETTLE_DELAY,
    APP_START_MAX_WAIT,
    APP_START_POLL_INTERVAL,  # noqa: F401 - retained for callers and tests
    FOREGROUND_CHECK_RETRIES,
    FOREGROUND_CHECK_RETRY_DELAY,  # noqa: F401 - retained for callers and tests
    current_foreground_package,
    deadline_allows,
    restart_until_foreground,
    wait_target_app_foreground,
)
from workflow.exploration_boundary import BoundaryHit, check_boundary, rule_for
from workflow.account_safety import account_session_action_violation
from workflow.llm import (
    _repair_common_json_typos,
    _parse_json_response,
    compute_retry_delay_seconds,
    disable_json_response_format,
    json_format_rejected,
)
from workflow.llm_usage import record_llm_failure, record_llm_request, record_llm_response, record_llm_diagnostic
from workflow.nodes.prompt import (
    COMPLIANCE_EXEC_ADDENDUM,
    SCENARIO_EXEC_SYSTEM_PROMPT_TEMPLATE,
)
from workflow.nodes._shared import (
    JSON_BLOCK_PATTERN,
    BoundedCache,
    normalize_llm_text,
    screenshot_change_region,
    screenshot_digest,
    screenshot_similarity,
)
from workflow.state import AgentState
from workflow.system_dialogs import is_permission_dialog
from workflow.trace_artifacts import save_trace_screenshot
from workflow.wechat_safety import (
    wechat_action_policy_violation,
    wechat_read_only_violation,
)

def _compact_execution_log_text(value: object, limit: int = 200) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


# Historical screenshots per decision; one gives a before/after pair, zero omits history explicitly.


KEEP_IMAGE_STEPS = max(0, int(os.getenv("EXEC_KEEP_IMAGE_STEPS", "1")))
# Maximum recent generation failures retained in model-visible history.
MAX_RECENT_FAILURES = 3
# Test actions per scenario; navigation has a separate MAX_NAVIGATION_STEPS budget.
MAX_EXEC_STEPS = max(1, int(os.getenv("MAX_EXEC_STEPS", "15")))
MAX_NAVIGATION_STEPS = max(1, int(os.getenv("MAX_NAVIGATION_STEPS", "10")))
# Allow four consecutive generation failures: some endpoints return prose despite JSON mode, then recover after format feedback.


MAX_CONSECUTIVE_ERRORS = max(1, int(os.getenv("EXEC_MAX_CONSECUTIVE_ERRORS", "4")))
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
SWIPE_COORDINATE_FIELDS = ("x1", "y1", "x2", "y2")
# An explicit default wait covers loading without forcing repeated wait actions.

MIN_WAIT_SECONDS = max(0.5, float(os.getenv("EXEC_WAIT_MIN_SECONDS", "1")))
MAX_WAIT_SECONDS = max(MIN_WAIT_SECONDS, float(os.getenv("EXEC_WAIT_MAX_SECONDS", "10")))
DEFAULT_WAIT_SECONDS = min(
    MAX_WAIT_SECONDS,
    max(MIN_WAIT_SECONDS, float(os.getenv("EXEC_WAIT_DEFAULT_SECONDS", "3"))),
)
# Swipe duration in seconds: shorter flicks versus controlled drawer or slider drags.
MIN_SWIPE_DURATION = max(0.05, float(os.getenv("EXEC_SWIPE_MIN_DURATION", "0.1")))
MAX_SWIPE_DURATION = max(MIN_SWIPE_DURATION, float(os.getenv("EXEC_SWIPE_MAX_DURATION", "1.5")))
DEFAULT_SWIPE_DURATION = min(
    MAX_SWIPE_DURATION,
    max(MIN_SWIPE_DURATION, float(os.getenv("EXEC_SWIPE_DEFAULT_DURATION", "0.5"))),
)
# Scroll and swipe share budgets and stall checks so a new gesture cannot bypass them.

SCROLL_LIKE_ACTIONS = {"scroll", "swipe"}
# Action parameter fields persisted in traces and forwarded to the device.
ACTION_PARAMETER_FIELDS = (
    *COORDINATE_FIELDS,
    *SWIPE_COORDINATE_FIELDS,
    "content",
    "direction",
    "app_name",
    "duration",
)
# Unchanged repeat limits: open_app can restart, while repeated list scrolling may be useful.

STALL_REPEAT_WINDOW = 3
WAIT_STALL_REPEAT_WINDOW = max(
    2,
    min(int(os.getenv("EXEC_WAIT_STALL_REPEAT_WINDOW", "2")), MAX_EXEC_STEPS),
)
OPEN_APP_STALL_REPEAT_WINDOW = 2
SCROLL_STALL_REPEAT_WINDOW = 5
STALL_REPEAT_WINDOWS = {
    "wait": WAIT_STALL_REPEAT_WINDOW,
    "open_app": OPEN_APP_STALL_REPEAT_WINDOW,
    "scroll": SCROLL_STALL_REPEAT_WINDOW,
    "swipe": SCROLL_STALL_REPEAT_WINDOW,
}
# Hard same-direction scroll budget in the test phase; further attempts are blocked by stall detection.

TEST_MAX_CONSECUTIVE_SAME_DIRECTION_SCROLLS = max(
    1,
    min(
        int(os.getenv("EXEC_TEST_MAX_CONSECUTIVE_SCROLLS", "3")),
        MAX_EXEC_STEPS,
    ),
)
PAGE_NAVIGATION_CYCLE_WINDOW = 6
# Pixel-change ratios distinguish noise, local changes and page changes; compute only recent effects.

EFFECT_NO_CHANGE_RATIO = float(os.getenv("EXEC_EFFECT_NO_CHANGE_RATIO", "0.0002"))
# Coordinate tolerance here uses uploaded-image pixels, not device-screen pixels.
EFFECT_SAME_POINT_TOLERANCE = int(os.getenv("EXEC_EFFECT_SAME_POINT_TOLERANCE", "50"))
EFFECT_MAJOR_CHANGE_RATIO = float(os.getenv("EXEC_EFFECT_MAJOR_CHANGE_RATIO", "0.15"))
EFFECT_HISTORY_STEPS = max(1, int(os.getenv("EXEC_EFFECT_HISTORY_STEPS", "6")))
EFFECT_ACTIONS = {"click", "long_press", "type", "press_back", "scroll", "swipe"}
REPEAT_COORD_TOLERANCE = int(os.getenv("EXEC_REPEAT_COORD_TOLERANCE", "80"))
# Refine taps only on low-texture backgrounds by magnifying the local region; avoid an extra model call for ordinary clicks.


TAP_REFINE_ENABLED = os.getenv("EXEC_TAP_REFINE", "1").strip().lower() not in {"0", "false", "no", "off"}
TAP_REFINE_FLAT_STD = float(os.getenv("EXEC_TAP_REFINE_FLAT_STD", "6"))
TAP_REFINE_RADIUS = max(60, int(os.getenv("EXEC_TAP_REFINE_RADIUS", "180")))
TAP_REFINE_MAX_SHIFT = max(10, int(os.getenv("EXEC_TAP_REFINE_MAX_SHIFT", "120")))
TAP_REFINE_TIMEOUT_SECONDS = max(1.0, float(os.getenv("EXEC_TAP_REFINE_TIMEOUT", "8")))
STALL_SCREENSHOT_SIMILARITY_THRESHOLD = float(os.getenv("EXEC_STALL_SCREENSHOT_SIMILARITY", "0.9995"))
PAGE_CYCLE_ACTIONS = {"click", "press_back", "wait"}
# Stop direction-independent probing among a few visual states before the action budget is exhausted.

NO_PROGRESS_STALL_WINDOW = max(
    3,
    min(int(os.getenv("EXEC_NO_PROGRESS_STALL_WINDOW", "6")), MAX_EXEC_STEPS),
)
NO_PROGRESS_MAX_VISUAL_STATES = max(
    1,
    int(os.getenv("EXEC_NO_PROGRESS_MAX_VISUAL_STATES", "2")),
)
# Visual-state clustering uses a looser threshold than identical-action detection to measure substantive change.

NO_PROGRESS_SIMILARITY_THRESHOLD = float(
    os.getenv("EXEC_NO_PROGRESS_SIMILARITY", "0.985")
)
NO_PROGRESS_STALL_ACTIONS = {"click", "long_press", "scroll", "swipe", "press_back", "wait"}
APP_RECOVERY_BACK_ATTEMPTS = max(1, int(os.getenv("APP_RECOVERY_BACK_ATTEMPTS", "3")))
# Foreground/restart constants are shared with workflow.app_foreground; retain imported names for callers and tests. Launcher recovery skips ineffective back presses.


LAUNCHER_PACKAGE_PATTERN = re.compile(r"launcher|home", re.IGNORECASE)
# Statuses: completed; failed_goal; blocked; stalled; incomplete; left_app; app_crashed; screenshot_unavailable; error; skipped. Each distinguishes task, environment and infrastructure outcomes.


EXECUTION_STATUSES = frozenset({
    "completed",
    "failed_goal",
    "blocked",
    "stalled",
    "incomplete",
    "left_app",
    "app_crashed",
    "screenshot_unavailable",
    "error",
    "skipped",
})
# These states already restored foreground or lost observations; avoid another final observation/recovery.
FOREGROUND_HANDLED_STATUSES = frozenset({"left_app", "app_crashed", "screenshot_unavailable"})
TARGET_APP_CRASHED_END = "Target app crashed; restarted it and ended the current scenario"
TARGET_APP_CRASH_RECOVERY_FAILED_END = "Target app crashed and remained unavailable after restart; ended the current scenario"
TARGET_APP_RECOVERY_FAILED_END = "Target app left foreground; back and restart failed to restore it; ended the scenario"
EXTERNAL_TRANSITION_BLOCKED_END = "Target app entered an uncontrolled external app; restored the target and ended the scenario"
# Boundary message template; label comes from the configured rule.
EXPLORATION_BOUNDARY_BLOCKED_END = "Entered {label} (outside target app core functionality); returned automatically and ended the scenario"
WECHAT_READ_ONLY_BLOCKED_END = "WeChat account safety blocked logout or account deletion"
VERIFICATION_BLOCKED_END = "Human verification cannot be completed unattended; exited the blocked page"
# Recheck non-foreground reports after a short delay because transitions may briefly report the launcher.


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
# Legacy multilingual verdicts and UI vocabulary remain supported for saved traces.
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
    "not completed",
    "not verified",
    "not executed",
    "not able",
    "unable",
    "cannot",
    "can't",
    "unreachable",
    "not found",
    "failed to",
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
)
MIXED_FINISH_FAILURE_PATTERN = re.compile(
    r"(?:\u76ee\u6807\u5df2\u8fbe\u6210|\u4efb\u52a1\u5df2\u5b8c\u6210|\u5df2\u5b8c\u6210\u53ef\u6267\u884c\u90e8\u5206).{0,120}"
    r"(?:\u4e0d\u53ef\u8fbe|\u672a\u5b8c\u6210|\u672a\u9a8c\u8bc1|\u672a\u627e\u5230|\u672a\u6267\u884c|\u65e0\u6cd5\u8fdb\u5165)"
    r"|(?:\u4e0d\u53ef\u8fbe|\u672a\u5b8c\u6210|\u672a\u9a8c\u8bc1|\u672a\u627e\u5230|\u672a\u6267\u884c|\u65e0\u6cd5\u8fdb\u5165).{0,120}"
    r"(?:\u76ee\u6807\u5df2\u8fbe\u6210|\u4efb\u52a1\u5df2\u5b8c\u6210|\u5df2\u5b8c\u6210\u53ef\u6267\u884c\u90e8\u5206)"
    r"|(?:goal achieved|task completed).{0,120}(?:unreachable|not completed|not verified|not executed|not found|unable|cannot)"
    r"|(?:unreachable|not completed|not verified|not executed|not found|unable|cannot).{0,120}(?:goal achieved|task completed)",
    re.IGNORECASE,
)
# A generic slider may be an ordinary control; only explicit CAPTCHA/verification phrases denote a challenge.

VERIFICATION_CHALLENGE_PATTERN = re.compile(
    r"\u9a8c\u8bc1\u7801|\u56fe\u5f62\u9a8c\u8bc1|\u4eba\u673a\u9a8c\u8bc1|\u5b89\u5168\u9a8c\u8bc1|\u62fc\u56fe|captcha|verification\s+challenge"
    r"|(?:\u6ed1\u5757|\u6ed1\u52a8)\u9a8c\u8bc1(?=\u7801|[，。、；,.;!！?？\s]|$)"
    r"|(?:\u62d6\u52a8|\u6ed1\u52a8)\u6ed1\u5757(?:\u4ee5|\u6765)?(?:\u5b8c\u6210|\u901a\u8fc7)\u9a8c\u8bc1",
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
EXTERNAL_APP_SCENARIO_PATTERNS = (
    r"\b(open|enter|visit)\b.{0,24}\b(system settings|permission settings|camera|album|browser)\b",
    r"\b(select|import|save|create|open)\b.{0,30}\b(file|document|directory)\b",
    r"(\u70b9\u51fb|\u9009\u62e9).{0,20}(\u5e94\u7528\u9879|\u5e94\u7528\u5217\u8868\u9879|app item)",
    # The target app itself does not authorize an external transition.
    r"(\u542f\u52a8|\u6253\u5f00|\u8df3\u8f6c\u5230|\u8fdb\u5165).{0,30}(\u5916\u90e8\u5e94\u7528|\u5176\u4ed6\u5e94\u7528|\u53e6\u4e00\u4e2a\u5e94\u7528|\u5bf9\u5e94\u5e94\u7528|\u5e94\u7528\u7a0b\u5e8f)",
    r"(\u542f\u52a8|\u6253\u5f00).{0,30}(Calculator|Calendar|Camera|Clock|Contacts|Files|Gmail|Maps|Messages|Phone|Photos)",
    r"(\u8fd4\u56de|\u56de\u5230).{0,30}(launcher|\u542f\u52a8\u5668|\u5e94\u7528\u5217\u8868|TextLauncher)",
    r"(\u7cfb\u7edf)?(\u6587\u4ef6|\u6587\u6863).{0,20}(\u9009\u62e9\u5668|\u7ba1\u7406\u5668|\u4fdd\u5b58|\u6253\u5f00|\u5bfc\u5165|\u521b\u5efa)",
    r"(\u6253\u5f00|\u8fdb\u5165|\u524d\u5f80).{0,24}(\u7cfb\u7edf\u8bbe\u7f6e|\u7cfb\u7edf\u6743\u9650\u9875|\u6743\u9650\u7ba1\u7406\u9875|\u76f8\u673a|\u76f8\u518c)",
    r"(?:\u5728|\u4f7f\u7528|\u901a\u8fc7)?\s*(?:\u7cfb\u7edf|\u5916\u90e8|\u9ed8\u8ba4)?\s*(?:\u6d4f\u89c8\u5668|browser).{0,16}(?:\u4e2d)?\s*(?:\u6253\u5f00|\u67e5\u770b)",
    r"(?:\u6253\u5f00|\u8fdb\u5165|\u8df3\u8f6c\u5230).{0,24}(?:\u6d4f\u89c8\u5668|browser)",
    r"(\u7cfb\u7edf\u6743\u9650|\u5b9a\u4f4d\u6743\u9650|\u76f8\u673a\u6743\u9650|\u76f8\u518c\u6743\u9650).{0,30}(\u8bbe\u7f6e|\u6388\u6743|\u5141\u8bb8|\u7ba1\u7406|\u9009\u62e9)",
    # Generic path wording is not sufficient to authorize external apps; require file/directory evidence.

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
    r"\b(return|go back)\b.{0,30}\b(home screen|launcher|previous app)\b",
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
            raise ValueError(f"{key} coordinate is not numeric: {value!r}") from exc
    else:
        raise ValueError(f"Unsupported {key} coordinate type: {type(value).__name__}")

    if not math.isfinite(number):
        raise ValueError(f"{key} coordinate is not finite: {value!r}")
    return number


def _parse_optional_seconds(value: object, key: str) -> float | None:
    """Parse optional seconds; absent values remain unset and invalid values trigger a model retry."""
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{key} must be numeric: {value!r}")
    try:
        number = float(str(value).strip())
    except ValueError as exc:
        raise ValueError(f"{key} must be numeric: {value!r}") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{key} must be positive: {value!r}")
    return number


def _clamp_seconds(value: float | None, default: float, lower: float, upper: float) -> float:
    """Clamp the supplied duration, or its default, to the configured interval."""
    return max(lower, min(upper, default if value is None else value))


def _scale_plan_coordinates(
    plan: dict, screen_w: int, screen_h: int, image_size: tuple[int, int],
) -> None:
    """Map uploaded-image pixels to screen pixels without inferring other coordinate systems."""
    image_w, image_h = image_size
    if min(image_w, image_h, screen_w, screen_h) <= 1:
        raise ValueError("Image and screen dimensions must exceed 1")
    for keys, source, target in (
        (("x", "x1", "x2"), image_w, screen_w),
        (("y", "y1", "y2"), image_h, screen_h),
    ):
        for key in keys:
            if key not in plan:
                continue
            number = _parse_coordinate_value(plan.get(key), key)
            if number is None:
                plan.pop(key, None)
                continue
            if not 0 <= number <= source - 1:
                raise ValueError(f"{key} is outside current-image pixel range 0-{source - 1}: {number}")
            plan[key] = round(number / (source - 1) * (target - 1))


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
        pair = plan.get("x")
        if plan.get("y") is None and isinstance(pair, (list, tuple)) and len(pair) == 2:
            plan["x"], plan["y"] = pair
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
        "swipe": common_fields | set(SWIPE_COORDINATE_FIELDS) | {"duration"},
        "open_app": common_fields | {"app_name"},
        "press_back": common_fields,
        "finished": common_fields,
        "wait": common_fields | {"duration"},
    }
    allowed_fields = allowed_fields_by_action.get(action_name)
    if not allowed_fields:
        return plan
    return {
        key: value
        for key, value in plan.items()
        if key in allowed_fields and value is not None
    }


def _validate_action_plan(plan: dict, image_size: tuple[int, int] | None = None) -> None:
    """Reject incomplete model actions so they are retried instead of touching the device."""
    action_name = str(plan.get("action_name", "")).strip()
    valid_actions = {
        "click", "long_press", "type", "scroll", "swipe",
        "open_app", "press_back", "finished", "wait",
    }
    if action_name not in valid_actions:
        raise ValueError(f"Unsupported or missing action type: {action_name!r}")
    if action_name in {"click", "long_press"} and (
        plan.get("x") is None or plan.get("y") is None
    ):
        raise ValueError(f"{action_name} requires x/y coordinates")
    if action_name in {"click", "long_press"}:
        for key in COORDINATE_FIELDS:
            number = _parse_coordinate_value(plan.get(key), key)
            if image_size is None or min(image_size) <= 1:
                raise ValueError("Current screenshot dimensions are unavailable; cannot validate pixel coordinates")
            limit = image_size[0 if key.startswith("x") else 1] - 1
            if number is None or not 0 <= number <= limit:
                raise ValueError(f"{key} must be within current-image pixel range 0-{limit}: {plan.get(key)!r}")
    if action_name == "type" and not str(plan.get("content", "")):
        raise ValueError("type requires content")
    if action_name == "scroll":
        direction = str(plan.get("direction", "")).strip().lower()
        if direction not in {"up", "down", "left", "right"}:
            raise ValueError(f"Invalid scroll direction: {direction!r}")
    if action_name == "swipe":
        for key in SWIPE_COORDINATE_FIELDS:
            number = _parse_coordinate_value(plan.get(key), key)
            if number is None:
                raise ValueError(f"swipe requires {key}")
            if image_size is None or min(image_size) <= 1:
                raise ValueError("Current screenshot dimensions are unavailable; cannot validate pixel coordinates")
            limit = image_size[0 if key.startswith("x") else 1] - 1
            if not 0 <= number <= limit:
                raise ValueError(f"swipe {key} must be within current-image pixel range 0-{limit}: {plan.get(key)!r}")
        if plan.get("x1") == plan.get("x2") and plan.get("y1") == plan.get("y2"):
            raise ValueError("swipe start and end coordinates must differ")
    if action_name == "wait":
        plan["duration"] = _clamp_seconds(
            _parse_optional_seconds(plan.get("duration"), "duration"),
            DEFAULT_WAIT_SECONDS,
            MIN_WAIT_SECONDS,
            MAX_WAIT_SECONDS,
        )
    elif action_name == "swipe":
        plan["duration"] = _clamp_seconds(
            _parse_optional_seconds(plan.get("duration"), "duration"),
            DEFAULT_SWIPE_DURATION,
            MIN_SWIPE_DURATION,
            MAX_SWIPE_DURATION,
        )
    if action_name == "open_app" and not str(plan.get("app_name", "")).strip():
        raise ValueError("open_app requires app_name")
    if not isinstance(plan.get("is_interruption", False), bool):
        raise ValueError("is_interruption must be boolean")


_ACTION_CLAUSE_MARKERS = re.compile(
    r"\u4e0b\u4e00\u6b65|\u63a5\u4e0b\u6765|\u9700\u8981|\u51c6\u5907|\u5c06\u8981|\u7acb\u5373|\u70b9\u51fb|\u957f\u6309|\u8f93\u5165|\u6eda\u52a8|\u6253\u5f00|\u8fdb\u5165|\u9009\u62e9|\u53d1\u9001|\u5207\u6362|\u5f00\u542f|\u5173\u95ed|\u5220\u9664|\u6807\u8bb0|\u7f6e\u9876|\u8fd4\u56de"
)


def _proposed_action_text(plan: dict) -> str:
    """Return only the intended operation, excluding unrelated visible-page prose."""
    description = str(plan.get("action_description", "")).strip()
    if description:
        return description
    content = str(plan.get("content", "")).strip()
    thought = str(plan.get("thought", "")).strip()
    clauses = [item.strip() for item in re.split(r"[。；;\n]+", thought) if item.strip()]
    actionable = [item for item in clauses if _ACTION_CLAUSE_MARKERS.search(item)]
    intent = actionable[-1] if actionable else ""
    return " ".join(item for item in (intent, content) if item)


# Finish conclusions begin with a recognized verdict, optionally after punctuation.
FINISHED_LEADING_PUNCTUATION = "【[（(「『<《“\"'*#-—:：·• \t"
FINISHED_FAILURE_PREFIX = "goal unreachable"
LEGACY_FINISHED_FAILURE_PREFIX = "\u76ee\u6807\u4e0d\u53ef\u8fbe"
FINISHED_SUCCESS_PREFIX = "goal achieved"
LEGACY_FINISHED_SUCCESS_PREFIX = "\u76ee\u6807\u5df2\u8fbe\u6210"


def _finished_declares_failure(thought: str) -> bool:
    text = str(thought or "").strip().lower()
    if not text:
        return False
    # An explicit leading failure verdict takes precedence over success words in the explanatory body.


    leading = text.lstrip(FINISHED_LEADING_PUNCTUATION)
    if leading.startswith((FINISHED_FAILURE_PREFIX, LEGACY_FINISHED_FAILURE_PREFIX)):
        return True
    if leading.startswith((FINISHED_SUCCESS_PREFIX, LEGACY_FINISHED_SUCCESS_PREFIX)):
        # A success prefix cannot override an admitted unfinished target or substituted entry.
        return bool(MIXED_FINISH_FAILURE_PATTERN.search(text)) or any(
            re.search(pattern, text, re.IGNORECASE)
            for pattern in FINISHED_OVERRIDE_FAILURE_PATTERNS
        )
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


def _unfinished_safe_external_reason(action_name: str, foreground_package: str) -> str:
    """Prevent a model from ending while a required system dialog is unresolved."""
    if str(action_name or "").strip() != "finished":
        return ""
    foreground_package = str(foreground_package or "").strip()
    if not _is_safe_external_package(foreground_package):
        return ""
    return (
        f"Still on required external/system page {foreground_package}; "
        "resolve the visible selection and observe its result before finished"
    )


def _uses_adb_backend(device) -> bool:
    operating_system = str(getattr(device, "operating_system", "adb") or "adb").strip().lower()
    return operating_system in {"adb", "android"}


def _current_package(device, retries: int = FOREGROUND_CHECK_RETRIES) -> str | None:
    return current_foreground_package(device, retries, log_prefix="[Scenario execution]")


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


def _looks_like_launcher_package(package_name: str) -> bool:
    return bool(LAUNCHER_PACKAGE_PATTERN.search(str(package_name or "").strip()))


def _settle_foreground_state(device, package_name: str):
    """Recheck after a short delay to avoid treating transient launcher reports during page animation as external transitions."""


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
    """Poll for the target foreground instead of assuming a fixed cold-start delay."""
    # Keep the module-level foreground probe replaceable by tests.
    return wait_target_app_foreground(
        device,
        package_name,
        max_wait=max_wait,
        state_probe=lambda dev, pkg: _target_app_foreground_state(dev, pkg),
    )


def _restart_target_app(device, package_name: str) -> bool:
    package_name = package_name.strip()
    if not package_name:
        return True
    restarted, _, _ = restart_until_foreground(
        device,
        package_name,
        wait_foreground=lambda dev, pkg: _wait_target_app_foreground(dev, pkg),
        attempts=APP_RESTART_ATTEMPTS,
        log_prefix="[Scenario execution]",
    )
    return restarted


def _ensure_target_app(device, package_name: str) -> bool:
    foreground_state, current_package = _target_app_foreground_state(device, package_name)
    if foreground_state == "foreground":
        return True
    if is_permission_dialog(current_package):
        print("[Scenario execution] Preserve the permission dialog for observation and selection; avoid repeated restarts.")
        return True

    if foreground_state == "unknown":
        print(f"[Scenario execution] Foreground app unknown; preparing to restart target {package_name}.")
    else:
        print(
            f"[Scenario execution] Foreground={current_package}, target={package_name}; "
            "preparing to restart the target app."
        )
    return _restart_target_app(device, package_name)


def _recover_target_app_by_back(
    device,
    package_name: str,
    attempts: int = APP_RECOVERY_BACK_ATTEMPTS,
    *,
    continue_message: bool = True,
) -> bool:
    foreground_state, current_package = _target_app_foreground_state(device, package_name)
    if foreground_state == "foreground":
        return True

    # Reuse the foreground query instead of issuing another identical request.
    current_package = str(current_package or "").strip()
    if _looks_like_launcher_package(current_package):
        # Back cannot restore the target from the launcher; let the caller restart it.

        print(
            "[Scenario execution] Foreground is the launcher; back cannot restore the target. "
            "Skip back recovery and try restart."
        )
        return False

    for attempt in range(1, attempts + 1):
        if foreground_state == "unknown":
            print(f"[Scenario execution] Foreground uncertain; rechecking after delay {attempt}/{attempts}.")
        else:
            try:
                print(f"[Scenario execution] Target left foreground; back recovery attempt {attempt}/{attempts}.")
                _perform_action(device=device, action_name="press_back")
            except Exception as exc:
                print(f"[Scenario execution] Back recovery action failed: {exc}")
        time.sleep(FAST_ACTION_DELAY)
        foreground_state, _ = _target_app_foreground_state(device, package_name)
        if foreground_state == "foreground":
            message = "Continue the scenario." if continue_message else "Prepare to end the scenario."
            print(f"[Scenario execution] Restored target with back. {message}")
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
        print(f"[Scenario execution] Capture after back recovery failed: {exc}")
        page = trace[-1].get("current_page") if trace else None

    action_name = "press_back" if recovery_method == "press_back" else "restart_app"
    recovery_text = "back" if recovery_method == "press_back" else "target app restart"
    scene_text = "At scenario end, " if final_recovery else "During execution, "
    thought = (
        f"{scene_text}the target left foreground; the runtime restored it using {recovery_text}. "
        "This is environment recovery, not a valid business action in the user scenario."
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
                f"Target left foreground: current={previous_package or 'unknown'}, "
                f"target={package_name or 'unknown'}; restored using {recovery_text}."
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
        print(f"[Scenario execution] External-page capture before recovery failed: {exc}")
        return
    trace.append({
        "action_name": "capture_screenshot",
        "thought": "The runtime preserves the external foreground page before automatic return for trace auditing.",
        "action_description": "Observe external page before automatic recovery",
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
        "[Scenario execution] Saved external-page observation before recovery: "
        f"current={external_package or 'unknown'}, target={package_name or 'unknown'}"
    )


def _recover_from_exploration_boundary(
    device,
    package_name: str,
    hit: BoundaryHit,
) -> bool:
    """Exit an out-of-scope in-app page and report recovery. Reuse Activity-first boundary detection because its package may still be the target package."""


    attempts = max(1, APP_RECOVERY_BACK_ATTEMPTS)
    for attempt in range(1, attempts + 1):
        try:
            print(
                f"[Exploration boundary] Attempt {attempt}/{attempts}: {hit.rule.exit_action} "
                f"to exit {hit.rule.label} (via={hit.via}, matched={hit.matched})."
            )
            _perform_action(device=device, action_name="press_back")
        except Exception as exc:
            print(f"[Exploration boundary] Exit action failed: {exc}")
        time.sleep(FAST_ACTION_DELAY)
        try:
            current_activity = str(device.current_activity() or "")
        except Exception as exc:
            print(f"[Exploration boundary] Cannot read current Activity: {exc}")
            current_activity = ""
        if _current_boundary_hit(device, package_name, current_activity) is None:
            return True
    return False


def _current_boundary_hit(
    device,
    package_name: str,
    activity: str | None,
    foreground_package: str | None = None,
) -> BoundaryHit | None:
    """Use Activity as the authoritative boundary signal. Fall back to process matching only when the target is foreground and its Activity is unavailable; persistent service processes alone do not prove an active boundary crossing."""


    rule = rule_for(package_name)
    if not rule.enabled:
        return None
    activity = str(activity or "").strip()
    hit = check_boundary(package_name, activity)
    if hit is not None or activity or not rule.process_markers:
        return hit
    if foreground_package is None:
        try:
            foreground_package = str(device.current_package() or "").strip()
        except Exception:
            foreground_package = ""
    if str(foreground_package or "").strip() != str(package_name or "").strip():
        return None
    try:
        snapshot = str(device.connector.shell("ps -A") or "")
    except Exception as exc:
        print(f"[Exploration boundary] Process fallback failed (ignored): {exc}")
        return None
    return check_boundary(package_name, activity=None, process_snapshot=snapshot)


def _recover_to_target(
    trace: list[dict],
    device,
    package_name: str,
    trace_phase: str,
    previous_package: str,
    crash_logcat: str,
    back,
    *,
    reason: str = "",
    final_recovery: bool = False,
    crashed: bool | None = None,
) -> tuple[bool, bool]:
    """Return (crashed, recovered): restart on a crash; otherwise try back before restart. Append a non-business recovery step on success. Reuse a supplied crash verdict to avoid parsing logcat twice."""


    if crashed is None:
        crashed = _logcat_has_target_process_crash(crash_logcat, package_name)
    method = ""
    if not crashed and back is not None and back():
        method = "press_back"
    elif _restart_target_app(device, package_name):
        method = "restart_app"
    if method:
        _append_target_app_recovery_step(
            trace,
            device,
            package_name,
            previous_package,
            trace_phase,
            recovery_method=method,
            final_recovery=final_recovery,
            reason=reason,
        )
    return crashed, bool(method)


def _recovery_outcome(crashed: bool, recovered: bool) -> tuple[str, str]:
    """Return a terminal state for crashes or failed recovery; callers decide successful non-crash recovery outcomes."""
    if crashed:
        return "app_crashed", (
            TARGET_APP_CRASHED_END if recovered else TARGET_APP_CRASH_RECOVERY_FAILED_END
        )
    return "left_app", TARGET_APP_RECOVERY_FAILED_END


def _clear_logcat(device):
    if not _uses_adb_backend(device):
        return
    device_serial = device.serial if hasattr(device, "serial") else ""
    if not device_serial:
        print("[Scenario execution] Device serial unavailable; skip clearing logcat.")
        return
    try:
        subprocess.run(
            ["adb", "-s", device_serial, "logcat", "-c"],
            timeout=5,
            capture_output=True,
        )
        print(f"[Scenario execution] Cleared logcat: device={device_serial}")
    except Exception as exc:
        print(f"[Scenario execution] Clearing logcat failed: {exc}")


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
        print(f"[Scenario execution] Reading crash logcat failed: {exc}")
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


def _sleep_for_llm_retry(consecutive_errors: int, deadline: float | None = None) -> bool:
    """Sleep for retry backoff only when the remaining phase budget permits it."""
    delay = compute_retry_delay_seconds(consecutive_errors)
    if not deadline_allows(deadline, delay):
        return False
    time.sleep(delay)
    return True


def _describe_generation_failure(
    raw_content: str,
    error: Exception,
    response_received: bool = True,
) -> str:
    """Record one concise failure reason in execution history, distinguishing transport errors, empty content, non-JSON prose and invalid JSON so the next decision can correct the actual problem."""


    if not response_received:
        return f"Model call failed (transport/timeout), no response received: {type(error).__name__}"
    if not str(raw_content or "").strip():
        return "Empty model response; no action executed. Return exactly one JSON object."
    if "{" not in str(raw_content or ""):
        return (
            "Response contained prose without JSON; no action executed. "
            "Return exactly one JSON object, not a textual description of the intended action."
        )
    return f"Response failed parsing/validation; no action executed: {str(error)[:300]}"


def _runtime_page_label(page: Page | None) -> str:
    """Use page name, description, then Activity as a runtime history label."""
    if not isinstance(page, Page):
        return "Unknown page"
    return page.name or page.description or page.activity or "Unknown page"


def _model_point(step: dict) -> tuple[int, int] | None:
    coordinates = step.get("model_coordinates")
    if isinstance(coordinates, dict) and "x" in coordinates and "y" in coordinates:
        try:
            return int(round(float(coordinates["x"]))), int(round(float(coordinates["y"])))
        except (TypeError, ValueError):
            return None
    return None


def _action_effect(
    before_screenshot: str,
    after_screenshot: str,
    image_size: tuple[int, int] | None,
) -> tuple[str, str]:
    """Return a pixel-based effect category and description: none, local, major or empty. Do not infer effects from the UI hierarchy."""


    result = screenshot_change_region(before_screenshot, after_screenshot)
    if result is None:
        return "", ""
    ratio, box = result
    if box is None or ratio < EFFECT_NO_CHANGE_RATIO:
        return "none", "No visible change"
    if ratio >= EFFECT_MAJOR_CHANGE_RATIO:
        return "major", "Major visible change"
    region = ""
    if image_size:
        width, height = image_size
        region = (
            f", approximate changed region ({int(box[0] * width)},{int(box[1] * height)})"
            f"–({int(box[2] * width)},{int(box[3] * height)})"
        )
    return "local", f"Local visible change (about {ratio * 100:.2f}%{region})"


# Cache effects by screenshot hashes and dimensions, outside serialized trace-step dictionaries.

_ACTION_EFFECT_CACHE = BoundedCache(max(16, EFFECT_HISTORY_STEPS * 8))


def _cached_action_effect(
    before_screenshot: str,
    after_screenshot: str,
    image_size: tuple[int, int] | None,
) -> tuple[str, str]:
    key = (
        screenshot_digest(before_screenshot),
        screenshot_digest(after_screenshot),
        tuple(image_size) if image_size else None,
    )
    if key in _ACTION_EFFECT_CACHE:
        return _ACTION_EFFECT_CACHE.get(key)
    effect = _action_effect(before_screenshot, after_screenshot, image_size)
    _ACTION_EFFECT_CACHE.put(key, effect)
    return effect


def _step_effects(
    trace: list[dict],
    current_screenshot: str,
    image_size: tuple[int, int] | None,
) -> dict[int, tuple[str, str]]:
    """Return recent effects by trace index. The next pre-action screenshot observes the prior result; cache already compared pairs."""


    effects: dict[int, tuple[str, str]] = {}
    start = max(0, len(trace) - EFFECT_HISTORY_STEPS)
    for index in range(start, len(trace)):
        step = trace[index]
        if str(step.get("action_name", "")).strip() not in EFFECT_ACTIONS:
            continue
        before = _page_screenshot(step)
        after = _page_screenshot(trace[index + 1]) if index + 1 < len(trace) else current_screenshot
        if not before or not after:
            continue
        kind, text = _cached_action_effect(before, after, image_size)
        if kind:
            effects[index] = (kind, text)
    return effects


def _no_response_note(
    trace: list[dict],
    effects: dict[int, tuple[str, str]],
) -> str:
    """Warn against repeating an unchanged click at the same position; suggest prerequisites or another target."""
    if not trace:
        return ""
    last = trace[-1]
    action = str(last.get("action_name", "")).strip()
    last_effect = effects.get(len(trace) - 1, ("", ""))[0]
    if action == "type" and last_effect in {"none", "local"}:
        if last_effect == "none":
            return (
                "[Previous effect] No visible change after type: the field may be unfocused or reject this input. "
                "Focus the intended field or use input matching its format; do not repeat unchanged input.\n"
            )
        return (
            "[Previous effect] type executed. Compare visible field content with the supplied content. "
            "A mismatch indicates character/length restrictions; adapt the value instead of repeating it.\n"
        )
    if action not in {"click", "long_press"} or last_effect != "none":
        return ""
    point = _model_point(last)
    repeats = 0
    for index in range(len(trace) - 1, -1, -1):
        step = trace[index]
        if str(step.get("action_name", "")).strip() != action:
            break
        if effects.get(index, ("", ""))[0] != "none":
            break
        other = _model_point(step)
        if point and other and (
            abs(point[0] - other[0]) > EFFECT_SAME_POINT_TOLERANCE
            or abs(point[1] - other[1]) > EFFECT_SAME_POINT_TOLERANCE
        ):
            break
        repeats += 1
    where = f" ({point[0]},{point[1]})" if point else ""
    if repeats >= 2:
        return (
            f"[Previous effect] {repeats} consecutive clicks near {where} produced no change; the control is currently unresponsive. "
            "Do not click this position again. Check prerequisites such as required fields, counts or selection, "
            "or test another unexecuted control. If neither is possible, finish with 'Goal unreachable: control unresponsive'.\n"
        )
    return (
        f"[Previous effect] No visible change after clicking {where}. Check the actual target position in the screenshot. "
        "If mis-aimed, retry once at its center; if correctly aimed, do not repeat the unresponsive action. "
        "Satisfy prerequisites or test another control.\n"
    )


def _repeated_success_note(
    trace: list[dict],
    effects: dict[int, tuple[str, str]],
) -> str:
    """Warn when an already successful click is repeated, including actions that keep reopening the same panel and otherwise consume steps before stall detection."""


    if not trace:
        return ""
    last = trace[-1]
    action = str(last.get("action_name", "")).strip()
    if action not in {"click", "long_press"}:
        return ""
    last_index = len(trace) - 1
    if effects.get(last_index, ("", ""))[0] not in {"local", "major"}:
        return ""
    point = _model_point(last)
    if point is None:
        return ""
    start = max(0, len(trace) - EFFECT_HISTORY_STEPS)
    for index in range(last_index - 1, start - 1, -1):
        step = trace[index]
        if str(step.get("action_name", "")).strip() != action:
            continue
        if effects.get(index, ("", ""))[0] not in {"local", "major"}:
            continue
        other = _model_point(step)
        if other and (
            abs(point[0] - other[0]) > EFFECT_SAME_POINT_TOLERANCE
            or abs(point[1] - other[1]) > EFFECT_SAME_POINT_TOLERANCE
        ):
            continue
        effect_text = effects.get(index, ("", ""))[1]
        return (
            f"[Already executed] The previous click ({point[0]},{point[1]}) succeeded earlier in this scenario ({effect_text}). "
            "The interface already reflects its result. Do not repeat it unchanged; "
            "test another unexecuted control, or finish if the goal is achieved and no new target remains.\n"
        )
    return ""


def _mark_tap_point(screenshot: str, point: tuple[int, int] | None) -> str:
    """Mark the actual historical tap with a red crosshair to distinguish inaccurate aim from an unresponsive control. Return the original image on failure."""


    if not screenshot or point is None:
        return screenshot
    image = decode_screenshot(screenshot)
    if image is None:
        return screenshot
    height, width = image.shape[:2]
    x = min(max(int(point[0]), 0), width - 1)
    y = min(max(int(point[1]), 0), height - 1)
    radius = max(8, min(width, height) // 40)
    cv2.circle(image, (x, y), radius, (0, 0, 255), 3)
    cv2.line(image, (x - radius * 2, y), (x + radius * 2, y), (0, 0, 255), 2)
    cv2.line(image, (x, y - radius * 2), (x, y + radius * 2), (0, 0, 255), 2)
    try:
        return encode_image_to_jpg_base64(image)
    except ValueError:
        return screenshot


def _format_execution_history(
    trace: list[dict],
    effects: dict[int, tuple[str, str]] | None = None,
) -> str:
    if not trace:
        return "No historical steps."

    effects = effects or {}
    lines = []
    for index, step in enumerate(trace, start=1):
        page = step.get("current_page")
        action = str(step.get("action_name", "")).strip() or "Unknown action"
        thought = str(step.get("thought", "")).strip()
        content = str(step.get("content", "")).strip()
        direction = str(step.get("direction", "")).strip()
        detail_parts = []
        if step.get("external_context"):
            external_package = str(step.get("external_package", "")).strip() or "External app"
            detail_parts.append(f"External page: {external_package}")
        if step.get("caused_external_transition"):
            external_package = str(step.get("external_package", "")).strip() or "External app"
            detail_parts.append(f"External transition: {external_package}")
        if step.get("is_interruption"):
            detail_parts.append("Interruption/recovery step")
        if step.get("foreground_recovery"):
            previous_package = str(step.get("previous_package", "")).strip() or "unknown"
            target_package = str(step.get("target_package", "")).strip() or "target app"
            detail_parts.append(f"Automatic back recovery: {previous_package} -> {target_package}")
        if content:
            detail_parts.append(f"Input/text: {content}")
        if direction:
            detail_parts.append(f"Direction: {direction}")
        if step.get("action_failed"):
            action_error = str(step.get("action_error", "")).strip() or "Device action failed"
            detail_parts.append(f"Action failed: {action_error[:200]}")
        point = _model_point(step) if action in {"click", "long_press"} else None
        if point:
            detail_parts.append(f"Image coordinates: ({point[0]},{point[1]})")
        effect = effects.get(index - 1)
        if effect:
            detail_parts.append(f"Effect: {effect[1]}")
        detail = f" | {'; '.join(detail_parts)}" if detail_parts else ""
        thought_text = f" | Assessment: {thought}" if thought else ""
        lines.append(f"{index}. Page: {_runtime_page_label(page)} | Action: {action}{detail}{thought_text}")
    return "\n".join(lines)


def _foreground_state_note(foreground_package: str, package_name: str) -> str:
    """Give the model package-authoritative foreground state information."""
    foreground_package = str(foreground_package or "").strip()
    package_name = str(package_name or "").strip()
    if not foreground_package or not package_name:
        return ""
    if foreground_package == package_name:
        return f"[Foreground] {foreground_package} is the target app. Its current screenshot is not the system launcher merely because Activity contains Launcher/Home.\n"
    if _looks_like_launcher_package(foreground_package):
        return f"[Foreground] {foreground_package} is the system launcher. Target {package_name} is not foreground and may still be starting.\n"
    return f"[Foreground] {foreground_package} is not the target app {package_name}.\n"


def _recent_failure_note(recent_failures: list[dict] | None) -> str:
    """Expose recent generation failures to the next model decision; return empty text for no failures."""
    failures = [f for f in (recent_failures or []) if f]
    if not failures:
        return ""
    entries = failures[-MAX_RECENT_FAILURES:]
    lines = []
    for failure in entries:
        reason = str(failure.get("reason", "") or "Unknown reason")
        attempt = int(failure.get("attempt", 1) or 1)
        attempt_text = f" (attempt {attempt})" if attempt > 1 else ""
        lines.append(f"- Step {failure.get('step', '?')}{attempt_text} generation failed: {reason}")
    body = "\n".join(lines)
    return (
        "[Recent failures] No actions were executed for these steps; the page did not change. "
        f"Adjust the output before deciding again:\n{body}\n\n"
    )


def _history_screenshot_label() -> str:
    if KEEP_IMAGE_STEPS <= 0:
        return "[Historical screenshots] None attached; use the execution history and current screenshot."
    if KEEP_IMAGE_STEPS == 1:
        return "[Page before the previous action]"
    return f"[Latest {KEEP_IMAGE_STEPS} historical screenshots]"


def _recent_history_steps(trace: list[dict]) -> list[dict]:
    """Return the latest KEEP_IMAGE_STEPS entries; handle zero explicitly because trace[-0:] returns everything."""
    return trace[-KEEP_IMAGE_STEPS:] if KEEP_IMAGE_STEPS > 0 else []


def _build_step_input(
    trace: list[dict],
    current_screenshot: str,
    step_index: int,
    max_steps: int,
    screenshot_health_error: str = "",
    foreground_note: str = "",
    image_size: tuple[int, int] | None = None,
    recent_failures: list[dict] | None = None,
) -> list[dict]:
    image_size = image_size or screenshot_dimensions(current_screenshot)
    coordinate_note = (
        f"[Coordinates] Current image: width={image_size[0]}, height={image_size[1]} pixels. "
        f"Origin=(0, 0), x=0-{image_size[0] - 1}, y=0-{image_size[1] - 1}. "
        "Use current-image pixels for clicks, long presses and swipes, never normalized or screen coordinates.\n"
        if image_size else "Current screenshot dimensions unavailable; do not emit coordinate actions.\n"
    )
    remaining_steps = max(0, max_steps - step_index + 1)
    effects = _step_effects(trace, current_screenshot, image_size)
    effect_note = _no_response_note(trace, effects)
    repeat_note = _repeated_success_note(trace, effects)
    scroll_budget_note = ""
    if trace and str(trace[-1].get("trace_phase", "")).strip() != "navigation":
        scroll_direction, consecutive_scrolls = (
            _consecutive_same_direction_scroll(trace)
        )
        if consecutive_scrolls >= TEST_MAX_CONSECUTIVE_SAME_DIRECTION_SCROLLS:
            scroll_budget_note = (
                f"[Scroll budget] Consecutive {_format_scroll_direction(scroll_direction)} gestures: "
                f"{consecutive_scrolls}; same-direction limit reached "
                f"({TEST_MAX_CONSECUTIVE_SAME_DIRECTION_SCROLLS}). "
                "Do not continue in this direction; select a list item, return or explore another target.\n"
            )
        elif consecutive_scrolls >= TEST_MAX_CONSECUTIVE_SAME_DIRECTION_SCROLLS - 1:
            scroll_budget_note = (
                f"[Scroll budget] Consecutive {_format_scroll_direction(scroll_direction)} gestures: "
                f"{consecutive_scrolls}; maximum "
                f"{TEST_MAX_CONSECUTIVE_SAME_DIRECTION_SCROLLS} consecutive gestures in one direction. "
                "Prepare to select a list item, return or explore another target.\n"
            )
    content: list[dict] = [
        {
            "type": "text",
            "text": (
                f"[Step {step_index}] Choose the next action from the task, full history and current screenshot.\n"
                f"This phase allows {max_steps} steps; current={step_index}, remaining including this step={remaining_steps}.\n"
                "If too few steps remain for valuable exploration, record the current verifiable result and finish.\n"
                "Use one representative among equivalent controls leading to the same result. "
                "Intermediate pages are not automatic endpoints; finish when the direction is complete or no relevant unexecuted control remains.\n"
                "Before responding, check whether the same action on the same page/control recently produced no change. "
                "If so, change strategy or finish rather than repeating it.\n"
                f"{scroll_budget_note}"
                f"{effect_note}"
                f"{repeat_note}"
                "[Complete execution history]\n"
                f"{_format_execution_history(trace, effects)}\n\n"
                f"{_recent_failure_note(recent_failures)}"
                f"{_history_screenshot_label()}"
            ),
        }
    ]

    recent_steps = _recent_history_steps(trace)
    for offset, step in enumerate(recent_steps, start=1):
        page = step.get("current_page")
        screenshot = page.representative_screenshot if isinstance(page, Page) else ""
        if not screenshot:
            continue
        absolute_index = len(trace) - len(recent_steps) + offset
        label = f"Historical screenshot: page before step {absolute_index}."
        tap_point = _model_point(step) if str(step.get("action_name", "")).strip() in {"click", "long_press"} else None
        if tap_point is not None and effects.get(absolute_index - 1, ("", ""))[0] == "none":
            # Mark ineffective taps so the next decision can distinguish aim from control response.
            screenshot = _mark_tap_point(screenshot, tap_point)
            label += (
                f"The red crosshair marks the actual tap ({tap_point[0]},{tap_point[1]}), after which no visible change occurred. "
                "If it misses the target, correct coordinates using the relative position of crosshair and control. "
                "If it is already at the center, the control was unresponsive."
            )
        content.append({
            "type": "text",
            "text": label,
        })
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{screenshot}"},
        })

    content.append({
        "type": "text",
        "text": (
            f"{foreground_note}"
            f"{coordinate_note}"
            "[Current screenshot] Return the next JSON action using this image."
            + (
                "\nScreenshot health suggests a black or unavailable page. "
                "Prefer one press_back if possible; do not repeatedly enter the cause of this state."
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

    # Share JSON typo repair with llm.invoke_json to avoid divergent parser behavior.

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

    try:
        parsed = _parse_json_response(text)
        if isinstance(parsed, dict):
            return parsed
    except ValueError:
        pass
    raise ValueError("Model output contains no valid JSON object")


def _recent_screenshot_count(trace: list[dict]) -> int:
    count = 0
    for step in _recent_history_steps(trace):
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
        int(step.get("x2", 0) or 0) // 20,
        int(step.get("y2", 0) or 0) // 20,
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
    if action_name == "swipe":
        same_gesture = len({
            (
                int(step.get("x1", 0) or 0) // 20,
                int(step.get("y1", 0) or 0) // 20,
                int(step.get("x2", 0) or 0) // 20,
                int(step.get("y2", 0) or 0) // 20,
            )
            for step in steps
        }) <= 1
        return same_gesture and _screenshots_stable(steps)
    if action_name == "open_app":
        same_app = len({str(step.get("app_name", "")).strip() for step in steps}) <= 1
        return same_app and _screenshots_stable(steps)
    return action_name in {"press_back", "wait"} and _screenshots_stable(steps)


def _step_scroll_direction(step: dict) -> str:
    """Use scroll.direction or derive finger motion from swipe endpoints so both gestures share budgets and stall detection."""


    action_name = str(step.get("action_name", "")).strip()
    if action_name == "scroll":
        direction = str(step.get("direction", "")).strip().lower()
        return direction if direction in {"up", "down", "left", "right"} else ""
    if action_name != "swipe":
        return ""
    try:
        x1 = float(step.get("x1", 0) or 0)
        y1 = float(step.get("y1", 0) or 0)
        x2 = float(step.get("x2", 0) or 0)
        y2 = float(step.get("y2", 0) or 0)
    except (TypeError, ValueError):
        return ""
    dx, dy = x2 - x1, y2 - y1
    if abs(dx) < 1 and abs(dy) < 1:
        return ""
    if abs(dy) >= abs(dx):
        return "up" if dy < 0 else "down"
    return "left" if dx < 0 else "right"


def _format_scroll_direction(direction: str) -> str:
    return {
        "up": "up",
        "down": "down",
        "left": "left",
        "right": "right",
    }.get(str(direction or "").strip(), str(direction or "").strip() or "unknown direction")


def _consecutive_same_direction_scroll(trace: list[dict]) -> tuple[str, int]:
    """Count the trailing scroll/swipe segment in one direction. Other actions or a direction change break it; unknown trailing direction yields zero."""


    if not trace:
        return "", 0
    latest_direction = _step_scroll_direction(trace[-1])
    if not latest_direction:
        return "", 0
    count = 0
    for step in reversed(trace):
        if _step_scroll_direction(step) != latest_direction:
            break
        count += 1
    return latest_direction, count


def _screen_point(step: dict) -> tuple[int, int] | None:
    if step.get("x") is None or step.get("y") is None:
        return None
    try:
        return int(round(float(step["x"]))), int(round(float(step["y"])))
    except (TypeError, ValueError):
        return None


def _points_pairwise_distinct(points: list[tuple[int, int]], tolerance: int) -> bool:
    for index, point in enumerate(points):
        for other in points[index + 1:]:
            if abs(point[0] - other[0]) <= tolerance and abs(point[1] - other[1]) <= tolerance:
                return False
    return True


def _alternation_explores_distinct_targets(
    steps: list[dict],
    action_names: list[str],
) -> bool:
    """Recognize entry-back-next-target exploration. Require press_back and comparable coordinates in one system for every click/long_press, with distinct targets beyond tolerance. Missing or repeated coordinates remain a loop."""


    if "press_back" not in action_names:
        return False
    forward_steps = [
        step
        for step, action in zip(steps, action_names)
        if action in {"click", "long_press"}
    ]
    if len(forward_steps) < 2:
        return False
    model_points = [_model_point(step) for step in forward_steps]
    if all(point is not None for point in model_points):
        return _points_pairwise_distinct(model_points, EFFECT_SAME_POINT_TOLERANCE)
    screen_points = [_screen_point(step) for step in forward_steps]
    if all(point is not None for point in screen_points):
        return _points_pairwise_distinct(screen_points, REPEAT_COORD_TOLERANCE)
    return False


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
        # Entry-back-next-control exploration can alternate the same two modeled pages; only repeated forward targets imply a loop.


        and not _alternation_explores_distinct_targets(recent, action_names)
    ):
        return "Page navigation loop detected"

    # For period-three cycles, a back step can identify legitimate entry-action-return exploration across different instances sharing one Page.


    if page_keys[-6:-3] == page_keys[-3:] and "press_back" not in action_names:
        return "Page navigation loop detected"

    return ""


def _detect_visual_action_cycle(trace: list[dict]) -> str:
    """Detect an A→B→A→B UI loop even when both states share one Activity."""
    if len(trace) < 4:
        return ""
    recent = trace[-4:]
    if any(
        str(step.get("action_name", "")).strip() not in PAGE_CYCLE_ACTIONS
        for step in recent
    ):
        return ""
    # A trailing wait observes the revisited state; it is not another navigation cycle.

    if str(recent[-1].get("action_name", "")).strip() == "wait":
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
        return "Visual-state alternation within the same Activity detected"
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


def _wait_observed_progress(trace: list[dict], start: int) -> bool:
    """Check whether a wait is followed by visible progress in the next pre-action screenshot."""
    for index in range(max(0, start), len(trace) - 1):
        if str(trace[index].get("action_name", "")).strip() != "wait":
            continue
        before, after = _page_screenshot(trace[index]), _page_screenshot(trace[index + 1])
        if before and after and _cached_action_effect(before, after, None)[0] in {"local", "major"}:
            return True
    return False


def _detect_no_progress_stall(trace: list[dict]) -> str:
    """Catch direction-independent blind probing using Activity and screenshots. Mixed clicks and scroll directions can evade action-specific checks; no UI hierarchy is needed."""


    if len(trace) < NO_PROGRESS_STALL_WINDOW:
        return ""

    recent = trace[-NO_PROGRESS_STALL_WINDOW:]
    if any(
        str(step.get("action_name", "")).strip() not in NO_PROGRESS_STALL_ACTIONS
        for step in recent
    ):
        return ""
    # Interruption recovery has dedicated cycle checks.
    if any(bool(step.get("is_interruption")) for step in recent):
        return ""
    # Visible progress after waiting is not blind probing; pixel comparisons detect small counters that thumbnails miss.

    if _wait_observed_progress(trace, len(trace) - len(recent)):
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
        f"Blind probing for {NO_PROGRESS_STALL_WINDOW} steps on the same interface; "
        f"only {visual_states} visual states observed; current goal is unreachable"
    )


def _detect_stall(trace: list[dict]) -> str:
    """Detect loops, no progress, recovery cycles, same-direction scroll exhaustion and unchanged repeated actions, using actions/screenshots rather than model prose."""


    if len(trace) < 2:
        return ""

    for detector in (
        _detect_page_navigation_cycle,
        _detect_visual_action_cycle,
        _detect_no_progress_stall,
    ):
        reason = detector(trace)
        if reason:
            return reason

    if len(trace) >= 4:
        recent_cycle = trace[-4:]
        cycle_signatures = [_step_signature(step) for step in recent_cycle]
        if (
            any(bool(step.get("is_interruption")) for step in recent_cycle)
            and cycle_signatures[:2] == cycle_signatures[2:]
        ):
            return "Repeated cycle between target action and interruption recovery detected"

    latest_action = str(trace[-1].get("action_name", "")).strip()
    if (
        latest_action in SCROLL_LIKE_ACTIONS
        and str(trace[-1].get("trace_phase", "")).strip() != "navigation"
    ):
        scroll_direction, consecutive_scrolls = _consecutive_same_direction_scroll(trace)
        if consecutive_scrolls > TEST_MAX_CONSECUTIVE_SAME_DIRECTION_SCROLLS:
            return (
                f"Consecutive {_format_scroll_direction(scroll_direction)} scrolls: "
                f"{consecutive_scrolls}, exceeding same-direction budget "
                f"({TEST_MAX_CONSECUTIVE_SAME_DIRECTION_SCROLLS}); "
                "select a list item, return or explore another direction"
            )

    repeat_window = STALL_REPEAT_WINDOWS.get(latest_action, STALL_REPEAT_WINDOW)
    if len(trace) < repeat_window:
        return ""
    recent = trace[-repeat_window:]
    if _is_consecutive_repeated_action(recent):
        if latest_action == "type":
            return f"Detected {repeat_window} consecutive repeated inputs"
        if latest_action in {"click", "long_press"}:
            return f"Detected {repeat_window} consecutive clicks in the same region"
        return f"Detected {repeat_window} consecutive repeated actions: {latest_action}"
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
        "thought": "The runtime observed a black/unavailable page; end this scenario and let the next round recover.",
        "action_description": "Observe black or unavailable page",
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
        print(f"[Scenario execution] Final observation unavailable: {exc}")
        return None, str(exc)
    except Exception as exc:
        print(f"[Scenario execution] Final observation failed: {exc}")
        return None, ""
    trace.append({
        "action_name": "capture_screenshot",
        "thought": "Observe the last side-effecting action's result before ending the scenario.",
        "action_description": "Final page observation before scenario termination",
        "current_page": page,
        "trace_phase": trace_phase,
        "observation_only": True,
        "terminal_observation": True,
        **({"screenshot_health_error": health_error} if health_error else {}),
    })
    print(
        "[Scenario execution] Added final observation: "
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
    print("[Scenario execution] Consecutive black-screen threshold reached; restart and wait for stabilization.")
    if not _restart_target_app(device, package_name):
        return None, "Target restart after consecutive black screens failed"
    try:
        page = capture_current_page(device)
    except ScreenshotUnavailableError as exc:
        return None, f"Screenshot still unavailable after black-screen restart: {exc}"
    except Exception as exc:
        return None, f"Capture after black-screen restart failed: {exc}"
    trace.append({
        "action_name": "restart_app",
        "thought": "The runtime restarts the target after consecutive black screens to restore a testable state.",
        "action_description": "Restart target after consecutive black screens",
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
        print("[Scenario execution] Executor already backed out of the blocked page; do not repeat back.")
        return None
    try:
        print("[Scenario execution] Cannot proceed; press back once to exit the blocked page.")
        _perform_action(device=device, action_name="press_back")
        time.sleep(FAST_ACTION_DELAY)
        page = capture_current_page(device)
    except Exception as exc:
        print(f"[Scenario execution] Failed to exit blocked page; switch functionalities: {exc}")
        return None
    trace.append({
        "action_name": "press_back",
        "thought": "The runtime exits a CAPTCHA or another confirmed impassable page.",
        "action_description": "Exit an impassable blocked page",
        "current_page": page,
        "trace_phase": trace_phase,
        "is_interruption": True,
        "blocked_page_exit": True,
        "observation_after_recovery": True,
    })
    return page


def _task_phase_label(phase: str) -> str:
    """Describe the planner-selected functionality phase to the executor."""
    phase = str(phase or "").strip().lower()
    if phase == "depth":
        return (
            "depth: complete one business action sequence for the current functionality; "
            "do not stop merely upon reaching the first new page"
        )
    if phase == "breadth":
        return "breadth: exercise entries and visible controls and observe direct results"
    return "Unspecified: follow the task's stated boundary"


@dataclass
class _ScenarioRun:
    """Mutable execution state shared by scenario-stage helpers."""

    llm: Any
    device: Any
    package_name: str
    trace_phase: str
    step_limit: int
    deadline: float | None
    exploration_mode: str
    allows_external_app: bool
    allows_target_exit: bool
    initial_page_id: str | None
    trace: list[dict] = field(default_factory=list)
    status: str = "completed"
    error_message: str = ""
    final_page_override: Page | None = None
    crash_logcat: str = ""
    # Observation failures and generation failures have independent consecutive-error budgets.

    observation_errors: int = 0
    llm_errors: int = 0
    # Bound screenshot-triggered restarts to avoid a persistent failure/restart loop.
    screenshot_restart_count: int = 0
    action_failure_recoveries: int = 0
    # Retain recent generation failures in every subsequent decision until corrected.
    recent_failures: list[dict] = field(default_factory=list)

    def stop(self, status: str, error_message: str) -> None:
        self.status, self.error_message = status, error_message

    def capture_final_page(self) -> None:
        try:
            self.final_page_override = capture_current_page(self.device)
        except Exception as capture_err:
            print(f"[Scenario execution] Capture after recovery failed: {capture_err}")

    def record_generation_failure(self, step_index: int, reason: str) -> None:
        attempt = 1 + sum(
            1 for failure in self.recent_failures if failure.get("step") == step_index
        )
        self.recent_failures.append({"step": step_index, "attempt": attempt, "reason": reason})
        self.recent_failures[:] = self.recent_failures[-MAX_RECENT_FAILURES:]

    def deadline_allows(self, seconds: float) -> bool:
        return deadline_allows(self.deadline, seconds)


@dataclass
class _StepObservation:
    """One decision observation: foreground, page screenshot, image dimensions and screen dimensions."""

    page: Page
    health_error: str
    foreground_package: str
    external_context: bool
    external_package: str
    screen_size: tuple[int, int] = (0, 0)
    image_size: tuple[int, int] = (0, 0)


# Retry delay after screenshot or dimension observations fail.
OBSERVATION_RETRY_DELAY = 2.0
# Helper control results: continue this step, reobserve, or stop the scenario.
_STEP_OK = "ok"
_STEP_RETRY = "retry"
_STEP_STOP = "stop"


def _prepare_scenario_start(
    device,
    package_name: str,
    scenario: str,
    clear_logcat: bool,
    trace_phase: str,
    compliance_cold_start: bool,
    deadline: float | None,
) -> tuple[str, str] | None:
    """Apply safety checks, establish the logcat start, handle cold start and confirm foreground. Return (status, error) on failure."""
    risky_scenario_intent = wechat_read_only_violation(package_name, scenario)
    if risky_scenario_intent:
        reason = f"{WECHAT_READ_ONLY_BLOCKED_END}: {risky_scenario_intent}"
        print(f"[Scenario execution] {reason}")
        return "blocked", reason

    if clear_logcat:
        _clear_logcat(device)
        print(f"[Scenario execution] Set scenario logcat start: phase={trace_phase}")

    if compliance_cold_start:
        print(
            "[Scenario execution] Compliance cold start restarts the target while retaining data. "
            "Data may be cleared only once, before the overall run starts."
        )
        if not deadline_allows(deadline, APP_RESTART_MAX_SECONDS):
            return "incomplete", "Insufficient phase time for compliance cold start"
        if not _restart_target_app(device, package_name):
            return "error", "Compliance cold start failed to launch target"
    if not _ensure_target_app(device, package_name):
        return "error", "Cannot launch target app"
    return None


def _check_target_foreground(run: _ScenarioRun) -> tuple[str, str, str]:
    """Recover and stop after unexpected external transitions or crashes. Return control result, foreground package and allowed external package; a nonempty allowed package permits continuation."""


    device, package_name, trace = run.device, run.package_name, run.trace
    foreground_state, current_package = _target_app_foreground_state(device, package_name)
    if foreground_state != "foreground":
        # Settle transient launcher reports before declaring an external transition or crash.

        foreground_state, current_package = _settle_foreground_state(device, package_name)
    if foreground_state == "foreground":
        if not current_package:
            # Without a target package the foreground probe is a no-op; query once for prompts and guards.
            try:
                current_package = str(device.current_package() or "").strip()
            except Exception:
                current_package = ""
        return _STEP_OK, current_package or "", ""

    scene_crash_logcat = _read_crash_logcat(device, package_name)
    # Parse each logcat snapshot for a crash once and reuse the verdict.
    target_crashed = _logcat_has_target_process_crash(scene_crash_logcat, package_name)
    controlled_external = (
        foreground_state == "background"
        and not target_crashed
        and (
            run.allows_external_app
            or run.allows_target_exit
            or _is_safe_external_package(current_package)
        )
    )
    if foreground_state == "background" and trace and not trace[-1].get("external_context"):
        trace[-1]["caused_external_transition"] = True
        trace[-1]["external_package"] = current_package or ""
    if controlled_external:
        print(
            f"[Scenario execution] Allowed external context; continue scenario: "
            f"current={current_package or 'unknown'}, target={package_name or 'unknown'}"
        )
        return _STEP_OK, current_package or "", current_package or ""

    print(
        f"[Scenario execution] Target left foreground; restore and end scenario: status={foreground_state}, "
        f"current={current_package or 'unknown'}, target={package_name or 'unknown'}"
    )
    if target_crashed:
        run.crash_logcat = scene_crash_logcat
        if trace:
            trace[-1]["caused_target_crash"] = True
    elif foreground_state == "background":
        _append_external_observation_step(
            trace, device, package_name, current_package or "unknown", run.trace_phase,
        )
    crashed, recovered = _recover_to_target(
        trace, device, package_name, run.trace_phase, current_package or "unknown",
        scene_crash_logcat,
        back=lambda: _recover_target_app_by_back(
            device, package_name, continue_message=False
        ),
        crashed=target_crashed,
    )
    if recovered and not crashed:
        run.stop("left_app", EXTERNAL_TRANSITION_BLOCKED_END)
    else:
        run.stop(*_recovery_outcome(crashed, recovered))
    if recovered:
        run.capture_final_page()
    return _STEP_STOP, current_package or "", ""


def _handle_screenshot_failure(run: _ScenarioRun, error: Exception) -> str:
    """Restart through foreground polling after repeated capture failures; stop when retry or time budget is exhausted."""
    print(f"[Error] Screenshot failed: {error}")
    run.observation_errors += 1
    if run.observation_errors >= MAX_CONSECUTIVE_ERRORS:
        if run.screenshot_restart_count >= APP_RESTART_ATTEMPTS:
            run.stop(
                "error",
                f"Repeated screenshot failures after {APP_RESTART_ATTEMPTS} app restarts; end scenario: {error}",
            )
            print(f"[Scenario execution] {run.error_message}")
            return _STEP_STOP
        if not run.deadline_allows(APP_RESTART_MAX_SECONDS):
            run.stop("incomplete", f"Repeated screenshot failures with insufficient time to restart: {error}")
            return _STEP_STOP
        # Restart through foreground polling to avoid capturing the launcher during cold start and inducing a restart storm.

        run.screenshot_restart_count += 1
        if _restart_target_app(run.device, run.package_name):
            run.observation_errors = 0
            return _STEP_RETRY
        run.stop("error", f"Repeated screenshot failures and target restart failed: {error}")
        return _STEP_STOP
    if not run.deadline_allows(OBSERVATION_RETRY_DELAY):
        run.stop("incomplete", f"Screenshot failure with insufficient time for retry: {error}")
        return _STEP_STOP
    time.sleep(OBSERVATION_RETRY_DELAY)
    return _STEP_RETRY


def _capture_step_page(run: _ScenarioRun) -> tuple[str, Page | None, str]:
    """Return control result, captured runtime page and screenshot-health error."""
    try:
        page, page_health_error = capture_runtime_page(
            run.device,
            page_id=run.initial_page_id if not run.trace else None,
        )
    except ScreenshotUnavailableError as e:
        print(f"[Scenario execution] Runtime screenshot unavailable; stop immediately: {e}")
        run.stop("screenshot_unavailable", f"Runtime screenshot unavailable; scenario stopped: {e}")
        return _STEP_STOP, None, ""
    except Exception as e:
        return _handle_screenshot_failure(run, e), None, ""
    if page_health_error:
        print(f"[Scenario execution] Unhealthy runtime screenshot: {page_health_error}")
        if run.trace:
            _append_unavailable_observation(
                run.trace, page, page_health_error, run.trace_phase,
            )
            run.stop(
                "screenshot_unavailable",
                f"Black/unavailable page observed; preserved it and ended scenario: {page_health_error}",
            )
            run.final_page_override = page
            return _STEP_STOP, page, page_health_error
    return _STEP_OK, page, page_health_error


def _handle_exploration_boundary(
    run: _ScenarioRun,
    page: Page,
    foreground_package: str,
) -> bool:
    """Exit and stop after an out-of-scope third-party in-app flow. Package-specific configured rules apply; apps without rules retain ordinary behavior."""


    if run.exploration_mode != "core":
        return False
    device, package_name, trace = run.device, run.package_name, run.trace
    boundary_hit = _current_boundary_hit(
        device, package_name, page.activity, foreground_package=foreground_package,
    )
    if boundary_hit is None:
        return False
    boundary_label = boundary_hit.rule.label
    print(
        f"[Exploration boundary] Entered {boundary_label}"
        f"（via={boundary_hit.via}, matched={boundary_hit.matched}），"
        "; exit automatically and end scenario."
    )
    reason = (
        f"Entering {boundary_label} crossed the exploration boundary"
        f"（via={boundary_hit.via}, matched={boundary_hit.matched}）"
    )
    if _recover_from_exploration_boundary(device, package_name, boundary_hit):
        _append_target_app_recovery_step(
            trace, device, package_name, package_name, run.trace_phase,
            recovery_method="press_back", reason=f"{reason}; exited with back.",
        )
        run.stop("blocked", EXPLORATION_BOUNDARY_BLOCKED_END.format(label=boundary_label))
        return True
    print("[Exploration boundary] Back failed; restart target and end scenario.")
    run.crash_logcat = _read_crash_logcat(device, package_name)
    crashed, recovered = _recover_to_target(
        trace, device, package_name, run.trace_phase, package_name, run.crash_logcat,
        back=None, reason=f"{reason}; back failed, so restarted the target.",
    )
    if recovered and not crashed:
        run.stop("left_app", EXPLORATION_BOUNDARY_BLOCKED_END.format(label=boundary_label))
    else:
        run.stop(*_recovery_outcome(crashed, recovered))
    if recovered:
        run.capture_final_page()
    return True


def _observe_step(run: _ScenarioRun) -> tuple[str, _StepObservation | None]:
    """Observe foreground, screenshot, exploration boundary and image/screen dimensions."""
    try:
        control, foreground_package, external_package = _check_target_foreground(run)
    except ScreenshotUnavailableError as e:
        print(f"[Scenario execution] Runtime screenshot unavailable; stop immediately: {e}")
        run.stop("screenshot_unavailable", f"Runtime screenshot unavailable; scenario stopped: {e}")
        return _STEP_STOP, None
    except Exception as e:
        return _handle_screenshot_failure(run, e), None
    if control != _STEP_OK:
        return control, None
    control, page, page_health_error = _capture_step_page(run)
    if control != _STEP_OK:
        return control, None
    if _handle_exploration_boundary(run, page, foreground_package):
        return _STEP_STOP, None
    try:
        img_w, img_h = run.device.display_size()
        image_size = screenshot_dimensions(page.representative_screenshot)
        if image_size is None or min(image_size) <= 1:
            raise ValueError("Current uploaded screenshot cannot be decoded or has invalid dimensions")
    except Exception as e:
        print(f"[Error] Reading screen/image dimensions failed: {e}")
        run.observation_errors += 1
        if run.observation_errors >= MAX_CONSECUTIVE_ERRORS:
            run.stop("error", f"Repeated screen-dimension failures: {e}")
            return _STEP_STOP, None
        if not run.deadline_allows(OBSERVATION_RETRY_DELAY):
            run.stop("incomplete", f"Dimension read failed with insufficient time to retry: {e}")
            return _STEP_STOP, None
        time.sleep(OBSERVATION_RETRY_DELAY)
        return _STEP_RETRY, None
    run.observation_errors = 0
    return _STEP_OK, _StepObservation(
        page=page,
        health_error=page_health_error,
        foreground_package=foreground_package,
        external_context=bool(external_package),
        external_package=external_package,
        screen_size=(img_w, img_h),
        image_size=image_size,
    )


def _handle_generation_failure(run: _ScenarioRun, error_message: str) -> str:
    """Count model-generation failures and back off; stop on exhausted retries or insufficient time."""
    run.llm_errors += 1
    if run.llm_errors >= MAX_CONSECUTIVE_ERRORS:
        run.stop("error", error_message)
        return _STEP_STOP
    if not _sleep_for_llm_retry(run.llm_errors, run.deadline):
        run.stop("incomplete", f"Insufficient phase time for model retry ({error_message})")
        return _STEP_STOP
    return _STEP_RETRY


def _decide_action(
    run: _ScenarioRun,
    system_msg: SystemMessage,
    observation: _StepObservation,
    step_index: int,
) -> tuple[str, dict | None, str, str]:
    """Build input, invoke the model, parse and validate its action. Return control result, plan, raw response and pre-action screenshot reference."""


    page = observation.page
    b64_image = page.representative_screenshot
    trace = run.trace
    previous_step = trace[-1] if trace else {}
    trace_screenshot_before = save_trace_screenshot(
        page,
        phase=run.trace_phase,
        step_index=step_index,
        moment="before_action",
        metadata={
            "foreground_package": observation.foreground_package,
            "target_package": run.package_name,
            "observes_result_of_step": len(trace) if trace else None,
            "previous_action_name": str(previous_step.get("action_name", "") or ""),
            "previous_action_description": str(
                previous_step.get("action_description", "") or ""
            ),
            "screenshot_health_error": observation.health_error,
        },
    )
    current_human_msg = HumanMessage(content=_build_step_input(
        trace,
        b64_image,
        step_index,
        run.step_limit,
        observation.health_error,
        foreground_note=_foreground_state_note(
            observation.foreground_package,
            run.package_name,
        ),
        image_size=observation.image_size,
        recent_failures=run.recent_failures,
    ))
    messages = [system_msg, current_human_msg]
    print(
        "[Scenario execution] LLM input summary: "
        f"phase={run.trace_phase}, step={step_index}, history_steps={len(trace)}, "
        f"recent_screenshots={_recent_screenshot_count(trace)}, "
        f"current_page={_runtime_page_label(page)}, activity={page.activity or 'unknown'}, "
        f"foreground={observation.foreground_package or 'unknown'}, "
        f"current_screenshot={'yes' if b64_image else 'no'}"
    )

    llm = run.llm
    raw_content = ""
    response = None
    request_started = time.monotonic()
    try:
        record_llm_request(llm, operation="[Scenario execution]")
        response = llm.invoke(messages)
        record_llm_response(llm, response, operation="[Scenario execution]")
        raw_content = normalize_llm_text(getattr(response, "content", ""))
        if not raw_content:
            raise ValueError("empty_content: response body is empty")
        if (getattr(response, "response_metadata", {}) or {}).get("finish_reason") == "length":
            raise ValueError("truncated_output: response was truncated")
        plan = _parse_action_json(raw_content)
        plan = _normalize_action_plan(plan)
        _validate_action_plan(plan, observation.image_size)
        record_llm_diagnostic(response, operation="[Scenario execution]", elapsed_seconds=time.monotonic() - request_started)
        log_plan = dict(plan)
        log_plan.pop("action_description", None)
        print(f"[Scenario execution] LLM output JSON: {log_plan}")
    except Exception as e:
        # Unsupported response_format is an endpoint configuration issue: retry without JSON mode, without charging the model error budget or polluting its history.


        if json_format_rejected(e) and disable_json_response_format(llm):
            print(
                "[Scenario execution] Endpoint rejected response_format=json_object. "
                "Disabled JSON mode and retried without charging the error budget."
            )
            return _STEP_RETRY, None, raw_content, trace_screenshot_before
        record_llm_diagnostic(response, operation="[Scenario execution]", error=f"{type(e).__name__}: {e}", elapsed_seconds=time.monotonic() - request_started)
        record_llm_failure(llm, operation="[Scenario execution]")
        no_json_hint = " (response contains no JSON)" if raw_content and "{" not in raw_content else ""
        print(f"[Error] LLM call/parsing failed{no_json_hint}: {e}\n[Raw content]: {raw_content}")
        run.record_generation_failure(
            step_index,
            _describe_generation_failure(raw_content, e, response_received=response is not None),
        )
        control = _handle_generation_failure(run, f"Repeated LLM parsing failures: {e}")
        return control, None, raw_content, trace_screenshot_before
    return _STEP_OK, plan, raw_content, trace_screenshot_before


def _tap_point_on_flat_background(screenshot: str, x: int, y: int) -> bool:
    """Detect low local high-frequency texture near a tap. Subtract Gaussian blur before measuring deviation so smooth gradients do not resemble control edges or text."""


    image = decode_screenshot(screenshot)
    if image is None:
        return False
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    if not (0 <= x < width and 0 <= y < height):
        return False
    y0, y1 = max(0, y - 24), min(height, y + 25)
    x0, x1 = max(0, x - 24), min(width, x + 25)
    patch = gray[y0:y1, x0:x1].astype("float32")
    high_pass = patch - cv2.GaussianBlur(patch, (0, 0), 3)
    window = high_pass[max(0, y - 8) - y0:y + 9 - y0, max(0, x - 8) - x0:x + 9 - x0]
    return float(window.std()) < TAP_REFINE_FLAT_STD


def _refine_tap_point(llm, screenshot: str, plan: dict) -> tuple[int, int] | None:
    """Ask the model to locate the intended control in a magnified crop. Accept only coordinates inside the crop and within TAP_REFINE_MAX_SHIFT; return None on missing targets, errors or suspicious results."""


    image = decode_screenshot(screenshot)
    if image is None:
        return None
    height, width = image.shape[:2]
    x, y = int(round(float(plan["x"]))), int(round(float(plan["y"])))
    x0, y0 = max(0, x - TAP_REFINE_RADIUS), max(0, y - TAP_REFINE_RADIUS)
    x1, y1 = min(width, x + TAP_REFINE_RADIUS), min(height, y + TAP_REFINE_RADIUS)
    crop = cv2.resize(image[y0:y1, x0:x1], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    crop_h, crop_w = crop.shape[:2]
    intent = " ".join(
        str(plan.get(key, "") or "").strip() for key in ("action_description", "thought")
    ).strip()
    question = (
        f"This is a magnified screenshot crop (width={crop_w}, height={crop_h} pixels, origin=(0, 0)). "
        f"Action intent: {intent}. Locate the intended control and return its clickable center "
        "in this crop's pixels. If absent, set found=false. Return only JSON: "
        '{"found": true, "x": 0, "y": 0}'
    )
    try:
        payload = encode_image_to_jpg_base64(crop, max_side=max(crop_w, crop_h), quality=85)
        caller = llm.bind(timeout=TAP_REFINE_TIMEOUT_SECONDS) if hasattr(llm, "bind") else llm
        record_llm_request(llm, operation="[Tap refinement]")
        request_started = time.monotonic()
        response = caller.invoke([HumanMessage(content=[
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{payload}"}},
        ])])
        # Persist refinement responses alongside primary execution calls so diagnostics retain timing and output.

        record_llm_diagnostic(response, operation="[Tap refinement]", elapsed_seconds=time.monotonic() - request_started)
        record_llm_response(llm, response, operation="[Tap refinement]")
        result = _parse_action_json(normalize_llm_text(getattr(response, "content", "")))
    except Exception as exc:  # noqa: BLE001 - optional refinement retains original coordinates on failure
        record_llm_failure(llm, operation="[Tap refinement]")
        record_llm_diagnostic(response=None, operation="[Tap refinement]", error=f"{type(exc).__name__}: {exc}")
        print(f"[Scenario execution] Tap refinement failed; retain original coordinates: {type(exc).__name__}: {exc}")
        return None
    if not isinstance(result, dict) or result.get("found") is False:
        return None
    try:
        crop_x, crop_y = float(result["x"]), float(result["y"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (0 <= crop_x < crop_w and 0 <= crop_y < crop_h):
        return None
    new_x = min(width - 1, max(0, int(round(x0 + crop_x / 2))))
    new_y = min(height - 1, max(0, int(round(y0 + crop_y / 2))))
    if math.hypot(new_x - x, new_y - y) > TAP_REFINE_MAX_SHIFT:
        return None
    return new_x, new_y


def _repeats_ineffective_tap(trace: list[dict], current_screenshot: str, x: int, y: int) -> bool:
    """Identify another tap near the preceding unchanged click, including edge or off-target hits."""
    if not trace:
        return False
    last = trace[-1]
    if str(last.get("action_name", "")).strip() not in {"click", "long_press"}:
        return False
    point = _model_point(last)
    if point is None or abs(point[0] - x) > EFFECT_SAME_POINT_TOLERANCE or abs(point[1] - y) > EFFECT_SAME_POINT_TOLERANCE:
        return False
    before = _page_screenshot(last)
    if not before or not current_screenshot:
        return False
    return _cached_action_effect(before, current_screenshot, None)[0] == "none"


def _maybe_refine_tap(run: "_ScenarioRun", plan: dict, observation: "_StepObservation") -> None:
    """Refine click/long_press coordinates in place when local texture is absent or the preceding nearby tap had no effect. Both trigger checks are model-free."""


    if (
        not TAP_REFINE_ENABLED
        or plan.get("action_name") not in {"click", "long_press"}
        or plan.get("x") is None
        or plan.get("y") is None
    ):
        return
    screenshot = observation.page.representative_screenshot
    try:
        x, y = int(round(float(plan["x"]))), int(round(float(plan["y"])))
    except (TypeError, ValueError):
        return
    if not screenshot:
        return
    if _tap_point_on_flat_background(screenshot, x, y):
        trigger = "tap on a flat background"
    elif _repeats_ineffective_tap(run.trace, screenshot, x, y):
        trigger = "repeated ineffective tap position"
    else:
        return
    refined = _refine_tap_point(run.llm, screenshot, plan)
    if refined is None or refined == (x, y):
        return
    print(f"[Scenario execution] {trigger}; magnified refinement: ({x},{y}) -> ({refined[0]},{refined[1]})")
    plan["x"], plan["y"] = refined
    plan["tap_refined_from"] = {"x": x, "y": y}


def _build_step_info(
    run: _ScenarioRun,
    plan: dict,
    observation: _StepObservation,
    model_coordinates: dict,
    trace_screenshot_before: str,
) -> dict:
    img_w, img_h = observation.screen_size
    image_size = observation.image_size
    step_info = {
        "action_name": plan.get("action_name", ""),
        "thought": plan.get("thought", ""),
        "action_description": str(plan.get("action_description", "")).strip(),
        "current_page": observation.page,
        "is_interruption": bool(plan.get("is_interruption", False)),
        "trace_phase": run.trace_phase,
        "coordinate_space": "screen_pixels",
        "model_coordinate_space": "image_pixels",
        "model_coordinates": model_coordinates,
        "image_size": {"width": image_size[0], "height": image_size[1]},
        "screen_size": {"width": img_w, "height": img_h},
    }
    if trace_screenshot_before:
        step_info["trace_screenshot_before"] = trace_screenshot_before
    if plan.get("tap_refined_from"):
        step_info["tap_refined_from"] = dict(plan["tap_refined_from"])
    if observation.health_error:
        step_info["screenshot_health_error"] = observation.health_error
    if observation.external_context:
        step_info["external_context"] = True
        step_info["external_package"] = observation.external_package
    for field_name in ACTION_PARAMETER_FIELDS:
        if plan.get(field_name) is not None:
            step_info[field_name] = plan[field_name]
    return step_info


def _apply_post_record_guards(
    run: _ScenarioRun,
    plan: dict,
    step_info: dict,
    observation: _StepObservation,
    proposed_action: str,
) -> str:
    """After recording but before dispatch, guard verification challenges, required external pages, stalls, finish verdicts and authenticated sessions. _STEP_OK permits execution."""

    action_name = plan.get("action_name", "")
    thought = plan.get("thought", "")

    def end_step(step_status: str, reason: str) -> None:
        run.stop(step_status, reason)
        step_info["action_failed"] = True
        step_info["action_error"] = reason

    verification_attempt_reason = _verification_attempt_reason(
        action_name,
        thought,
        proposed_action,
    )
    if verification_attempt_reason:
        end_step("blocked", verification_attempt_reason)
        print(f"[Scenario execution] Pre-action guard: {verification_attempt_reason}")
        return _STEP_STOP

    unresolved_external_reason = _unfinished_safe_external_reason(
        action_name,
        observation.foreground_package,
    )
    if unresolved_external_reason:
        step_info["action_failed"] = True
        step_info["action_error"] = unresolved_external_reason
        print(f"[Scenario execution] Rejected premature finish: {unresolved_external_reason}")
        return _STEP_RETRY

    stall_reason = _detect_stall(run.trace)
    if stall_reason:
        print(f"[Warning] {stall_reason}; ending scenario early.")
        end_step("stalled", stall_reason)
        return _STEP_STOP

    if action_name == "finished":
        if observation.health_error:
            end_step("screenshot_unavailable", "Executor finished on a black/unavailable page; defer recovery to next round")
            run.final_page_override = observation.page
            return _STEP_STOP
        verification_finish_reason = _verification_finish_reason(thought)
        if verification_finish_reason:
            end_step("blocked", verification_finish_reason)
            print(f"[Scenario execution] Verification challenge blocks functionality: {thought[:160]}")
        elif _finished_declares_failure(thought):
            end_step("failed_goal", f"Executor declared goal unreachable: {thought[:240]}")
            print(f"[Scenario execution] finished expresses failure; downgraded to failed_goal: {thought[:160]}")
        return _STEP_STOP

    # Inspect the action clause, not the entire thought, which may merely restate session-preservation restrictions.

    session_violation = account_session_action_violation(action_name, proposed_action)
    if session_violation:
        print(
            "[Scenario execution] Blocked action that would invalidate the authenticated session: "
            f"{session_violation} | {(plan.get('action_description') or '')[:80]}"
        )
        end_step(
            "failed_goal",
            "Runtime blocked logout/account-switch action "
            f"({session_violation}); ended scenario with session preserved",
        )
        step_info["session_guard_blocked"] = True
        return _STEP_STOP
    return _STEP_OK


def _execute_planned_action(run: _ScenarioRun, plan: dict, step_info: dict) -> str:
    """Dispatch the action and wait for stabilization according to its type."""
    action_name = plan.get("action_name", "")
    package_name = run.package_name
    if action_name != "wait":
        action_kwargs = {
            key: plan[key]
            for key in ACTION_PARAMETER_FIELDS
            if plan.get(key) is not None
        }
        if action_name == "open_app" and package_name:
            action_kwargs["package_name"] = package_name
        if action_kwargs:
            print(f"[Scenario execution] Action parameters: {action_kwargs}")
        try:
            action_result = _perform_action(device=run.device, action_name=action_name, **action_kwargs)
            if action_name in {"click", "long_press"} and isinstance(action_result, dict):
                for key in ("x", "y"):
                    if key in action_result:
                        step_info[key] = action_result[key]
        except Exception as e:
            print(f"[Error] Action execution failed: {e}")
            step_info["is_interruption"] = True
            step_info["action_failed"] = True
            step_info["action_error"] = str(e)
            if run.action_failure_recoveries < MAX_ACTION_FAILURE_RECOVERIES:
                run.action_failure_recoveries += 1
                print(
                    "[Scenario execution] Return the action failure to the model; allow one alternative action."
                )
                return _STEP_RETRY
            run.stop("error", f"Action and alternative strategy both failed; scenario ended: {e}")
            return _STEP_STOP

    if action_name == "wait":
        # Honor explicit wait durations to avoid repeated short waits being classified as stalls.

        wait_seconds = max(
            MIN_WAIT_SECONDS,
            min(MAX_WAIT_SECONDS, float(plan.get("duration") or DEFAULT_WAIT_SECONDS)),
        )
        if wait_seconds > ACTION_DELAY:
            print(f"[Scenario execution] Explicit wait {wait_seconds:.1f}s for page loading")
        time.sleep(wait_seconds)
    elif action_name == "open_app" and package_name:
        # Poll after cold start rather than showing a premature launcher/loading screenshot to the model.

        _wait_target_app_foreground(
            run.device,
            package_name,
            max_wait=min(APP_START_MAX_WAIT, 8.0),
        )
        time.sleep(APP_RESTART_SETTLE_DELAY)
    elif action_name in FAST_ACTIONS:
        time.sleep(FAST_ACTION_DELAY)
    else:
        time.sleep(ACTION_DELAY)
    return _STEP_OK


def _run_scenario_step(run: _ScenarioRun, system_msg: SystemMessage) -> str:
    """Run observation, decision, guards and execution; return the control result."""
    control, observation = _observe_step(run)
    if control != _STEP_OK:
        return control

    step_index = len(run.trace) + 1
    control, plan, raw_content, trace_screenshot_before = _decide_action(
        run, system_msg, observation, step_index,
    )
    if control != _STEP_OK:
        return control

    if run.deadline is not None and time.time() >= run.deadline:
        run.stop("incomplete", "Phase budget expired during model response; suggested action was not executed")
        return _STEP_STOP
    action_name = plan.get("action_name", "")
    thought = plan.get("thought", "")
    is_interruption = bool(plan.get("is_interruption", False))

    proposed_action = _proposed_action_text(plan)
    risky_action_intent = wechat_action_policy_violation(
        run.package_name,
        action_name,
        proposed_action,
    )
    if risky_action_intent:
        run.stop("blocked", f"{WECHAT_READ_ONLY_BLOCKED_END}: {risky_action_intent}")
        print(f"[Scenario execution] Pre-action guard: {run.error_message}")
        return _STEP_STOP

    print(
        f"\n[Step {step_index}]{' [Interruption]' if is_interruption else ''} "
        f"page={observation.page.id or 'unknown'} action={action_name} | Assessment: {thought}"
    )

    _maybe_refine_tap(run, plan, observation)
    model_coordinates = {
        key: plan[key] for key in (*COORDINATE_FIELDS, *SWIPE_COORDINATE_FIELDS)
        if key in plan
    }
    img_w, img_h = observation.screen_size
    try:
        _scale_plan_coordinates(plan, img_w, img_h, observation.image_size)
    except ValueError as e:
        print(f"[Error] LLM coordinate parsing failed: {e}\n[Raw content]: {raw_content}")
        run.record_generation_failure(step_index, f"Coordinate conversion failed; action not executed: {str(e)[:300]}")
        return _handle_generation_failure(run, f"Repeated LLM coordinate parsing failures: {e}")

    # A valid action clears generation failures so stale warnings do not affect later decisions.

    run.llm_errors = 0
    run.recent_failures.clear()
    step_info = _build_step_info(
        run, plan, observation, model_coordinates, trace_screenshot_before,
    )
    run.trace.append(step_info)

    control = _apply_post_record_guards(run, plan, step_info, observation, proposed_action)
    if control != _STEP_OK:
        return control
    return _execute_planned_action(run, plan, step_info)


def _finalize_scenario(run: _ScenarioRun) -> None:
    """Exit blocked pages, observe the last result, restore target foreground and persist screenshots."""
    device, package_name, trace, trace_phase = run.device, run.package_name, run.trace, run.trace_phase
    if run.status in {"blocked", "stalled"}:
        blocked_exit_page = _exit_blocked_page(trace, device, trace_phase)
        if blocked_exit_page is not None:
            run.final_page_override = blocked_exit_page

    if run.status not in FOREGROUND_HANDLED_STATUSES:
        terminal_page, terminal_health_error = _append_terminal_observation(
            trace,
            device,
            trace_phase,
        )
        if terminal_page is not None:
            run.final_page_override = terminal_page
        if terminal_health_error:
            run.stop(
                "screenshot_unavailable",
                "Page unavailable after the last side-effecting action; scenario ended: "
                f"{terminal_health_error}",
            )

    # Restore foreground at scenario end, except when the time budget is exhausted.
    if package_name and run.status not in FOREGROUND_HANDLED_STATUSES and run.status != "incomplete":
        foreground_state, current_package = _target_app_foreground_state(device, package_name)
        if foreground_state != "foreground":
            scene_end_logcat = _read_crash_logcat(device, package_name)
            crashed, recovered = _recover_to_target(
                trace, device, package_name, trace_phase, current_package or "unknown",
                scene_end_logcat,
                back=lambda: _recover_target_app_by_back(
                    device, package_name, continue_message=False
                ),
                final_recovery=True,
            )
            if crashed:
                run.crash_logcat = run.crash_logcat or scene_end_logcat
            if crashed or not recovered:
                run.stop(*_recovery_outcome(crashed, recovered))

    supplemental_screenshots = _persist_unwritten_trace_screenshots(trace, trace_phase)
    print(
        "[Trace screenshots] Scenario screenshot indexing complete: "
        f"phase={trace_phase}, trace_steps={len(trace)}, "
        f"supplemental_saved={supplemental_screenshots}"
    )


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
    compliance_context: dict[str, Any] | None = None,
    deadline: float | None = None,
    task_phase: str = "",
) -> tuple[list[dict], Page | None, str, str, str]:
    """Return (trace, final_page, status, error, crash_logcat). Each step observes, decides, applies guards and executes; see EXECUTION_STATUSES for outcomes."""


    phase_label = {"navigation": "navigation", "test": "test"}.get(
        trace_phase, trace_phase
    )
    step_limit = (
        MAX_NAVIGATION_STEPS if trace_phase == "navigation" else MAX_EXEC_STEPS
    )
    if not scenario:
        print(f"[Scenario execution] Skipped {phase_label}: no scenario configured.")
        return [], initial_page, "skipped", "", ""
    if deadline is not None and time.time() >= deadline:
        return [], initial_page, "incomplete", "Phase time budget exhausted", ""

    scene_started_at = time.time()
    compliance_cold_start = bool(
        (compliance_context or {}).get("cold_start")
    ) and trace_phase == "test"
    start_failure = _prepare_scenario_start(
        device, package_name, scenario, clear_logcat, trace_phase,
        compliance_cold_start, deadline,
    )
    if start_failure is not None:
        return [], initial_page, start_failure[0], start_failure[1], ""
    target_pid = _target_process_pid(device, package_name)
    print(
        "[Scenario execution] Scenario start: "
        f"phase={trace_phase}, target={package_name or 'unknown'}, "
        f"pid={target_pid or 'unknown'}, started_at={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(scene_started_at))}"
    )

    system_prompt = SCENARIO_EXEC_SYSTEM_PROMPT_TEMPLATE.format(
        # Keep the fixed mode addendum before dynamic task information for prefix caching.
        mode_addendum=COMPLIANCE_EXEC_ADDENDUM if compliance_context else "",
        app_name=app_name,
        package_name=package_name,
        task_phase=_task_phase_label(task_phase),
        scenario=scenario,
        max_exec_steps=step_limit,
    )
    system_msg = SystemMessage(content=system_prompt)
    allows_external_app = _scenario_allows_external_app(scenario)
    if allows_external_app:
        print("[Scenario execution] Controlled external transitions allowed; restore target at scenario end.")
    allows_target_exit = _scenario_allows_target_exit(scenario)
    if allows_target_exit:
        print("[Scenario execution] Target may leave foreground; restore it at scenario end.")

    run = _ScenarioRun(
        llm=llm,
        device=device,
        package_name=package_name,
        trace_phase=trace_phase,
        step_limit=step_limit,
        deadline=deadline,
        exploration_mode=exploration_mode,
        allows_external_app=allows_external_app,
        allows_target_exit=allows_target_exit,
        initial_page_id=initial_page.id if isinstance(initial_page, Page) else None,
    )

    print(f"[Scenario execution] Start {phase_label}: {scenario}")
    if isinstance(initial_page, Page):
        print(f"[Scenario execution] Initial page: {initial_page.id or 'unknown'}")

    while True:
        if deadline is not None and time.time() >= deadline:
            run.stop("incomplete", "Phase time budget exhausted; stop new actions and save final observation")
            break
        if len(run.trace) >= step_limit:
            print(f"[Warning] Maximum step limit ({step_limit}) reached; stopping.")
            run.stop("incomplete", f"Maximum step limit {step_limit} reached")
            break
        if _run_scenario_step(run, system_msg) == _STEP_STOP:
            break

    _finalize_scenario(run)

    trace = run.trace
    final_page = run.final_page_override or (trace[-1]["current_page"] if trace else initial_page)
    interruption_count = sum(1 for s in trace if s.get("is_interruption"))
    final_page_id = (
        str(final_page.id or "unknown")
        if isinstance(final_page, Page)
        else "unknown"
    )
    reason = f" | reason={_compact_execution_log_text(run.error_message)}" if run.error_message else ""
    print(
        f"[Scenario execution] {phase_label} finished | status={run.status} | "
        f"steps={len(trace)} | interruptions={interruption_count} | "
        f"page={final_page_id}{reason}"
    )
    return trace, final_page, run.status, run.error_message, run.crash_logcat


def create_scenario_execution_pass(execution_llm, device):
    """Create the navigation + test execution pass used by the exec subgraph."""

    def execute_scenarios(state: AgentState) -> dict:
        limit = float(state.get("total_time_limit", 0) or 0)
        deadline = float(state.get("start_time", time.time())) + limit if limit > 0 else None
        app_name = state.get("app_name", "")
        package_name = state.get("package_name", "")
        exploration_mode = state.get("exploration_mode", "core")
        navigation_scenario = state.get("current_navigation_scenario", "")
        test_scenario = state.get("current_test_scenario", "")
        current_page = state.get("current_page")
        current_plan = state.get("current_plan", {}) or {}
        scenario_type = current_plan.get("scenario_type", "")
        navigation_failure_count = int(state.get("navigation_failure_count", 0) or 0)
        unavailable_count = _screenshot_unavailable_count(state)

        def execution_result(
            navigation_status: str,
            navigation_error: str,
            navigation_trace: list[dict],
            scenario_status: str = "skipped",
            scenario_error: str = "",
            test_trace: list[dict] | None = None,
            current_scenario: str = "",
            crash_logcat: str = "",
        ) -> dict[str, Any]:
            test_trace = test_trace or []
            return {
                "navigation_status": navigation_status,
                "navigation_error": navigation_error,
                "navigation_trace_length": len(navigation_trace),
                "scenario_status": scenario_status,
                "scenario_error": scenario_error,
                "scenario_trace_length": len(test_trace),
                "current_scenario": current_scenario,
                "scenario_type": scenario_type,
                "crash_logcat": crash_logcat,
                "crash_detected": "app_crashed" in {navigation_status, scenario_status},
                "recovered_to_target_app": any(
                    step.get("recovered_to_target_app")
                    for step in (*navigation_trace, *test_trace)
                ),
            }

        def handle_unavailable(trace: list[dict], trace_phase: str, page):
            """Count consecutive unavailable screenshots, restart at the threshold, and optionally terminate the run if recovery still fails."""
            count = unavailable_count + 1
            restart_error = ""
            if count >= MAX_CONSECUTIVE_SCREENSHOT_UNAVAILABLE:
                reset_page, restart_error = _restart_after_repeated_black_screen(
                    trace, device, package_name, trace_phase,
                )
                if reset_page is not None:
                    page, count = reset_page, 0
            workflow_complete = (
                STOP_AFTER_SCREENSHOT_UNAVAILABLE
                and count >= MAX_CONSECUTIVE_SCREENSHOT_UNAVAILABLE
            )
            print(
                f"[Scenario execution] Screenshot unavailable in {trace_phase}; "
                f"{'end current app run' if workflow_complete else 'end scenario and explore other functionalities'}: "
                f"{count}/{MAX_CONSECUTIVE_SCREENSHOT_UNAVAILABLE}"
            )
            return page, count, restart_error, workflow_complete

        if package_name and not _ensure_target_app(device, package_name):
            return {
                "navigation_execution_trace": [],
                "scenario_execution_trace": [],
                "navigation_failure_count": navigation_failure_count + 1,
                "scenario_exec_error_count": 0,
                "execution_result": execution_result("error", "Target app could not be launched", []),
            }

        (
            navigation_trace,
            nav_final_page,
            navigation_status,
            navigation_error,
            navigation_crash_logcat,
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
            deadline=deadline,
        )

        # If navigation did not finish, skip testing and retain the navigation outcome.
        if navigation_scenario and navigation_status != "completed":
            final_page = nav_final_page or current_page
            workflow_complete = False
            if navigation_status == "screenshot_unavailable":
                final_page, unavailable_count, restart_error, workflow_complete = handle_unavailable(
                    navigation_trace, "navigation", final_page,
                )
                navigation_error = restart_error or navigation_error
                navigation_failure_count = 0
            elif navigation_status in {"left_app", "app_crashed"}:
                # Recovery already occurred during execution; do not count another navigation failure.
                navigation_failure_count = 0
            else:
                navigation_failure_count += 1
                print(f"[Scenario execution] Navigation failed; skip test scenario. Consecutive failures: {navigation_failure_count}")
                if navigation_failure_count >= 2:
                    print("[Scenario execution] Two consecutive navigation failures; restart app at its initial page.")
                    if package_name and _restart_target_app(device, package_name):
                        try:
                            final_page = capture_current_page(device)
                        except Exception as e:
                            print(f"[Scenario execution] Capture after restart failed: {e}")
                    navigation_failure_count = 0
            updates = {
                "current_page": final_page,
                "current_test_scenario": "",
                "navigation_execution_trace": navigation_trace,
                "scenario_execution_trace": [],
                "navigation_failure_count": navigation_failure_count,
                "scenario_exec_error_count": 0,
                "execution_result": execution_result(
                    navigation_status,
                    navigation_error,
                    navigation_trace,
                    scenario_error=navigation_error,
                    current_scenario=navigation_scenario,
                    crash_logcat=navigation_crash_logcat,
                ),
            }
            if navigation_status == "screenshot_unavailable":
                updates["screenshot_unavailable_count"] = unavailable_count
                updates["workflow_complete"] = workflow_complete
            return updates

        test_initial_page = nav_final_page if isinstance(nav_final_page, Page) else current_page
        compliance_context = None
        if str(scenario_type).strip() == "compliance_check":
            compliance_context = {
                "cold_start": bool(current_plan.get("compliance_cold_start")),
                "check_ids": [
                    str(item).strip()
                    for item in current_plan.get("compliance_check_ids", []) or []
                    if str(item).strip()
                ],
            }
            print(
                "[Scenario execution] Compliance scenario: "
                f"checks={len(compliance_context['check_ids'])}, "
                f"cold_start={compliance_context['cold_start']}"
            )
        (
            test_trace,
            test_final_page,
            scenario_status,
            scenario_error,
            test_crash_logcat,
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
            compliance_context=compliance_context,
            deadline=deadline,
            task_phase=str(current_plan.get("function_phase", "") or ""),
        )
        final_page = test_final_page or nav_final_page or current_page
        workflow_complete = False
        if scenario_status == "screenshot_unavailable":
            final_page, unavailable_count, restart_error, workflow_complete = handle_unavailable(
                test_trace, "test", final_page,
            )
            scenario_error = restart_error or scenario_error
        else:
            unavailable_count = 0
        return {
            "current_page": final_page,
            "navigation_execution_trace": navigation_trace,
            "scenario_execution_trace": test_trace,
            "navigation_failure_count": 0,
            "scenario_exec_error_count": 0,
            "workflow_complete": workflow_complete,
            "screenshot_unavailable_count": unavailable_count,
            "execution_result": execution_result(
                navigation_status,
                navigation_error,
                navigation_trace,
                scenario_status,
                scenario_error,
                test_trace,
                current_scenario=test_scenario,
                crash_logcat=test_crash_logcat,
            ),
        }

    return execute_scenarios
