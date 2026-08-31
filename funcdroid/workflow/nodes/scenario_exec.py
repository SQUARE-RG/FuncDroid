"""场景执行节点。"""

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
from workflow.wechat_safety import (
    wechat_action_policy_violation,
    wechat_read_only_violation,
)

JSON_BLOCK_PATTERN = re.compile(r"^```(?:json)?\s*|\s*```$")


def _compact_execution_log_text(value: object, limit: int = 200) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


KEEP_IMAGE_STEPS = int(os.getenv("EXEC_KEEP_IMAGE_STEPS", "3"))
# 场景执行（test 阶段）单轮步数上限；导航阶段使用独立的 MAX_NAVIGATION_STEPS。
MAX_EXEC_STEPS = max(1, int(os.getenv("MAX_EXEC_STEPS", "15")))
MAX_NAVIGATION_STEPS = max(1, int(os.getenv("MAX_NAVIGATION_STEPS", "10")))
MAX_CONSECUTIVE_ERRORS = 3
# 单轮场景内允许的模型错误总数（含超时）。仅限制累计次数，不改变连续失败的判定。
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
# test 阶段同方向连续滚动的硬预算：达到该次数后必须转向其他探索，
# 继续同方向滚动会被 _detect_stall 提前拦截。
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
# 方向无关、动作无关的兜底停滞判据：在同一 Activity 内反复试探却始终在
# 少数几个视觉状态之间打转时提前收尾，避免把 MAX_EXEC_STEPS 走满。
NO_PROGRESS_STALL_WINDOW = max(
    3,
    min(int(os.getenv("EXEC_NO_PROGRESS_STALL_WINDOW", "6")), MAX_EXEC_STEPS),
)
NO_PROGRESS_MAX_VISUAL_STATES = max(
    1,
    int(os.getenv("EXEC_NO_PROGRESS_MAX_VISUAL_STATES", "2")),
)
# 聚类同一视觉状态时用比重复动作判定更宽松的阈值：这里要回答的是
# “界面有没有实质变化”，而不是“两张截图是否逐像素一致”。
NO_PROGRESS_SIMILARITY_THRESHOLD = float(
    os.getenv("EXEC_NO_PROGRESS_SIMILARITY", "0.985")
)
NO_PROGRESS_STALL_ACTIONS = {"click", "long_press", "scroll", "press_back", "wait"}
APP_RECOVERY_BACK_ATTEMPTS = max(1, int(os.getenv("APP_RECOVERY_BACK_ATTEMPTS", "3")))
APP_RESTART_ATTEMPTS = max(1, int(os.getenv("APP_RESTART_ATTEMPTS", "2")))
APP_RESTART_RETRY_DELAY = float(os.getenv("APP_RESTART_RETRY_DELAY", "1.0"))
# 冷启动等待：固定 sleep 改为轮询前台包名。微信等重型应用冷启动常超过
# APP_RESTART_DELAY（2s），固定等待会截到桌面/启动页截图，诱发执行模型
# 误判并重复 open_app（历史上形成 force-stop 重启风暴）。
APP_START_POLL_INTERVAL = max(0.1, float(os.getenv("APP_START_POLL_INTERVAL", "0.5")))
APP_START_MAX_WAIT = max(1.0, float(os.getenv("APP_START_MAX_WAIT", "12")))
APP_RESTART_SETTLE_DELAY = max(0.0, float(os.getenv("APP_RESTART_SETTLE_DELAY", "1.0")))
# open_app 重复窗口：每次 open_app 都可能触发真实重启，比普通动作代价高，
# 连续 2 次即提前拦截。
OPEN_APP_STALL_REPEAT_WINDOW = 2
# 前台是桌面（launcher）时按返回键无效，恢复流程直接跳到重启。
LAUNCHER_PACKAGE_PATTERN = re.compile(r"launcher|home", re.IGNORECASE)
TARGET_APP_RESTARTED_END = "目标应用离开前台且无法通过返回恢复，已重启并结束本轮场景"
TARGET_APP_CRASHED_END = "检测到目标应用崩溃，已重启目标应用并结束当前场景"
TARGET_APP_CRASH_RECOVERY_FAILED_END = "检测到目标应用崩溃，但重启后仍未恢复到目标应用，已结束当前场景"
TARGET_APP_RECOVERY_FAILED_END = "目标应用离开前台，自动返回和重启均未能恢复，已结束本轮场景"
EXTERNAL_TRANSITION_BLOCKED_END = "目标应用跳转到未受控外部应用，已恢复目标应用并结束当前场景"
# 探索边界（core 模式）结束文案模板：{label} 来自边界规则配置，如"小程序/第三方服务"。
EXPLORATION_BOUNDARY_BLOCKED_END = "检测到进入{label}（非目标应用核心功能），已自动返回并结束当前场景"
WECHAT_READ_ONLY_BLOCKED_END = "微信账号安全策略拦截了退出登录或删除账户操作"
VERIFICATION_BLOCKED_END = "验证码或人机验证无法由无人值守测试完成，已退出当前阻塞页面"
FOREGROUND_CHECK_RETRIES = max(1, int(os.getenv("FOREGROUND_CHECK_RETRIES", "2")))
FOREGROUND_CHECK_RETRY_DELAY = float(os.getenv("FOREGROUND_CHECK_RETRY_DELAY", "0.2"))
# 前台状态复检：页面切换动画期间 current_package 可能瞬时上报桌面
# launcher（应用自身过渡态），单次立即检测会把过渡态误判为外部跳转
# （external_transition_blocked）。非前台结论必须经短暂等待复检确认。
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
    "目标不可达",
    "当前不可达",
    "任务无法",
    "无法完成",
    "无法继续",
    "无法切换",
    "无法找到",
    "无法进入",
    "未找到",
    "未成功",
    "不能继续",
    "均未成功",
    "均无法",
    "not able",
    "unable",
    "cannot",
    "can't",
    "unreachable",
    "not found",
    "failed to",
)
FINISHED_OVERRIDE_FAILURE_PATTERNS = (
    r"(替代|更换).{0,12}(入口|路径|控件|按钮)",
    r"(原|指定).{0,20}(未找到|未成功|未完成|无法|不能|不符合)",
    r"(instead|alternate|alternative).{0,20}(original|specified|requested)",
)
FINISHED_SUCCESS_PATTERNS = (
    r"目标已达成",
    r"任务目标已达成",
    r"(所有|全部).{0,8}(任务)?目标.{0,4}(满足|达成|完成)",
    r"当前任务已完成",
    r"(所有|全部).{0,16}(验证项|检查项).{0,8}(均已确认|已确认|完成)",
    r"已完成.{0,12}(所有|全部)",
    r"测试任务已完成",
    r"验证了.{0,40}正常",
    r"(所有|全部).{0,12}验证(点|要求|目标).{0,12}(满足|达成|完成)",
    r"功能正常",
    r"(主要|核心).{0,16}(验证点|目标).{0,20}(覆盖|达成|满足|完成|已验证)",
    r"\bgoal\b.{0,30}\b(achieved|complete|completed)\b",
    r"\btask\b.{0,30}\b(complete|completed)\b",
)
MIXED_FINISH_FAILURE_PATTERN = re.compile(
    r"(?:目标已达成|任务已完成|已完成可执行部分).{0,120}"
    r"(?:不可达|未完成|未验证|未找到|未执行|无法进入)"
    r"|(?:不可达|未完成|未验证|未找到|未执行|无法进入).{0,120}"
    r"(?:目标已达成|任务已完成|已完成可执行部分)",
    re.IGNORECASE,
)
VERIFICATION_CHALLENGE_PATTERN = re.compile(
    r"验证码|图形验证|人机验证|安全验证|滑块|拼图|captcha|verification\s+challenge",
    re.IGNORECASE,
)
VERIFICATION_ATTEMPT_PATTERN = re.compile(
    r"识别|拖动|滑动|对齐|拼合|提交|完成|通过|solve|drag|swipe|submit|verify",
    re.IGNORECASE,
)
VERIFICATION_EXIT_PATTERN = re.compile(
    r"关闭|取消|返回|退出|close|cancel|back|exit",
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
    "寻找",
    "找到",
    "未显示",
    "未看到",
    "没看到",
    "目标",
    "露出",
    "按钮",
    "控件",
    "入口",
    "visible",
    "find",
    "target",
    "not found",
)
EXTERNAL_APP_SCENARIO_PATTERNS = (
    r"(点击|选择).{0,20}(应用项|应用列表项|app item)",
    # “目标应用”是被测应用本身，常出现在导航恢复文案中，不能据此放行外部应用。
    r"(启动|打开|跳转到|进入).{0,30}(外部应用|其他应用|另一个应用|对应应用|应用程序)",
    r"(启动|打开).{0,30}(Calculator|Calendar|Camera|Clock|Contacts|Files|Gmail|Maps|Messages|Phone|Photos)",
    r"(返回|回到).{0,30}(launcher|启动器|应用列表|TextLauncher)",
    r"(系统)?(文件|文档).{0,20}(选择器|管理器|保存|打开|导入|创建)",
    r"(打开|进入|前往).{0,24}(系统设置|系统权限页|权限管理页|相机|相册)",
    r"(?:在|使用|通过)?\s*(?:系统|外部|默认)?\s*(?:浏览器|browser).{0,16}(?:中)?\s*(?:打开|查看)",
    r"(?:打开|进入|跳转到).{0,24}(?:浏览器|browser)",
    r"(系统权限|定位权限|相机权限|相册权限).{0,30}(设置|授权|允许|管理|选择)",
    # 单独的“路径”也会命中“打开的菜单属于导航路径”这类通用导航提示。
    # 文件和目录已经覆盖真实的系统选择器场景，不需要这个过宽关键词。
    r"(选择|导入|保存|创建|打开).{0,30}(文件|文档|目录|TiddlyWiki)",
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
    r"(退出|关闭).{0,20}(应用|app|App|程序)",
    r"(返回|回到|退回).{0,30}(系统桌面|桌面|launcher|启动器|前一个应用)",
    r"(验证|测试).{0,30}(退出逻辑|退出功能|关闭逻辑)",
    r"\b(exit|quit|close)\b.{0,20}\b(app|application)\b",
)


def _parse_coordinate_value(value, key: str) -> float | None:
    """Parse an LLM coordinate value while tolerating JSON numeric strings."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{key} 坐标不能是布尔值: {value!r}")
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            number = float(stripped)
        except ValueError as exc:
            raise ValueError(f"{key} 坐标不是数字: {value!r}") from exc
    else:
        raise ValueError(f"{key} 坐标类型不支持: {type(value).__name__}")

    if not math.isfinite(number):
        raise ValueError(f"{key} 坐标不是有限数字: {value!r}")
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
        raise ValueError(f"不支持或缺失的操作类型: {action_name!r}")
    if action_name in {"click", "long_press"} and (
        plan.get("x") is None or plan.get("y") is None
    ):
        raise ValueError(f"{action_name} 缺少 x/y 坐标")
    if action_name in {"click", "long_press"}:
        for key in COORDINATE_FIELDS:
            number = _parse_coordinate_value(plan.get(key), key)
            if number is None or not 0 <= number <= 1000:
                raise ValueError(f"{key} 坐标必须在 0–1000 范围内: {plan.get(key)!r}")
    if action_name == "type" and not str(plan.get("content", "")):
        raise ValueError("type 缺少 content")
    if action_name == "scroll":
        direction = str(plan.get("direction", "")).strip().lower()
        if direction not in {"up", "down", "left", "right"}:
            raise ValueError(f"scroll direction 非法: {direction!r}")
    if action_name == "open_app" and not str(plan.get("app_name", "")).strip():
        raise ValueError("open_app 缺少 app_name")
    if not isinstance(plan.get("is_interruption", False), bool):
        raise ValueError("is_interruption 必须是 boolean")


_ACTION_CLAUSE_MARKERS = re.compile(
    r"下一步|接下来|需要|准备|将要|立即|点击|长按|输入|滚动|打开|进入|选择|发送|切换|开启|关闭|删除|标记|置顶|返回"
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
    if re.search(r"验证成功|已通过验证|成功通过|verification\s+succeeded", text, re.IGNORECASE):
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
            print(f"[场景执行] 获取当前前台包名失败: {exc}")
        if attempt < retries:
            time.sleep(FOREGROUND_CHECK_RETRY_DELAY)
    if last_error is None:
        print("[场景执行] 当前前台包名为空，无法确认目标应用状态。")
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
    """短暂等待后复检前台状态，过滤页面切换动画期间的过渡态误报。

    实测（微信服务号卡片跳转）：点击后 current_package 瞬时上报桌面
    launcher，最终稳定在目标应用内的新 Activity（webview 等）；单次
    立即检测会把过渡态误判为外部跳转（external_transition_blocked），
    白白终止场景并给功能记一次失败。
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
    """轮询等待目标应用进入前台，代替固定时长的冷启动等待。"""
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
            print(f"[场景执行] 第 {attempt}/{APP_RESTART_ATTEMPTS} 次重启目标应用。")
            device.restart_app(package_name)
            if _wait_target_app_foreground(device, package_name):
                time.sleep(APP_RESTART_SETTLE_DELAY)
                return True
            print("[场景执行] 重启后目标应用仍未进入前台。")
        except Exception as exc:
            print(f"[场景执行] 重启目标应用失败: {exc}")
        if attempt < APP_RESTART_ATTEMPTS:
            time.sleep(APP_RESTART_RETRY_DELAY)
    return False


def _ensure_target_app(device, package_name: str) -> bool:
    foreground_state, current_package = _target_app_foreground_state(device, package_name)
    if foreground_state == "foreground":
        return True

    if foreground_state == "unknown":
        print(f"[场景执行] 无法确认当前前台应用，目标应用为 {package_name}，准备重启目标应用。")
    else:
        print(
            f"[场景执行] 检测到当前前台应用为 {current_package}，目标应用为 {package_name}，"
            "准备重启目标应用。"
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
        # 桌面上按返回键既回不到目标应用，也可能误触系统手势；
        # 直接放弃返回恢复，让调用方走重启路径。
        print(
            "[场景执行] 当前前台为桌面（launcher），返回键无法恢复目标应用，"
            "跳过返回恢复直接尝试重启。"
        )
        return False

    for attempt in range(1, attempts + 1):
        if foreground_state == "unknown":
            print(f"[场景执行] 无法确认目标应用是否在前台，等待后第 {attempt}/{attempts} 次重查。")
        else:
            try:
                print(f"[场景执行] 目标应用离开前台，尝试第 {attempt}/{attempts} 次返回恢复。")
                _perform_action(device=device, action_name="press_back")
            except Exception as exc:
                print(f"[场景执行] 返回恢复动作失败: {exc}")
        time.sleep(FAST_ACTION_DELAY)
        foreground_state, _ = _target_app_foreground_state(device, package_name)
        if foreground_state == "foreground":
            message = "继续执行当前场景。" if continue_message else "准备结束当前场景。"
            print(f"[场景执行] 已通过返回恢复到目标应用，{message}")
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
        print(f"[场景执行] 返回恢复后抓取页面失败: {exc}")
        page = trace[-1].get("current_page") if trace else None

    action_name = "press_back" if recovery_method == "press_back" else "restart_app"
    recovery_text = "返回键" if recovery_method == "press_back" else "重启目标应用"
    scene_text = "场景结束时" if final_recovery else "执行过程中"
    thought = (
        f"{scene_text}检测到目标应用离开前台，运行器已自动通过{recovery_text}恢复到目标应用。"
        "这是测试环境恢复步骤，不代表用户场景内的有效操作。"
    )
    trace.append({
        "action_name": action_name,
        "thought": thought,
        "action_description": "目标应用离开前台后的自动恢复",
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
                f"目标应用离开前台: current={previous_package or '未知'}, "
                f"target={package_name or '未知'}；已通过{recovery_text}恢复。"
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
        print(f"[场景执行] 自动恢复前抓取外部页面失败: {exc}")
        return
    trace.append({
        "action_name": "capture_screenshot",
        "thought": "运行器在自动返回前保留外部前台页面，供执行轨迹审计。",
        "action_description": "自动恢复前的外部页面观测",
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
        "[场景执行] 已保存自动恢复前的外部页面观测: "
        f"current={external_package or '未知'}, target={package_name or '未知'}"
    )


def _recover_from_exploration_boundary(
    device,
    package_name: str,
    hit: BoundaryHit,
) -> bool:
    """自动退出越过探索边界的页面（如小程序），返回是否已离开边界。

    与外部应用恢复不同：边界页面仍运行在目标应用包内（前台包名不变），
    因此按 Activity 名复检是否已退出边界，而非按包名复检。
    """
    attempts = max(1, APP_RECOVERY_BACK_ATTEMPTS)
    for attempt in range(1, attempts + 1):
        try:
            print(
                f"[探索边界] 尝试第 {attempt}/{attempts} 次 {hit.rule.exit_action} "
                f"退出{hit.rule.label}（via={hit.via}, matched={hit.matched}）。"
            )
            _perform_action(device=device, action_name="press_back")
        except Exception as exc:
            print(f"[探索边界] 退出动作失败: {exc}")
        time.sleep(FAST_ACTION_DELAY)
        try:
            current_activity = str(device.current_activity() or "")
        except Exception as exc:
            print(f"[探索边界] 读取当前 Activity 失败: {exc}")
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

    previous_package = current_package or "未知"
    print(
        "[场景执行] 场景结束时仍不在目标应用，准备恢复: "
        f"status={foreground_state}, current={previous_package}, target={package_name or '未知'}"
    )
    crash_logcat = _read_crash_logcat(device, package_name)
    if _logcat_has_target_process_crash(crash_logcat, package_name):
        print("[场景执行] 场景结束时检测到目标应用崩溃，重启目标应用供后续场景继续。")
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
        print("[场景执行] 未获取到设备序列号，跳过清空 logcat。")
        return
    try:
        subprocess.run(
            ["adb", "-s", device_serial, "logcat", "-c"],
            timeout=5,
            capture_output=True,
        )
        print(f"[场景执行] 已清空 logcat: device={device_serial}")
    except Exception as exc:
        print(f"[场景执行] 清空 logcat 失败: {exc}")


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
        print(f"[场景执行] 读取 crash logcat 失败: {exc}")
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
        return "未知页面"
    return page.name or page.description or page.activity or "未知页面"


def _format_execution_history(trace: list[dict]) -> str:
    if not trace:
        return "暂无历史步骤。"

    lines = []
    for index, step in enumerate(trace, start=1):
        page = step.get("current_page")
        action = str(step.get("action_name", "")).strip() or "未知动作"
        thought = str(step.get("thought", "")).strip()
        content = str(step.get("content", "")).strip()
        direction = str(step.get("direction", "")).strip()
        detail_parts = []
        if step.get("external_context"):
            external_package = str(step.get("external_package", "")).strip() or "外部应用"
            detail_parts.append(f"外部页面: {external_package}")
        if step.get("caused_external_transition"):
            external_package = str(step.get("external_package", "")).strip() or "外部应用"
            detail_parts.append(f"触发外部跳转: {external_package}")
        if step.get("is_interruption"):
            detail_parts.append("干扰/恢复步骤")
        if step.get("foreground_recovery"):
            previous_package = str(step.get("previous_package", "")).strip() or "未知"
            target_package = str(step.get("target_package", "")).strip() or "目标应用"
            detail_parts.append(f"自动返回恢复: {previous_package} -> {target_package}")
        if content:
            detail_parts.append(f"输入/文本: {content}")
        if direction:
            detail_parts.append(f"方向: {direction}")
        if step.get("action_failed"):
            action_error = str(step.get("action_error", "")).strip() or "设备动作执行失败"
            detail_parts.append(f"动作失败: {action_error[:200]}")
        detail = f" | {'; '.join(detail_parts)}" if detail_parts else ""
        thought_text = f" | 判断: {thought}" if thought else ""
        lines.append(f"{index}. 页面: {_page_label(page)} | 动作: {action}{detail}{thought_text}")
    return "\n".join(lines)


def _foreground_state_note(foreground_package: str, package_name: str) -> str:
    """Give the model package-authoritative foreground state information."""
    foreground_package = str(foreground_package or "").strip()
    package_name = str(package_name or "").strip()
    if not foreground_package or not package_name:
        return ""
    if foreground_package == package_name:
        return f"【前台状态】前台应用: {foreground_package}（目标应用，截图是它的当前界面；Activity 名即使包含 Launcher/Home 也不能据此判定为手机桌面）。\n"
    if _looks_like_launcher_package(foreground_package):
        return f"【前台状态】前台应用: {foreground_package}（手机桌面，目标应用 {package_name} 不在前台，可能仍在启动）。\n"
    return f"【前台状态】前台应用: {foreground_package}（非目标应用 {package_name}）。\n"


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
                f"【滚动预算】已连续向{_format_scroll_direction(scroll_direction)}滑动"
                f"{consecutive_scrolls} 次，达到同方向滑动上限"
                f"（{TEST_MAX_CONSECUTIVE_SAME_DIRECTION_SCROLLS} 次）。"
                "禁止继续向该方向滑动；请改为点击列表项、返回或转向其他探索。\n"
            )
        elif consecutive_scrolls >= TEST_MAX_CONSECUTIVE_SAME_DIRECTION_SCROLLS - 1:
            scroll_budget_note = (
                f"【滚动预算】已连续向{_format_scroll_direction(scroll_direction)}滑动"
                f"{consecutive_scrolls} 次，最多允许"
                f"{TEST_MAX_CONSECUTIVE_SAME_DIRECTION_SCROLLS} 次同方向连续滑动。"
                "下一步请准备转向点击列表项、返回或其他探索。\n"
            )
    content: list[dict] = [
        {
            "type": "text",
            "text": (
                f"【第 {step_index} 步】请根据测试任务、完整执行历史和当前截图决定下一步操作。\n"
                f"本阶段最多执行 {max_steps} 步；当前是第 {step_index} 步，包含本步还剩 {remaining_steps} 步。\n"
                "如果剩余步数不足以继续有价值探索，应优先记录当前可验证结果并选择 finished。\n"
                "具有相同功能并进入相同结果的重复控件只操作一个代表项；"
                "中间结果页不是默认终点；只有探索方向已完成或没有明确相关的未执行控件时才 finished。\n"
                "在输出前先检查：最近是否已经在相同页面对相同控件做过相同动作且没有变化；"
                "如果是，请不要重复原动作，应换策略或 finished。\n"
                f"{scroll_budget_note}"
                "【完整执行历史】\n"
                f"{_format_execution_history(trace)}\n\n"
                f"【最近 {KEEP_IMAGE_STEPS} 张历史截图】"
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
            "text": f"历史截图：第 {absolute_index} 步执行前的页面。",
        })
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{screenshot}"},
        })

    content.append({
        "type": "text",
        "text": (
            f"{foreground_note}"
            "【当前截图】请基于这张截图输出下一步 JSON 动作。"
            + (
                "\n截图健康检查认为当前界面可能是黑屏或不可用状态。"
                "如果能通过返回键退出，优先执行一次 press_back；不要重复进入造成该状态的入口。"
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

    # 已知高频笔误统一修复（如 "x": 226, 190 丢失 "y": 键名）：
    # 与 llm.invoke_json 共用同一修复层，避免各解析路径规则漂移。
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
    raise ValueError("模型输出中没有合法 JSON object")


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
        r"当前|已经|已成功|本次|此前|再次|准确|尝试|找到|页面|按钮|入口|该|进行|"
        r"任务目标|进入对应|进入|点击|操作|查看|结果|完成|达成|后结束|\s+|[，。、“”'\"]",
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
    return directions[0] or "未知方向"


def _format_scroll_direction(direction: str) -> str:
    return {
        "up": "上",
        "down": "下",
        "left": "左",
        "right": "右",
    }.get(str(direction or "").strip(), str(direction or "").strip() or "未知方向")


def _consecutive_same_direction_scroll(trace: list[dict]) -> tuple[str, int]:
    """从尾部向前统计连续同方向 scroll 步数。

    只统计 action_name == "scroll" 的连续段；中间插入其他动作或改变方向
    都会中断计数。尾部动作不是 scroll（或缺少 direction）时返回 0。
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
    text = re.sub(r"[0-9一二三四五六七八九十多]+次", "", text)
    text = re.sub(r"(经过|已经|仍然|当前|需要|继续|页面|底部|区域|完全|尚未|目标)", "", text)
    text = re.sub(r"[\"'“”‘’（）()【】\[\]{}<>《》、，,。.：:；;!！?？\-_/\s]+", "", text)
    return text


def _scroll_intent_tokens(step: dict) -> set[str]:
    text = " ".join((
        str(step.get("thought", "")).strip(),
        str(step.get("action_description", "")).strip(),
        str(step.get("content", "")).strip(),
    ))
    tokens: set[str] = set()
    for match in re.findall(r"[\"'“”‘’「」『』`]{1}([^\"'“”‘’「」『』`]{1,32})[\"'“”‘’「」『』`]{1}", text):
        token = _normalize_scroll_intent_text(match)
        if len(token) >= 2:
            tokens.add(token)
    for match in re.findall(r"(?<![A-Za-z0-9_])[A-Za-z][A-Za-z0-9_-]{1,30}(?![A-Za-z0-9_])", text):
        token = match.lower()
        if token not in SCROLL_INTENT_STOPWORDS:
            tokens.add(token)
    for match in re.findall(r"(?:目标|找到|寻找|显示|露出|点击)?([\u4e00-\u9fffA-Za-z0-9_-]{1,16})(?:按钮|控件|文本|入口)", text):
        token = _normalize_scroll_intent_text(match)
        if len(token) >= 2 and token not in {"目标", "按钮", "控件", "文本", "入口"}:
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
        return f"连续 {repeat_window} 次向{direction}滚动寻找同一目标“{target}”"

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
        return f"连续 {repeat_window} 次向{direction}滚动执行高度相似的寻找动作"

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
        return "检测到页面导航循环"

    if page_keys[-6:-3] == page_keys[-3:]:
        return "检测到页面导航循环"

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
        return "检测到相同 Activity 内的界面状态往返循环"
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

    重复动作、同方向滚动和 A→B→A→B 往返都各有专门判据，但混合方向的盲试
    （如 scroll up/left/right 与 click 交替）会绕过全部判据，把步数走满。
    这里只依赖 Activity 与截图，不依赖 UI 层级，也不要求动作或方向一致。
    """
    if len(trace) < NO_PROGRESS_STALL_WINDOW:
        return ""

    recent = trace[-NO_PROGRESS_STALL_WINDOW:]
    if any(
        str(step.get("action_name", "")).strip() not in NO_PROGRESS_STALL_ACTIONS
        for step in recent
    ):
        return ""
    # 干扰恢复本身会切换界面，另有专门判据处理，这里不介入。
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
        f"连续 {NO_PROGRESS_STALL_WINDOW} 步在同一界面反复试探，"
        f"视觉状态仅 {visual_states} 种，判定当前目标不可达"
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
            return "检测到目标动作与干扰恢复之间的重复循环"

    latest_action = str(trace[-1].get("action_name", "")).strip()
    if latest_action in {"click", "long_press"} and _is_semantic_click_stall(trace):
        return "相同点击意图重试一次后页面仍无明显变化"
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
                    f"连续向{_format_scroll_direction(scroll_direction)}滚动"
                    f"{consecutive_scrolls} 次，超出同方向滑动预算"
                    f"（{TEST_MAX_CONSECUTIVE_SAME_DIRECTION_SCROLLS} 次），"
                    "应转向点击列表项、返回或其他方向探索"
                )
        repeated_intent_reason = _repeated_scroll_intent_reason(trace)
        if repeated_intent_reason:
            return repeated_intent_reason
        scroll_direction = _repeated_scroll_stall_direction(trace)
        if scroll_direction:
            return f"连续 {SCROLL_STALL_REPEAT_WINDOW} 次向{_format_scroll_direction(scroll_direction)}滚动且页面无明显变化"

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
            return f"检测到连续 {repeat_window} 次重复输入"
        if action_name in {"click", "long_press"}:
            return f"检测到连续 {repeat_window} 次重复点击同一区域"
        return f"检测到连续 {repeat_window} 次重复动作: {action_name}"

    if len(trace) >= 8:
        latest = [_step_signature(step) for step in trace[-4:]]
        previous = [_step_signature(step) for step in trace[-8:-4]]
        if latest == previous and len(set(latest)) <= 2:
            return "检测到重复模式循环"

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
        "thought": "运行时观察到黑屏或不可用界面，结束当前场景并交给下一轮恢复。",
        "action_description": "观察到黑屏或页面不可用",
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
        print(f"[场景执行] 终止前最终观测不可用: {exc}")
        return None, str(exc)
    except Exception as exc:
        print(f"[场景执行] 终止前最终观测失败: {exc}")
        return None, ""
    trace.append({
        "action_name": "capture_screenshot",
        "thought": "场景结束前补充最后一个有副作用动作的结果观测。",
        "action_description": "场景终止前最终页面观测",
        "current_page": page,
        "trace_phase": trace_phase,
        "observation_only": True,
        "terminal_observation": True,
        **({"screenshot_health_error": health_error} if health_error else {}),
    })
    print(
        "[场景执行] 已补充终止前最终观测: "
        f"phase={trace_phase}, page={page.id or '未知'}, "
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
    print("[场景执行] 连续黑屏达到阈值，重启目标 App 并等待界面稳定。")
    if not _restart_target_app(device, package_name):
        return None, "连续黑屏后重启目标 App 失败"
    try:
        page = capture_current_page(device)
    except ScreenshotUnavailableError as exc:
        return None, f"连续黑屏重启后界面仍不可用: {exc}"
    except Exception as exc:
        return None, f"连续黑屏重启后抓取页面失败: {exc}"
    trace.append({
        "action_name": "restart_app",
        "thought": "连续黑屏达到阈值，运行器重启目标 App 以恢复可测试状态。",
        "action_description": "连续黑屏后重启目标 App",
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
        print("[场景执行] 阻塞页已由执行器返回，不重复执行返回动作。")
        return None
    try:
        print("[场景执行] 当前页面无法继续，自动按一次返回退出阻塞页。")
        _perform_action(device=device, action_name="press_back")
        time.sleep(FAST_ACTION_DELAY)
        page = capture_current_page(device)
    except Exception as exc:
        print(f"[场景执行] 退出阻塞页失败，结束当前功能并切换其它功能: {exc}")
        return None
    trace.append({
        "action_name": "press_back",
        "thought": "运行器退出验证码或已确认无法继续的阻塞页面。",
        "action_description": "退出无法继续的阻塞页面",
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
    phase_label = {"navigation": "导航", "test": "场景"}.get(
        trace_phase, trace_phase
    )
    step_limit = (
        MAX_NAVIGATION_STEPS if trace_phase == "navigation" else MAX_EXEC_STEPS
    )
    if not scenario:
        print(f"[场景执行] {phase_label}阶段跳过: 未配置场景。")
        return [], initial_page, "skipped", "", "", None

    representative_scenario = representative_control_family_scenario(scenario)
    if representative_scenario != scenario:
        print("[场景执行] 检测到同功能控件穷尽式场景，强制改为单个代表项。")
        scenario = representative_scenario

    risky_scenario_intent = wechat_read_only_violation(package_name, scenario)
    if risky_scenario_intent:
        reason = f"{WECHAT_READ_ONLY_BLOCKED_END}: {risky_scenario_intent}"
        print(f"[场景执行] {reason}")
        return [], initial_page, "blocked_by_safety_policy", reason, "", None

    if clear_logcat:
        _clear_logcat(device)
        print(f"[场景执行] 已设置场景 logcat 起点: phase={trace_phase}")

    scene_started_at = time.time()
    if not _ensure_target_app(device, package_name):
        return [], initial_page, "error", "无法拉起目标应用", "", None
    target_pid = _target_process_pid(device, package_name)
    print(
        "[场景执行] 场景起点: "
        f"phase={trace_phase}, target={package_name or '未知'}, "
        f"pid={target_pid or '未知'}, started_at={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(scene_started_at))}"
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
        print("[场景执行] 当前场景允许受控外部应用跳转；场景结束时会恢复目标应用。")
    allows_target_exit = _scenario_allows_target_exit(scenario)
    if allows_target_exit:
        print("[场景执行] 当前场景允许目标应用退出前台；场景结束时会恢复目标应用。")

    trace: list[dict] = []
    consecutive_errors = 0
    # 单轮场景内因截图失败触发的应用重启次数。达到上限后直接结束场景，
    # 防止设备持续故障（如 adb 掉线、uiautomator2 挂死）时
    # “截图失败-重启-再失败”无限循环占用整个运行预算。
    screenshot_restart_count = 0
    # 单轮累计的模型错误次数。consecutive_errors 只要中间成功一次就清零，
    # 无法阻止“超时—成功—超时”交替在一轮内累积出分钟级空等。
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

    print(f"[场景执行] 开始{phase_label}阶段: {scenario}")
    if isinstance(initial_page, Page):
        print(f"[场景执行] 起始页面: {initial_page.id or '未知'}")

    while True:
        if len(trace) >= step_limit:
            print(f"[警告] 达到最大步数限制 ({step_limit})，强制结束。")
            status = "max_steps_reached"
            error_message = f"达到最大步数限制 {step_limit}"
            break

        page_health_error = ""
        try:
            foreground_state, current_package = _target_app_foreground_state(device, package_name)
            if foreground_state != "foreground":
                # 页面切换动画期间可能瞬时上报 launcher（应用自身过渡态），
                # 复检确认后再定性为外部跳转/崩溃，避免误终止场景。
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
                        "[场景执行] 检测到目标应用崩溃，结束当前场景: "
                        f"current={current_package or '未知'}, target={package_name or '未知'}"
                    )
                    restarted = _restart_target_app(device, package_name)
                    if restarted:
                        recovered_to_target_app = True
                        final_recovery_method = "restart_app"
                        _append_target_app_recovery_step(
                            trace,
                            device,
                            package_name,
                            current_package or "未知",
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
                            print(f"[场景执行] 崩溃重启后抓取页面失败: {capture_err}")
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
                        f"[场景执行] 检测到受控外部应用上下文，继续执行当前场景: "
                        f"current={current_package or '未知'}, target={package_name or '未知'}"
                    )
                else:
                    if foreground_state == "background":
                        external_app_used = True
                        if trace and not trace[-1].get("external_context"):
                            trace[-1]["caused_external_transition"] = True
                            trace[-1]["external_package"] = current_package or ""
                    print(
                        f"[场景执行] 执行中检测到未受控外部应用状态: status={foreground_state}, "
                        f"current={current_package or '未知'}, target={package_name or '未知'}"
                    )
                    _append_external_observation_step(
                        trace,
                        device,
                        package_name,
                        current_package or "未知",
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
                    print("[场景执行] 返回恢复失败，重启目标应用并结束本轮场景。")
                    crash_logcat = _read_crash_logcat(device, package_name)
                    if _restart_target_app(device, package_name):
                        recovered_to_target_app = True
                        final_recovery_method = "restart_app"
                        _append_target_app_recovery_step(
                            trace,
                            device,
                            package_name,
                            current_package or "未知",
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
                            print(f"[场景执行] 重启后抓取页面失败: {capture_err}")
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
                print(f"[场景执行] 运行时观察到异常截图: {page_health_error}")
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
                        "观察到黑屏或页面不可用，已保留当前界面并结束场景: "
                        f"{page_health_error}"
                    )
                    final_page_override = page
                    break
        except ScreenshotUnavailableError as e:
            print(f"[场景执行] 完全无法取得运行时截图，立即停止当前场景: {e}")
            status = "screenshot_unavailable"
            error_message = f"无法取得运行时截图，当前场景已停止: {e}"
            break
        except Exception as e:
            print(f"[错误] 截图失败: {e}")
            consecutive_errors += 1
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                if screenshot_restart_count >= APP_RESTART_ATTEMPTS:
                    status = "error"
                    error_message = (
                        f"连续截图失败且应用重启已达 {APP_RESTART_ATTEMPTS} 次，"
                        f"结束当前场景: {e}"
                    )
                    print(f"[场景执行] {error_message}")
                    break
                try:
                    device.restart_app(package_name)
                    time.sleep(APP_RESTART_DELAY)
                    consecutive_errors = 0
                    screenshot_restart_count += 1
                    continue
                except Exception as restart_err:
                    status = "error"
                    error_message = f"连续截图失败且重启失败: {restart_err}"
                    break
            time.sleep(2)
            continue

        # 探索边界检测（core 模式）：目标应用内进入第三方服务内容（如小程序、
        # 内嵌网页等）时自动退出并结束当前场景。规则按包名从配置加载，未配置
        # 的应用不启用，行为与无本检测时完全一致。
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
                        print(f"[探索边界] 进程兜底检测失败（忽略）: {exc}")
        if boundary_hit is not None:
            print(
                f"[探索边界] 检测到进入{boundary_hit.rule.label}"
                f"（via={boundary_hit.via}, matched={boundary_hit.matched}），"
                "自动退出并结束当前场景。"
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
                        f"进入{boundary_hit.rule.label}越过探索边界"
                        f"（via={boundary_hit.via}, matched={boundary_hit.matched}）；"
                        "已通过返回键退出。"
                    ),
                )
                status = "exploration_boundary_blocked"
                error_message = EXPLORATION_BOUNDARY_BLOCKED_END.format(
                    label=boundary_hit.rule.label
                )
                break
            print("[探索边界] 返回退出失败，重启目标应用并结束本轮场景。")
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
                        f"进入{boundary_hit.rule.label}越过探索边界且返回退出失败；"
                        "已重启目标应用。"
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
                    print(f"[场景执行] 重启后抓取页面失败: {capture_err}")
                status = "target_app_restarted"
                error_message = TARGET_APP_RESTARTED_END
            else:
                status = "target_app_recovery_failed"
                error_message = TARGET_APP_RECOVERY_FAILED_END
            break

        try:
            img_w, img_h = device.display_size()
        except Exception as e:
            print(f"[错误] 获取屏幕尺寸失败: {e}")
            consecutive_errors += 1
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                status = "error"
                error_message = f"连续获取屏幕尺寸失败: {e}"
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
            "[场景执行] LLM 输入摘要: "
            f"phase={trace_phase}, step={step_index}, history_steps={len(trace)}, "
            f"recent_screenshots={_recent_screenshot_count(trace)}, "
            f"current_page={_page_label(page)}, activity={page.activity or '未知'}, "
            f"foreground={foreground_package or '未知'}, "
            f"current_screenshot={'yes' if b64_image else 'no'}"
        )

        raw_content = "未知"
        try:
            record_llm_request(llm, operation="[场景执行]")
            response = llm.invoke(messages)
            record_llm_response(llm, response, operation="[场景执行]")
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
            print(f"[场景执行] LLM 输出 JSON: {log_plan}")
        except Exception as e:
            record_llm_failure(llm, operation="[场景执行]")
            print(f"[错误] LLM调用或解析失败: {e}\n[原始内容]: {raw_content}")
            consecutive_errors += 1
            total_llm_errors += 1
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                status = "error"
                error_message = f"连续 LLM 解析失败: {e}"
                break
            if total_llm_errors >= MAX_TOTAL_LLM_ERRORS:
                status = "error"
                error_message = (
                    f"本轮模型错误累计达到 {total_llm_errors} 次，提前结束场景: {e}"
                )
                print(f"[场景执行] {error_message}")
                break
            _sleep_for_llm_retry(consecutive_errors)
            continue

        action_name = plan.get("action_name", "")
        thought = plan.get("thought", "")
        action_description = str(plan.get("action_description", "")).strip()
        is_interruption = bool(plan.get("is_interruption", False))

        risky_action_intent = ""
        proposed_action = _proposed_action_text(plan)
        whitelist_violation = wechat_action_policy_violation(
            package_name,
            action_name,
            proposed_action,
        )
        if whitelist_violation:
            risky_action_intent = whitelist_violation
        if risky_action_intent:
            status = "blocked_by_safety_policy"
            error_message = f"{WECHAT_READ_ONLY_BLOCKED_END}: {risky_action_intent}"
            print(f"[场景执行] 动作执行前拦截: {error_message}")
            break

        print(
            f"\n[步骤 {step_index}]{' [干扰]' if is_interruption else ''} "
            f"页面={page.id or '未知'} 动作={action_name} | 思考: {thought}"
        )

        try:
            _scale_plan_coordinates(plan, img_w, img_h)
        except ValueError as e:
            print(f"[错误] LLM坐标解析失败: {e}\n[原始内容]: {raw_content}")
            consecutive_errors += 1
            total_llm_errors += 1
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                status = "error"
                error_message = f"连续 LLM 坐标解析失败: {e}"
                break
            if total_llm_errors >= MAX_TOTAL_LLM_ERRORS:
                status = "error"
                error_message = (
                    f"本轮模型错误累计达到 {total_llm_errors} 次，提前结束场景: {e}"
                )
                print(f"[场景执行] {error_message}")
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
            print(f"[场景执行] 动作执行前拦截: {verification_attempt_reason}")
            break

        stall_reason = _detect_stall(trace)
        if stall_reason:
            print(f"[警告] {stall_reason}，提前结束当前场景。")
            step_info["action_failed"] = True
            step_info["action_error"] = stall_reason
            status = "stalled"
            error_message = stall_reason
            break

        if action_name == "finished":
            if page_health_error:
                status = "screenshot_unavailable"
                error_message = "执行器在黑屏或不可用界面结束，等待下一轮恢复"
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
                print(f"[场景执行] 验证挑战阻塞当前功能: {thought[:160]}")
            elif _finished_declares_failure(thought):
                status = "failed_goal"
                error_message = f"执行器判断当前目标不可达: {thought[:240]}"
                step_info["action_failed"] = True
                step_info["action_error"] = error_message
                print(f"[场景执行] finished 包含失败语义，降级为 failed_goal: {thought[:160]}")
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
                print(f"[场景执行] 动作参数: {action_kwargs}")
            try:
                _perform_action(device=device, action_name=action_name, **action_kwargs)
            except Exception as e:
                print(f"[错误] 执行动作失败: {e}")
                step_info["is_interruption"] = True
                step_info["action_failed"] = True
                step_info["action_error"] = str(e)
                if action_failure_recoveries < MAX_ACTION_FAILURE_RECOVERIES:
                    action_failure_recoveries += 1
                    print(
                        "[场景执行] 将动作失败及原因交给执行模型，允许改用一次替代动作继续当前场景。"
                    )
                    continue
                status = "action_failed"
                error_message = f"动作执行失败且替代策略仍失败，已结束当前场景: {e}"
                break

        if action_name == "wait":
            time.sleep(ACTION_DELAY)
        elif action_name == "open_app" and package_name:
            # 冷启动轮询代替固定延时：避免在应用仍在启动时截图，
            # 让执行模型看到桌面/启动页而误判重复 open_app。
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
                "最后一个有副作用动作后的页面不可用，已结束当前场景: "
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
        "[轨迹截图] 场景截图索引完成: "
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
        f"[场景执行] {phase_label}阶段完成 | status={status} | "
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
                    "navigation_error": "目标应用无法拉起",
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
                    "[场景执行] 导航阶段截图持续不可用，结束当前 app 运行: "
                    f"{unavailable_count}/{MAX_CONSECUTIVE_SCREENSHOT_UNAVAILABLE}"
                )
            else:
                print(
                    "[场景执行] 导航阶段截图不可用，当前场景结束；后续探索其它功能: "
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
            print("[场景执行] 导航阶段离开目标应用，本轮不再继续执行测试场景。")
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
            print(f"[场景执行] 导航失败，本轮跳过执行场景。连续失败次数: {navigation_failure_count}")
            reset_page = nav_final_page or current_page
            if navigation_failure_count >= 2:
                if package_name:
                    navigation_restarted_after_failures = True
                    print("[场景执行] 连续两次导航失败，重启 app 回到初始页面。")
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
                            print(f"[场景执行] 重启后抓取页面失败: {e}")
                            reset_page = nav_final_page or current_page
                    else:
                        print("[场景执行] 连续导航失败后仍无法恢复目标应用。")
                else:
                    print("[场景执行] 连续两次导航失败，但目标包名为空；重置失败计数。")
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
                    "[场景执行] 场景阶段截图持续不可用，结束当前 app 运行: "
                    f"{unavailable_count}/{MAX_CONSECUTIVE_SCREENSHOT_UNAVAILABLE}"
                )
            else:
                print(
                    "[场景执行] 场景阶段截图不可用，当前场景结束；后续探索其它功能: "
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
