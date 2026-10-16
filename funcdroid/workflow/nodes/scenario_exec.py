"""场景执行节点。"""

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
    APP_RESTART_DELAY,  # noqa: F401  保留原模块名供引用/测试
    APP_RESTART_MAX_SECONDS,
    APP_RESTART_RETRY_DELAY,  # noqa: F401  保留原模块名供引用/测试
    APP_RESTART_SETTLE_DELAY,
    APP_START_MAX_WAIT,
    APP_START_POLL_INTERVAL,  # noqa: F401  保留原模块名供引用/测试
    FOREGROUND_CHECK_RETRIES,
    FOREGROUND_CHECK_RETRY_DELAY,  # noqa: F401  保留原模块名供引用/测试
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


# 每步随消息发送的历史截图张数；默认 1，仅带上一步执行前截图，与当前截图
# 构成最近一次动作的前后对照。需要更长视觉记忆时调大（如 EXEC_KEEP_IMAGE_STEPS=3）；
# 设为 0 表示不发送历史截图（注意 trace[-0:] 会取全量，必须单独处理）。
KEEP_IMAGE_STEPS = max(0, int(os.getenv("EXEC_KEEP_IMAGE_STEPS", "1")))
# 生成失败（解析/校验/调用失败）写回执行历史的滚动窗口上限。
MAX_RECENT_FAILURES = 3
# 场景执行（test 阶段）单轮步数上限；导航阶段使用独立的 MAX_NAVIGATION_STEPS。
MAX_EXEC_STEPS = max(1, int(os.getenv("MAX_EXEC_STEPS", "15")))
MAX_NAVIGATION_STEPS = max(1, int(os.getenv("MAX_NAVIGATION_STEPS", "10")))
# 实测：部分本地模型即使开启 response_format=json_object，仍会偶发用纯文字
# 描述动作意图而不带 JSON（见合规批量实验 run.log），通常 2-3 次重试后恢复。
# 默认从 3 调到 4，给这种“格式回退”留出更充分的纠偏机会，避免整个场景
# （及其覆盖的一组合规条款）因临界的连续失败直接判 error/deferred。
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
# wait 的显式等待时长：默认 3s 覆盖启动页/WebView 加载，避免模型为了等加载
# 连续输出两次 wait 而被 WAIT_STALL_REPEAT_WINDOW 误判为重复动作。
MIN_WAIT_SECONDS = max(0.5, float(os.getenv("EXEC_WAIT_MIN_SECONDS", "1")))
MAX_WAIT_SECONDS = max(MIN_WAIT_SECONDS, float(os.getenv("EXEC_WAIT_MAX_SECONDS", "10")))
DEFAULT_WAIT_SECONDS = min(
    MAX_WAIT_SECONDS,
    max(MIN_WAIT_SECONDS, float(os.getenv("EXEC_WAIT_DEFAULT_SECONDS", "3"))),
)
# swipe 拖动时长（秒）：短=快速甩动，长=可控拖动（通知栏下拉、滑块）。
MIN_SWIPE_DURATION = max(0.05, float(os.getenv("EXEC_SWIPE_MIN_DURATION", "0.1")))
MAX_SWIPE_DURATION = max(MIN_SWIPE_DURATION, float(os.getenv("EXEC_SWIPE_MAX_DURATION", "1.5")))
DEFAULT_SWIPE_DURATION = min(
    MAX_SWIPE_DURATION,
    max(MIN_SWIPE_DURATION, float(os.getenv("EXEC_SWIPE_DEFAULT_DURATION", "0.5"))),
)
# scroll 与 swipe 共用同一套滚动类判据（同方向预算、无进展），
# 避免新增手势成为绕过停滞检测的新通道。
SCROLL_LIKE_ACTIONS = {"scroll", "swipe"}
# 模型动作中需要记入轨迹并透传给设备的参数字段。
ACTION_PARAMETER_FIELDS = (
    *COORDINATE_FIELDS,
    *SWIPE_COORDINATE_FIELDS,
    "content",
    "direction",
    "app_name",
    "duration",
)
# 同一动作在界面无变化时连续重复多少次判停滞。open_app 每次都可能真实重启，
# 代价高，2 次即拦截；滚动在长列表中合理地多次重复，放宽到 5 次。
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
# 动作效果反馈：变化像素占比低于该值视为界面无变化（光标闪烁、JPEG 噪声），
# 高于 EFFECT_MAJOR_CHANGE_RATIO 视为整页级变化。只给最近几步计算，控制解码开销。
EFFECT_NO_CHANGE_RATIO = float(os.getenv("EXEC_EFFECT_NO_CHANGE_RATIO", "0.0002"))
# “同一位置”按模型输出的图片坐标判断（与 REPEAT_COORD_TOLERANCE 的屏幕像素口径不同）。
EFFECT_SAME_POINT_TOLERANCE = int(os.getenv("EXEC_EFFECT_SAME_POINT_TOLERANCE", "50"))
EFFECT_MAJOR_CHANGE_RATIO = float(os.getenv("EXEC_EFFECT_MAJOR_CHANGE_RATIO", "0.15"))
EFFECT_HISTORY_STEPS = max(1, int(os.getenv("EXEC_EFFECT_HISTORY_STEPS", "6")))
EFFECT_ACTIONS = {"click", "long_press", "type", "press_back", "scroll", "swipe"}
REPEAT_COORD_TOLERANCE = int(os.getenv("EXEC_REPEAT_COORD_TOLERANCE", "80"))
# 点击落点二次定位：执行模型在完整提示下对小控件常有稳定的数十像素偏差。
# 落点周围没有局部纹理（高频分量标准差低于阈值；背景通常 <5，控件上 >9）
# 时才裁剪放大落点附近区域、请模型再定位一次；其余点击不额外调用，保持速度。
TAP_REFINE_ENABLED = os.getenv("EXEC_TAP_REFINE", "1").strip().lower() not in {"0", "false", "no", "off"}
TAP_REFINE_FLAT_STD = float(os.getenv("EXEC_TAP_REFINE_FLAT_STD", "6"))
TAP_REFINE_RADIUS = max(60, int(os.getenv("EXEC_TAP_REFINE_RADIUS", "180")))
TAP_REFINE_MAX_SHIFT = max(10, int(os.getenv("EXEC_TAP_REFINE_MAX_SHIFT", "120")))
TAP_REFINE_TIMEOUT_SECONDS = max(1.0, float(os.getenv("EXEC_TAP_REFINE_TIMEOUT", "8")))
STALL_SCREENSHOT_SIMILARITY_THRESHOLD = float(os.getenv("EXEC_STALL_SCREENSHOT_SIMILARITY", "0.9995"))
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
NO_PROGRESS_STALL_ACTIONS = {"click", "long_press", "scroll", "swipe", "press_back", "wait"}
APP_RECOVERY_BACK_ATTEMPTS = max(1, int(os.getenv("APP_RECOVERY_BACK_ATTEMPTS", "3")))
# APP_RESTART_* / APP_START_* / FOREGROUND_CHECK_* 冷启动轮询与重启参数统一定义在
# workflow/app_foreground.py（规划节点同口径），此处导入后保留原名供调用与测试引用。
# 前台是桌面（launcher）时按返回键无效，恢复流程直接跳到重启。
LAUNCHER_PACKAGE_PATTERN = re.compile(r"launcher|home", re.IGNORECASE)
# 执行结果状态（navigation_status / scenario_status 共用）：
#   completed               执行模型确认目标达成
#   failed_goal             执行模型判定目标不可达，或运行器拦截了破坏登录态的动作
#   blocked                 硬墙：验证码、账号安全策略、探索边界，当前环境不可测
#   stalled                 停滞检测提前结束
#   incomplete              步数或时间预算耗尽
#   left_app                目标应用离开前台（非崩溃），已尝试返回/重启恢复
#   app_crashed             目标应用崩溃
#   screenshot_unavailable  截图黑屏或不可用
#   error                   模型、设备或拉起应用连续出错
#   skipped                 未执行（未配置场景，或导航未完成导致测试跳过）
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
# 已由运行器自行恢复前台（或无法取得画面）的状态：结束时不再补观测、不再恢复。
FOREGROUND_HANDLED_STATUSES = frozenset({"left_app", "app_crashed", "screenshot_unavailable"})
TARGET_APP_CRASHED_END = "检测到目标应用崩溃，已重启目标应用并结束当前场景"
TARGET_APP_CRASH_RECOVERY_FAILED_END = "检测到目标应用崩溃，但重启后仍未恢复到目标应用，已结束当前场景"
TARGET_APP_RECOVERY_FAILED_END = "目标应用离开前台，自动返回和重启均未能恢复，已结束本轮场景"
EXTERNAL_TRANSITION_BLOCKED_END = "目标应用跳转到未受控外部应用，已恢复目标应用并结束当前场景"
# 探索边界（core 模式）结束文案模板：{label} 来自边界规则配置，如"小程序/第三方服务"。
EXPLORATION_BOUNDARY_BLOCKED_END = "检测到进入{label}（非目标应用核心功能），已自动返回并结束当前场景"
WECHAT_READ_ONLY_BLOCKED_END = "微信账号安全策略拦截了退出登录或删除账户操作"
VERIFICATION_BLOCKED_END = "验证码或人机验证无法由无人值守测试完成，已退出当前阻塞页面"
# 前台状态复检：页面切换动画期间 current_package 可能瞬时上报桌面
# launcher（应用自身过渡态），单次立即检测会把过渡态误判为外部跳转
# （left_app）。非前台结论必须经短暂等待复检确认。
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
# 单独的“滑块”也可能是音量/亮度/进度等普通控件；“拖动滑块验证读数”里的“验证”
# 只是核对。只有成词的“滑块验证/滑动验证”（后接“码”、标点或结尾）才算挑战。
VERIFICATION_CHALLENGE_PATTERN = re.compile(
    r"验证码|图形验证|人机验证|安全验证|拼图|captcha|verification\s+challenge"
    r"|(?:滑块|滑动)验证(?=码|[，。、；,.;!！?？\s]|$)"
    r"|(?:拖动|滑动)滑块(?:以|来)?(?:完成|通过)验证",
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


def _parse_optional_seconds(value: object, key: str) -> float | None:
    """解析可选的秒数参数；空值表示“未提供”，非法值交给模型重试。"""
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{key} 必须是数字: {value!r}")
    try:
        number = float(str(value).strip())
    except ValueError as exc:
        raise ValueError(f"{key} 必须是数字: {value!r}") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{key} 必须是正数: {value!r}")
    return number


def _clamp_seconds(value: float | None, default: float, lower: float, upper: float) -> float:
    """把秒数收敛到配置区间；缺省值时用默认值参与收敛。"""
    return max(lower, min(upper, default if value is None else value))


def _scale_plan_coordinates(
    plan: dict, screen_w: int, screen_h: int, image_size: tuple[int, int],
) -> None:
    """将上传图片像素坐标换算为屏幕像素；不推断或兼容其他坐标制。"""
    image_w, image_h = image_size
    if min(image_w, image_h, screen_w, screen_h) <= 1:
        raise ValueError("图片和屏幕宽高必须大于 1")
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
                raise ValueError(f"{key} 超出当前图片像素范围 0–{source - 1}: {number}")
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
        raise ValueError(f"不支持或缺失的操作类型: {action_name!r}")
    if action_name in {"click", "long_press"} and (
        plan.get("x") is None or plan.get("y") is None
    ):
        raise ValueError(f"{action_name} 缺少 x/y 坐标")
    if action_name in {"click", "long_press"}:
        for key in COORDINATE_FIELDS:
            number = _parse_coordinate_value(plan.get(key), key)
            if image_size is None or min(image_size) <= 1:
                raise ValueError("缺少有效的当前截图尺寸，无法校验图片像素坐标")
            limit = image_size[0 if key.startswith("x") else 1] - 1
            if number is None or not 0 <= number <= limit:
                raise ValueError(f"{key} 坐标必须在当前图片像素范围 0–{limit} 内: {plan.get(key)!r}")
    if action_name == "type" and not str(plan.get("content", "")):
        raise ValueError("type 缺少 content")
    if action_name == "scroll":
        direction = str(plan.get("direction", "")).strip().lower()
        if direction not in {"up", "down", "left", "right"}:
            raise ValueError(f"scroll direction 非法: {direction!r}")
    if action_name == "swipe":
        for key in SWIPE_COORDINATE_FIELDS:
            number = _parse_coordinate_value(plan.get(key), key)
            if number is None:
                raise ValueError(f"swipe 的 {key} 缺失")
            if image_size is None or min(image_size) <= 1:
                raise ValueError("缺少有效的当前截图尺寸，无法校验图片像素坐标")
            limit = image_size[0 if key.startswith("x") else 1] - 1
            if not 0 <= number <= limit:
                raise ValueError(f"swipe 的 {key} 必须在当前图片像素范围 0–{limit} 内: {plan.get(key)!r}")
        if plan.get("x1") == plan.get("x2") and plan.get("y1") == plan.get("y2"):
            raise ValueError("swipe 的起点与终点不能相同")
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


# finished 的 thought 按系统提示词要求以结论开头；开头前容许少量括号/标点。
FINISHED_LEADING_PUNCTUATION = "【[（(「『<《“\"'*#-—:：·• \t"
FINISHED_FAILURE_PREFIX = "目标不可达"
FINISHED_SUCCESS_PREFIX = "目标已达成"


def _finished_declares_failure(thought: str) -> bool:
    text = str(thought or "").strip().lower()
    if not text:
        return False
    # 系统提示词规定 thought 以“目标已达成 / 目标不可达”开头：显式结论优先于
    # 正文中的成功/失败关键词。否则“目标不可达：……已完成所有可见控件检查”
    # 会先命中成功模式（已完成…所有）而被记为 completed。
    leading = text.lstrip(FINISHED_LEADING_PUNCTUATION)
    if leading.startswith(FINISHED_FAILURE_PREFIX):
        return True
    if leading.startswith(FINISHED_SUCCESS_PREFIX):
        # 以“目标已达成”开头但后文又承认未完成/不可达或改用替代入口，仍判失败。
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


def _unfinished_safe_external_reason(action_name: str, foreground_package: str) -> str:
    """Prevent a model from ending while a required system dialog is unresolved."""
    if str(action_name or "").strip() != "finished":
        return ""
    foreground_package = str(foreground_package or "").strip()
    if not _is_safe_external_package(foreground_package):
        return ""
    return (
        f"当前仍停留在任务流程相关的系统页面 {foreground_package}；"
        "必须先处理可见选择并观察结果，不能 finished"
    )


def _uses_adb_backend(device) -> bool:
    operating_system = str(getattr(device, "operating_system", "adb") or "adb").strip().lower()
    return operating_system in {"adb", "android"}


def _current_package(device, retries: int = FOREGROUND_CHECK_RETRIES) -> str | None:
    return current_foreground_package(device, retries, log_prefix="[场景执行]")


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
    """短暂等待后复检前台状态，过滤页面切换动画期间的过渡态误报。

    实测（微信服务号卡片跳转）：点击后 current_package 瞬时上报桌面
    launcher，最终稳定在目标应用内的新 Activity（webview 等）；单次
    立即检测会把过渡态误判为外部跳转（left_app），
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
    # 通过模块级 _target_app_foreground_state 探测，保留测试替换该函数的能力。
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
        log_prefix="[场景执行]",
    )
    return restarted


def _ensure_target_app(device, package_name: str) -> bool:
    foreground_state, current_package = _target_app_foreground_state(device, package_name)
    if foreground_state == "foreground":
        return True
    if is_permission_dialog(current_package):
        print("[场景执行] 保留系统权限弹窗，先取证并执行可见选择，避免反复重启。")
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
    foreground_state, current_package = _target_app_foreground_state(device, package_name)
    if foreground_state == "foreground":
        return True

    # 复用同一次前台查询结果，不再紧接着重复查询一次。
    current_package = str(current_package or "").strip()
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
    因此复检与进入检测共用 _current_boundary_hit 判据（Activity 优先，
    仅在前台为目标包且 Activity 不可读时才看进程），而非按包名复检。
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
        if _current_boundary_hit(device, package_name, current_activity) is None:
            return True
    return False


def _current_boundary_hit(
    device,
    package_name: str,
    activity: str | None,
    foreground_package: str | None = None,
) -> BoundaryHit | None:
    """进入与退出探索边界共用的判据。

    Activity 是权威信号：读到 Activity 时只按 Activity 特征判断。进程特征
    （如微信 :appbrand）在小程序退出后仍常驻，“进程存在”不等于“正在边界内”，
    因此只有前台确实是目标包、且 Activity 为空/不可读时才用 ``ps -A`` 兜底；
    这也避免了每一步都执行一次 ``ps -A``。foreground_package 为 None 时按需查询。
    """
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
        print(f"[探索边界] 进程兜底检测失败（忽略）: {exc}")
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
    """统一的前台恢复：崩溃直接重启；否则先 back() 退出，失败再重启。

    返回 (crashed, recovered)。恢复成功时追加一条非业务的恢复步骤。
    调用方已对同一份 logcat 做过崩溃判定时通过 crashed 传入，避免重复解析。
    """
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
    """崩溃或恢复失败时的统一结束状态；恢复成功的非崩溃情形由调用方决定。"""
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


def _sleep_for_llm_retry(consecutive_errors: int, deadline: float | None = None) -> bool:
    """按退避时长等待后重试；剩余阶段时间不足以等待时不睡眠并返回 False。"""
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
    """把一次生成失败压缩成写回执行历史的一行原因（agent 式循环：失败也是上下文）。

    区分“模型调用本身失败（传输/超时）”“响应正文为空”“纯文字、零 JSON”与
    “JSON 有但不合法/未通过校验”（实测合规批量 run.log 中零 JSON 占比高），
    让后续步骤能针对性纠正而不是重复同样错误；传输失败不是模型输出问题，
    不能描述成“响应是纯文字说明”。
    """
    if not response_received:
        return f"模型调用失败（传输/超时），未获得响应：{type(error).__name__}"
    if not str(raw_content or "").strip():
        return "模型响应正文为空，动作未执行；必须只输出一个 JSON object"
    if "{" not in str(raw_content or ""):
        return (
            "响应是纯文字说明，未包含任何 JSON 结构，动作未执行；"
            "必须只输出一个 JSON object，禁止用文字复述动作意图"
        )
    return f"响应未通过解析/校验，动作未执行：{str(error)[:300]}"


def _runtime_page_label(page: Page | None) -> str:
    """执行历史/日志中的运行时页面名：名称 → 描述 → Activity 依次兜底。"""
    if not isinstance(page, Page):
        return "未知页面"
    return page.name or page.description or page.activity or "未知页面"


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
    """描述一次动作前后的界面变化，返回 (类别, 文字)；类别为 none/local/major/""。

    只比较截图像素，不读取控件树：点击是否起作用完全以画面为准。
    """
    result = screenshot_change_region(before_screenshot, after_screenshot)
    if result is None:
        return "", ""
    ratio, box = result
    if box is None or ratio < EFFECT_NO_CHANGE_RATIO:
        return "none", "界面无变化"
    if ratio >= EFFECT_MAJOR_CHANGE_RATIO:
        return "major", "界面明显变化"
    region = ""
    if image_size:
        width, height = image_size
        region = (
            f"，变化区域约 ({int(box[0] * width)},{int(box[1] * height)})"
            f"–({int(box[2] * width)},{int(box[3] * height)})"
        )
    return "local", f"界面局部变化（约 {ratio * 100:.2f}%{region}）"


# 动作效果缓存：键为 (前截图哈希, 后截图哈希, 图片尺寸)。不写回 step dict，
# 因为执行轨迹会按 step 的全部字段序列化进产物（main.py 只剔除 current_page）。
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
    """最近几步动作的效果；键为 trace 下标。每步的“后”截图是下一步执行前截图。

    已比较过的前后截图对直接复用缓存结果，每步只新算最后一个动作的效果。
    """
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
    """末尾连续“点击同一位置且界面无变化”时，提醒换策略而不是再点。"""
    if not trace:
        return ""
    last = trace[-1]
    action = str(last.get("action_name", "")).strip()
    last_effect = effects.get(len(trace) - 1, ("", ""))[0]
    if action == "type" and last_effect in {"none", "local"}:
        if last_effect == "none":
            return (
                "【上一步效果】type 后界面无变化：输入框可能未聚焦或不接受该内容。"
                "先点击目标输入框使其聚焦，或改用符合其格式的内容；不要原样重复输入。\n"
            )
        return (
            "【上一步效果】已执行 type。核对输入框实际显示的内容是否与输入的 content 一致；"
            "不一致说明该输入框限制了字符类型或长度，改用符合其格式的内容，不要原样重复输入。\n"
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
            f"【上一步效果】已连续 {repeats} 次点击{where}附近，界面都没有变化：该控件当前无响应。"
            "禁止再次点击这个位置；改为检查并满足它的前置条件（如必填内容、所需数量、先选中对象），"
            "或改测其他未执行控件；都不可行时输出 finished，写“目标不可达：控件无响应”。\n"
        )
    return (
        f"【上一步效果】上一步点击{where}后界面无变化。先核对截图中目标控件的真实位置："
        "若位置有偏差，换到控件中心再试一次；若位置无误，说明控件未响应，不要原样重复，"
        "改为满足其前置条件或测试其他控件。\n"
    )


def _repeated_success_note(
    trace: list[dict],
    effects: dict[int, tuple[str, str]],
) -> str:
    """上一步点击的点在本场景中已成功执行过一次时，提醒不要原样重复。

    “无响应提醒”只覆盖界面无变化的情形；实测模型还会陷入另一类循环：
    动作每次都生效（如反复打开同一个切换面板、重复点同一个档位），模型却
    意识不到目标已经达成，原样重试直到停滞检测兜底，每轮浪费 3–5 步。
    """
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
            f"【已执行过】上一步点击 ({point[0]},{point[1]}) 在本场景早前已成功执行过（{effect_text}），"
            "界面已处于该动作产生的结果状态。不要原样重复这个动作；"
            "应改为测试当前界面的其他未执行控件；若目标已达成且没有新控件，输出 finished。\n"
        )
    return ""


def _mark_tap_point(screenshot: str, point: tuple[int, int] | None) -> str:
    """在历史截图上标出上一步实际点击的位置（红色准星）；失败时返回原图。

    模型对小控件（开关、弹窗按钮）的定位常有稳定偏差，只看“界面无变化”的文字
    提示时会原样重复同一坐标。把真实落点画在它刚看过的截图上，模型能直接比对
    落点与控件，从而修正坐标或确认控件确实无响应。
    """
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
        return "暂无历史步骤。"

    effects = effects or {}
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
        point = _model_point(step) if action in {"click", "long_press"} else None
        if point:
            detail_parts.append(f"图片坐标: ({point[0]},{point[1]})")
        effect = effects.get(index - 1)
        if effect:
            detail_parts.append(f"效果: {effect[1]}")
        detail = f" | {'; '.join(detail_parts)}" if detail_parts else ""
        thought_text = f" | 判断: {thought}" if thought else ""
        lines.append(f"{index}. 页面: {_runtime_page_label(page)} | 动作: {action}{detail}{thought_text}")
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


def _recent_failure_note(recent_failures: list[dict] | None) -> str:
    """把最近几次生成失败写进模型可见历史；空列表返回空串。"""
    failures = [f for f in (recent_failures or []) if f]
    if not failures:
        return ""
    entries = failures[-MAX_RECENT_FAILURES:]
    lines = []
    for failure in entries:
        reason = str(failure.get("reason", "") or "未知原因")
        attempt = int(failure.get("attempt", 1) or 1)
        attempt_text = f"（第 {attempt} 次尝试）" if attempt > 1 else ""
        lines.append(f"- 第 {failure.get('step', '?')} 步{attempt_text}生成失败：{reason}")
    body = "\n".join(lines)
    return (
        "【近期异常记录】以下步骤没有执行任何动作，页面未变化；"
        f"请据此调整输出后再决策：\n{body}\n\n"
    )


def _history_screenshot_label() -> str:
    if KEEP_IMAGE_STEPS <= 0:
        return "【历史截图】本轮不附带历史截图，仅参考上方执行历史与当前截图。"
    if KEEP_IMAGE_STEPS == 1:
        return "【上一步执行前的页面】"
    return f"【最近 {KEEP_IMAGE_STEPS} 张历史截图】"


def _recent_history_steps(trace: list[dict]) -> list[dict]:
    """最近 KEEP_IMAGE_STEPS 步；0 时返回空列表（trace[-0:] 会取全量）。"""
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
        f"【坐标约定】当前图片宽 {image_size[0]} 像素、高 {image_size[1]} 像素；"
        f"左上角为 (0, 0)，x 范围 0–{image_size[0] - 1}，y 范围 0–{image_size[1] - 1}。"
        "点击、长按及滑动均输出当前图片的像素坐标，不输出归一化坐标或手机屏幕坐标。\n"
        if image_size else "当前截图尺寸不可用，不得输出坐标操作。\n"
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
                f"{effect_note}"
                f"{repeat_note}"
                "【完整执行历史】\n"
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
        label = f"历史截图：第 {absolute_index} 步执行前的页面。"
        tap_point = _model_point(step) if str(step.get("action_name", "")).strip() in {"click", "long_press"} else None
        if tap_point is not None and effects.get(absolute_index - 1, ("", ""))[0] == "none":
            # 无效点击：把实际落点画出来，便于模型判断是点偏了还是控件无响应。
            screenshot = _mark_tap_point(screenshot, tap_point)
            label += (
                f"图中红色准星是该步实际点击位置 ({tap_point[0]},{tap_point[1]})，点击后界面无变化："
                "若准星不在目标控件上，说明坐标偏了，按准星与控件的相对位置修正后再点；"
                "若准星已落在控件中心，说明控件无响应。"
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

    try:
        parsed = _parse_json_response(text)
        if isinstance(parsed, dict):
            return parsed
    except ValueError:
        pass
    raise ValueError("模型输出中没有合法 JSON object")


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
    """解析滚动类动作的“手指方向”。

    scroll 直接读 direction；swipe 由起止点推导，使两种手势共用同一套
    滚动判据（同方向预算、意图重复、页面无变化），避免新增动作绕过检测。
    """
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
        "up": "上",
        "down": "下",
        "left": "左",
        "right": "右",
    }.get(str(direction or "").strip(), str(direction or "").strip() or "未知方向")


def _consecutive_same_direction_scroll(trace: list[dict]) -> tuple[str, int]:
    """从尾部向前统计连续同方向滚动步数（scroll 与 swipe 共用预算）。

    只统计滚动类动作（scroll/swipe）的连续段；中间插入其他动作或改变方向
    都会中断计数。尾部动作不是滚动类（或无法解析方向）时返回 0。
    """
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
    """两页交替中，前进动作是否每次都点了不同目标（“进入-返回-换下一个控件”）。

    窗口内必须出现 press_back；所有前进动作（click/long_press）都要在同一坐标系
    （优先模型图片坐标，否则屏幕坐标）下有可比较的坐标，且两两不在同一目标容差内。
    坐标缺失或任意两次前进命中同一目标都按真循环处理。
    """
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
        # 与下方 period-3 分支同理：规划模板“依次探索多个控件，进入新页面就返回”
        # 会产生 列表→点控件1→详情→返回→列表→点控件2→… 的严格两页交替，
        # 详情页在 FFG 中合并为同一 Page 节点。只有每次前进都点同一目标才是死循环。
        and not _alternation_explores_distinct_targets(recent, action_names)
    ):
        return "检测到页面导航循环"

    # M2：period-3 的通用回环检测（前三步状态与后三步状态完全相同）会把
    # "点条目→页内操作→返回，换下一个条目重复同样动作"这个正常的多条目遍历
    # 误判为死循环——因为条目详情页在 FFG 里通常合并成同一个 Page 节点，
    # 不同条目会产生相同的 page_key。真正的死循环（如页面间纯前进式互相
    # 跳转）不会在窗口内出现 press_back；而"进入-操作-返回"模式一定会。
    # 用窗口内是否出现过 press_back 区分这两种情况，不放宽 2 页严格交替
    # （上面那条分支）的判定，因为那种情况本身就只在两个页面间打转。
    if page_keys[-6:-3] == page_keys[-3:] and "press_back" not in action_names:
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
    # 最新一步是 wait：模型在回到 B 后停下观察（如“返回再进入，看进度是否续接”），
    # 不是又一次往返导航；若之后继续来回切换，下一个窗口仍会被识别。
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


def _wait_observed_progress(trace: list[dict], start: int) -> bool:
    """窗口内是否有 wait 步之后界面出现了可见变化（以下一步执行前截图为准）。"""
    for index in range(max(0, start), len(trace) - 1):
        if str(trace[index].get("action_name", "")).strip() != "wait":
            continue
        before, after = _page_screenshot(trace[index]), _page_screenshot(trace[index + 1])
        if before and after and _cached_action_effect(before, after, None)[0] in {"local", "major"}:
            return True
    return False


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
    # 等待期间界面在变（进度百分比、计时、加载内容）说明流程在推进，不是盲试；
    # 64×64 缩略图看不出这类小区域文字变化，用逐像素变化区域判断。
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
        f"连续 {NO_PROGRESS_STALL_WINDOW} 步在同一界面反复试探，"
        f"视觉状态仅 {visual_states} 种，判定当前目标不可达"
    )


def _detect_stall(trace: list[dict]) -> str:
    """停滞检测：只看动作与截图，不解析模型的自然语言。

    依次检查：页面/界面往返循环、同一界面无进展、干扰恢复往返、
    test 阶段同方向滚动预算、同一动作在界面无变化时的连续重复。
    """
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
            return "检测到目标动作与干扰恢复之间的重复循环"

    latest_action = str(trace[-1].get("action_name", "")).strip()
    if (
        latest_action in SCROLL_LIKE_ACTIONS
        and str(trace[-1].get("trace_phase", "")).strip() != "navigation"
    ):
        scroll_direction, consecutive_scrolls = _consecutive_same_direction_scroll(trace)
        if consecutive_scrolls > TEST_MAX_CONSECUTIVE_SAME_DIRECTION_SCROLLS:
            return (
                f"连续向{_format_scroll_direction(scroll_direction)}滚动"
                f"{consecutive_scrolls} 次，超出同方向滑动预算"
                f"（{TEST_MAX_CONSECUTIVE_SAME_DIRECTION_SCROLLS} 次），"
                "应转向点击列表项、返回或其他方向探索"
            )

    repeat_window = STALL_REPEAT_WINDOWS.get(latest_action, STALL_REPEAT_WINDOW)
    if len(trace) < repeat_window:
        return ""
    recent = trace[-repeat_window:]
    if _is_consecutive_repeated_action(recent):
        if latest_action == "type":
            return f"检测到连续 {repeat_window} 次重复输入"
        if latest_action in {"click", "long_press"}:
            return f"检测到连续 {repeat_window} 次重复点击同一区域"
        return f"检测到连续 {repeat_window} 次重复动作: {latest_action}"
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


def _task_phase_label(phase: str) -> str:
    """把规划器给出的功能阶段转成执行器可读的任务说明。"""
    phase = str(phase or "").strip().lower()
    if phase == "depth":
        return (
            "depth（深度业务场景）：本轮沿功能的业务逻辑走完一条完整操作，"
            "不要在进入第一个新页面时就结束"
        )
    if phase == "breadth":
        return "breadth（铺入口）：把入口与可见控件触发一遍并观察直接结果"
    return "未指定：按任务文本自行判断边界"


@dataclass
class _ScenarioRun:
    """单个场景执行在各阶段 helper 之间共享的可变状态。"""

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
    # 连续失败预算分开计：截图/屏幕尺寸等观测失败与模型生成失败互不叠加，
    # 各自连续 MAX_CONSECUTIVE_ERRORS 次结束场景。
    observation_errors: int = 0
    llm_errors: int = 0
    # 截图失败触发的应用重启次数，防止设备持续故障时无限“失败-重启”。
    screenshot_restart_count: int = 0
    action_failure_recoveries: int = 0
    # 生成失败滚动记录：写回后续每步的执行历史（agent 式循环），而非一次性反馈。
    recent_failures: list[dict] = field(default_factory=list)

    def stop(self, status: str, error_message: str) -> None:
        self.status, self.error_message = status, error_message

    def capture_final_page(self) -> None:
        try:
            self.final_page_override = capture_current_page(self.device)
        except Exception as capture_err:
            print(f"[场景执行] 恢复后抓取页面失败: {capture_err}")

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
    """一步决策所需的观测：前台状态、截图页面与图片/屏幕尺寸。"""

    page: Page
    health_error: str
    foreground_package: str
    external_context: bool
    external_package: str
    screen_size: tuple[int, int] = (0, 0)
    image_size: tuple[int, int] = (0, 0)


# 截图/屏幕尺寸等观测失败后的重试等待（秒）。
OBSERVATION_RETRY_DELAY = 2.0
# 阶段 helper 的控制结果：继续本步 / 重新观测（下一轮循环）/ 结束场景。
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
    """场景起点：安全策略、logcat 起点、冷启动与前台确认。失败时返回 (status, error)。"""
    risky_scenario_intent = wechat_read_only_violation(package_name, scenario)
    if risky_scenario_intent:
        reason = f"{WECHAT_READ_ONLY_BLOCKED_END}: {risky_scenario_intent}"
        print(f"[场景执行] {reason}")
        return "blocked", reason

    if clear_logcat:
        _clear_logcat(device)
        print(f"[场景执行] 已设置场景 logcat 起点: phase={trace_phase}")

    if compliance_cold_start:
        print(
            "[场景执行] 合规冷启动仅重启目标应用并保留数据；"
            "清除应用数据只允许在整次运行启动前执行一次。"
        )
        if not deadline_allows(deadline, APP_RESTART_MAX_SECONDS):
            return "incomplete", "阶段剩余时间不足以完成合规冷启动"
        if not _restart_target_app(device, package_name):
            return "error", "合规冷启动失败：无法重新拉起应用"
    if not _ensure_target_app(device, package_name):
        return "error", "无法拉起目标应用"
    return None


def _check_target_foreground(run: _ScenarioRun) -> tuple[str, str, str]:
    """前台检查：外部/崩溃时恢复并结束场景。

    返回 (控制结果, 前台包名, 受控外部包名)；受控外部包名非空表示当前位于
    场景允许的外部页面，继续执行。
    """
    device, package_name, trace = run.device, run.package_name, run.trace
    foreground_state, current_package = _target_app_foreground_state(device, package_name)
    if foreground_state != "foreground":
        # 页面切换动画期间可能瞬时上报 launcher（应用自身过渡态），
        # 复检确认后再定性为外部跳转/崩溃，避免误终止场景。
        foreground_state, current_package = _settle_foreground_state(device, package_name)
    if foreground_state == "foreground":
        if not current_package:
            # 未配置目标包名时前台检查不查询设备；此时单独读取一次供提示与守卫使用。
            try:
                current_package = str(device.current_package() or "").strip()
            except Exception:
                current_package = ""
        return _STEP_OK, current_package or "", ""

    scene_crash_logcat = _read_crash_logcat(device, package_name)
    # 同一份 logcat 只判定一次崩溃，后续分支与恢复流程复用该结论。
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
            f"[场景执行] 检测到受控外部应用上下文，继续执行当前场景: "
            f"current={current_package or '未知'}, target={package_name or '未知'}"
        )
        return _STEP_OK, current_package or "", current_package or ""

    print(
        f"[场景执行] 目标应用离开前台，恢复后结束当前场景: status={foreground_state}, "
        f"current={current_package or '未知'}, target={package_name or '未知'}"
    )
    if target_crashed:
        run.crash_logcat = scene_crash_logcat
        if trace:
            trace[-1]["caused_target_crash"] = True
    elif foreground_state == "background":
        _append_external_observation_step(
            trace, device, package_name, current_package or "未知", run.trace_phase,
        )
    crashed, recovered = _recover_to_target(
        trace, device, package_name, run.trace_phase, current_package or "未知",
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
    """截图异常：连续失败达到阈值时经前台轮询重启应用；预算或时间耗尽则结束。"""
    print(f"[错误] 截图失败: {error}")
    run.observation_errors += 1
    if run.observation_errors >= MAX_CONSECUTIVE_ERRORS:
        if run.screenshot_restart_count >= APP_RESTART_ATTEMPTS:
            run.stop(
                "error",
                f"连续截图失败且应用重启已达 {APP_RESTART_ATTEMPTS} 次，结束当前场景: {error}",
            )
            print(f"[场景执行] {run.error_message}")
            return _STEP_STOP
        if not run.deadline_allows(APP_RESTART_MAX_SECONDS):
            run.stop("incomplete", f"连续截图失败且阶段剩余时间不足以重启目标应用: {error}")
            return _STEP_STOP
        # 走 _restart_target_app：轮询前台包名而非固定 sleep，避免冷启动未完成
        # 时截到桌面而诱发新一轮“失败-重启”（重启风暴）。
        run.screenshot_restart_count += 1
        if _restart_target_app(run.device, run.package_name):
            run.observation_errors = 0
            return _STEP_RETRY
        run.stop("error", f"连续截图失败且重启目标应用失败: {error}")
        return _STEP_STOP
    if not run.deadline_allows(OBSERVATION_RETRY_DELAY):
        run.stop("incomplete", f"截图失败且阶段剩余时间不足以等待重试: {error}")
        return _STEP_STOP
    time.sleep(OBSERVATION_RETRY_DELAY)
    return _STEP_RETRY


def _capture_step_page(run: _ScenarioRun) -> tuple[str, Page | None, str]:
    """抓取当前运行时页面，返回 (控制结果, 页面, 截图健康错误)。"""
    try:
        page, page_health_error = capture_runtime_page(
            run.device,
            page_id=run.initial_page_id if not run.trace else None,
        )
    except ScreenshotUnavailableError as e:
        print(f"[场景执行] 完全无法取得运行时截图，立即停止当前场景: {e}")
        run.stop("screenshot_unavailable", f"无法取得运行时截图，当前场景已停止: {e}")
        return _STEP_STOP, None, ""
    except Exception as e:
        return _handle_screenshot_failure(run, e), None, ""
    if page_health_error:
        print(f"[场景执行] 运行时观察到异常截图: {page_health_error}")
        if run.trace:
            _append_unavailable_observation(
                run.trace, page, page_health_error, run.trace_phase,
            )
            run.stop(
                "screenshot_unavailable",
                f"观察到黑屏或页面不可用，已保留当前界面并结束场景: {page_health_error}",
            )
            run.final_page_override = page
            return _STEP_STOP, page, page_health_error
    return _STEP_OK, page, page_health_error


def _handle_exploration_boundary(
    run: _ScenarioRun,
    page: Page,
    foreground_package: str,
) -> bool:
    """探索边界检测（core 模式）；命中时自动退出并结束场景，返回是否已结束。

    目标应用内进入第三方服务内容（如小程序、内嵌网页等）时自动退出并结束
    当前场景。规则按包名从配置加载，未配置的应用不启用，行为与无本检测时完全一致。
    """
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
        f"[探索边界] 检测到进入{boundary_label}"
        f"（via={boundary_hit.via}, matched={boundary_hit.matched}），"
        "自动退出并结束当前场景。"
    )
    reason = (
        f"进入{boundary_label}越过探索边界"
        f"（via={boundary_hit.via}, matched={boundary_hit.matched}）"
    )
    if _recover_from_exploration_boundary(device, package_name, boundary_hit):
        _append_target_app_recovery_step(
            trace, device, package_name, package_name, run.trace_phase,
            recovery_method="press_back", reason=f"{reason}；已通过返回键退出。",
        )
        run.stop("blocked", EXPLORATION_BOUNDARY_BLOCKED_END.format(label=boundary_label))
        return True
    print("[探索边界] 返回退出失败，重启目标应用并结束本轮场景。")
    run.crash_logcat = _read_crash_logcat(device, package_name)
    crashed, recovered = _recover_to_target(
        trace, device, package_name, run.trace_phase, package_name, run.crash_logcat,
        back=None, reason=f"{reason}且返回退出失败；已重启目标应用。",
    )
    if recovered and not crashed:
        run.stop("left_app", EXPLORATION_BOUNDARY_BLOCKED_END.format(label=boundary_label))
    else:
        run.stop(*_recovery_outcome(crashed, recovered))
    if recovered:
        run.capture_final_page()
    return True


def _observe_step(run: _ScenarioRun) -> tuple[str, _StepObservation | None]:
    """观测阶段：前台检查 → 截图 → 探索边界 → 屏幕与图片尺寸。"""
    try:
        control, foreground_package, external_package = _check_target_foreground(run)
    except ScreenshotUnavailableError as e:
        print(f"[场景执行] 完全无法取得运行时截图，立即停止当前场景: {e}")
        run.stop("screenshot_unavailable", f"无法取得运行时截图，当前场景已停止: {e}")
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
            raise ValueError("当前上传截图无法解码或尺寸无效")
    except Exception as e:
        print(f"[错误] 获取屏幕或截图尺寸失败: {e}")
        run.observation_errors += 1
        if run.observation_errors >= MAX_CONSECUTIVE_ERRORS:
            run.stop("error", f"连续获取屏幕尺寸失败: {e}")
            return _STEP_STOP, None
        if not run.deadline_allows(OBSERVATION_RETRY_DELAY):
            run.stop("incomplete", f"获取屏幕尺寸失败且阶段剩余时间不足以等待重试: {e}")
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
    """模型生成失败计数与退避；预算耗尽或剩余时间不足时结束场景。"""
    run.llm_errors += 1
    if run.llm_errors >= MAX_CONSECUTIVE_ERRORS:
        run.stop("error", error_message)
        return _STEP_STOP
    if not _sleep_for_llm_retry(run.llm_errors, run.deadline):
        run.stop("incomplete", f"阶段剩余时间不足以等待模型重试（{error_message}）")
        return _STEP_STOP
    return _STEP_RETRY


def _decide_action(
    run: _ScenarioRun,
    system_msg: SystemMessage,
    observation: _StepObservation,
    step_index: int,
) -> tuple[str, dict | None, str, str]:
    """决策阶段：构造输入、调用模型并解析校验动作。

    返回 (控制结果, plan, 原始响应, 动作前截图引用)。
    """
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
        "[场景执行] LLM 输入摘要: "
        f"phase={run.trace_phase}, step={step_index}, history_steps={len(trace)}, "
        f"recent_screenshots={_recent_screenshot_count(trace)}, "
        f"current_page={_runtime_page_label(page)}, activity={page.activity or '未知'}, "
        f"foreground={observation.foreground_package or '未知'}, "
        f"current_screenshot={'yes' if b64_image else 'no'}"
    )

    llm = run.llm
    raw_content = ""
    response = None
    request_started = time.monotonic()
    try:
        record_llm_request(llm, operation="[场景执行]")
        response = llm.invoke(messages)
        record_llm_response(llm, response, operation="[场景执行]")
        raw_content = normalize_llm_text(getattr(response, "content", ""))
        if not raw_content:
            raise ValueError("empty_content: 响应正文为空")
        if (getattr(response, "response_metadata", {}) or {}).get("finish_reason") == "length":
            raise ValueError("truncated_output: 响应被截断")
        plan = _parse_action_json(raw_content)
        plan = _normalize_action_plan(plan)
        _validate_action_plan(plan, observation.image_size)
        record_llm_diagnostic(response, operation="[场景执行]", elapsed_seconds=time.monotonic() - request_started)
        log_plan = dict(plan)
        log_plan.pop("action_description", None)
        print(f"[场景执行] LLM 输出 JSON: {log_plan}")
    except Exception as e:
        # 服务端不识别 response_format 时，属配置问题而非模型错误：
        # 关闭 JSON 模式后立即重试，不计入错误预算，也不写入模型可见的失败记录
        # （模型无法修正配置问题，写进历史只会干扰决策）。
        if json_format_rejected(e) and disable_json_response_format(llm):
            print(
                "[场景执行] 服务端不接受 response_format=json_object，"
                "已自动关闭 JSON 模式并重试（不计入错误预算）。"
            )
            return _STEP_RETRY, None, raw_content, trace_screenshot_before
        record_llm_diagnostic(response, operation="[场景执行]", error=f"{type(e).__name__}: {e}", elapsed_seconds=time.monotonic() - request_started)
        record_llm_failure(llm, operation="[场景执行]")
        no_json_hint = " (响应零JSON结构)" if raw_content and "{" not in raw_content else ""
        print(f"[错误] LLM调用或解析失败{no_json_hint}: {e}\n[原始内容]: {raw_content}")
        run.record_generation_failure(
            step_index,
            _describe_generation_failure(raw_content, e, response_received=response is not None),
        )
        control = _handle_generation_failure(run, f"连续 LLM 解析失败: {e}")
        return control, None, raw_content, trace_screenshot_before
    return _STEP_OK, plan, raw_content, trace_screenshot_before


def _tap_point_on_flat_background(screenshot: str, x: int, y: int) -> bool:
    """落点 8px 邻域没有局部纹理 → 大概率点在控件之外的背景上。

    先减去高斯模糊得到高频分量再算标准差：渐变背景（灰度缓慢变化）原始标准差
    可能不小，但高频分量接近 0；控件的文字、边缘、图标会产生明显的高频分量。
    """
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
    """裁剪放大落点附近区域，请模型对意图中的控件再定位一次。

    只返回落在裁剪区内、与原落点距离不超过 TAP_REFINE_MAX_SHIFT 的新坐标；
    模型找不到、调用失败或结果可疑时返回 None，调用方保留原坐标。
    """
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
        f"这是手机截图的局部放大图（宽 {crop_w} 像素、高 {crop_h} 像素，左上角为 (0, 0)）。"
        f"执行意图：{intent}。找到意图要点击的那个控件，给出其可点击本体中心在这张放大图中的"
        "像素坐标；图中没有该控件时 found 为 false。只输出 JSON："
        '{"found": true, "x": 0, "y": 0}'
    )
    try:
        payload = encode_image_to_jpg_base64(crop, max_side=max(crop_w, crop_h), quality=85)
        caller = llm.bind(timeout=TAP_REFINE_TIMEOUT_SECONDS) if hasattr(llm, "bind") else llm
        record_llm_request(llm, operation="[点击定位]")
        request_started = time.monotonic()
        response = caller.invoke([HumanMessage(content=[
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{payload}"}},
        ])])
        # 与主执行路径同口径落盘 llm_responses.jsonl：只记计数不落响应，
        # 离线复盘时就算不出二次定位的真实耗时、也看不到模型输出。
        record_llm_diagnostic(response, operation="[点击定位]", elapsed_seconds=time.monotonic() - request_started)
        record_llm_response(llm, response, operation="[点击定位]")
        result = _parse_action_json(normalize_llm_text(getattr(response, "content", "")))
    except Exception as exc:  # noqa: BLE001 - 二次定位只是优化，失败沿用原坐标
        record_llm_failure(llm, operation="[点击定位]")
        record_llm_diagnostic(response=None, operation="[点击定位]", error=f"{type(exc).__name__}: {exc}")
        print(f"[场景执行] 点击二次定位失败，沿用原坐标: {type(exc).__name__}: {exc}")
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
    """上一步在几乎同一位置点击后界面无变化：落点大概率擦在控件边缘或之外。"""
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
    """click/long_press 可能点偏时做一次放大二次定位，原地修正 plan 坐标。

    触发条件（都不需要模型调用即可判断）：落点在无纹理背景上；或上一步在几乎
    同一位置点击后界面无变化（落点擦在控件边缘时纹理不低，只能靠这一条兜住）。
    """
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
        trigger = "落点位于空白背景"
    elif _repeats_ineffective_tap(run.trace, screenshot, x, y):
        trigger = "重复上一步无效点击的位置"
    else:
        return
    refined = _refine_tap_point(run.llm, screenshot, plan)
    if refined is None or refined == (x, y):
        return
    print(f"[场景执行] {trigger}，放大二次定位修正坐标: ({x},{y}) -> ({refined[0]},{refined[1]})")
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
    """动作已记入轨迹、尚未下发设备前的守卫：验证挑战、未处理系统页、停滞、
    finished 结论与登录态。返回 _STEP_OK 表示可执行动作。"""
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
        print(f"[场景执行] 动作执行前拦截: {verification_attempt_reason}")
        return _STEP_STOP

    unresolved_external_reason = _unfinished_safe_external_reason(
        action_name,
        observation.foreground_package,
    )
    if unresolved_external_reason:
        step_info["action_failed"] = True
        step_info["action_error"] = unresolved_external_reason
        print(f"[场景执行] 拒绝提前结束: {unresolved_external_reason}")
        return _STEP_RETRY

    stall_reason = _detect_stall(run.trace)
    if stall_reason:
        print(f"[警告] {stall_reason}，提前结束当前场景。")
        end_step("stalled", stall_reason)
        return _STEP_STOP

    if action_name == "finished":
        if observation.health_error:
            end_step("screenshot_unavailable", "执行器在黑屏或不可用界面结束，等待下一轮恢复")
            run.final_page_override = observation.page
            return _STEP_STOP
        verification_finish_reason = _verification_finish_reason(thought)
        if verification_finish_reason:
            end_step("blocked", verification_finish_reason)
            print(f"[场景执行] 验证挑战阻塞当前功能: {thought[:160]}")
        elif _finished_declares_failure(thought):
            end_step("failed_goal", f"执行器判断当前目标不可达: {thought[:240]}")
            print(f"[场景执行] finished 包含失败语义，降级为 failed_goal: {thought[:160]}")
        return _STEP_STOP

    # 只看动作本身（action_description，缺失时取 thought 的动作子句），不看完整
    # thought：thought 常复述“保持登录态、不点退出登录”之类约束，会误伤正常动作。
    session_violation = account_session_action_violation(action_name, proposed_action)
    if session_violation:
        print(
            "[场景执行] 硬拦截会破坏登录态的动作: "
            f"{session_violation} | {(plan.get('action_description') or '')[:80]}"
        )
        end_step(
            "failed_goal",
            "运行器拦截了会退出登录/切换账号的动作"
            f"（{session_violation}）；已保持登录态结束本场景",
        )
        step_info["session_guard_blocked"] = True
        return _STEP_STOP
    return _STEP_OK


def _execute_planned_action(run: _ScenarioRun, plan: dict, step_info: dict) -> str:
    """执行阶段：下发设备动作并按动作类型等待界面稳定。"""
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
            print(f"[场景执行] 动作参数: {action_kwargs}")
        try:
            action_result = _perform_action(device=run.device, action_name=action_name, **action_kwargs)
            if action_name in {"click", "long_press"} and isinstance(action_result, dict):
                for key in ("x", "y"):
                    if key in action_result:
                        step_info[key] = action_result[key]
        except Exception as e:
            print(f"[错误] 执行动作失败: {e}")
            step_info["is_interruption"] = True
            step_info["action_failed"] = True
            step_info["action_error"] = str(e)
            if run.action_failure_recoveries < MAX_ACTION_FAILURE_RECOVERIES:
                run.action_failure_recoveries += 1
                print(
                    "[场景执行] 将动作失败及原因交给执行模型，允许改用一次替代动作继续当前场景。"
                )
                return _STEP_RETRY
            run.stop("error", f"动作执行失败且替代策略仍失败，已结束当前场景: {e}")
            return _STEP_STOP

    if action_name == "wait":
        # wait 携带显式时长时按模型要求等待，避免“等 1 秒不够 → 再等一次”
        # 被 WAIT_STALL_REPEAT_WINDOW 误判为重复动作而提前结束场景。
        wait_seconds = max(
            MIN_WAIT_SECONDS,
            min(MAX_WAIT_SECONDS, float(plan.get("duration") or DEFAULT_WAIT_SECONDS)),
        )
        if wait_seconds > ACTION_DELAY:
            print(f"[场景执行] 显式等待 {wait_seconds:.1f}s（等待页面加载）")
        time.sleep(wait_seconds)
    elif action_name == "open_app" and package_name:
        # 冷启动轮询代替固定延时：避免在应用仍在启动时截图，
        # 让执行模型看到桌面/启动页而误判重复 open_app。
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
    """执行一步：观测 → 决策 → 守卫 → 执行。返回控制结果。"""
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
        run.stop("incomplete", "模型返回时阶段时间预算已耗尽，未执行其建议动作")
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
        print(f"[场景执行] 动作执行前拦截: {run.error_message}")
        return _STEP_STOP

    print(
        f"\n[步骤 {step_index}]{' [干扰]' if is_interruption else ''} "
        f"页面={observation.page.id or '未知'} 动作={action_name} | 思考: {thought}"
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
        print(f"[错误] LLM坐标解析失败: {e}\n[原始内容]: {raw_content}")
        run.record_generation_failure(step_index, f"坐标换算失败，动作未执行：{str(e)[:300]}")
        return _handle_generation_failure(run, f"连续 LLM 坐标解析失败: {e}")

    # 本步得到了可执行动作：重置模型错误预算，并清空已被纠正的失败记录，
    # 避免过期警告持续出现在后续步骤的历史里。
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
    """收尾：退出阻塞页、补终止观测、恢复目标应用前台、补存截图。"""
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
                "最后一个有副作用动作后的页面不可用，已结束当前场景: "
                f"{terminal_health_error}",
            )

    # 场景结束时保证目标应用在前台（受控外部跳转等情形），超时结束的场景不再额外恢复。
    if package_name and run.status not in FOREGROUND_HANDLED_STATUSES and run.status != "incomplete":
        foreground_state, current_package = _target_app_foreground_state(device, package_name)
        if foreground_state != "foreground":
            scene_end_logcat = _read_crash_logcat(device, package_name)
            crashed, recovered = _recover_to_target(
                trace, device, package_name, trace_phase, current_package or "未知",
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
        "[轨迹截图] 场景截图索引完成: "
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
    """执行一个场景，返回 (trace, final_page, status, error, crash_logcat)。

    status 取值见 EXECUTION_STATUSES。每步依次经过 观测（前台/截图/边界/尺寸）
    → 决策（模型调用与解析）→ 守卫（安全策略、停滞、finished 结论）→ 执行。
    """
    phase_label = {"navigation": "导航", "test": "场景"}.get(
        trace_phase, trace_phase
    )
    step_limit = (
        MAX_NAVIGATION_STEPS if trace_phase == "navigation" else MAX_EXEC_STEPS
    )
    if not scenario:
        print(f"[场景执行] {phase_label}阶段跳过: 未配置场景。")
        return [], initial_page, "skipped", "", ""
    if deadline is not None and time.time() >= deadline:
        return [], initial_page, "incomplete", "阶段时间预算已耗尽", ""

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
        "[场景执行] 场景起点: "
        f"phase={trace_phase}, target={package_name or '未知'}, "
        f"pid={target_pid or '未知'}, started_at={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(scene_started_at))}"
    )

    system_prompt = SCENARIO_EXEC_SYSTEM_PROMPT_TEMPLATE.format(
        # 合规专项要求是固定内容，同样放在动态任务信息之前，保持可缓存前缀。
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
        print("[场景执行] 当前场景允许受控外部应用跳转；场景结束时会恢复目标应用。")
    allows_target_exit = _scenario_allows_target_exit(scenario)
    if allows_target_exit:
        print("[场景执行] 当前场景允许目标应用退出前台；场景结束时会恢复目标应用。")

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

    print(f"[场景执行] 开始{phase_label}阶段: {scenario}")
    if isinstance(initial_page, Page):
        print(f"[场景执行] 起始页面: {initial_page.id or '未知'}")

    while True:
        if deadline is not None and time.time() >= deadline:
            run.stop("incomplete", "阶段时间预算已耗尽，停止新动作并保存最终观测")
            break
        if len(run.trace) >= step_limit:
            print(f"[警告] 达到最大步数限制 ({step_limit})，强制结束。")
            run.stop("incomplete", f"达到最大步数限制 {step_limit}")
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
        f"[场景执行] {phase_label}阶段完成 | status={run.status} | "
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
            """连续黑屏计数；达到阈值时重启应用，重启仍失败且配置允许时结束整次运行。"""
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
                f"[场景执行] {trace_phase} 阶段截图不可用，"
                f"{'结束当前 app 运行' if workflow_complete else '当前场景结束；后续探索其它功能'}: "
                f"{count}/{MAX_CONSECUTIVE_SCREENSHOT_UNAVAILABLE}"
            )
            return page, count, restart_error, workflow_complete

        if package_name and not _ensure_target_app(device, package_name):
            return {
                "navigation_execution_trace": [],
                "scenario_execution_trace": [],
                "navigation_failure_count": navigation_failure_count + 1,
                "scenario_exec_error_count": 0,
                "execution_result": execution_result("error", "目标应用无法拉起", []),
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

        # 导航未完成：本轮跳过测试场景，由 navigation_status 说明原因。
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
                # 已在执行中恢复过目标应用，不再计入导航失败。
                navigation_failure_count = 0
            else:
                navigation_failure_count += 1
                print(f"[场景执行] 导航失败，本轮跳过执行场景。连续失败次数: {navigation_failure_count}")
                if navigation_failure_count >= 2:
                    print("[场景执行] 连续两次导航失败，重启 app 回到初始页面。")
                    if package_name and _restart_target_app(device, package_name):
                        try:
                            final_page = capture_current_page(device)
                        except Exception as e:
                            print(f"[场景执行] 重启后抓取页面失败: {e}")
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
                "[场景执行] 合规检测场景: "
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
