"""统一场景规划节点。"""

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
    FlowGraph,
    FunctionNode,
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
    APP_RESTART_SETTLE_DELAY,  # noqa: F401  保留原模块名供引用/测试
    APP_START_MAX_WAIT,
    APP_START_POLL_INTERVAL,  # noqa: F401  保留原模块名供引用/测试
    restart_until_foreground,
    wait_target_app_foreground,
)
from workflow.exploration_boundary import rule_for
from workflow.nodes.ffg_update import BLACK_SCREEN_PAGE_NAME, apply_ffg_patch
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
# 每个规划轮次最多对几个已完成功能发起覆盖遗漏审计（每次审计可能是多批带图
# 大模型调用）；其余功能留到后续轮次，避免收尾时单轮串行审计全部功能。
PLAN_MAX_AUDITS_PER_ROUND = max(1, int(os.getenv("PLAN_MAX_AUDITS_PER_ROUND", "2")))
MAX_CROSS_TEST_PAIRS = int(os.getenv("PLAN_MAX_CROSS_TEST_PAIRS", "20"))
# 语义校验重试次数；底层 invoke_llm_json_with_retry 自身还会做传输/解析重试，
# 两者相乘，因此外层默认只给 2 次。
PLAN_VALIDATION_RETRY_ATTEMPTS = max(1, int(os.getenv("PLAN_VALIDATION_RETRY_ATTEMPTS", "2")))
# 作用域摘要中每个功能只展示最近 K 个已测试场景，更早的只给数量汇总，
# 避免规划提示词随运行时长无界增长。
PLAN_SCOPE_RECENT_SCENARIOS = max(1, int(os.getenv("PLAN_SCOPE_RECENT_SCENARIOS", "20")))
# 作用域摘要中最多展示的相关页面跳转条数（作用域内部跳转优先）。
PLAN_SCOPE_MAX_TRANSITIONS = max(1, int(os.getenv("PLAN_SCOPE_MAX_TRANSITIONS", "40")))
# 单个补测场景一次覆盖的控件族数量。每轮的固定开销（规划 + FFG 更新的带图
# 大模型调用 + 截图 + logcat）与场景长度无关，一次只探一个控件会让固定开销
# 摊在 1-2 个有效动作上。
PLAN_SCENARIO_CONTROL_BATCH = max(
    1,
    int(os.getenv("PLAN_SCENARIO_CONTROL_BATCH", "4")),
)
# 程序派生的控件/操作覆盖率达到该值后进入完成与遗漏审计。
PLAN_FUNCTION_COVERAGE_TARGET = max(
    1,
    min(100, int(os.getenv("PLAN_FUNCTION_COVERAGE_TARGET", "80"))),
)
# 功能内两阶段：先广度铺入口，入口铺开后沿业务逻辑做深度场景。
# 广度阶段的计数闸门——做满这么多个广度场景就转深度，否则单功能会一直铺入口。
PLAN_BREADTH_MAX_SCENARIOS = max(
    1,
    int(os.getenv("PLAN_BREADTH_MAX_SCENARIOS", "3")),
)
# 深度场景数随功能规模增长：每 PLAN_DEPTH_CONTROLS_PER_SCENARIO 个控件需要 1 个，
# 结果夹在 [PLAN_MIN_DEPTH_SCENARIOS, PLAN_MAX_DEPTH_SCENARIOS]。
PLAN_DEPTH_CONTROLS_PER_SCENARIO = max(
    1,
    int(os.getenv("PLAN_DEPTH_CONTROLS_PER_SCENARIO", "5")),
)
PLAN_MIN_DEPTH_SCENARIOS = max(0, int(os.getenv("PLAN_MIN_DEPTH_SCENARIOS", "1")))
PLAN_MAX_DEPTH_SCENARIOS = max(
    PLAN_MIN_DEPTH_SCENARIOS,
    int(os.getenv("PLAN_MAX_DEPTH_SCENARIOS", "6")),
)
# 单功能场景上限 = 广度闸门 + 深度场景数 + 余量；达到后满足完成判定则 completed，
# 否则 deferred，避免一个庞大的功能长期占用执行权。
PLAN_SCENARIO_CAP_MARGIN = max(0, int(os.getenv("PLAN_SCENARIO_CAP_MARGIN", "2")))
# 跨功能组合因导航失败/截图不可用而 retryable 时，同一组合最多尝试的次数。
PLAN_MAX_CROSS_ATTEMPTS = max(1, int(os.getenv("PLAN_MAX_CROSS_ATTEMPTS", "2")))
# 规划器与程序模板连续多少轮给不出场景才延后功能（单轮空场景常是模型偶发）。
PLAN_UNPLANNABLE_ROUNDS_BEFORE_DEFER = max(
    1, int(os.getenv("PLAN_UNPLANNABLE_ROUNDS_BEFORE_DEFER", "2"))
)
FUNCTION_PHASE_BREADTH = "breadth"
FUNCTION_PHASE_DEPTH = "depth"
# 导航参考路径的级别（导航只是参考，不作为排程闸门）：
# 空=当前页已在作用域内，无需导航；A=只走已记录正向边；B=允许把已记录边反向
# 当作“返回”；C=先重开应用回到首页再走 A/B；D=图中找不到路径，只给目标描述，
# 由执行器探索式导航，失败一次即搁置。
NAV_LEVEL_HERE = ""
NAV_LEVEL_FORWARD = "A"
NAV_LEVEL_BACKTRACK = "B"
NAV_LEVEL_RESTART = "C"
NAV_LEVEL_EXPLORE = "D"
KNOWN_NAV_LEVELS = {NAV_LEVEL_HERE, NAV_LEVEL_FORWARD, NAV_LEVEL_BACKTRACK, NAV_LEVEL_RESTART}
# 当前只支持 core 探索范围（EXPLORATION_MODE_INSTRUCTIONS 仅含 core）。
PLAN_EXPLORATION_MODE = "core"
ENTRY_FUNCTION_NAME_PATTERN = re.compile(r"入口\s*$")

COMPLIANCE_GROUP_DISCOVERY_TERMS: dict[str, tuple[str, ...]] = {
    "first_launch_flow": ("隐私", "协议", "同意", "拒绝", "开屏"),
    "permission_denial_flow": ("权限", "相机", "定位", "位置", "通讯录", "相册", "存储", "麦克风"),
    "consent_observation": ("敏感", "人脸", "位置", "通讯录", "健康", "支付", "会员", "订阅", "续费", "青少年"),
    "registration_form": ("登录", "注册", "账号", "手机号", "资料", "实名认证", "个人信息"),
    "settings_search": ("设置", "隐私", "推荐", "推送", "权限", "自启动", "设备标识", "资费"),
    "rights_entry_search": ("设置", "账号", "注销", "删除", "更正", "撤回", "客服", "投诉", "反馈", "帮助"),
    "policy_text_review": ("隐私政策", "隐私", "用户协议", "个人信息", "第三方", "共享清单"),
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
# APP_RESTART_RETRY_DELAY / APP_START_* / APP_RESTART_SETTLE_DELAY 与执行节点同口径，
# 统一定义在 workflow/app_foreground.py：重启后轮询前台包名直到目标应用可见，
# 避免固定 sleep 在重型应用（如微信）冷启动期间截到桌面截图。
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
    """规划节点的前台探测：单次读取，读取失败按非前台处理（不重试、不刷日志）。"""
    try:
        current = str(device.current_package() or "").strip()
    except Exception:
        current = ""
    return ("foreground" if current == package_name else "background"), current


def _wait_target_app_foreground(device, package_name: str) -> bool:
    """轮询等待目标应用进入前台；上限 APP_START_MAX_WAIT 秒。"""
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
        print(f"[场景规划] 获取当前前台包名失败: {exc}")
        current_package = ""

    if is_permission_dialog(current_package):
        print("[场景规划] 系统权限弹窗覆盖目标应用，保留现场截图并交给执行器处理。")
        return capture_current_page(device)

    needs_reset = False
    if current_package and current_package != package_name:
        print(
            f"[场景规划] 当前前台应用为 {current_package}，目标应用为 {package_name}，"
            "准备重启目标应用。"
        )
        needs_reset = True
    elif current_page is None and current_package == package_name:
        print("[场景规划] 当前页面为空，目标应用已在前台，直接抓取页面。")
        try:
            return capture_current_page(device)
        except ScreenshotUnavailableError as exc:
            print(f"[场景规划] 前台截图不可用，先通过返回键恢复: {exc}")
            for attempt in range(1, PLAN_SCREENSHOT_BACK_ATTEMPTS + 1):
                try:
                    print(
                        "[场景规划] 黑屏恢复："
                        f"第 {attempt}/{PLAN_SCREENSHOT_BACK_ATTEMPTS} 次按返回键。"
                    )
                    device.back()
                    time.sleep(APP_RESTART_RETRY_DELAY)
                    recovered_package = str(device.current_package() or "").strip()
                    if recovered_package != package_name:
                        print(
                            "[场景规划] 返回后目标应用已离开前台，转入重启恢复: "
                            f"current={recovered_package or '未知'}"
                        )
                        break
                    return capture_current_page(device)
                except ScreenshotUnavailableError as back_exc:
                    print(f"[场景规划] 返回后截图仍不可用: {back_exc}")
                except Exception as back_exc:
                    print(f"[场景规划] 返回恢复失败: {back_exc}")
            needs_reset = True
    elif current_page is None:
        print("[场景规划] 无法确认目标应用在前台，准备重启目标应用。")
        needs_reset = True

    if not needs_reset:
        return current_page

    restarted, page, errors = restart_until_foreground(
        device,
        package_name,
        wait_foreground=lambda dev, pkg: _wait_target_app_foreground(dev, pkg),
        on_ready=lambda: capture_current_page(device),
        attempts=PLAN_APP_RESTART_ATTEMPTS,
        log_prefix="[场景规划]",
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
        print(f"[场景规划] 抓取当前页面失败: {exc}")
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
    """跨功能测试只接受未处于失败/崩溃/让出关闭状态的功能。"""
    if node is None:
        return False
    return (node.test_status or "not_started") not in {
        "blocked",
        "deferred",
        "unavailable",
    }


def _node_intra_attempt_count(node: FunctionNode | None) -> int:
    """实际执行次数；旧快照的去重场景数仅作为下界。"""
    return node.intra_attempts() if node is not None else 0


def _node_depth_count(node: FunctionNode | None) -> int:
    """本功能已进入测试阶段的深度场景数（测试阶段失败也计入）。"""
    return int(node.depth_scenario_count or 0) if node is not None else 0


def _node_breadth_count(node: FunctionNode | None) -> int:
    return max(0, _node_intra_attempt_count(node) - _node_depth_count(node))


def _required_depth_count(graph: FlowGraph, node: FunctionNode | None) -> int:
    """本功能需要的深度场景数：每 PLAN_DEPTH_CONTROLS_PER_SCENARIO 个控件 1 个。"""
    if node is None:
        return PLAN_MIN_DEPTH_SCENARIOS
    controls = len(graph.function_controls(node.id))
    required = -(-controls // PLAN_DEPTH_CONTROLS_PER_SCENARIO)
    return max(PLAN_MIN_DEPTH_SCENARIOS, min(PLAN_MAX_DEPTH_SCENARIOS, required))


def _function_scenario_cap(graph: FlowGraph, node: FunctionNode | None) -> int:
    """单功能场景上限 = 广度闸门 + 深度场景数 + 余量；随控件增加而上调。"""
    return (
        PLAN_BREADTH_MAX_SCENARIOS
        + _required_depth_count(graph, node)
        + PLAN_SCENARIO_CAP_MARGIN
    )


def _function_phase(graph: FlowGraph, node: FunctionNode | None) -> str:
    """功能当前所处阶段：广度铺入口 / 深度业务场景。

    depth：深度场景数未达目标，且入口已铺开（广度场景数达到闸门，或控件已全部覆盖）。
    其余情况都是 breadth：深度做够之后缺的只会是覆盖率，回到广度补控件。
    """
    if node is None:
        return FUNCTION_PHASE_BREADTH
    if node.successful_depth_scenario_count >= _required_depth_count(graph, node):
        return FUNCTION_PHASE_BREADTH
    if _node_breadth_count(node) >= PLAN_BREADTH_MAX_SCENARIOS:
        # 计数闸门优先于覆盖事实：保证广度阶段有界，不被“控件永远覆盖不完”拖住。
        return FUNCTION_PHASE_DEPTH
    details = graph.function_coverage_details(node.id)
    if int(details["total_controls"] or 0) > 0 and not details["uncovered_controls"]:
        return FUNCTION_PHASE_DEPTH
    return FUNCTION_PHASE_BREADTH


def _describe_function_phase(graph: FlowGraph, node: FunctionNode | None) -> str:
    """给规划模型看的阶段与预算摘要，也用在同一行日志里。"""
    if node is None:
        return "无目标功能"
    phase = _function_phase(graph, node)
    uncovered = len(graph.function_coverage_details(node.id)["uncovered_controls"])
    return (
        f"{phase} | 广度 {_node_breadth_count(node)}/{PLAN_BREADTH_MAX_SCENARIOS}"
        f" | 深度 {_node_depth_count(node)}/{_required_depth_count(graph, node)}"
        f"（有效完成 {node.successful_depth_scenario_count}）"
        f" | 未覆盖控件 {uncovered}"
        f" | 场景 {_node_intra_attempt_count(node)}/{_function_scenario_cap(graph, node)}"
    )


def _cross_pair_ready(graph: FlowGraph, src_id: str, dst_id: str) -> bool:
    """两端功能都已有自身场景和成功业务操作，跨功能闭环才有可复用的前置。"""
    return all(
        graph.get_node(node_id) is not None
        and graph.function_has_successful_business_operation(node_id)
        and _node_intra_attempt_count(graph.get_node(node_id)) > 0
        for node_id in (src_id, dst_id)
    )


def _function_completion_met(graph: FlowGraph, node: FunctionNode) -> bool:
    """完成判定：覆盖率达标 + 有业务操作 + 有效完成的深度场景数达标。

    覆盖率可能由共享页面的兄弟功能“继承”，但深度场景只在本功能的 depth 阶段产生，
    而进入 depth 要求本功能自身铺过入口，所以深度场景数本身就是本功能自身的证据。
    """
    return (
        node.coverage_percent >= PLAN_FUNCTION_COVERAGE_TARGET
        and graph.function_has_successful_business_operation(node.id)
        and node.successful_depth_scenario_count >= _required_depth_count(graph, node)
    )


def _update_function_statuses(graph: FlowGraph) -> None:
    """每轮按 FFG 统一迁移功能状态；之后的调度不再改状态。"""
    reopened = _normalize_and_reopen_undercovered_functions(graph)
    if reopened:
        print("[场景规划] 已完成节点按程序派生覆盖率/深度目标恢复待测: " + ", ".join(reopened))
    graph.recompute_all_coverages()
    completed: list[str] = []
    capped: list[str] = []
    for node in list(graph.functions.values()):
        status = node.test_status or "not_started"
        # deferred 只表示“暂未满足完成判定”，事后补齐证据时仍可转为 completed。
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
            "[场景规划] 完成判定通过，标记 completed 并等待遗漏审计: "
            + ", ".join(f"{fid} | {_function_label(graph.get_node(fid))}" for fid in completed)
        )
    if capped:
        print(
            "[场景规划] 达到单功能场景上限但未满足完成判定，标记 deferred: "
            + ", ".join(f"{fid} | {_function_label(graph.get_node(fid))}" for fid in capped)
        )


def _normalize_and_reopen_undercovered_functions(graph: FlowGraph) -> list[str]:
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
        # 被规划过却没有留下任何控件与操作记录：其 completed 状态没有任何
        # 可核对的覆盖依据，需要重新审计。未被规划过的节点不在此列，避免抢占
        # 真正未开始功能的广度优先调度权。
        never_observed = (
            bool(node.scenarios)
            and int(coverage_details["total_controls"] or 0) == 0
            and not graph.function_executed_operation_keys(node.id)
        )
        # 重开必须由“程序派生的覆盖事实”驱动：仍有未操作控件，或该功能名下
        # 根本没有记录过任何控件（说明它从未被真正观察过）。与覆盖无关的启发式
        # （例如入口命名 + 页面数量）会和完成判定互相翻转，白烧调度轮次。
        needs_reopen = (
            (has_uncovered or never_observed)
            and node.coverage_percent < PLAN_FUNCTION_COVERAGE_TARGET
        )
        # 旧 completed 快照同样必须有有效深度证据；历史尝试数不能追认为成功数。
        needs_more_depth = node.successful_depth_scenario_count < _required_depth_count(graph, node)
        # 已达场景上限的功能不再重开，剩余入口交给覆盖遗漏审计。
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


def _unfinished_function_nodes(graph: FlowGraph) -> list[FunctionNode]:
    return [node for node in graph.functions.values() if not _node_is_terminal(node)]


def _function_sort_index(node: FunctionNode) -> int:
    match = re.search(r"\d+", str(node.id or ""))
    return int(match.group(0)) if match else 10**9


def _intra_sort_key(
    graph: FlowGraph,
    node: FunctionNode,
    current_owner_ids: list[str],
    failure_counts: dict[str, int],
    navigation_level: str = NAV_LEVEL_HERE,
) -> tuple:
    """功能内目标的唯一排序键。

    无失败优先 → depth 阶段优先（深度场景尽快排上）→ 当前页面所属功能优先（少导航）
    → 有已知路径优先（D 级探索式导航排后）→ 未尝试优先 → 尝试少优先 → 功能编号。
    """
    return (
        0 if int(failure_counts.get(node.id, 0) or 0) == 0 else 1,
        0 if _function_phase(graph, node) == FUNCTION_PHASE_DEPTH else 1,
        0 if node.id in current_owner_ids else 1,
        0 if navigation_level in KNOWN_NAV_LEVELS else 1,
        0 if not _node_intra_attempt_count(node) else 1,
        _node_intra_attempt_count(node),
        _function_sort_index(node),
    )


def _cross_pair_key(src_function_id: str, dst_function_id: str) -> tuple[str, str]:
    return (str(src_function_id or "").strip(), str(dst_function_id or "").strip())


def _cross_resource(reason: str) -> str:
    return str(reason or "").partition(": ")[2] or str(reason or "").strip()


def _add_cross_candidate(
    candidates: list[tuple[str, str, str]],
    seen: set[tuple[str, str, str]],
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
    key = graph.cross_test_key(src_function_id, dst_function_id, _cross_resource(reason))
    if key in seen:
        return
    seen.add(key)
    candidates.append((src_function_id, dst_function_id, reason))


def _candidate_cross_pairs(graph: FlowGraph) -> list[tuple[str, str, str]]:
    candidates: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    for dep in graph.function_dependencies:
        reason = _clean_natural_text(dep.resource_variable) or "功能依赖"
        _add_cross_candidate(
            candidates,
            seen,
            graph,
            dep.src_function_id,
            dep.dst_function_id,
            f"功能依赖: {reason}",
        )

    # 共享资源变量没有显式方向：两个方向都可能是“写入→读取”，双向各生成一个候选；
    # 某资源已有显式依赖方向时只保留显式方向。
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

    for owners in resource_owners.values():
        if len(owners) < 2:
            continue
        for src_index, (src_node, resource) in enumerate(owners[:-1]):
            for dst_node, _ in owners[src_index + 1:]:
                # Explicit producer/consumer directions take precedence for this
                # resource only. Do not guess a reverse direction for shared data.
                if (
                    graph.cross_test_key(src_node.id, dst_node.id, resource) in seen
                    or graph.cross_test_key(dst_node.id, src_node.id, resource) in seen
                ):
                    continue
                for first, second in ((src_node, dst_node), (dst_node, src_node)):
                    _add_cross_candidate(
                        candidates,
                        seen,
                        graph,
                        first.id,
                        second.id,
                        f"共享资源变量: {resource}",
                    )

    return candidates


def _cross_pair_exhausted_keys(graph: FlowGraph) -> set[tuple[str, str, str]]:
    """不再排队的组合：已完成、不可重试的失败，或已用完重试次数的 retryable 结果。"""
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


def _pending_cross_pairs(graph: FlowGraph) -> list[tuple[str, str, str]]:
    # 已完成与确定失败的组合离开队列；导航失败或截图不可用的组合按次数有限重试。
    processed = _cross_pair_exhausted_keys(graph)
    return [
        (src_id, dst_id, reason)
        for src_id, dst_id, reason in _candidate_cross_pairs(graph)
        if graph.cross_test_key(src_id, dst_id, _cross_resource(reason)) not in processed
        and _node_cross_testable(graph.get_node(src_id))
        and _node_cross_testable(graph.get_node(dst_id))
    ][:MAX_CROSS_TEST_PAIRS]


def _function_label(node) -> str:
    if node is None:
        return "未知功能"
    return _clean_natural_text(node.name or node.description) or "未命名功能"


def _page_display_name(page: Page | None) -> str:
    """规划提示词中的页面自然语言名：名称 → 描述，剔除内部编号。"""
    if page is None:
        return "未知页面"
    return _clean_natural_text(page.name or page.description) or "未命名页面"


def _page_node_label(page: Page | None) -> str:
    if page is None:
        return "未知页面"
    return _clean_natural_text(page.name) or str(page.id or "").strip() or "未命名页面"


def _is_black_screen_page(
    graph: FlowGraph,
    current_page: Page | None,
    current_page_id: str | None = None,
) -> bool:
    page_id = (
        _resolve_current_page_id(graph, current_page)
        if current_page_id is None else current_page_id
    )
    page = graph.get_page(page_id) if page_id else None
    return bool(page and page.name == BLACK_SCREEN_PAGE_NAME)


def _matches_black_screen_path(graph: FlowGraph, scenario: str) -> bool:
    scenario_key = _normalize_similarity_key(scenario)
    if not scenario_key:
        return False
    black_page_ids = {
        str(page.id or "")
        for page in graph.active_pages()
        if page.name == BLACK_SCREEN_PAGE_NAME
    }
    for transition in graph.page_transitions:
        if transition.dst_page_id not in black_page_ids:
            continue
        failed_action = re.split(
            r"[；;]?\s*结果[:：]",
            str(transition.operation or ""),
            maxsplit=1,
        )[0].strip()
        action_key = _normalize_similarity_key(failed_action)
        if action_key and action_key in scenario_key:
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


def _attach_current_page_identity_with_id(
    graph: FlowGraph,
    current_page: Page | None,
) -> tuple[Page | None, str]:
    """把运行时页面对齐到已记录 Page，同时返回解析出的页面 id（未匹配为空串）。"""
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


def _attach_current_page_identity(graph: FlowGraph, current_page: Page | None) -> Page | None:
    return _attach_current_page_identity_with_id(graph, current_page)[0]


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
        print(f"[场景规划] 执行前页面刷新失败，沿用上一轮状态: {exc}")
        return current_page
    if not isinstance(observed, Page):
        return current_page
    previous_id = str(current_page.id or "").strip() if isinstance(current_page, Page) else ""
    observed = _attach_current_page_identity(graph, observed)
    observed_id = str(observed.id or "").strip()
    if previous_id and observed_id != previous_id:
        print(
            "[场景规划] 执行前刷新发现当前页面已变化: "
            f"state={previous_id}, observed={observed_id or '未匹配新页面'}"
        )
    return observed


def _build_current_location_summary(
    graph: FlowGraph,
    current_page: Page | None,
    current_page_id: str | None = None,
) -> str:
    page_id = (
        _resolve_current_page_id(graph, current_page)
        if current_page_id is None else current_page_id
    )
    page = graph.get_page(page_id) if page_id else None
    owner_ids = graph.get_page_owners(page_id) if page_id else []

    lines = ["【当前位置】"]
    if owner_ids:
        owner_labels = [
            f"{owner_id} | {_function_label(graph.get_node(owner_id))}"
            for owner_id in owner_ids
        ]
        lines.append(f"- 当前页面所属功能候选: {'；'.join(owner_labels)}")
    else:
        lines.append("- 当前页面所属功能候选: 未匹配到已记录功能")

    if page is not None:
        lines.append(f"- 当前页面节点: {page.id} | {_page_node_label(page)}")
        lines.append(f"- 当前页面 owners: {owner_ids}")
    else:
        lines.append("- 当前页面节点: 未匹配到已记录页面")
        if isinstance(current_page, Page) and current_page.activity:
            lines.append(f"- 当前 Activity: {_clean_natural_text(current_page.activity)}")

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
    text = re.sub(r"进入\s*对应的\s*页面", "进入目标页面", text)
    text = re.sub(r"在当前\s*页面", "在当前页面", text)
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
        log_prefix="[场景规划]",
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
            print("[场景规划] 底层调用已耗尽重试，改用程序模板场景。")
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
            "[场景规划] LLM 返回字段不合法"
            f"（第 {attempt}/{PLAN_VALIDATION_RETRY_ATTEMPTS} 次）: {errors}"
        )
        current_message_content = list(message_content)
        current_message_content.append({
            "type": "text",
            "text": (
                '上一次输出不合法，请重新输出 {"test_scenario": "..."}。'
                "同功能且进入同一结果的重复控件只选一个代表项；"
                "不同结果的并列控件可以按准确名称列出 2–3 个。"
                "不得要求所有、全部、每个、逐个、逐一或遍历某类控件。"
                f"字段问题: {errors}"
            ),
        })
    return {}, last_errors


def _skip_response(
    graph: FlowGraph,
    current_page: Page | None,
    target_function_id: str = "",
    *,
    testing_function_id: str = "",
    testing_function_status: str = "not_started",
    workflow_complete: bool = False,
    **extra,
) -> dict:
    """本轮不执行场景（冷启动建图、状态迁移、等待或收尾）时的统一返回。"""
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


def _forced_depth_scenario(graph: FlowGraph, node: FunctionNode) -> str:
    """在深度阶段生成一次通用的业务场景兜底。

    只描述“沿业务逻辑走完一个完整操作”的写法，不规定具体业务——做哪条业务
    由执行器依据它看到的界面决定。这里刻意不写按文案匹配的候选规则表：那类
    规则泛化能力差，换一个 App 就容易选错目标。
    """
    # 结构性守卫：没有任何功能承载页时无从执行。返回空让调用方走既有的
    # “连续无可执行场景则 release 功能”通路，避免把功能卡死在深度阶段。
    if not node.coverage_page_ids:
        return ""
    label = _function_label(node)
    representative_page = _node_representative_page(graph, node)
    entry_label = (
        _page_display_name(representative_page) if representative_page is not None else "该功能的已记录页面"
    )
    hints = _node_uncovered_control_hints(graph, node, limit=3)
    hint_text = (
        "可优先结合仍未覆盖的可见控件：" + "、".join(f"“{hint}”" for hint in hints) + "。"
        if hints
        else ""
    )
    return (
        f"围绕“{label}”执行一条完整的业务操作：先尝试进入“{entry_label}”并截图记录当前可见状态，"
        "然后在该功能已记录的控件中选择一条会产生可观察结果的操作链走完"
        "（例如新建并保存、编辑已有条目、移除条目、切换开关或筛选条件），"
        "动作完成后离开该页面再返回查看结果。"
        f"{hint_text}"
        "不要重复该功能已执行过的场景；每一步等待界面稳定后再截图；只使用已记录的页面与控件；"
        "只操作用 FuncDroid_ 后缀自建的数据，不得执行注销账号、退出登录、切换账号、支付、下单、实名认证等不可逆操作；"
        "入口不可达或动作确实无法执行时，记录原因后结束。"
    )


def _forced_intra_scenario(
    graph: FlowGraph,
    node: FunctionNode,
    hint_index: int = 0,
    package_name: str = "",
) -> str:
    if _function_phase(graph, node) == FUNCTION_PHASE_DEPTH:
        # 深度阶段：优先给深度场景。返回空会让调用方把功能判为“无可产出场景”
        # 并 release，所以深度模板不可用时必须回退到下面的广度模板。
        depth_scenario = _forced_depth_scenario(graph, node)
        if depth_scenario:
            print(
                "[场景规划] 阶段=depth，生成深度业务场景: "
                f"{node.id} | {_function_label(node)}"
            )
            return depth_scenario
    label = _function_label(node)
    representative_page = _node_representative_page(graph, node)
    entry_label = _page_display_name(representative_page) if representative_page is not None else "该功能的已记录页面"
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
                f"点击{hint}",
            )
            if violation:
                print(
                    "[场景规划] 微信账号保护跳过退出登录/删除账户控件: "
                    f"{hint} | {violation}"
                )
                continue
            safe_hints.append(hint)
        hints = safe_hints
    if not hints:
        if not node.coverage_page_ids:
            return ""
        # 零控件功能（页面已记录但尚无可见控件记录）先观察，不能直接判为“无可测”。
        return (
            f"观察“{label}”的可操作内容：从当前页面尝试进入“{entry_label}”，"
            "等待界面稳定后截图，记录页面上所有可见的按钮、输入框、列表项、开关与标签；"
            "若页面可滚动，向下滑动一屏再截图记录；选择一个低风险控件（如筛选、标签切换、"
            "展开详情）点击一次并观察直接结果，进入新界面则返回；"
            "不执行删除、支付、退出登录等不可逆操作；入口不可达时记录原因后结束。"
        )
    # 单轮只探一个控件会让每个场景仅剩 1-2 个有效动作，而每轮的固定开销
    # （规划 1 次 + FFG 更新 1 次以上的大模型带图调用 + 截图 + logcat）不变，
    # 摊薄后吞吐极低。这里一次给出多个控件族，让单个场景把同页可测点走完。
    batch: list[str] = []
    for offset in range(min(PLAN_SCENARIO_CONTROL_BATCH, len(hints))):
        candidate = hints[(hint_index + offset) % len(hints)]
        if candidate not in batch:
            batch.append(candidate)
    hint_text = "、".join(f"“{hint}”" for hint in batch)
    if len(batch) == 1:
        coverage_clause = (
            f"本轮探索以下尚无执行记录的可见控件（{hint_text}），观察它的直接结果；"
        )
    else:
        coverage_clause = (
            f"本轮依次探索以下 {len(batch)} 个尚无执行记录的可见控件（{hint_text}），"
            "每个控件点击后观察其直接结果，若进入了新界面就返回上一层再继续下一个；"
        )
    return (
        f"围绕“{label}”执行一次低风险补充探索：从当前页面尝试进入“{entry_label}”，"
        + coverage_clause
        + "不要重复执行已有操作记录中的控件；"
        "如果入口不可达、目标已无可测内容或只能重复历史路径，记录原因后结束。"
    )


def _forced_cross_scenario(graph: FlowGraph, src_id: str, dst_id: str, reason: str) -> str:
    src_label = _function_label(graph.get_node(src_id))
    dst_label = _function_label(graph.get_node(dst_id))
    reason = _clean_natural_text(reason) or "既有功能依赖"
    data_values = [item["value"] for item in graph.function_test_data(src_id)][-3:]
    data_clause = (
        "源功能此前已创建的测试数据：" + "、".join(f"“{value}”" for value in data_values)
        + "，优先在目标功能中查找这些数据；"
        if data_values
        else f"若源功能中没有可用数据，先用唯一名称 FuncDroid_{src_id} 创建一条；"
    )
    return (
        f"测试“{src_label}”到“{dst_label}”的跨功能关系：先确认源功能中与“{reason}”相关的已记录资源、"
        f"状态或结果，{data_clause}再进入目标功能观察该资源、状态或结果是否被正确读取、展示或使用；"
        "如果路径不可达或条件不足，记录不可达原因后结束。"
    )


def _test_data_instruction(
    graph: FlowGraph,
    scenario_type: str,
    target: FunctionNode | None,
    src: FunctionNode | None,
) -> str:
    """测试数据后缀按功能稳定（FuncDroid_<func_id>），跨轮次、跨功能可追溯。"""
    owner = src if scenario_type == "cross_test" else target
    owner_id = owner.id if owner is not None else "data"
    text = f"【本轮测试数据】创建或命名资源时使用唯一后缀 FuncDroid_{owner_id}。"
    existing = [item["value"] for item in graph.function_test_data(owner_id)][-5:]
    if existing:
        text += "该功能此前已创建的测试数据：" + "、".join(existing) + "；深度场景与跨功能读取优先复用这些数据。"
    return text


def _release_unplannable_function(
    graph: FlowGraph,
    current_page: Page | None,
    node: FunctionNode,
    reason: str,
    unplannable_counts: dict[str, int] | None = None,
) -> dict:
    """规划器和程序模板都给不出可执行场景：连续多轮如此才延后该功能。

    单轮空场景常是模型偶发或页面记录尚未补齐，先计数并跳过本轮；连续达到
    PLAN_UNPLANNABLE_ROUNDS_BEFORE_DEFER 轮才标记 deferred（reason=unplannable）。
    deferred 仍会在收尾时补测一次，且事后满足完成判定时可转为 completed。
    """
    counts = dict(unplannable_counts or {})
    counts[node.id] = int(counts.get(node.id, 0) or 0) + 1
    if counts[node.id] < PLAN_UNPLANNABLE_ROUNDS_BEFORE_DEFER:
        print(
            f"[场景规划] {reason}，本轮跳过（{counts[node.id]}/"
            f"{PLAN_UNPLANNABLE_ROUNDS_BEFORE_DEFER}）: {node.id} | {_function_label(node)}"
        )
        return {**_skip_response(graph, current_page, node.id), "unplannable_counts": counts}
    print(f"[场景规划] {reason}，连续无可执行场景，标记 deferred 以释放调度: {node.id} | {_function_label(node)}")
    if not _node_is_terminal(node):
        graph.set_node_test_status(node.id, "deferred", reason="unplannable")
    counts.pop(node.id, None)
    return {**_skip_response(graph, current_page, node.id), "unplannable_counts": counts}


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
        "text": f"目标应用: {app_name or package_name or '未知'} | package={package_name or '未知'}",
    }]
    if current_page.representative_screenshot:
        message_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{current_page.representative_screenshot}"},
        })
    else:
        message_content.append({
            "type": "text",
            "text": '当前没有页面截图；请返回 {"functions": [], "pages": []}。',
        })

    result = _invoke_llm(llm, BOOTSTRAP_FFG_SYSTEM_PROMPT, message_content)
    # 冷启动只有一张截图，证据步骤固定为 1，由程序填写而不是让模型复述。
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
        print("[场景规划] 冷启动 patch 未产生功能节点，使用最小主页面 patch 兜底。")
        fallback_patch = {
            "functions": [
                {
                    "function_ref": "new_func_1",
                    "source_step": 1,
                    "name": _page_display_name(current_page) or "当前页面导航",
                    "description": current_page.description or "冷启动识别的当前页面和可见入口。",
                }
            ],
            "pages": [
                {
                    "page_ref": "new_page_1",
                    "function_refs": ["new_func_1"],
                    "source_step": 1,
                    "name": _page_display_name(current_page) or "当前页面",
                    "description": current_page.description or "冷启动截图对应的当前页面。",
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
        "[场景规划] 冷启动建图完成: "
        f"functions={len(graph.functions)}, page={_page_display_name(first_page)}, "
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
        "【覆盖遗漏检查】",
        f"目标应用: {app_name or package_name or '未知'} | package={package_name or '未知'}",
        EXPLORATION_MODE_INSTRUCTIONS[PLAN_EXPLORATION_MODE],
        boundary_hint,
        "",
        f"功能: {node.id} | {_function_label(node)}",
    ]
    if node.description:
        lines.append(f"描述: {_clean_natural_text(node.description)}")
    lines.append(f"测试状态: {node.test_status or 'not_started'}")
    lines.append(f"覆盖率: {node.coverage_percent}%")
    coverage_details = graph.function_coverage_details(node.id)
    lines.append(
        "控件/操作: "
        f"{coverage_details['executed_operations']}/{coverage_details['total_controls']}"
    )
    if node.resource_variables:
        lines.append(f"资源变量: {'、'.join(_clean_natural_text(item) for item in node.resource_variables)}")
    lines.append("")
    lines.append("已记录页面、控件与操作:")
    for page_id in node.coverage_page_ids:
        page = graph.get_page(page_id)
        if page is None:
            continue
        lines.append(f"- 页面: {page.id} | {_page_display_name(page)}")
        if page.description:
            lines.append(f"  描述: {_clean_natural_text(page.description)}")
        if page.controls:
            lines.append("  控件全集（跨截图状态）:")
            for control in page.controls:
                lines.append(f"    * {_clean_natural_text(control)}")
        if page.operations:
            lines.append("  已执行/已知操作:")
            for operation in page.operations:
                lines.append(f"    * {_clean_natural_text(operation)}")
        else:
            lines.append("  已执行/已知操作: 无")
    transitions = graph.get_function_page_transitions(node.id)
    if transitions:
        lines.append("")
        lines.append("页面跳转:")
        for transition in transitions:
            src_page = graph.get_page(transition.src_page_id)
            dst_page = graph.get_page(transition.dst_page_id)
            lines.append(
                f"- 从「{_page_display_name(src_page)}」通过「{_clean_natural_text(transition.operation)}」到「{_page_display_name(dst_page)}」"
            )
    if node.scenarios:
        lines.append("")
        lines.append("已测试场景:")
        for scenario in node.scenarios:
            lines.append(f"- {_clean_natural_text(scenario)}")
    else:
        lines.append("")
        lines.append("已测试场景: 无")
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
                    f"【分批视觉复核】当前是第 {batch_index}/{len(page_batches)} 批。"
                    "只根据本批附带截图寻找可见遗漏；本批没有遗漏时返回 false，"
                    "不要据此推断未附图页面。程序会继续检查其余批次。"
                ),
            },
        ]
        if validation_feedback:
            message_content.append({"type": "text", "text": validation_feedback})
        for page in page_batch:
            message_content.append({
                "type": "text",
                "text": f"截图对应页面: {page.id} | {_page_display_name(page)}",
            })
            message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{page.representative_screenshot}"},
            })

        print(
            "[场景规划] 覆盖遗漏检查: "
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
    """两个字段要么都空（无遗漏），要么都非空（给出一个入口）。"""
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
        f"在“{_page_display_name(page)}”页面，点击“{entry_name}”，"
        "观察并记录实际出现的页面、弹窗或状态变化后结束。"
    )


def _function_audit_signature(graph: FlowGraph, node: FunctionNode) -> str:
    """结构覆盖签名：只有页面/控件/跳转变化才触发重新审计。

    刻意排除场景数量与操作数量：审计探索本身会 add_scenario 并新增操作记录，
    若把它们计入签名，每次审计探索都会改变签名，导致下一轮对同一已完成功能
    再次审计（最多 MAX_COMPLETION_AUDITS_PER_FUNCTION 次），即使探索没有
    发现任何新的页面或控件。
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
    graph: FlowGraph,
    exploration_mode: str,
    previous_signatures: dict[str, str] | None = None,
    audit_attempts: dict[str, int] | None = None,
    app_name: str = "",
    package_name: str = "",
    deadline: float | None = None,
    max_audits: int | None = None,
) -> tuple[dict | None, bool, dict[str, str]]:
    """逐个审计已完成功能，返回 (待执行入口, 是否全部审计完, 已审计签名)。

    模型连续返回非法结构或不存在的页面时，按“无可执行遗漏”记下签名，
    避免同一功能每轮都重审而阻塞收尾；到达运行截止时间时停止审计。
    每轮最多实际审计 max_audits（默认 PLAN_MAX_AUDITS_PER_ROUND）个功能，
    其余留到后续规划轮次（返回“未审计完”）。
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
            print("[场景规划] 已到运行截止时间，停止覆盖遗漏检查。")
            return None, False, audited_signatures
        signature = _function_audit_signature(graph, node)
        if previous_signatures.get(node.id) == signature:
            audited_signatures[node.id] = signature
            print(f"[场景规划] 覆盖遗漏检查跳过未变化功能: {_function_label(node)}")
            continue
        if int(audit_attempts.get(node.id, 0) or 0) >= MAX_COMPLETION_AUDITS_PER_FUNCTION:
            audited_signatures[node.id] = signature
            print(
                "[场景规划] 单功能覆盖审计达到安全上限，结束该签名审计: "
                f"{_function_label(node)} | {MAX_COMPLETION_AUDITS_PER_FUNCTION}"
            )
            continue
        if audits_this_round >= audit_budget:
            print(
                "[场景规划] 本轮覆盖遗漏检查已达上限，其余功能留待后续轮次: "
                f"{audits_this_round}/{audit_budget}"
            )
            return None, False, audited_signatures
        audits_this_round += 1
        validation_feedback = ""
        for _ in range(2):
            result = _audit_function_for_gap(
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
                "上一次输出不符合覆盖审计协议。无遗漏时 entry_page_ref 与 entry_name 都返回空字符串；"
                "有遗漏时两者都必须非空：entry_page_ref 必须是摘要中该功能已有的 page_x，"
                "entry_name 必须是该页截图或 description 中真实可见且尚未执行的入口名称。"
                f"字段问题: {protocol_errors or ['entry_page_ref_not_in_function']}"
            )
        if protocol_errors or (has_gap and not valid_entry_page):
            print(
                "[场景规划] 覆盖遗漏检查连续返回不可执行结果，按无可执行遗漏记录该签名: "
                f"{_function_label(node)} | "
                f"{protocol_errors or f'page={entry_page_ref} | entry={entry_name}'}"
            )
            audited_signatures[node.id] = signature
            continue
        action_policy_violation = ""
        if has_gap:
            # 通用登录态守卫（不限包名）：忽略“点击退出登录/切换账号”类候选，
            # 避免为查看登录表单而主动登出，导致后续依赖登录态的取证全部失效。
            action_policy_violation = wechat_action_policy_violation(
                package_name, "click", f"点击{entry_name}"
            ) or account_session_action_violation("click", f"点击{entry_name}")
        if action_policy_violation:
            print(
                "[场景规划] 覆盖遗漏候选涉及退出登录或删除账户，已忽略: "
                f"{_function_label(node)} | entry={entry_name} | "
                f"reason={action_policy_violation}"
            )
        elif has_gap:
            test_scenario = _audit_entry_scenario(graph.get_page(entry_page_ref), entry_name)
            print(
                "[场景规划] 覆盖遗漏检查发现待验证入口，转为真实执行而非直接建图: "
                f"{node.id} | page={entry_page_ref} | scenario={test_scenario}"
            )
            return {
                "owner_function_id": node.id,
                "entry_page_id": entry_page_ref,
                "test_scenario": test_scenario,
            }, False, audited_signatures
        else:
            print(f"[场景规划] 覆盖遗漏检查通过: {_function_label(node)}")
        audited_signatures[node.id] = signature
    return None, True, audited_signatures


def _page_path_to_entry(
    graph: FlowGraph,
    node: FunctionNode,
    start_page_id: str,
    entry_page_id: str,
    allow_backtrack: bool = False,
) -> list[tuple[PageTransition, bool]]:
    """功能内部页面之间到入口页的路径，只经过该功能自己的页面。"""
    return _leveled_page_path(
        graph,
        start_page_id,
        [entry_page_id],
        allow_backtrack=allow_backtrack,
        allowed_page_ids=set(node.page_ids),
    )


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

    # A 级：已记录正向边；B 级：允许把已记录边反向当作返回。
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
                "当前页仍属于目标功能但没有已记录的正向导航路径：若它明显是本轮入口页打开的"
                f"子页面、选择页或弹层，先执行一次 back，截图确认是否直接回到“{_page_display_name(entry_page)}”；"
                "若没有回到目标入口页或已离开目标应用，再执行一次 open_app 打开目标应用主页面，"
                "并按后续已记录路径导航"
            )
        else:
            instructions.append(
                "当前不在目标功能的已知导航路径上：先执行一次 open_app 直接打开目标应用主页面；"
                "若仍未到主页面，再关闭导航开始时已有的临时菜单或弹窗，并使用页面返回回到已记录的"
                f"“{_page_display_name(start_page)}”页面；若返回时离开目标应用，则再次 open_app"
            )
    instructions.extend(_path_step_instructions(graph, path))
    if not path and start_page_id != entry_page_id:
        instructions.append(f"继续安全返回，直到进入已记录的“{_page_display_name(entry_page)}”页面")
    instructions.append(
        f"截图确认已到达“{_page_display_name(entry_page)}”后结束导航，"
        "不要在导航阶段点击本轮待验证入口；为到达目标页而主动打开的菜单属于导航路径，不要再把它关闭"
    )
    return "；".join(instructions) + "。"


def _audit_exploration_response(
    graph: FlowGraph,
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
        "[场景规划] 覆盖遗漏检查转为证据驱动探索执行: "
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
    graph: FlowGraph,
    current_page: Page | None,
    failure_counts: dict[str, int] | None = None,
    compliance_terms: tuple[str, ...] = (),
    current_page_id: str | None = None,
) -> FunctionNode | None:
    """在未完成的功能中按唯一排序键取第一个。

    导航只是参考路径，不作为候选闸门：没有已知路径的功能走 D 级探索式导航，
    只是在排序上排到有已知路径的功能之后。
    """
    failure_counts = failure_counts or {}
    # 当前页面只解析一次（可能是对全部页面截图的相似度扫描），邻接表只建一次，
    # 供所有候选功能的导航级别计算复用。
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
    # 合规闭环存在未决目标时，已发现相关页面/控件的功能优先，
    # 让普通探索为下一次合规复判补充入口和路径证据。
    if compliance_terms:
        relevant = [
            node for node in candidates
            if _function_compliance_score(graph, node, compliance_terms) > 0
        ]
        # 刚失败的功能同样让出优先级：合规相关但刚失败的功能不压过健康的
        # 非相关功能，此时回落到排序键的第一个。max 在同分时保留排序键顺序。
        healthy = [node for node in relevant if not failure_counts.get(node.id)]
        if healthy:
            return max(
                healthy,
                key=lambda node: _function_compliance_score(graph, node, compliance_terms),
            )
    return candidates[0] if candidates else None


def _run_deadline(state: AgentState) -> float | None:
    """运行截止的绝对时间戳；未设置总时长时返回 None。"""
    limit = float(state.get("total_time_limit", 0) or 0)
    start = float(state.get("start_time", 0) or 0)
    return start + limit if limit > 0 and start > 0 else None


def _select_deferred_retry(
    graph: FlowGraph,
    current_page: Page | None,
    retried_ids: list[str],
    current_page_id: str | None = None,
) -> FunctionNode | None:
    """收尾时每个 deferred 功能只再给一次补测机会，且只给有已知路径（A/B/C 级）的功能。

    D 级探索式导航已经失败过一次，没有新路径时重试只会重复同样的探索。
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
    """页面跳转的正向/反向邻接表；每次规划调用建一次，供多次 BFS 复用。"""

    forward: dict[str, list[PageTransition]] = field(default_factory=dict)
    reverse: dict[str, list[PageTransition]] = field(default_factory=dict)


def _page_adjacency(graph: FlowGraph) -> _PageAdjacency:
    adjacency = _PageAdjacency()
    # 按 page_transitions 原顺序建表，保证 BFS 的边遍历顺序与逐条扫描一致。
    for transition in graph.page_transitions:
        adjacency.forward.setdefault(transition.src_page_id, []).append(transition)
        adjacency.reverse.setdefault(transition.dst_page_id, []).append(transition)
    return adjacency


def _leveled_page_path(
    graph: FlowGraph,
    start_page_id: str,
    target_page_ids: list[str],
    allow_backtrack: bool = False,
    allowed_page_ids: set[str] | None = None,
    adjacency: _PageAdjacency | None = None,
) -> list[tuple[PageTransition, bool]]:
    """BFS 最短页面路径；每步为 (转移, 是否反向)。

    allow_backtrack 时把已记录的 src→dst 边也当作 dst→src 的“返回”候选：
    进入过的页面大多能退回来源页，但这只是参考，执行器需截图确认。
    adjacency 由调用方预建时直接复用，否则本次调用内建一次。
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
    graph: FlowGraph,
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


def _root_page(graph: FlowGraph) -> Page | None:
    """应用入口页：第一个功能的代表页面（open_app 重开后预期落在这里）。"""
    return _node_representative_page(graph, next(iter(graph.functions.values()), None))


def _route_via_restart(
    graph: FlowGraph,
    current_page_id: str,
    target_page_ids: list[str],
    adjacency: _PageAdjacency,
    allow_backtrack: bool,
) -> tuple[list[tuple[PageTransition, bool]], bool]:
    """当前位置无路径时的 C 级兜底：重开应用回到入口页，再走入口页的已记录路径。

    返回 (路径, 是否需要先重开)。入口页本身就是目标时只需重开，不需要路径。
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


# C 级导航的重开指令；执行器提示词中 open_app 规则对“导航任务明确要求重新打开”留有例外。
RESTART_NAVIGATION_INSTRUCTION = "执行 open_app 重新打开目标应用，并等待界面稳定"


def _path_step_instructions(
    graph: FlowGraph,
    path: list[tuple[PageTransition, bool]],
) -> list[str]:
    instructions: list[str] = []
    for transition, reversed_edge in path:
        src_page = graph.get_page(transition.src_page_id)
        dst_page = graph.get_page(transition.dst_page_id)
        if reversed_edge:
            instructions.append(
                f"在“{_page_display_name(dst_page)}”尝试执行一次 back，截图确认是否回到“{_page_display_name(src_page)}”"
            )
            continue
        operation = _clean_natural_text(transition.operation) or "执行已记录的页面操作"
        instructions.append(
            f"在“{_page_display_name(src_page)}”执行“{operation}”，进入“{_page_display_name(dst_page)}”"
        )
    return instructions


def _exploratory_navigation_text(graph: FlowGraph, target: FunctionNode) -> str:
    """D 级：图中没有路径，只给目标描述，由执行器自行寻找进入方式。"""
    target_pages = [
        page for page_id in target.coverage_page_ids
        if (page := graph.get_page(page_id)) is not None
    ][:3]
    entry_pages = [
        page for page_id in target.entry_page_ids
        if (page := graph.get_page(page_id)) is not None
    ][:3]
    parts = [f"FFG 中没有到“{_function_label(target)}”的已记录路径，本次为探索式导航"]
    if target.description:
        parts.append(f"该功能：{_clip_summary_text(target.description, 160)}")
    if target_pages:
        parts.append(
            "目标页面："
            + "、".join(
                f"“{_page_display_name(page)}”"
                + (f"（{_clip_summary_text(page.description, 80)}）" if page.description else "")
                for page in target_pages
            )
        )
    if entry_pages:
        parts.append("已知入口页：" + "、".join(f"“{_page_display_name(page)}”" for page in entry_pages))
    parts.append(
        "从当前页面自行寻找进入方式（可先 open_app 回到应用首页），每步截图确认；"
        "看到目标页面后结束导航，找不到时说明原因后结束"
    )
    return "；".join(parts) + "。"


def _navigation_to_function(
    graph: FlowGraph,
    current_page: Page | None,
    target: FunctionNode,
    current_page_id: str | None = None,
    adjacency: _PageAdjacency | None = None,
) -> tuple[str, str, str]:
    """生成到目标功能的参考导航，返回 (导航文本, 原因, 级别)。

    current_page_id / adjacency 可由调用方在一轮规划内预先算好传入，避免
    对每个候选功能重复做截图相似度匹配和邻接扫描。

    级别见 NAV_LEVEL_*：空=已在目标页，A=正向边，B=允许返回，C=重开应用后走已知路径，
    D=探索式导航。导航只是参考，不决定功能能否被排程。
    """
    # entry_page_ids 只提供进入功能的入口，不承载目标功能本身。把入口页
    # 当成导航终点会让执行器停在应用首页，再到其它模块里寻找同名控件。
    target_page_ids = list(target.coverage_page_ids)
    if current_page_id is None:
        current_page_id = _resolve_current_page_id(graph, current_page)
    if target_page_ids and current_page_id in target_page_ids:
        return "", "当前位置已经属于目标功能页面", NAV_LEVEL_HERE
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
            "当前位置和应用入口都没有到目标功能的已记录页面路径，改为探索式导航",
            NAV_LEVEL_EXPLORE,
        )

    instructions: list[str] = []
    if restart_first:
        instructions.append(RESTART_NAVIGATION_INSTRUCTION)
        root_page = _root_page(graph)
        if root_page is not None:
            instructions.append(
                f"若打开后尚未处于“{_page_display_name(root_page)}”，"
                "逐次执行安全返回并每次截图确认，直到回到该已记录入口页"
            )
    instructions.extend(_path_step_instructions(graph, path))
    if path:
        last_transition, reversed_edge = path[-1]
        target_page = graph.get_page(
            last_transition.src_page_id if reversed_edge else last_transition.dst_page_id
        )
    else:
        target_page = _node_representative_page(graph, target)
    instructions.append(f"确认到达“{_page_display_name(target_page)}”后结束导航")
    reason = {
        NAV_LEVEL_FORWARD: "使用 FFG 中已记录的最短页面路径",
        NAV_LEVEL_BACKTRACK: "已记录路径需要先返回到来源页面",
        NAV_LEVEL_RESTART: "需要先重开应用再走已记录路径",
    }[level]
    return "；".join(instructions) + "。", reason, level


def _build_scope_planning_summary(
    graph: FlowGraph,
    current_page: Page | None,
    scenario_type: str,
    target: FunctionNode | None = None,
    src: FunctionNode | None = None,
    dst: FunctionNode | None = None,
    dependency_reason: str = "",
    current_page_id: str | None = None,
) -> tuple[str, str, str]:
    """作用域摘要，同时返回 (摘要, 参考导航, 导航级别)。"""
    nodes = [node for node in (target, src, dst) if node is not None]
    navigation_target = src if scenario_type == "cross_test" else target
    navigation_scenario = ""
    navigation_reason = "没有目标功能"
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
        "【当前位置】",
        _build_current_location_summary(graph, current_page, current_page_id),
        "",
        "【参考导航】（由执行器先行执行，仅供参考）",
        navigation_scenario or navigation_reason,
    ]
    if scenario_type == "cross_test":
        lines.extend([
            "",
            "【本轮跨功能依赖】",
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
        lines.append("【源功能到目标功能的已知路径】")
        shared_pages = set(src.page_ids if src else []) & set(dst.page_ids if dst else [])
        if shared_pages:
            lines.append(f"- 两个功能共享页面: {sorted(shared_pages)}")
        elif cross_path:
            for transition in cross_path:
                lines.append(
                    f"- {transition.src_page_id}（{_page_display_name(graph.get_page(transition.src_page_id))}）"
                    f" --[{_clean_natural_text(transition.operation)}]--> "
                    f"{transition.dst_page_id}（{_page_display_name(graph.get_page(transition.dst_page_id))}）"
                )
        else:
            lines.append("- 无已记录路径；在源功能完成后按界面自行寻找进入目标功能的方式")

    scope_page_ids = {page_id for node in nodes for page_id in node.page_ids}
    for node in nodes:
        title = "当前功能" if scenario_type == "intra_test" else (
            "源功能" if src is not None and node.id == src.id else "目标功能"
        )
        lines.extend([
            "",
            f"【{title}】",
            f"- {node.id} | {_function_label(node)}",
            f"- 描述: {_clip_summary_text(node.description)}",
            f"- 状态: {node.test_status} | 覆盖率: {node.coverage_percent}%",
            f"- 阶段: {_describe_function_phase(graph, node)}",
            f"- 关联页面: {node.page_ids or '无'}",
            f"- 仅导航入口页: {node.entry_page_ids or '无'}",
            f"- 资源变量: {node.resource_variables or '无'}",
            "- 已测试场景:",
        ])
        if node.scenarios:
            # 只展示最近 K 个：深度场景要避开已测过的业务，但全量展示会让提示词随
            # 运行时长无界增长；更早的场景给出数量汇总，提醒模型它们同样已测。
            older_count = max(0, len(node.scenarios) - PLAN_SCOPE_RECENT_SCENARIOS)
            if older_count:
                lines.append(
                    f"  - （另有更早的 {older_count} 个已测试场景未逐条列出，"
                    "同样不要重复；以下为最近的场景）"
                )
            lines.extend(
                f"  - {_clip_summary_text(item, 360)}"
                for item in node.scenarios[older_count:]
            )
        else:
            lines.append("  - 无")
        lines.append("- 页面:")
        for page_id in node.page_ids:
            page = graph.get_page(page_id)
            if page is None:
                continue
            page_role = "仅导航入口" if page_id in node.entry_page_ids else "功能页面"
            lines.append(
                f"  - {page.id} | {_page_display_name(page)} | role={page_role} | "
                f"activity={page.activity or '未知'}"
            )
            if page.description:
                lines.append(f"    描述: {_clip_summary_text(page.description)}")
            if page.controls:
                lines.append("    控件: " + "；".join(_clip_summary_text(value, 180) for value in page.controls[-30:]))
            if page.operations:
                lines.append("    已执行: " + "；".join(_clip_summary_text(value, 180) for value in page.operations[-12:]))

    # 只列与作用域页面相邻的跳转：作用域内部跳转优先，其次是进出作用域的边界跳转，
    # 总数有上限，避免大图下把整张跳转表塞进提示词。
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
    lines.extend(["", "【相关页面路径】"])
    if relevant_transitions:
        for transition in relevant_transitions:
            lines.append(
                f"- {transition.src_page_id}（{_page_display_name(graph.get_page(transition.src_page_id))}）"
                f" --[{_clean_natural_text(transition.operation)}]--> "
                f"{transition.dst_page_id}（{_page_display_name(graph.get_page(transition.dst_page_id))}）"
            )
        if omitted_transitions > 0:
            lines.append(f"- （另有 {omitted_transitions} 条作用域边界跳转未列出）")
    else:
        lines.append("- 无")
    return "\n".join(lines), navigation_scenario, navigation_level


def _compliance_evidence_signature(
    graph: FlowGraph,
    rules: list[dict[str, Any]] | None = None,
) -> str:
    """对会影响 GUI 导航/取证的 FFG 事实签名，不包含截图大字段。

    传入规则时只签名与该检测组关键词相关的功能、页面和相邻路径。这样普通
    商品浏览不会错误重开“设置/注销”等未决条款，而新发现隐私设置、客服、
    权限入口时会触发对应组复判。未传规则时保留全图签名供诊断和兼容测试。
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
                # 只保留摘要，不把截图大字段写入签名 payload。普通探索在相关
                # 页面采集到新的稳定 GUI 状态时，也应能触发未决条款复判。
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
    graph: FlowGraph,
    rules: list[dict[str, Any]],
) -> dict[str, str]:
    """同组规则共享相关 FFG 签名，返回 rule_id -> signature。"""
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
    graph: FlowGraph,
    node: FunctionNode,
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
    graph: FlowGraph,
    current_page: Page | None,
    rules: list[dict[str, Any]],
    current_page_id: str | None = None,
) -> tuple[str, str]:
    """返回基于已知 FFG 的导航场景与候选页面提示。"""
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
        return "", "【FFG 合规线索】当前尚未记录与本组目标明显相关的页面；优先探索设置、我的、登录或帮助等候选入口。"

    candidate_lines = []
    for page in candidates:
        matched_controls = [
            control
            for control in page.controls
            if any(term.lower() in str(control).lower() for term in terms)
        ][:4]
        candidate_lines.append(
            f"- {_page_display_name(page)}"
            + (f"｜相关控件：{'、'.join(matched_controls)}" if matched_controls else "")
        )
    guidance = "【FFG 已发现的合规线索】\n" + "\n".join(candidate_lines)

    if current_page_id is None:
        current_page_id = _resolve_current_page_id(graph, current_page)
    adjacency = _page_adjacency(graph)
    for candidate in candidates:
        if current_page_id == candidate.id:
            return "", guidance + f"\n当前位置已在候选页面“{_page_display_name(candidate)}”，直接从此处取证。"
        # 只走已记录的正向边；当前位置无路径时与功能导航共用 C 级重开兜底。
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
        instructions.append(f"确认到达“{_page_display_name(candidate)}”后结束导航")
        return "；".join(instructions) + "。", guidance
    return "", guidance + "\n当前没有到这些候选页的已记录路径；后续功能探索应优先补全入口路径。"


def _deferred_compliance_guidance(rules_path: str | None) -> tuple[str, tuple[str, ...]]:
    from workflow.compliance_worker import compliance_worker

    checklist = ComplianceChecklist.from_state(compliance_worker.snapshot())
    rules = checklist.deferred_rules(rules_path)
    if not rules:
        return "", ()
    groups = _compliance_groups(rules)
    lines = [
        "【合规闭环未决目标】以下检测组上一轮证据不足。若本轮固定功能内存在相关入口，"
        "优先探索能补充这些 GUI 证据的不同路径或控件，不要简单重复上一轮动作："
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
            f"- {GROUP_LABELS.get(group, group)}："
            + "；".join(objectives[:2])
        )
    return "\n".join(lines), _compliance_terms(rules)


def _maybe_schedule_compliance_check(
    state: AgentState,
    graph: FlowGraph,
    current_page: Page | None,
    *,
    force_deferred: bool = False,
    current_page_id: str | None = None,
) -> dict | None:
    """合规检测调度：合规模式下优先于功能探索，按组派发未完成的合规检测点。

    检测点清单的权威状态保存在 compliance worker 单例里（后台线程异步写入
    verdict）；规划器每轮从 worker 读取快照，避免读到过期的 state 副本。
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
            "[场景规划] 合规闭环重开证据不足条款: "
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
            f"\n【闭环复判｜第 {next_attempt} 次】上一轮证据不足："
            + ("；".join(previous_gaps[:6]) if previous_gaps else "未记录具体缺口")
            + "。优先使用上述新 FFG 路径或不同入口补证，不要机械重复上一轮动作。"
        )
    plan_extra["compliance_attempt"] = next_attempt
    compliance_worker.mark_executing(
        rule_ids,
        _compliance_evidence_signatures(graph, group_rules),
    )
    checklist = ComplianceChecklist.from_state(compliance_worker.snapshot())
    print(
        "[场景规划] 合规检测调度: "
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
    """定向补测等待异步判定完成；不为已检查目标扩展无关功能。"""
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
        print("[定向合规检测] 指定规则已完成判定，结束补测；证据不足结果保留原状态。")
        return {**update, "workflow_complete": True, "compliance_checklist": snapshot}
    return None


def _planning_skip_unavailable(
    state: AgentState,
    graph: FlowGraph,
    *,
    screenshot_unavailable: bool,
) -> dict:
    """截图不可用 / 无法确认目标应用页面时的跳过响应与连续失败计数。"""
    if screenshot_unavailable:
        unavailable_count = int(state.get("screenshot_unavailable_count", 0) or 0) + 1
        workflow_complete = (
            STOP_AFTER_SCREENSHOT_UNAVAILABLE
            and unavailable_count >= MAX_CONSECUTIVE_SCREENSHOT_UNAVAILABLE
        )
        if workflow_complete:
            print(
                "[场景规划] 当前 app 连续截图不可用或黑屏，结束该 app 运行: "
                f"{unavailable_count}/{MAX_CONSECUTIVE_SCREENSHOT_UNAVAILABLE}"
            )
        else:
            print(
                "[场景规划] 当前截图不可用或黑屏，本轮跳过、恢复后继续重试: "
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
            "[场景规划] 连续无法恢复目标应用，结束运行以避免空转: "
            f"{unavailable_count}/{MAX_CONSECUTIVE_TARGET_APP_UNAVAILABLE}"
        )
    else:
        print(
            "[场景规划] 未确认目标应用页面，本轮跳过、重启后继续重试: "
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
    graph: FlowGraph,
    device,
    cold_start: bool,
) -> tuple[Page | None, dict | None]:
    """规划前确认目标应用在前台并刷新当前页面；返回 (当前页面, 提前返回的跳过响应)。"""
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


def _cold_start_response(llm, state: AgentState, graph: FlowGraph, current_page: Page) -> dict:
    """FFG 为空：只根据当前截图建图，本轮不生成执行场景。"""
    print("[场景规划] FFG 为空，进入冷启动截图建图。")
    _bootstrap_initial_ffg(
        llm,
        graph,
        current_page,
        app_name=str(state.get("app_name", "") or ""),
        package_name=str(state.get("package_name", "") or ""),
    )
    current_page = next(iter(graph.active_pages()), current_page)
    stats = _count_ffg_stats(graph)
    print("[场景规划] 输入上下文")
    print(
        f"  探索范围: 核心功能 | 当前页面: {_page_display_name(current_page)} | 输入截图: yes"
    )
    print(
        "  FFG 统计: "
        f"功能={stats['functions']}, 页面={stats['pages']}, 操作={stats['operations']}, "
        f"页面跳转={stats['page_transitions']}, "
        f"功能依赖={stats['function_dependencies']}, 功能内场景={stats['intra_scenarios']}, "
        f"跨功能场景={stats['cross_scenarios']}"
    )
    print("[场景规划] 冷启动仅建图，本轮不生成执行场景；下一轮基于 FFG 摘要继续规划。")
    return _skip_response(graph, current_page)


@dataclass
class _RoundSelection:
    """一轮规划选中的作用域。"""

    scenario_type: str = ""
    target_node: FunctionNode | None = None
    src_node: FunctionNode | None = None
    dst_node: FunctionNode | None = None
    dependency_reason: str = ""
    extra_updates: dict = field(default_factory=dict)


@dataclass
class _PlanningRound:
    """一轮规划内各阶段共享的上下文；current_page_id 只解析一次。"""

    llm: Any
    state: AgentState
    graph: FlowGraph
    current_page: Page | None
    current_page_id: str
    current_testing_function_id: str
    current_testing_function_status: str
    compliance_guidance: str
    compliance_terms: tuple[str, ...]

    def skip_round(self, reason: str, **extra) -> dict:
        print(f"[场景规划] {reason}，本轮跳过执行。")
        return _skip_response(
            self.graph,
            self.current_page,
            testing_function_id=self.current_testing_function_id,
            testing_function_status=self.current_testing_function_status,
            **extra,
        )


def _select_round_scope(round_ctx: _PlanningRound) -> tuple[_RoundSelection, dict | None]:
    """按 跨功能 → 功能内 → 收尾（合规复判/deferred 补测/覆盖审计）的顺序选作用域。

    返回 (选中的作用域, 提前返回的响应)；后者非空时本轮不生成场景。
    """
    state, graph, current_page = round_ctx.state, round_ctx.graph, round_ctx.current_page
    selection = _RoundSelection()

    # 1) 跨功能：两端都已有自身场景与成功业务操作的依赖即可立即排上，
    #    不等待无关功能完成，也不被功能内探索永久抢占；导航只作参考，不做闸门。
    if _normalize_run_phase(state.get("run_phase", "all")) == "all":
        for src_id, dst_id, candidate_reason in _pending_cross_pairs(graph):
            if not _cross_pair_ready(graph, src_id, dst_id):
                continue
            selection.src_node, selection.dst_node = graph.get_node(src_id), graph.get_node(dst_id)
            selection.dependency_reason = candidate_reason
            selection.scenario_type = "cross_test"
            return selection, None

    # 2) 功能内：未完成的功能按唯一排序键取第一个。
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

    # 3) 收尾：没有可执行的常规目标。
    if bool(state.get("compliance_detection_enabled", False)):
        from workflow.compliance_worker import compliance_worker

        if compliance_worker.pending_count() > 0:
            print(
                "[场景规划] 普通探索已无目标，等待异步合规判定完成后再决定复判或收尾。"
            )
            if not compliance_worker.flush(
                timeout=COMPLIANCE_CLOSE_LOOP_WAIT_SECONDS
            ):
                return selection, round_ctx.skip_round("仍有异步合规判定任务，暂不结束工作流")
    # 普通功能探索已经没有可执行目标时，给仍处于 deferred 且尚未
    # 耗尽直接重试预算的检测组一次最终机会。达到预算的规则继续保留
    # 为未决，但没有新的相关发现时不阻塞本次 App 运行收尾。
    compliance_update = _maybe_schedule_compliance_check(
        state,
        graph,
        current_page,
        force_deferred=True,
        current_page_id=round_ctx.current_page_id,
    )
    if compliance_update is not None:
        return selection, compliance_update

    # 3a) 每个 deferred 功能收尾时只再补测一次（仅限有已知路径的功能），
    #     结果仍由下一轮状态迁移判定。
    retried_ids = [str(item) for item in (state.get("deferred_retry_ids") or [])]
    selection.target_node = _select_deferred_retry(
        graph, current_page, retried_ids, round_ctx.current_page_id,
    )
    if selection.target_node is not None:
        selection.scenario_type = "intra_test"
        selection.extra_updates["deferred_retry_ids"] = [*retried_ids, selection.target_node.id]
        print(
            "[场景规划] 收尾：deferred 功能最后补测一次: "
            f"{selection.target_node.id} | {_function_label(selection.target_node)}"
        )
        return selection, None

    # 3b) 对已完成功能做覆盖遗漏审计；审计不改变任何功能状态。
    return selection, _completion_audit_round(round_ctx)


def _completion_audit_round(round_ctx: _PlanningRound) -> dict:
    """收尾阶段的覆盖遗漏审计：有遗漏转为执行，未审计完则跳过，全部通过则结束。"""
    state, graph, current_page = round_ctx.state, round_ctx.graph, round_ctx.current_page
    audit_attempts = state.get("completion_audit_attempts", {}) or {}
    previous_signatures = state.get("completion_audit_signatures", {}) or {}
    print("[场景规划] 已无待测功能或跨功能依赖，开始覆盖遗漏检查。")
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
            "覆盖遗漏检查未完成，暂不判定工作流完成",
            completion_audit_signatures=merged_signatures,
        )
    print("[场景规划] 所有功能、跨功能依赖和覆盖遗漏均已处理。")
    return _skip_response(
        graph,
        current_page,
        workflow_complete=True,
        completion_audit_signatures=merged_signatures,
    )


def _scope_instruction(
    graph: FlowGraph,
    selection: _RoundSelection,
) -> tuple[str, str]:
    """返回 (功能阶段, 本轮作用域说明)。"""
    if selection.scenario_type == "cross_test":
        return FUNCTION_PHASE_DEPTH, (
            "【本轮作用域】cross_test："
            f"源功能“{_function_label(selection.src_node)}”→目标功能“{_function_label(selection.dst_node)}”。"
            "只围绕给定资源依赖生成一次源功能产生/修改、目标功能读取/展示/消费的闭环。"
        )
    target_node = selection.target_node
    function_phase = _function_phase(graph, target_node)
    if function_phase == FUNCTION_PHASE_DEPTH:
        return function_phase, (
            f"【本轮作用域】intra_test：功能“{_function_label(target_node)}”。"
            "【本轮阶段=depth 深度业务场景】入口已铺开，本轮沿该功能的业务逻辑写一个完整场景。"
            "可参考的方向（仅供参考，由你结合已记录的页面、控件与操作自行选择）："
            "操作结果离开后再回来是否保留；新建→编辑→删除的完整生命周期；"
            "边界或异常输入（空值、超长、特殊字符）；状态联动或互斥选项；中途放弃或返回后的状态。"
            "必须避开【已测试场景】中已经做过的业务；"
            "确实没有新的业务场景可做时，test_scenario 留空。"
        )
    return function_phase, (
        f"【本轮作用域】intra_test：功能“{_function_label(target_node)}”。"
        "【本轮阶段=breadth 铺入口】若仍有不同的可探索业务方向，返回非空 test_scenario；"
        "若已充分覆盖或只能重复已测试场景，test_scenario 留空。"
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
                f"【目标应用】name={state.get('app_name', '') or '未知'}，"
                f"package={package_name or '未知'}"
            ),
        },
        {
            "type": "text",
            "text": (
                "【探索范围】"
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
    # 截图只在设备已经停在本轮功能页面时有用；否则导航后的画面与之无关。
    scope_node = selection.src_node if selection.scenario_type == "cross_test" else selection.target_node
    if (
        current_page is not None
        and current_page.representative_screenshot
        and round_ctx.current_page_id in scope_node.coverage_page_ids
    ):
        message_content.extend([
            {
                "type": "text",
                "text": "【当前页面截图】只用于核对当前位置，不得使用 FFG 中没有记录的控件。",
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
    """调用规划模型得到 test_scenario；不可用时回落到程序模板（可能为空）。"""
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
        graph, f"{known_navigation}；{test_scenario}"
    ):
        print("[场景规划] 规划结果重复了已知会导致黑屏或页面不可用的操作路径，改用程序模板。")
        test_scenario = ""
    if test_scenario:
        return test_scenario, validation_errors
    # 大模型没给出可用场景：用程序模板兜底，本轮仍然执行。
    if selection.scenario_type == "cross_test":
        test_scenario = _forced_cross_scenario(
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
            "[场景规划] 规划器未给出可用场景"
            f"（{validation_errors or '空场景'}），使用程序模板。"
        )
    return test_scenario, validation_errors


def create_scenario_plan_node(llm, device):
    """创建统一场景规划节点。"""

    def scenario_plan_node(state: AgentState) -> dict:
        targeted_stop = _targeted_compliance_stop(state)
        if targeted_stop is not None:
            return targeted_stop
        graph = state.get("ffg")
        if not isinstance(graph, FlowGraph):
            graph = FlowGraph()
        cold_start = not bool(graph.functions)

        # 阶段 1：观测当前页面（确认前台、刷新截图）。
        current_page, early_response = _observe_planning_page(state, graph, device, cold_start)
        if early_response is not None:
            return early_response
        if cold_start:
            return _cold_start_response(llm, state, graph, current_page)
        # 当前页面 id 每轮只解析一次（可能需要对全部页面截图做相似度匹配），
        # 后续目标选择、导航与摘要构造全部复用。
        current_page, current_page_id = _attach_current_page_identity_with_id(graph, current_page)

        # 阶段 2：状态迁移。覆盖率是“活”的（共享页面被其他功能测试会持续推高），
        # 所有状态迁移在每轮开头统一按 FFG 事实完成，后面的目标选择只读状态、不再改状态。
        _update_function_statuses(graph)

        rules_path = str(state.get("compliance_rules_path", "") or "").strip() or None
        compliance_guidance, compliance_terms = (
            _deferred_compliance_guidance(rules_path)
            if bool(state.get("compliance_detection_enabled", False))
            else ("", ())
        )

        # 合规模式下合规检测点优先于功能探索：冷启动流程要趁登录态尚未建立时
        # 尽早执行（清数据代价最小），权限拒绝路径会破坏应用状态也需要先跑。
        compliance_update = _maybe_schedule_compliance_check(
            state, graph, current_page, current_page_id=current_page_id,
        )
        if compliance_update is not None:
            return compliance_update

        if state.get("run_mode") == "compliance":
            from workflow.compliance_worker import compliance_worker

            summary = ComplianceChecklist.from_state(compliance_worker.snapshot()).summary(rules_path)
            if summary["resolved"] == summary["gui_total"] and not _unfinished_function_nodes(graph):
                print("[合规检测] 全部适用检测点已形成结论，结束本次运行。")
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

        # 阶段 3：选作用域（跨功能 → 功能内 → 收尾）。
        selection, early_response = _select_round_scope(round_ctx)
        if early_response is not None:
            return early_response
        scenario_type = selection.scenario_type
        target_node, src_node, dst_node = selection.target_node, selection.src_node, selection.dst_node
        extra_updates = selection.extra_updates

        # 阶段 4：构造规划输入并生成场景。
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
            "[场景规划] 固定作用域: "
            + (
                f"cross {src_node.id}->{dst_node.id} | {selection.dependency_reason}"
                if scenario_type == "cross_test"
                else f"intra {target_node.id} | {_function_label(target_node)} | "
                     f"阶段={_describe_function_phase(graph, target_node)}"
            )
            + f" | 导航级别={navigation_level or '无需导航'}"
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
                    f"规划器与程序模板均无可执行场景（{validation_errors or '空场景'}）",
                    state.get("unplannable_counts") or {},
                ),
                **extra_updates,
            }

        # 阶段 5：组装导航与调度状态。
        navigation_scenario = known_navigation
        if _is_black_screen_page(graph, current_page, current_page_id):
            navigation_scenario = (
                "当前停留在已记录的黑屏异常页面：先按一次返回键（press_back）并截图确认，"
                "若已离开目标应用则执行 open_app；然后"
                + (known_navigation or "确认回到目标功能页面后结束导航。")
            )

        if scenario_type == "intra_test":
            # deferred 的收尾补测保持终态，只有正常调度的功能才进入 in_progress。
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
                "resource_variable": _cross_resource(selection.dependency_reason) if scenario_type == "cross_test" else "",
            },
            "workflow_complete": False,
            **extra_updates,
        }

    return scenario_plan_node
