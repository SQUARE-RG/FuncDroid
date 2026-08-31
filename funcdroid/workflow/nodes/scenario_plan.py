"""统一场景规划节点。"""

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
from workflow.wechat_safety import (
    wechat_action_policy_violation,
    wechat_read_only_violation,
)

FUNCTION_ID_PATTERN = re.compile(r"\bfunc_\d+\b")
PAGE_ID_PATTERN = re.compile(r"\bpage_\d+\b")
PAGE_ID_ANNOTATION_PATTERN = re.compile(r"\bpage_\d+（[^）]*）")
TEXT_NORMALIZE_PATTERN = re.compile(r"[\"'“”‘’（）()【】\[\]{}<>《》、，,。.：:；;!！?？\-_/\s]+")
DUPLICATE_SCENARIO_ACTION_MARKERS = (
    "下拉刷新",
    "刷新手势",
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
# 单功能功能内场景的硬预算：达到后强制收尾，交给覆盖审计按截图复核剩余入口，
# 避免一个健康但庞大的功能长期占用执行权（广度优先）。
PLAN_MAX_SCENARIOS_PER_FUNCTION = max(
    1,
    int(os.getenv("PLAN_MAX_SCENARIOS_PER_FUNCTION", "6")),
)
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
# 覆盖率可能由共享页面的兄弟功能操作“继承”而来，因此收尾前要求本功能
# 自身至少跑过这么多个功能内场景（除非已无未覆盖控件）。
PLAN_MIN_SCENARIOS_BEFORE_COMPLETE = max(
    1,
    int(os.getenv("PLAN_MIN_SCENARIOS_BEFORE_COMPLETE", "2")),
)
ENTRY_FUNCTION_NAME_PATTERN = re.compile(r"入口\s*$")
CROSS_SCENARIO_MARKER_PATTERN = re.compile(r"^\[cross:([^\]>]+)->([^\]]+)\]")
# 连续无法产出可执行场景的功能，达到该次数后写入终态 unavailable，
# 避免它既排在调度末尾又永远阻塞整体收敛判定（长时空转）。
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
# 与执行节点同口径的冷启动轮询参数：重启后轮询前台包名直到目标应用可见，
# 避免固定 sleep 在重型应用（如微信）冷启动期间截到桌面截图。
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
    r"点击|长按|输入|滑动|滚动|返回|打开|启动|切换|选择|勾选|发送|保存|提交|删除|清除"
)
PARAMETERIZED_CONTROL_PATTERN = re.compile(
    r"(点击|选择|切换到|勾选)\s*[\"'“‘]([^\"'”’]{1,40})[\"'”’]\s*"
    r"(活动按钮|活动标签|筛选项|排序项|分类项|单选项|复选项|标签页|选项)"
)
NAMED_MENU_CONTROL_PATTERN = re.compile(
    r"(点击|打开|选择)\s*[\"'“‘]([^\"'”’]{1,40})[\"'”’]\s*"
    r"(菜单项|导航项|功能项|入口)",
    re.IGNORECASE,
)
UNQUOTED_AUDIT_CONTROL_PATTERN = re.compile(
    r"(?:点击/打开/选择|点击|打开|选择|长按|切换到)\s*([^→，,.。；;]{1,40}?)"
    r"(?=\s*(?:→|[，,.。；;]|并观察|并记录|$))",
    re.IGNORECASE,
)
QUOTED_ACTION_CONTROL_PATTERN = re.compile(
    r"(?:点击/打开|点击|打开|选择)\s*"
    r"(?:截图(?:和页面描述)?中\s*)?(?:真实)?(?:可见的?)?\s*"
    r"[\"'“‘]\s*([^\"'”’]{1,40}?)\s*[\"'”’]",
    re.IGNORECASE,
)
NAMED_ICON_CONTROL_PATTERN = re.compile(
    r"([A-Za-z][A-Za-z0-9_+\-/]{0,39})\s*(icon|图标)",
    re.IGNORECASE,
)
QUOTED_ICON_CONTROL_PATTERN = re.compile(
    r"[\"'“‘]\s*([^\"'”’]{1,20})\s*[\"'”’]\s*图标"
)
PARENTHESIZED_ICON_CONTROL_PATTERN = re.compile(
    r"[（(]\s*([^（）()]{1,20}?)\s*图标\s*[）)]"
)
CHINESE_ICON_CONTROL_PATTERN = re.compile(
    r"(?:点击|选择|找到)\s*"
    r"(?:(?:顶部工具栏|左上角|右上角|左下角|右下角|底部导航栏)(?:中|上|的)?\s*)?"
    r"(?:真实)?(?:可见的?)?\s*([一-鿿A-Za-z0-9_+\-/]{1,12})\s*图标",
    re.IGNORECASE,
)
SCENARIO_PAGE_PATTERN = re.compile(
    r"在[\"'“‘](.{1,60}?)[\"'”’](?:页面|页|中)"
)
SCENARIO_INPUT_VALUE_PATTERN = re.compile(
    r"(输入(?:文本)?\s*)[\"'“‘][^\"'”’]{1,100}[\"'”’]"
)
SCENARIO_EMPTY_INPUT_PATTERN = re.compile(
    r"不输入任何文本到\s*([\"'“‘][^\"'”’]{1,60}[\"'”’]\s*输入框)"
)
SCENARIO_QUOTED_TOKEN_PATTERN = re.compile(
    r"[\"'“‘]([^\"'”’]{1,60})[\"'”’]"
)


def _normalize_exploration_mode(value: str | None) -> str:
    return "core"


def _wait_target_app_foreground(device, package_name: str) -> bool:
    """轮询等待目标应用进入前台；上限 APP_START_MAX_WAIT 秒。"""
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
        print(f"[场景规划] 获取当前前台包名失败: {exc}")
        current_package = ""

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

    last_screenshot_error: ScreenshotUnavailableError | None = None
    for attempt in range(1, PLAN_APP_RESTART_ATTEMPTS + 1):
        try:
            print(f"[场景规划] 第 {attempt}/{PLAN_APP_RESTART_ATTEMPTS} 次重启目标应用。")
            device.restart_app(package_name)
            if _wait_target_app_foreground(device, package_name):
                time.sleep(APP_RESTART_SETTLE_DELAY)
                return capture_current_page(device)
            print(
                "[场景规划] 重启后目标应用仍未进入前台。"
            )
        except ScreenshotUnavailableError as exc:
            last_screenshot_error = exc
            print(f"[场景规划] 重启后截图仍不可用: {exc}")
        except Exception as exc:
            print(f"[场景规划] 拉起目标应用失败: {exc}")
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
    """跨功能测试只接受未处于失败/崩溃/让出关闭状态的功能。"""
    if node is None:
        return False
    return (node.test_status or "not_started") not in {
        "blocked",
        "deferred",
        "unavailable",
    }


def _node_intra_attempt_count(node: FunctionNode | None) -> int:
    """功能内场景数，不含跨功能场景，用于单功能预算。"""
    if node is None:
        return 0
    return sum(
        1
        for scenario in node.scenarios
        if not str(scenario or "").startswith("[cross:")
    )


def _function_completion_met(graph: FlowGraph, node: FunctionNode) -> bool:
    """完成判定：覆盖率达标 + 有成功业务操作 + 本功能自身证据充分。

    同一页面可被多个功能共享，页面上的操作会被记到全部 owner 名下，
    因此一个功能可能在自己一次都没被测过时就“继承”到达标覆盖率。
    收尾前必须要求本功能自身的证据：已无未覆盖控件，或它自己至少
    跑过 PLAN_MIN_SCENARIOS_BEFORE_COMPLETE 个功能内场景（或达到场景预算）。
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
            and (node.test_status or "not_started") == "completed"
        )
        if not needs_reopen:
            continue
        attempts = _node_intra_attempt_count(node)
        if attempts >= PLAN_MAX_SCENARIOS_PER_FUNCTION:
            print(
                "[场景规划] 已完成节点的程序覆盖率不足，尽管达到旧场景预算仍重新审计: "
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
    """未完成功能按“无失败优先、未开始优先、尝试少优先”排序。

    最近执行失败的功能排到后面，让调度器先转向其它未完成功能，
    避免困在单一失败功能上反复重试。
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
    """已测跨功能组合的无向集合：同一关系的任一方向测过即视为覆盖。

    直接扫描场景上的 ``[cross:a->b]`` 标记，而不是遍历 function_dependencies。
    ``add_cross_scenario`` 并不会创建依赖边，依赖遍历会让所有由共享资源派生的
    候选永远停留在 pending，导致每轮重复规划同一组合。
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
        reason = _clean_natural_text(dep.resource_variable) or "功能依赖"
        _add_cross_candidate(
            candidates,
            seen,
            graph,
            dep.src_function_id,
            dep.dst_function_id,
            f"功能依赖: {reason}",
        )

    # 共享资源变量没有额外方向时，按稳定功能顺序补充候选；显式依赖始终优先。
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
                    f"共享资源变量: {resource}",
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
        labels = "、".join(_function_label(node) for node in unfinished[:8])
        return False, f"仍有未完成的功能节点: {labels}"
    if _normalize_run_phase(run_phase) == "all":
        pending_cross = _pending_cross_pairs(graph)
        if pending_cross:
            labels = "、".join(
                f"{_function_label(graph.get_node(src_id))}->{_function_label(graph.get_node(dst_id))}"
                for src_id, dst_id, _ in pending_cross[:6]
            )
            return False, f"仍有关键跨功能组合未测试: {labels}"
    return True, "功能内和关键跨功能覆盖均已完成"


def _function_label(node) -> str:
    if node is None:
        return "未知功能"
    return _clean_natural_text(node.name or node.description) or "未命名功能"


def _page_label(page: Page | None) -> str:
    if page is None:
        return "未知页面"
    return _clean_natural_text(page.name or page.description) or "未命名页面"


def _page_node_label(page: Page | None) -> str:
    if page is None:
        return "未知页面"
    return _clean_natural_text(page.name) or str(page.id or "").strip() or "未命名页面"


def _is_black_screen_page(graph: FlowGraph, current_page: Page | None) -> bool:
    page_id = _resolve_current_page_id(graph, current_page)
    page = graph.get_page(page_id) if page_id else None
    return bool(page and _normalize_similarity_key(page.name) == _normalize_similarity_key("黑屏或页面不可用"))


def _matches_black_screen_path(graph: FlowGraph, scenario: str) -> bool:
    scenario_key = _normalize_similarity_key(scenario)
    if not scenario_key:
        return False
    black_page_ids = {
        str(page.id or "")
        for page in graph.active_pages()
        if _normalize_similarity_key(page.name) == _normalize_similarity_key("黑屏或页面不可用")
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


def _build_current_location_summary(graph: FlowGraph, current_page: Page | None) -> str:
    page_id = _resolve_current_page_id(graph, current_page)
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
        # 仅当操作族集合完全一致时才判重。部分重叠（例如 3 步里共享 2 步）说明
        # 存在一个尚未测过的不同控件，判重会直接丢掉那个控件的覆盖机会。
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
        r"(?:→|->)?\s*(?:确认|观察|验证|检查|记录)(?:实际)?",
        str(value or "").strip(),
        maxsplit=1,
    )[0]
    key = _normalize_similarity_key(action_chain)
    return key if SCENARIO_ACTION_PATTERN.search(action_chain) else ""


def _scenario_operation_family_keys(value: str) -> frozenset[str]:
    """Canonical device-action families, excluding result observations."""
    action_text = re.split(
        r"(?:→|->)?\s*(?:确认|观察|验证|检查|记录|看到)(?:实际)?",
        str(value or "").strip(),
        maxsplit=1,
    )[0]
    clauses = re.split(
        r"(?:→|->|，|,|；|;|。|(?:然后|随后|之后|后再)|并(?=点击|选择|切换|勾选|输入|返回|打开))",
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
    return _normalize_similarity_key(match.group(1)) if match else ""


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
    if "输入框" not in text:
        return ""
    text = SCENARIO_EMPTY_INPUT_PATTERN.sub(r"在 \1 中输入文本 '<value>'", text)
    text = SCENARIO_INPUT_VALUE_PATTERN.sub(r"\1<value>", text)
    return _normalize_similarity_key(text)


def _input_scenario_control_tokens(value: str) -> tuple[str, ...]:
    text = str(value or "").strip()
    text = SCENARIO_EMPTY_INPUT_PATTERN.sub(r"在 \1 中输入文本 '<value>'", text)
    text = SCENARIO_INPUT_VALUE_PATTERN.sub(r"\1<value>", text)
    action_text = re.split(r"观察|验证|确认", text, maxsplit=1)[0]
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
    action = control_match.group(1)
    action_family = "select" if action in {"点击", "选择", "切换到", "勾选"} else action
    control_type = control_match.group(3)
    control_family = _normalize_similarity_key(
        control_match.group(2) if control_type == "选项" else control_type
    )
    return page_key, action_family, control_family


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


def _sanitize_navigation_scenario(text: str, graph: FlowGraph, current_page: Page | None) -> str:
    text = _sanitize_planned_scenario(text)
    if not text:
        return ""
    page_id = _resolve_current_page_id(graph, current_page)
    page = graph.get_page(page_id) if page_id else None
    current_label = _page_node_label(page) if page is not None else _page_label(current_page)
    current_key = _normalize_similarity_key(current_label)
    starts_from_home = re.match(r"^(从|先从)?(首页|主页|主页面|启动页|应用首页)", text)
    current_is_home = bool(re.search(r"(首页|主页|主页面|home|main)", current_key, re.IGNORECASE))
    if starts_from_home and not current_is_home:
        text = re.sub(r"^(从|先从)?(首页|主页|主页面|启动页|应用首页)", "从当前位置", text, count=1)
        print(
            "[场景规划] 导航场景疑似默认从首页开始，已改为从当前位置出发: "
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
        log_prefix="[场景规划]",
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
    risky_intent = wechat_read_only_violation(
        package_name,
        result.get("navigation_scenario", ""),
        test_scenario,
    )
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
    package_name: str = "",
) -> tuple[dict, list[str]]:
    last_errors: list[str] = []
    current_message_content = list(message_content)
    for attempt in range(1, PLAN_VALIDATION_RETRY_ATTEMPTS + 1):
        result = _invoke_llm(llm, system_prompt, current_message_content)
        if llm_json_result_failed(result):
            last_errors = ["llm_request_or_parse_failed"]
            print(
                "[场景规划] 底层调用已耗尽重试，停止字段级重复请求；"
                "交由调度器后续轮次继续。"
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
            "[场景规划] LLM 返回字段不合法"
            f"（第 {attempt}/{PLAN_VALIDATION_RETRY_ATTEMPTS} 次）: {errors}"
        )
        current_message_content = list(message_content)
        current_message_content.append({
            "type": "text",
            "text": (
                "上一次输出字段不合法，请重新输出完整合法 JSON，不要省略任何字段。"
                "同功能且进入同一结果的重复控件只选一个代表项；"
                "不同结果的并列控件可以按准确名称列出 2–3 个。"
                "不得要求所有、全部、每个、逐个、逐一或遍历某类控件。"
                "不得点击、打开、编辑或删除当前作用域没有记录且本场景未先创建的动态对象。"
                "下游页面、结果或入口未记录时，允许用条件开放结尾：若出现明确且未执行的结果/入口，"
                "选择一个代表项并确认直接结果；若未出现、为空或不可达则结束。"
                f"字段问题: {errors}"
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
    entry_label = _page_label(representative_page) if representative_page is not None else "该功能的已记录页面"
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
        return ""
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
    return (
        f"测试“{src_label}”到“{dst_label}”的跨功能关系：先确认源功能中与“{reason}”相关的已记录资源、"
        f"状态或结果，再进入目标功能观察该资源、状态或结果是否被正确读取、展示或使用；"
        "如果路径不可达或条件不足，记录不可达原因后结束。"
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
            # 已连续多轮既没有可补测控件、也没有可达路径：写终态，
            # 否则它会一直留在 unfinished 里阻塞收敛并触发长时空转。
            graph.set_node_test_status(node.id, "unavailable")
            failure_counts.pop(node.id, None)
            print(
                "[场景规划] 功能连续无可执行场景，标记 unavailable 以释放调度: "
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
                "[场景规划] 覆盖不足，直接生成代表性跨功能补测场景: ",
                "[场景规划] 连续空规划达到阈值，强制推进待测跨功能组合: ",
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
    if current_node is not None and "相似历史场景" in reason:
        return release_unproductive_function(
            current_node,
            "[场景规划] 同类重复候选连续达到阈值，释放当前功能并降低其调度优先级"
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
                "[场景规划] 连续空规划后未发现可补充执行的控件，释放当前功能并重新调度",
            )
        if (current_node.test_status or "not_started") == "not_started":
            graph.set_node_test_status(current_node.id, "in_progress")
        print(
            progress_log(
                "[场景规划] 覆盖不足，直接生成代表性补测场景: ",
                "[场景规划] 连续空规划达到阈值，强制收尾上一轮功能: ",
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
                "[场景规划] 未完成功能没有可补充执行的控件，降低其调度优先级并重新规划",
            )
        if (node.test_status or "not_started") == "not_started":
            graph.set_node_test_status(node.id, "in_progress")
        print(
            progress_log(
                "[场景规划] 覆盖不足，直接推进未完成功能的代表性补测: ",
                "[场景规划] 连续空规划达到阈值，强制推进未完成功能: ",
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
            "[场景规划] 连续空规划达到阈值，且无剩余功能/跨功能，直接结束规划: "
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
        "程序派生覆盖率不足且仍有未处理控件" in reason
        or "入口主路径已经验证，必须改选目标页面未执行的不同功能控件" in reason
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
                    "[场景规划] 覆盖不足但仍有未处理控件，"
                    "直接生成代表性补测场景，不产生纯状态规划轮。"
                )
            return forced
    print(
        f"[场景规划] {reason}，本轮跳过执行并继续下一轮规划。"
        f" 空规划计数={noop_count}/{MAX_CONSECUTIVE_PLANNER_NOOPS}"
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
    bootstrap_patch = {
        "functions": result.get("functions", []) if isinstance(result, dict) else [],
        "pages": result.get("pages", []) if isinstance(result, dict) else [],
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
                    "name": _page_label(current_page) or "当前页面导航",
                    "description": current_page.description or "冷启动识别的当前页面和可见入口。",
                    "resource_variables": [],
                }
            ],
            "pages": [
                {
                    "page_ref": "new_page_1",
                    "function_refs": ["new_func_1"],
                    "source_step": 1,
                    "name": _page_label(current_page) or "当前页面",
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
        "【覆盖遗漏检查】",
        f"目标应用: {app_name or package_name or '未知'} | package={package_name or '未知'}",
        EXPLORATION_MODE_INSTRUCTIONS[_normalize_exploration_mode(exploration_mode)],
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
        lines.append(f"- 页面: {page.id} | {_page_label(page)}")
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
                f"- 从「{_page_label(src_page)}」通过「{_clean_natural_text(transition.operation)}」到「{_page_label(dst_page)}」"
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
                "text": f"截图对应页面: {page.id} | {_page_label(page)}",
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
        print(f"[场景规划] 覆盖审计缺少 entry_name，已从执行场景提取: {derived_entry}")
    return normalized


def _function_audit_signature(graph: FlowGraph, node: FunctionNode) -> str:
    """结构覆盖签名：只有页面/控件/操作/跳转变化才触发重新审计。

    刻意排除场景数量：审计探索本身会 add_scenario，若把 scenario 数计入签名，
    每次探索都会改变签名，导致下一轮对同一已完成功能再次全量审计，
    即使探索没有产生任何新的覆盖事实。
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
            print(f"[场景规划] 覆盖遗漏检查跳过未变化功能: {_function_label(node)}")
            continue
        if int(audit_attempts.get(node.id, 0) or 0) >= MAX_COMPLETION_AUDITS_PER_FUNCTION:
            audited_signatures[node.id] = signature
            print(
                "[场景规划] 单功能覆盖审计达到安全上限，结束该签名审计: "
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
        action_policy_violation = wechat_action_policy_violation(
            package_name,
            "click",
            f"点击{entry_name}",
        ) if has_gap else ""
        if protocol_errors or (has_gap and (not valid_entry_page or not valid_scenario)):
            result = _normalize_audit_result(_audit_function_for_gap(
                llm,
                graph,
                node,
                exploration_mode,
                validation_feedback=(
                    "上一次输出不符合覆盖审计协议。has_exploration_gap 必须是 JSON boolean；"
                    "false 时其余字段必须为空字符串；true 时 entry_page_ref、entry_name、test_scenario 必须全部非空。"
                    "entry_page_ref 必须是摘要中的已有 page_x；entry_name 必须是该页截图或 description 中"
                    "真实可见且尚未执行的入口名称；test_scenario 必须明确写出在该页点击 entry_name 并观察实际结果。"
                    f"字段问题: {protocol_errors}"
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
            action_policy_violation = wechat_action_policy_violation(
                package_name,
                "click",
                f"点击{entry_name}",
            ) if has_gap else ""
        if protocol_errors:
            print(
                "[场景规划] 覆盖遗漏检查连续返回非法结构，本轮不据此判定完成: "
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
                f"在“{_page_label(page)}”页面，点击“{entry_name}”，"
                "观察并记录实际出现的页面、弹窗或状态变化。"
            )
            valid_scenario = True
        if has_gap and (not valid_entry_page or not valid_scenario):
            print(
                "[场景规划] 覆盖遗漏检查坚持存在遗漏，但未给出可验证入口；"
                "本轮不据此判定完成: "
                f"{_function_label(node)} | page={entry_page_ref or 'none'} | "
                f"scenario={test_scenario or 'none'}"
            )
            return None, False, audited_signatures
        if action_policy_violation:
            print(
                "[场景规划] 覆盖遗漏候选涉及退出登录或删除账户，已忽略: "
                f"{_function_label(node)} | entry={entry_name or 'none'} | "
                f"reason={action_policy_violation}"
            )
        if has_gap and valid_entry_page and valid_scenario and not action_policy_violation:
            print(
                "[场景规划] 覆盖遗漏检查发现待验证入口，转为真实执行而非直接建图: "
                f"{node.id} | page={entry_page_ref} | scenario={test_scenario}"
            )
            return {
                "owner_function_id": node.id,
                "entry_page_id": entry_page_ref,
                "test_scenario": test_scenario,
            }, False, audited_signatures
        if has_gap:
            print(
                "[场景规划] 覆盖遗漏检查候选缺少有效已有页面或执行场景，已忽略: "
                f"{_function_label(node)} | entry_page_ref={entry_page_ref or 'none'} | "
                f"entry_name={entry_name or 'none'} | scenario={test_scenario or 'none'}"
            )
        else:
            print(f"[场景规划] 覆盖遗漏检查通过: {_function_label(node)}")
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
                "当前页仍属于目标功能但没有已记录的正向导航路径：若它明显是本轮入口页打开的"
                f"子页面、选择页或弹层，先执行一次 back，截图确认是否直接回到“{_page_label(entry_page)}”；"
                "若没有回到目标入口页或已离开目标应用，再执行一次 open_app 打开目标应用主页面，"
                "并按后续已记录路径导航"
            )
        else:
            instructions.append(
                "当前不在目标功能的已知导航路径上：先执行一次 open_app 直接打开目标应用主页面；"
                "若仍未到主页面，再关闭导航开始时已有的临时菜单或弹窗，并使用页面返回回到已记录的"
                f"“{_page_label(start_page)}”页面；若返回时离开目标应用，则再次 open_app"
            )
    for transition in path:
        src_page = graph.get_page(transition.src_page_id)
        dst_page = graph.get_page(transition.dst_page_id)
        operation = _clean_natural_text(transition.operation) or "执行已记录的页面操作"
        instructions.append(
            f"在“{_page_label(src_page)}”执行“{operation}”，进入“{_page_label(dst_page)}”"
        )
    if not path and start_page_id != entry_page_id:
        instructions.append(f"继续安全返回，直到进入已记录的“{_page_label(entry_page)}”页面")
    instructions.append(
        f"截图确认已到达“{_page_label(entry_page)}”后结束导航，"
        "不要在导航阶段点击本轮待验证入口；为到达目标页而主动打开的菜单属于导航路径，不要再把它关闭"
    )
    return "；".join(instructions) + "。"


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
                f"“{label}”",
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

    # 同一入口页可能同时属于当前父功能和刚从明确业务入口发现的多个新功能。
    # 优先给尚未开始的共享 owner 一次入口验证机会，避免父功能耗尽场景预算后
    # 才开始识别相册、文件、位置等独立能力；其余情况仍保持当前功能的连续性。
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
    # entry_page_ids 只提供进入功能的入口，不承载目标功能本身。把入口页
    # 当成导航终点会让执行器停在应用首页，再到其它模块里寻找同名控件。
    target_page_ids = list(target.coverage_page_ids)
    if not target_page_ids:
        return "", "目标功能尚无已记录的功能承载页", False

    current_page_id = _resolve_current_page_id(graph, current_page)
    if current_page_id in target_page_ids:
        return "", "当前位置已经属于目标功能页面", True

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
        return "", "当前位置和应用入口都没有到目标功能的已记录页面路径", False

    instructions: list[str] = []
    if restart_first:
        instructions.append("执行 open_app 重新打开目标应用，并等待界面稳定")
        root_page = _node_representative_page(graph, next(iter(graph.functions.values()), None))
        if root_page is not None:
            instructions.append(
                f"若打开后尚未处于“{_page_label(root_page)}”，"
                "逐次执行安全返回并每次截图确认，直到回到该已记录入口页"
            )
    for transition in path:
        src_page = graph.get_page(transition.src_page_id)
        dst_page = graph.get_page(transition.dst_page_id)
        instructions.append(
            f"在“{_page_label(src_page)}”执行“{_clean_natural_text(transition.operation)}”，"
            f"进入“{_page_label(dst_page)}”"
        )
    target_page = graph.get_page(path[-1].dst_page_id) if path else _node_representative_page(graph, target)
    instructions.append(f"确认到达“{_page_label(target_page)}”后结束导航")
    return "；".join(instructions) + "。", "使用 FFG 中已记录的最短页面路径", True


def _build_global_planning_summary(
    graph: FlowGraph,
    run_phase: str,
) -> str:
    lines = [
        "【全局 FFG 摘要】",
        f"运行模式: {_normalize_run_phase(run_phase)}",
        "功能状态:",
    ]
    sorted_functions = sorted(graph.functions.values(), key=_function_sort_index)
    for node in sorted_functions[:PLAN_MAX_GLOBAL_SUMMARY_FUNCTIONS]:
        lines.append(
            f"- {node.id} | {_function_label(node)} | status={node.test_status}"
            f" | coverage={node.coverage_percent}% | pages={len(node.page_ids)}"
        )
    omitted = len(sorted_functions) - PLAN_MAX_GLOBAL_SUMMARY_FUNCTIONS
    if omitted > 0:
        lines.append(f"- …[已省略 {omitted} 个功能，详情见本轮作用域摘要]")
    lines.append("功能依赖:")
    if graph.function_dependencies:
        for dependency in graph.function_dependencies:
            lines.append(
                f"- {dependency.src_function_id} -> {dependency.dst_function_id}"
                f" | resource={_clip_summary_text(dependency.resource_variable, 240)}"
            )
    else:
        lines.append("- 无")
    pending = _pending_cross_pairs(graph) if _normalize_run_phase(run_phase) == "all" else []
    lines.append("待测跨功能组合:")
    if pending:
        for src_id, dst_id, reason in pending:
            lines.append(f"- {src_id} -> {dst_id} | {_clip_summary_text(reason, 240)}")
    else:
        lines.append("- 无")
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
    navigation_reason = "没有目标功能"
    navigation_reachable = False
    if navigation_target is not None:
        navigation_scenario, navigation_reason, navigation_reachable = _navigation_to_function(
            graph,
            current_page,
            navigation_target,
        )

    lines = [
        "【当前位置】",
        _build_current_location_summary(graph, current_page),
        "",
        "【已知导航】",
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
                    f"- {transition.src_page_id}（{_page_label(graph.get_page(transition.src_page_id))}）"
                    f" --[{_clean_natural_text(transition.operation)}]--> "
                    f"{transition.dst_page_id}（{_page_label(graph.get_page(transition.dst_page_id))}）"
                )
        else:
            lines.append("- 无已记录路径；不得猜测跨功能导航")
            navigation_reachable = False

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
            f"- 关联页面: {node.page_ids or '无'}",
            f"- 仅导航入口页: {node.entry_page_ids or '无'}",
            f"- 资源变量: {node.resource_variables or '无'}",
            "- 已测试场景:",
        ])
        if node.scenarios:
            lines.extend(f"  - {_clip_summary_text(item, 360)}" for item in node.scenarios[-8:])
        else:
            lines.append("  - 无")
        lines.append("- 页面:")
        for page_id in node.page_ids:
            page = graph.get_page(page_id)
            if page is None:
                continue
            page_role = "仅导航入口" if page_id in node.entry_page_ids else "功能页面"
            lines.append(
                f"  - {page.id} | {_page_label(page)} | role={page_role} | "
                f"activity={page.activity or '未知'}"
            )
            if page.description:
                lines.append(f"    描述: {_clip_summary_text(page.description)}")
            if page.controls:
                lines.append("    控件: " + "；".join(_clip_summary_text(value, 180) for value in page.controls[-30:]))
            if page.operations:
                lines.append("    已执行: " + "；".join(_clip_summary_text(value, 180) for value in page.operations[-12:]))

    relevant_transitions = [
        transition
        for transition in graph.page_transitions
        if transition.src_page_id in scope_page_ids or transition.dst_page_id in scope_page_ids
    ]
    lines.extend(["", "【相关页面路径】"])
    if relevant_transitions:
        for transition in relevant_transitions:
            lines.append(
                f"- {transition.src_page_id}（{_page_label(graph.get_page(transition.src_page_id))}）"
                f" --[{_clean_natural_text(transition.operation)}]--> "
                f"{transition.dst_page_id}（{_page_label(graph.get_page(transition.dst_page_id))}）"
            )
    else:
        lines.append("- 无")
    return "\n".join(lines), navigation_scenario, navigation_reachable


def create_scenario_plan_node(llm, device):
    """创建统一场景规划节点。"""

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
                    "[场景规划] 当前 app 连续截图不可用或黑屏，结束该 app 运行: "
                    f"{unavailable_count}/{MAX_CONSECUTIVE_SCREENSHOT_UNAVAILABLE}"
                )
            else:
                print(
                    "[场景规划] 当前截图不可用或黑屏，本轮跳过、恢复后继续重试: "
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
                    "[场景规划] 连续无法恢复目标应用，结束运行以避免空转: "
                    f"{unavailable_count}/{MAX_CONSECUTIVE_TARGET_APP_UNAVAILABLE}"
                )
            else:
                print(
                    "[场景规划] 未确认目标应用页面，本轮跳过、重启后继续重试: "
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
                f"  探索范围: 核心功能 | 当前页面: {_page_label(current_page)} | 输入截图: yes"
            )
            print(
                "  FFG 统计: "
                f"功能={stats['functions']}, 页面={stats['pages']}, 操作={stats['operations']}, "
                f"页面跳转={stats['page_transitions']}, "
                f"功能依赖={stats['function_dependencies']}, 功能内场景={stats['intra_scenarios']}, "
                f"跨功能场景={stats['cross_scenarios']}"
            )
            print("[场景规划] 冷启动仅建图，本轮不生成执行场景；下一轮基于 FFG 摘要继续规划。")
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
                "[场景规划] 已完成节点按程序派生覆盖率恢复待测: "
                + ", ".join(reopened_functions)
            )

        # 覆盖率是“活”的（共享页面被其他功能测试会持续推高），完成判定
        # 不能只在功能成为本轮 target 时执行一次。每轮开始前对全部
        # in_progress 功能批量复查，满足条件的立即收尾。
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
                "[场景规划] 批量完成判定: 覆盖率已达阈值且自身证据充分，标记 completed: "
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

        # 先测试实际所在功能；只有没有可继续的功能内目标时，才调度跨功能场景。
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
                    "[场景规划] 覆盖率来自共享页面而非本功能自身测试，暂不收尾: "
                    f"{target_node.id} | {_function_label(target_node)} | "
                    f"coverage={target_node.coverage_percent}% | "
                    f"自身场景={intra_attempts}/{PLAN_MIN_SCENARIOS_BEFORE_COMPLETE} | "
                    f"未覆盖控件={len(uncovered_controls)}"
                )
            if coverage_reached:
                graph.set_node_test_status(target_node.id, "completed")
                print(
                    "[场景规划] 程序派生控件/操作覆盖率达到阈值，标记 completed 并等待遗漏审计: "
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
                    "[场景规划] 功能内场景达到普通预算但程序覆盖不足，"
                    "继续按未覆盖控件审计，不直接完成: "
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
                "FFG 中没有到剩余待测功能的已记录页面路径",
                state,
            )

        if not scenario_type:
            audit_updates = int(state.get("completion_audit_updates", 0) or 0)
            audit_attempts = state.get("completion_audit_attempts", {}) or {}
            previous_signatures = state.get("completion_audit_signatures", {}) or {}
            print("[场景规划] 已无待测功能或跨功能依赖，开始覆盖遗漏检查。")
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
                    "覆盖遗漏检查未返回可验证的完整 JSON，不能据此判定工作流完成",
                    state,
                )
                response["completion_audit_signatures"] = merged_signatures
                return response
            print("[场景规划] 所有功能、跨功能依赖和覆盖遗漏均已处理。")
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
                "FFG 中没有从当前位置或应用入口到目标功能的已记录页面路径",
                state,
            )
        if scenario_type == "cross_test":
            scope_instruction = (
                "【本轮固定作用域】必须返回 cross_test，"
                f"src_function={src_node.id}，dst_function={dst_node.id}。"
                "只围绕给定资源依赖生成一次源功能产生/修改、目标功能读取/展示/消费的闭环。"
            )
        else:
            scope_instruction = (
                "【本轮固定作用域】必须返回 intra_test，"
                f"target_function={target_node.id}。"
                "若仍有不同的可探索业务方向，返回非空 test_scenario；"
                "若已充分覆盖或只能重复已测试场景，把 navigation_scenario 和 test_scenario 留空。"
            )
        if _is_black_screen_page(graph, current_page):
            scope_instruction += (
                " 当前设备停留在已记录的黑屏异常 Page；navigation_scenario 必须先执行一次返回恢复，"
                "恢复后再按已知路径进入本轮固定功能。不得重新执行任何通向黑屏 Page 的历史操作。"
            )

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
                    "【本轮测试数据】创建或命名资源时使用唯一后缀 "
                    f"FuncDroid_{int(state.get('iteration_count', 0) or 0)}。"
                ),
            },
        ]
        planner_feedback = _clean_natural_text(state.get("planner_feedback", ""))
        rejected_scenario = _clean_natural_text(state.get("planner_rejected_scenario", ""))
        if planner_feedback:
            message_content.append({
                "type": "text",
                "text": (
                    f"【上一轮校验反馈】{planner_feedback}。"
                    + (f"被拒绝场景：{rejected_scenario}。" if rejected_scenario else "")
                    + (
                        "同类参数化入口的其他取值也视为代表性覆盖；"
                        "必须改选不同功能的控件，或把两个场景字段留空。"
                        if rejected_scenario else ""
                    )
                ),
            })
        if current_page and current_page.representative_screenshot:
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

        print(
            "[场景规划] 固定作用域: "
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
                f"规划器连续返回无效字段: {validation_errors}",
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
                f"规划器偏离固定功能 {target_node.id}",
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
                f"规划器偏离固定跨功能依赖 {src_node.id}->{dst_node.id}",
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
            r"返回|press_back|关闭|取消|open_app|重新打开|重启",
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
                "当前处于黑屏异常 Page，但规划结果没有先返回或重新打开 App 的恢复导航",
                state,
                rejected_scenario=test_scenario,
            )
        if _matches_black_screen_path(
            graph,
            f"{navigation_scenario}；{test_scenario}",
        ):
            return _planner_status_update_response(
                graph,
                current_page,
                exploration_mode,
                current_testing_function_id,
                current_testing_function_status,
                "规划结果重复了已知会导致黑屏或页面不可用的操作路径",
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
                    "程序派生覆盖率不足且仍有未处理控件，不能直接完成",
                    state,
                )
            graph.set_node_test_status(target_node.id, "completed")
            print(f"[场景规划] 功能规划完成: {_function_label(target_node)}")
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
