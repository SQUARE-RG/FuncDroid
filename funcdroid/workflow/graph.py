"""LangGraph 图组装入口。

统一规划工作流：
  scenario_plan → scenario_exec（提交所选模式的异步判定）→ ffg_update → (循环)
"""

import os
import time as _time
import traceback
from typing import Callable

from langgraph.graph import END, START, StateGraph

os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
os.environ.setdefault("LANGSMITH_TRACING", "false")

from model.flow_graph import FlowGraph
from workflow.nodes.scenario_plan import (
    FUNCTION_PHASE_DEPTH,
    _function_phase,
    create_scenario_plan_node,
)
from workflow.nodes.ffg_update import (
    _merge_page_audit_results,
    _record_execution_facts,
    _submit_page_audit_batch,
)
from workflow.page_audit_worker import page_audit_worker
from workflow.subgraphs.ffg_update import create_ffg_update_subgraph_node
from workflow.subgraphs.scenario_exec import create_scenario_exec_subgraph_node
from workflow.llm_usage import get_llm_usage_totals
from workflow.state import AgentState

DEFAULT_MAX_ITERATIONS = 0
LONG_RUN_IDLE_SECONDS = max(0.0, float(os.getenv("LONG_RUN_IDLE_SECONDS", "15")))
MAX_CONSECUTIVE_NODE_ERRORS = max(1, int(os.getenv("MAX_CONSECUTIVE_NODE_ERRORS", "3")))
MAX_SCENARIO_EXEC_NODE_ERRORS = MAX_CONSECUTIVE_NODE_ERRORS
STOP_AFTER_CONSECUTIVE_NODE_ERRORS = os.getenv(
    "STOP_AFTER_CONSECUTIVE_NODE_ERRORS", "0"
).strip().lower() in {"1", "true", "yes", "on"}
# 节点连续异常的指数退避上限（秒）。设备断连等“瞬间失败”的异常若不退避，
# 关闭 STOP_AFTER_CONSECUTIVE_NODE_ERRORS 时工作流会以满速空转直到时长用尽。
# 首次异常立即重试（多为瞬时故障），之后按 2、4、8… 秒退避，封顶该值；0 关闭退避。
NODE_ERROR_BACKOFF_MAX_SECONDS = max(
    0.0, float(os.getenv("NODE_ERROR_BACKOFF_MAX_SECONDS", "60"))
)
# 无条件硬上限：即使未开启 STOP_AFTER_CONSECUTIVE_NODE_ERRORS，同一节点连续
# 异常达到该次数也结束本次运行，避免不可恢复的基础设施故障让运行无限空转。
HARD_STOP_CONSECUTIVE_NODE_ERRORS = max(
    MAX_CONSECUTIVE_NODE_ERRORS,
    int(os.getenv(
        "HARD_STOP_CONSECUTIVE_NODE_ERRORS",
        str(MAX_CONSECUTIVE_NODE_ERRORS * 10),
    )),
)
# 无执行的跳过轮只每隔 N 轮输出一次轮次指标（首个跳过轮总会输出）；
# 指标需要遍历全部功能计算阶段/覆盖，空转轮逐轮打印没有信息量。
SKIP_ROUND_METRICS_EVERY = max(1, int(os.getenv("SKIP_ROUND_METRICS_EVERY", "10")))
NODE_ERROR_COUNT_KEYS = {
    "scenario_plan": "scenario_plan_error_count",
    "scenario_exec": "scenario_exec_error_count",
    "ffg_update": "ffg_update_error_count",
}


def _safe_node(node_name: str, node_func: Callable, fallback: Callable[[AgentState, Exception], dict]):
    """Wrap a LangGraph node so node-level exceptions don't stop the workflow."""

    def wrapped(state: AgentState) -> dict:
        try:
            result = node_func(state)
            counter_key = NODE_ERROR_COUNT_KEYS.get(node_name)
            if counter_key and isinstance(result, dict) and state.get(counter_key, 0):
                result = dict(result)
                result[counter_key] = 0
            return result
        except Exception as exc:
            print(f"[节点异常] {node_name} 执行失败: {exc}")
            print(traceback.format_exc())
            try:
                return fallback(state, exc)
            except Exception as fallback_exc:
                print(f"[节点异常] {node_name} fallback 也失败: {fallback_exc}")
                print(traceback.format_exc())
                return {}

    return wrapped


def _skip_plan() -> dict:
    """本轮不执行任何场景时使用的空计划。"""
    return {
        "scenario_type": "skip",
        "target_function_id": "",
        "src_function_id": "",
        "dst_function_id": "",
    }


def _threshold_reached(node_name: str, error_count: int, limit: int) -> tuple[bool, bool, str]:
    """判断节点连续异常是否应结束工作流，返回 (是否结束, 是否硬上限, 结束原因)。

    STOP_AFTER_CONSECUTIVE_NODE_ERRORS 开启时按 limit 提前结束；无论是否开启，
    达到 HARD_STOP_CONSECUTIVE_NODE_ERRORS 都无条件结束。
    """
    if error_count >= HARD_STOP_CONSECUTIVE_NODE_ERRORS:
        hard = True
        reason = (
            f"{node_name} 连续异常达到硬上限 "
            f"{error_count}/{HARD_STOP_CONSECUTIVE_NODE_ERRORS}，判定为不可恢复故障"
        )
    elif STOP_AFTER_CONSECUTIVE_NODE_ERRORS and error_count >= limit:
        hard = False
        reason = f"{node_name} 连续异常达到阈值 {error_count}/{limit}"
    else:
        return False, False, ""
    print(f"[节点异常] {reason}，提前结束工作流。")
    return True, hard, reason


def _node_error_backoff(node_name: str, error_count: int) -> None:
    """连续异常时指数退避，避免瞬间失败的节点形成满速空转。"""
    if error_count <= 1 or NODE_ERROR_BACKOFF_MAX_SECONDS <= 0:
        return
    delay = min(NODE_ERROR_BACKOFF_MAX_SECONDS, float(2 ** min(error_count - 1, 16)))
    print(f"[节点异常] {node_name} 连续异常 {error_count} 次，退避 {delay:g}s 后继续。")
    _time.sleep(delay)


def _error_stop_updates(node_name: str, error_count: int, limit: int) -> dict:
    """三个 fallback 共用：达到阈值则结束并记录原因，否则退避后继续。"""
    stop, hard, reason = _threshold_reached(node_name, error_count, limit)
    if not stop:
        _node_error_backoff(node_name, error_count)
        return {"workflow_complete": False}
    return {
        "workflow_complete": True,
        "workflow_stop_reason": reason,
        "workflow_hard_stop": hard,
    }


def _fallback_scenario_plan(state: AgentState, exc: Exception) -> dict:
    error_count = int(state.get("scenario_plan_error_count", 0) or 0) + 1
    iteration_count = int(state.get("iteration_count", 0) or 0) + 1
    return {
        "current_navigation_scenario": "",
        "current_test_scenario": "",
        "current_plan": _skip_plan(),
        "scenario_plan_error_count": error_count,
        "iteration_count": iteration_count,
        **_error_stop_updates("scenario_plan", error_count, MAX_CONSECUTIVE_NODE_ERRORS),
    }


def _fallback_scenario_exec(state: AgentState, exc: Exception) -> dict:
    if state.get("compliance_detection_enabled", False):
        from workflow.compliance_worker import compliance_worker

        compliance_worker.defer_checks(
            (state.get("current_plan") or {}).get("compliance_check_ids", []),
            f"合规场景执行失败，等待补测: {exc}",
        )
    error_count = int(state.get("scenario_exec_error_count", 0) or 0) + 1
    # 执行节点自身异常不是一次功能尝试：清空本轮场景，并以 infrastructure_error
    # 标记让 FFG 更新跳过场景记账与失败计数，避免把没执行过的场景记成“已尝试”。
    return {
        "current_page": None,
        "current_test_scenario": "",
        "navigation_execution_trace": [],
        "scenario_execution_trace": [],
        "scenario_exec_error_count": error_count,
        **_error_stop_updates("scenario_exec", error_count, MAX_SCENARIO_EXEC_NODE_ERRORS),
        "execution_result": {
            "navigation_status": "error",
            "navigation_error": "",
            "navigation_trace_length": 0,
            "scenario_status": "error",
            "scenario_error": f"场景执行节点异常: {exc}",
            "scenario_trace_length": 0,
            "infrastructure_error": True,
            "current_scenario": state.get("current_test_scenario", ""),
            "scenario_type": state.get("current_plan", {}).get("scenario_type", ""),
        },
    }


def _fallback_ffg_update(state: AgentState, exc: Exception) -> dict:
    error_count = int(state.get("ffg_update_error_count", 0) or 0) + 1
    graph = state.get("ffg")
    if not isinstance(graph, FlowGraph):
        graph = FlowGraph()
    # FFG 更新直接写在状态里的图上（不再使用副本），异常前已合并的页面审计结果
    # 与已应用的观测都保留；这里只补记本轮执行事实，避免场景与失败计数丢失。
    updates: dict = {}
    try:
        updates["function_execution_failure_counts"] = _record_execution_facts(
            graph, state, state.get("execution_result", {}) or {},
        )
    except Exception as facts_exc:  # noqa: BLE001
        print(f"[FFG 更新] 补记执行事实失败: {facts_exc}")
    print(f"[FFG 更新] 本轮更新异常，保留已写入的 FFG 内容: {exc}")
    return {
        "ffg": graph,
        **updates,
        "ffg_update_error_count": error_count,
        **_error_stop_updates("ffg_update", error_count, MAX_CONSECUTIVE_NODE_ERRORS),
    }


def _long_run_converged(
    graph: FlowGraph | None,
    submitted_fingerprints: set[tuple[str, str]] | None = None,
) -> bool:
    """规划器已无事可做时，只有页面审计还能给图带来新遗漏。

    跳过轮不经过 FFG 更新，审计结果在这里合并、未审页面在这里补交；
    没有新变化、也没有在途审计时，继续等时长只是空转，应提前结束。

    submitted_fingerprints 记录本次空闲期内已提交过的 (page_id, 指纹)：
    失败的审计会让页面重新变为“未审计”，若每个空闲轮都把它重新提交并
    视为“尚未收敛”，一个持续失败的页面就能让运行空转到时长耗尽。同一
    空闲期内只提交新指纹，只有新结果/合并变化或新指纹才算未收敛。
    """
    if graph is None:
        return True
    if submitted_fingerprints is None:
        submitted_fingerprints = set()
    changes = _merge_page_audit_results(graph)
    submitted = _submit_page_audit_batch(
        graph,
        exclude_fingerprints=submitted_fingerprints,
    )
    pending = page_audit_worker.pending_count()
    if changes or submitted or pending:
        return False
    print("[长时运行] 覆盖复核与页面审计均无新结果，提前结束，不再空转等待时长。")
    return True


def _finish_without_round(
    iteration_count: int,
    iteration_started_at: float,
    usage_totals: dict,
) -> dict:
    """达到轮次/时长上限时直接收尾：不调用规划器，轮次数保持不变。"""
    return {
        "iteration_count": iteration_count,
        "iteration_started_at": iteration_started_at,
        "iteration_llm_requests_start": int(usage_totals.get("requests", 0) or 0),
        "iteration_llm_tokens_start": int(usage_totals.get("total_tokens", 0) or 0),
        "current_navigation_scenario": "",
        "current_test_scenario": "",
        "current_plan": _skip_plan(),
        "workflow_complete": True,
        "navigation_execution_trace": [],
        "scenario_execution_trace": [],
        "execution_result": {},
    }


def _create_scenario_plan_node_with_iteration(planning_node: Callable):
    """把轮次/收尾控制收进规划节点，保持工作流只有三个业务节点。"""

    def scenario_plan_with_iteration(state: AgentState) -> dict:
        iteration_started_at = _time.time()
        usage_totals = get_llm_usage_totals()
        previous_count = int(state.get("iteration_count", 0) or 0)
        max_iterations = int(state.get("max_iterations", DEFAULT_MAX_ITERATIONS) or 0)
        if max_iterations > 0 and previous_count >= max_iterations:
            print(
                f"[轮次信息] 已完成最大轮次 {max_iterations}，"
                "不再为断点恢复或异常重入额外调用规划模型。"
            )
            return _finish_without_round(previous_count, iteration_started_at, usage_totals)
        time_limit_reason = _time_limit_reason(state)
        if time_limit_reason:
            print(f"[轮次信息] {time_limit_reason}，不再启动新的规划或场景。")
            # 本轮并未执行任何规划或场景，轮次数保持不变。
            return _finish_without_round(previous_count, iteration_started_at, usage_totals)
        count = previous_count + 1
        run_phase = state.get("run_phase", "all")
        run_phase_label = {
            "all": "统一规划",
            "intra": "仅功能内测试",
        }.get(run_phase, "统一规划")
        print(f"\n{'='*60}")
        print(f"  第 {count} 轮 — 模式: {run_phase_label} | 探索范围: 核心功能")
        print(f"{'='*60}\n")
        navigation_failure_count = int(state.get("navigation_failure_count", 0) or 0)
        if navigation_failure_count:
            print(f"[轮次] 连续导航失败: {navigation_failure_count}")
        planning_state = dict(state)
        planning_state["iteration_count"] = count
        # 状态迁移已在规划节点开头统一完成；跳过轮直接进入下一轮，不在轮内重复调用规划器。
        planning_updates = dict(planning_node(planning_state))
        if planning_updates.get("workflow_complete", False):
            projected_state = dict(planning_state)
            projected_state.update(planning_updates)
            has_time_budget = float(projected_state.get("total_time_limit", 0) or 0) > 0
            idle_audit_fingerprints = {
                (str(item[0]), str(item[1]))
                for item in (state.get("long_run_audit_fingerprints") or [])
                if isinstance(item, (list, tuple)) and len(item) == 2
            }
            if (
                has_time_budget
                and state.get("run_mode") != "compliance"
                and not _time_limit_reason(projected_state)
                and not _long_run_converged(
                    projected_state.get("ffg"),
                    idle_audit_fingerprints,
                )
            ):
                idle_count = int(state.get("long_run_idle_count", 0) or 0) + 1
                print(
                    "[长时运行] 当前 FFG 暂无新遗漏，但页面审计仍有新结果或未完成任务；"
                    "等待审计后继续覆盖复核。"
                )
                if LONG_RUN_IDLE_SECONDS > 0:
                    remaining = max(
                        0.0,
                        float(projected_state.get("total_time_limit", 0) or 0)
                        - (
                            _time.time()
                            - float(projected_state.get("start_time", 0) or _time.time())
                        ),
                    )
                    # 审计一完成就回到规划，不必睡满空闲间隔。
                    page_audit_worker.flush(timeout=min(LONG_RUN_IDLE_SECONDS, remaining))
                planning_updates = dict(planning_updates)
                # Keep completion_audit_signatures so the coverage-gap audit
                # stays incremental across idle rounds. Clearing them here forced
                # every completed function to be re-audited from scratch each
                # idle cycle, which is exactly the "反复重审已完成功能" waste.
                planning_updates.update({
                    "workflow_complete": False,
                    "current_navigation_scenario": "",
                    "current_test_scenario": "",
                    "current_testing_function_id": "",
                    "current_testing_function_status": "not_started",
                    "current_plan": _skip_plan(),
                    "long_run_idle_count": idle_count,
                    "long_run_audit_fingerprints": sorted(idle_audit_fingerprints),
                })
            else:
                planning_updates["long_run_idle_count"] = 0
                planning_updates["long_run_audit_fingerprints"] = []
        elif (planning_updates.get("current_plan", {}) or {}).get("scenario_type") in {
            "intra_test",
            "cross_test",
            "compliance_check",
        }:
            # 真正执行场景后图会变化，空闲期的已提交指纹记录随之作废。
            planning_updates["long_run_idle_count"] = 0
            planning_updates["long_run_audit_fingerprints"] = []
        if "target_app_unavailable_count" not in planning_updates:
            planning_updates["target_app_unavailable_count"] = 0
        return {
            "iteration_count": count,
            "iteration_started_at": iteration_started_at,
            "iteration_llm_requests_start": int(usage_totals.get("requests", 0) or 0),
            "iteration_llm_tokens_start": int(usage_totals.get("total_tokens", 0) or 0),
            "navigation_execution_trace": [],
            "scenario_execution_trace": [],
            "execution_result": {},
            **planning_updates,
        }

    return scenario_plan_with_iteration


# 连续跳过轮计数：只用于控制跳过轮指标日志的输出频率。
_CONSECUTIVE_SKIP_ROUNDS = 0


def _route_after_plan(state: AgentState) -> str:
    """场景规划之后，决定是否需要执行设备动作。"""
    global _CONSECUTIVE_SKIP_ROUNDS
    if state.get("workflow_complete", False):
        _print_iteration_metrics(state, "plan_end")
        stop_reason = str(state.get("workflow_stop_reason", "") or "")
        if stop_reason:
            print(f"\n[路由] {stop_reason}，结束工作流。")
        else:
            print("\n[路由] 程序判断整体探索已完成，跳过执行。")
        return "end"
    scenario_type = (state.get("current_plan", {}) or {}).get("scenario_type")
    time_limit_reason = _time_limit_reason(state)
    if time_limit_reason:
        _print_iteration_metrics(state, "plan_hard_limit")
        print(f"\n[路由] {time_limit_reason}，不再启动新的场景执行。")
        return "end"
    iteration_count = int(state.get("iteration_count", 0) or 0)
    max_iterations = int(state.get("max_iterations", DEFAULT_MAX_ITERATIONS) or 0)
    past_iteration_limit = max_iterations > 0 and iteration_count > max_iterations
    at_iteration_limit_without_action = (
        max_iterations > 0
        and iteration_count == max_iterations
        and scenario_type not in {"intra_test", "cross_test", "compliance_check"}
    )
    if past_iteration_limit or at_iteration_limit_without_action:
        _print_iteration_metrics(state, "plan_hard_limit")
        print(f"\n[路由] 已达到最大轮次 {max_iterations}，不再进入无动作规划循环。")
        return "end"
    if scenario_type == "skip":
        _CONSECUTIVE_SKIP_ROUNDS += 1
        if (
            _CONSECUTIVE_SKIP_ROUNDS == 1
            or _CONSECUTIVE_SKIP_ROUNDS % SKIP_ROUND_METRICS_EVERY == 0
        ):
            _print_iteration_metrics(state, "skip")
        print("\n[路由] 本轮无执行场景，进入下一轮规划。")
        return "continue"
    _CONSECUTIVE_SKIP_ROUNDS = 0
    return "scenario_exec"


def _hard_limit_reason(state: AgentState) -> str:
    """Return the reason for a hard workflow stop, if any."""
    iteration_count = int(state.get("iteration_count", 0) or 0)
    max_iterations = int(state.get("max_iterations", DEFAULT_MAX_ITERATIONS) or 0)
    if max_iterations > 0 and iteration_count >= max_iterations:
        return f"已达到最大轮次 {max_iterations}"

    return _time_limit_reason(state)


def _time_limit_reason(state: AgentState) -> str:
    start_time = float(state.get("start_time", 0) or 0)
    total_time_limit = float(state.get("total_time_limit", 0) or 0)
    if total_time_limit > 0 and start_time > 0:
        elapsed = _time.time() - start_time
        if elapsed >= total_time_limit:
            return f"已用时 {elapsed / 60:.1f} 分钟，达到设定最长时间"
    return ""


def _route_after_update(state: AgentState) -> str:
    """FFG 更新之后，决定下一步。

    检查顺序：
    1. 是否达到最大轮次 → 结束
    2. 是否达到设定最长时间 → 结束
    3. 是否整体测试完成 → 结束
    4. 否则 → 继续下一轮
    """
    iteration_count = int(state.get("iteration_count", 0) or 0)
    _print_iteration_metrics(state, "executed")
    hard_limit_reason = _hard_limit_reason(state)
    if hard_limit_reason:
        print(f"\n[路由] {hard_limit_reason}，结束工作流。")
        return "end"

    if state.get("workflow_complete", False):
        stop_reason = str(state.get("workflow_stop_reason", "") or "")
        if stop_reason:
            print(f"\n[路由] {stop_reason}，结束工作流。")
        else:
            print("\n[路由] 整体探索已完成，工作流结束。")
        return "end"

    print(f"\n[路由] 第 {iteration_count} 轮完成，继续统一规划。")
    return "continue"


def _print_iteration_metrics(state: AgentState, outcome: str) -> None:
    started_at = float(state.get("iteration_started_at", 0) or 0)
    elapsed_seconds = max(0.0, _time.time() - started_at) if started_at else 0.0
    usage_totals = get_llm_usage_totals()
    request_delta = max(
        0,
        int(usage_totals.get("requests", 0) or 0)
        - int(state.get("iteration_llm_requests_start", 0) or 0),
    )
    token_delta = max(
        0,
        int(usage_totals.get("total_tokens", 0) or 0)
        - int(state.get("iteration_llm_tokens_start", 0) or 0),
    )
    graph = state.get("ffg")
    functions = pages = completed = blocked = unavailable = scenarios = 0
    breadth_functions = depth_functions = depth_scenarios = 0
    if isinstance(graph, FlowGraph):
        active_nodes = graph.active_nodes()
        functions = len(active_nodes)
        pages = len(graph.pages)
        # 单次遍历汇总全部计数，状态直接读节点字段（不再逐项 get_node 查找）。
        for node in active_nodes:
            status = node.test_status or "not_started"
            if status == "completed":
                completed += 1
            elif status in {"blocked", "deferred"}:
                blocked += 1
            elif status == "unavailable":
                unavailable += 1
            scenarios += len(node.scenarios)
            depth_scenarios += int(node.depth_scenario_count or 0)
            if _function_phase(graph, node) == FUNCTION_PHASE_DEPTH:
                depth_functions += 1
            else:
                breadth_functions += 1
    completion_rate = (completed / functions) if functions else 0.0
    execution_result = state.get("execution_result", {}) or {}
    current_plan = state.get("current_plan", {}) or {}
    scenario_type = str(current_plan.get("scenario_type", "") or "none")
    if scenario_type == "cross_test":
        scope = (
            f"{current_plan.get('src_function_id', '') or 'unknown'}>"
            f"{current_plan.get('dst_function_id', '') or 'unknown'}"
        )
    else:
        scope = str(
            current_plan.get("target_function_id", "")
            or state.get("current_testing_function_id", "")
            or "none"
        )
    current_page = state.get("current_page")
    page_id = str(getattr(current_page, "id", "") or "none")
    page_activity = str(getattr(current_page, "activity", "") or "none").replace(",", ";")
    print(
        "[轮次指标] "
        f"iteration={int(state.get('iteration_count', 0) or 0)}, outcome={outcome}, "
        f"scenario_type={scenario_type}, scope={scope}, page={page_id}, "
        f"activity={page_activity}, "
        f"elapsed_seconds={elapsed_seconds:.3f}, llm_requests={request_delta}, "
        f"llm_tokens={token_delta}, functions={functions}, pages={pages}, "
        f"completed_functions={completed}, blocked_functions={blocked}, "
        f"unavailable_functions={unavailable}, completion_rate={completion_rate:.3f}, "
        f"attempted_scenarios={scenarios}, "
        f"breadth_functions={breadth_functions}, depth_functions={depth_functions}, "
        f"depth_scenarios={depth_scenarios}, "
        f"scenario_status={execution_result.get('scenario_status', '') or 'none'}"
    )


def build_graph(
    planning_llm,
    execution_llm,
    device,
    ffg_update_llm=None,
):
    """构建场景规划、场景执行（含缺陷检测）、FFG 更新三节点循环工作流。

    Args:
        planning_llm: 规划使用的 LLM 客户端
        execution_llm: 场景执行和执行后缺陷复核使用的 LLM 客户端
        device: 设备驱动
        ffg_update_llm: FFG 更新专用 LLM 客户端；None 时沿用 planning_llm。
            FFG 更新单请求携带大量轨迹截图（mandatory 强制图不受
            image_budget 约束），响应时间显著长于规划，需要独立超时。
    """
    if ffg_update_llm is None:
        ffg_update_llm = planning_llm
    graph = StateGraph(AgentState)

    # ─── 注册节点 ───
    planning_node = create_scenario_plan_node(planning_llm, device)
    graph.add_node(
        "scenario_plan",
        _safe_node(
            "scenario_plan",
            _create_scenario_plan_node_with_iteration(planning_node),
            _fallback_scenario_plan,
        ),
    )
    graph.add_node(
        "scenario_exec",
        _safe_node(
            "scenario_exec",
            create_scenario_exec_subgraph_node(execution_llm, device),
            _fallback_scenario_exec,
        ),
    )
    graph.add_node(
        "ffg_update",
        _safe_node(
            "ffg_update",
            create_ffg_update_subgraph_node(ffg_update_llm, device),
            _fallback_ffg_update,
        ),
    )
    # ─── 定义边 ───

    graph.add_edge(START, "scenario_plan")

    # 统一规划节点 → 执行节点，或直接结束
    graph.add_conditional_edges(
        "scenario_plan",
        _route_after_plan,
        {
            "scenario_exec": "scenario_exec",
            "continue": "scenario_plan",
            "end": END,
        },
    )

    # 场景执行节点已提交异步判定，随后更新 FFG。
    graph.add_edge("scenario_exec", "ffg_update")

    # FFG 更新后 → 路由决策
    graph.add_conditional_edges(
        "ffg_update",
        _route_after_update,
        {
            "continue": "scenario_plan",            # 继续当前阶段
            "end": END,                              # 结束
        },
    )

    return graph.compile()
