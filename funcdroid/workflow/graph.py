"""LangGraph 图组装入口。

统一规划工作流：
  scenario_plan → scenario_exec（含缺陷检测）→ ffg_update → (循环)
"""

import os
import time as _time
import traceback
from typing import Callable

from langgraph.graph import END, START, StateGraph

os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
os.environ.setdefault("LANGSMITH_TRACING", "false")

from model.flow_graph import FlowGraph
from workflow.nodes.scenario_plan import create_scenario_plan_node
from workflow.subgraphs.ffg_update import create_ffg_update_subgraph_node
from workflow.subgraphs.scenario_exec import create_scenario_exec_subgraph_node
from workflow.llm_usage import get_llm_usage_summary
from workflow.state import AgentState

DEFAULT_MAX_ITERATIONS = 0
LONG_RUN_IDLE_SECONDS = max(0.0, float(os.getenv("LONG_RUN_IDLE_SECONDS", "15")))
MAX_CONSECUTIVE_NODE_ERRORS = max(1, int(os.getenv("MAX_CONSECUTIVE_NODE_ERRORS", "3")))
MAX_SCENARIO_EXEC_NODE_ERRORS = MAX_CONSECUTIVE_NODE_ERRORS
MAX_INLINE_STATUS_TRANSITIONS = max(
    1,
    int(os.getenv("PLAN_MAX_INLINE_STATUS_TRANSITIONS", "2")),
)
STOP_AFTER_CONSECUTIVE_NODE_ERRORS = os.getenv(
    "STOP_AFTER_CONSECUTIVE_NODE_ERRORS", "0"
).strip().lower() in {"1", "true", "yes", "on"}
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


def _fallback_scenario_plan(state: AgentState, exc: Exception) -> dict:
    error_count = int(state.get("scenario_plan_error_count", 0) or 0) + 1
    workflow_complete = (
        STOP_AFTER_CONSECUTIVE_NODE_ERRORS
        and error_count >= MAX_CONSECUTIVE_NODE_ERRORS
    )
    if workflow_complete:
        print(
            "[节点异常] scenario_plan 连续异常达到阈值，提前结束工作流: "
            f"{error_count}/{MAX_CONSECUTIVE_NODE_ERRORS}"
        )
    return {
        "current_navigation_scenario": "",
        "current_test_scenario": "",
        "current_plan": {
            "scenario_type": "status_update",
            "target_function_id": "",
            "src_function_id": "",
            "dst_function_id": "",
        },
        "scenario_plan_error_count": error_count,
        "workflow_complete": workflow_complete,
    }


def _fallback_scenario_exec(state: AgentState, exc: Exception) -> dict:
    error_count = int(state.get("scenario_exec_error_count", 0) or 0) + 1
    workflow_complete = (
        STOP_AFTER_CONSECUTIVE_NODE_ERRORS
        and error_count >= MAX_SCENARIO_EXEC_NODE_ERRORS
    )
    if workflow_complete:
        print(
            "[节点异常] scenario_exec 连续异常达到阈值，提前结束工作流: "
            f"{error_count}/{MAX_SCENARIO_EXEC_NODE_ERRORS}"
        )
    return {
        "current_page": None,
        "navigation_execution_trace": [],
        "scenario_execution_trace": [],
        "scenario_exec_error_count": error_count,
        "workflow_complete": workflow_complete,
        "execution_result": {
            "navigation_status": "error",
            "navigation_error": "",
            "navigation_trace_length": 0,
            "scenario_status": "error",
            "scenario_error": f"场景执行节点异常: {exc}",
            "scenario_trace_length": 0,
            "current_scenario": state.get("current_test_scenario", ""),
            "scenario_type": state.get("current_plan", {}).get("scenario_type", ""),
        },
    }


def _fallback_ffg_update(state: AgentState, exc: Exception) -> dict:
    error_count = int(state.get("ffg_update_error_count", 0) or 0) + 1
    workflow_complete = (
        STOP_AFTER_CONSECUTIVE_NODE_ERRORS
        and error_count >= MAX_CONSECUTIVE_NODE_ERRORS
    )
    graph = state.get("ffg")
    if not isinstance(graph, FlowGraph):
        graph = FlowGraph()
    print(f"[FFG 更新] 已跳过本轮更新，保留原 FFG: {exc}")
    if workflow_complete:
        print(
            "[节点异常] ffg_update 连续异常达到阈值，提前结束工作流: "
            f"{error_count}/{MAX_CONSECUTIVE_NODE_ERRORS}"
        )
    return {
        "ffg": graph,
        "ffg_update_error_count": error_count,
        "workflow_complete": workflow_complete,
    }


def _create_scenario_plan_node_with_iteration(planning_node: Callable):
    """把轮次/收尾控制收进规划节点，保持工作流只有三个业务节点。"""

    def scenario_plan_with_iteration(state: AgentState) -> dict:
        iteration_started_at = _time.time()
        usage_summary = get_llm_usage_summary()
        usage_totals = usage_summary.get("totals", {}) if isinstance(usage_summary, dict) else {}
        previous_count = int(state.get("iteration_count", 0) or 0)
        max_iterations = int(state.get("max_iterations", DEFAULT_MAX_ITERATIONS) or 0)
        if max_iterations > 0 and previous_count >= max_iterations:
            print(
                f"[轮次信息] 已完成最大轮次 {max_iterations}，"
                "不再为断点恢复或异常重入额外调用规划模型。"
            )
            return {
                "iteration_count": previous_count,
                "iteration_started_at": iteration_started_at,
                "iteration_llm_requests_start": int(usage_totals.get("requests", 0) or 0),
                "iteration_llm_tokens_start": int(usage_totals.get("total_tokens", 0) or 0),
                "current_navigation_scenario": "",
                "current_test_scenario": "",
                "current_plan": {
                    "scenario_type": "status_update",
                    "target_function_id": "",
                    "src_function_id": "",
                    "dst_function_id": "",
                },
                "workflow_complete": True,
                "navigation_execution_trace": [],
                "scenario_execution_trace": [],
                "execution_result": {},
            }
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
        time_limit_reason = _time_limit_reason(state)
        if time_limit_reason:
            print(f"[轮次信息] {time_limit_reason}，不再启动新的规划或场景。")
            return {
                "iteration_count": count,
                "iteration_started_at": iteration_started_at,
                "iteration_llm_requests_start": int(usage_totals.get("requests", 0) or 0),
                "iteration_llm_tokens_start": int(usage_totals.get("total_tokens", 0) or 0),
                "current_navigation_scenario": "",
                "current_test_scenario": "",
                "current_plan": {
                    "scenario_type": "status_update",
                    "target_function_id": "",
                    "src_function_id": "",
                    "dst_function_id": "",
                },
                "workflow_complete": True,
                "navigation_execution_trace": [],
                "scenario_execution_trace": [],
                "execution_result": {},
            }
        planning_state = dict(state)
        planning_state["iteration_count"] = count
        planning_updates = planning_node(planning_state)
        accumulated_planning_updates = dict(planning_updates)
        previous_status_signature: tuple | None = None
        for _ in range(MAX_INLINE_STATUS_TRANSITIONS):
            scenario_type = (
                accumulated_planning_updates.get("current_plan", {}) or {}
            ).get("scenario_type")
            if (
                scenario_type != "status_update"
                or accumulated_planning_updates.get("workflow_complete", False)
            ):
                break
            status_signature = (
                str(accumulated_planning_updates.get("current_testing_function_id", "") or ""),
                str(accumulated_planning_updates.get("current_testing_function_status", "") or ""),
                int(accumulated_planning_updates.get("planner_noop_count", 0) or 0),
                str(accumulated_planning_updates.get("planner_feedback", "") or ""),
                str(
                    (accumulated_planning_updates.get("current_plan", {}) or {}).get(
                        "target_function_id", ""
                    )
                    or ""
                ),
            )
            if status_signature == previous_status_signature:
                break
            previous_status_signature = status_signature
            projected_state = dict(planning_state)
            projected_state.update(accumulated_planning_updates)
            if _time_limit_reason(projected_state):
                break
            print(
                "[调度器] 状态切换已在当前轮内完成，"
                "继续选择可执行场景，不单独产生纯状态轮。"
            )
            next_updates = planning_node(projected_state)
            accumulated_planning_updates.update(next_updates)
        planning_updates = accumulated_planning_updates
        if planning_updates.get("workflow_complete", False):
            projected_state = dict(planning_state)
            projected_state.update(planning_updates)
            has_time_budget = float(projected_state.get("total_time_limit", 0) or 0) > 0
            if has_time_budget and not _time_limit_reason(projected_state):
                idle_count = int(state.get("long_run_idle_count", 0) or 0) + 1
                print(
                    "[长时运行] 当前 FFG 暂无新遗漏，但指定时长尚未到达；"
                    "保留同一张图并继续覆盖复核。"
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
                    _time.sleep(min(LONG_RUN_IDLE_SECONDS, remaining))
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
                    "current_plan": {
                        "scenario_type": "status_update",
                        "target_function_id": "",
                        "src_function_id": "",
                        "dst_function_id": "",
                    },
                    "long_run_idle_count": idle_count,
                })
            else:
                planning_updates["long_run_idle_count"] = 0
        elif (planning_updates.get("current_plan", {}) or {}).get("scenario_type") in {
            "intra_test",
            "cross_test",
        }:
            planning_updates["long_run_idle_count"] = 0
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


def _route_after_plan(state: AgentState) -> str:
    """场景规划之后，决定是否需要执行设备动作。"""
    if state.get("workflow_complete", False):
        _print_iteration_metrics(state, "plan_end")
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
        and scenario_type not in {"intra_test", "cross_test"}
    )
    if past_iteration_limit or at_iteration_limit_without_action:
        _print_iteration_metrics(state, "plan_hard_limit")
        print(f"\n[路由] 已达到最大轮次 {max_iterations}，不再进入无动作规划循环。")
        return "end"
    if scenario_type == "bootstrap":
        _print_iteration_metrics(state, "bootstrap")
        print("\n[路由] 冷启动建图完成，本轮跳过执行，进入下一轮摘要驱动规划。")
        return "continue"
    if scenario_type == "ffg_audit_update":
        _print_iteration_metrics(state, "ffg_audit_update")
        print("\n[路由] 终止审计已更新 FFG，本轮跳过执行，进入下一轮摘要驱动规划。")
        return "continue"
    if scenario_type == "status_update":
        _print_iteration_metrics(state, "status_update")
        print("\n[路由] 规划器仅更新功能状态，本轮跳过执行，进入下一轮规划。")
        return "continue"
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
        print("\n[路由] 整体探索已完成，工作流结束。")
        return "end"

    print(f"\n[路由] 第 {iteration_count} 轮完成，继续统一规划。")
    return "continue"


def _print_iteration_metrics(state: AgentState, outcome: str) -> None:
    started_at = float(state.get("iteration_started_at", 0) or 0)
    elapsed_seconds = max(0.0, _time.time() - started_at) if started_at else 0.0
    usage_summary = get_llm_usage_summary()
    usage_totals = usage_summary.get("totals", {}) if isinstance(usage_summary, dict) else {}
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
    if isinstance(graph, FlowGraph):
        active_nodes = graph.active_nodes()
        functions = len(active_nodes)
        pages = len(graph.active_pages())
        completed = sum(
            1 for node in active_nodes if graph.get_node_test_status(node.id) == "completed"
        )
        blocked = sum(
            1 for node in active_nodes if graph.get_node_test_status(node.id) in {"blocked", "deferred"}
        )
        unavailable = sum(
            1 for node in active_nodes if graph.get_node_test_status(node.id) == "unavailable"
        )
        scenarios = sum(len(node.scenarios) for node in active_nodes)
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
        f"activity={page_activity}, planner_noop_count={int(state.get('planner_noop_count', 0) or 0)}, "
        f"elapsed_seconds={elapsed_seconds:.3f}, llm_requests={request_delta}, "
        f"llm_tokens={token_delta}, functions={functions}, pages={pages}, "
        f"completed_functions={completed}, blocked_functions={blocked}, "
        f"unavailable_functions={unavailable}, completion_rate={completion_rate:.3f}, "
        f"attempted_scenarios={scenarios}, "
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

    # 场景执行节点内部已完成缺陷检测，随后更新 FFG。
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
