"""LangGraph graph assembly entry point.

Unified planning workflow:
  scenario_plan -> scenario_exec (including defect detection) -> ffg_update -> (loop)
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
            print(f"[Node Exception] {node_name} execution failed: {exc}")
            print(traceback.format_exc())
            try:
                return fallback(state, exc)
            except Exception as fallback_exc:
                print(f"[Node Exception] {node_name} fallback also failed: {fallback_exc}")
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
            "[Node Exception] scenario_plan consecutive exceptions reached threshold, terminating workflow early: "
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
            "[Node Exception] scenario_exec consecutive exceptions reached threshold, terminating workflow early: "
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
            "scenario_error": f"Scenario execution node exception: {exc}",
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
    print(f"[FFG Update] Skipped this update round, retaining original FFG: {exc}")
    if workflow_complete:
        print(
            "[Node Exception] ffg_update consecutive exceptions reached threshold, terminating workflow early: "
            f"{error_count}/{MAX_CONSECUTIVE_NODE_ERRORS}"
        )
    return {
        "ffg": graph,
        "ffg_update_error_count": error_count,
        "workflow_complete": workflow_complete,
    }


def _create_scenario_plan_node_with_iteration(planning_node: Callable):
    """Incorporate round/finalization control into the planning node, keeping the workflow to only three business nodes."""

    def scenario_plan_with_iteration(state: AgentState) -> dict:
        iteration_started_at = _time.time()
        usage_summary = get_llm_usage_summary()
        usage_totals = usage_summary.get("totals", {}) if isinstance(usage_summary, dict) else {}
        previous_count = int(state.get("iteration_count", 0) or 0)
        max_iterations = int(state.get("max_iterations", DEFAULT_MAX_ITERATIONS) or 0)
        if max_iterations > 0 and previous_count >= max_iterations:
            print(
                f"[Round Info] Maximum iterations {max_iterations} reached, "
                "will not make additional planning model calls for checkpoint recovery or exception reentry."
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
            "all": "Unified Planning",
            "intra": "Intra-function Only Testing",
        }.get(run_phase, "Unified Planning")
        print(f"\n{'='*60}")
        print(f"  Round {count} - Mode: {run_phase_label} | Exploration Scope: Core Functions")
        print(f"{'='*60}\n")
        navigation_failure_count = int(state.get("navigation_failure_count", 0) or 0)
        if navigation_failure_count:
            print(f"[Round] Consecutive navigation failures: {navigation_failure_count}")
        time_limit_reason = _time_limit_reason(state)
        if time_limit_reason:
            print(f"[Round Info] {time_limit_reason}, not starting new planning or scenarios.")
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
                "[Scheduler] State transition completed within current round, "
                "continuing to select executable scenario, no separate pure state round produced."
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
                    "[Long Run] Current FFG has no new omissions, but specified duration not yet reached; "
                    "retaining the same graph and continuing coverage review."
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
                # idle cycle, which is exactly the "repeatedly reviewing completed functions" waste.
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
    """Decide whether device actions need to be executed after scenario planning."""
    if state.get("workflow_complete", False):
        _print_iteration_metrics(state, "plan_end")
        print("\n[Routing] Program determined overall exploration is complete, skipping execution.")
        return "end"
    scenario_type = (state.get("current_plan", {}) or {}).get("scenario_type")
    time_limit_reason = _time_limit_reason(state)
    if time_limit_reason:
        _print_iteration_metrics(state, "plan_hard_limit")
        print(f"\n[Routing] {time_limit_reason}, not starting new scenario execution.")
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
        print(f"\n[Routing] Maximum iterations {max_iterations} reached, not entering actionless planning loop.")
        return "end"
    if scenario_type == "bootstrap":
        _print_iteration_metrics(state, "bootstrap")
        print("\n[Routing] Cold start graph building completed, skipping execution this round, entering next round of summary-driven planning.")
        return "continue"
    if scenario_type == "ffg_audit_update":
        _print_iteration_metrics(state, "ffg_audit_update")
        print("\n[Routing] Termination audit updated FFG, skipping execution this round, entering next round of summary-driven planning.")
        return "continue"
    if scenario_type == "status_update":
        _print_iteration_metrics(state, "status_update")
        print("\n[Routing] Planner only updated function state, skipping execution this round, entering next round of planning.")
        return "continue"
    return "scenario_exec"


def _hard_limit_reason(state: AgentState) -> str:
    """Return the reason for a hard workflow stop, if any."""
    iteration_count = int(state.get("iteration_count", 0) or 0)
    max_iterations = int(state.get("max_iterations", DEFAULT_MAX_ITERATIONS) or 0)
    if max_iterations > 0 and iteration_count >= max_iterations:
        return f"Maximum iterations {max_iterations} reached"

    return _time_limit_reason(state)


def _time_limit_reason(state: AgentState) -> str:
    start_time = float(state.get("start_time", 0) or 0)
    total_time_limit = float(state.get("total_time_limit", 0) or 0)
    if total_time_limit > 0 and start_time > 0:
        elapsed = _time.time() - start_time
        if elapsed >= total_time_limit:
            return f"Elapsed time {elapsed / 60:.1f} minutes, reached configured maximum time"
    return ""


def _route_after_update(state: AgentState) -> str:
    """Decide next step after FFG update.

    Check order:
    1. Whether maximum iterations reached -> End
    2. Whether configured maximum time reached -> End
    3. Whether overall testing completed -> End
    4. Otherwise -> Continue to next round
    """
    iteration_count = int(state.get("iteration_count", 0) or 0)
    _print_iteration_metrics(state, "executed")
    hard_limit_reason = _hard_limit_reason(state)
    if hard_limit_reason:
        print(f"\n[Routing] {hard_limit_reason}, terminating workflow.")
        return "end"

    if state.get("workflow_complete", False):
        print("\n[Routing] Overall exploration completed, workflow ended.")
        return "end"

    print(f"\n[Routing] Round {iteration_count} completed, continuing unified planning.")
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
        "[Round Metrics] "
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
    """Build a three-node cyclic workflow of scenario planning, scenario execution (including defect detection), and FFG update.

    Args:
        planning_llm: LLM client used for planning
        execution_llm: LLM client used for scenario execution and post-execution defect review
        device: Device driver
        ffg_update_llm: Dedicated LLM client for FFG update; falls back to planning_llm when None.
            FFG update single requests carry a large number of trace screenshots (mandatory forced images are not
            constrained by image_budget), response times are significantly longer than planning, requiring independent timeouts.
    """
    if ffg_update_llm is None:
        ffg_update_llm = planning_llm
    graph = StateGraph(AgentState)

    # Register nodes.
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
    # Define edges.

    graph.add_edge(START, "scenario_plan")

    # Unified planning node -> execution node, or end directly.
    graph.add_conditional_edges(
        "scenario_plan",
        _route_after_plan,
        {
            "scenario_exec": "scenario_exec",
            "continue": "scenario_plan",
            "end": END,
        },
    )

    # Defect detection is completed internally within the scenario execution node, followed by FFG update.
    graph.add_edge("scenario_exec", "ffg_update")

    # After FFG update -> routing decision.
    graph.add_conditional_edges(
        "ffg_update",
        _route_after_update,
        {
            "continue": "scenario_plan",            # Continue current phase
            "end": END,                              # End
        },
    )

    return graph.compile()
