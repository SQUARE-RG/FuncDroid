"""Assemble the LangGraph workflow.

    scenario_plan -> scenario_exec (asynchronous review) -> ffg_update -> repeat
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
# Bound exponential backoff after consecutive node failures, in seconds.
# Without backoff, disconnected devices can cause a busy loop until timeout.
# Retry the first failure immediately, then wait 2, 4, 8... seconds up to this cap.
NODE_ERROR_BACKOFF_MAX_SECONDS = max(
    0.0, float(os.getenv("NODE_ERROR_BACKOFF_MAX_SECONDS", "60"))
)
# Stop unconditionally at this hard limit, even if the optional early-stop
# policy is disabled, so unrecoverable infrastructure cannot loop indefinitely.
HARD_STOP_CONSECUTIVE_NODE_ERRORS = max(
    MAX_CONSECUTIVE_NODE_ERRORS,
    int(os.getenv(
        "HARD_STOP_CONSECUTIVE_NODE_ERRORS",
        str(MAX_CONSECUTIVE_NODE_ERRORS * 10),
    )),
)
# Log metrics only every N skipped rounds, always including the first skip.
# Computing them scans all functionalities and adds no information on idle rounds.
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
            print(f"[Node error] {node_name} failed: {exc}")
            print(traceback.format_exc())
            try:
                return fallback(state, exc)
            except Exception as fallback_exc:
                print(f"[Node error] {node_name} fallback also failed: {fallback_exc}")
                print(traceback.format_exc())
                return {}

    return wrapped


def _skip_plan() -> dict:
    """Build an empty plan for a round with no executable scenario."""
    return {
        "scenario_type": "skip",
        "target_function_id": "",
        "src_function_id": "",
        "dst_function_id": "",
    }


def _threshold_reached(node_name: str, error_count: int, limit: int) -> tuple[bool, bool, str]:
    """Return (stop, hard_stop, reason) for consecutive node failures.

    The optional policy stops at its threshold. The hard limit always applies.
    """
    if error_count >= HARD_STOP_CONSECUTIVE_NODE_ERRORS:
        hard = True
        reason = (
            f"{node_name} reached the consecutive failure hard limit "
            f"{error_count}/{HARD_STOP_CONSECUTIVE_NODE_ERRORS}; treating this as an unrecoverable failure"
        )
    elif STOP_AFTER_CONSECUTIVE_NODE_ERRORS and error_count >= limit:
        hard = False
        reason = f"{node_name} reached the consecutive failure threshold {error_count}/{limit}"
    else:
        return False, False, ""
    print(f"[Node error] {reason}; stopping the workflow early.")
    return True, hard, reason


def _node_error_backoff(node_name: str, error_count: int) -> None:
    """Back off after repeated node failures to avoid a busy retry loop."""
    if error_count <= 1 or NODE_ERROR_BACKOFF_MAX_SECONDS <= 0:
        return
    delay = min(NODE_ERROR_BACKOFF_MAX_SECONDS, float(2 ** min(error_count - 1, 16)))
    print(f"[Node error] {node_name} failed consecutively {error_count} times; retry after {delay:g}s.")
    _time.sleep(delay)


def _error_stop_updates(node_name: str, error_count: int, limit: int) -> dict:
    """Shared fallback policy: stop at the threshold, otherwise back off and continue."""
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
    error_count = int(state.get("scenario_exec_error_count", 0) or 0) + 1
    # Executor infrastructure errors are not functionality attempts. Clear this
    # scenario and flag it so FFG updates do not count an unexecuted attempt.
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
            "scenario_error": f"Scenario execution node failed: {exc}",
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
    # FFG updates mutate the shared graph; retain audits and observations applied
    # before failure, then record execution facts so attempts are not lost.
    updates: dict = {}
    try:
        updates["function_execution_failure_counts"] = _record_execution_facts(
            graph, state, state.get("execution_result", {}) or {},
        )
    except Exception as facts_exc:  # noqa: BLE001
        print(f"[FFG update] Failed to record execution facts: {facts_exc}")
    print(f"[FFG update] Retain applied graph changes after update failure: {exc}")
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
    """When planning is idle, only page audits can expose new coverage gaps.

    Merge completed audits and submit unaudited pages here because skipped
    rounds bypass FFG updating. Stop when neither changes nor jobs remain.
    Track (page_id, fingerprint) submissions per idle period: repeatedly
    resubmitting a failed audit must not keep the workflow alive indefinitely.
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
    print("[Exploration] Coverage review and page audits found no new evidence; stop instead of waiting out the budget.")
    return True


def _finish_without_round(
    iteration_count: int,
    iteration_started_at: float,
    usage_totals: dict,
) -> dict:
    """Finalize at the iteration or time limit without planning or incrementing the counter."""
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
    """Keep iteration/finalization control inside planning so the workflow has three nodes."""

    def scenario_plan_with_iteration(state: AgentState) -> dict:
        iteration_started_at = _time.time()
        usage_totals = get_llm_usage_totals()
        previous_count = int(state.get("iteration_count", 0) or 0)
        max_iterations = int(state.get("max_iterations", DEFAULT_MAX_ITERATIONS) or 0)
        if max_iterations > 0 and previous_count >= max_iterations:
            print(
                f"[Iteration] Maximum iteration count reached: {max_iterations},"
                "Do not invoke planning again after resume or error reentry."
            )
            return _finish_without_round(previous_count, iteration_started_at, usage_totals)
        time_limit_reason = _time_limit_reason(state)
        if time_limit_reason:
            print(f"[Iteration] {time_limit_reason}; do not start another plan or scenario.")
            # No planning or execution occurred, so do not increment the iteration count.
            return _finish_without_round(previous_count, iteration_started_at, usage_totals)
        count = previous_count + 1
        run_phase = state.get("run_phase", "all")
        run_phase_label = {
            "all": "Unified planning",
            "intra": "Intra-functionality testing only",
        }.get(run_phase, "Unified planning")
        print(f"\n{'='*60}")
        print(f"  Iteration {count} - scope: {run_phase_label} | exploration: core functionalities")
        print(f"{'='*60}\n")
        navigation_failure_count = int(state.get("navigation_failure_count", 0) or 0)
        if navigation_failure_count:
            print(f"[Iteration] Consecutive navigation failures: {navigation_failure_count}")
        planning_state = dict(state)
        planning_state["iteration_count"] = count
        # Planning handles status changes once; skipped rounds proceed to the next iteration.
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
                and not _time_limit_reason(projected_state)
                and not _long_run_converged(
                    projected_state.get("ffg"),
                    idle_audit_fingerprints,
                )
            ):
                idle_count = int(state.get("long_run_idle_count", 0) or 0) + 1
                print(
                    "[Exploration] No new FFG gaps, but page audits have results or pending jobs; "
                    "continue coverage review after those audits."
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
                    # Resume planning as soon as an audit completes, without waiting the full interval.
                    page_audit_worker.flush(timeout=min(LONG_RUN_IDLE_SECONDS, remaining))
                planning_updates = dict(planning_updates)
                # Keep completion_audit_signatures so the coverage-gap audit
                # stays incremental across idle rounds. Clearing them here forced
                # every completed function to be re-audited from scratch each
                # idle cycle, repeatedly re-auditing already completed functionalities.
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
        }:
            # Executed scenarios can change the graph; reset fingerprints for the next idle period.
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


# Consecutive skip count only throttles skipped-round metric logging.
_CONSECUTIVE_SKIP_ROUNDS = 0


def _route_after_plan(state: AgentState) -> str:
    """Decide whether a planned scenario requires device actions."""
    global _CONSECUTIVE_SKIP_ROUNDS
    if state.get("workflow_complete", False):
        _print_iteration_metrics(state, "plan_end")
        stop_reason = str(state.get("workflow_stop_reason", "") or "")
        if stop_reason:
            print(f"\n[Route] {stop_reason}; finish the workflow.")
        else:
            print("\n[Route] Exploration is complete; skip execution.")
        return "end"
    scenario_type = (state.get("current_plan", {}) or {}).get("scenario_type")
    time_limit_reason = _time_limit_reason(state)
    if time_limit_reason:
        _print_iteration_metrics(state, "plan_hard_limit")
        print(f"\n[Route] {time_limit_reason}; do not start another scenario.")
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
        print(f"\n[Route] Maximum iterations reached: {max_iterations}; stop idle planning rounds.")
        return "end"
    if scenario_type == "skip":
        _CONSECUTIVE_SKIP_ROUNDS += 1
        if (
            _CONSECUTIVE_SKIP_ROUNDS == 1
            or _CONSECUTIVE_SKIP_ROUNDS % SKIP_ROUND_METRICS_EVERY == 0
        ):
            _print_iteration_metrics(state, "skip")
        print("\n[Route] No executable scenario this round; continue planning.")
        return "continue"
    _CONSECUTIVE_SKIP_ROUNDS = 0
    return "scenario_exec"


def _hard_limit_reason(state: AgentState) -> str:
    """Return the reason for a hard workflow stop, if any."""
    iteration_count = int(state.get("iteration_count", 0) or 0)
    max_iterations = int(state.get("max_iterations", DEFAULT_MAX_ITERATIONS) or 0)
    if max_iterations > 0 and iteration_count >= max_iterations:
        return f"Maximum iterations reached: {max_iterations}"

    return _time_limit_reason(state)


def _time_limit_reason(state: AgentState) -> str:
    start_time = float(state.get("start_time", 0) or 0)
    total_time_limit = float(state.get("total_time_limit", 0) or 0)
    if total_time_limit > 0 and start_time > 0:
        elapsed = _time.time() - start_time
        if elapsed >= total_time_limit:
            return f"Elapsed time: {elapsed / 60:.1f} minutes; the time budget is exhausted"
    return ""


def _route_after_update(state: AgentState) -> str:
    """Route after updating the FFG.

    Stop at the iteration limit, time limit, or overall completion; otherwise
    continue to the next planning round.
    """
    iteration_count = int(state.get("iteration_count", 0) or 0)
    _print_iteration_metrics(state, "executed")
    hard_limit_reason = _hard_limit_reason(state)
    if hard_limit_reason:
        print(f"\n[Route] {hard_limit_reason}; finish the workflow.")
        return "end"

    if state.get("workflow_complete", False):
        stop_reason = str(state.get("workflow_stop_reason", "") or "")
        if stop_reason:
            print(f"\n[Route] {stop_reason}; finish the workflow.")
        else:
            print("\n[Route] Exploration is complete; finish the workflow.")
        return "end"

    print(f"\n[Route] Iteration {iteration_count} completed; continue unified planning.")
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
        # Aggregate counts in one pass using node fields instead of repeated lookups.
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
        "[Iteration metrics] "
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
    """Build the planning, scenario-driven execution/review, and FFG update loop.

    Args:
        planning_llm: Model client for scenario planning.
        execution_llm: Model client for GUI execution and subsequent bug review.
        device: Device driver.
        ffg_update_llm: Dedicated update client, or planning_llm when omitted.
            Mandatory trace images are not capped by the optional image budget;
            FFG updates may need a longer request timeout than planning.
    """
    if ffg_update_llm is None:
        ffg_update_llm = planning_llm
    graph = StateGraph(AgentState)

    # Register workflow nodes.
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
    # Define workflow edges.

    graph.add_edge(START, "scenario_plan")

    # Planning routes to execution, another planning round, or termination.
    graph.add_conditional_edges(
        "scenario_plan",
        _route_after_plan,
        {
            "scenario_exec": "scenario_exec",
            "continue": "scenario_plan",
            "end": END,
        },
    )

    # Execution has submitted asynchronous review; update the FFG next.
    graph.add_edge("scenario_exec", "ffg_update")

    # Route after the FFG update.
    graph.add_conditional_edges(
        "ffg_update",
        _route_after_update,
        {
            "continue": "scenario_plan",            # Continue planning.
            "end": END,                              # Finish.
        },
    )

    return graph.compile()
