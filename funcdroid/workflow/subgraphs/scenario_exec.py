"""Independent execution subgraph for one planned scenario."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from workflow.bug_detection_worker import bug_detection_worker, build_defect_detection_job
from workflow.nodes.scenario_exec import create_scenario_execution_pass
from workflow.state import AgentState


class ScenarioExecSubgraphState(AgentState, total=False):
    scenario_execution_context: dict[str, Any] | None
    scenario_exec_subgraph_phase: str


def create_scenario_exec_subgraph(execution_llm, device):
    """Compile prepare -> action agent -> submit async defect review -> finalize.

    Defect detection no longer blocks this subgraph: it only reads the
    completed navigation/test trace (no device access), so it is handed off
    to a background worker (`workflow.bug_detection_worker`) while the outer
    workflow moves straight on to `ffg_update` / the next planned scenario,
    which do need the device.
    """
    execution_pass = create_scenario_execution_pass(execution_llm, device)

    def prepare(state: ScenarioExecSubgraphState) -> dict[str, Any]:
        review_status = (
            "async"
            if bool(state.get("bug_detection_enabled", True))
            else "off"
        )
        print(
            "[场景执行] 本轮准备 | "
            f"navigation={'yes' if state.get('current_navigation_scenario') else 'no'} | "
            f"scenario={'yes' if state.get('current_test_scenario') else 'no'} | "
            f"defect_review={review_status}"
        )
        return {
            "scenario_execution_context": None,
            "scenario_exec_subgraph_phase": "prepared",
        }

    def run_action_agent(state: ScenarioExecSubgraphState) -> dict[str, Any]:
        updates, context = execution_pass(state)
        return {
            **updates,
            "scenario_execution_context": context,
            "scenario_exec_subgraph_phase": "executed",
        }

    def submit_defect_review(state: ScenarioExecSubgraphState) -> dict[str, Any]:
        if not bool(state.get("bug_detection_enabled", True)):
            print("[缺陷检测] 已通过运行参数关闭，本轮不提交检测任务。")
            return {"scenario_exec_subgraph_phase": "review_disabled"}
        try:
            job_state = build_defect_detection_job(state)
            accepted = bug_detection_worker.submit(job_state)
        except Exception as exc:
            print(f"[缺陷检测] 提交异步检测任务失败，本轮缺陷复核已跳过: {exc}")
            accepted = False
        return {
            "scenario_exec_subgraph_phase": (
                "review_submitted" if accepted else "review_skipped_backpressure"
            )
        }

    def finalize(state: ScenarioExecSubgraphState) -> dict[str, Any]:
        return {"scenario_exec_subgraph_phase": "finished"}

    graph = StateGraph(ScenarioExecSubgraphState)
    graph.add_node("prepare_execution", prepare)
    graph.add_node("action_agent", run_action_agent)
    graph.add_node("defect_review", submit_defect_review)
    graph.add_node("finalize_execution", finalize)
    graph.add_edge(START, "prepare_execution")
    graph.add_edge("prepare_execution", "action_agent")
    graph.add_edge("action_agent", "defect_review")
    graph.add_edge("defect_review", "finalize_execution")
    graph.add_edge("finalize_execution", END)
    return graph.compile()


def create_scenario_exec_subgraph_node(execution_llm, device):
    """Expose the compiled subgraph through the outer workflow node contract."""
    subgraph = create_scenario_exec_subgraph(execution_llm, device)

    def invoke(state: AgentState) -> dict[str, Any]:
        result = subgraph.invoke(dict(state))
        return {
            key: value
            for key, value in result.items()
            if key not in {"scenario_execution_context"}
        }

    invoke.subgraph = subgraph  # type: ignore[attr-defined]
    return invoke
