"""Independent execution subgraph for one planned scenario."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from workflow.bug_detection_worker import bug_detection_worker, build_defect_detection_job
from workflow.nodes.scenario_exec import create_scenario_execution_pass
from workflow.state import AgentState


def create_scenario_exec_subgraph(execution_llm, device):
    """Compile the action agent followed by asynchronous review submission.

    Review completed navigation/test traces without touching the device. Submit only functional bug detection."""
    execution_pass = create_scenario_execution_pass(execution_llm, device)

    def submit_reviews(state: AgentState) -> dict[str, Any]:

        if bool(state.get("bug_detection_enabled", True)):
            try:
                bug_detection_worker.submit(build_defect_detection_job(state))
            except Exception as exc:
                bug_detection_worker.record_failure(state, exc)
                print(f"[Bug detection] Background submission failed; skipping this review: {exc}")
        return {}

    graph = StateGraph(AgentState)
    graph.add_node("action_agent", execution_pass)
    graph.add_node("async_reviews", submit_reviews)
    graph.add_edge(START, "action_agent")
    graph.add_edge("action_agent", "async_reviews")
    graph.add_edge("async_reviews", END)
    return graph.compile()


def create_scenario_exec_subgraph_node(execution_llm, device):
    """Expose the compiled subgraph through the outer workflow node contract."""
    subgraph = create_scenario_exec_subgraph(execution_llm, device)

    def invoke(state: AgentState) -> dict[str, Any]:
        return subgraph.invoke(dict(state))

    invoke.subgraph = subgraph  # type: ignore[attr-defined]
    return invoke
