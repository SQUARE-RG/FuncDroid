"""Bounded agent-style subgraph for incremental FFG refinement."""

from __future__ import annotations

import copy
import os
from typing import Any

from langgraph.graph import END, START, StateGraph

from model.flow_graph import FlowGraph
from workflow.nodes.ffg_update import (
    create_ffg_update_node,
    partition_alignment_issues,
    validate_ffg_invariants,
)
from workflow.state import AgentState


FFG_UPDATE_MAX_STEPS = max(1, int(os.getenv("FFG_UPDATE_MAX_STEPS", "4")))


class FFGUpdateSubgraphState(AgentState, total=False):
    ffg_update_step: int
    ffg_update_max_steps: int
    ffg_update_decision: str
    ffg_update_reason: str
    ffg_update_validation_issues: list[str]
    ffg_update_prev_issues: list[str]
    ffg_update_base_ffg: FlowGraph
    ffg_update_base_target: str
    ffg_update_base_status: str
    ffg_update_base_current_page: Any


def create_ffg_update_subgraph(llm, device=None):
    """Compile an update subgraph whose LLM decides whether refinement continues."""
    update_pass = create_ffg_update_node(llm, device)

    def prepare(state: FFGUpdateSubgraphState) -> dict[str, Any]:
        graph = state.get("ffg")
        if not isinstance(graph, FlowGraph):
            graph = FlowGraph()
        return {
            "ffg": copy.deepcopy(graph),
            "ffg_update_base_ffg": copy.deepcopy(graph),
            "ffg_update_base_target": str(state.get("current_testing_function_id", "") or ""),
            "ffg_update_base_status": str(state.get("current_testing_function_status", "not_started") or "not_started"),
            "ffg_update_base_current_page": copy.deepcopy(state.get("current_page")),
            "ffg_update_step": 1,
            "ffg_update_max_steps": FFG_UPDATE_MAX_STEPS,
            "ffg_update_decision": "continue",
            "ffg_update_reason": "",
            "ffg_update_validation_issues": [],
        }

    def run_agent_step(state: FFGUpdateSubgraphState) -> dict[str, Any]:
        print(
            "[FFG Update] Update attempt: "
            f"{int(state.get('ffg_update_step', 1) or 1)}/"
            f"{int(state.get('ffg_update_max_steps', FFG_UPDATE_MAX_STEPS) or FFG_UPDATE_MAX_STEPS)}"
        )
        return update_pass(state)

    def route_after_step(state: FFGUpdateSubgraphState) -> str:
        step = int(state.get("ffg_update_step", 1) or 1)
        max_steps = int(state.get("ffg_update_max_steps", FFG_UPDATE_MAX_STEPS) or FFG_UPDATE_MAX_STEPS)
        issues = list(state.get("ffg_update_validation_issues", []) or [])
        prev_issues = list(state.get("ffg_update_prev_issues", []) or [])
        # Only continue refinement if structural validation actually reports issues. The model's self-reported "continue"
        # is not sufficient to trigger a new round of multimodal calls: it keeps returning continue even when the structure is already valid,
        # which is the main source of redundant refinement and image overhead.
        wants_more = bool(issues)
        # If the same set of validation errors appears identically across two consecutive rounds, the model cannot fix this type of issue;
        # continuing retries will only repeatedly consume full screenshot payloads, so enter finalize early.
        if wants_more and prev_issues and issues == prev_issues:
            print(
                "[FFG Update] Validation errors are identical to the previous round; stopping retries early and entering finalization: "
                + "; ".join(issues[:3])
            )
            return "finish"
        if wants_more and step < max_steps:
            return "continue"
        return "finish"

    def advance(state: FFGUpdateSubgraphState) -> dict[str, Any]:
        return {
            "ffg_update_step": int(state.get("ffg_update_step", 1) or 1) + 1,
            "ffg_update_prev_issues": list(
                state.get("ffg_update_validation_issues", []) or []
            ),
        }

    def finalize(state: FFGUpdateSubgraphState) -> dict[str, Any]:
        graph = state.get("ffg")
        blocking_issues, advisory_issues = partition_alignment_issues(
            list(state.get("ffg_update_validation_issues", []) or [])
        )
        blocking_issues.extend(
            validate_ffg_invariants(graph)
            if isinstance(graph, FlowGraph)
            else ["FFG missing"]
        )
        blocking_issues = list(
            dict.fromkeys(str(issue) for issue in blocking_issues if str(issue))
        )
        if blocking_issues:
            print(
                "[FFG Update] Structural validation failed; rolling back this update round: "
                + "; ".join(blocking_issues[:8])
            )
            return {
                "ffg": copy.deepcopy(state.get("ffg_update_base_ffg", FlowGraph())),
                "current_testing_function_id": str(state.get("ffg_update_base_target", "") or ""),
                "current_testing_function_status": str(state.get("ffg_update_base_status", "not_started") or "not_started"),
                "current_page": copy.deepcopy(state.get("ffg_update_base_current_page")),
                "ffg_update_validation_issues": blocking_issues,
                "ffg_update_decision": "finish",
                "ffg_update_reason": "invalid graph rolled back",
            }
        if advisory_issues:
            # The Page indeed exists and the graph is self-consistent, but the reused object falls outside the retrieval candidate set. Retain this round's observation,
            # otherwise an entire round of new Pages, operations, and transitions would be lost due to a potentially inaccurate alignment.
            print(
                "[FFG Update] Graph structure is self-consistent; keeping update and recording alignment warning: "
                + "; ".join(advisory_issues[:8])
            )
        print(
            "[FFG Update] Update completed: "
            f"attempts={int(state.get('ffg_update_step', 1) or 1)}, "
            f"decision={state.get('ffg_update_decision', 'finish')}"
        )
        return {"ffg_update_validation_issues": []}

    graph = StateGraph(FFGUpdateSubgraphState)
    graph.add_node("prepare_context", prepare)
    graph.add_node("agent_update", run_agent_step)
    graph.add_node("advance_refinement", advance)
    graph.add_node("finalize_update", finalize)
    graph.add_edge(START, "prepare_context")
    graph.add_edge("prepare_context", "agent_update")
    graph.add_conditional_edges(
        "agent_update",
        route_after_step,
        {"continue": "advance_refinement", "finish": "finalize_update"},
    )
    graph.add_edge("advance_refinement", "agent_update")
    graph.add_edge("finalize_update", END)
    return graph.compile()


def create_ffg_update_subgraph_node(llm, device=None):
    """Return a normal callable while retaining a separately testable subgraph."""
    subgraph = create_ffg_update_subgraph(llm, device)

    def invoke(state: AgentState) -> dict[str, Any]:
        result = subgraph.invoke(dict(state))
        internal_keys = {
            "ffg_update_base_ffg",
            "ffg_update_base_target",
            "ffg_update_base_status",
            "ffg_update_base_current_page",
            "ffg_update_max_steps",
            "ffg_update_prev_issues",
        }
        return {key: value for key, value in result.items() if key not in internal_keys}

    invoke.subgraph = subgraph  # type: ignore[attr-defined]
    return invoke
