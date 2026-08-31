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
            "[FFG 更新] 更新尝试: "
            f"{int(state.get('ffg_update_step', 1) or 1)}/"
            f"{int(state.get('ffg_update_max_steps', FFG_UPDATE_MAX_STEPS) or FFG_UPDATE_MAX_STEPS)}"
        )
        return update_pass(state)

    def route_after_step(state: FFGUpdateSubgraphState) -> str:
        step = int(state.get("ffg_update_step", 1) or 1)
        max_steps = int(state.get("ffg_update_max_steps", FFG_UPDATE_MAX_STEPS) or FFG_UPDATE_MAX_STEPS)
        issues = list(state.get("ffg_update_validation_issues", []) or [])
        prev_issues = list(state.get("ffg_update_prev_issues", []) or [])
        # 只有结构校验真的报出问题才继续 refinement。模型自报的 "continue"
        # 不足以触发新一轮多模态调用：它在结构已合法时也会持续返回 continue，
        # 是重复 refinement 与图片开销的主要来源。
        wants_more = bool(issues)
        # 同一组校验错误连续两轮原样出现，说明模型没有能力修复该类问题；
        # 继续重试只会重复消耗完整截图 payload，提前进入 finalize。
        if wants_more and prev_issues and issues == prev_issues:
            print(
                "[FFG 更新] 校验错误与上一轮完全相同，提前停止重试并进入收尾: "
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
                "[FFG 更新] 结构校验未通过，回滚本轮更新: "
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
            # 页面确实存在且图自洽，只是复用对象落在检索候选集之外。保留本轮观测，
            # 否则会为一条可能归属不准的对齐丢掉整轮的新页面、操作与跳转。
            print(
                "[FFG 更新] 图结构自洽，保留更新并记录对齐告警: "
                + "; ".join(advisory_issues[:8])
            )
        print(
            "[FFG 更新] 更新完成: "
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
