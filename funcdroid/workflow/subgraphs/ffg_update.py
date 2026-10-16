"""FFG update subgraph: one model pass, best-effort apply, no rollback."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from model.flow_graph import FlowGraph
from workflow.nodes.ffg_update import create_ffg_update_node
from workflow.state import AgentState


class FFGUpdateSubgraphState(AgentState, total=False):
    pass


def create_ffg_update_subgraph(llm, device=None):
    """Compile the FFG update as a single-pass subgraph.

    旧设计在副本上更新、校验失败时 refinement 重试并最终整轮回滚，导致本轮
    已执行的页面、操作与跳转经常整体丢失。现在一轮只调用一次模型：patch
    尽力应用，非法条目跳过，模型遗漏由程序兜底，结构不一致确定性修复。
    """
    update_pass = create_ffg_update_node(llm, device)

    def agent_update(state: FFGUpdateSubgraphState) -> dict[str, Any]:
        if not isinstance(state.get("ffg"), FlowGraph):
            state = {**state, "ffg": FlowGraph()}
        result = update_pass(state)
        issues = list(result.get("ffg_update_validation_issues", []) or [])
        print(
            "[FFG 更新] 更新完成: "
            f"diagnostics={len(issues)}"
            + (f" | {'; '.join(map(str, issues[:3]))}" if issues else "")
        )
        return result

    graph = StateGraph(FFGUpdateSubgraphState)
    graph.add_node("agent_update", agent_update)
    graph.add_edge(START, "agent_update")
    graph.add_edge("agent_update", END)
    return graph.compile()


def create_ffg_update_subgraph_node(llm, device=None):
    """Return a normal callable while retaining a separately testable subgraph."""
    subgraph = create_ffg_update_subgraph(llm, device)

    def invoke(state: AgentState) -> dict[str, Any]:
        return subgraph.invoke(dict(state))

    invoke.subgraph = subgraph  # type: ignore[attr-defined]
    return invoke
