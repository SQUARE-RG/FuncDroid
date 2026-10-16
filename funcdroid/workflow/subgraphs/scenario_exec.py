"""Independent execution subgraph for one planned scenario."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from workflow.bug_detection_worker import bug_detection_worker, build_defect_detection_job
from workflow.compliance_worker import (
    build_compliance_detection_job,
    compliance_worker,
    plan_check_ids,
)
from workflow.nodes.scenario_exec import create_scenario_execution_pass
from workflow.state import AgentState


def create_scenario_exec_subgraph(execution_llm, device):
    """Compile action agent -> submit async reviews.

    判定不阻塞本子图：只读取已完成的 navigation/test trace（不触碰设备），
    交给后台线程，外层工作流直接进入 `ffg_update`。每次运行只启用所选模式
    的判定器；缺陷模式提交缺陷轨迹，合规模式提交条款取证（普通探索场景
    没有关联检测点，跳过合规判定）。
    """
    execution_pass = create_scenario_execution_pass(execution_llm, device)

    def submit_reviews(state: AgentState) -> dict[str, Any]:
        if bool(state.get("compliance_detection_enabled", False)):
            check_ids = plan_check_ids(state.get("current_plan"))
            if not check_ids:
                print("[合规检测] 本轮为普通探索场景，无关联合规检测点，跳过判定。")
            else:
                accepted = False
                try:
                    accepted = compliance_worker.submit(build_compliance_detection_job(state))
                except Exception as exc:
                    print(f"[合规检测] 提交异步判定任务失败，本轮判定已跳过: {exc}")
                if not accepted:
                    compliance_worker.defer_checks(check_ids, "执行轨迹未成功进入合规判定队列，等待补测")

        if bool(state.get("bug_detection_enabled", True)):
            try:
                bug_detection_worker.submit(build_defect_detection_job(state))
            except Exception as exc:
                bug_detection_worker.record_failure(state, exc)
                print(f"[缺陷检测] 提交异步检测任务失败，本轮缺陷复核已跳过: {exc}")
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
