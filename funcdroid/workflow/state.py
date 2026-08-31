"""工作流共享状态定义。"""

from typing import Any, TypedDict

from model.flow_graph import FlowGraph, Page


class AgentState(TypedDict, total=False):
    app_name: str
    package_name: str
    current_page: Page | None
    current_navigation_scenario: str
    current_test_scenario: str
    current_plan: dict[str, Any]
    current_testing_function_id: str
    current_testing_function_status: str
    navigation_execution_trace: list[dict[str, Any]]
    scenario_execution_trace: list[dict[str, Any]]
    execution_result: dict[str, Any]
    detected_bugs: list[dict[str, Any]]
    bug_trace_records: dict[str, list[dict[str, Any]]]
    bug_detection_enabled: bool
    ffg: FlowGraph
    navigation_failure_count: int
    scenario_exec_error_count: int
    scenario_plan_error_count: int
    ffg_update_error_count: int
    screenshot_unavailable_count: int
    target_app_unavailable_count: int
    function_execution_failure_counts: dict[str, int]
    cross_execution_failure_counts: dict[str, int]
    planner_noop_count: int
    planner_feedback: str
    planner_rejected_scenario: str
    # 循环控制
    iteration_count: int           # 当前已执行轮次
    max_iterations: int            # 最大执行轮次
    iteration_started_at: float
    iteration_llm_requests_start: int
    iteration_llm_tokens_start: int
    # 时间控制（秒）
    start_time: float              # 工作流开始时间戳
    total_time_limit: float        # 总时间限制（秒），0 表示不限制
    workflow_complete: bool        # 统一规划器判断整体测试是否完成
    long_run_idle_count: int       # 尚有时间预算时完成覆盖复核的轮次
    # 运行模式: "all" | "intra"
    run_phase: str
    # 探索模式: "core" 启用探索边界限制（按包名规则拦截第三方服务内容）
    exploration_mode: str
    # 终止审计只保存轻量运行内状态，不用于断点续跑；不得回退 Function 完成状态
    completion_audit_updates: int
    completion_audit_signatures: dict[str, str]
    completion_audit_attempts: dict[str, int]
    # 动态粒度复核的尝试计数（防止每轮空转复核）
    granularity_review_attempts: dict[str, int]
    ffg_granularity_review_function_id: str
    ffg_granularity_review_metrics: dict[str, Any]
    # 子图运行指标（不参与断点语义）
    ffg_update_step: int
    ffg_update_decision: str
    ffg_update_reason: str
    ffg_update_validation_issues: list[str]
    scenario_exec_subgraph_phase: str
    # 运行时持续产物；不进入 FFG 结构。
    _activity_timeline: dict[str, Any]
