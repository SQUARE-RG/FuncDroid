"""工作流共享状态定义。"""

from typing import Any, TypedDict

from model.flow_graph import FlowGraph, Page


class AgentState(TypedDict, total=False):
    run_mode: str  # defect | compliance
    reset_app_data: bool
    grant_permissions: bool
    permission_grant_report: dict[str, Any] | None
    seed_provenance: dict[str, Any]
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
    defect_review_failures: list[dict[str, Any]]
    # 合规检测：检测点清单状态 + 条款结论。
    compliance_detection_enabled: bool
    compliance_targeted: bool
    # status/findings + attempts/evidence_signatures/discovery_baselines
    # （undetermined 相关证据发现冷却与闭环复判）
    compliance_checklist: dict[str, Any]
    compliance_rules_path: str           # 自定义规则库路径，空串用默认 rules.json
    compliance_review_failures: list[dict[str, Any]]
    ffg: FlowGraph
    navigation_failure_count: int
    scenario_exec_error_count: int
    scenario_plan_error_count: int
    ffg_update_error_count: int
    screenshot_unavailable_count: int
    target_app_unavailable_count: int
    function_execution_failure_counts: dict[str, int]
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
    # 本次空闲期内已提交过页面审计的 (page_id, 指纹)；失败重提不算“未收敛”
    long_run_audit_fingerprints: list[tuple[str, str]]
    # 节点连续异常导致的提前结束：原因文本；硬上限触发时 workflow_hard_stop=True
    workflow_stop_reason: str
    workflow_hard_stop: bool
    # 运行模式: "all" | "intra"
    run_phase: str
    # 探索模式: "core" 启用探索边界限制（按包名规则拦截第三方服务内容）
    exploration_mode: str
    # 终止审计只保存轻量运行内状态，不用于断点续跑；不得回退 Function 完成状态
    completion_audit_signatures: dict[str, str]
    completion_audit_attempts: dict[str, int]
    # 收尾阶段已给过一次补测机会的 deferred 功能
    deferred_retry_ids: list[str]
    # 功能连续“规划器与模板均无场景”的轮数；达到阈值才延后。
    unplannable_counts: dict[str, int]
    # 动态粒度复核的尝试计数（防止每轮空转复核）
    granularity_review_attempts: dict[str, int]
    ffg_granularity_review_function_id: str
    ffg_granularity_review_metrics: dict[str, Any]
    # FFG 更新诊断（不参与断点语义）
    ffg_update_decision: str
    ffg_update_reason: str
    ffg_update_validation_issues: list[str]
    # 运行时持续产物；不进入 FFG 结构。
    _activity_timeline: dict[str, Any]
