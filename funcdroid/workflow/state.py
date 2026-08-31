"""Workflow shared state definition."""

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
    # Loop control
    iteration_count: int           # Current executed iterations
    max_iterations: int            # Maximum executed iterations
    iteration_started_at: float
    iteration_llm_requests_start: int
    iteration_llm_tokens_start: int
    # Time control (seconds)
    start_time: float              # Workflow start timestamp
    total_time_limit: float        # Total time limit (seconds), 0 means unlimited
    workflow_complete: bool        # Unified planner judges whether overall testing is complete
    long_run_idle_count: int       # Iterations of coverage rechecks completed while time budget remains
    # Run mode: "all" | "intra"
    run_phase: str
    # Exploration mode: "core" enables exploration boundary limits (intercept third-party service content by package name rules)
    exploration_mode: str
    # Termination audit only saves lightweight in-run state, not used for resumption; must not roll back Function completion state
    completion_audit_updates: int
    completion_audit_signatures: dict[str, str]
    completion_audit_attempts: dict[str, int]
    # Attempt count for dynamic granularity recheck (prevents idling recheck in every iteration)
    granularity_review_attempts: dict[str, int]
    ffg_granularity_review_function_id: str
    ffg_granularity_review_metrics: dict[str, Any]
    # Subgraph runtime metrics (do not participate in checkpoint semantics)
    ffg_update_step: int
    ffg_update_decision: str
    ffg_update_reason: str
    ffg_update_validation_issues: list[str]
    scenario_exec_subgraph_phase: str
    # Runtime continuous artifacts; do not enter FFG structure.
    _activity_timeline: dict[str, Any]
