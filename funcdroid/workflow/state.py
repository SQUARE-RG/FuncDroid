"""Shared workflow state definitions."""

from typing import Any, TypedDict

from model.flow_graph import FunctionalFlowGraph, Page


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
    # Compliance checklist state and per-rule conclusions.
    compliance_detection_enabled: bool
    compliance_targeted: bool
    # status/findings + attempts/evidence_signatures/discovery_baselines
    # Cooldown and evidence-driven rechecks for undetermined conclusions.
    compliance_checklist: dict[str, Any]
    compliance_rules_path: str           # Custom rule path; empty selects the default rules.json.
    compliance_review_failures: list[dict[str, Any]]
    ffg: FunctionalFlowGraph
    navigation_failure_count: int
    scenario_exec_error_count: int
    scenario_plan_error_count: int
    ffg_update_error_count: int
    screenshot_unavailable_count: int
    target_app_unavailable_count: int
    function_execution_failure_counts: dict[str, int]
    # Iteration control.
    iteration_count: int           # Completed iteration count.
    max_iterations: int            # Maximum iteration count.
    iteration_started_at: float
    iteration_llm_requests_start: int
    iteration_llm_tokens_start: int
    # Time control, in seconds.
    start_time: float              # Workflow start timestamp.
    total_time_limit: float        # Total time budget; zero means unlimited.
    workflow_complete: bool        # Planner decision that all scheduled testing is complete.
    long_run_idle_count: int       # Coverage-review rounds completed while time remains.
    # Audit (page_id, fingerprint) submissions in this idle period; failed retries do not prevent convergence.
    long_run_audit_fingerprints: list[tuple[str, str]]
    # Consecutive-failure stop reason; the hard limit also sets workflow_hard_stop.
    workflow_stop_reason: str
    workflow_hard_stop: bool
    # Planning scope: "all" or "intra".
    run_phase: str
    # "core" applies package-specific boundaries to exclude third-party services.
    exploration_mode: str
    # Lightweight in-run audit state, not resume state; never reset functionality completion.
    completion_audit_signatures: dict[str, str]
    completion_audit_attempts: dict[str, int]
    # Deferred functionalities already given a final retry.
    deferred_retry_ids: list[str]
    # Consecutive rounds with no model or fallback scenario before deferral.
    unplannable_counts: dict[str, int]
    # Granularity-review attempt counts to prevent repeated idle reviews.
    granularity_review_attempts: dict[str, int]
    ffg_granularity_review_function_id: str
    ffg_granularity_review_metrics: dict[str, Any]
    # FFG update diagnostics, excluded from resume semantics.
    ffg_update_decision: str
    ffg_update_reason: str
    ffg_update_validation_issues: list[str]
    # Persistent runtime artifacts outside the FFG structure.
    _activity_timeline: dict[str, Any]
