# Multilingual UI and model-response fixtures use Unicode escapes to preserve their exact text.
import time

import pytest

from model.flow_graph import (
    FlowGraph,
    FunctionDependency,
    FunctionNode,
    Page,
    PageTransition,
)
from workflow.nodes import scenario_plan as scenario_plan_module
from workflow.graph import _fallback_scenario_exec, _route_after_plan
from workflow.nodes.ffg_update import _resolve_plan_target_function_id
from workflow.nodes.scenario_plan import (
    _audit_function_for_gap,
    _build_audit_navigation_scenario,
    _forced_intra_scenario,
    _candidate_cross_pairs,
    _pending_cross_pairs,
    _release_unplannable_function,
    _navigation_to_function,
    _run_completion_audit,
    _validate_audit_result,
    _validate_plan_result,
    create_scenario_plan_node,
    _bootstrap_initial_ffg,
)
from workflow.nodes.prompt import (
    BOOTSTRAP_FFG_SYSTEM_PROMPT,
    FFG_AUDIT_SYSTEM_PROMPT,
    FFG_UPDATE_SYSTEM_PROMPT,
    SCENARIO_EXEC_SYSTEM_PROMPT_TEMPLATE,
    UNIFIED_SCENARIO_PLAN_SYSTEM_PROMPT,
)


@pytest.fixture(autouse=True)
def _legacy_text_matching_enabled(monkeypatch):
    """Explicitly enable legacy text fallback to preserve these pre-binding tests' intended coverage semantics."""
    from model import flow_graph as _flow_graph

    monkeypatch.setattr(_flow_graph, "COVERAGE_LEGACY_TEXT_MATCHING", True)


def _graph_with_two_functions() -> FlowGraph:
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="\u6e90\u529f\u80fd"))
    graph.add_node(FunctionNode(id="func_2", name="\u76ee\u6807\u529f\u80fd"))
    return graph


def test_navigation_reaches_function_page_instead_of_stopping_at_entry_page():
    graph = _graph_with_two_functions()
    root = Page(id="page_1", name="\u5fae\u4fe1\u9996\u9875")
    detail = Page(id="page_2", name="\u6587\u4ef6\u4f20\u8f93\u52a9\u624b\u8be6\u60c5\u9875")
    graph.add_page(root, "func_1")
    graph.add_entry_page(root, "func_2")
    graph.add_page(detail, "func_2")
    graph.add_page_transition(PageTransition("page_1", "page_2", "\u70b9\u51fb\u6587\u4ef6\u4f20\u8f93\u52a9\u624b"))

    scenario, _reason, level = _navigation_to_function(
        graph,
        root,
        graph.get_node("func_2"),
    )

    assert level == "A"
    assert "\u70b9\u51fb\u6587\u4ef6\u4f20\u8f93\u52a9\u624b" in scenario
    assert "Confirm arrival at \"\u6587\u4ef6\u4f20\u8f93\u52a9\u624b\u8be6\u60c5\u9875\"" in scenario


def test_completion_audit_batches_cover_every_page_image(monkeypatch):
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="\u591a\u9875\u9762\u529f\u80fd"))
    for index in range(5):
        graph.add_page(Page(
            id=f"page_{index + 1}",
            name=f"\u9875\u9762{index + 1}",
            representative_screenshot=f"image-{index + 1}",
        ), "func_1")

    class Response:
        content = (
            '{"entry_page_ref":"","entry_name":""}'
        )

    class Llm:
        def __init__(self):
            self.image_counts = []

        def invoke(self, messages):
            content = messages[1].content
            self.image_counts.append(sum(
                1 for block in content if block.get("type") == "image_url"
            ))
            return Response()

    monkeypatch.setattr(scenario_plan_module, "MAX_AUDIT_IMAGES_PER_FUNCTION", 2)
    llm = Llm()

    result = _audit_function_for_gap(
        llm,
        graph,
        graph.get_node("func_1"),
        "core",
    )

    assert result == {"entry_page_ref": "", "entry_name": ""}
    assert llm.image_counts == [2, 2, 1]


def test_time_limit_blocks_new_normal_scenario_execution():
    state = {
        "workflow_complete": False,
        "iteration_count": 1,
        "max_iterations": 0,
        "start_time": time.time() - 10,
        "total_time_limit": 1,
        "current_plan": {"scenario_type": "intra_test"},
    }

    assert _route_after_plan(state) == "end"


def test_planner_self_loop_stops_at_max_iterations():
    state = {
        "iteration_count": 3,
        "max_iterations": 3,
        "workflow_complete": False,
        "current_plan": {"scenario_type": "skip"},
    }

    assert _route_after_plan(state) == "end"


def test_executor_prompt_explains_scroll_direction_semantics():
    assert "up reveals lower/later content" in SCENARIO_EXEC_SYSTEM_PROMPT_TEMPLATE
    assert "down reveals upper/earlier content" in SCENARIO_EXEC_SYSTEM_PROMPT_TEMPLATE


def test_prompts_do_not_claim_unsupported_physical_inputs():
    assert "cannot supply human speech" in UNIFIED_SCENARIO_PLAN_SYSTEM_PROMPT
    assert "do not plan solving, dragging, submitting or successful verification" in UNIFIED_SCENARIO_PLAN_SYSTEM_PROMPT
    assert "only up to the preparation/challenge page" in FFG_AUDIT_SYSTEM_PROMPT
    assert "You cannot provide human speech" in SCENARIO_EXEC_SYSTEM_PROMPT_TEMPLATE
    assert "Do not attempt CAPTCHA, slider/puzzle challenges or other human verification" in UNIFIED_SCENARIO_PLAN_SYSTEM_PROMPT
    assert "Stop when a challenge appears" in UNIFIED_SCENARIO_PLAN_SYSTEM_PROMPT
    assert "Never request automated solving, dragging, submission or completion of human verification" in FFG_AUDIT_SYSTEM_PROMPT
    assert "On a challenge, press_back once" in SCENARIO_EXEC_SYSTEM_PROMPT_TEMPLATE
    assert "verification blocked; exited" in SCENARIO_EXEC_SYSTEM_PROMPT_TEMPLATE


def test_prompts_require_compact_exploration_directions():
    assert "one novel scenario with a clear scope" in UNIFIED_SCENARIO_PLAN_SYSTEM_PROMPT
    assert "Open-ended guidance belongs to breadth" in UNIFIED_SCENARIO_PLAN_SYSTEM_PROMPT
    assert "Batch only controls on the same stable page or in the same result domain" in UNIFIED_SCENARIO_PLAN_SYSTEM_PROMPT
    assert "Describe unknown downstream behavior as a direction and observable continuation conditions, not as established facts" in UNIFIED_SCENARIO_PLAN_SYSTEM_PROMPT
    assert "Do not repeat the task, history, control inventory" in SCENARIO_EXEC_SYSTEM_PROMPT_TEMPLATE
    assert "one brief English sentence" in SCENARIO_EXEC_SYSTEM_PROMPT_TEMPLATE
    assert "an exploration direction, not a fixed script" in SCENARIO_EXEC_SYSTEM_PROMPT_TEMPLATE
    assert "An intermediate page is not automatically the endpoint" in SCENARIO_EXEC_SYSTEM_PROMPT_TEMPLATE
    assert "when no valuable new direction remains" in UNIFIED_SCENARIO_PLAN_SYSTEM_PROMPT
    assert "return an empty test_scenario" in UNIFIED_SCENARIO_PLAN_SYSTEM_PROMPT
    assert "Re-entering, waiting, paraphrasing or repeating a path is not a new scenario" in UNIFIED_SCENARIO_PLAN_SYSTEM_PROMPT


def test_ffg_prompts_treat_visible_entries_as_business_functions():
    assert "Name the goal, not an entry location" in BOOTSTRAP_FFG_SYSTEM_PROMPT
    assert "use goal names rather than entry-location names" in FFG_UPDATE_SYSTEM_PROMPT
    assert "entry_function_refs" in BOOTSTRAP_FFG_SYSTEM_PROMPT
    assert "entry_function_refs" in FFG_UPDATE_SYSTEM_PROMPT
    assert "The runtime records successes/failures from the actual trace" in FFG_UPDATE_SYSTEM_PROMPT
    assert "need their own functionality rather than belonging permanently to the parent list" in FFG_UPDATE_SYSTEM_PROMPT
    assert "Merely reaching a page may establish that capability's existence" in FFG_UPDATE_SYSTEM_PROMPT
    assert "coverage_percent" not in FFG_UPDATE_SYSTEM_PROMPT
    assert "coverage_percent" not in BOOTSTRAP_FFG_SYSTEM_PROMPT


def test_ffg_prompt_requires_splitting_sub_functions_of_large_functions():
    """Split sub-functionalities of large functionalities into separate nodes, preferring partial splits."""
    assert "Independent subdomains sharing settings/profile/management entry pages must be split" in FFG_UPDATE_SYSTEM_PROMPT
    assert "sharing settings/profile/management entry pages" in FFG_UPDATE_SYSTEM_PROMPT
    assert "must be split" in FFG_UPDATE_SYSTEM_PROMPT
    assert "Partial splits use retire_source=false" in FFG_UPDATE_SYSTEM_PROMPT
    assert "a full split need not happen at once" in FFG_UPDATE_SYSTEM_PROMPT


def test_prompt_json_examples_do_not_use_literal_union_placeholders():
    assert "has_exploration_gap" not in FFG_AUDIT_SYSTEM_PROMPT
    assert '"entry_name": ""' in FFG_AUDIT_SYSTEM_PROMPT
    assert "target_function" not in UNIFIED_SCENARIO_PLAN_SYSTEM_PROMPT
    assert "navigation_scenario" not in UNIFIED_SCENARIO_PLAN_SYSTEM_PROMPT
    assert '"test_scenario": "' in UNIFIED_SCENARIO_PLAN_SYSTEM_PROMPT
    assert "post-scenario defect review" not in SCENARIO_EXEC_SYSTEM_PROMPT_TEMPLATE


def test_audit_protocol_rejects_empty_or_ambiguous_results():
    assert _validate_audit_result({}) == ["empty_or_non_object_result"]
    assert _validate_audit_result({"entry_page_ref": "", "entry_name": ""}) == []
    assert _validate_audit_result({"entry_page_ref": "page_1", "entry_name": "\u8bbe\u7f6e"}) == []
    assert "entry_page_ref_and_entry_name_must_both_be_set_or_empty" in _validate_audit_result({
        "entry_page_ref": "page_1",
        "entry_name": "",
    })


def test_invalid_audit_result_is_capped_instead_of_blocking_completion(monkeypatch):
    """Remember the signature after two invalid audit responses to avoid repeated audits and blocked completion."""
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="\u8bbe\u7f6e", test_status="completed"))
    graph.add_page(Page(id="page_1", name="\u8bbe\u7f6e\u9875"), "func_1")
    calls = []
    monkeypatch.setattr(
        scenario_plan_module,
        "_audit_function_for_gap",
        lambda *_args, **_kwargs: calls.append(1) or {},
    )

    audit_result, audit_complete, signatures = _run_completion_audit(
        object(),
        graph,
        "core",
    )

    assert audit_result is None
    assert audit_complete is True
    assert len(calls) == 2
    assert list(signatures) == ["func_1"]

    _result, _complete, _signatures = _run_completion_audit(
        object(), graph, "core", signatures
    )
    assert len(calls) == 2


def test_completion_audit_stops_at_run_deadline(monkeypatch):
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="\u8bbe\u7f6e", test_status="completed"))
    graph.add_page(Page(id="page_1", name="\u8bbe\u7f6e\u9875"), "func_1")
    monkeypatch.setattr(
        scenario_plan_module,
        "_audit_function_for_gap",
        lambda *_args, **_kwargs: pytest.fail("audit must not run after the deadline"),
    )

    audit_result, audit_complete, signatures = _run_completion_audit(
        object(), graph, "core", deadline=0.0
    )

    assert (audit_result, audit_complete, signatures) == (None, False, {})


def test_ffg_prompt_can_update_existing_function_resource_variables():
    assert "Update an existing func_x only to correct a boundary or add observed resources" in FFG_UPDATE_SYSTEM_PROMPT
    assert "Empty arrays do not delete existing data" in FFG_UPDATE_SYSTEM_PROMPT
    assert '"function_ref": "new_func_1"' in FFG_UPDATE_SYSTEM_PROMPT
    assert '"function_dependencies": [{' in FFG_UPDATE_SYSTEM_PROMPT


def test_ffg_prompt_separates_planning_scope_from_function_ownership():
    assert "scenario_type limits scheduling scope, not discovery" in FFG_UPDATE_SYSTEM_PROMPT
    assert "never inherit current_plan.target_function by default" in FFG_UPDATE_SYSTEM_PROMPT
    assert '"function_refs": ["new_func_1"]' in FFG_UPDATE_SYSTEM_PROMPT
    assert '"page_ownership": [{' in FFG_UPDATE_SYSTEM_PROMPT
    assert "Visible business entries may establish functionalities" in FFG_UPDATE_SYSTEM_PROMPT
    assert "entry-only owners with add_entry_function_refs" in FFG_UPDATE_SYSTEM_PROMPT
    assert "behavior owners with add_function_refs" in FFG_UPDATE_SYSTEM_PROMPT
    assert '"test_status": "unavailable"' not in FFG_UPDATE_SYSTEM_PROMPT
    assert "Do not output test_status" in FFG_UPDATE_SYSTEM_PROMPT


def test_ffg_prompt_leaves_trace_edges_to_deterministic_code():
    assert '"operations": [' not in FFG_UPDATE_SYSTEM_PROMPT
    assert '"page_transitions": [' not in FFG_UPDATE_SYSTEM_PROMPT


def test_bootstrap_does_not_rewrite_function_by_package_specific_rule():
    graph = FlowGraph()
    current_page = Page(
        name="\u5fae\u4fe1\u4f1a\u8bdd\u5217\u8868",
        description="\u663e\u793a\u5f53\u524d\u4f1a\u8bdd\u6761\u76ee",
    )

    class BootstrapLlm:
        class Response:
            content = (
                '{"functions":[{"function_ref":"new_func_1","source_step":1,'
                '"name":"\u5fae\u4fe1\u4f1a\u8bdd","description":"\u5fae\u4fe1\u4e2d\u7684\u6240\u6709\u4f1a\u8bdd\u80fd\u529b",'
                '"entry_page_ref":"new_page_1"}],"pages":[{'
                '"page_ref":"new_page_1","function_refs":["new_func_1"],'
                '"source_step":1,"name":"\u5fae\u4fe1\u4f1a\u8bdd\u5217\u8868",'
                '"description":"\u663e\u793a\u5f53\u524d\u4f1a\u8bdd\u6761\u76ee","controls":[]}]}'
            )

        def invoke(self, _messages):
            return self.Response()

    _bootstrap_initial_ffg(
        BootstrapLlm(),
        graph,
        current_page,
        app_name="\u5fae\u4fe1",
        package_name="com.tencent.mm",
    )

    node = graph.get_node("func_1")
    assert node.name == "\u5fae\u4fe1\u4f1a\u8bdd"
    assert node.description == "\u5fae\u4fe1\u4e2d\u7684\u6240\u6709\u4f1a\u8bdd\u80fd\u529b"


def test_bootstrap_visible_business_entries_share_the_observed_page():
    graph = FlowGraph()
    current_page = Page(
        name="\u4f1a\u8bdd\u804a\u5929\u9875\u9762",
        description="\u663e\u793a\u6d88\u606f\u548c\u6269\u5c55\u529f\u80fd\u9762\u677f",
    )

    class BootstrapLlm:
        class Response:
            content = (
                '{"functions":['
                '{"function_ref":"new_func_1","source_step":1,'
                '"name":"\u4f1a\u8bdd\u6d88\u606f\u6d4f\u89c8\u4e0e\u6536\u53d1","description":"\u6d4f\u89c8\u548c\u53d1\u9001\u6d88\u606f","coverage_percent":0},'
                '{"function_ref":"new_func_2","source_step":1,'
                '"name":"\u4ece\u76f8\u518c\u9009\u62e9\u5e76\u53d1\u9001\u5a92\u4f53","description":"\u7531\u76f8\u518c\u5165\u53e3\u53d1\u73b0，\u6d41\u7a0b\u5f85\u786e\u8ba4","coverage_percent":0},'
                '{"function_ref":"new_func_3","source_step":1,'
                '"name":"\u9009\u62e9\u5e76\u53d1\u9001\u6587\u4ef6","description":"\u7531\u6587\u4ef6\u5165\u53e3\u53d1\u73b0，\u6d41\u7a0b\u5f85\u786e\u8ba4","coverage_percent":0}'
                '],"pages":[{'
                '"page_ref":"new_page_1",'
                '"function_refs":["new_func_1","new_func_2","new_func_3"],'
                '"source_step":1,"name":"\u4f1a\u8bdd\u804a\u5929\u9875\u9762",'
                '"description":"\u663e\u793a\u6d88\u606f\u548c\u6269\u5c55\u529f\u80fd\u9762\u677f",'
                '"controls":["\u76f8\u518c\u5165\u53e3","\u6587\u4ef6\u5165\u53e3"]}]}'
            )

        def invoke(self, _messages):
            return self.Response()

    _bootstrap_initial_ffg(
        BootstrapLlm(),
        graph,
        current_page,
        app_name="\u6d4b\u8bd5\u5e94\u7528",
        package_name="com.example.app",
    )

    assert list(graph.functions) == ["func_1", "func_2", "func_3"]
    assert graph.get_page_owners("page_1") == ["func_1", "func_2", "func_3"]
    assert all(graph.get_node(function_id).page_ids == ["page_1"] for function_id in graph.functions)
    assert graph.get_node("func_1").entry_page_ids == []
    assert graph.get_node("func_2").entry_page_ids == ["page_1"]
    assert graph.get_node("func_3").entry_page_ids == ["page_1"]
    assert graph.get_node("func_2").coverage_percent == 0
    assert graph.get_node("func_3").test_status == "not_started"


def test_last_executable_plan_is_not_skipped_by_planner_limit():
    state = {
        "iteration_count": 3,
        "max_iterations": 3,
        "workflow_complete": False,
        "current_plan": {"scenario_type": "intra_test"},
    }

    assert _route_after_plan(state) == "scenario_exec"


def test_cross_candidates_ignore_navigation_transition_and_prefer_dependency():
    graph = _graph_with_two_functions()
    graph.add_function_dependency(FunctionDependency("func_1", "func_2", "\u5df2\u521b\u5efa\u8bb0\u5f55"))

    candidates = _candidate_cross_pairs(graph)

    assert candidates == [("func_1", "func_2", "Functional dependency: \u5df2\u521b\u5efa\u8bb0\u5f55")]


def test_cross_candidates_fall_back_to_shared_resource_variables():
    graph = _graph_with_two_functions()
    graph.get_node("func_1").resource_variables = ["\u4efb\u52a1\u8bb0\u5f55"]
    graph.get_node("func_2").resource_variables = ["\u4efb\u52a1\u8bb0\u5f55"]

    candidates = _candidate_cross_pairs(graph)

    assert candidates == [
        ("func_1", "func_2", "Shared state variable: \u4efb\u52a1\u8bb0\u5f55"),
        ("func_2", "func_1", "Shared state variable: \u4efb\u52a1\u8bb0\u5f55"),
    ]


def test_navigation_transition_alone_does_not_create_cross_test_candidate():
    graph = _graph_with_two_functions()

    assert _candidate_cross_pairs(graph) == []


def test_shared_resource_fallback_uses_stable_function_order():
    graph = _graph_with_two_functions()
    graph.get_node("func_1").resource_variables = ["\u4efb\u52a1\u8bb0\u5f55"]
    graph.get_node("func_2").resource_variables = ["\u4efb\u52a1\u8bb0\u5f55"]
    candidates = _candidate_cross_pairs(graph)

    assert candidates == [
        ("func_1", "func_2", "Shared state variable: \u4efb\u52a1\u8bb0\u5f55"),
        ("func_2", "func_1", "Shared state variable: \u4efb\u52a1\u8bb0\u5f55"),
    ]


def test_cross_candidates_dedupe_reverse_directions():
    graph = _graph_with_two_functions()
    graph.add_function_dependency(FunctionDependency("func_1", "func_2", "\u5df2\u521b\u5efa\u8bb0\u5f55"))

    candidates = _candidate_cross_pairs(graph)

    assert candidates == [("func_1", "func_2", "Functional dependency: \u5df2\u521b\u5efa\u8bb0\u5f55")]


def test_cross_pair_done_does_not_block_reverse():
    graph = _graph_with_two_functions()
    graph.add_function_dependency(FunctionDependency("func_1", "func_2", "\u8be6\u60c5\u8bb0\u5f55"))
    graph.add_cross_scenario("func_2", "func_1", "\u5df2\u9a8c\u8bc1\u521b\u5efa\u5230\u5217\u8868")
    graph.record_cross_test("func_2", "func_1", "\u8be6\u60c5\u8bb0\u5f55", "\u5df2\u9a8c\u8bc1\u521b\u5efa\u5230\u5217\u8868", "done")
    graph.get_node("func_1").test_status = "completed"
    graph.get_node("func_2").test_status = "completed"

    assert _pending_cross_pairs(graph) == [("func_1", "func_2", "Functional dependency: \u8be6\u60c5\u8bb0\u5f55")]
    tested_keys = _done_cross_keys(graph)
    assert ("func_1", "func_2", "\u8be6\u60c5\u8bb0\u5f55") not in tested_keys
    assert ("func_2", "func_1", "\u8be6\u60c5\u8bb0\u5f55") in tested_keys


def test_completion_audit_returns_execution_candidate_without_mutating_ffg():
    graph = FlowGraph()
    node = FunctionNode(id="func_1", name="\u4e3b\u5bfc\u822a", test_status="completed")
    graph.add_node(node)
    graph.add_page(Page(
        id="page_1",
        name="\u9996\u9875",
        description="\u5305\u542b\u5df2\u8bb0\u5f55\u7684\u5bfc\u5165\u5165\u53e3",
        representative_screenshot="stored-screenshot",
    ), "func_1")

    class AuditLlm:
        class Response:
            content = '{"entry_page_ref":"page_1","entry_name":"\u5bfc\u5165\u5165\u53e3"}'

        def invoke(self, _messages):
            return self.Response()

    before_ids = list(graph.functions)
    result, audit_complete, _signatures = _run_completion_audit(
        AuditLlm(), graph, "core"
    )

    assert audit_complete is False
    assert result["owner_function_id"] == "func_1"
    assert result["entry_page_id"] == "page_1"
    assert list(graph.functions) == before_ids
    assert "functions" not in result
    assert "Return one entry exploration" in FFG_AUDIT_SYSTEM_PROMPT


def test_completion_audit_retries_a_gap_on_an_unknown_page():
    graph = FlowGraph()
    node = FunctionNode(id="func_1", name="\u4e3b\u5bfc\u822a", test_status="completed")
    graph.add_node(node)
    graph.add_page(Page(
        id="page_1",
        name="\u9996\u9875",
        description="\u5305\u542b\u5c1a\u672a\u63a2\u7d22\u7684\u5bfc\u5165\u5165\u53e3",
    ), "func_1")

    class AuditLlm:
        def __init__(self):
            self.call_count = 0

        def invoke(self, _messages):
            self.call_count += 1
            content = (
                '{"entry_page_ref":"page_9","entry_name":"\u5bfc\u5165\u5165\u53e3"}'
                if self.call_count == 1
                else '{"entry_page_ref":"page_1","entry_name":"\u5bfc\u5165\u5165\u53e3"}'
            )
            return type("Response", (), {"content": content})()

    llm = AuditLlm()
    result, audit_complete, _signatures = _run_completion_audit(llm, graph, "core")

    assert llm.call_count == 2
    assert audit_complete is False
    # Use the program's scenario template instead of the model's original wording.
    assert result["test_scenario"].startswith("On \"\u9996\u9875\", click \"\u5bfc\u5165\u5165\u53e3\"")


def test_completion_audit_allows_wechat_conversation_entry():
    graph = FlowGraph()
    node = FunctionNode(id="func_1", name="\u5fae\u4fe1\u9996\u9875", test_status="completed")
    graph.add_node(node)
    graph.add_page(Page(
        id="page_1",
        name="\u5fae\u4fe1\u9996\u9875",
        description="\u5305\u542b\u6587\u4ef6\u4f20\u8f93\u52a9\u624b\u4f1a\u8bdd",
    ), "func_1")

    class AuditLlm:
        class Response:
            content = (
                '{"entry_page_ref":"page_1","entry_name":"\u6587\u4ef6\u4f20\u8f93\u52a9\u624b\u4f1a\u8bdd"}'
            )

        def invoke(self, _messages):
            return self.Response()

    result, audit_complete, _signatures = _run_completion_audit(
        AuditLlm(),
        graph,
        "core",
        package_name="com.tencent.mm",
    )

    assert result is not None
    assert result["entry_page_id"] == "page_1"
    assert audit_complete is False


def test_new_unfinished_function_runs_before_resuming_completed_function_audit():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="\u5df2\u5b8c\u6210\u5165\u53e3", test_status="completed"))
    graph.add_page(Page(id="page_1", name="\u9996\u9875", representative_screenshot="stored"), "func_1")
    graph.add_node(FunctionNode(id="func_2", name="\u65b0\u53d1\u73b0\u529f\u80fd", test_status="not_started"))
    graph.add_page(Page(id="page_2", name="\u65b0\u9875\u9762", representative_screenshot="stored"), "func_2")

    class PlanningLlm:
        calls = 0

        def invoke(self, _messages):
            self.calls += 1
            return type("Response", (), {"content": (
                '{"workflow_complete":false,"scenario_type":"intra_test",'
                '"target_function":"func_2",'
                '"src_function":"","dst_function":"",'
                '"navigation_scenario":"","test_scenario":"\u70b9\u51fb\u65b0\u9875\u9762\u4e2d\u7684\u5df2\u8bb0\u5f55\u5165\u53e3 → \u786e\u8ba4\u7ed3\u679c\u540e\u7ed3\u675f"}'
            )})()

    llm = PlanningLlm()
    planner = create_scenario_plan_node(llm, object())
    result = planner({
        "ffg": graph,
        "current_page": graph.get_page("page_2"),
        "current_plan": {
            "scenario_type": "intra_test",
            "target_function_id": "func_1",
            "audit_exploration": True,
        },
        "current_testing_function_id": "func_1",
        "current_testing_function_status": "completed",
        "run_phase": "all",
        "completion_audit_attempts": {"func_1": 1},
    })

    assert llm.calls == 1
    assert result["current_plan"]["target_function_id"] == "func_2"
    assert result["current_plan"].get("audit_exploration") is not True


def test_existing_tested_scenarios_do_not_bypass_program_limits():
    graph = FlowGraph()
    node = FunctionNode(id="func_1", name="\u4e3b\u9875\u529f\u80fd", test_status="in_progress")
    graph.add_node(node)
    graph.add_page(Page(
        id="page_1",
        name="\u4e3b\u9875",
        description="\u5305\u542b\u5c1a\u672a\u70b9\u51fb\u7684\u4fa7\u8fb9\u83dc\u5355\u5165\u53e3",
        controls=["\u53ef\u70b9\u51fb\u7684‘\u4fa7\u8fb9\u83dc\u5355’\u6309\u94ae"],
    ), "func_1")
    for index in range(5):
        node.add_scenario(f"\u5df2\u5b8c\u6210\u7684\u4e0d\u540c\u573a\u666f {index}")

    class Llm:
        calls = 0

        def invoke(self, _messages):
            self.calls += 1
            content = (
                '{"workflow_complete":false,"scenario_type":"intra_test",'
                '"target_function":"func_1",'
                '"src_function":"","dst_function":"",'
                '"navigation_scenario":"","test_scenario":'
                "\"\u5728\u4e3b\u9875\u70b9\u51fb\u2018\u4fa7\u8fb9\u83dc\u5355\u2019\u6309\u94ae\uff0c\u89c2\u5bdf\u5e76\u8bb0\u5f55\u5b9e\u9645\u51fa\u73b0\u7684\u9875\u9762\u6216\u62bd\u5c49\"}"
            )
            return type("Response", (), {"content": content})()

    llm = Llm()
    planner = create_scenario_plan_node(llm, object())
    result = planner({
        "ffg": graph,
        "current_page": graph.get_page("page_1"),
        "current_testing_function_id": "func_1",
        "current_testing_function_status": "in_progress",
        "current_plan": {"scenario_type": "intra_test"},
        "run_phase": "all",
        "completion_audit_signatures": {},
    })

    assert llm.calls == 1
    assert result["current_plan"]["scenario_type"] == "intra_test"
    assert result["current_plan"]["target_function_id"] == "func_1"
    assert graph.get_node_test_status("func_1") == "in_progress"


def test_unplannable_function_is_deferred_without_marking_it_completed():
    graph = FlowGraph()
    node = FunctionNode(id="func_1", name="\u8bb0\u5f55\u6d3b\u52a8", test_status="in_progress")
    graph.add_node(node)
    graph.add_page(Page(id="page_1", name="\u6d3b\u52a8\u4e3b\u9875"), "func_1")

    first = _release_unplannable_function(
        graph,
        graph.get_page("page_1"),
        node,
        "\u89c4\u5212\u5668\u4e0e\u7a0b\u5e8f\u6a21\u677f\u5747\u65e0\u53ef\u6267\u884c\u573a\u666f",
    )
    # One empty scenario increments skips; defer only after the consecutive threshold.
    assert graph.get_node("func_1").test_status == "in_progress"
    assert first["unplannable_counts"] == {"func_1": 1}

    result = _release_unplannable_function(
        graph,
        graph.get_page("page_1"),
        node,
        "\u89c4\u5212\u5668\u4e0e\u7a0b\u5e8f\u6a21\u677f\u5747\u65e0\u53ef\u6267\u884c\u573a\u666f",
        first["unplannable_counts"],
    )

    assert graph.get_node("func_1").test_status == "deferred"
    assert graph.get_node("func_1").status_reason == "unplannable"
    assert result["unplannable_counts"] == {}
    assert result["current_plan"]["scenario_type"] == "skip"
    assert result["current_test_scenario"] == ""
    assert result["current_testing_function_id"] == ""


def test_low_coverage_empty_plan_immediately_generates_representative_scenario():
    graph = FlowGraph()
    node = FunctionNode(id="func_1", name="\u8bb0\u5f55\u6d3b\u52a8", test_status="in_progress")
    graph.add_node(node)
    graph.add_page(Page(
        id="page_1",
        name="\u6d3b\u52a8\u4e3b\u9875",
        controls=["\u641c\u7d22\u6309\u94ae"],
    ), "func_1")

    class Llm:
        def invoke(self, _messages):
            return type("Response", (), {"content": '{"test_scenario":""}'})()

    planner = create_scenario_plan_node(Llm(), object())
    result = planner({
        "ffg": graph,
        "current_page": graph.get_page("page_1"),
        "current_testing_function_id": "func_1",
        "current_testing_function_status": "in_progress",
        "run_phase": "all",
    })

    assert result["current_plan"]["scenario_type"] == "intra_test"
    assert result["current_plan"]["target_function_id"] == "func_1"
    assert "\u641c\u7d22\u6309\u94ae" in result["current_test_scenario"]


def test_shared_resource_fallback_does_not_depend_on_navigation_direction():
    graph = _graph_with_two_functions()
    graph.get_node("func_1").resource_variables = ["\u4efb\u52a1\u8bb0\u5f55"]
    graph.get_node("func_2").resource_variables = ["\u4efb\u52a1\u8bb0\u5f55"]
    candidates = _candidate_cross_pairs(graph)

    assert candidates == [
        ("func_1", "func_2", "Shared state variable: \u4efb\u52a1\u8bb0\u5f55"),
        ("func_2", "func_1", "Shared state variable: \u4efb\u52a1\u8bb0\u5f55"),
    ]


def test_empty_intra_scenario_is_valid_but_cross_requires_one():
    assert _validate_plan_result({"test_scenario": ""}) == []
    assert _validate_plan_result({"test_scenario": None}) == []
    assert "cross_test_requires_nonempty_test_scenario" in _validate_plan_result(
        {"test_scenario": ""}, scenario_type="cross_test"
    )


def test_planner_no_longer_rejects_transient_control_mentions():
    """Transient-control validation was removed, so UNDO/Toast mentions do not cause programmatic rejection."""
    standalone_undo = {
        "workflow_complete": False,
        "scenario_type": "intra_test",
        "target_function": "func_1",
        "src_function": "",
        "dst_function": "",
        "navigation_scenario": "",
        "test_scenario": "\u63a2\u7d22\u7f16\u8f91\u529f\u80fd\uff0c\u82e5\u51fa\u73b0 UNDO \u6309\u94ae\u5219\u89c2\u5bdf\u5176\u6548\u679c",
    }
    immediate_undo = {
        **standalone_undo,
        "test_scenario": "\u70b9\u51fb\u6d3b\u52a8\u6309\u94ae\u542f\u52a8\u8bb0\u5f55\uff0c\u7d27\u63a5\u7740\u70b9\u51fb UNDO \u6309\u94ae\u5e76\u89c2\u5bdf\u7ed3\u679c",
    }

    assert not any(
        error.startswith("test_scenario_requires_live_transient_control=")
        for error in _validate_plan_result(standalone_undo)
    )
    assert not any(
        error.startswith("test_scenario_requires_live_transient_control=")
        for error in _validate_plan_result(immediate_undo)
    )


def test_planner_accepts_open_ended_conditional_scenario():
    """Allow conditional open endings when downstream results are unknown: continue if a result appears, otherwise end."""
    search_scenario = {
        "workflow_complete": False,
        "scenario_type": "intra_test",
        "target_function": "func_2",
        "src_function": "",
        "dst_function": "",
        "navigation_scenario": "",
        "test_scenario": (
            "\u70b9\u51fb\u641c\u7d22\u6846\u5e76\u8f93\u5165\u5173\u952e\u8bcd，\u82e5\u51fa\u73b0\u641c\u7d22\u7ed3\u679c，\u70b9\u51fb\u9996\u4e2a\u7ed3\u679c\u67e5\u770b\u8be6\u60c5；"
            "\u82e5\u7ed3\u679c\u4e3a\u7a7a\u5219\u7ed3\u675f"
        ),
    }
    conditional_nav_scenario = {
        **search_scenario,
        "test_scenario": (
            "\u70b9\u51fb\u6253\u5f00\u76f8\u673a\u5165\u53e3，\u82e5\u51fa\u73b0\u76f8\u673a\u53d6\u666f\u9875\u9762，\u70b9\u51fb\u62cd\u7167\u6309\u94ae\u786e\u8ba4\u8fdb\u5165；"
            "\u82e5\u672a\u51fa\u73b0\u5219\u7ed3\u675f"
        ),
    }

    assert _validate_plan_result(search_scenario) == []
    assert _validate_plan_result(conditional_nav_scenario) == []
    # An open ending must not trigger exhaustive-control or action-count rejection.
    errors = _validate_plan_result(search_scenario)
    assert "test_scenario_exhaustive_control_family" not in errors
    assert "test_scenario_too_many_actions" not in errors


def test_open_ended_scenario_no_longer_action_counted():
    """Do not reject long exploration directions by counting actions."""
    padded = {
        "workflow_complete": False,
        "scenario_type": "intra_test",
        "target_function": "func_1",
        "src_function": "",
        "dst_function": "",
        "navigation_scenario": "",
        "test_scenario": "，".join(["\u70b9\u51fb\u6309\u94ae"] * 8) + "，\u82e5\u51fa\u73b0\u7ed3\u679c\u5219\u70b9\u51fb\u67e5\u770b",
    }

    assert not any(
        error.startswith("test_scenario_too_many_actions=")
        for error in _validate_plan_result(padded)
    )


def test_plan_prompt_mentions_open_ending_support():
    """The planning system prompt describes general conditional open endings."""
    assert "Open-ended guidance" in UNIFIED_SCENARIO_PLAN_SYSTEM_PROMPT
    assert "downstream" in UNIFIED_SCENARIO_PLAN_SYSTEM_PROMPT
    assert "continue to an explicit unexecuted related result" in UNIFIED_SCENARIO_PLAN_SYSTEM_PROMPT
    assert "Let the executor choose concrete actions from each screenshot" in UNIFIED_SCENARIO_PLAN_SYSTEM_PROMPT
    assert "After entering a keyword, if search results appear" not in UNIFIED_SCENARIO_PLAN_SYSTEM_PROMPT


def test_plan_prompt_splits_independent_stable_flows():
    assert "One scenario has one independent business goal" in UNIFIED_SCENARIO_PLAN_SYSTEM_PROMPT
    assert "Independent flows such as albums, locations, files, contacts, record search and settings must be separate scenarios" in UNIFIED_SCENARIO_PLAN_SYSTEM_PROMPT
    assert "albums, locations, files, contacts" in UNIFIED_SCENARIO_PLAN_SYSTEM_PROMPT
    assert "observe and stop at a new full-screen page, independent Activity" in UNIFIED_SCENARIO_PLAN_SYSTEM_PROMPT


def test_executor_stops_at_first_independent_stable_flow():
    assert "Follow the task phase and independent stable-flow boundaries" in SCENARIO_EXEC_SYSTEM_PROMPT_TEMPLATE
    assert "A task to expand a panel stops at that panel; entering settings does not authorize unrelated" in SCENARIO_EXEC_SYSTEM_PROMPT_TEMPLATE
    assert "does not authorize unrelated search, media, file, location or contact flows" in SCENARIO_EXEC_SYSTEM_PROMPT_TEMPLATE


def test_plan_transport_failure_does_not_multiply_semantic_retries(monkeypatch):
    class FailingLlm:
        calls = 0

        def invoke(self, _messages):
            self.calls += 1
            raise RuntimeError("temporary gateway failure")

    llm = FailingLlm()
    monkeypatch.setattr(scenario_plan_module, "PLAN_VALIDATION_RETRY_ATTEMPTS", 3)
    monkeypatch.setenv("LLM_RETRY_ATTEMPTS", "1")
    result, errors = scenario_plan_module._invoke_valid_plan(llm, "prompt", [])

    assert result == {}
    assert errors == ["llm_request_or_parse_failed"]
    assert llm.calls == 1


def test_unfinished_current_function_is_dispatched_before_pending_cross_test():
    graph = _graph_with_two_functions()
    graph.add_page(Page(id="page_1", name="\u4efb\u52a1\u7f16\u8f91"), "func_1")
    graph.add_page(Page(id="page_2", name="\u4efb\u52a1\u8bb0\u5f55"), "func_2")
    graph.add_page_transition(PageTransition("page_1", "page_2", "\u6253\u5f00\u4efb\u52a1\u8bb0\u5f55"))
    graph.set_node_test_status("func_1", "in_progress")
    graph.add_function_dependency(FunctionDependency("func_1", "func_2", "\u4efb\u52a1\u8bb0\u5f55"))

    class Response:
        content = (
            '{"workflow_complete":false,"scenario_type":"intra_test",'
            '"target_function":"func_1",'
            '"src_function":"","dst_function":"",'
            '"navigation_scenario":"",'
            '"test_scenario":"\u5728\u6e90\u529f\u80fd\u7f16\u8f91\u5e76\u4fdd\u5b58\u4e00\u6761\u4efb\u52a1，\u786e\u8ba4\u4fdd\u5b58\u7ed3\u679c"}'
        )

    class Llm:
        def __init__(self):
            self.calls = 0

        def invoke(self, _messages):
            self.calls += 1
            return Response()

    llm = Llm()
    node = create_scenario_plan_node(llm, object())
    result = node({
        "ffg": graph,
        "current_page": Page(id="page_1", name="\u4efb\u52a1\u7f16\u8f91"),
        "current_testing_function_id": "func_1",
        "current_testing_function_status": "in_progress",
        "run_phase": "all",
    })

    assert llm.calls == 1
    assert result["current_plan"]["scenario_type"] == "intra_test"
    assert result["current_testing_function_id"] == "func_1"
    assert "\u7f16\u8f91\u5e76\u4fdd\u5b58" in result["current_test_scenario"]


def test_planner_unfinished_function_is_dispatched_before_completed_function_audit():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="\u4e3b\u5bfc\u822a", test_status="completed",
                                depth_scenario_count=1, successful_depth_scenario_count=1))
    graph.add_node(FunctionNode(id="func_2", name="\u5f85\u6d4b\u529f\u80fd", test_status="not_started"))
    graph.add_page(Page(
        id="page_1",
        name="Navigation Drawer",
        description="\u5305\u542b Diary、Settings \u548c About \u83dc\u5355\u9879",
    ), "func_1")
    graph.add_page(Page(
        id="page_2",
        name="\u5f85\u6d4b\u9875\u9762",
        description="\u5305\u542b\u5f85\u6d4b\u5165\u53e3",
    ), "func_2")
    graph.add_page_transition(PageTransition("page_1", "page_2", "\u70b9\u51fb\u5f85\u6d4b\u529f\u80fd\u5165\u53e3"))

    class AuditLlm:
        calls = 0

        def invoke(self, _messages):
            self.calls += 1
            return type("Response", (), {"content": (
                '{"workflow_complete":false,"scenario_type":"intra_test",'
                '"target_function":"func_2",'
                '"src_function":"","dst_function":"",'
                '"navigation_scenario":"","test_scenario":'
                '"\u70b9\u51fb\u5f85\u6d4b\u9875\u9762\u4e2d\u7684\u5f85\u6d4b\u5165\u53e3 → \u786e\u8ba4\u7ed3\u679c\u540e\u7ed3\u675f"}'
            )})()

    llm = AuditLlm()
    node = create_scenario_plan_node(llm, object())
    result = node({
        "ffg": graph,
        "current_page": Page(id="page_1", name="Navigation Drawer"),
        "current_plan": {
            "scenario_type": "skip",
            "target_function_id": "func_1",
        },
        "current_testing_function_id": "",
        "current_testing_function_status": "not_started",
        "run_phase": "all",
    })

    assert llm.calls == 1
    assert result["current_plan"].get("audit_exploration") is not True
    assert result["current_plan"]["target_function_id"] == "func_2"
    assert "\u5f85\u6d4b\u5165\u53e3" in result["current_test_scenario"]


def test_completed_coverage_finishes_early_when_frontier_is_exhausted():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="\u4e3b\u529f\u80fd", test_status="completed",
                                depth_scenario_count=1, successful_depth_scenario_count=1))
    graph.add_page(Page(id="page_1", name="\u4e3b\u9875", operations=["\u70b9\u51fb\u5237\u65b0\u6309\u94ae"]), "func_1")

    class DoneLlm:
        def invoke(self, _messages):
            return type("Response", (), {"content": (
                '{"entry_page_ref":"","entry_name":""}'
            )})()

    node = create_scenario_plan_node(DoneLlm(), object())
    result = node({
        "ffg": graph,
        "current_page": Page(id="page_1", name="\u4e3b\u9875"),
        "current_plan": {},
        "current_testing_function_id": "",
        "current_testing_function_status": "not_started",
        "run_phase": "all",
        "start_time": time.time(),
        "total_time_limit": 1800,
        "iteration_count": 3,
    })

    assert result["workflow_complete"] is True
    assert result["current_plan"]["scenario_type"] == "skip"
    assert result["current_test_scenario"] == ""


def test_completion_audit_gets_recorded_navigation_to_its_entry_page():
    graph = FlowGraph()
    node = FunctionNode(id="func_1", name="\u7b14\u8bb0\u5217\u8868")
    graph.add_node(node)
    graph.add_page(Page(id="page_1", name="\u7b14\u8bb0\u9996\u9875"), "func_1")
    graph.add_page(Page(id="page_2", name="\u65b0\u589e\u7c7b\u578b\u83dc\u5355"), "func_1")
    graph.add_page(Page(id="page_3", name="\u7b14\u8bb0\u7f16\u8f91\u9875"), "func_1")
    graph.add_page_transition(PageTransition("page_1", "page_2", "\u70b9\u51fb\u65b0\u589e\u6309\u94ae"))

    navigation = _build_audit_navigation_scenario(
        graph,
        graph.get_page("page_3"),
        "func_1",
        "page_2",
    )

    assert "try back once" in navigation
    assert "returns to \"\u65b0\u589e\u7c7b\u578b\u83dc\u5355\"" in navigation
    assert "use open_app once" in navigation
    assert "\u70b9\u51fb\u65b0\u589e\u6309\u94ae" in navigation
    assert "Do not click the entry under test during navigation" in navigation
    assert "Menus opened deliberately to reach the target are part of the path; do not close them again" in navigation
    assert _build_audit_navigation_scenario(
        graph,
        graph.get_page("page_2"),
        "func_1",
        "page_2",
    ) == ""


def test_completion_audit_uses_open_app_first_for_unrelated_current_page():
    graph = FlowGraph()
    node = FunctionNode(id="func_1", name="\u7b14\u8bb0\u5217\u8868")
    graph.add_node(node)
    graph.add_node(FunctionNode(id="func_2", name="\u5176\u4ed6\u529f\u80fd"))
    graph.add_page(Page(id="page_1", name="\u7b14\u8bb0\u9996\u9875"), "func_1")
    graph.add_page(Page(id="page_2", name="\u65b0\u589e\u7c7b\u578b\u83dc\u5355"), "func_1")
    graph.add_page(Page(id="page_other", name="\u5176\u4ed6\u529f\u80fd\u9875\u9762"), "func_2")
    graph.add_page_transition(PageTransition("page_1", "page_2", "\u70b9\u51fb\u65b0\u589e\u6309\u94ae"))

    navigation = _build_audit_navigation_scenario(
        graph,
        graph.get_page("page_other"),
        "func_1",
        "page_2",
    )

    assert "First use open_app once" in navigation
    assert "try back once" not in navigation


def test_audit_entry_scenario_uses_page_names_not_internal_ids():
    from workflow.nodes.scenario_plan import _audit_entry_scenario

    scenario = _audit_entry_scenario(Page(id="page_1", name="\u804a\u5929\u9875\u9762"), "\u8868\u60c5\u641c\u7d22\u5165\u53e3")

    assert "page_1" not in scenario
    assert "\u804a\u5929\u9875\u9762" in scenario
    assert "\u8868\u60c5\u641c\u7d22\u5165\u53e3" in scenario


def test_forced_progress_selects_only_one_uncovered_control():
    graph = FlowGraph()
    node = FunctionNode(id="func_1", name="\u7b14\u8bb0\u5217\u8868")
    graph.add_node(node)
    graph.add_page(Page(
        id="page_1",
        name="\u7b14\u8bb0\u9996\u9875",
        controls=[
            "\u53ef\u70b9\u51fb\u7684‘\u641c\u7d22’\u6309\u94ae",
            "\u53ef\u70b9\u51fb\u7684‘\u6392\u5e8f’\u6309\u94ae",
            "\u53ef\u70b9\u51fb\u7684‘\u4fa7\u8fb9\u680f’\u6309\u94ae",
        ],
        operations=["\u70b9\u51fb\u641c\u7d22\u6309\u94ae", "\u6253\u5f00\u4fa7\u8fb9\u680f"],
    ), "func_1")

    scenario = _forced_intra_scenario(graph, node, hint_index=1)

    assert "\u6392\u5e8f" in scenario
    assert "\u641c\u7d22" not in scenario
    assert "\u4fa7\u8fb9\u680f" not in scenario
    assert "with no execution record" in scenario


def test_forced_progress_may_choose_wechat_conversation_control():
    graph = FlowGraph()
    node = FunctionNode(id="func_1", name="\u5fae\u4fe1\u9996\u9875")
    graph.add_node(node)
    graph.add_page(Page(
        id="page_1",
        name="\u5fae\u4fe1\u9996\u9875",
        controls=[
            "\u53ef\u70b9\u51fb\u7684‘\u6587\u4ef6\u4f20\u8f93\u52a9\u624b\u4f1a\u8bdd’",
            "\u53ef\u70b9\u51fb\u7684‘\u901a\u8baf\u5f55\u6807\u7b7e’",
        ],
    ), "func_1")

    scenario = _forced_intra_scenario(
        graph,
        node,
        package_name="com.tencent.mm",
    )

    assert "\u6587\u4ef6\u4f20\u8f93\u52a9\u624b" in scenario


def test_exec_fallback_marks_current_page_unknown():
    state = {
        "current_page": Page(id="page_old"),
        "current_test_scenario": "\u6d4b\u8bd5\u4fdd\u5b58",
        "current_plan": {"scenario_type": "intra_test"},
    }

    updates = _fallback_scenario_exec(state, RuntimeError("boom"))

    assert updates["current_page"] is None
    assert updates["execution_result"]["scenario_status"] == "error"
    assert updates["execution_result"]["scenario_error"] == "Scenario execution node failed: boom"
    assert updates["execution_result"]["scenario_trace_length"] == 0
    assert not {
        "test_status",
        "test_error",
        "test_trace_length",
    }.intersection(updates["execution_result"])


def test_missing_plan_target_does_not_fall_back_to_previous_function():
    graph = _graph_with_two_functions()

    assert _resolve_plan_target_function_id(graph, {}) == ""
    assert _resolve_plan_target_function_id(
        graph,
        {"target_function_id": "func_2"},
    ) == "func_2"


def test_finished_with_partial_success_is_not_a_failure():
    from workflow.nodes.scenario_exec import _finished_declares_failure

    thought = (
        "\u5f53\u524d\u5df2\u6210\u529f\u8fd4\u56deNotes\u7b14\u8bb0\u5217\u8868\u9875\u9762，\u5217\u8868\u6700\u9876\u90e8\u5df2\u663e\u793a\u6807\u9898\u4e3a“FuncDroid_29”\u7684\u6e05\u5355\u7b14\u8bb0，"
        "\u5df2\u5b8c\u6210\u6807\u9898\u8f93\u5165、\u786e\u8ba4\u9ed8\u8ba4\u5f85\u529e\u9879“New item”\u53ca\u5176\u590d\u9009\u6846\u5b58\u5728\u7684\u6b65\u9aa4，"
        "\u4ec5\u65b0\u589e\u989d\u5916\u5f85\u529e\u9879\u7684\u64cd\u4f5c\u672a\u6210\u529f\u89e6\u53d1，\u6838\u5fc3\u4efb\u52a1\u7684\u4e3b\u8981\u9a8c\u8bc1\u70b9\u5df2\u8986\u76d6，\u53ef\u7ed3\u675f\u672c\u6b21\u6267\u884c\u4efb\u52a1。"
    )

    assert _finished_declares_failure(thought) is False


def test_intra_target_prefers_reachable_unpenalized_function():
    graph = _graph_with_two_functions()
    graph.add_page(Page(id="page_1", name="\u6e90\u9875\u9762"), "func_1")
    graph.add_page(Page(id="page_2", name="\u76ee\u6807\u9875\u9762"), "func_2")
    graph.add_page_transition(PageTransition("page_1", "page_2", "\u8fdb\u5165\u76ee\u6807\u529f\u80fd"))

    selected = scenario_plan_module._select_intra_target(
        graph,
        graph.get_page("page_1"),
        {"func_1": 1},
    )

    assert selected is not None
    assert selected.id == "func_2"


def _done_cross_keys(graph):
    """Return done inter-functional combination keys, replacing the removed _tested_cross_pair_keys production helper."""
    return {
        graph.cross_test_key(item["src_function_id"], item["dst_function_id"], item["resource_variable"])
        for item in graph.cross_test_results if item["status"] == "done"
    }
