# Multilingual UI and model-response fixtures use Unicode escapes to preserve their exact text.
"""Regressions for the 2026-10-05 real-device false completion."""

import numpy as np
import pytest

from model.flow_graph import FunctionalFlowGraph, FunctionalityNode, Page
from tools.utils import encode_image_to_jpg_base64
from workflow.nodes.ffg_update import _record_intra_attempt
from workflow.nodes.scenario_plan import _function_completion_met, _update_function_statuses


def graph_and_state(status="completed", changed=True, **flags):
    graph = FunctionalFlowGraph()
    node = FunctionalityNode(id="f1", scenarios=["\u5f00\u542f\u8bbe\u7f6e"], test_status="in_progress")
    graph.add_node(node)
    before = np.full((400, 200, 3), 240, dtype=np.uint8)
    after = before.copy()
    if changed:
        after[100:200, 30:100] = 20
    page = Page(id="p1", controls=["\u8bbe\u7f6e\u6309\u94ae"], operations=["\u70b9\u51fb\u8bbe\u7f6e\u6309\u94ae"],
                representative_screenshot=encode_image_to_jpg_base64(before))
    page.add_operation_binding("\u70b9\u51fb\u8bbe\u7f6e\u6309\u94ae", "\u8bbe\u7f6e\u6309\u94ae")
    graph.add_page(page, "f1")
    graph.recompute_all_coverages()
    state = {
        "current_plan": {"scenario_type": "intra_test", "target_function_id": "f1", "function_phase": "depth"},
        "execution_result": {"scenario_status": status},
        "scenario_execution_trace": [
            {"action_name": "click", "current_page": page, **flags},
            {"action_name": "finished", "current_page": Page(representative_screenshot=encode_image_to_jpg_base64(after))},
        ],
    }
    return graph, node, state


@pytest.mark.parametrize("status,changed,flags", [
    ("stalled", True, {}), ("incomplete", True, {}), ("completed", False, {}),
    ("completed", True, {"action_failed": True}),
    ("completed", True, {"is_interruption": True}),
    ("completed", True, {"external_context": True}),
])
def test_attempt_or_unobserved_response_cannot_complete_function(status, changed, flags):
    graph, node, state = graph_and_state(status, changed, **flags)
    _record_intra_attempt(graph, state)
    assert node.depth_scenario_count == 1
    assert node.successful_depth_scenario_count == 0
    assert not _function_completion_met(graph, node)


def test_observed_response_completes_and_survives_copy_and_reload():
    graph, node, state = graph_and_state()
    _record_intra_attempt(graph, state)
    assert node.successful_depth_scenario_count == 1
    assert _function_completion_met(graph, node)
    for restored in (graph.clone(), FunctionalFlowGraph.from_dict(graph.to_dict())):
        assert restored.get_node("f1").successful_depth_scenario_count == 1
        assert _function_completion_met(restored, restored.get_node("f1"))


def test_old_completed_snapshot_without_success_evidence_reopens():
    graph, node, state = graph_and_state(status="stalled")
    _record_intra_attempt(graph, state)
    node.test_status = "completed"
    _update_function_statuses(graph)
    assert node.test_status == "in_progress"
    node.intra_attempt_count = 100
    node.test_status = "completed"
    _update_function_statuses(graph)
    assert node.test_status == "deferred"
