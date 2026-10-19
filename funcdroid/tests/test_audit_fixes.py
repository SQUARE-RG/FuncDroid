# Multilingual UI and model-response fixtures use Unicode escapes to preserve their exact text.
"""Regressions for audit findings that silently changed coverage or defect conclusions without raising errors."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from model.flow_graph import FunctionalFlowGraph, FunctionalityNode, Page, operation_is_failure
from tools.utils import encode_image_to_jpg_base64
from workflow.bug_dedupe import bug_identity_key, dedupe_bugs
from workflow.trace_defect_detection import (
    _bug_has_required_trace_evidence,
    _build_full_trace_messages,
    _filter_trace,
)

TEST_SCREENSHOT = encode_image_to_jpg_base64(np.full((32, 32, 3), 180, dtype=np.uint8))


# ---------------------------------------------------------------------------
# 1. Failure recognition must accept the closing bracket in black-screen diagnostics.
# ---------------------------------------------------------------------------

def test_failure_pattern_accepts_trailing_bracket():
    """Recognize black-screen/unavailable-screenshot failures even with a closing bracket."""
    assert operation_is_failure("\u6253\u5f00\u8bbe\u7f6e（\u9ed1\u5c4f，\u622a\u56fe\u5065\u5eb7\u68c0\u67e5\u5931\u8d25）") is True
    assert operation_is_failure("\u8f93\u5165\u6587\u672c（\u5931\u8d25）") is True
    assert operation_is_failure(
        "\u70b9\u51fb\u5371\u9669\u5165\u53e3；\u7ed3\u679c：\u9ed1\u5c4f\u6216\u9875\u9762\u4e0d\u53ef\u7528（\u622a\u56fe\u5065\u5eb7\u68c0\u67e5\u5931\u8d25）"
    ) is True
    # Relaxing the suffix rule must still recognize text describing a failed result.
    assert operation_is_failure("\u70b9\u51fb\u5931\u8d25\u63d0\u793a\u5f39\u7a97") is False
    assert operation_is_failure("\u5931\u8d25\u91cd\u8bd5\u6309\u94ae") is False
    assert operation_is_failure("\u70b9\u51fb\u63d0\u4ea4（\u6210\u529f）") is False


def test_failed_black_screen_operation_is_not_counted_as_coverage():
    """A failed black-screen operation provides neither control coverage nor successful-operation evidence."""
    graph = FunctionalFlowGraph()
    node = FunctionalityNode(id="func_1", name="\u5e94\u7528\u8bbe\u7f6e", test_status="in_progress")
    node.add_scenario("\u63a2\u7d22\u8bbe\u7f6e\u9875\u5165\u53e3")
    graph.add_node(node)
    page = Page(id="page_1", name="\u8bbe\u7f6e\u9875", controls=["\u6e05\u9664\u7f13\u5b58"], operations=[])
    graph.add_page(page, "func_1")

    page.add_operation("\u6253\u5f00\u7f13\u5b58\u7ba1\u7406（\u9ed1\u5c4f，\u622a\u56fe\u5065\u5eb7\u68c0\u67e5\u5931\u8d25）")
    page.add_operation_binding("\u6253\u5f00\u7f13\u5b58\u7ba1\u7406（\u9ed1\u5c4f，\u622a\u56fe\u5065\u5eb7\u68c0\u67e5\u5931\u8d25）", "\u6e05\u9664\u7f13\u5b58")

    details = graph.function_coverage_details("func_1")
    assert details["uncovered_controls"] == ["\u6e05\u9664\u7f13\u5b58"]
    assert graph.function_has_successful_business_operation("func_1") is False


# ---------------------------------------------------------------------------
# 2. Operation-family merges must migrate bindings to preserve coverage after save/load.
# ---------------------------------------------------------------------------

def test_operation_family_merge_migrates_binding(tmp_path: Path):
    """Preserve bindings across family merges: the regression changed coverage from 100% in memory to 0% after save/load."""
    graph = FunctionalFlowGraph()
    node = FunctionalityNode(id="func_1", name="\u6211\u7684\u6536\u85cf", test_status="in_progress")
    graph.add_node(node)
    page = Page(id="page_1", name="\u6536\u85cf\u9875", controls=["\u5220\u9664\u6309\u94ae"], operations=[])
    graph.add_page(page, "func_1")

    page.add_operation("\u70b9\u51fb\u5220\u9664\u6309\u94ae")
    assert page.add_operation_binding("\u70b9\u51fb\u5220\u9664\u6309\u94ae", "\u5220\u9664\u6309\u94ae") is True
    assert graph.function_coverage_details("func_1")["coverage_percent"] == 100

    # The same family with text over 12 characters longer replaces the representative text.
    page.add_operation("\u70b9\u51fb\u5220\u9664\u6309\u94ae\u5e76\u786e\u8ba4\u5220\u9664\u8be5\u6536\u85cf\u6761\u76ee\u7acb\u5373\u751f\u6548")
    assert page.operations == ["\u70b9\u51fb\u5220\u9664\u6309\u94ae\u5e76\u786e\u8ba4\u5220\u9664\u8be5\u6536\u85cf\u6761\u76ee\u7acb\u5373\u751f\u6548"]
    assert page.operation_bindings == [
        {"operation": "\u70b9\u51fb\u5220\u9664\u6309\u94ae\u5e76\u786e\u8ba4\u5220\u9664\u8be5\u6536\u85cf\u6761\u76ee\u7acb\u5373\u751f\u6548", "control": "\u5220\u9664\u6309\u94ae"}
    ]

    before = graph.function_coverage_details("func_1")["coverage_percent"]
    graph.save(str(tmp_path / "out"))
    reloaded = FunctionalFlowGraph.load(str(tmp_path / "out"))
    after = reloaded.function_coverage_details("func_1")["coverage_percent"]
    assert before == after == 100, "\u8986\u76d6\u7387\u4e0d\u5f97\u56e0\u4e00\u6b21\u843d\u76d8/\u7eed\u8dd1\u800c\u4e0b\u8dcc"


# ---------------------------------------------------------------------------
# 3. Crash identity must remove the PID that changes between runs.
# ---------------------------------------------------------------------------

_CRASH = (
    "FATAL EXCEPTION: main\n"
    "Process: com.example.app, PID: {pid}\n"
    "java.lang.NullPointerException: Attempt to invoke virtual method\n"
    "\tat com.example.app.MainActivity.onCreate(MainActivity.java:88)"
)


def test_crash_identity_key_ignores_pid():
    first = {"type": "crash", "actual": _CRASH.format(pid=12345),
             "target_function_id": "func_1", "description": "\u6253\u5f00\u8bbe\u7f6e\u540e\u5d29\u6e83"}
    second = {"type": "crash", "actual": _CRASH.format(pid=67890),
              "target_function_id": "func_1", "description": "\u6253\u5f00\u8bbe\u7f6e\u540e\u5d29\u6e83"}
    assert bug_identity_key(first) == bug_identity_key(second)
    assert len(dedupe_bugs([first, second])) == 1


def test_different_crash_stacks_stay_separate():
    first = {"type": "crash", "actual": _CRASH.format(pid=1),
             "target_function_id": "func_1", "description": "\u5d29\u6e83"}
    other = {"type": "crash", "target_function_id": "func_1", "description": "\u5d29\u6e83",
             "actual": "FATAL EXCEPTION: main\nProcess: com.example.app, PID: 2\n"
                       "java.lang.IllegalStateException: boom\n"
                       "\tat com.example.app.Other.run(Other.java:5)"}
    assert len(dedupe_bugs([first, other])) == 2


# ---------------------------------------------------------------------------
# 4. Image numbering and evidence validation must use the same step index.
# ---------------------------------------------------------------------------

def _raw_trace() -> list[dict]:
    return [
        {
            "action_name": "click",
            "trace_phase": "test",
            "action_description": "\u70b9\u51fb\u5220\u9664\u6309\u94ae",
            "thought": "\u70b9\u51fb\u540e\u6761\u76ee\u4ecd\u5728",
            "current_page": Page(representative_screenshot=TEST_SCREENSHOT, id="page_1", name="\u6211\u7684\u6536\u85cf"),
        },
        {
            "action_name": "click",
            "trace_phase": "test",
            "is_interruption": True,
            "current_page": Page(representative_screenshot=TEST_SCREENSHOT, id="page_x", name="\u5916\u90e8\u5e94\u7528"),
        },
        {
            "action_name": "click",
            "trace_phase": "test",
            "action_description": "\u70b9\u51fb\u4fdd\u5b58\u8bbe\u7f6e",
            "thought": "\u4fdd\u5b58\u540e\u8fd4\u56de\u5217\u8868",
            "current_page": Page(representative_screenshot=TEST_SCREENSHOT, id="page_2", name="\u8bbe\u7f6e\u9875"),
        },
    ]


def test_image_ref_uses_raw_step_index():
    """Image references use raw STEP indices even when intervening steps are filtered."""
    trace = _filter_trace(_raw_trace())
    assert [step["raw_step_index"] for step in trace] == [1, 3]

    body = "\n".join(
        block["text"]
        for block in _build_full_trace_messages({"current_plan": {}}, trace)[-1].content
        if isinstance(block, dict) and block.get("type") == "text"
    )
    assert "[IMAGE_REF] BEFORE_STEP_1" in body
    assert "[IMAGE_REF] BEFORE_STEP_3" in body
    assert "[IMAGE_REF] BEFORE_STEP_2" not in body
    assert "The result of STEP 1 is normally visible in BEFORE_STEP_3" in body


def test_evidence_referencing_image_number_passes_validation():
    """Image numbers visible to the model must be valid global_step references."""
    trace = _filter_trace(_raw_trace())
    bug = {
        "evidence_category": "persisted_result_error",
        "description": "\u5220\u9664\u540e\u91cd\u65b0\u8fdb\u5165\u6761\u76ee\u4ecd\u5728",
        "expected": "\u6761\u76ee\u6d88\u5931",
        "actual": "\u6761\u76ee\u8fd8\u5728",
        "evidence": [
            {"global_step": 1, "role": "action"},
            {"global_step": 3, "role": "after"},
        ],
    }
    assert _bug_has_required_trace_evidence(bug, trace) is True

    # Sequential image numbers used before the fix are absent from the index and reject the candidate.
    stale = dict(bug, evidence=[
        {"global_step": 1, "role": "action"},
        {"global_step": 2, "role": "after"},
    ])
    assert _bug_has_required_trace_evidence(stale, trace) is False


# ---------------------------------------------------------------------------
# 5. Detection prompts must communicate the evidence structure enforced by validation.
#
# The 2026-09-16 run produced one high-confidence candidate that validation rejected.
# The rejection was accepted=0, candidates=1 because the model supplied before+after+after.
# The validator requires exactly one action for these mutation defect categories.
# Neither the old prompt nor its example communicated that action requirement.
# ---------------------------------------------------------------------------

def test_defect_prompt_states_action_role_requirement():
    from workflow.nodes.prompt import BUG_DETECT_SYSTEM_PROMPT

    assert "exactly one role=action" in BUG_DETECT_SYSTEM_PROMPT
    # The example must include action so copying its structure passes validation.
    assert '"role": "action"' in BUG_DETECT_SYSTEM_PROMPT


def _action_trace() -> list[dict]:
    return _filter_trace([
        {"action_name": "click", "trace_phase": "test",
         "action_description": "\u70b9\u51fb\u9996\u9875Tab", "thought": "\u9996\u9875\u8bed\u8a00\u680f\u663e\u793a\u82f1\u8bed⇄\u65e5\u8bed",
         "current_page": Page(representative_screenshot=TEST_SCREENSHOT, id="page_1", name="\u6587\u672c\u7ffb\u8bd1\u4e3b\u754c\u9762")},
        {"action_name": "click", "trace_phase": "test",
         "action_description": "\u70b9\u51fb\u8bed\u8a00\u5207\u6362\u680f", "thought": "\u8bed\u8a00\u9009\u62e9\u9762\u677f\u6e90\u8bed\u8a00\u4e3a\u82f1\u8bed",
         "current_page": Page(representative_screenshot=TEST_SCREENSHOT, id="page_2", name="\u9996\u9875")},
        {"action_name": "click", "trace_phase": "test",
         "action_description": "\u70b9\u51fb\u8fd4\u56de", "thought": "\u8fd4\u56de\u540e\u4ecd\u4e3a\u82f1\u8bed⇄\u65e5\u8bed",
         "current_page": Page(representative_screenshot=TEST_SCREENSHOT, id="page_3", name="\u9996\u9875")},
    ])


def _mutation_bug(evidence: list[dict]) -> dict:
    return {
        "evidence_category": "single_action_resource_mutation",
        "confidence": "high",
        "description": "\u5207\u6362Tab\u540e\u8bed\u8a00\u65b9\u5411\u88ab\u53cd\u8f6c\u4e3a\u82f1\u8bed→\u65e5\u8bed",
        "expected": "\u5207\u6362\u540e\u7ffb\u8bd1\u65b9\u5411\u4fdd\u6301\u4e00\u81f4",
        "actual": "\u65b9\u5411\u88ab\u53cd\u8f6c\u4e3a\u82f1\u8bed→\u65e5\u8bed",
        "evidence": evidence,
    }


def test_mutation_bug_without_action_role_is_rejected():
    """Reject the original before+after+after mutation evidence because it lacks an action."""
    assert _bug_has_required_trace_evidence(
        _mutation_bug([
            {"global_step": 1, "role": "before"},
            {"global_step": 2, "role": "after"},
            {"global_step": 3, "role": "after"},
        ]),
        _action_trace(),
    ) is False


def test_mutation_bug_with_action_role_is_accepted():
    assert _bug_has_required_trace_evidence(
        _mutation_bug([
            {"global_step": 1, "role": "action"},
            {"global_step": 2, "role": "after"},
            {"global_step": 3, "role": "after"},
        ]),
        _action_trace(),
    ) is True


def test_action_role_must_precede_after_observation():
    """Require action before after so a later screenshot can establish its result."""
    assert _bug_has_required_trace_evidence(
        _mutation_bug([
            {"global_step": 3, "role": "action"},
            {"global_step": 1, "role": "after"},
        ]),
        _action_trace(),
    ) is False


# ---------------------------------------------------------------------------
# 6. Page-loop detection must allow entering an item, operating, returning, and selecting another.
# Pure forward navigation cycles must still be detected.
# ---------------------------------------------------------------------------

def _cycle_step(action_name: str, page_id: str) -> dict:
    return {"action_name": action_name, "current_page": Page(id=page_id, activity="A")}


def test_enter_operate_return_pattern_is_not_a_navigation_cycle():
    """Allow ABBABB traversal: enter a detail, operate, return, then select another item. Merged detail Page nodes share page_key; a back action distinguishes traversal from a pure forward cycle."""
    from workflow.nodes.scenario_exec import _detect_page_navigation_cycle

    trace = [
        _cycle_step("click", "list"),
        _cycle_step("click", "detail"),
        _cycle_step("click", "detail"),
        _cycle_step("press_back", "list"),
        _cycle_step("click", "detail"),
        _cycle_step("click", "detail"),
    ]
    assert _detect_page_navigation_cycle(trace) == ""


def test_pure_forward_three_page_loop_is_still_detected():
    """Detect a pure forward ABCABC loop without any back action."""
    from workflow.nodes.scenario_exec import _detect_page_navigation_cycle

    trace = [
        _cycle_step("click", "A"),
        _cycle_step("click", "B"),
        _cycle_step("click", "C"),
        _cycle_step("click", "A"),
        _cycle_step("click", "B"),
        _cycle_step("click", "C"),
    ]
    assert _detect_page_navigation_cycle(trace) == "Page navigation loop detected"


def test_missing_screenshot_file_fails_fast_on_load_instead_of_corrupting_next_save(
    tmp_path: Path,
):
    """Fail during load when a referenced screenshot file is missing. Returning its filename as screenshot content previously delayed a base64 error until an unrelated save checkpoint."""
    graph = FunctionalFlowGraph()
    node = FunctionalityNode(id="func_1", name="\u6211\u7684\u6536\u85cf")
    graph.add_node(node)
    page = Page(id="page_1", name="\u6536\u85cf\u9875", representative_screenshot="aGVsbG8=")
    graph.add_page(page, "func_1")
    graph.save(str(tmp_path / "out"))

    (tmp_path / "out" / "screenshots" / "page_1_representative.jpg").unlink()

    with pytest.raises(ValueError, match="screenshot file is missing"):
        FunctionalFlowGraph.load(str(tmp_path / "out"))


def test_strict_two_page_alternation_is_still_detected_even_with_back():
    """Strict ABABAB alternation remains a loop even when back actions are present."""
    from workflow.nodes.scenario_exec import _detect_page_navigation_cycle

    trace = [
        _cycle_step("click", "A"),
        _cycle_step("press_back", "B"),
        _cycle_step("click", "A"),
        _cycle_step("press_back", "B"),
        _cycle_step("click", "A"),
        _cycle_step("press_back", "B"),
    ]
    assert _detect_page_navigation_cycle(trace) == "Page navigation loop detected"
