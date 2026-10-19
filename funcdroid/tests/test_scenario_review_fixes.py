# Multilingual UI and model-response fixtures use Unicode escapes to preserve their exact text.
"""Scenario planning and execution review regressions."""

from __future__ import annotations

import base64
import json
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from model.flow_graph import FunctionalFlowGraph, FunctionalityNode, Page, PageTransition
from workflow import app_foreground
from workflow import exploration_boundary as boundary
from workflow.account_safety import account_session_action_violation, account_session_violation
from workflow.nodes import _shared
from workflow.nodes import scenario_exec as executor
from workflow.nodes import scenario_plan as planner
from workflow.nodes.prompt import (
    SCENARIO_EXEC_SYSTEM_PROMPT_TEMPLATE,
    UNIFIED_SCENARIO_PLAN_SYSTEM_PROMPT,
)
from workflow.wechat_safety import wechat_read_only_violation


def _screenshot(value: int, size: tuple[int, int] = (48, 32)) -> str:
    image = np.full((*size, 3), value, dtype=np.uint8)
    success, buffer = cv2.imencode(".png", image)
    assert success
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


# 1. Explicit finished-result prefixes take precedence.

@pytest.mark.parametrize("thought", [
    "\u76ee\u6807\u4e0d\u53ef\u8fbe：\u672a\u627e\u5230\u5bfc\u51fa\u5165\u53e3，\u5df2\u5b8c\u6210\u6240\u6709\u53ef\u89c1\u63a7\u4ef6\u68c0\u67e5",
    "【\u76ee\u6807\u4e0d\u53ef\u8fbe】\u672a\u627e\u5230\u5bfc\u51fa\u5165\u53e3，\u5df2\u5b8c\u6210\u5168\u90e8\u68c0\u67e5",
    "[\u76ee\u6807\u4e0d\u53ef\u8fbe] \u529f\u80fd\u6b63\u5e38\u4f46\u5165\u53e3\u7f3a\u5931",
    "  \u76ee\u6807\u4e0d\u53ef\u8fbe：\u6240\u6709\u9a8c\u8bc1\u70b9\u5747\u5df2\u6ee1\u8db3\u4e4b\u5916\u7684\u5165\u53e3\u4e0d\u5b58\u5728",
])
def test_finished_unreachable_prefix_wins_over_success_words(thought):
    assert executor._finished_declares_failure(thought) is True


@pytest.mark.parametrize("thought", [
    "\u76ee\u6807\u5df2\u8fbe\u6210：\u5df2\u5b8c\u6210\u6240\u6709\u63a7\u4ef6\u68c0\u67e5",
    "【\u76ee\u6807\u5df2\u8fbe\u6210】\u51fa\u73b0 CAN'T SAVE \u63d0\u793a\u4f46\u7ed3\u679c\u7b26\u5408\u9884\u671f",
])
def test_finished_success_prefix_is_success(thought):
    assert executor._finished_declares_failure(thought) is False


def test_finished_success_prefix_with_unreachable_subgoal_is_still_failure():
    assert executor._finished_declares_failure("\u76ee\u6807\u5df2\u8fbe\u6210；\u4f46\u6587\u4ef6\u53d1\u9001\u5165\u53e3\u4e0d\u53ef\u8fbe，\u5269\u4f59\u9a8c\u8bc1\u672a\u5b8c\u6210。")


# 2. Exempt enter-return traversal from two-page alternation detection.

def _cycle(action: str, page_id: str, point: tuple[int, int] | None = None) -> dict:
    step = {"action_name": action, "current_page": Page(id=page_id, activity="A")}
    if point is not None:
        step["model_coordinates"] = {"x": point[0], "y": point[1]}
        step["x"], step["y"] = point
    return step


def test_enter_return_over_distinct_controls_is_not_a_loop():
    trace = [
        _cycle("click", "list", (100, 200)),
        _cycle("press_back", "detail"),
        _cycle("click", "list", (100, 500)),
        _cycle("press_back", "detail"),
        _cycle("click", "list", (100, 800)),
        _cycle("press_back", "detail"),
    ]
    assert executor._detect_page_navigation_cycle(trace) == ""


def test_enter_return_over_same_control_is_still_a_loop():
    trace = [
        _cycle("click", "list", (100, 200)),
        _cycle("press_back", "detail"),
        _cycle("click", "list", (105, 210)),
        _cycle("press_back", "detail"),
        _cycle("click", "list", (98, 195)),
        _cycle("press_back", "detail"),
    ]
    assert executor._detect_page_navigation_cycle(trace) == "Page navigation loop detected"


def test_two_page_alternation_without_back_is_still_a_loop():
    trace = [
        _cycle("click", "A", (100, 200)),
        _cycle("click", "B", (100, 500)),
        _cycle("click", "A", (100, 800)),
        _cycle("click", "B", (300, 500)),
        _cycle("click", "A", (500, 800)),
        _cycle("click", "B", (700, 500)),
    ]
    assert executor._detect_page_navigation_cycle(trace) == "Page navigation loop detected"


# 3. Recent exception history.

def test_recent_failure_note_is_line_joined():
    note = executor._recent_failure_note([
        {"step": 2, "reason": "\u539f\u56e0\u7532"},
        {"step": 2, "attempt": 2, "reason": "\u539f\u56e0\u4e59"},
    ])
    assert "['" not in note and "[\"" not in note
    assert "- Step 2 generation failed: \u539f\u56e0\u7532\n- Step 2 (attempt 2) generation failed: \u539f\u56e0\u4e59" in note


def test_transport_failure_is_not_described_as_plain_text_response():
    reason = executor._describe_generation_failure("", TimeoutError("read timeout"), response_received=False)
    assert "Model call failed (transport/timeout), no response received" in reason
    assert "prose without JSON" not in reason
    assert "prose without JSON" in executor._describe_generation_failure("\u70b9\u51fb\u6309\u94ae", ValueError("x"))


def _run_scenario_with_replies(monkeypatch, replies, *, page=None, deadline=None):
    shot = _screenshot(120, (64, 32))
    page = page or Page(id="page_1", activity="MainActivity", representative_screenshot=shot)
    reply_iter = iter(replies)
    messages_seen: list[list] = []

    class Llm:
        def invoke(self, messages):
            messages_seen.append(messages)
            reply = next(reply_iter)
            if isinstance(reply, Exception):
                raise reply
            return SimpleNamespace(content=reply)

    device = SimpleNamespace(display_size=lambda: (320, 640), current_package=lambda: "com.example")
    monkeypatch.setattr(executor, "_ensure_target_app", lambda *a: True)
    monkeypatch.setattr(executor, "_target_process_pid", lambda *a: "")
    monkeypatch.setattr(executor, "_target_app_foreground_state", lambda *a: ("foreground", "com.example"))
    monkeypatch.setattr(executor, "capture_runtime_page", lambda *a, **kw: (page, ""))
    monkeypatch.setattr(executor, "capture_current_page", lambda *a, **kw: page)
    monkeypatch.setattr(executor, "save_trace_screenshot", lambda *a, **kw: "")
    monkeypatch.setattr(executor, "_perform_action", lambda **kw: None)
    monkeypatch.setattr(executor.time, "sleep", lambda *_a: None)
    result = executor._run_single_scenario(
        Llm(), device, "App", "com.example", "\u70b9\u51fb\u8bbe\u7f6e", page, False, "test", deadline=deadline,
    )
    return result, messages_seen


def _step_text(messages) -> str:
    return "\n".join(
        block["text"] for block in messages[-1].content if block.get("type") == "text"
    )


def test_recent_failures_cleared_after_successful_step(monkeypatch):
    replies = [
        "\u6211\u8981\u70b9\u51fb\u8bbe\u7f6e\u6309\u94ae",
        json.dumps({"action_name": "click", "x": 10, "y": 10, "thought": "\u70b9\u8bbe\u7f6e"}),
        json.dumps({"action_name": "finished", "thought": "\u76ee\u6807\u5df2\u8fbe\u6210"}),
    ]
    (trace, _page, status, error, _crash), messages = _run_scenario_with_replies(monkeypatch, replies)
    assert status == "completed", error
    assert "Recent failures" in _step_text(messages[1])
    assert "prose without JSON" in _step_text(messages[1])
    # A successful step clears the old failure from subsequent history.
    assert "Recent failures" not in _step_text(messages[2])


def test_transport_failure_recorded_as_transport(monkeypatch):
    replies = [
        TimeoutError("gateway timeout"),
        json.dumps({"action_name": "finished", "thought": "\u76ee\u6807\u5df2\u8fbe\u6210"}),
    ]
    (_trace, _page, status, error, _crash), messages = _run_scenario_with_replies(monkeypatch, replies)
    assert status == "completed", error
    text = _step_text(messages[1])
    assert "Model call failed (transport/timeout), no response received" in text
    assert "prose without JSON" not in text


def test_response_format_rejection_is_not_recorded_as_model_failure(monkeypatch):
    monkeypatch.setattr(executor, "json_format_rejected", lambda e: "response_format" in str(e))
    monkeypatch.setattr(executor, "disable_json_response_format", lambda llm: True)
    replies = [
        RuntimeError("response_format not supported"),
        json.dumps({"action_name": "finished", "thought": "\u76ee\u6807\u5df2\u8fbe\u6210"}),
    ]
    (_trace, _page, status, error, _crash), messages = _run_scenario_with_replies(monkeypatch, replies)
    assert status == "completed", error
    assert "Recent failures" not in _step_text(messages[1])


def test_llm_retry_stops_when_deadline_is_too_close(monkeypatch):
    monkeypatch.setattr(executor, "compute_retry_delay_seconds", lambda _n: 30.0)
    replies = ["prose without JSON", "prose without JSON"]
    (_trace, _page, status, error, _crash), messages = _run_scenario_with_replies(
        monkeypatch, replies, deadline=time.time() + 5,
    )
    assert status == "incomplete"
    assert "insufficient" in error.lower()
    # A 30-second backoff exceeds the 5 seconds remaining: neither sleep nor call the model again.
    assert len(messages) == 1


# 4. EXEC_KEEP_IMAGE_STEPS=0 must not send the entire history.

def test_keep_image_steps_zero_sends_no_history(monkeypatch):
    monkeypatch.setattr(executor, "KEEP_IMAGE_STEPS", 0)
    shot = _screenshot(80)
    trace = [
        {"action_name": "click", "current_page": Page(id=f"p{i}", representative_screenshot=shot)}
        for i in range(5)
    ]
    content = executor._build_step_input(trace, shot, 6, 15, image_size=(32, 48))
    images = [block for block in content if block["type"] == "image_url"]
    assert len(images) == 1  # Only the current screenshot.
    assert executor._recent_screenshot_count(trace) == 0
    assert "None attached" in content[0]["text"]
    assert "Latest 0 historical screenshots" not in content[0]["text"]


# 5. Process fallback semantics at exploration boundaries.

@pytest.fixture()
def wechat_boundary_rules(tmp_path, monkeypatch):
    rules = tmp_path / "rules.json"
    rules.write_text(json.dumps({
        "com.tencent.mm": {
            "enabled": True,
            "activity_markers": ["appbrand"],
            "process_markers": [":appbrand"],
        }
    }), encoding="utf-8")
    monkeypatch.setenv("BOUNDARY_RULES_FILE", str(rules))
    boundary.reset_cache()
    yield
    boundary.reset_cache()


class _BoundaryDevice:
    def __init__(self, foreground="com.tencent.mm"):
        self.ps_calls = 0
        self.foreground = foreground
        self.connector = SimpleNamespace(shell=self._shell)

    def _shell(self, command):
        assert command == "ps -A"
        self.ps_calls += 1
        return "u0_a12 1234 com.tencent.mm:appbrand0"

    def current_package(self):
        return self.foreground


def test_resident_appbrand_process_does_not_hit_when_activity_known(wechat_boundary_rules):
    device = _BoundaryDevice()
    assert executor._current_boundary_hit(
        device, "com.tencent.mm", "com.tencent.mm.ui.LauncherUI", "com.tencent.mm",
    ) is None
    assert device.ps_calls == 0


def test_process_fallback_only_when_activity_unreadable_and_target_foreground(wechat_boundary_rules):
    device = _BoundaryDevice()
    hit = executor._current_boundary_hit(device, "com.tencent.mm", "", "com.tencent.mm")
    assert hit is not None and hit.via == "process"
    other = _BoundaryDevice(foreground="com.other")
    assert executor._current_boundary_hit(other, "com.tencent.mm", "", "com.other") is None
    assert other.ps_calls == 0


def test_activity_hit_still_detected(wechat_boundary_rules):
    hit = executor._current_boundary_hit(
        _BoundaryDevice(), "com.tencent.mm", "com.tencent.mm.plugin.appbrand.ui.AppBrandUI", "com.tencent.mm",
    )
    assert hit is not None and hit.via == "activity"


# 7. Session guards inspect actions and skip negated wording.

@pytest.mark.parametrize("text", [
    "\u4fdd\u6301\u767b\u5f55\u6001，\u4e0d\u70b9\u9000\u51fa\u767b\u5f55，\u70b9\u51fb\u9690\u79c1\u8bbe\u7f6e",
    "\u907f\u514d\u5207\u6362\u8d26\u53f7",
    "\u7981\u6b62\u70b9\u51fb\u767b\u51fa",
    "\u4e0d\u8981\u70b9\u51fb“\u9000\u51fa\u767b\u5f55”",
    "\u4e0d\u5f97\u6267\u884c\u6ce8\u9500\u8d26\u53f7、\u9000\u51fa\u767b\u5f55、\u5207\u6362\u8d26\u53f7",
])
def test_negated_session_phrases_are_not_blocked(text):
    assert not account_session_violation(text)


@pytest.mark.parametrize("text", ["\u70b9\u51fb\u9000\u51fa\u767b\u5f55", "\u4e0d\u70b9\u8bbe\u7f6e，\u70b9\u51fb\u9000\u51fa\u767b\u5f55", "tap Log out"])
def test_affirmative_session_phrases_are_blocked(text):
    assert account_session_violation(text)


def test_wechat_guard_also_skips_negated_mentions():
    assert not wechat_read_only_violation("com.tencent.mm", "\u4e0d\u5f97\u6267\u884c\u6ce8\u9500\u8d26\u53f7、\u9000\u51fa\u767b\u5f55")
    assert wechat_read_only_violation("com.tencent.mm", "\u70b9\u51fb\u9000\u51fa\u767b\u5f55")


def test_executor_session_guard_ignores_thought_constraints(monkeypatch):
    replies = [
        json.dumps({
            "action_name": "click", "x": 10, "y": 10,
            "thought": "\u4fdd\u6301\u767b\u5f55\u6001，\u4e0d\u9000\u51fa\u767b\u5f55，\u70b9\u51fb\u9690\u79c1\u8bbe\u7f6e",
            "action_description": "\u70b9\u51fb\u9690\u79c1\u8bbe\u7f6e",
        }),
        json.dumps({"action_name": "finished", "thought": "\u76ee\u6807\u5df2\u8fbe\u6210"}),
    ]
    (trace, _page, status, error, _crash), _messages = _run_scenario_with_replies(monkeypatch, replies)
    assert status == "completed", error
    assert not trace[0].get("session_guard_blocked")


def test_executor_session_guard_blocks_logout_action(monkeypatch):
    replies = [json.dumps({
        "action_name": "click", "x": 10, "y": 10,
        "thought": "\u9700\u8981\u9000\u51fa\u767b\u5f55", "action_description": "\u70b9\u51fb\u9000\u51fa\u767b\u5f55\u6309\u94ae",
    })]
    (trace, _page, status, _error, _crash), _messages = _run_scenario_with_replies(monkeypatch, replies)
    assert status == "failed_goal"
    assert trace[0]["session_guard_blocked"] is True
    assert account_session_action_violation("click", "\u70b9\u51fb\u9000\u51fa\u767b\u5f55\u6309\u94ae")


# 8. Parse the current page once per planning iteration; BFS uses adjacency lists.

def _chain_graph(count: int = 4) -> FunctionalFlowGraph:
    graph = FunctionalFlowGraph()
    for index in range(1, count + 1):
        graph.add_node(FunctionalityNode(id=f"func_{index}", name=f"\u529f\u80fd{index}"))
        graph.add_page(Page(id=f"page_{index}", name=f"\u9875\u9762{index}"), f"func_{index}")
    for index in range(1, count):
        graph.add_page_transition(PageTransition(f"page_{index}", f"page_{index + 1}", f"\u8fdb\u5165\u9875\u9762{index + 1}"))
    return graph


def test_select_intra_target_resolves_current_page_once(monkeypatch):
    graph = _chain_graph()
    calls = []
    original = planner._resolve_current_page_id
    monkeypatch.setattr(
        planner, "_resolve_current_page_id",
        lambda g, p: calls.append(1) or original(g, p),
    )
    planner._select_intra_target(graph, graph.get_page("page_1"), {})
    assert len(calls) == 1


def test_backtrack_bfs_uses_reverse_adjacency():
    graph = _chain_graph()
    path = planner._leveled_page_path(graph, "page_4", ["page_1"], allow_backtrack=True)
    assert [(t.src_page_id, reversed_edge) for t, reversed_edge in path] == [
        ("page_3", True), ("page_2", True), ("page_1", True),
    ]


def test_screenshot_thumbnail_is_cached_by_content(monkeypatch):
    calls = []
    real = _shared._decode_thumbnail
    monkeypatch.setattr(_shared, "_decode_thumbnail", lambda s, n: calls.append(1) or real(s, n))
    _shared._THUMBNAIL_CACHE.clear()
    shot = _screenshot(33)
    first = _shared.screenshot_thumbnail(shot)
    second = _shared.screenshot_thumbnail(str(shot))
    assert first is second and len(calls) == 1


def test_bounded_cache_evicts_oldest():
    cache = _shared.BoundedCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")
    cache.put("c", 3)
    assert "a" in cache and "c" in cache and "b" not in cache


# 9. Cache action effects per screenshot pair without mutating steps.

def test_step_effects_reuse_cached_comparisons(monkeypatch):
    calls = []
    real = executor._action_effect
    monkeypatch.setattr(executor, "_action_effect", lambda *a: calls.append(1) or real(*a))
    executor._ACTION_EFFECT_CACHE.clear()
    trace = [
        {"action_name": "click", "current_page": Page(representative_screenshot=_screenshot(10 + i * 40))}
        for i in range(3)
    ]
    current = _screenshot(200)
    executor._step_effects(trace, current, (32, 48))
    assert len(calls) == 3
    executor._step_effects(trace, current, (32, 48))
    assert len(calls) == 3
    assert all(not any(key.startswith("_") for key in step) for step in trace)


# 10. Bound planning prompt size.

def test_scope_summary_keeps_only_recent_scenarios(monkeypatch):
    monkeypatch.setattr(planner, "PLAN_SCOPE_RECENT_SCENARIOS", 3)
    graph = _chain_graph(2)
    node = graph.get_node("func_1")
    for index in range(10):
        node.add_scenario(f"\u5386\u53f2\u573a\u666f{index:02d}")
    summary, _nav, _level = planner._build_scope_planning_summary(
        graph, graph.get_page("page_1"), "intra_test", target=node,
    )
    assert "\u5386\u53f2\u573a\u666f09" in summary and "\u5386\u53f2\u573a\u666f07" in summary
    assert "\u5386\u53f2\u573a\u666f06" not in summary
    assert "7 earlier tested scenarios" in summary


def test_scope_summary_lists_only_adjacent_transitions():
    graph = _chain_graph(5)
    summary, _nav, _level = planner._build_scope_planning_summary(
        graph, graph.get_page("page_1"), "intra_test", target=graph.get_node("func_1"),
    )
    paths = summary.split("[Related page paths]", 1)[1]
    assert "page_1" in paths and "page_2" in paths
    assert "page_4（" not in paths


def test_plan_validation_retries_default_to_two(monkeypatch):
    import importlib

    monkeypatch.delenv("PLAN_VALIDATION_RETRY_ATTEMPTS", raising=False)
    source = importlib.util.find_spec(planner.__name__).origin
    text = open(source, encoding="utf-8").read()
    assert 'os.getenv("PLAN_VALIDATION_RETRY_ATTEMPTS", "2")' in text


# 11. Per-iteration coverage-audit limit.

def test_completion_audit_caps_functions_per_round(monkeypatch):
    graph = FunctionalFlowGraph()
    for index in range(1, 5):
        graph.add_node(FunctionalityNode(id=f"func_{index}", name=f"\u529f\u80fd{index}", test_status="completed"))
        graph.add_page(Page(id=f"page_{index}", name=f"\u9875\u9762{index}"), f"func_{index}")
    audited = []
    monkeypatch.setattr(
        planner, '_audit_functionality_completeness',
        lambda _llm, _g, node, *_a, **_k: audited.append(node.id) or {"entry_page_ref": "", "entry_name": ""},
    )
    result, complete, signatures = planner._run_completion_audit(object(), graph, "core", max_audits=2)
    assert result is None and complete is False
    assert audited == ["func_1", "func_2"] and set(signatures) == {"func_1", "func_2"}
    result, complete, more = planner._run_completion_audit(object(), graph, "core", signatures, max_audits=2)
    assert complete is True and audited[2:] == ["func_3", "func_4"]


# 12. Compliance priority must not override failure yielding.

def test_compliance_relevant_failed_function_does_not_beat_healthy_candidate():
    graph = FunctionalFlowGraph()
    graph.add_node(FunctionalityNode(id="func_1", name="\u9690\u79c1\u8bbe\u7f6e", test_status="in_progress"))
    graph.add_node(FunctionalityNode(id="func_2", name="\u5185\u5bb9\u6d4f\u89c8", test_status="in_progress"))
    graph.add_page(Page(id="page_1", name="\u9690\u79c1\u8bbe\u7f6e\u9875"), "func_1")
    graph.add_page(Page(id="page_2", name="\u5185\u5bb9\u9875"), "func_2")
    selected = planner._select_intra_target(graph, graph.get_page("page_1"), {"func_1": 1}, ("\u9690\u79c1",))
    assert selected.id == "func_2"


# 13. Screenshot failures use foreground polling and a separate model-failure counter.

def test_screenshot_failure_uses_polling_restart(monkeypatch):
    restarts = []
    monkeypatch.setattr(executor, "_ensure_target_app", lambda *a: True)
    monkeypatch.setattr(executor, "_target_process_pid", lambda *a: "")
    monkeypatch.setattr(executor, "_target_app_foreground_state", lambda *a: ("foreground", "com.example"))
    monkeypatch.setattr(executor, "capture_runtime_page", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("down")))
    monkeypatch.setattr(executor, "_restart_target_app", lambda *a: restarts.append(a) or True)
    monkeypatch.setattr(executor, "_append_terminal_observation", lambda *a, **kw: (None, ""))
    monkeypatch.setattr(executor.time, "sleep", lambda *_a: None)
    device = SimpleNamespace(restart_app=lambda *_a: pytest.fail("must use _restart_target_app"))
    _trace, _page, status, _error, _crash = executor._run_single_scenario(
        SimpleNamespace(invoke=lambda *_a: pytest.fail("llm must not run")),
        device, "App", "com.example", "\u70b9\u51fb\u8bbe\u7f6e", None, False, "test",
    )
    assert status == "error"
    assert len(restarts) == executor.APP_RESTART_ATTEMPTS


def test_screenshot_restart_skipped_when_deadline_too_close(monkeypatch):
    monkeypatch.setattr(executor, "_ensure_target_app", lambda *a: True)
    monkeypatch.setattr(executor, "_target_process_pid", lambda *a: "")
    monkeypatch.setattr(executor, "_target_app_foreground_state", lambda *a: ("foreground", "com.example"))
    monkeypatch.setattr(executor, "capture_runtime_page", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("down")))
    monkeypatch.setattr(executor, "_restart_target_app", lambda *a: pytest.fail("no time to restart"))
    monkeypatch.setattr(executor, "_append_terminal_observation", lambda *a, **kw: (None, ""))
    monkeypatch.setattr(executor.time, "sleep", lambda *_a: None)
    _trace, _page, status, error, _crash = executor._run_single_scenario(
        object(), SimpleNamespace(), "App", "com.example", "\u70b9\u51fb\u8bbe\u7f6e", None, False, "test",
        deadline=time.time() + 1.0,
    )
    assert status == "incomplete"
    assert "insufficient" in error.lower()


def test_recover_by_back_queries_foreground_once_before_launcher_check(monkeypatch):
    calls = []
    monkeypatch.setattr(
        executor, "_target_app_foreground_state",
        lambda *a: calls.append(1) or ("background", "com.android.launcher3"),
    )
    assert executor._recover_target_app_by_back(object(), "com.example") is False
    assert len(calls) == 1


def test_crash_logcat_is_parsed_once_when_leaving_app(monkeypatch):
    calls = []
    real = executor._logcat_has_target_process_crash
    monkeypatch.setattr(
        executor, "_logcat_has_target_process_crash",
        lambda *a: calls.append(1) or real(*a),
    )
    monkeypatch.setattr(executor, "_ensure_target_app", lambda *a: True)
    monkeypatch.setattr(executor, "_target_process_pid", lambda *a: "")
    monkeypatch.setattr(executor, "_target_app_foreground_state", lambda *a: ("background", "com.other"))
    monkeypatch.setattr(executor, "_read_crash_logcat", lambda *a: "")
    monkeypatch.setattr(executor, "_recover_target_app_by_back", lambda *a, **kw: False)
    monkeypatch.setattr(executor, "_restart_target_app", lambda *a: False)
    monkeypatch.setattr(executor.time, "sleep", lambda *_a: None)
    executor._run_single_scenario(
        object(), object(), "App", "com.example", "\u70b9\u51fb\u8bbe\u7f6e", None, False, "test",
    )
    # One foreground loss inside the loop plus finalization; failed left_app recovery is not rechecked.
    assert len(calls) == 1


# 14. Shared foreground logic.

def test_foreground_constants_shared_between_nodes():
    for name in ("APP_START_MAX_WAIT", "APP_START_POLL_INTERVAL", "APP_RESTART_SETTLE_DELAY", "APP_RESTART_RETRY_DELAY"):
        assert getattr(executor, name) == getattr(app_foreground, name)
        assert getattr(planner, name) == getattr(app_foreground, name)
    assert executor.JSON_BLOCK_PATTERN is _shared.JSON_BLOCK_PATTERN


def test_deadline_allows_helper():
    assert app_foreground.deadline_allows(None, 999)
    assert not app_foreground.deadline_allows(time.time() + 1, 5)
    assert app_foreground.deadline_allows(time.time() + 60, 5)


# 6. Consistent prompt rules.

def test_executor_prompt_list_governs_and_allows_requested_reopen():
    assert "A named control checklist is authoritative" in SCENARIO_EXEC_SYSTEM_PROMPT_TEMPLATE
    assert "navigation explicitly requests reopening" in SCENARIO_EXEC_SYSTEM_PROMPT_TEMPLATE
    assert "A named checklist is completed item by item" in UNIFIED_SCENARIO_PLAN_SYSTEM_PROMPT
    assert "A same-result-domain checklist is the stated exception" in UNIFIED_SCENARIO_PLAN_SYSTEM_PROMPT
    assert planner.RESTART_NAVIGATION_INSTRUCTION.startswith("Execute open_app")


# Device regression 2026-10-05: identical coordinate retries after missing a small control.

def test_failed_tap_is_marked_on_history_screenshot():
    import numpy as np

    from tools.utils import decode_screenshot, encode_image_to_jpg_base64
    from workflow.nodes import scenario_exec as exec_module

    # Other tests reload model.flow_graph; use the executor's Page class for consistent isinstance checks.
    Page = exec_module.Page

    image = np.full((400, 200, 3), 230, dtype=np.uint8)
    shot = encode_image_to_jpg_base64(image)
    trace = [{
        "action_name": "click",
        "model_coordinates": {"x": 100, "y": 300},
        "current_page": Page(representative_screenshot=shot),
    }]
    content = exec_module._build_step_input(trace, shot, 2, 10, image_size=(200, 400))
    texts = " ".join(block.get("text", "") for block in content if block.get("type") == "text")
    assert "red crosshair" in texts and "(100,300)" in texts
    history_url = [block for block in content if block.get("type") == "image_url"][0]["image_url"]["url"]
    assert history_url.startswith("data:image/jpeg;base64,")
    marked = decode_screenshot(history_url.split(",", 1)[1])
    b, g, r = (int(v) for v in marked[300, 100 + 12])
    assert r > 150 and g < 120 and b < 120


def test_effective_tap_history_screenshot_is_not_marked():
    import numpy as np

    from tools.utils import encode_image_to_jpg_base64
    from workflow.nodes import scenario_exec as exec_module

    Page = exec_module.Page

    before = np.full((400, 200, 3), 230, dtype=np.uint8)
    after = before.copy()
    after[100:300, 20:180] = 20
    trace = [{
        "action_name": "click",
        "model_coordinates": {"x": 100, "y": 300},
        "current_page": Page(representative_screenshot=encode_image_to_jpg_base64(before)),
    }]
    content = exec_module._build_step_input(
        trace, encode_image_to_jpg_base64(after), 2, 10, image_size=(200, 400),
    )
    texts = " ".join(block.get("text", "") for block in content if block.get("type") == "text")
    assert "red crosshair" not in texts


# Device regression 2026-10-05: repeating a control action that already succeeded.

def _click_trace_pair_same_point():
    """Two taps at the same point both change the screen; the second repeats an already successful action."""
    import numpy as np

    from tools.utils import encode_image_to_jpg_base64
    from workflow.nodes import scenario_exec as exec_module

    Page = exec_module.Page
    base = np.full((400, 200, 3), 230, dtype=np.uint8)
    changed = base.copy()
    changed[100:300, 20:180] = 20
    changed2 = base.copy()
    changed2[100:300, 20:180] = 60
    point = {"x": 100, "y": 300}
    return [
        {"action_name": "click", "model_coordinates": dict(point),
         "current_page": Page(representative_screenshot=encode_image_to_jpg_base64(base))},
        {"action_name": "click", "model_coordinates": dict(point),
         "current_page": Page(representative_screenshot=encode_image_to_jpg_base64(changed))},
    ], encode_image_to_jpg_base64(changed2)


def test_repeat_of_successful_action_gets_reminder():
    from workflow.nodes import scenario_exec as exec_module

    trace, current = _click_trace_pair_same_point()
    content = exec_module._build_step_input(trace, current, 3, 10, image_size=(200, 400))
    texts = " ".join(block.get("text", "") for block in content if block.get("type") == "text")
    assert "[Already executed]" in texts


def test_first_effective_action_gets_no_repeat_reminder():
    from workflow.nodes import scenario_exec as exec_module

    trace, current = _click_trace_pair_same_point()
    # Keep only the first tap: a successful action with no prior equivalent should not warn.
    content = exec_module._build_step_input(trace[:1], current, 2, 10, image_size=(200, 400))
    texts = " ".join(block.get("text", "") for block in content if block.get("type") == "text")
    assert "[Already executed]" not in texts


def test_repeat_at_different_point_gets_no_reminder():
    from workflow.nodes import scenario_exec as exec_module

    trace, current = _click_trace_pair_same_point()
    # Moving the second tap to another control is not a repeated successful action.
    trace[1]["model_coordinates"] = {"x": 100, "y": 60}
    content = exec_module._build_step_input(trace, current, 3, 10, image_size=(200, 400))
    texts = " ".join(block.get("text", "") for block in content if block.get("type") == "text")
    assert "[Already executed]" not in texts


# Tap refinement triggers on a uniform background at the proposed point.

def _flat_screen_with_button():
    import numpy as np

    from tools.utils import encode_image_to_jpg_base64

    image = np.full((800, 400, 3), 240, dtype=np.uint8)
    image[500:540, 150:250] = (200, 120, 30)  # Button body.
    return encode_image_to_jpg_base64(image)


class _RefineLlm:
    def __init__(self, content):
        self.content = content
        self.calls = 0

    def invoke(self, _messages):
        self.calls += 1
        return type("R", (), {"content": self.content})()


def test_tap_on_flat_background_is_refined_onto_control(monkeypatch):
    from types import SimpleNamespace

    from workflow.nodes import scenario_exec as exec_module

    monkeypatch.setattr(exec_module, "TAP_REFINE_ENABLED", True)
    shot = _flat_screen_with_button()
    # The original point (200,470) lies on the blank background above the button.
    llm = _RefineLlm('{"found": true, "x": 360, "y": 410}')
    plan = {"action_name": "click", "x": 200, "y": 470, "thought": "\u70b9\u51fb\u6309\u94ae"}
    obs = SimpleNamespace(page=exec_module.Page(representative_screenshot=shot), external_context=False)
    exec_module._maybe_refine_tap(SimpleNamespace(llm=llm, trace=[]), plan, obs)
    assert llm.calls == 1
    # Crop origin (20,290), scale 2: (20+180,290+205)=(200,495).
    assert (plan["x"], plan["y"]) == (200, 495)
    assert plan["tap_refined_from"] == {"x": 200, "y": 470}


def test_tap_on_control_skips_refinement_call(monkeypatch):
    from types import SimpleNamespace

    from workflow.nodes import scenario_exec as exec_module

    monkeypatch.setattr(exec_module, "TAP_REFINE_ENABLED", True)
    shot = _flat_screen_with_button()
    llm = _RefineLlm('{"found": true, "x": 0, "y": 0}')
    plan = {"action_name": "click", "x": 200, "y": 540, "thought": "\u70b9\u51fb\u6309\u94ae"}
    obs = SimpleNamespace(page=exec_module.Page(representative_screenshot=shot), external_context=False)
    exec_module._maybe_refine_tap(SimpleNamespace(llm=llm, trace=[]), plan, obs)
    assert llm.calls == 0
    assert (plan["x"], plan["y"]) == (200, 540)


def test_refinement_keeps_original_on_not_found_or_far_jump(monkeypatch):
    from types import SimpleNamespace

    from workflow.nodes import scenario_exec as exec_module

    monkeypatch.setattr(exec_module, "TAP_REFINE_ENABLED", True)
    shot = _flat_screen_with_button()
    for content in ('{"found": false}', '{"found": true, "x": 719, "y": 719}', "not json"):
        plan = {"action_name": "click", "x": 200, "y": 300, "thought": "\u70b9\u51fb\u6309\u94ae"}
        obs = SimpleNamespace(page=exec_module.Page(representative_screenshot=shot), external_context=False)
        exec_module._maybe_refine_tap(SimpleNamespace(llm=_RefineLlm(content), trace=[]), plan, obs)
        assert (plan["x"], plan["y"]) == (200, 300)
        assert "tap_refined_from" not in plan


# Device regression 2026-10-05: waiting for progress was mistaken for repeated blind probing.

def _progress_frames(count):
    import numpy as np

    from tools.utils import encode_image_to_jpg_base64

    frames = []
    for index in range(count):
        image = np.full((800, 400, 3), 235, dtype=np.uint8)
        # The small central progress bar barely changes a 64x64 thumbnail.
        image[380:400, 150:150 + 15 * (index + 1)] = 30
        frames.append(encode_image_to_jpg_base64(image))
    return frames


def test_waiting_on_visible_progress_is_not_a_no_progress_stall():
    from workflow.nodes import scenario_exec as exec_module

    Page = exec_module.Page
    frames = _progress_frames(6)
    actions = ["click", "wait", "wait", "click", "wait", "wait"]
    trace = [
        {"action_name": action, "current_page": Page(activity="CoolingActivity", representative_screenshot=frame)}
        for action, frame in zip(actions, frames)
    ]
    assert exec_module._detect_no_progress_stall(trace) == ""


def test_identical_screen_probing_is_still_a_no_progress_stall():
    from workflow.nodes import scenario_exec as exec_module

    Page = exec_module.Page
    frame = _progress_frames(1)[0]
    trace = [
        {"action_name": action, "current_page": Page(activity="CoolingActivity", representative_screenshot=frame)}
        for action in ["click", "wait", "click", "wait", "click", "wait"]
    ]
    assert "Blind probing" in exec_module._detect_no_progress_stall(trace)


def test_reentering_page_then_waiting_is_not_a_visual_cycle():
    """Reentering a task screen and waiting for progress must not be classified as a navigation cycle."""
    import numpy as np

    from tools.utils import encode_image_to_jpg_base64
    from workflow.nodes import scenario_exec as exec_module

    Page = exec_module.Page
    home = np.full((800, 400, 3), 235, dtype=np.uint8)
    detail = home.copy()
    detail[100:700, 50:350] = 40
    home_b64, detail_b64 = encode_image_to_jpg_base64(home), encode_image_to_jpg_base64(detail)
    frames = [home_b64, detail_b64, home_b64, detail_b64]

    def trace_with(last_action):
        return [
            {"action_name": action, "current_page": Page(activity="A", representative_screenshot=frame)}
            for action, frame in zip(["click", "click", "click", last_action], frames)
        ]

    assert exec_module._detect_visual_action_cycle(trace_with("wait")) == ""
    # Repeated back-and-forth taps still constitute a loop.
    assert "Visual-state alternation" in exec_module._detect_visual_action_cycle(trace_with("click"))


def test_repeated_ineffective_tap_on_textured_edge_triggers_refinement(monkeypatch):
    """Refine repeated ineffective taps near a textured button edge as well as taps on blank backgrounds."""
    from types import SimpleNamespace

    import numpy as np

    from tools.utils import encode_image_to_jpg_base64
    from workflow.nodes import scenario_exec as exec_module

    monkeypatch.setattr(exec_module, "TAP_REFINE_ENABLED", True)
    image = np.full((800, 400, 3), 120, dtype=np.uint8)
    image[::4, :] = 200  # Textured background avoids the blank-background trigger.
    shot = encode_image_to_jpg_base64(image)
    Page = exec_module.Page
    plan = {"action_name": "click", "x": 200, "y": 300, "thought": "\u70b9\u51fb\u6309\u94ae"}
    obs = SimpleNamespace(page=Page(representative_screenshot=shot))
    llm = _RefineLlm('{"found": true, "x": 180, "y": 260}')

    exec_module._maybe_refine_tap(SimpleNamespace(llm=llm, trace=[]), plan, obs)
    assert llm.calls == 0

    trace = [{"action_name": "click", "model_coordinates": {"x": 200, "y": 300},
              "current_page": Page(representative_screenshot=shot)}]
    exec_module._maybe_refine_tap(SimpleNamespace(llm=llm, trace=trace), plan, obs)
    assert llm.calls == 1
    assert plan["tap_refined_from"] == {"x": 200, "y": 300}
