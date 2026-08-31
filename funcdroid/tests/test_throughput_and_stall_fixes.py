"""Regressions for the throughput fixes: blind-probe early stop, LLM error
budget, zero-yield scheduling release, and batched coverage scenarios."""

from __future__ import annotations

import base64

import cv2
import numpy as np
import pytest

from model.flow_graph import FlowGraph, FunctionNode, Page
from workflow.nodes.ffg_update import (
    FUNCTION_FAILURE_BLOCK_THRESHOLD,
    ZERO_YIELD_MIN_COVERAGE,
    _record_zero_structural_yield,
    _update_function_execution_progress,
)
from workflow.nodes.scenario_exec import (
    MAX_CONSECUTIVE_ERRORS,
    MAX_TOTAL_LLM_ERRORS,
    NO_PROGRESS_STALL_WINDOW,
    _detect_no_progress_stall,
    _detect_stall,
)
from workflow.nodes.scenario_plan import (
    PLAN_SCENARIO_CONTROL_BATCH,
    _forced_intra_scenario,
)


@pytest.fixture(autouse=True)
def _legacy_text_matching_enabled(monkeypatch):
    """本文件用例写于操作-控件显式绑定之前，依赖文本匹配计入覆盖。

    显式绑定是覆盖率的权威来源且文本兜底默认关闭
    （COVERAGE_LEGACY_TEXT_MATCHING），此处显式开启以保持用例原语义。
    """
    from model import flow_graph as _flow_graph

    monkeypatch.setattr(_flow_graph, "COVERAGE_LEGACY_TEXT_MATCHING", True)


def _screenshot(marker_width: int) -> str:
    image = np.full((256, 256, 3), 255, dtype=np.uint8)
    if marker_width:
        image[20:40, 20:20 + marker_width] = 0
    success, encoded = cv2.imencode(".jpg", image)
    assert success
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _full_screen_screenshot(level: int) -> str:
    """整屏内容更换，模拟真实的列表翻页。

    相似度口径是全图灰度均值差，因此只有整屏级别的变化才会被判为不同状态；
    局部小面积的展开/收起会被视为同一状态，这正是盲试要被拦下的形态。
    """
    image = np.full((256, 256, 3), max(0, 255 - level * 40), dtype=np.uint8)
    success, encoded = cv2.imencode(".jpg", image)
    assert success
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _probe_step(action_name: str, screenshot: str, **fields) -> dict:
    step = {
        "action_name": action_name,
        "current_page": Page(
            id="",
            activity=".plugin.gallery.ui.AlbumPreviewUI",
            representative_screenshot=screenshot,
        ),
    }
    step.update(fields)
    return step


def _blind_probe_trace() -> list[dict]:
    """Mixed-direction probing that every pre-existing detector misses."""
    collapsed = _screenshot(10)
    expanded = _screenshot(200)
    plan = [
        ("click", collapsed, {"x": 719, "y": 186}),
        ("click", expanded, {"x": 600, "y": 1591}),
        ("scroll", collapsed, {"direction": "up"}),
        ("scroll", collapsed, {"direction": "left"}),
        ("scroll", collapsed, {"direction": "right"}),
        ("click", expanded, {"x": 719, "y": 186}),
    ]
    return [
        _probe_step(action, screenshot, **fields)
        for action, screenshot, fields in plan[:NO_PROGRESS_STALL_WINDOW]
    ]


def test_mixed_direction_blind_probing_is_detected_as_stalled():
    trace = _blind_probe_trace()

    reason = _detect_no_progress_stall(trace)

    assert reason
    assert "反复试探" in reason
    # 必须由统一入口生效，而不是只有单独调用时才生效。
    assert _detect_stall(trace) == reason


def test_blind_probe_detection_needs_the_full_window():
    trace = _blind_probe_trace()[: NO_PROGRESS_STALL_WINDOW - 1]

    assert _detect_no_progress_stall(trace) == ""


def test_genuine_list_browsing_is_not_treated_as_stalled():
    """每步都带来整屏变化时，兜底检测器不得早停；同方向滚动预算另行拦截。"""
    trace = [
        _probe_step("scroll", _full_screen_screenshot(index), direction="up")
        for index in range(NO_PROGRESS_STALL_WINDOW)
    ]

    # 无进展兜底检测器不早停（每步都有整屏变化）。
    assert _detect_no_progress_stall(trace) == ""
    # 但同方向滚动硬预算在统一入口生效：连续同方向滚动超限必须转向。
    stall_reason = _detect_stall(trace)
    assert stall_reason
    assert "同方向滑动预算" in stall_reason


def test_two_state_ping_pong_across_full_screens_is_stalled():
    """整屏切换但只在两个状态间往返，同样属于无进展。"""
    first = _full_screen_screenshot(0)
    second = _full_screen_screenshot(4)
    trace = [
        _probe_step("click", first if index % 2 == 0 else second, x=100, y=200 + index)
        for index in range(NO_PROGRESS_STALL_WINDOW)
    ]

    assert _detect_no_progress_stall(trace)


def test_activity_change_within_window_is_not_treated_as_stalled():
    trace = _blind_probe_trace()
    trace[-1]["current_page"] = Page(
        id="",
        activity=".ui.LauncherUI",
        representative_screenshot=_screenshot(10),
    )

    assert _detect_no_progress_stall(trace) == ""


def test_interruption_recovery_is_left_to_its_own_detector():
    trace = _blind_probe_trace()
    trace[-1]["is_interruption"] = True

    assert _detect_no_progress_stall(trace) == ""


def test_total_llm_error_budget_is_stricter_than_nothing_but_allows_retries():
    """单轮累计上限必须存在，且不得比连续失败阈值更严。"""
    assert MAX_TOTAL_LLM_ERRORS >= MAX_CONSECUTIVE_ERRORS


def _zero_yield_graph(coverage_controls: int, handled: int) -> FlowGraph:
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="公众号图文内容详情查看"))
    graph.add_page(Page(
        id="page_1",
        activity=".plugin.brandservice.ui.timeline.preload.ui.TmplWebViewMMUI",
        controls=[f"正文段落{index}" for index in range(coverage_controls)],
        operations=[f"点击正文段落{index}" for index in range(handled)],
    ), "func_1")
    graph.set_node_test_status("func_1", "in_progress")
    return graph


def _zero_yield_state() -> dict:
    return {
        "current_plan": {"scenario_type": "intra_test", "target_function_id": "func_1"},
        "execution_result": {"scenario_status": "completed"},
    }


def test_successful_round_without_new_pages_counts_as_no_progress():
    graph = _zero_yield_graph(coverage_controls=4, handled=3)
    counts = graph.function_coverage_details("func_1")
    assert counts["coverage_percent"] >= ZERO_YIELD_MIN_COVERAGE

    snapshot = {"functions": 12, "pages": 17}
    failure_counts, function_id, status = _record_zero_structural_yield(
        graph,
        _zero_yield_state(),
        snapshot,
        dict(snapshot),
        {},
        "func_1",
        "in_progress",
    )

    assert failure_counts["func_1"] == 1
    # 首次零产出只降优先级，不该立刻让出当前功能。
    assert function_id == "func_1"
    assert status == "in_progress"


def test_repeated_zero_yield_releases_the_current_scheduling_target():
    graph = _zero_yield_graph(coverage_controls=4, handled=3)
    snapshot = {"functions": 12, "pages": 17}

    failure_counts, function_id, status = _record_zero_structural_yield(
        graph,
        _zero_yield_state(),
        snapshot,
        dict(snapshot),
        {"func_1": FUNCTION_FAILURE_BLOCK_THRESHOLD - 1},
        "func_1",
        "in_progress",
    )

    # _select_intra_target 会硬优先返回 preferred_id，只加计数无法解除粘连。
    assert function_id == ""
    assert status == "not_started"
    assert failure_counts["func_1"] == FUNCTION_FAILURE_BLOCK_THRESHOLD


def test_new_page_in_the_round_keeps_the_function_uncounted():
    graph = _zero_yield_graph(coverage_controls=4, handled=3)

    failure_counts, function_id, status = _record_zero_structural_yield(
        graph,
        _zero_yield_state(),
        {"functions": 12, "pages": 17},
        {"functions": 12, "pages": 18},
        {},
        "func_1",
        "in_progress",
    )

    assert failure_counts == {}
    assert function_id == "func_1"
    assert status == "in_progress"


def test_early_exploration_is_not_penalised_for_zero_yield():
    """覆盖率还低的功能本就需要几轮铺垫，不能一上来就降优先级。"""
    graph = _zero_yield_graph(coverage_controls=20, handled=1)
    assert graph.function_coverage_details("func_1")["coverage_percent"] < ZERO_YIELD_MIN_COVERAGE

    snapshot = {"functions": 12, "pages": 17}
    failure_counts, function_id, _status = _record_zero_structural_yield(
        graph,
        _zero_yield_state(),
        snapshot,
        dict(snapshot),
        {},
        "func_1",
        "in_progress",
    )

    assert failure_counts == {}
    assert function_id == "func_1"


def test_failed_round_is_not_double_counted_as_zero_yield():
    graph = _zero_yield_graph(coverage_controls=4, handled=3)
    state = _zero_yield_state()
    state["execution_result"] = {
        "scenario_status": "failed_goal",
        "scenario_error": "执行器判断当前目标不可达",
    }
    snapshot = {"functions": 12, "pages": 17}

    failure_counts, _function_id, _status = _record_zero_structural_yield(
        graph,
        state,
        snapshot,
        dict(snapshot),
        {},
        "func_1",
        "in_progress",
    )

    assert failure_counts == {}


def test_zero_yield_count_survives_a_successful_round():
    """成功轮不得清零零产出计数，否则计数永远累积不到释放阈值。

    _update_function_execution_progress 在 scenario_status == completed 时会 pop
    掉目标功能的计数，而它先于 _record_zero_structural_yield 执行。线上表现是
    日志里的零产出计数恒为 (1/3)，功能永远不会被让出。
    """
    graph = _zero_yield_graph(coverage_controls=4, handled=3)
    state = _zero_yield_state()
    counts: dict[str, int] = {}

    for expected in range(1, FUNCTION_FAILURE_BLOCK_THRESHOLD + 1):
        counts, function_id, _status = _record_zero_structural_yield(
            graph,
            state,
            {"functions": 12, "pages": 17},
            {"functions": 12, "pages": 17},
            counts,
            "func_1",
            "in_progress",
        )
        assert counts.get("func_1") == expected
        # 模拟真实链路：每轮成功执行后都会走一次执行进度更新，它从 state 读计数。
        state["function_execution_failure_counts"] = dict(counts)
        counts, _fid, _st = _update_function_execution_progress(
            graph,
            state,
            state["execution_result"],
            "func_1",
            "in_progress",
        )
        assert counts.get("func_1") == expected, "成功轮不得清零零产出计数"

    assert function_id == ""


def test_structural_yield_resets_the_zero_yield_streak():
    """只有连续零产出才释放；本轮有新页面就必须清零。"""
    graph = _zero_yield_graph(coverage_controls=4, handled=3)

    counts, _fid, _st = _record_zero_structural_yield(
        graph,
        _zero_yield_state(),
        {"functions": 12, "pages": 17},
        {"functions": 12, "pages": 18},
        {"func_1": FUNCTION_FAILURE_BLOCK_THRESHOLD - 1},
        "func_1",
        "in_progress",
    )

    assert "func_1" not in counts


def test_forced_intra_scenario_batches_multiple_uncovered_controls():
    graph = FlowGraph()
    node = FunctionNode(id="func_1", name="聊天信息管理")
    graph.add_node(node)
    graph.add_page(Page(
        id="page_1",
        name="聊天信息页",
        controls=["查找聊天内容", "设置聊天背景", "清空聊天记录入口", "消息免打扰"],
        operations=[],
    ), "func_1")

    scenario = _forced_intra_scenario(graph, graph.get_node("func_1"))

    assert scenario
    hints = [
        hint for hint in ("查找聊天内容", "设置聊天背景", "消息免打扰")
        if hint in scenario
    ]
    assert len(hints) >= min(2, PLAN_SCENARIO_CONTROL_BATCH)
    assert "依次探索" in scenario
    # 旧文案会把执行器锁死在单个控件上。
    assert "不要继续串行探索其他控件" not in scenario


def test_forced_intra_scenario_degrades_to_single_control():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="扫码"))
    graph.add_page(Page(
        id="page_1",
        name="扫码页",
        controls=["相册"],
        operations=[],
    ), "func_1")

    scenario = _forced_intra_scenario(graph, graph.get_node("func_1"))

    assert "相册" in scenario
    assert "依次探索" not in scenario


def _scroll_step(direction: str, phase: str = "") -> dict:
    return _probe_step(
        "scroll",
        _full_screen_screenshot(1),
        direction=direction,
        trace_phase=phase,
    )


def test_consecutive_same_direction_scroll_counting():
    from workflow.nodes.scenario_exec import (
        _consecutive_same_direction_scroll,
    )

    # 3 连 down 计数为 3。
    trace = [_scroll_step("down") for _ in range(3)]
    assert _consecutive_same_direction_scroll(trace) == ("down", 3)
    # 中间插入其他动作后计数重置。
    trace.append(_probe_step("click", _full_screen_screenshot(2), x=100, y=200))
    trace.append(_scroll_step("down"))
    assert _consecutive_same_direction_scroll(trace) == ("down", 1)
    # 改变方向后计数重置。
    trace.extend([_scroll_step("up") for _ in range(2)])
    assert _consecutive_same_direction_scroll(trace) == ("up", 2)
    # 尾部不是 scroll 时返回 0。
    trace.append(_probe_step("wait", _full_screen_screenshot(2)))
    assert _consecutive_same_direction_scroll(trace) == ("", 0)


def test_scroll_budget_triggers_stall_on_fourth_same_direction_scroll():
    """第 4 步仍是同方向滚动时，_detect_stall 在统一入口拦截。"""
    trace = [_scroll_step("down") for _ in range(4)]

    reason = _detect_stall(trace)

    assert reason
    assert "同方向滑动预算" in reason
    assert "向下" in reason


def test_scroll_budget_allows_three_same_direction_scrolls():
    trace = [_scroll_step("down") for _ in range(3)]

    assert _detect_stall(trace) == ""


def test_scroll_budget_resets_after_other_actions():
    """3 连滚动后改为点击，接着再滚动不会误触发预算拦截。"""
    trace = [_scroll_step("down") for _ in range(3)]
    trace.append(_probe_step("click", _full_screen_screenshot(2), x=100, y=200))
    trace.append(_scroll_step("down"))

    assert _detect_stall(trace) == ""


def test_scroll_budget_skips_navigation_phase():
    """导航阶段的滚动走原有意图检测，不受同方向滚动硬预算约束。"""
    trace = [_scroll_step("down", phase="navigation") for _ in range(4)]

    assert _detect_stall(trace) == ""


def test_step_input_warns_when_scroll_budget_nearly_exhausted():
    from workflow.nodes.scenario_exec import _build_step_input

    trace = [_scroll_step("down") for _ in range(3)]
    content = _build_step_input(trace, "", 4, 15)

    text_blocks = [
        item["text"] for item in content if item.get("type") == "text"
    ]
    assert any("滚动预算" in text for text in text_blocks)
    assert any("禁止继续" in text for text in text_blocks)
