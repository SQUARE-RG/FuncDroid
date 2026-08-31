import base64

import cv2
import numpy as np

from model.flow_graph import FlowGraph, FunctionNode, Page, PageTransition
from workflow.nodes import ffg_update as ffg_update_module
from workflow.nodes.ffg_update import (
    _ScreenshotThumbnailCache,
    _build_observation_candidate_map,
    _find_matched_existing_page,
    _reference_pages_for_update,
    create_ffg_update_node,
)


def _screenshot(value: int) -> str:
    image = np.full((48, 32, 3), value, dtype=np.uint8)
    success, buffer = cv2.imencode(".png", image)
    assert success
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def test_thumbnail_cache_decodes_each_graph_screenshot_once(monkeypatch):
    decode_calls: list[str] = []

    def fake_thumbnail(screenshot: str, size: int):
        decode_calls.append(screenshot)
        return np.zeros((size, size), dtype=np.uint8)

    monkeypatch.setattr(ffg_update_module, "screenshot_thumbnail", fake_thumbnail)
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="功能"))
    representative = _screenshot(10)
    evidence = _screenshot(20)
    graph.add_page(Page(
        id="page_1",
        representative_screenshot=representative,
        evidence_screenshots=[evidence],
    ), "func_1")
    cache = _ScreenshotThumbnailCache()

    cache.preload_graph(graph)
    cache.similarity(representative, evidence)
    cache.similarity(representative, evidence)

    assert decode_calls == [representative, evidence]


def test_reference_lookup_reuses_candidate_similarity_results():
    class CountingCache:
        def __init__(self):
            self.calls = 0

        def similarity(self, _left, _right):
            self.calls += 1
            return 0.9

    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="功能"))
    graph.add_page(Page(
        id="page_1",
        activity="MainActivity",
        representative_screenshot=_screenshot(10),
    ), "func_1")
    trace = [{
        "raw_step_index": 1,
        "current_page": Page(
            activity="MainActivity",
            representative_screenshot=_screenshot(20),
        ),
    }]
    cache = CountingCache()

    candidate_map, similarity_cache = _build_observation_candidate_map(
        graph,
        {"target_function_id": "func_1"},
        trace,
        thumbnail_cache=cache,
    )
    candidate_calls = cache.calls
    _reference_pages_for_update(
        graph,
        {"target_function_id": "func_1"},
        trace,
        candidate_map=candidate_map,
        thumbnail_cache=cache,
        step_similarity_cache=similarity_cache,
    )

    assert candidate_calls == 1
    assert cache.calls == candidate_calls


def test_screenshot_summary_substitution_requires_explicit_same_activity():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="功能"))
    graph.add_page(Page(id="page_1", activity="MainActivity"), "func_1")
    scores = {1: {"page_1": 1.0}}

    assert _find_matched_existing_page(
        graph,
        {"current_page": Page(activity="OtherActivity")},
        1,
        scores,
    ) == ""
    assert _find_matched_existing_page(
        graph,
        {"current_page": Page(activity="")},
        1,
        scores,
    ) == ""
    assert _find_matched_existing_page(
        graph,
        {"current_page": Page(activity="MainActivity")},
        1,
        {1: {"page_1": 0.9799}},
    ) == ""
    assert _find_matched_existing_page(
        graph,
        {"current_page": Page(activity="MainActivity")},
        1,
        {1: {"page_1": 0.99}},
    ) == "page_1"
    assert _find_matched_existing_page(
        graph,
        {"current_page": Page(activity="MainActivity")},
        1,
        {1: {"page_1": 0.99}},
        threshold=0.995,
    ) == ""


def test_matched_trace_uses_page_summary_without_reuploading_reference_image(monkeypatch):
    monkeypatch.setattr(
        ffg_update_module,
        "FFG_UPDATE_OMIT_MATCHED_TRACE_IMAGES",
        True,
    )
    screenshot = _screenshot(30)
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="功能"))
    graph.add_page(Page(
        id="page_1",
        name="已知页面",
        description="显示已记录的业务内容",
        controls=["保存按钮"],
        activity="MainActivity",
        representative_screenshot=screenshot,
    ), "func_1")

    class Response:
        content = '{"functions":[],"pages":[],"function_dependencies":[]}'

    class CapturingLlm:
        def __init__(self):
            self.messages = []

        def invoke(self, messages):
            self.messages = messages
            return Response()

    llm = CapturingLlm()
    create_ffg_update_node(llm)({
        "ffg": graph,
        "ffg_update_step": 1,
        "current_plan": {"target_function_id": "func_1"},
        "scenario_execution_trace": [{
            "action_name": "finished",
            "current_page": Page(
                activity="MainActivity",
                representative_screenshot=screenshot,
            ),
        }],
        "navigation_execution_trace": [],
        "execution_result": {"scenario_status": "completed"},
    })

    human_content = llm.messages[1].content
    text = "\n".join(
        block.get("text", "")
        for block in human_content
        if block.get("type") == "text"
    )
    images = [block for block in human_content if block.get("type") == "image_url"]
    assert "[OBSERVATION_IMAGE_MATCHED]" in text
    assert "显示已记录的业务内容" in text
    assert "保存按钮" in text
    assert images == []


def test_no_change_fast_path_skips_llm_and_reference_compression(monkeypatch):
    monkeypatch.setattr(ffg_update_module, "FFG_UPDATE_FAST_PATH_ENABLED", True)
    monkeypatch.setattr(
        ffg_update_module,
        "FFG_UPDATE_OMIT_MATCHED_TRACE_IMAGES",
        True,
    )
    screenshot = _screenshot(35)
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="功能", test_status="in_progress"))
    graph.add_page(Page(
        id="page_1",
        activity="MainActivity",
        representative_screenshot=screenshot,
        controls=["保存按钮"],
        operations=["点击保存按钮"],
    ), "func_1")
    graph.add_page(Page(
        id="page_2",
        activity="OtherActivity",
        representative_screenshot=_screenshot(180),
    ), "func_1")
    graph.add_page_transition(PageTransition(
        src_page_id="page_1",
        dst_page_id="page_1",
        operation="点击保存按钮",
    ))

    class NeverCalledLlm:
        def invoke(self, _messages):
            raise AssertionError("no-change trace should not invoke the LLM")

    # 页面审计是独立子系统（有自己的后台 worker），与本测试关注的
    # “fast path 不压缩 FFG 参考图”无关；显式关闭避免其截图压缩误触发断言。
    monkeypatch.setattr(ffg_update_module, "PAGE_AUDIT_ENABLED", False)
    monkeypatch.setattr(
        ffg_update_module,
        "_compact_screenshot_for_llm",
        lambda _value: (_ for _ in ()).throw(
            AssertionError("no-change fast path should not compact reference images")
        ),
    )

    result = create_ffg_update_node(NeverCalledLlm())({
        "ffg": graph,
        "ffg_update_step": 1,
        "current_plan": {
            "scenario_type": "intra_test",
            "target_function_id": "func_1",
        },
        "current_testing_function_id": "func_1",
        "current_testing_function_status": "in_progress",
        "current_test_scenario": "点击保存按钮并观察结果",
        "scenario_execution_trace": [
            {
                "action_name": "click",
                "action_description": "点击保存按钮",
                "current_page": Page(
                    activity="MainActivity",
                    representative_screenshot=screenshot,
                ),
            },
            {
                "action_name": "finished",
                "current_page": Page(
                    activity="MainActivity",
                    representative_screenshot=screenshot,
                ),
            },
        ],
        "navigation_execution_trace": [],
        "execution_result": {"scenario_status": "completed"},
    })

    assert result["ffg_update_decision"] == "no_change"
    assert result["ffg_update_validation_issues"] == []


def test_known_page_fast_path_records_new_operation_without_llm(monkeypatch):
    monkeypatch.setattr(ffg_update_module, "FFG_UPDATE_FAST_PATH_ENABLED", True)
    monkeypatch.setattr(
        ffg_update_module,
        "FFG_UPDATE_OMIT_MATCHED_TRACE_IMAGES",
        True,
    )
    screenshot = _screenshot(36)
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="功能", test_status="in_progress"))
    graph.add_page(Page(
        id="page_1",
        activity="MainActivity",
        representative_screenshot=screenshot,
        controls=["保存按钮"],
    ), "func_1")

    class NeverCalledLlm:
        def invoke(self, _messages):
            raise AssertionError("known-page trace should not invoke the LLM")

    result = create_ffg_update_node(NeverCalledLlm())({
        "ffg": graph,
        "ffg_update_step": 1,
        "current_plan": {
            "scenario_type": "intra_test",
            "target_function_id": "func_1",
        },
        "current_testing_function_id": "func_1",
        "current_testing_function_status": "in_progress",
        "current_test_scenario": "点击保存按钮并观察结果",
        "scenario_execution_trace": [
            {
                "action_name": "click",
                "action_description": "点击保存按钮",
                "current_page": Page(
                    activity="MainActivity",
                    representative_screenshot=screenshot,
                ),
            },
            {
                "action_name": "finished",
                "current_page": Page(
                    activity="MainActivity",
                    representative_screenshot=screenshot,
                ),
            },
        ],
        "navigation_execution_trace": [],
        "execution_result": {"scenario_status": "completed"},
    })

    assert result["ffg_update_decision"] == "no_change"
    assert graph.get_page("page_1").operations == ["点击保存按钮"]


def test_compacted_images_are_reused_only_within_one_refinement_cycle(monkeypatch):
    screenshot = _screenshot(40)
    compact_calls: list[str] = []

    def fake_compact(value: str) -> str:
        compact_calls.append(value)
        return "compact-image"

    monkeypatch.setattr(ffg_update_module, "_compact_screenshot_for_llm", fake_compact)

    class Response:
        content = '{"functions":[],"pages":[],"function_dependencies":[]}'

    class Llm:
        def invoke(self, _messages):
            return Response()

    node = create_ffg_update_node(Llm())
    graph = FlowGraph()
    state = {
        "ffg": graph,
        "current_plan": {},
        "scenario_execution_trace": [{
            "action_name": "finished",
            "current_page": Page(
                activity="NovelActivity",
                representative_screenshot=screenshot,
            ),
        }],
        "navigation_execution_trace": [],
        "execution_result": {"scenario_status": "completed"},
    }

    node({**state, "ffg_update_step": 1})
    node({**state, "ffg_update_step": 2})
    assert compact_calls == [screenshot]

    node({**state, "ffg_update_step": 1})
    assert compact_calls == [screenshot, screenshot]


def _captured_image_payloads(llm) -> list[str]:
    return [
        block["image_url"]["url"].split(",", 1)[1]
        for block in llm.messages[1].content
        if block.get("type") == "image_url"
    ]


class _CapturingPatchLlm:
    class Response:
        content = '{"functions":[],"pages":[],"function_dependencies":[]}'

    def __init__(self):
        self.messages = []

    def invoke(self, messages):
        self.messages = messages
        return self.Response()


def test_total_image_budget_never_drops_unmatched_new_pages(monkeypatch):
    monkeypatch.setattr(ffg_update_module, "FFG_UPDATE_MAX_TOTAL_IMAGES", 6)
    monkeypatch.setattr(ffg_update_module, "_compact_screenshot_for_llm", lambda value: value)
    screenshots = [_screenshot(10 + index * 20) for index in range(8)]
    llm = _CapturingPatchLlm()

    create_ffg_update_node(llm)({
        "ffg": FlowGraph(),
        "ffg_update_step": 1,
        "current_plan": {},
        "scenario_execution_trace": [
            {
                "action_name": "finished",
                "current_page": Page(
                    activity=f"NovelActivity{index}",
                    representative_screenshot=screenshot,
                ),
            }
            for index, screenshot in enumerate(screenshots)
        ],
        "navigation_execution_trace": [],
        "execution_result": {"scenario_status": "completed"},
    })

    assert _captured_image_payloads(llm) == screenshots


def test_default_fidelity_mode_keeps_known_and_new_trace_images(monkeypatch):
    monkeypatch.setattr(ffg_update_module, "FFG_UPDATE_MAX_TOTAL_IMAGES", 6)
    monkeypatch.setattr(ffg_update_module, "_compact_screenshot_for_llm", lambda value: value)
    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="功能"))
    known_screenshots = [_screenshot(10 + index * 20) for index in range(10)]
    trace = []
    for index, screenshot in enumerate(known_screenshots):
        activity = f"KnownActivity{index}"
        graph.add_page(Page(
            id=f"page_{index + 1}",
            activity=activity,
            representative_screenshot=screenshot,
        ), "func_1")
        trace.append({
            "action_name": "wait",
            "current_page": Page(
                activity=activity,
                representative_screenshot=screenshot,
            ),
        })
    new_screenshot = _screenshot(250)
    trace.append({
        "action_name": "finished",
        "current_page": Page(
            activity="NovelActivity",
            representative_screenshot=new_screenshot,
        ),
    })
    llm = _CapturingPatchLlm()

    create_ffg_update_node(llm)({
        "ffg": graph,
        "ffg_update_step": 1,
        "current_plan": {"target_function_id": "func_1"},
        "scenario_execution_trace": trace,
        "navigation_execution_trace": [],
        "execution_result": {"scenario_status": "completed"},
    })

    payloads = _captured_image_payloads(llm)
    assert new_screenshot in payloads
    assert payloads == [*known_screenshots, new_screenshot]


def test_refinement_resends_only_issue_step_and_neighbors(monkeypatch):
    monkeypatch.setattr(ffg_update_module, "FFG_UPDATE_MAX_REFINEMENT_IMAGES", 3)
    monkeypatch.setattr(ffg_update_module, "_compact_screenshot_for_llm", lambda value: value)
    screenshots = [_screenshot(20 + index * 40) for index in range(5)]
    llm = _CapturingPatchLlm()

    create_ffg_update_node(llm)({
        "ffg": FlowGraph(),
        "ffg_update_step": 2,
        "ffg_update_validation_issues": [
            "ignore alignment lacks an ignorable observation at source_step 3"
        ],
        "current_plan": {},
        "scenario_execution_trace": [
            {
                "action_name": "wait",
                "current_page": Page(
                    activity=f"NovelActivity{index}",
                    representative_screenshot=screenshot,
                ),
            }
            for index, screenshot in enumerate(screenshots)
        ],
        "navigation_execution_trace": [],
        "execution_result": {"scenario_status": "completed"},
    })

    assert _captured_image_payloads(llm) == screenshots[1:4]
