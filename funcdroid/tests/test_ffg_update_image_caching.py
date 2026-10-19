# Multilingual UI and model-response fixtures use Unicode escapes to preserve their exact text.
import base64

import cv2
import numpy as np

from model.flow_graph import FlowGraph, FunctionNode, Page
from workflow.nodes import ffg_update as ffg_update_module
from workflow.nodes.ffg_update import (
    _ScreenshotThumbnailCache,
    _build_observation_candidate_map,
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
    graph.add_node(FunctionNode(id="func_1", name="\u529f\u80fd"))
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
    graph.add_node(FunctionNode(id="func_1", name="\u529f\u80fd"))
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


def test_compacted_images_are_compressed_once_across_cycles(monkeypatch):
    """Compress the same screenshot once across update iterations using the shared process cache. Count resize_image calls to measure actual compression rather than thumbnail caching."""
    import tools.utils as tools_utils_module
    import workflow.image_compact as image_compact_module

    resize_calls: list[int] = []
    real_resize = tools_utils_module.resize_image

    def counting_resize(image, *, max_side):
        resize_calls.append(1)
        return real_resize(image, max_side=max_side)

    monkeypatch.setattr(image_compact_module, "resize_image", counting_resize)

    screenshot = _screenshot(40)

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

    node(dict(state))
    node(dict(state))
    node(dict(state))
    assert len(resize_calls) == 1


def _captured_image_payloads(llm) -> list[str]:
    return [
        block["image_url"]["url"].split(",", 1)[1]
        for block in llm.messages[1].content
        if block.get("type") == "image_url"
    ]


class _CapturingPatchLlm:
    """Capture the first request with all trace images; repair requests contain only affected steps."""

    class Response:
        content = '{"functions":[],"pages":[],"function_dependencies":[]}'

    def __init__(self):
        self.messages = []
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        if not self.messages:
            self.messages = messages
        return self.Response()


def test_total_image_budget_never_drops_unmatched_new_pages(monkeypatch):
    monkeypatch.setattr(ffg_update_module, "FFG_UPDATE_MAX_TOTAL_IMAGES", 6)
    monkeypatch.setattr(ffg_update_module, "_compact_screenshot_for_llm", lambda value: value)
    screenshots = [_screenshot(10 + index * 20) for index in range(8)]
    llm = _CapturingPatchLlm()

    create_ffg_update_node(llm)({
        "ffg": FlowGraph(),
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
    graph.add_node(FunctionNode(id="func_1", name="\u529f\u80fd"))
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
        "current_plan": {"target_function_id": "func_1"},
        "scenario_execution_trace": trace,
        "navigation_execution_trace": [],
        "execution_result": {"scenario_status": "completed"},
    })

    payloads = _captured_image_payloads(llm)
    assert new_screenshot in payloads
    assert payloads == [*known_screenshots, new_screenshot]
