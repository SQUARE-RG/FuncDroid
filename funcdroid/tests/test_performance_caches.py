# Multilingual UI and model-response fixtures use Unicode escapes to preserve their exact text.
"""Performance cache tests cover shared image compression, per-audit screenshot limits, screenshot hash reuse, and device launcher/display-size caches."""
import base64

import cv2
import numpy as np

import automator.u2 as u2_module
import model.flow_graph as flow_graph_module
import workflow.image_compact as image_compact_module
from automator.u2 import U2
from model.flow_graph import FlowGraph, FunctionNode, Page
from workflow.image_compact import compact_screenshot
from workflow.nodes import ffg_update as ffg_update_module


def _tiny_screenshot(value: int) -> str:
    image = np.full((48, 32, 3), value, dtype=np.uint8)
    success, buffer = cv2.imencode(".png", image)
    assert success
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def test_shared_compact_cache_compresses_once(monkeypatch):
    calls: list[int] = []
    real_imdecode = image_compact_module.cv2.imdecode

    def counting_imdecode(*args, **kwargs):
        calls.append(1)
        return real_imdecode(*args, **kwargs)

    monkeypatch.setattr(image_compact_module.cv2, "imdecode", counting_imdecode)

    screenshot = _tiny_screenshot(77)
    first = compact_screenshot(screenshot, max_side=32, quality=80)
    second = compact_screenshot(screenshot, max_side=32, quality=80)
    assert first != ""
    assert first == second
    assert len(calls) == 1

    # Different parameters use separate cache entries and require recompression.
    assert compact_screenshot(screenshot, max_side=16, quality=80) != ""
    assert len(calls) == 2


def test_page_audit_batch_caps_images(monkeypatch):
    class Llm:
        def invoke(self, _messages):  # pragma: no cover - Audits enqueue without synchronous execution.
            raise AssertionError("\u5ba1\u8ba1\u4efb\u52a1\u4e0d\u5e94\u540c\u6b65\u6267\u884c")

    captured: dict = {}

    def fake_submit(job):
        captured.update(job)
        return True

    monkeypatch.setattr(ffg_update_module, "PAGE_AUDIT_MAX_IMAGES", 4)
    monkeypatch.setattr(ffg_update_module.page_audit_worker, "submit", fake_submit)

    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="\u529f\u80fd\u4e00"))
    graph.add_page(Page(
        id="page_1",
        activity="MainActivity",
        name="\u9996\u9875",
        representative_screenshot=_tiny_screenshot(10),
    ), "func_1")
    for graph_page in graph.pages.values():
        graph_page.evidence_screenshots = [
            _tiny_screenshot(20 + index) for index in range(10)
        ]

    submitted = ffg_update_module._submit_page_audit_batch(graph, Llm())
    assert submitted == 1

    content = captured["messages"][1].content
    image_blocks = [block for block in content if block.get("type") == "image_url"]
    assert len(image_blocks) == 4
    texts = [block["text"] for block in content if block.get("type") == "text"]
    assert any("PAGE_STATE_IMAGES_OMITTED" in text for text in texts)


def test_flow_graph_save_skips_unchanged_screenshot_decode(tmp_path, monkeypatch):
    calls: list[int] = []
    real_b64decode = flow_graph_module.base64.b64decode

    def counting_b64decode(*args, **kwargs):
        calls.append(1)
        return real_b64decode(*args, **kwargs)

    monkeypatch.setattr(flow_graph_module.base64, "b64decode", counting_b64decode)

    graph = FlowGraph()
    graph.add_node(FunctionNode(id="func_1", name="\u529f\u80fd\u4e00"))
    graph.add_page(Page(
        id="page_1",
        activity="MainActivity",
        name="\u9996\u9875",
        representative_screenshot=_tiny_screenshot(5),
    ), "func_1")
    output_dir = tmp_path / "run"
    output_dir.mkdir()

    graph.save(str(output_dir))
    assert len(calls) == 1

    graph.save(str(output_dir))
    # The second save hits the screenshot hash and existing file without decoding or rereading it.
    assert len(calls) == 1


def test_u2_display_size_cached(monkeypatch):
    calls: list[int] = []

    def fake_resolve(self):
        calls.append(1)
        return (1080, 2400)

    monkeypatch.setattr(U2, "_resolve_display_size", fake_resolve)
    device = U2.__new__(U2)
    device._display_size_cache = None

    assert device.display_size() == (1080, 2400)
    assert device.display_size() == (1080, 2400)
    assert len(calls) == 1


def test_u2_launcher_activity_cached(monkeypatch):
    monkeypatch.setattr(u2_module, "_LAUNCHER_ACTIVITY_CACHE", {})
    run_calls: list[int] = []

    class FakeResult:
        stdout = (
            "priority=0 preferredOrder=0 matchAll=0 current=0\n"
            "  com.example.app/.ui.HomeActivity\n"
        )

    def fake_run(*args, **kwargs):
        run_calls.append(1)
        return FakeResult()

    monkeypatch.setattr(u2_module.subprocess, "run", fake_run)
    device = U2.__new__(U2)
    device._serial = "serial-1"

    # Expand relative Activity names before scoring; both am start -n spellings are equivalent.
    assert device._preferred_launcher_activity("com.example.app") == "com.example.app.ui.HomeActivity"
    assert device._preferred_launcher_activity("com.example.app") == "com.example.app.ui.HomeActivity"
    assert len(run_calls) == 1
