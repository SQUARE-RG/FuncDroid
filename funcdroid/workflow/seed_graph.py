"""Start a new testing budget from prior navigation knowledge, not prior coverage."""

import base64
import hashlib
import json
from pathlib import Path

from model.flow_graph import FlowGraph


def load_seed_graph(source: str, package_name: str) -> tuple[FlowGraph, dict]:
    root = Path(source).resolve()
    result = json.loads((root / "run_result.json").read_text(encoding="utf-8"))
    if not package_name or result.get("package_name") != package_name:
        raise ValueError("FFG 种子包名与本次目标不一致")
    if result.get("status") != "success":
        raise ValueError("FFG 种子必须来自成功收尾的运行")
    manifest = json.loads((root / "artifact_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("FFG 种子产物尚未完整收尾")
    graph = FlowGraph.load(str(root))
    graph.cross_test_results = []
    graph.cross_test_history = []
    graph.test_data_records = []
    # FlowGraph.load supports inline images and file references. Fail before
    # device access if a referenced screenshot was lost instead of forwarding
    # a filename to the vision model as though it were image bytes.
    # 存在 screenshots/ 目录时，FlowGraph.load 已逐个读取引用文件并自行
    # base64 编码（文件缺失直接抛错），这些值必然合法，无需再全量解码；
    # 只有无截图目录、ffg.json 内联原始 base64 时才需要校验内容。
    inline_screenshots = not (root / "screenshots").is_dir()
    for page in graph.pages.values():
        if inline_screenshots:
            for screenshot in [page.representative_screenshot, *page.evidence_screenshots]:
                if screenshot:
                    try:
                        base64.b64decode(screenshot, validate=True)
                    except ValueError as exc:
                        raise ValueError(f"FFG 种子截图缺失或损坏: {page.id}") from exc
        page.operations = []
        page.operation_bindings = []
    for function in graph.functions.values():
        function.test_status = "not_started"
        function.coverage_percent = 0
        function.scenarios = []
        function.intra_attempt_count = 0
        # 深度阶段计数同样属于上一次运行的覆盖进度；不清零会让种子运行
        # 直接跳过深度阶段（_function_phase 依据该计数判断深度场景是否已满）。
        function.depth_scenario_count = 0
        function.successful_depth_scenario_count = 0
        function.navigation_failure_count = 0
        function.status_reason = ""
    provenance = {
        "source_output": str(root),
        "source_mode": result.get("mode"),
        "ffg_sha256": hashlib.sha256((root / "ffg.json").read_bytes()).hexdigest(),
        "source_elapsed_seconds": result.get("elapsed_seconds"),
        "inherited_functions": len(graph.functions),
        "inherited_pages": len(graph.pages),
        "coverage_reset": True,
    }
    return graph, provenance
