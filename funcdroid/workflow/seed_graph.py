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
        raise ValueError("FFG seed package does not match the current target")
    if result.get("status") != "success":
        raise ValueError("FFG seed must come from a successfully finalized run")
    manifest = json.loads((root / "artifact_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("FFG seed artifacts are incomplete")
    graph = FlowGraph.load(str(root))
    graph.cross_test_results = []
    graph.cross_test_history = []
    graph.test_data_records = []
    # FlowGraph.load supports inline images and file references. Fail before
    # device access if a referenced screenshot was lost instead of forwarding
    # a filename to the vision model as though it were image bytes.
    # Referenced screenshot files were already validated and encoded by FlowGraph.load; validate only inline base64 here.
    inline_screenshots = not (root / "screenshots").is_dir()
    for page in graph.pages.values():
        if inline_screenshots:
            for screenshot in [page.representative_screenshot, *page.evidence_screenshots]:
                if screenshot:
                    try:
                        base64.b64decode(screenshot, validate=True)
                    except ValueError as exc:
                        raise ValueError(f"FFG seed screenshot is missing or corrupt: {page.id}") from exc
        page.operations = []
        page.operation_bindings = []
    for function in graph.functions.values():
        function.test_status = "not_started"
        function.coverage_percent = 0
        function.scenarios = []
        function.intra_attempt_count = 0
        # Reset inherited depth coverage so a seed run cannot skip its own depth scenarios.
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
