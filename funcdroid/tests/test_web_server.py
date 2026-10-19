# Multilingual UI and model-response fixtures use Unicode escapes to preserve their exact text.
"""Test visualization artifact parsing through OutputStore and normalization helpers without opening a listening port."""

import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location("funcdroid_web_server", Path("web") / "server.py")
server = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(server)


@pytest.fixture()
def run_dir(tmp_path):
    """Create defect-run artifacts containing seed and resume metadata."""
    run = tmp_path / "experiments" / "output" / "demo_run"
    run.mkdir(parents=True)
    (run / "run_result.json").write_text(
        json.dumps(
            {
                "status": "success",
                "exit_code": 0,
                "elapsed_seconds": 300.5,
                "iteration_count": 3,
                "package_name": "com.example.app",
                "high_confidence_bug_count": 1,
                "seed_provenance": {
                    "source_output": "/tmp/other_run",
                    "source_mode": "defect",
                    "inherited_functions": 6,
                    "inherited_pages": 3,
                    "coverage_reset": True,
                    "ffg_sha256": "a" * 64,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (run / "ffg.json").write_text(
        json.dumps(
            {
                "functions": [
                    {"id": "func_1", "name": "\u964d\u6e29", "test_status": "completed", "coverage_percent": 80,
                     "page_ids": ["page_1"], "entry_page_ids": ["page_2"]},
                    {"id": "func_2", "name": "\u7701\u7535", "test_status": "not_started", "coverage_percent": 0,
                     "page_ids": ["page_1"]},
                ],
                "pages": [{"id": "page_1", "name": "\u9996\u9875", "representative_screenshot": "screenshots/a.png"}],
                "page_transitions": [{"src_page_id": "page_1", "dst_page_id": "page_1", "operation": "click"}],
                "function_dependencies": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (run / "bugs.json").write_text(json.dumps([{"bug_id": "BUG-001"}]), encoding="utf-8")
    (run / "artifact_manifest.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")
    (run / "run.log").write_text(
        "\u7b2c 1 \u8f6e — \u6a21\u5f0f: defect | \u63a2\u7d22\u8303\u56f4: func_1\n"
        "[\u8f6e\u6b21\u6307\u6807] llm_requests=5, steps=9\n"
        "[\u72b6\u6001\u66f4\u65b0] \u6267\u884c\u573a\u666f: func_1\n",
        encoding="utf-8",
    )
    return run


def test_list_runs_reads_metadata_and_counts(tmp_path, run_dir):
    store = server.OutputStore(tmp_path / "experiments" / "output")
    runs = store.list_runs()
    assert len(runs) == 1
    item = runs[0]
    assert item["name"] == "demo_run"
    assert item["status"] == "success"
    assert item["bug_count"] == 1
    assert item["high_confidence_bug_count"] == 1
    assert item["has_ffg"] is True
    # Aggregate test_status: one completed functionality out of two.
    assert item["ffg_counts"]["test_status"]["completed"] == 1
    assert item["ffg_counts"]["functions"] == 2


def test_summary_exposes_manifest_and_seed(run_dir):
    store = server.OutputStore(run_dir.parent)
    data = store.summary(run_dir)
    assert data["manifest"]["status"] == "complete"
    assert data["run_result"]["seed_provenance"]["inherited_functions"] == 6
    assert "artifact_manifest.json" in data["files"]


def test_ffg_normalization_merges_entry_pages(run_dir):
    store = server.OutputStore(run_dir.parent)
    graph = store.ffg(run_dir)
    func_1 = graph["functions"][0]
    # Merge entry_page_ids into page_ids without duplicates.
    assert func_1["page_ids"] == ["page_1", "page_2"]
    assert func_1["coverage_percent"] == 80
    # Pass through current-format pages and retain representative screenshot fields.
    assert graph["pages"][0]["representative_screenshot"] == "screenshots/a.png"


def test_metrics_parses_round_headers_and_kv(run_dir):
    store = server.OutputStore(run_dir.parent)
    data = store.metrics(run_dir)
    assert data["rounds"][0]["mode"] == "defect"
    assert data["iterations"][0]["llm_requests"] == 5
    assert data["iterations"][0]["steps"] == 9


def test_path_traversal_is_rejected(tmp_path, run_dir):
    store = server.OutputStore(run_dir.parent)
    assert store.get_run_dir("../other") is None
    assert store.get_run_dir("..") is None
    assert store.compliance_trace(run_dir, "../secret.json") is None
