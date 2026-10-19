"""Regression coverage for English runtime descriptions and preserved legacy matching."""

from model.flow_graph import (
    FlowGraph, FunctionalFlowGraph, FunctionNode, FunctionalityNode,
    control_family_key, operation_family_key, operation_is_failure, operation_is_recovery,
)
from workflow.account_safety import account_session_violation


def test_english_failure_and_recovery_share_operation_family():
    normal = "Click Save button"
    failure = "Failed attempt: Click Save button; error: target unavailable"
    recovery = "Automatic recovery: Click Save button"
    assert operation_is_failure(failure)
    assert not operation_is_failure(normal)
    assert operation_is_recovery(recovery)
    assert not operation_is_recovery(normal)
    assert operation_family_key(normal) == "click:save"
    assert operation_family_key(failure) == operation_family_key(normal)
    assert operation_family_key(recovery) == operation_family_key(normal)
    assert operation_family_key("Back") == "back:back"
    assert control_family_key("Top-left Back arrow") == control_family_key("\u9876\u90e8\u8fd4\u56de\u6309\u94ae") == "back"
    assert control_family_key("Back to home button") == control_family_key("\u8fd4\u56de\u9996\u9875\u6309\u94ae") == "back:home"
    assert control_family_key("Back to top button") != control_family_key("Back button")


def test_paper_names_preserve_existing_model_types():
    assert FunctionalFlowGraph is FlowGraph
    assert FunctionalityNode is FunctionNode


def test_english_session_negation_ends_at_clause_boundary(monkeypatch):
    monkeypatch.setenv("ACCOUNT_SESSION_GUARD", "1")
    assert account_session_violation("Do not click Logout") == ""
    assert account_session_violation("Avoid logout or switch account") == ""
    assert account_session_violation("Do not logout. Then click Logout")
    assert account_session_violation("Click Logout")


def test_web_reads_current_english_runtime_events(tmp_path):
    import importlib.util
    import json
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("english_runtime_server", Path(__file__).parents[1] / "web" / "server.py")
    server = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server)
    (tmp_path / "ffg.json").write_text(json.dumps({
        "functions": [],
        "pages": [{"id": "page_1", "name": "Home", "activity": "MainActivity"}],
    }), encoding="utf-8")
    (tmp_path / "run.log").write_text(
        "Iteration 2 - scope: defect | exploration: core functionalities\n"
        "[Plan] Scenario: Open preferences\n"
        "[Scenario execution] Set scenario logcat start: phase=test\n"
        "[Scenario execution] Initial page: page_1\n"
        "[Scenario execution] LLM input summary: phase=test, step=1, history_steps=0, current_page=page_1, activity=MainActivity, foreground=com.example\n"
        "[Step 1] [Interruption] page=unknown action=click | Assessment: Open preferences\n"
        "[FFG update] Increment: functionalities=1, pages=1, operations=2, page_transitions=1, function_dependencies=0\n"
        "[State] scenario_exec | navigation=success(0 steps) | scenario=success(1 steps) | page=page_1\n"
        "[Iteration metrics] llm_requests=2, steps=1, scenario_type=single_test\n",
        encoding="utf-8",
    )
    store = server.OutputStore(tmp_path.parent)
    metrics = store.metrics(tmp_path)
    assert metrics["rounds"] == [{"round": 2, "mode": "defect", "scope": "core functionalities"}]
    state = store.live_state(tmp_path)
    assert state["scenario"] == "Open preferences"
    assert state["current_page_id"] == "page_1"
    assert state["stage"] == "round_complete"
    assert state["trace"][0]["phase"] == "test"
    assert state["trace"][0]["page_inferred"] is True
    assert state["trace"][0]["thought"] == "Open preferences"
    assert state["result"]["scenario_status"] == "success"
    assert state["delta"]["operations"] == 2


def test_web_reads_legacy_events_without_rewriting_them(tmp_path):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("legacy_runtime_server", Path(__file__).parents[1] / "web" / "server.py")
    server = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server)
    # Literal legacy event vocabulary is preserved for historical artifact compatibility.
    legacy = (
        "\u7b2c 3 \u8f6e — \u6a21\u5f0f: defect | \u63a2\u7d22\u8303\u56f4: func_1\n"
        "[\u72b6\u6001\u66f4\u65b0] \u6267\u884c\u573a\u666f: Open preferences\n"
        "[\u573a\u666f\u6267\u884c] \u5df2\u8bbe\u7f6e\u573a\u666f logcat \u8d77\u70b9: phase=test\n"
        "[\u6b65\u9aa4 1] \u9875\u9762=page_1 \u52a8\u4f5c=click | \u601d\u8003: Open preferences\n"
        "[FFG \u66f4\u65b0] \u589e\u91cf: \u529f\u80fd=1, \u9875\u9762=1, \u64cd\u4f5c=2, \u9875\u9762\u8df3\u8f6c=1, \u529f\u80fd\u4f9d\u8d56=0\n"
        '[\u72b6\u6001\u66f4\u65b0] \u6267\u884c\u7ed3\u679c: {"scenario_status": "success"}\n'
        "[\u8f6e\u6b21\u6307\u6807] steps=1\n"
    )
    log = tmp_path / "run.log"
    log.write_text(legacy, encoding="utf-8")
    state = server.OutputStore(tmp_path.parent).live_state(tmp_path)
    assert state["latest_round"] == 3
    assert state["scenario"] == "Open preferences"
    assert state["trace"][0]["thought"] == "Open preferences"
    assert state["result"]["scenario_status"] == "success"
    assert state["delta"]["operations"] == 2
    assert log.read_text(encoding="utf-8") == legacy
