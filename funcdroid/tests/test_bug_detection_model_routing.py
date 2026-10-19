from types import SimpleNamespace

import pytest
import main


@pytest.mark.parametrize("bug,compliance,expected", [(True, False, 4), (False, True, 4), (False, False, 3)])
def test_runtime_creates_shared_endpoint_instances_for_enabled_mode(monkeypatch, bug, compliance, expected):
    calls = []
    def fake_create_llm(**kwargs):
        calls.append(kwargs)
        return object()
    monkeypatch.setattr(main, "create_llm", fake_create_llm)
    monkeypatch.setattr(main, "Driver", lambda device_id: {"device_id": device_id})
    monkeypatch.setenv("FFG_UPDATE_LLM_TIMEOUT", "300")
    monkeypatch.setenv("EXECUTION_REASONING_EFFORT", "low")
    monkeypatch.setenv("EXEC_JSON_RESPONSE_FORMAT", "1")
    planning, execution, update, review, device = main._init_runtime(
        SimpleNamespace(device_id="device-1", bug_detection=bug, compliance=compliance)
    )
    # Only the scenario executor gets low reasoning effort and JSON mode; other roles use global settings.
    assert calls == [
        {},
        {"reasoning_effort": "low", "json_response_format": True},
        {"timeout_seconds": 300.0},
    ] + ([{}] if expected == 4 else [])
    assert planning is not execution and update is not planning
    assert (review is not None) == (expected == 4)
    assert device == {"device_id": "device-1"}


def test_runtime_allows_disabling_execution_json_mode(monkeypatch):
    calls = []

    def fake_create_llm(**kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(main, "create_llm", fake_create_llm)
    monkeypatch.setattr(main, "Driver", lambda device_id: object())
    monkeypatch.setenv("EXECUTION_REASONING_EFFORT", "low")
    monkeypatch.setenv("EXEC_JSON_RESPONSE_FORMAT", "0")

    main._init_runtime(
        SimpleNamespace(device_id="device-1", bug_detection=False, compliance=False)
    )

    assert calls[1] == {"reasoning_effort": "low", "json_response_format": False}
