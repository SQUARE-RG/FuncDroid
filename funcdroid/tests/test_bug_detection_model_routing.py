from types import SimpleNamespace

import main


def test_runtime_creates_shared_endpoint_llm_instances(monkeypatch):
    calls: list[dict] = []

    def fake_create_llm(**kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(main, "create_llm", fake_create_llm)
    monkeypatch.setattr(main, "Driver", lambda device_id: {"device_id": device_id})
    monkeypatch.setenv("FFG_UPDATE_LLM_TIMEOUT", "300")

    planning_llm, execution_llm, ffg_update_llm, bug_detection_llm, device = main._init_runtime(
        SimpleNamespace(device_id="device-1")
    )

    assert len(calls) == 4
    assert calls[0] == {}
    assert calls[1] == {}
    assert calls[2] == {"timeout_seconds": 300.0}
    assert calls[3] == {}
    assert planning_llm is not execution_llm
    assert ffg_update_llm is not planning_llm
    assert bug_detection_llm is not execution_llm
    assert device == {"device_id": "device-1"}


def test_runtime_skips_bug_detection_model_when_disabled(monkeypatch):
    calls: list[dict] = []

    def fake_create_llm(**kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(main, "create_llm", fake_create_llm)
    monkeypatch.setattr(main, "Driver", lambda device_id: {"device_id": device_id})
    monkeypatch.setenv("FFG_UPDATE_LLM_TIMEOUT", "300")

    _planning_llm, _execution_llm, _ffg_update_llm, bug_detection_llm, _device = main._init_runtime(
        SimpleNamespace(device_id="device-1", bug_detection=False)
    )

    assert len(calls) == 3
    assert bug_detection_llm is None
