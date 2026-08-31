import pytest

from workflow import llm as llm_module


class FakeChatOpenAI:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.__class__.instances.append(self)


def _set_env(monkeypatch, **overrides):
    values = {
        "API_KEY": "key",
        "BASE_URL": "https://api.example/v1",
        "MODEL": "test-model",
    }
    values.update(overrides)
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def test_create_llm_reads_single_endpoint_config(monkeypatch):
    FakeChatOpenAI.instances = []
    monkeypatch.setattr(llm_module, "ChatOpenAI", FakeChatOpenAI)
    monkeypatch.setattr(llm_module, "register_llm", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("LLM_REQUEST_TIMEOUT", "77")
    _set_env(monkeypatch)

    llm_module.create_llm()

    assert FakeChatOpenAI.instances[0].kwargs == {
        "api_key": "key",
        "base_url": "https://api.example/v1",
        "model": "test-model",
        "temperature": 0,
        "max_retries": 0,
        "request_timeout": 77.0,
    }


def test_create_llm_honors_role_timeout_override(monkeypatch):
    FakeChatOpenAI.instances = []
    monkeypatch.setattr(llm_module, "ChatOpenAI", FakeChatOpenAI)
    monkeypatch.setattr(llm_module, "register_llm", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("LLM_REQUEST_TIMEOUT", "30")
    _set_env(monkeypatch)

    llm_module.create_llm(timeout_seconds=300)

    assert FakeChatOpenAI.instances[0].kwargs["request_timeout"] == 300.0


def test_create_llm_rejects_missing_required_fields(monkeypatch):
    monkeypatch.setattr(llm_module, "ChatOpenAI", FakeChatOpenAI)
    _set_env(monkeypatch, API_KEY="")

    with pytest.raises(ValueError, match="API_KEY"):
        llm_module.create_llm()
