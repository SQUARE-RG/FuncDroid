# Multilingual UI and model-response fixtures use Unicode escapes to preserve their exact text.
import pytest

from workflow import llm as llm_module


@pytest.fixture(autouse=True)
def _no_proxy_env(monkeypatch):
    # ChatOpenAI builds httpx clients from proxy variables; SOCKS proxies require socksio.
    for name in ("ALL_PROXY", "all_proxy", "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        monkeypatch.delenv(name, raising=False)


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
        "LLM_REASONING_EFFORT": "",
    }
    values.update(overrides)
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def test_create_llm_reads_single_endpoint_config(monkeypatch):
    FakeChatOpenAI.instances = []
    monkeypatch.setattr(llm_module, "_ReasoningAwareChatOpenAI", FakeChatOpenAI)
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
    monkeypatch.setattr(llm_module, "_ReasoningAwareChatOpenAI", FakeChatOpenAI)
    monkeypatch.setattr(llm_module, "register_llm", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("LLM_REQUEST_TIMEOUT", "30")
    _set_env(monkeypatch)

    llm_module.create_llm(timeout_seconds=300)

    assert FakeChatOpenAI.instances[0].kwargs["request_timeout"] == 300.0


def test_create_llm_rejects_missing_required_fields(monkeypatch):
    monkeypatch.setattr(llm_module, "_ReasoningAwareChatOpenAI", FakeChatOpenAI)
    # Reject configuration only when both generic and DEEPSEEK fallback settings are empty.
    _set_env(monkeypatch, API_KEY="", BASE_URL="", MODEL="",
             DEEPSEEK_API_KEY="", DEEPSEEK_BASE_URL="", DEEPSEEK_MODEL="")

    with pytest.raises(ValueError, match="API_KEY"):
        llm_module.create_llm()


def test_create_llm_falls_back_to_deepseek_env(monkeypatch):
    """Fall back to DEEPSEEK settings when generic credentials, endpoint, and model are empty."""
    FakeChatOpenAI.instances = []
    monkeypatch.setattr(llm_module, "_ReasoningAwareChatOpenAI", FakeChatOpenAI)
    monkeypatch.setattr(llm_module, "register_llm", lambda *_args, **_kwargs: None)
    _set_env(
        monkeypatch,
        API_KEY="", BASE_URL="", MODEL="",
        DEEPSEEK_API_KEY="deepseek-key",
        DEEPSEEK_BASE_URL="https://api.deepseek.com",
        DEEPSEEK_MODEL="deepseek-flash",
    )

    llm_module.create_llm()

    kwargs = FakeChatOpenAI.instances[0].kwargs
    assert kwargs["api_key"] == "deepseek-key"
    assert kwargs["base_url"] == "https://api.deepseek.com"
    assert kwargs["model"] == "deepseek-flash"


@pytest.mark.parametrize("timeout", [None, 300])
def test_high_reasoning_effort_is_sent_in_chat_request(monkeypatch, timeout):
    _set_env(monkeypatch, MODEL="glm-5.3-flash", LLM_REASONING_EFFORT=" high ")
    # Exercise the real installed adapter's serialization without any network call.
    llm = llm_module.create_llm(timeout_seconds=timeout)
    payload = llm._get_request_payload([{"role": "user", "content": "test"}])
    assert payload["reasoning_effort"] == "high"
    assert payload["model"] == "glm-5.3-flash"


def test_empty_effort_preserves_provider_default(monkeypatch):
    _set_env(monkeypatch)
    payload = llm_module.create_llm()._get_request_payload([{"role": "user", "content": "test"}])
    assert "reasoning_effort" not in payload


def test_json_response_format_is_sent_when_enabled(monkeypatch):
    _set_env(monkeypatch)

    llm = llm_module.create_llm(json_response_format=True)
    payload = llm._get_request_payload([{"role": "user", "content": "test"}])

    assert payload["response_format"] == {"type": "json_object"}


def test_json_response_format_absent_by_default(monkeypatch):
    _set_env(monkeypatch)

    payload = llm_module.create_llm()._get_request_payload([{"role": "user", "content": "test"}])

    assert "response_format" not in payload


def test_disable_json_response_format_falls_back_to_free_text(monkeypatch):
    _set_env(monkeypatch)
    llm = llm_module.create_llm(json_response_format=True)

    assert llm_module.disable_json_response_format(llm) is True
    payload = llm._get_request_payload([{"role": "user", "content": "test"}])
    assert "response_format" not in payload

    # Repeated calls report no removable settings so the executor knows degradation already happened.
    assert llm_module.disable_json_response_format(llm) is False


@pytest.mark.parametrize("message", [
    "Error code: 400 - {'error': {'message': \"Unrecognized request argument supplied: response_format\"}}",
    "response_format is not supported by this model",
    "Unsupported parameter: 'json_object' is not supported with this model.",
])
def test_json_format_rejected_detects_provider_errors(message):
    assert llm_module.json_format_rejected(RuntimeError(message)) is True


def test_json_format_rejected_ignores_unrelated_errors():
    assert llm_module.json_format_rejected(RuntimeError("Connection timed out")) is False
    assert llm_module.json_format_rejected(ValueError("\u6a21\u578b\u8f93\u51fa\u4e2d\u6ca1\u6709\u5408\u6cd5 JSON object")) is False


class _FakeRawMessage:
    """Fake OpenAI SDK response message preserving third-party attributes with extra=allow."""

    def __init__(self, content, reasoning_content=None):
        self.role = "assistant"
        self.content = content
        self.reasoning_content = reasoning_content

    def model_dump(self):
        return {"role": self.role, "content": self.content}


class _FakeRawChoice:
    def __init__(self, message, finish_reason="stop"):
        self.message = message
        self.finish_reason = finish_reason

    def model_dump(self):
        return {"message": self.message.model_dump(), "finish_reason": self.finish_reason}


class _FakeRawChatCompletion:
    """Fake raw non-dict completion returned to ChatOpenAI._generate."""

    def __init__(self, choices):
        self.choices = choices
        self.usage = None
        self.model = "test-model"
        self.id = "resp-1"
        self.system_fingerprint = ""
        self.service_tier = None

    def model_dump(self):
        return {
            "choices": [choice.model_dump() for choice in self.choices],
            "usage": self.usage,
            "model": self.model,
            "id": self.id,
            "system_fingerprint": self.system_fingerprint,
        }


def test_reasoning_content_from_third_party_endpoint_is_preserved(monkeypatch):
    """Preserve third-party reasoning_content in additional_kwargs, following refusal/parsed handling, so diagnostic artifacts retain the reasoning behind short final answers."""
    _set_env(monkeypatch)
    llm = llm_module.create_llm()

    raw_response = _FakeRawChatCompletion(
        [_FakeRawChoice(_FakeRawMessage("\u6700\u7ec8\u7b54\u6848", reasoning_content="\u9010\u6b65\u63a8\u7406\u8fc7\u7a0b..."))]
    )

    result = llm._create_chat_result(raw_response)

    assert (
        result.generations[0].message.additional_kwargs["reasoning_content"]
        == "\u9010\u6b65\u63a8\u7406\u8fc7\u7a0b..."
    )


def test_missing_reasoning_content_does_not_add_the_key(monkeypatch):
    """Do not invent a reasoning_content key when none exists."""
    _set_env(monkeypatch)
    llm = llm_module.create_llm()

    raw_response = _FakeRawChatCompletion([_FakeRawChoice(_FakeRawMessage("\u6700\u7ec8\u7b54\u6848"))])

    result = llm._create_chat_result(raw_response)

    assert "reasoning_content" not in result.generations[0].message.additional_kwargs


def test_default_reasoning_is_low(monkeypatch):
    _set_env(monkeypatch)
    monkeypatch.setattr(llm_module, '_ENV_LOADED', True)
    monkeypatch.delenv('LLM_REASONING_EFFORT')
    payload = llm_module.create_llm()._get_request_payload([{'role': 'user', 'content': 'test'}])
    assert payload['reasoning_effort'] == 'low'


def test_deepseek_execution_disables_thinking_with_json(monkeypatch):
    _set_env(monkeypatch, MODEL='deepseek-flash', LLM_REASONING_EFFORT='low')
    payload = llm_module.create_llm(reasoning_effort='none', json_response_format=True)._get_request_payload(
        [{'role': 'user', 'content': 'test'}])
    assert payload['extra_body']['thinking'] == {'type': 'disabled'}
    assert 'reasoning_effort' not in payload
    assert payload['response_format'] == {'type': 'json_object'}
