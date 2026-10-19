"""Model initialization and structured-response handling."""

import json
import math
import os
import re
import time
from typing import Pattern

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from workflow.llm_usage import (
    record_llm_failure,
    record_llm_diagnostic,
    record_llm_request,
    record_llm_response,
    register_llm,
)

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


DEFAULT_JSON_BLOCK_PATTERN = re.compile(r"^```(?:json)?\s*|\s*```$")
# Request provider-enforced JSON for execution; disable unsupported response_format through the caller fallback.
JSON_RESPONSE_FORMAT_PAYLOAD = {"type": "json_object"}
JSON_FORMAT_REJECTION_PATTERN = re.compile(
    r"response_format|json_object|json\s*mode|unsupported\s+parameter|"
    r"unrecognized\s+request\s+argument|invalid\s+parameter",
    re.IGNORECASE,
)


def json_format_rejected(exc: BaseException) -> bool:
    """Return whether the provider rejected response_format."""
    return bool(JSON_FORMAT_REJECTION_PATTERN.search(str(exc) or ""))


def disable_json_response_format(llm) -> bool:
    """Remove response_format after provider rejection.

    LangChain merges model_kwargs on every request, so removing the key affects subsequent calls. Return whether it was present."""
    model_kwargs = getattr(llm, "model_kwargs", None)
    if isinstance(model_kwargs, dict) and "response_format" in model_kwargs:
        model_kwargs.pop("response_format", None)
        print("[LLM] Disabled response_format=json_object; subsequent requests use free-text responses.")
        return True
    return False


class _ReasoningAwareChatOpenAI(ChatOpenAI):
    """Preserve third-party reasoning_content in additional_kwargs.

    The OpenAI-compatible response may retain this extra field even when ChatOpenAI does not copy it into the resulting message."""

    def _create_chat_result(self, response, generation_info=None):
        result = super()._create_chat_result(response, generation_info)
        raw_choices = getattr(response, "choices", None)
        if raw_choices:
            for generation, raw_choice in zip(result.generations, raw_choices):
                reasoning_content = getattr(
                    getattr(raw_choice, "message", None), "reasoning_content", None
                )
                if isinstance(reasoning_content, str) and reasoning_content.strip():
                    generation.message.additional_kwargs["reasoning_content"] = reasoning_content
        return result


class _FailedJsonResult(dict):
    """Empty mapping that preserves why no JSON result was produced."""

    def __init__(self, reason: str):
        super().__init__()
        self.failure_reason = reason


def llm_json_result_failed(result: object) -> bool:
    """Whether the retry helper exhausted before producing valid JSON."""
    return isinstance(result, _FailedJsonResult)

# Repair only the known missing-y coordinate typo after strict parsing fails; the executor validates image bounds.
MISSING_COORD_KEY_PATTERN = re.compile(
    r'"x"\s*:\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*(?=[,}\]\n])'
)
# Some providers emit x as a two-coordinate array without a separate y key.
ARRAY_COORD_PATTERN = re.compile(
    r'"x"\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]'
)


def _repair_common_json_typos(text: str) -> str:
    """Repair well-known LLM JSON typos; returns text unchanged when nothing matches."""
    repaired = MISSING_COORD_KEY_PATTERN.sub(
        lambda match: (
            f'"x": {match.group(1)}, "y": {match.group(2)}'
            if _is_coordinate_number(match.group(1))
            and _is_coordinate_number(match.group(2))
            else match.group(0)
        ),
        text,
    )
    if repaired != text:
        print("[LLM] Repaired missing coordinate key (\"x\": a, b → \"x\": a, \"y\": b）")
    array_repaired = ARRAY_COORD_PATTERN.sub(
        lambda match: (
            f'"x": {match.group(1)}, "y": {match.group(2)}'
            if _is_coordinate_number(match.group(1))
            and _is_coordinate_number(match.group(2))
            else match.group(0)
        ),
        repaired,
    )
    if array_repaired != repaired:
        print("[LLM] Repaired array-form coordinates (\"x\": [a, b] → \"x\": a, \"y\": b）")
    return array_repaired


def _is_coordinate_number(value: str) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number >= 0


def _parse_json_response(text: str):
    """Parse exact JSON or the last top-level object embedded in model prose."""
    try:
        return json.loads(text)
    except json.JSONDecodeError as exact_error:
        decoder = json.JSONDecoder()
        selected = None
        # Decode linearly in the original string and skip entire decoded objects so nested evidence does not become a separate candidate.
        idx = text.find("{")
        while idx >= 0:
            try:
                value, end = decoder.raw_decode(text, idx)
            except json.JSONDecodeError:
                idx = text.find("{", idx + 1)
                continue
            if isinstance(value, dict):
                selected = value
            idx = text.find("{", max(end, idx + 1))
        if selected is not None:
            print("[LLM] Extracted the last top-level JSON object from a response containing prose.")
            return selected
        raise exact_error

_ENV_LOADED = False


def _load_env_once():
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    if load_dotenv is not None:
        load_dotenv()
    _ENV_LOADED = True


def _normalize_response_text(content) -> str:
    """Normalize text and list-of-blocks response content to a string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                value = block.get("text") or block.get("content") or ""
                if isinstance(value, str):
                    parts.append(value)
        return "".join(parts).strip()
    return str(content).strip()


# Always supply a finite request timeout to prevent one stalled call from hanging the workflow.
DEFAULT_LLM_REQUEST_TIMEOUT_SECONDS = 180.0


def _resolve_request_timeout(timeout_seconds: float | None) -> float:
    """Resolve timeout from explicit value, environment, then the 180-second default.

    Nonpositive or invalid values use the finite default rather than disabling the timeout."""
    candidates = [timeout_seconds]
    if timeout_seconds is None:
        candidates = [os.getenv("LLM_REQUEST_TIMEOUT", "").strip() or None]
    for value in candidates:
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number) and number > 0:
            return number
    return DEFAULT_LLM_REQUEST_TIMEOUT_SECONDS


def create_llm(
    timeout_seconds: float | None = None,
    reasoning_effort: str | None = None,
    json_response_format: bool = False,
):
    """Create a chat model using the shared configured endpoint.

    Explicit reasoning_effort overrides the global setting. JSON response mode is requested only when enabled and can be disabled after provider rejection."""
    _load_env_once()

    # Prefer generic API_KEY/BASE_URL/MODEL; retain DEEPSEEK_* compatibility fallbacks.
    api_key = os.getenv("API_KEY", "").strip() or os.getenv("DEEPSEEK_API_KEY", "").strip()
    base_url = os.getenv("BASE_URL", "").strip() or os.getenv("DEEPSEEK_BASE_URL", "").strip()
    model = os.getenv("MODEL", "").strip() or os.getenv("DEEPSEEK_MODEL", "").strip()
    if reasoning_effort is None:
        resolved_effort = os.getenv("LLM_REASONING_EFFORT", "low").strip().lower()
    else:
        resolved_effort = str(reasoning_effort).strip().lower()
    resolved_timeout = _resolve_request_timeout(timeout_seconds)

    if not api_key or not base_url or not model:
        raise ValueError(
            "Missing model configuration: check API_KEY, BASE_URL, and MODEL in .env"
        )

    kwargs = {
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
        "temperature": 0,
        # FuncDroid already owns bounded retry/backoff and node recovery. SDK
        # retries would multiply request_timeout and can stall a long run for
        # several minutes on one request.
        "max_retries": 0,
    }
    kwargs["request_timeout"] = resolved_timeout
    if resolved_effort in {"none", "disabled", "off"}:
        if model.lower().startswith("deepseek"):
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        else:
            kwargs["reasoning_effort"] = "none"
    elif resolved_effort:
        kwargs["reasoning_effort"] = resolved_effort
    if json_response_format:
        # Inject through model_kwargs without changing the model type, usage tracking, or retries.
        kwargs["model_kwargs"] = {
            "response_format": dict(JSON_RESPONSE_FORMAT_PAYLOAD),
        }

    llm = _ReasoningAwareChatOpenAI(**kwargs)
    register_llm(llm, model=model)
    return llm


def compute_retry_delay_seconds(
    attempt: int,
    *,
    base_delay_seconds: float | None = None,
    backoff_multiplier: float | None = None,
    max_delay_seconds: float | None = None,
) -> float:
    """Compute exponential retry backoff; attempts are one-based."""
    _load_env_once()
    normalized_attempt = max(1, int(attempt))
    base_delay = float(os.getenv("LLM_RETRY_DELAY", "2.0")) if base_delay_seconds is None else float(base_delay_seconds)
    multiplier = float(os.getenv("LLM_RETRY_BACKOFF_MULTIPLIER", "2.0")) if backoff_multiplier is None else float(backoff_multiplier)
    max_delay = float(os.getenv("LLM_RETRY_MAX_DELAY", "20.0")) if max_delay_seconds is None else float(max_delay_seconds)
    delay = base_delay * (multiplier ** (normalized_attempt - 1))
    return max(0.0, min(delay, max_delay))


def _build_json_retry_feedback(raw_content: str, exc: BaseException) -> str:
    """Explain the previous parsing failure instead of repeating an unchanged prompt.

    Give explicit JSON-only feedback when the response contained no object structure."""
    if "{" not in str(raw_content or ""):
        return (
            "The previous response contained prose without a JSON object and could not be parsed. "
            "Return only one JSON object, without explanation or Markdown fences."
        )
    return (
        f"The previous response could not be parsed: {str(exc)[:300]}。"
        "Return only one complete valid JSON object, without explanatory prose or Markdown fences."
    )


def invoke_llm_json_with_retry(
    llm,
    messages,
    *,
    log_prefix: str = "[LLM]",
    json_block_pattern: Pattern[str] | None = None,
    max_attempts: int | None = None,
    retry_delay_seconds: float | None = None,
    deadline_monotonic: float | None = None,
):
    """Call the model, parse its JSON response, and retry using the common policy."""
    _load_env_once()
    pattern = json_block_pattern or DEFAULT_JSON_BLOCK_PATTERN
    attempts = max(1, int(max_attempts or os.getenv("LLM_RETRY_ATTEMPTS", "3")))
    delay_seconds = float(os.getenv("LLM_RETRY_DELAY", "2.0")) if retry_delay_seconds is None else float(retry_delay_seconds)
    base_messages = list(messages)
    current_messages = base_messages
    raw_content = ""
    failure_reason = "no_attempt_completed"

    for attempt in range(1, attempts + 1):
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            print(f"{log_prefix} Request deadline reached; stop retrying.")
            break
        parse_failed = False
        response = None
        raw_content = ""
        request_started = time.monotonic()
        try:
            record_llm_request(llm, operation=log_prefix)
            response = llm.invoke(current_messages)
            record_llm_response(llm, response, operation=log_prefix)
            raw_content = _normalize_response_text(getattr(response, "content", ""))
            try:
                cleaned = pattern.sub("", raw_content).strip()
                try:
                    parsed = _parse_json_response(cleaned)
                except json.JSONDecodeError:
                    # Repair recognized coordinate typos before spending a model retry.
                    repaired = _repair_common_json_typos(cleaned)
                    if repaired != cleaned:
                        parsed = _parse_json_response(repaired)
                    else:
                        raise
                if not isinstance(parsed, dict):
                    raise ValueError(
                        "Model JSON must contain a top-level object; "
                        f"received  {type(parsed).__name__}"
                    )
                record_llm_diagnostic(response, operation=log_prefix, elapsed_seconds=time.monotonic() - request_started)
                return parsed
            except (json.JSONDecodeError, ValueError) as parse_exc:
                parse_failed = True
                raise parse_exc
        except Exception as exc:
            record_llm_diagnostic(response, operation=log_prefix, error=f"{type(exc).__name__}: {exc}", elapsed_seconds=time.monotonic() - request_started)
            record_llm_failure(llm, operation=log_prefix)
            kind = "parse" if parse_failed else "call"
            failure_reason = f"{kind} failed: {type(exc).__name__}"
            no_json_hint = " (response contains no JSON object)" if parse_failed and raw_content and "{" not in raw_content else ""
            print(f"{log_prefix} LLM {kind} failed{no_json_hint} (attempt {attempt}/{attempts}): {exc}\n[Raw response]: {raw_content}")
            if attempt >= attempts:
                break
            # Use short delays for parsing errors and exponential backoff for transport errors.
            if parse_failed:
                sleep_seconds = min(delay_seconds, 1.0)
                # Rebuild from base_messages with one corrective message so repeated failures do not grow prompt history.
                current_messages = [
                    *base_messages,
                    HumanMessage(content=_build_json_retry_feedback(raw_content, exc)),
                ]
            else:
                sleep_seconds = compute_retry_delay_seconds(
                    attempt,
                    base_delay_seconds=delay_seconds,
                )
            if deadline_monotonic is not None:
                remaining = deadline_monotonic - time.monotonic()
                if remaining <= 0:
                    print(f"{log_prefix} Request deadline reached; stop retrying.")
                    break
                sleep_seconds = min(sleep_seconds, remaining)
            time.sleep(sleep_seconds)

    return _FailedJsonResult(failure_reason)
