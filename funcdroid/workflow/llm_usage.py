"""LLM token usage accounting for each workflow run."""

from __future__ import annotations

import json
import os
import threading
import time as _time
from datetime import datetime
from typing import Any


_LOCK = threading.Lock()
_OUTPUT_DIR = ""
_USAGE_PATH = ""
_LLM_REGISTRY: dict[int, dict[str, str]] = {}
_USAGE: dict[str, Any] = {}

# Throttle usage persistence to update-count/time thresholds; explicit and final saves flush durably, while frequent writes use atomic replacement without fsync.
_DIRTY_WRITES = 0
_LAST_SAVE_MONOTONIC = 0.0
_LAST_SAVE_ERROR_LOG_MONOTONIC = 0.0
_SAVE_EVERY_N = max(1, int(os.getenv("LLM_USAGE_SAVE_EVERY_N", "20")))
_SAVE_MIN_INTERVAL = max(0.0, float(os.getenv("LLM_USAGE_SAVE_MIN_INTERVAL", "30.0")))
_SAVE_ERROR_LOG_INTERVAL = max(
    1.0,
    float(os.getenv("LLM_USAGE_SAVE_ERROR_LOG_INTERVAL", "60")),
)
# Maximum retained reasoning characters per response for later review.
LLM_DIAGNOSTIC_REASONING_MAX_CHARS = max(
    0,
    int(os.getenv("LLM_DIAGNOSTIC_REASONING_MAX_CHARS", "4000")),
)


def _now_iso() -> str:
    return datetime.now().isoformat()


def _empty_counter() -> dict[str, Any]:
    return {
        "requests": 0,
        "responses": 0,
        "failures": 0,
        "prompt_tokens": 0,
        "cached_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


def _empty_usage(output_dir: str) -> dict[str, Any]:
    return {
        "started_at": _now_iso(),
        "updated_at": _now_iso(),
        "output_dir": output_dir,
        "totals": _empty_counter(),
        "by_model": {},
        "by_operation": {},
    }


def configure_llm_usage(output_dir: str, filename: str = "llm_usage.json", reset: bool = True):
    """Configure where token usage should be saved."""
    global _OUTPUT_DIR, _USAGE_PATH, _USAGE
    normalized_dir = str(output_dir or "").strip()
    if not normalized_dir:
        return
    try:
        os.makedirs(normalized_dir, exist_ok=True)
    except OSError:
        # Keep the in-memory accounting active.  `_save_locked` will report a
        # rate-limited warning and retry on later checkpoints.
        pass

    with _LOCK:
        _OUTPUT_DIR = normalized_dir
        _USAGE_PATH = os.path.join(normalized_dir, filename)
        if not reset:
            try:
                if os.path.exists(_USAGE_PATH):
                    with open(_USAGE_PATH, "r", encoding="utf-8") as f:
                        _USAGE = json.load(f)
                else:
                    _USAGE = _empty_usage(normalized_dir)
            except Exception:
                _USAGE = _empty_usage(normalized_dir)
        else:
            _USAGE = _empty_usage(normalized_dir)
        _save_locked()
    print(f"[LLM usage] Token usage will be saved to: {_USAGE_PATH}")


def register_llm(llm, *, model: str):
    """Remember model metadata for a LangChain LLM instance."""
    if llm is None:
        return
    _LLM_REGISTRY[id(llm)] = {
        "model": str(model or "").strip(),
    }


def record_llm_request(llm, *, operation: str = ""):
    """Record an attempted LLM call."""
    with _LOCK:
        if not _USAGE:
            return
        _increment_counters(llm, operation, requests=1)
        _USAGE["updated_at"] = _now_iso()
        _save_locked_throttled()


def record_llm_response(llm, response, *, operation: str = ""):
    """Extract token usage from a LangChain response and save it."""
    usage = _extract_usage(response)
    with _LOCK:
        if not _USAGE:
            return
        _increment_counters(
            llm,
            operation,
            responses=1,
            prompt_tokens=usage["prompt_tokens"],
            cached_tokens=usage["cached_tokens"],
            completion_tokens=usage["completion_tokens"],
            total_tokens=usage["total_tokens"],
        )
        _USAGE["updated_at"] = _now_iso()
        _save_locked_throttled()


def record_llm_failure(llm, *, operation: str = ""):
    """Record a failed LLM attempt or parse failure."""
    with _LOCK:
        if not _USAGE:
            return
        _increment_counters(llm, operation, failures=1)
        _USAGE["updated_at"] = _now_iso()
        _save_locked_throttled()


def get_llm_usage_summary() -> dict[str, Any]:
    with _LOCK:
        return json.loads(json.dumps(_USAGE, ensure_ascii=False)) if _USAGE else {}


def get_llm_usage_totals() -> dict[str, Any]:
    """Return a shallow copy of totals without copying per-model/per-operation history."""
    with _LOCK:
        totals = _USAGE.get("totals") if _USAGE else None
        return dict(totals) if isinstance(totals, dict) else {}


def save_llm_usage_summary(*, durable: bool = True):
    """Persist usage explicitly.

    With durable=False, throttle writes and omit fsync for frequent checkpoints. Final and explicit saves remain durable."""
    with _LOCK:
        if not _USAGE:
            return
        if durable:
            _USAGE["updated_at"] = _now_iso()
            _save_locked()
        elif _DIRTY_WRITES > 0 and _save_due_locked():
            _save_locked(fsync=False)


def _llm_key(llm) -> tuple[str, str]:
    meta = _LLM_REGISTRY.get(id(llm), {})
    model = meta.get("model", "") or _infer_model_name(llm) or "unknown"
    return model, model


def _infer_model_name(llm) -> str:
    for attr in ("model_name", "model", "deployment_name"):
        value = getattr(llm, attr, "")
        if value:
            return str(value)
    return ""


def _increment_counters(
    llm,
    operation: str,
    *,
    requests: int = 0,
    responses: int = 0,
    failures: int = 0,
    prompt_tokens: int = 0,
    cached_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
):
    model, model_key = _llm_key(llm)
    operation_key = str(operation or "unknown").strip() or "unknown"
    fields = {
        "requests": requests,
        "responses": responses,
        "failures": failures,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }

    for target in (
        _USAGE.setdefault("totals", _empty_counter()),
        _USAGE.setdefault("by_model", {}).setdefault(
            model_key,
            {"model": model, **_empty_counter()},
        ),
        _USAGE.setdefault("by_operation", {}).setdefault(operation_key, _empty_counter()),
    ):
        for key, value in fields.items():
            target[key] = int(target.get(key, 0) or 0) + int(value or 0)


def _extract_usage(response) -> dict[str, int]:
    candidates = []
    usage_metadata = getattr(response, "usage_metadata", None)
    if isinstance(usage_metadata, dict):
        candidates.append(usage_metadata)
    response_metadata = getattr(response, "response_metadata", None)
    if isinstance(response_metadata, dict):
        for key in ("token_usage", "usage", "usage_metadata"):
            value = response_metadata.get(key)
            if isinstance(value, dict):
                candidates.append(value)
        candidates.append(response_metadata)

    prompt_tokens = completion_tokens = total_tokens = 0
    for usage in candidates:
        prompt_tokens = prompt_tokens or _int_first(
            usage,
            "prompt_tokens",
            "input_tokens",
            "prompt_token_count",
            "input_token_count",
        )
        completion_tokens = completion_tokens or _int_first(
            usage,
            "completion_tokens",
            "output_tokens",
            "completion_token_count",
            "output_token_count",
        )
        total_tokens = total_tokens or _int_first(
            usage,
            "total_tokens",
            "total_token_count",
        )

    if not total_tokens and (prompt_tokens or completion_tokens):
        total_tokens = prompt_tokens + completion_tokens
    # Track cached-input tokens separately for provider prefix-cache accounting.
    cached_tokens = 0
    for usage in candidates:
        if cached_tokens:
            break
        input_details = usage.get("input_token_details")
        if isinstance(input_details, dict):
            cached_tokens = _int_first(input_details, "cache_read", "cached_tokens")
        if not cached_tokens:
            prompt_details = usage.get("prompt_tokens_details")
            if isinstance(prompt_details, dict):
                cached_tokens = _int_first(prompt_details, "cached_tokens")
    cached_tokens = max(0, min(cached_tokens, prompt_tokens))
    return {
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _int_first(mapping: dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str) and value.strip().isdigit():
            return int(value.strip())
    return 0


def _save_due_locked() -> bool:
    """Check the dirty-count threshold or minimum save interval."""
    return (
        _DIRTY_WRITES >= _SAVE_EVERY_N
        or (_time.monotonic() - _LAST_SAVE_MONOTONIC) >= _SAVE_MIN_INTERVAL
    )


def _save_locked_throttled():
    """Throttle full serialization by update count or elapsed time.

    Frequent updates use atomic replacement without fsync; explicit and final saves provide durability."""
    global _DIRTY_WRITES
    _DIRTY_WRITES += 1
    if _save_due_locked():
        _save_locked(fsync=False)


def _log_save_error_locked(exc: BaseException) -> None:
    global _LAST_SAVE_ERROR_LOG_MONOTONIC
    now = _time.monotonic()
    if (
        _LAST_SAVE_ERROR_LOG_MONOTONIC > 0
        and now - _LAST_SAVE_ERROR_LOG_MONOTONIC < _SAVE_ERROR_LOG_INTERVAL
    ):
        return
    _LAST_SAVE_ERROR_LOG_MONOTONIC = now
    try:
        print(
            "[LLM usage] Persistence failed; counters remain in memory for the next checkpoint: "
            f"{exc}"
        )
    except Exception:
        pass


def _save_locked(*, fsync: bool = True) -> bool:
    """Best-effort persistence; usage accounting must never break an LLM call."""
    global _DIRTY_WRITES, _LAST_SAVE_MONOTONIC
    if not _USAGE_PATH:
        return False
    tmp_path = f"{_USAGE_PATH}.tmp"
    try:
        if _OUTPUT_DIR:
            os.makedirs(_OUTPUT_DIR, exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(_USAGE, f, ensure_ascii=False, indent=2)
            f.flush()
            if fsync:
                os.fsync(f.fileno())
        os.replace(tmp_path, _USAGE_PATH)
    except Exception as exc:  # noqa: BLE001 - telemetry must not affect workflow
        # Dirty data remains in `_USAGE`; reset only the throttle counter so a
        # persistent disk fault does not cause a write attempt on every token
        # accounting event.  Explicit checkpoints still retry immediately.
        _DIRTY_WRITES = 0
        _LAST_SAVE_MONOTONIC = _time.monotonic()
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        _log_save_error_locked(exc)
        return False
    _DIRTY_WRITES = 0
    _LAST_SAVE_MONOTONIC = _time.monotonic()
    return True


def record_llm_diagnostic(response=None, *, operation="", error="", elapsed_seconds=0.0):
    """Persist an allowlisted response record without prompts or credentials."""
    if not _OUTPUT_DIR:
        return
    metadata = getattr(response, "response_metadata", {}) or {}
    additional = getattr(response, "additional_kwargs", {}) or {}
    # Persist bounded reasoning text so reviewers can inspect how short final verdicts were reached.
    reasoning = additional.get("reasoning_content") or additional.get("reasoning")
    if not isinstance(reasoning, str):
        reasoning = ""
    reasoning = reasoning.strip()[:LLM_DIAGNOSTIC_REASONING_MAX_CHARS]
    record = {
        "timestamp": _now_iso(), "operation": operation,
        "content": getattr(response, "content", None),
        "finish_reason": metadata.get("finish_reason"),
        "request_id": metadata.get("request_id") or metadata.get("id"),
        "usage": _extract_usage(response), "elapsed_seconds": elapsed_seconds,
        "has_reasoning": bool(reasoning),
        "reasoning": reasoning,
        "error": error,
    }
    try:
        with _LOCK:
            with open(os.path.join(_OUTPUT_DIR, "llm_responses.jsonl"), "a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except OSError as exc:
        print(f"[LLM diagnostics] Persistence failed: {type(exc).__name__}")
