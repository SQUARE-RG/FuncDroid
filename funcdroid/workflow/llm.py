"""LLM 初始化。"""

import json
import os
import re
import time
from typing import Pattern

from langchain_openai import ChatOpenAI

from workflow.llm_usage import (
    record_llm_failure,
    record_llm_request,
    record_llm_response,
    register_llm,
)

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


DEFAULT_JSON_BLOCK_PATTERN = re.compile(r"^```(?:json)?\s*|\s*```$")


class _FailedJsonResult(dict):
    """Empty mapping that preserves why no JSON result was produced."""

    def __init__(self, reason: str):
        super().__init__()
        self.failure_reason = reason


def llm_json_result_failed(result: object) -> bool:
    """Whether the retry helper exhausted before producing valid JSON."""
    return isinstance(result, _FailedJsonResult)

# 已知的高频 JSON 笔误修复模式：本地小模型常把坐标写成
# "x": 226, 190（丢失 "y": 键名）。仅在严格解析失败后作为兜底尝试，
# 且要求两个数字都落在 0–1000 的坐标范围内，避免误修普通数组。
MISSING_COORD_KEY_PATTERN = re.compile(
    r'"x"\s*:\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*(?=[,}\]\n])'
)
# 部分模型把坐标写成数组：\"x\": [682, 954]（缺独立 y 键）。
ARRAY_COORD_PATTERN = re.compile(
    r'"x"\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]'
)


def _repair_common_json_typos(text: str) -> str:
    """Repair well-known LLM JSON typos; returns text unchanged when nothing matches."""
    repaired = MISSING_COORD_KEY_PATTERN.sub(
        lambda match: (
            f'"x": {match.group(1)}, "y": {match.group(2)}'
            if _is_normalized_coordinate(match.group(1))
            and _is_normalized_coordinate(match.group(2))
            else match.group(0)
        ),
        text,
    )
    if repaired != text:
        print("[LLM] 已修复缺失坐标键名的 JSON 笔误（\"x\": a, b → \"x\": a, \"y\": b）")
    array_repaired = ARRAY_COORD_PATTERN.sub(
        lambda match: (
            f'"x": {match.group(1)}, "y": {match.group(2)}'
            if _is_normalized_coordinate(match.group(1))
            and _is_normalized_coordinate(match.group(2))
            else match.group(0)
        ),
        repaired,
    )
    if array_repaired != repaired:
        print("[LLM] 已修复数组坐标的 JSON 笔误（\"x\": [a, b] → \"x\": a, \"y\": b）")
    return array_repaired


def _is_normalized_coordinate(value: str) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return 0 <= number <= 1000


def _parse_json_response(text: str):
    """Parse exact JSON or the last complete object embedded in weak-model prose."""
    try:
        return json.loads(text)
    except json.JSONDecodeError as exact_error:
        decoder = json.JSONDecoder()
        parsed_objects: list[dict] = []
        for match in re.finditer(r"\{", text):
            try:
                value, _end = decoder.raw_decode(text[match.start():])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                parsed_objects.append(value)
        if parsed_objects:
            print("[LLM] 已从附带解释文字的响应中提取最后一个完整 JSON object。")
            return parsed_objects[-1]
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
    """把 LLM 响应内容归一化为文本，兼容 list-of-blocks 返回。"""
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


def create_llm(timeout_seconds: float | None = None):
    """根据环境变量创建聊天模型；所有角色共用同一个端点。"""
    _load_env_once()

    api_key = os.getenv("API_KEY", "").strip()
    base_url = os.getenv("BASE_URL", "").strip()
    model = os.getenv("MODEL", "").strip()
    resolved_timeout = timeout_seconds
    if resolved_timeout is None:
        global_timeout = os.getenv("LLM_REQUEST_TIMEOUT", "").strip()
        resolved_timeout = float(global_timeout) if global_timeout else None

    if not api_key or not base_url or not model:
        raise ValueError(
            "缺少 LLM 配置，请检查 .env 中的 API_KEY、BASE_URL、MODEL"
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
    if resolved_timeout is not None and resolved_timeout > 0:
        kwargs["request_timeout"] = float(resolved_timeout)

    llm = ChatOpenAI(**kwargs)
    register_llm(llm, model=model)
    return llm


def compute_retry_delay_seconds(
    attempt: int,
    *,
    base_delay_seconds: float | None = None,
    backoff_multiplier: float | None = None,
    max_delay_seconds: float | None = None,
) -> float:
    """根据重试次数计算指数退避时长。attempt 从 1 开始。"""
    _load_env_once()
    normalized_attempt = max(1, int(attempt))
    base_delay = float(os.getenv("LLM_RETRY_DELAY", "2.0")) if base_delay_seconds is None else float(base_delay_seconds)
    multiplier = float(os.getenv("LLM_RETRY_BACKOFF_MULTIPLIER", "2.0")) if backoff_multiplier is None else float(backoff_multiplier)
    max_delay = float(os.getenv("LLM_RETRY_MAX_DELAY", "20.0")) if max_delay_seconds is None else float(max_delay_seconds)
    delay = base_delay * (multiplier ** (normalized_attempt - 1))
    return max(0.0, min(delay, max_delay))


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
    """调用 LLM 并解析 JSON，失败时按统一策略重试。"""
    _load_env_once()
    pattern = json_block_pattern or DEFAULT_JSON_BLOCK_PATTERN
    attempts = max(1, int(max_attempts or os.getenv("LLM_RETRY_ATTEMPTS", "3")))
    delay_seconds = float(os.getenv("LLM_RETRY_DELAY", "2.0")) if retry_delay_seconds is None else float(retry_delay_seconds)
    raw_content = ""
    failure_reason = "no_attempt_completed"

    for attempt in range(1, attempts + 1):
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            print(f"{log_prefix} 已到调用截止时间，停止后续重试。")
            break
        parse_failed = False
        try:
            record_llm_request(llm, operation=log_prefix)
            response = llm.invoke(messages)
            record_llm_response(llm, response, operation=log_prefix)
            raw_content = _normalize_response_text(getattr(response, "content", ""))
            try:
                cleaned = pattern.sub("", raw_content).strip()
                try:
                    parsed = _parse_json_response(cleaned)
                except json.JSONDecodeError:
                    # 已知笔误（如 "x": 226, 190 缺 "y": 键）先修复再试一次，
                    # 避免为可机器修复的输出浪费整轮重试。
                    repaired = _repair_common_json_typos(cleaned)
                    if repaired != cleaned:
                        parsed = _parse_json_response(repaired)
                    else:
                        raise
                if not isinstance(parsed, dict):
                    raise ValueError(
                        "LLM JSON 顶层必须是 object，"
                        f"实际为 {type(parsed).__name__}"
                    )
                return parsed
            except (json.JSONDecodeError, ValueError) as parse_exc:
                parse_failed = True
                raise parse_exc
        except Exception as exc:
            record_llm_failure(llm, operation=log_prefix)
            kind = "解析" if parse_failed else "调用"
            failure_reason = f"{kind}失败: {type(exc).__name__}"
            print(f"{log_prefix} LLM {kind}失败（第 {attempt}/{attempts} 次）: {exc}\n[原始内容]: {raw_content}")
            if attempt >= attempts:
                break
            # 解析失败重试同样的 prompt 意义不大，用较短固定间隔快速重试；
            # 网络/调用失败走指数退避。
            if parse_failed:
                sleep_seconds = min(delay_seconds, 1.0)
            else:
                sleep_seconds = compute_retry_delay_seconds(
                    attempt,
                    base_delay_seconds=delay_seconds,
                )
            if deadline_monotonic is not None:
                remaining = deadline_monotonic - time.monotonic()
                if remaining <= 0:
                    print(f"{log_prefix} 已到调用截止时间，停止后续重试。")
                    break
                sleep_seconds = min(sleep_seconds, remaining)
            time.sleep(sleep_seconds)

    return _FailedJsonResult(failure_reason)
