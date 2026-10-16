"""LLM 初始化。"""

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
# 场景执行阶段要求纯 JSON 输出。开启后由服务端保证返回合法 JSON，根治
# “模型把整个响应写成自然语言”的解析失败；OpenAI 兼容端点若不识别该参数，
# 调用侧会依据下面的特征词自动关闭（见 disable_json_response_format）。
JSON_RESPONSE_FORMAT_PAYLOAD = {"type": "json_object"}
JSON_FORMAT_REJECTION_PATTERN = re.compile(
    r"response_format|json_object|json\s*mode|unsupported\s+parameter|"
    r"unrecognized\s+request\s+argument|invalid\s+parameter",
    re.IGNORECASE,
)


def json_format_rejected(exc: BaseException) -> bool:
    """判断异常是否为服务端拒绝 response_format 参数。"""
    return bool(JSON_FORMAT_REJECTION_PATTERN.search(str(exc) or ""))


def disable_json_response_format(llm) -> bool:
    """服务端拒绝 response_format 后，把该实例切回自由文本输出。

    LangChain 每次请求都会合并 model_kwargs，因此原地移除即可让后续请求
    不再携带该参数。返回是否真的移除了（未开启时返回 False）。
    """
    model_kwargs = getattr(llm, "model_kwargs", None)
    if isinstance(model_kwargs, dict) and "response_format" in model_kwargs:
        model_kwargs.pop("response_format", None)
        print("[LLM] 已关闭 response_format=json_object，后续请求回退到自由文本输出。")
        return True
    return False


class _ReasoningAwareChatOpenAI(ChatOpenAI):
    """补上 ChatOpenAI 官方声明会丢弃的第三方 reasoning_content 字段。

    langchain_openai 的文档明确写着：非标准响应字段（如 DeepSeek 等第三方
    端点返回的 reasoning_content）不会被提取或保留，只解析官方 OpenAI 字段。
    openai SDK 的响应对象 extra="allow"，原始字段其实还在，只是
    ChatOpenAI._create_chat_result 没读它——库内对 refusal/parsed 就是用
    同样的手法从原始响应对象里搬进 additional_kwargs 的，这里照抄一份。
    """

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

# 已知的高频 JSON 笔误修复模式：本地小模型常把坐标写成
# "x": 226, 190（丢失 "y": 键名）。仅在严格解析失败后作为兜底尝试，
# 只修复 x/y 坐标键的有限非负数字；实际范围由执行器按当前图片宽高校验。
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
            if _is_coordinate_number(match.group(1))
            and _is_coordinate_number(match.group(2))
            else match.group(0)
        ),
        text,
    )
    if repaired != text:
        print("[LLM] 已修复缺失坐标键名的 JSON 笔误（\"x\": a, b → \"x\": a, \"y\": b）")
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
        print("[LLM] 已修复数组坐标的 JSON 笔误（\"x\": [a, b] → \"x\": a, \"y\": b）")
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
        # 线性扫描：raw_decode 直接在原文上按偏移解析（不切片复制），一次
        # 成功解析后跳到该对象末尾继续找下一个 "{"，因此外层对象内部的
        # evidence 等嵌套对象不会再成为候选，只保留“最后一个顶层对象”。
        # 解析失败的 "{" 只前进一个字符，与原先逐个左花括号尝试的语义一致。
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
            print("[LLM] 已从附带解释文字的响应中提取最后一个顶层 JSON object。")
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


# 未配置 LLM_REQUEST_TIMEOUT 时的默认单请求超时（秒）。不传 request_timeout
# 时 openai 客户端的 timeout=None 表示永不超时，一次挂起的请求就会让整个
# 长时运行卡死在某个节点；因此始终下发一个有限超时。
DEFAULT_LLM_REQUEST_TIMEOUT_SECONDS = 180.0


def _resolve_request_timeout(timeout_seconds: float | None) -> float:
    """解析单请求超时：显式参数 > LLM_REQUEST_TIMEOUT > 默认 180s。

    显式传入或环境变量给出 0/负数/非法值时同样回退到默认值，而不是
    关闭超时——框架自身已有重试与节点恢复，“无限等待”没有正当用途。
    """
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
    """根据环境变量创建聊天模型；所有角色共用同一个端点。

    reasoning_effort 显式传入时优先（用于按角色差异化思考强度，例如场景执行
    用 none 关闭思考、其余阶段用全局值）；传 None 时回退到全局 LLM_REASONING_EFFORT。

    json_response_format 为 True 时要求服务端返回合法 JSON（仅场景执行使用；
    端点不支持时由调用侧捕获并自动关闭）。
    """
    _load_env_once()

    # 通用变量（API_KEY/BASE_URL/MODEL）优先；未配置时回退到 DEEPSEEK_* 变量，
    # 便于在同一份 .env 中保留两套接入点，注释掉通用三件套即可切到 DeepSeek。
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
    kwargs["request_timeout"] = resolved_timeout
    if resolved_effort in {"none", "disabled", "off"}:
        if model.lower().startswith("deepseek"):
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        else:
            kwargs["reasoning_effort"] = "none"
    elif resolved_effort:
        kwargs["reasoning_effort"] = resolved_effort
    if json_response_format:
        # 通过 model_kwargs 注入，保持返回对象仍是 ChatOpenAI 实例，
        # 用量统计与重试逻辑不受影响。
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
    """根据重试次数计算指数退避时长。attempt 从 1 开始。"""
    _load_env_once()
    normalized_attempt = max(1, int(attempt))
    base_delay = float(os.getenv("LLM_RETRY_DELAY", "2.0")) if base_delay_seconds is None else float(base_delay_seconds)
    multiplier = float(os.getenv("LLM_RETRY_BACKOFF_MULTIPLIER", "2.0")) if backoff_multiplier is None else float(backoff_multiplier)
    max_delay = float(os.getenv("LLM_RETRY_MAX_DELAY", "20.0")) if max_delay_seconds is None else float(max_delay_seconds)
    delay = base_delay * (multiplier ** (normalized_attempt - 1))
    return max(0.0, min(delay, max_delay))


def _build_json_retry_feedback(raw_content: str, exc: BaseException) -> str:
    """解析失败后追加一条纠偏消息，而不是原样重发相同 prompt 盲重试。

    温度已固定为 0，若不告知模型上一次输出哪里有问题，重试很大概率复现
    同类偏差（例如整段自然语言、无花括号）。零花括号是最常见且最容易
    识别的偏差，单独给出更强硬的措辞。
    """
    if "{" not in str(raw_content or ""):
        return (
            "上一次响应是纯文字说明，完全没有包含任何 JSON 结构，无法解析。"
            "请只输出一个 JSON object 本身，不要输出解释性文字、不要用 markdown 代码块围栏。"
        )
    return (
        f"上一次响应解析失败：{str(exc)[:300]}。"
        "请只输出一个完整、合法的 JSON object 本身，不要包含解释性文字或 markdown 代码块围栏。"
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
    """调用 LLM 并解析 JSON，失败时按统一策略重试。"""
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
            print(f"{log_prefix} 已到调用截止时间，停止后续重试。")
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
                record_llm_diagnostic(response, operation=log_prefix, elapsed_seconds=time.monotonic() - request_started)
                return parsed
            except (json.JSONDecodeError, ValueError) as parse_exc:
                parse_failed = True
                raise parse_exc
        except Exception as exc:
            record_llm_diagnostic(response, operation=log_prefix, error=f"{type(exc).__name__}: {exc}", elapsed_seconds=time.monotonic() - request_started)
            record_llm_failure(llm, operation=log_prefix)
            kind = "解析" if parse_failed else "调用"
            failure_reason = f"{kind}失败: {type(exc).__name__}"
            no_json_hint = " (响应零JSON结构)" if parse_failed and raw_content and "{" not in raw_content else ""
            print(f"{log_prefix} LLM {kind}失败{no_json_hint}（第 {attempt}/{attempts} 次）: {exc}\n[原始内容]: {raw_content}")
            if attempt >= attempts:
                break
            # 解析失败重试同样的 prompt 意义不大，用较短固定间隔快速重试；
            # 网络/调用失败走指数退避。
            if parse_failed:
                sleep_seconds = min(delay_seconds, 1.0)
                # 温度为 0，原样重发同一 prompt 命中同类偏差的概率不低；
                # 追加一条纠偏消息而非盲重试。基于 base_messages 重建，
                # 避免连续失败时反馈消息在历史里无限堆叠。
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
                    print(f"{log_prefix} 已到调用截止时间，停止后续重试。")
                    break
                sleep_seconds = min(sleep_seconds, remaining)
            time.sleep(sleep_seconds)

    return _FailedJsonResult(failure_reason)
