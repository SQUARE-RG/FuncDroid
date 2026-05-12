from dotenv import load_dotenv
import os
from openai import OpenAI
from threading import Lock
from typing import Any, Dict

TOKEN_LOCK = Lock()

TOKEN_STATS: Dict[str, int] = {
    "calls": 0,
    "input_tokens": 0,
    "output_tokens": 0,
    "total_tokens": 0,
}

TOKEN_LOGS = []  # list[dict]

def _extract_usage(resp: Any) -> Dict[str, int]:
    """
    Try to extract usage tokens from OpenAI Responses API compatible object.
    Return dict with keys: input_tokens, output_tokens, total_tokens
    """
    usage = getattr(resp, "usage", None)
    if usage is None and isinstance(resp, dict):
        usage = resp.get("usage")

    # OpenAI Responses API: usage.{input_tokens, output_tokens, total_tokens}
    it = getattr(usage, "input_tokens", None) if usage is not None else None
    ot = getattr(usage, "output_tokens", None) if usage is not None else None
    tt = getattr(usage, "total_tokens", None) if usage is not None else None

    # dict fallback
    if usage is not None and isinstance(usage, dict):
        it = usage.get("input_tokens", it)
        ot = usage.get("output_tokens", ot)
        tt = usage.get("total_tokens", tt)

    # Some gateways may use prompt/completion naming
    if it is None and usage is not None:
        it = getattr(usage, "prompt_tokens", None)
    if ot is None and usage is not None:
        ot = getattr(usage, "completion_tokens", None)
    if tt is None and usage is not None:
        tt = getattr(usage, "total_tokens", None)

    # ensure int
    it = int(it) if it is not None else 0
    ot = int(ot) if ot is not None else 0
    tt = int(tt) if tt is not None else (it + ot)

    return {"input_tokens": it, "output_tokens": ot, "total_tokens": tt}

def _add_usage(resp: Any, tag: str = "", model: str = "") -> Dict[str, int]:
    """
    Update global TOKEN_STATS and (optional) TOKEN_LOGS.
    Return this-call usage dict.
    """
    u = _extract_usage(resp)

    with TOKEN_LOCK:
        TOKEN_STATS["calls"] += 1
        TOKEN_STATS["input_tokens"] += u["input_tokens"]
        TOKEN_STATS["output_tokens"] += u["output_tokens"]
        TOKEN_STATS["total_tokens"] += u["total_tokens"]

        # 可选：记录每次调用明细
        TOKEN_LOGS.append({
            "tag": tag,
            "model": model,
            "input_tokens": u["input_tokens"],
            "output_tokens": u["output_tokens"],
            "total_tokens": u["total_tokens"],
        })
        # 可选：限制长度，避免跑太久内存涨
        if len(TOKEN_LOGS) > 2000:
            del TOKEN_LOGS[:1000]

    return u


load_dotenv()



client_llm = OpenAI(
    base_url="https://api2.aigcbest.top/v1",
    # api_key="sk-DxXcnt6PCCKHOO3BQ27K2h8Cdbo4zWALMUuOUG8dwe6hEktk",
    api_key = "sk-WtUTeeyKVeQNsBqYVZLaOe1h5iAyHaTLqp2hdCOliGPRFwxo",
)

client_uitars = OpenAI(
    base_url=os.getenv("SPECIALIZED_BASE_URL"),
    api_key=os.getenv("SPECIALIZED_API_KEY"),
)

# def ask_llm(content):
#     resp = client_llm.responses.create(
#         model="gpt-4o",
#         input=[{
#             "role": "user",
#             "content": content
#         }],
#         temperature=0,
#     )
#     return resp.output_text


def ask_llm(content):
    resp = client_uitars.responses.create(
        model=os.getenv("SPECIALIZED_MODEL"),
        input=[{
            "role": "user",
            "content": content
        }],
        temperature=0,
        extra_body={
            "thinking": {
                "type": "disabled",
            },
        },
    )
    # print(resp.output_text)
    _add_usage(resp, tag="ask_llm", model=os.getenv("SPECIALIZED_MODEL") or "")
    return resp.output_text


def ask_uitars(content):
    resp = client_uitars.responses.create(
        model=os.getenv("SPECIALIZED_MODEL"),
        input=[{
            "role": "user",
            "content": content
        }],
        temperature=0,
        extra_body={
            "thinking": {
                "type": "disabled",
            },
        },
    )
    # print(resp.output_text)
    _add_usage(resp, tag="ask_uitars", model=os.getenv("SPECIALIZED_MODEL") or "")
    return resp.output_text

def ask_uitars_without_thinking(content):
    resp = client_uitars.responses.create(
        model=os.getenv("SPECIALIZED_MODEL"),
        input=[{
            "role": "user",
            "content": content
        }],
        temperature=0,
        extra_body={
            "thinking": {
                "type": "disabled",
            },
        },
    )
    _add_usage(resp, tag="ask_uitars_without_thinking", model=os.getenv("SPECIALIZED_MODEL") or "")
    return resp.output_text


def ask_uitars_messages(messages):
    resp = client_uitars.responses.create(
        model=os.getenv("SPECIALIZED_MODEL"),
        input=messages,
        temperature=0,
        extra_body={
            "thinking": {
                "type": "disabled",
            },
        },
    )
    _add_usage(resp, tag="ask_uitars_messages", model=os.getenv("SPECIALIZED_MODEL") or "")
    return resp.output_text




    


