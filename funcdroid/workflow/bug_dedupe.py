"""Shared bug deduplication helpers."""

from __future__ import annotations

import re
from typing import Any


TEXT_NORMALIZE_PATTERN = re.compile(r"[\"'“”‘’（）()【】\[\]{}<>《》、，,。.：:；;!！?？\-_/\s]+")
# 每次运行 PID/TID 都不同；不剥离会让同一次崩溃被判成两个缺陷重复上报。
CRASH_VOLATILE_PATTERN = re.compile(r"\b(pid|tid)\s*[:：]?\s*\d+", re.IGNORECASE)
# 对象哈希（如 Foo@1a2b3c4）每次运行也不同。
CRASH_OBJECT_HASH_PATTERN = re.compile(r"@[0-9a-f]{5,}\b", re.IGNORECASE)
# logcat 行前缀（时间戳、级别/标签、进程号）每次出现都不同，必须先剥离再提取特征：
#   -v time:       10-05 12:00:01.123 E/AndroidRuntime( 4321): FATAL EXCEPTION: main
#   -v threadtime: 10-05 12:00:01.123  4321  4350 E AndroidRuntime: FATAL EXCEPTION: main
#   -v brief:      E/AndroidRuntime( 4321): FATAL EXCEPTION: main
_LOGCAT_TIMESTAMP = r"(?:\d{4}-)?\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?"
LOGCAT_PREFIX_PATTERNS = (
    re.compile(rf"^{_LOGCAT_TIMESTAMP}\s+\d+\s+\d+\s+[VDIWEFA]\s+[^:]*?:\s?"),
    re.compile(rf"^(?:{_LOGCAT_TIMESTAMP}\s+)?[VDIWEFA]/[^(:]*?\(\s*\d+\)\s*:\s?"),
)
CRASH_EXCEPTION_LINE_PATTERN = re.compile(
    r"^(?:caused by:\s*)?(?:[a-z_$][\w$]*\.)+[A-Za-z_$][\w$]*(?:exception|error)\b",
    re.IGNORECASE,
)


def normalize_bug_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    return TEXT_NORMALIZE_PATTERN.sub("", text)


def strip_logcat_prefix(line: str) -> str:
    """剥离单行 logcat 的时间戳/级别/标签/进程号前缀，返回正文。"""
    text = str(line or "")
    for pattern in LOGCAT_PREFIX_PATTERNS:
        stripped, count = pattern.subn("", text, count=1)
        if count:
            return stripped.strip()
    return text.strip()


def _crash_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw_line in str(text or "").splitlines():
        line = strip_logcat_prefix(raw_line)
        if not line:
            continue
        # PID/对象哈希是每次运行的易变字段，必须在选特征行之前剥离，
        # 否则同一崩溃在两次复现里会得到不同的身份键。
        line = CRASH_VOLATILE_PATTERN.sub(lambda match: f"{match.group(1).lower()}:", line)
        line = CRASH_OBJECT_HASH_PATTERN.sub("@", line)
        lines.append(line)
    return lines


def _extract_crash_signature(text: str) -> str:
    if not text:
        return ""
    lines = _crash_lines(text)
    # 首选“异常类行 + 第一帧 at”：与时间、进程号、线程名无关，最能代表同一根因。
    for position, line in enumerate(lines):
        if not CRASH_EXCEPTION_LINE_PATTERN.search(line):
            continue
        frame = next((item for item in lines[position + 1:] if item.startswith("at ")), "")
        return normalize_bug_text(f"{line} {frame}")[:240]

    interesting: list[str] = []
    for line in lines:
        lowered = line.lower()
        if "fatal exception" in lowered:
            interesting.append(line)
            continue
        if "process:" in lowered:
            interesting.append(line)
            continue
        if "exception:" in lowered or lowered.startswith("java.") or lowered.startswith("kotlin."):
            interesting.append(line)
            continue
        if "signal " in lowered and "pid:" in lowered:
            interesting.append(line)
            continue
        if interesting and line.startswith("at "):
            interesting.append(line)
            break
    signature = normalize_bug_text(" ".join(interesting[:4]))
    return signature[:240]


def bug_identity_key(bug: dict[str, Any]) -> str:
    bug_type = str(bug.get("type") or "").strip().lower()
    description_key = normalize_bug_text(bug.get("description"))
    actual_key = normalize_bug_text(bug.get("actual"))
    target_key = normalize_bug_text(bug.get("target_function_id"))

    if bug_type == "crash":
        actual_text = str(bug.get("actual") or "")
        crash_signature = _extract_crash_signature(actual_text)
        # 兜底同样基于剥离 logcat 前缀后的正文，避免时间戳让每次出现都成为新键。
        stable_actual = normalize_bug_text(" ".join(_crash_lines(actual_text)))
        evidence = crash_signature or stable_actual or description_key
        if not evidence:
            return ""
        return f"crash|{target_key[:64]}|{evidence[:240]}"

    # 场景文本由 LLM 每轮重新生成，纳入身份键会让跨轮去重完全失效；
    # 同一功能下用户可见后果相同即视为同一缺陷。
    evidence = actual_key if len(actual_key) >= 12 else description_key
    if not evidence:
        return ""
    return f"functional|{target_key[:64]}|{evidence[:240]}"


def is_similar_bug_key(left: str, right: str) -> bool:
    if not left or not right:
        return False
    if left == right:
        return True
    shorter, longer = sorted((left, right), key=len)
    return len(shorter) >= 24 and shorter in longer


def dedupe_bugs(bugs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen_keys: list[str] = []
    for bug in bugs:
        if not isinstance(bug, dict):
            continue
        key = bug_identity_key(bug)
        if key and any(is_similar_bug_key(key, seen) for seen in seen_keys):
            continue
        deduped.append(bug)
        if key:
            seen_keys.append(key)
    return deduped
