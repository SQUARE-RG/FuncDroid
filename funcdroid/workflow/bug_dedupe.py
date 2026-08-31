"""Shared bug deduplication helpers."""

from __future__ import annotations

import re
from typing import Any


TEXT_NORMALIZE_PATTERN = re.compile(r"[\"'\u201c\u201d\u2018\u2019\uff08\uff09()\u3010\u3011\[\]{}<>\u300a\u300b\u3001\uff0c,\u3002.\uff1a:\uff1b;!\uff01?\uff1f\-_/\s]+")


def normalize_bug_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    return TEXT_NORMALIZE_PATTERN.sub("", text)


def _extract_crash_signature(text: str) -> str:
    if not text:
        return ""
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
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
    scenario_key = normalize_bug_text(bug.get("scenario"))
    target_key = normalize_bug_text(bug.get("target_function_id"))

    if bug_type == "crash":
        crash_signature = _extract_crash_signature(str(bug.get("actual") or ""))
        evidence = crash_signature or actual_key or description_key
        if not evidence:
            return ""
        return f"crash|{target_key[:64]}|{evidence[:240]}"

    evidence = actual_key if len(actual_key) >= 12 else description_key
    if not evidence:
        return ""
    return f"functional|{target_key[:64]}|{scenario_key[:120]}|{evidence[:240]}"


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
