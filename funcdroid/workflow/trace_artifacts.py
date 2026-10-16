"""Persist every runtime observation screenshot independently of the FFG.

These files are diagnostic evidence, not Page representative/evidence images.  They
are written as soon as an observation is captured so a later empty FFG patch,
Page reuse, process interruption, or evidence-image cap cannot erase the raw run
history.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from model.flow_graph import Page


_LOCK = threading.Lock()
_OUTPUT_DIR: Path | None = None
_SCREENSHOTS_DIR: Path | None = None
_INDEX_PATH: Path | None = None
_SEQUENCE = 0
_SAFE_TOKEN_PATTERN = re.compile(r"[^a-zA-Z0-9_-]+")


def _safe_token(value: object, fallback: str) -> str:
    token = _SAFE_TOKEN_PATTERN.sub("_", str(value or "").strip()).strip("_")
    return token[:40] or fallback


def _decode_screenshot(value: str) -> bytes | None:
    screenshot = str(value or "").strip()
    if not screenshot:
        return None
    if screenshot.startswith("data:image") and "," in screenshot:
        screenshot = screenshot.split(",", 1)[1]
    try:
        payload = base64.b64decode(screenshot, validate=False)
    except Exception:
        return None
    return payload or None


def _page_screenshot(page: Page | dict[str, Any] | None) -> str:
    if isinstance(page, Page):
        return str(page.representative_screenshot or "")
    if isinstance(page, dict):
        return str(
            page.get("representative_screenshot")
            or page.get("screenshot")
            or page.get("screenshot_base64")
            or ""
        )
    return ""


def _page_field(page: Page | dict[str, Any] | None, field: str) -> str:
    if isinstance(page, Page):
        return str(getattr(page, field, "") or "")
    if isinstance(page, dict):
        return str(page.get(field, "") or "")
    return ""


def _existing_max_sequence(index_path: Path) -> int:
    if not index_path.is_file():
        return 0
    maximum = 0
    try:
        with index_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                    maximum = max(maximum, int(record.get("sequence", 0) or 0))
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
    except OSError:
        return 0
    return maximum


def configure_trace_artifacts(output_dir: str, *, reset: bool = True) -> None:
    """Configure the per-run raw trace screenshot directory."""
    global _OUTPUT_DIR, _SCREENSHOTS_DIR, _INDEX_PATH, _SEQUENCE
    root = Path(str(output_dir or "").strip())
    with _LOCK:
        _OUTPUT_DIR = root
        _SCREENSHOTS_DIR = root / "trace_screenshots"
        _INDEX_PATH = _SCREENSHOTS_DIR / "index.jsonl"
        _SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        if reset and _INDEX_PATH.exists():
            # New runs are prevented from targeting a non-empty directory by main;
            # this only keeps direct/test callers deterministic.
            _INDEX_PATH.unlink()
        _SEQUENCE = _existing_max_sequence(_INDEX_PATH)
    print(
        "[轨迹截图] 已启用逐步原始截图落盘: "
        f"{_SCREENSHOTS_DIR} (mode={'reset' if reset else 'append'})"
    )


def disable_trace_artifacts() -> None:
    """Disable persistence; mainly used to isolate direct unit-test callers."""
    global _OUTPUT_DIR, _SCREENSHOTS_DIR, _INDEX_PATH, _SEQUENCE
    with _LOCK:
        _OUTPUT_DIR = None
        _SCREENSHOTS_DIR = None
        _INDEX_PATH = None
        _SEQUENCE = 0


def trace_screenshot_count() -> int:
    with _LOCK:
        return _SEQUENCE


def save_trace_screenshot(
    page: Page | dict[str, Any] | None,
    *,
    phase: str,
    step_index: int,
    moment: str,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Write one screenshot and append a crash-tolerant JSONL index record.

    Returns a path relative to the run output directory, or an empty string when
    no decodable screenshot/output directory is available.
    """
    global _SEQUENCE
    payload = _decode_screenshot(_page_screenshot(page))
    if not payload:
        return ""
    with _LOCK:
        if _OUTPUT_DIR is None or _SCREENSHOTS_DIR is None or _INDEX_PATH is None:
            return ""
        try:
            _SEQUENCE += 1
            sequence = _SEQUENCE
            phase_token = _safe_token(phase, "unknown")
            moment_token = _safe_token(moment, "observation")
            filename = (
                f"{sequence:05d}_{phase_token}_step_{max(0, int(step_index)):03d}_"
                f"{moment_token}.jpg"
            )
            target = _SCREENSHOTS_DIR / filename
            tmp = target.with_suffix(target.suffix + ".tmp")
            with tmp.open("wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, target)
            relative = target.relative_to(_OUTPUT_DIR).as_posix()
            record = {
                "sequence": sequence,
                "captured_at": datetime.now().astimezone().isoformat(),
                "phase": str(phase or ""),
                "step_index": int(step_index),
                "moment": str(moment or ""),
                "file": relative,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "page_id": _page_field(page, "id"),
                "page_name": _page_field(page, "name"),
                "activity": _page_field(page, "activity"),
                "metadata": dict(metadata or {}),
            }
            with _INDEX_PATH.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        except Exception as exc:
            print(
                "[轨迹截图] 落盘失败（不中断主流程）: "
                f"phase={phase or '未知'}, step={step_index}, "
                f"moment={moment or '观测'}, error={exc}"
            )
            return ""
    print(
        "[轨迹截图] 已保存: "
        f"seq={sequence}, phase={phase or '未知'}, step={step_index}, "
        f"moment={moment or '观测'}, page={record['page_id'] or '未知'}, "
        f"activity={record['activity'] or '未知'}, "
        f"observes_previous_action="
        f"{record['metadata'].get('previous_action_name') or '无'}, "
        f"file={relative}, bytes={len(payload)}, sha256={record['sha256'][:12]}"
    )
    return relative
