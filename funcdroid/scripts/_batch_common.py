"""Shared process, adb, and artifact helpers for APK and installed-app batch runners."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
from pathlib import Path
from typing import Sequence

DEFAULT_BATCH_ADB_COMMAND_TIMEOUT = max(
    1.0,
    float(os.getenv("BATCH_ADB_COMMAND_TIMEOUT", "30")),
)
# Finite subprocess timeout when main.py uses total-time=0 (unlimited).
DEFAULT_BATCH_PROCESS_TIMEOUT = max(
    60.0,
    float(os.getenv("BATCH_PROCESS_TIMEOUT", "7200")),
)
BATCH_PROCESS_TIMEOUT_GRACE = max(
    0.0,
    float(os.getenv("BATCH_PROCESS_TIMEOUT_GRACE", "300")),
)
BATCH_PROCESS_TERMINATE_GRACE = max(
    1.0,
    float(os.getenv("BATCH_PROCESS_TERMINATE_GRACE", "15")),
)


class BatchInterrupted(RuntimeError):
    """Raised on SIGINT/SIGTERM so the queue stops after cleaning up the current app."""


def signal_handler(signum, _frame):
    raise BatchInterrupted(f"received_signal_{signum}")


def run_command(
    command: Sequence[str],
    *,
    check: bool = False,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        check=check,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def adb(
    device_id: str,
    *args: str,
    check: bool = False,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    return run_command(
        ["adb", "-s", device_id, *args],
        check=check,
        timeout=DEFAULT_BATCH_ADB_COMMAND_TIMEOUT if timeout is None else timeout,
    )


def query_installed(adb_call, device_id: str, package_name: str) -> bool:
    """Query installation; transport/PackageManager errors mean unknown state and raise RuntimeError.

    Unlike pm path, a successful pm list packages <filter> can explicitly return no matches.
    """
    try:
        result = adb_call(device_id, "shell", "pm", "list", "packages", package_name)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"Cannot query app installation state ({package_name}): {exc}") from exc
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if result.returncode != 0 or result.stderr.strip() or any(not line.startswith("package:") for line in lines):
        raise RuntimeError(
            f"Cannot query app installation state ({package_name}, exit={result.returncode}): "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    return f"package:{package_name}" in lines


def resolve_process_timeout(
    total_time_minutes: float,
    explicit_timeout_seconds: float | None = None,
    grace_seconds: float = BATCH_PROCESS_TIMEOUT_GRACE,
    unlimited_timeout_seconds: float = DEFAULT_BATCH_PROCESS_TIMEOUT,
) -> float:
    """Use an explicit hard timeout, budget plus grace, or a finite fallback for an unlimited budget."""
    if explicit_timeout_seconds is not None:
        return max(1.0, float(explicit_timeout_seconds))
    if total_time_minutes > 0:
        return total_time_minutes * 60 + grace_seconds
    return unlimited_timeout_seconds


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be finite and >= 0")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def signal_process_tree(process: subprocess.Popen, sig: int) -> None:
    if process.poll() is not None:
        return
    if os.name == "posix":
        try:
            os.killpg(process.pid, sig)
            return
        except ProcessLookupError:
            return
        except OSError:
            # If the process group is unavailable, signal the direct child as a fallback.
            pass
    if sig == signal.SIGTERM:
        process.terminate()
    else:
        process.kill()


def terminate_process_tree(
    process: subprocess.Popen,
    grace_seconds: float = BATCH_PROCESS_TERMINATE_GRACE,
    log_prefix: str = "[App queue]",
) -> None:
    if process.poll() is not None:
        return
    signal_process_tree(process, signal.SIGTERM)
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    signal_process_tree(process, signal.SIGKILL)
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        print(f"{log_prefix} Child process survived forced termination; stop waiting: pid={process.pid}")


def append_jsonl(path: Path, payload: dict) -> None:
    """Append one JSON record and fsync so interruption or power loss cannot discard it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        file_obj.flush()
        os.fsync(file_obj.fileno())


def run_artifacts_complete(
    output_dir: Path,
    mode: str,
    package_name: str,
    *,
    missing_mode_ok: bool = False,
    require_checks: bool = False,
) -> bool:
    """Require a final run result and complete artifacts, rather than exit code zero alone.

    missing_mode_ok accepts legacy artifacts without a mode field, as used by the installed-app runner.
    require_checks requires a report object with a checks list.
    """
    try:
        result = json.loads((output_dir / "run_result.json").read_text(encoding="utf-8"))
        manifest = json.loads((output_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
        report_name = "compliance_report.json" if mode == "compliance" else "bugs.json"
        report = json.loads((output_dir / report_name).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not (isinstance(result, dict) and isinstance(manifest, dict)):
        return False
    result_mode = result.get("mode", mode if missing_mode_ok else None)
    report_ok = (
        isinstance(report, dict) and isinstance(report.get("checks"), list)
        if require_checks
        else isinstance(report, (dict, list))
    )
    return (
        report_ok
        and result_mode == mode
        and result.get("status") == "success"
        and result.get("package_name") == package_name
        and manifest.get("status") == "complete"
    )
