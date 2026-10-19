"""Identify target-process crash and ANR evidence in logcat.

Exclude graphics-stack noise and other processes; report a defect only when evidence belongs to the tested package."""

from __future__ import annotations

import re

# A log containing only these SurfaceFlinger timestamp fragments is graphics noise, not a crash.
NON_CRASH_LOGCAT_PATTERNS = (
    "BufferLayer",
    "Timestamp",
    "seems implausible",
    "expectedPresent",
)


def _logcat_has_target_process_crash(crash_logcat: str, package_name: str) -> bool:
    text = str(crash_logcat or "")
    package_name = str(package_name or "").strip()
    if not text or not package_name:
        return False
    non_empty_lines = [line for line in text.splitlines() if line.strip()]
    if non_empty_lines and all(
        all(pattern in line for pattern in NON_CRASH_LOGCAT_PATTERNS)
        for line in non_empty_lines
    ):
        return False

    escaped_package = re.escape(package_name)
    if re.search(rf"\bANR in\s+{escaped_package}\b", text):
        return True
    if re.search(rf"\bProcess:\s*{escaped_package}\b", text) and "FATAL EXCEPTION" in text:
        return True
    if "Fatal signal" in text and (
        f">>> {package_name} <<<" in text
        or re.search(rf"\bCmdline:\s*{escaped_package}\b", text)
    ):
        return True
    strong_crash_signal = any(
        signal in text
        for signal in ("FATAL EXCEPTION", "AndroidRuntime", "Fatal signal", "App crash", "ANR in")
    )
    if strong_crash_signal and re.search(rf"\bForce finishing activity\s+{escaped_package}/", text):
        return True
    if strong_crash_signal and re.search(rf"\bProcess\s+{escaped_package}\s+\(pid\s+\d+\)\s+has died\b", text):
        return True
    return False


# Public alias retained alongside the private compatibility name.
logcat_has_target_process_crash = _logcat_has_target_process_crash
