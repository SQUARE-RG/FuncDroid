"""Shared foreground probing, startup polling, and restart handling.

Polling avoids capturing the launcher during slow startup and provoking repeated force-stop/restart loops. Injectable probes preserve testability through caller wrappers."""

from __future__ import annotations

import os
import time
from typing import Any, Callable

from workflow.system_dialogs import is_permission_dialog

# Poll foreground state instead of sleeping a fixed interval during slow cold starts.
APP_START_POLL_INTERVAL = max(0.1, float(os.getenv("APP_START_POLL_INTERVAL", "0.5")))
APP_START_MAX_WAIT = max(1.0, float(os.getenv("APP_START_MAX_WAIT", "12")))
APP_RESTART_SETTLE_DELAY = max(0.0, float(os.getenv("APP_RESTART_SETTLE_DELAY", "1.0")))
APP_RESTART_DELAY = float(os.getenv("APP_RESTART_DELAY", "2.0"))
APP_RESTART_ATTEMPTS = max(1, int(os.getenv("APP_RESTART_ATTEMPTS", "2")))
APP_RESTART_RETRY_DELAY = float(os.getenv("APP_RESTART_RETRY_DELAY", "1.0"))
FOREGROUND_CHECK_RETRIES = max(1, int(os.getenv("FOREGROUND_CHECK_RETRIES", "2")))
FOREGROUND_CHECK_RETRY_DELAY = float(os.getenv("FOREGROUND_CHECK_RETRY_DELAY", "0.2"))
# Reserve enough budget for launch, foreground polling, and stabilization before restarting.
APP_RESTART_MAX_SECONDS = APP_START_MAX_WAIT + APP_RESTART_SETTLE_DELAY


def current_foreground_package(
    device,
    retries: int = FOREGROUND_CHECK_RETRIES,
    *,
    log_prefix: str = "[Scenario execution]",
) -> str | None:
    """Read the foreground package; return None when repeated probes fail or remain empty."""
    last_error: Exception | None = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            current_package = str(device.current_package() or "").strip()
            if current_package:
                return current_package
        except Exception as exc:
            last_error = exc
            print(f"{log_prefix} Could not query foreground package: {exc}")
        if attempt < retries:
            time.sleep(FOREGROUND_CHECK_RETRY_DELAY)
    if last_error is None:
        print(f"{log_prefix} Foreground package is empty; target state is unknown.")
    return None


def target_app_foreground_state(device, package_name: str) -> tuple[str, str]:
    """Return (foreground/background/unknown, observed package)."""
    package_name = str(package_name or "").strip()
    if not package_name:
        return "foreground", ""
    current_package = current_foreground_package(device)
    if current_package is None:
        return "unknown", ""
    if current_package == package_name:
        return "foreground", current_package
    return "background", current_package


def foreground_or_permission(state: str, current_package: str) -> bool:
    """Accept the target foreground or a system permission dialog covering it."""
    return state == "foreground" or is_permission_dialog(current_package)


def poll_until_foreground(
    probe: Callable[[], bool],
    *,
    max_wait: float | None = None,
    poll_interval: float | None = None,
) -> bool:
    """Poll until the probe succeeds or the deadline expires."""
    wait = APP_START_MAX_WAIT if max_wait is None else max(0.0, float(max_wait))
    interval = APP_START_POLL_INTERVAL if poll_interval is None else max(0.0, float(poll_interval))
    deadline = time.monotonic() + wait
    while True:
        if probe():
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(interval)


def wait_target_app_foreground(
    device,
    package_name: str,
    *,
    max_wait: float | None = None,
    state_probe: Callable[[Any, str], tuple[str, str]] | None = None,
) -> bool:
    """Wait for the target foreground or an overlaid system permission dialog."""
    package_name = str(package_name or "").strip()
    if not package_name:
        return True
    probe_state = state_probe or target_app_foreground_state
    return poll_until_foreground(
        lambda: foreground_or_permission(*probe_state(device, package_name)),
        max_wait=max_wait,
    )


def restart_until_foreground(
    device,
    package_name: str,
    *,
    wait_foreground: Callable[[Any, str], bool],
    on_ready: Callable[[], Any] | None = None,
    attempts: int = APP_RESTART_ATTEMPTS,
    log_prefix: str = "[Scenario execution]",
) -> tuple[bool, Any, list[Exception]]:
    """Restart and poll up to attempts times.

    Return success, on_ready result, and captured exceptions. Invoke on_ready after stabilization; its exceptions count as failed attempts."""
    errors: list[Exception] = []
    attempts = max(1, int(attempts))
    for attempt in range(1, attempts + 1):
        try:
            print(f"{log_prefix} Restart target application, attempt {attempt}/{attempts}.")
            device.restart_app(package_name)
            if wait_foreground(device, package_name):
                time.sleep(APP_RESTART_SETTLE_DELAY)
                return True, (on_ready() if on_ready is not None else None), errors
            print(f"{log_prefix} Target is still not in the foreground after restart.")
        except Exception as exc:
            errors.append(exc)
            print(f"{log_prefix} Target restart failed: {exc}")
        if attempt < attempts:
            time.sleep(APP_RESTART_RETRY_DELAY)
    return False, None, errors


def remaining_seconds(deadline: float | None) -> float | None:
    """Return remaining wall-clock seconds, or None without a deadline."""
    if deadline is None:
        return None
    return float(deadline) - time.time()


def deadline_allows(deadline: float | None, seconds: float) -> bool:
    """Check whether enough time remains for the requested wait or restart."""
    remaining = remaining_seconds(deadline)
    return remaining is None or remaining > max(0.0, float(seconds))
