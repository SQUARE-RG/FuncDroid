"""Configurable application exploration boundaries.

External JSON rules identify embedded third-party services and end the current scenario when crossed. No application-specific rules are hard-coded. Missing rules disable the feature; BOUNDARY_RULES_FILE overrides boundary_rules.json."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

_RULES_CACHE: tuple[str | None, dict[str, Any]] | None = None
_EMPTY_RULES: dict[str, Any] = {}


@dataclass(frozen=True)
class BoundaryRule:
    """Boundary rules for one application."""

    package_name: str
    enabled: bool = False
    label: str = "third-party service content"
    activity_markers: tuple[str, ...] = ()
    process_markers: tuple[str, ...] = ()
    exit_action: str = "press_back"

    def to_prompt_hint(self) -> str:
        """Build a generic boundary constraint for the planning prompt."""
        if not self.enabled:
            return ""
        return (
            f"Application boundary: do not plan entry into {self.label}"
            " (such as mini-apps or embedded third-party pages). "
            "An in-app entry can open this content, but doing so crosses the target application boundary."
        )


@dataclass(frozen=True)
class BoundaryHit:
    """One observed boundary match."""

    rule: BoundaryRule
    via: str          # "activity" | "process"
    matched: str      # Matched feature string.
    evidence: str = ""  # Original observed activity or process-snapshot fragment.


def _rules_file() -> str:
    return os.getenv(
        "BOUNDARY_RULES_FILE",
        os.path.join(os.getcwd(), "boundary_rules.json"),
    )


def _load_rules() -> tuple[str | None, dict[str, Any]]:
    """Load and cache rules; absent or invalid files yield empty rules."""
    global _RULES_CACHE
    path = _rules_file()
    if _RULES_CACHE is not None and _RULES_CACHE[0] == path:
        return _RULES_CACHE
    try:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        rules = raw if isinstance(raw, dict) else _EMPTY_RULES
    except FileNotFoundError:
        rules = _EMPTY_RULES
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[Exploration boundary] Cannot load rules; boundaries are disabled: {exc}")
        rules = _EMPTY_RULES
    _RULES_CACHE = (path, rules)
    return _RULES_CACHE


def _lower_markers(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(m).strip().lower() for m in value if str(m).strip())


def rule_for(package_name: str) -> BoundaryRule:
    """Return configured rules for a package, or a disabled rule."""
    package_name = (package_name or "").strip()
    if not package_name:
        return BoundaryRule(package_name=package_name)
    _, rules = _load_rules()
    entry = rules.get(package_name)
    if not isinstance(entry, dict) or not entry:
        return BoundaryRule(package_name=package_name)
    enabled = bool(entry.get("enabled", False))
    if not enabled:
        return BoundaryRule(package_name=package_name)
    return BoundaryRule(
        package_name=package_name,
        enabled=True,
        label=str(entry.get("label") or "third-party service content").strip(),
        activity_markers=_lower_markers(entry.get("activity_markers")),
        process_markers=_lower_markers(entry.get("process_markers")),
        exit_action=str(entry.get("exit_action") or "press_back").strip(),
    )


def check_boundary(
    package_name: str,
    activity: str | None = None,
    process_snapshot: str | None = None,
) -> BoundaryHit | None:
    """Detect whether execution crossed an application boundary.

    Prefer the current activity. Use an optional process snapshot only when the activity is unavailable: persistent child processes do not prove the boundary is currently active. The caller must first confirm the target foreground. Return BoundaryHit or None."""
    rule = rule_for(package_name)
    if not rule.enabled:
        return None

    activity = (activity or "").strip()
    if activity and rule.activity_markers:
        lowered = activity.lower()
        for marker in rule.activity_markers:
            if marker and marker in lowered:
                return BoundaryHit(rule=rule, via="activity", matched=marker, evidence=activity)

    if activity:
        return None
    snapshot = (process_snapshot or "").strip()
    if snapshot and rule.process_markers:
        lowered = snapshot.lower()
        for marker in rule.process_markers:
            if marker and marker in lowered:
                return BoundaryHit(rule=rule, via="process", matched=marker, evidence=snapshot[:300])
    return None


def enabled_packages() -> list[str]:
    """Return packages with enabled rules for diagnostics and tests."""
    _, rules = _load_rules()
    return sorted(
        name for name, entry in rules.items()
        if isinstance(entry, dict) and entry.get("enabled")
    )


def reset_cache() -> None:
    """Clear the rules cache for tests."""
    global _RULES_CACHE
    _RULES_CACHE = None
