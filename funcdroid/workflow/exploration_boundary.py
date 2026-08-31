"""Exploration boundary rule engine (generic).

Allows configuring "exploration boundary restrictions" for different target apps: when entering
third-party service content within the target app during runtime (such as mini-programs, embedded web pages, game centers, etc.),
automatically exits and ends the current scenario, keeping exploration focused on the target app's own core functionality.

Design constraints:
- This module contains no app-specific logic; rules come entirely from external JSON config files;
- Apps without configured rules behave identically to when this module is not introduced (default zero impact);
- The rule file path is specified by the environment variable ``BOUNDARY_RULES_FILE``, defaulting to
  ``boundary_rules.json`` under the project root; gracefully degrades to "all disabled" if the file does not exist.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

_RULES_CACHE: tuple[str | None, dict[str, Any]] | None = None
_EMPTY_RULES: dict[str, Any] = {}


@dataclass(frozen=True)
class BoundaryRule:
    """Boundary rules for a single app."""

    package_name: str
    enabled: bool = False
    label: str = "third-party service content"
    activity_markers: tuple[str, ...] = ()
    process_markers: tuple[str, ...] = ()
    exit_action: str = "press_back"

    def to_prompt_hint(self) -> str:
        """Generate generic constraint text injectable into planning prompts."""
        if not self.enabled:
            return ""
        return (
            f"Boundary restriction for this app: Do not plan navigation into {self.label}"
            " (such as mini-programs, embedded web pages, and other third-party service content);"
            " In-app entry points may open such content, but entering is considered leaving the target app boundary."
        )


@dataclass(frozen=True)
class BoundaryHit:
    """A boundary hit result."""

    rule: BoundaryRule
    via: str          # "activity" | "process"
    matched: str      # Matched feature string
    evidence: str = ""  # Raw observation value (Activity name or process snapshot fragment)


def _rules_file() -> str:
    return os.getenv(
        "BOUNDARY_RULES_FILE",
        os.path.join(os.getcwd(), "boundary_rules.json"),
    )


def _load_rules() -> tuple[str | None, dict[str, Any]]:
    """Load the rule file; includes module-level caching, degrades to empty rules when file is missing/corrupted."""
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
        print(f"[Exploration Boundary] Failed to read rule file, degraded to disabling boundary restrictions: {exc}")
        rules = _EMPTY_RULES
    _RULES_CACHE = (path, rules)
    return _RULES_CACHE


def _lower_markers(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(m).strip().lower() for m in value if str(m).strip())


def rule_for(package_name: str) -> BoundaryRule:
    """Return boundary rules for the specified app; returns disabled rule if unconfigured."""
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
    """Check if the target app exploration boundary has currently been crossed.

    Args:
        package_name: Target app package name.
        activity: Current foreground Activity name (Method 1: Activity name feature matching).
        process_snapshot: Optional process snapshot (e.g. ``ps -A`` output), used for fallback determination
            only when Activity evaluation cannot reach a conclusion.

    Returns:
        Returns :class:`BoundaryHit` on hit, otherwise returns None.
    """
    rule = rule_for(package_name)
    if not rule.enabled:
        return None

    activity = (activity or "").strip()
    if activity and rule.activity_markers:
        lowered = activity.lower()
        for marker in rule.activity_markers:
            if marker and marker in lowered:
                return BoundaryHit(rule=rule, via="activity", matched=marker, evidence=activity)

    snapshot = (process_snapshot or "").strip()
    if snapshot and rule.process_markers:
        lowered = snapshot.lower()
        for marker in rule.process_markers:
            if marker and marker in lowered:
                return BoundaryHit(rule=rule, via="process", matched=marker, evidence=snapshot[:300])
    return None


def enabled_packages() -> list[str]:
    """Return all package names with enabled boundary rules (for testing/diagnostics)."""
    _, rules = _load_rules()
    return sorted(
        name for name, entry in rules.items()
        if isinstance(entry, dict) and entry.get("enabled")
    )


def reset_cache() -> None:
    """Clear rule cache (for testing)."""
    global _RULES_CACHE
    _RULES_CACHE = None
