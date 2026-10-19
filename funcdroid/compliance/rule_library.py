"""Load rule libraries and select checkpoints."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

DEFAULT_RULES_PATH = Path(__file__).resolve().parent / "rules.json"
LEGACY_RULES_PATH = Path(__file__).resolve().parent / "legacy_rules_v1.json"

# Prioritize initial launch, then permission denial, then form and settings discovery that can use the existing FFG.
CHECKPOINT_GROUP_ORDER = (
    "first_launch_flow",
    "permission_denial_flow",
    "consent_observation",
    "registration_form",
    "settings_search",
    "rights_entry_search",
    "policy_text_review",
)

GROUP_LABELS = {
    "first_launch_flow": "First launch and consent",
    "permission_denial_flow": "Permission-denial path",
    "consent_observation": "Separate consent for sensitive data",
    "registration_form": "Registration data minimization",
    "settings_search": "Settings disclosures and controls",
    "rights_entry_search": "User-rights entrypoints",
    "policy_text_review": "Privacy-policy text review",
}

GUI_DETECTABILITY = {"gui_direct", "gui_collect_llm"}


def _normalize_rules(raw: Any) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        rule_id = str(item.get("rule_id", "")).strip()
        clause = str(item.get("clause", "")).strip()
        if not rule_id or not clause:
            continue
        rules.append(item)
    return rules


@lru_cache(maxsize=4)
def _load_rules_cached(path: str) -> tuple[dict[str, Any], ...]:
    return tuple(_load_rules_uncached(path))


def _load_rules_uncached(path: str) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"Compliance rule library does not exist: {target}")
    with target.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return _normalize_rules(data.get("rules"))


def load_rules(rules_path: str | os.PathLike[str] | None = None) -> list[dict[str, Any]]:
    """Load rules using a process-local cache."""
    path = str(rules_path) if rules_path else str(DEFAULT_RULES_PATH)
    return list(_load_rules_cached(path))


def rule_by_id(rule_id: str, rules_path: str | os.PathLike[str] | None = None) -> dict[str, Any] | None:
    rule_id = str(rule_id or "").strip()
    for rule in load_rules(rules_path):
        if rule["rule_id"] == rule_id:
            return rule
    # Offline review can load legacy R IDs; new runs schedule current P01-P20 rules.
    if rules_path is None and rule_id.startswith("R"):
        for rule in load_rules(LEGACY_RULES_PATH):
            if rule["rule_id"] == rule_id:
                return rule
    return None


def gui_rules(rules_path: str | os.PathLike[str] | None = None) -> list[dict[str, Any]]:
    """Return rules screenable through GUI execution or captured-text review."""
    return [
        rule
        for rule in load_rules(rules_path)
        if rule.get("detectability") in GUI_DETECTABILITY
    ]


def pending_gui_checkpoints(
    checklist_status: dict[str, str] | None,
    rules_path: str | os.PathLike[str] | None = None,
) -> list[dict[str, Any]]:
    """Return pending GUI checkpoints in group order.

    Skip done, executing, and deferred checks. Deferred checks must be reopened explicitly after related FFG evidence changes."""
    from compliance.checkpoints import (
        CHECK_STATUS_DEFERRED,
        CHECK_STATUS_DONE,
        CHECK_STATUS_EXECUTING,
    )

    status_map = checklist_status if isinstance(checklist_status, dict) else {}
    pending: list[dict[str, Any]] = []
    for rule in gui_rules(rules_path):
        status = str(status_map.get(rule["rule_id"], "") or "").strip()
        if status in {CHECK_STATUS_DONE, CHECK_STATUS_EXECUTING, CHECK_STATUS_DEFERRED}:
            continue
        pending.append(rule)
    group_rank = {group: rank for rank, group in enumerate(CHECKPOINT_GROUP_ORDER)}
    pending.sort(
        key=lambda rule: (
            group_rank.get(str(rule.get("checkpoint", {}).get("group", "")), 99),
            rule["rule_id"],
        )
    )
    return pending
