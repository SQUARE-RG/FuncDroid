# Multilingual UI and model-response fixtures use Unicode escapes to preserve their exact text.
"""Regression tests for preserving authenticated sessions. A compliance executor previously logged out while gathering registration evidence, destroying the session retained by --keep-app and invalidating later checks."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workflow.account_safety import (  # noqa: E402
    account_session_action_violation,
    account_session_violation,
    session_guard_enabled,
)


@pytest.mark.parametrize(
    "text",
    [
        "\u70b9\u51fb\u9000\u51fa\u767b\u5f55",
        "\u9000\u51fa\u5f53\u524d\u8d26\u53f7",
        "\u5207\u6362\u8d26\u53f7",
        "\u70b9\u51fb\u767b\u51fa\u6309\u94ae",
        "tap Log out",
        "Sign Out",
    ],
)
def test_session_destroying_text_is_blocked(text):
    assert account_session_violation(text), f"\u5e94\u62e6\u622a: {text}"


@pytest.mark.parametrize(
    "text",
    [
        # Allow compliance rules R031/R038/R084/R096 to inspect whether an account deletion entry exists.
        "\u67e5\u627e\u8d26\u53f7\u6ce8\u9500\u5165\u53e3",
        "\u8d26\u53f7\u6ce8\u9500",
        "\u6ce8\u9500\u8d26\u53f7",
        "\u5220\u9664\u8d26\u53f7\u5165\u53e3",
        "\u67e5\u770b\u9690\u79c1\u653f\u7b56",
        "\u70b9\u51fb\u767b\u5f55\u6309\u94ae",
    ],
)
def test_deletion_entry_exploration_is_not_blocked(text):
    assert not account_session_violation(text), f"\u4e0d\u5e94\u62e6\u622a: {text}"


def test_non_committal_actions_are_never_blocked():
    for action in ("finished", "wait", "press_back", "scroll", "swipe", "open_app"):
        assert not account_session_action_violation(action, "\u9000\u51fa\u767b\u5f55"), action


def test_click_on_logout_is_blocked_regardless_of_package():
    violation = account_session_action_violation(
        "click",
        "\u70b9\u51fb\u9000\u51fa\u767b\u5f55\u6309\u94ae",
        "\u5f53\u524d\u5df2\u767b\u5f55，\u9700\u5148\u9000\u51fa\u767b\u5f55\u4ee5\u5230\u8fbe\u6ce8\u518c\u8868\u5355",
    )
    assert violation
    assert "\u9000\u51fa\u767b\u5f55" in violation


def test_guard_can_be_disabled(monkeypatch):
    monkeypatch.setenv("ACCOUNT_SESSION_GUARD", "0")
    assert not session_guard_enabled()
    assert not account_session_violation("\u9000\u51fa\u767b\u5f55")
    monkeypatch.setenv("ACCOUNT_SESSION_GUARD", "1")
    assert session_guard_enabled()
    assert account_session_violation("\u9000\u51fa\u767b\u5f55")
