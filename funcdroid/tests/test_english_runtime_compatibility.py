"""Regression coverage for English runtime descriptions and preserved legacy matching."""

from model.flow_graph import (
    FlowGraph, FunctionalFlowGraph, FunctionNode, FunctionalityNode,
    control_family_key, operation_family_key, operation_is_failure, operation_is_recovery,
)
from workflow.account_safety import account_session_violation


def test_english_failure_and_recovery_share_operation_family():
    normal = "Click Save button"
    failure = "Failed attempt: Click Save button; error: target unavailable"
    recovery = "Automatic recovery: Click Save button"
    assert operation_is_failure(failure)
    assert not operation_is_failure(normal)
    assert operation_is_recovery(recovery)
    assert not operation_is_recovery(normal)
    assert operation_family_key(normal) == "click:save"
    assert operation_family_key(failure) == operation_family_key(normal)
    assert operation_family_key(recovery) == operation_family_key(normal)
    assert operation_family_key("Back") == "back:back"
    assert control_family_key("Top-left Back arrow") == control_family_key("\u9876\u90e8\u8fd4\u56de\u6309\u94ae") == "back"
    assert control_family_key("Back to home button") == control_family_key("\u8fd4\u56de\u9996\u9875\u6309\u94ae") == "back:home"
    assert control_family_key("Back to top button") != control_family_key("Back button")


def test_paper_names_preserve_existing_model_types():
    assert FunctionalFlowGraph is FlowGraph
    assert FunctionalityNode is FunctionNode


def test_english_session_negation_ends_at_clause_boundary(monkeypatch):
    monkeypatch.setenv("ACCOUNT_SESSION_GUARD", "1")
    assert account_session_violation("Do not click Logout") == ""
    assert account_session_violation("Avoid logout or switch account") == ""
    assert account_session_violation("Do not logout. Then click Logout")
    assert account_session_violation("Click Logout")
