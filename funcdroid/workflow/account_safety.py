"""Generic account-safety guard for unattended app exploration."""

from __future__ import annotations

import re


# Keep source text in English/ASCII while still recognizing common Chinese UI
# labels. This policy applies to every package instead of special-casing an app.
_ACCOUNT_DESTRUCTIVE_RE = re.compile(
    r"\u9000\u51fa\u767b\u5f55|\u767b\u51fa(?:\u8d26\u53f7|\u8d26\u6237)?|"
    r"\u6ce8\u9500(?:\u8d26\u53f7|\u8d26\u6237)|\u5220\u9664(?:\u8d26\u53f7|\u8d26\u6237)|"
    r"\u6c38\u4e45\u5220\u9664(?:\u8d26\u53f7|\u8d26\u6237)|"
    r"\blog\s*out\b|\bsign\s*out\b|\bdelete\s+(?:the\s+)?account\b|"
    r"\bclose\s+(?:the\s+)?account\b|\bdeactivate\s+(?:the\s+)?account\b|"
    r"\bterminate\s+(?:the\s+)?account\b",
    re.IGNORECASE,
)


def account_destructive_intent(*texts: object) -> str:
    """Return the matched logout or account-destruction intent, if any."""
    combined = " ".join(str(text or "") for text in texts if text is not None)
    match = _ACCOUNT_DESTRUCTIVE_RE.search(combined)
    return match.group(0) if match else ""


def account_action_policy_violation(action_text: object = "") -> str:
    """Return a package-independent safety violation for a proposed action."""
    violation = account_destructive_intent(action_text)
    return f"Logging out or deleting the account is prohibited: {violation}" if violation else ""
