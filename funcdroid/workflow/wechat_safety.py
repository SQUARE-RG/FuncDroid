"""Minimal hard safety guard for unattended WeChat exploration."""

# Unicode escapes preserve literal UI labels and legacy evidence matching.
from __future__ import annotations

import re

from workflow.account_safety import first_affirmative_match


WECHAT_PACKAGE = "com.tencent.mm"

# Keep the runtime guard intentionally small. Exploration breadth and prioritization
# belong to the model prompts; code only prevents losing access to the test account.
_WECHAT_ACCOUNT_DESTRUCTIVE_RE = re.compile(
    r"\u9000\u51fa\u767b\u5f55|\u767b\u51fa(?:\u5fae\u4fe1|\u8d26\u53f7|\u8d26\u6237)?|"
    r"\u6ce8\u9500(?:\u5fae\u4fe1|\u8d26\u53f7|\u8d26\u6237)|\u5220\u9664(?:\u5fae\u4fe1)?(?:\u8d26\u53f7|\u8d26\u6237)|\u6c38\u4e45\u5220\u9664(?:\u8d26\u53f7|\u8d26\u6237)|"
    r"log\s*out|sign\s*out|delete\s+(?:the\s+)?account|close\s+(?:the\s+)?account",
    re.IGNORECASE,
)


def wechat_read_only_violation(package_name: str, *texts: object) -> str:
    """Return only logout/account-deletion intent; all other actions are allowed."""
    if str(package_name or "").strip() != WECHAT_PACKAGE:
        return ""
    combined = " ".join(str(text or "") for text in texts if text is not None)
    # A restated prohibition is not affirmative action intent.
    return first_affirmative_match(_WECHAT_ACCOUNT_DESTRUCTIVE_RE, combined)


def wechat_action_policy_violation(
    package_name: str,
    action_name: str,
    action_text: object = "",
) -> str:
    """Apply the same minimal account guard to any proposed action."""
    violation = wechat_read_only_violation(package_name, action_text)
    return f"Logout or account deletion is forbidden: {violation}" if violation else ""
