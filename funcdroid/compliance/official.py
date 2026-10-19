"""Map published issue categories to related screening rule IDs."""

# Unicode escapes preserve literal UI labels and legacy evidence matching.
from __future__ import annotations

import re



def normalize_issue(issue: str) -> str:
    return re.sub(r"\s+", "", str(issue)).removeprefix("APP")


OFFICIAL_PROBLEM_RULES = {
    "\u4fe1\u606f\u7a97\u53e3\u65e0\u6cd5\u5173\u95ed": ("P14",),
    "\u4fe1\u606f\u7a97\u53e3\u70b9\u51fb\u4e71\u8df3\u8f6c": ("P15",),
    "\u5f3a\u5236\u7528\u6237\u4f7f\u7528\u5b9a\u5411\u63a8\u9001\u529f\u80fd": ("P08",),
    "\u8fdd\u89c4\u6536\u96c6\u4e2a\u4eba\u4fe1\u606f": ("P01", "P02", "P03", "P04", "P05", "P06", "P07", "P09"),
    "\u8fdd\u89c4\u4f7f\u7528\u4e2a\u4eba\u4fe1\u606f": ("P01", "P02", "P03", "P05", "P08", "P09", "P11"),
    "\u5f3a\u5236、\u9891\u7e41、\u8fc7\u5ea6\u7d22\u53d6\u6743\u9650": ("P06", "P07"),
    "\u9891\u7e41\u7d22\u8981\u975e\u5fc5\u8981\u6743\u9650": ("P06", "P07"),
    "\u5f3a\u5236\u6216\u9891\u7e41\u7d22\u8981\u975e\u5fc5\u8981\u6743\u9650": ("P06", "P07"),
    "\u9891\u7e41\u81ea\u542f\u52a8\u548c\u5173\u8054\u542f\u52a8": ("P19",),
    "\u8fdd\u89c4\u63d0\u4f9b\u81ea\u52a8\u7eed\u8d39\u670d\u52a1": ("P16",),
    "\u6b3a\u9a97\u8bef\u5bfc\u7528\u6237\u4e0b\u8f7dAPP": ("P15",),
    "\u8d85\u8303\u56f4\u6536\u96c6\u4e2a\u4eba\u4fe1\u606f": ("P04",),
    "SDK\u4fe1\u606f\u516c\u793a\u4e0d\u5230\u4f4d": ("P09",),
    "\u672a\u516c\u5f00\u4e2a\u4eba\u4fe1\u606f\u6536\u96c6\u4f7f\u7528\u89c4\u5219": ("P01",),
    "\u672a\u9010\u4e00\u5217\u51fa\u6536\u96c6\u4f7f\u7528\u4e2a\u4eba\u4fe1\u606f\u7684SDK": ("P09",),
    "\u672a\u5b8c\u6574\u51c6\u786e\u5217\u660e\u6536\u96c6\u4f7f\u7528\u4e2a\u4eba\u4fe1\u606f\u60c5\u51b5": ("P02", "P09"),
    "\u672a\u63d0\u4f9b\u6709\u6548\u8d26\u53f7\u6ce8\u9500\u529f\u80fd": ("P10",),
}


# English categories use the same normalization as historical category lookups.
OFFICIAL_PROBLEM_RULES.update({
    normalize_issue(label): rule_ids
    for label, rule_ids in {
        'Information window cannot be closed': ('P14',),
        'Misleading information-window redirection': ('P15',),
        'Forced targeted push': ('P08',),
        'Illegal collection of personal information': ('P01', 'P02', 'P03', 'P04', 'P05', 'P06', 'P07', 'P09'),
        'Illegal use of personal information': ('P01', 'P02', 'P03', 'P05', 'P08', 'P09', 'P11'),
        'Forced, frequent, or excessive permission requests': ('P06', 'P07'),
        'Frequent unnecessary permission requests': ('P06', 'P07'),
        'Frequent self-start or associated start': ('P19',),
        'Improper automatic-renewal service': ('P16',),
        'Misleading application downloads': ('P15',),
        'Excessive personal-information collection': ('P04',),
        'Incomplete SDK disclosures': ('P09',),
        'Undisclosed personal-information collection and use rules': ('P01',),
        'No effective account-deletion functionality': ('P10',),
        'Forced or frequent unnecessary permission requests': ('P06', 'P07'),
        'SDKs collecting and using personal information are not individually listed': ('P09',),
        'Incomplete or inaccurate personal-information collection and use disclosures': ('P02', 'P09'),
    }.items()
})


def official_problem_rule_ids(issue: str) -> list[str]:
    """Return related rule IDs; a category match does not reproduce the published trigger path."""
    return list(OFFICIAL_PROBLEM_RULES.get(normalize_issue(issue), ()))
