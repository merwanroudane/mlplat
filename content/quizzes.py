"""Question bank (section 73). Each module id maps to a list of questions:

{"q": text, "type": "mcq"|"tf"|"scenario"|"code"|"chart"|"hyper"|"method",
 "options": [...], "answer": index, "explain": why, "code": optional snippet}

Questions are grouped per curriculum group in content/quiz_bank/*.py and merged here.
"""

from __future__ import annotations

from content.quiz_bank import QUIZ_GROUPS

QUIZZES: dict[str, list[dict]] = {}
for _group in QUIZ_GROUPS:
    QUIZZES.update(_group)
