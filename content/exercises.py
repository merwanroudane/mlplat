"""Exercises (section 73): Beginner / Intermediate / Research challenge, each with hint and solution.

EXERCISES[module_id] = {"beginner": {...}, "intermediate": {...}, "research": {...}}
with keys title, task, hint, solution. Grouped in content/exercise_bank/*.py.
"""

from __future__ import annotations

from content.exercise_bank import EXERCISE_GROUPS

EXERCISES: dict[str, dict] = {}
for _group in EXERCISE_GROUPS:
    EXERCISES.update(_group)
