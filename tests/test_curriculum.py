"""Curriculum registry integrity: ids, files, prerequisites, groups, paths, roadmap, quizzes."""

import os

from content.algorithm_metadata import ALGORITHMS
from content.exercises import EXERCISES
from content.glossary import GLOSSARY
from content.quizzes import QUIZZES
from core.curriculum import DIFFICULTIES, GROUPS, LEARNING_PATHS, MODULES, ROADMAP, get_module

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IDS = {m.id for m in MODULES}


def test_unique_ids_and_files():
    assert len(IDS) == len(MODULES)
    assert len({m.file for m in MODULES}) == len(MODULES)


def test_page_files_exist_and_match_group_folder():
    for m in MODULES:
        assert os.path.isfile(os.path.join(ROOT, m.file)), m.file
        assert m.file.startswith(f"app_pages/{m.group}/"), (m.id, m.file)


def test_prereqs_exist_and_no_self_or_cycles():
    for m in MODULES:
        for p in m.prereqs:
            assert p in IDS, (m.id, p)
            assert p != m.id
    state = {}

    def visit(mid):
        if state.get(mid) == 1:
            raise AssertionError(f"cycle at {mid}")
        if state.get(mid) == 2:
            return
        state[mid] = 1
        for p in get_module(mid).prereqs:
            visit(p)
        state[mid] = 2

    for m in MODULES:
        visit(m.id)


def test_groups_difficulty_and_kinds():
    for m in MODULES:
        assert m.group in GROUPS
        assert m.difficulty in DIFFICULTIES, (m.id, m.difficulty)
        assert m.kind in {"lesson", "lab", "project", "resource", "start"}
        assert m.title_ar and m.title_en and m.description


def test_paths_and_roadmap_reference_real_modules():
    for p in LEARNING_PATHS.values():
        for mid in p["modules"]:
            assert mid in IDS, mid
    for _, _, mid in ROADMAP:
        assert mid in IDS
    assert len(ROADMAP) == 14
    assert len(LEARNING_PATHS) >= 7


def test_every_tracked_lesson_has_quiz_and_exercises():
    lessons = [m for m in MODULES if m.tracked and m.kind == "lesson"]
    missing_q = [m.id for m in lessons if m.id not in QUIZZES]
    missing_e = [m.id for m in lessons if m.id not in EXERCISES]
    assert not missing_q, missing_q
    assert not missing_e, missing_e


def test_quiz_and_exercise_structure():
    for mid, qs in QUIZZES.items():
        assert mid in IDS, mid
        for q in qs:
            assert 0 <= q["answer"] < len(q["options"]), (mid, q["q"])
            assert q["explain"]
    for mid, ex in EXERCISES.items():
        assert mid in IDS, mid
        assert set(ex) == {"beginner", "intermediate", "research"}, mid
        for e in ex.values():
            assert {"title", "task", "hint", "solution"} <= set(e)


def test_algorithm_metadata_links():
    for a in ALGORITHMS:
        assert a.module in IDS, a.id
        assert a.tasks and a.strengths and a.weaknesses


def test_glossary_links_and_minimum_terms():
    for en, ar, d, mid in GLOSSARY:
        assert mid in IDS, en
    required = {"Estimator", "Parameter", "Hyperparameter", "Feature", "Target", "Loss", "Objective", "Metric", "Regularization",
                "Generalization", "Overfitting", "Margin", "Kernel", "Impurity", "Ensemble", "Calibration", "Nuisance function",
                "Neyman orthogonality", "Cross-fitting", "Treatment", "CATE"}
    assert required <= {g[0] for g in GLOSSARY}


def test_dml_group_is_substantial():
    causal = [m for m in MODULES if m.group == "causal_ml"]
    assert len(causal) >= 14
