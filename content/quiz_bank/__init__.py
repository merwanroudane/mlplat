"""Quiz banks per curriculum group."""

from content.quiz_bank import (breiman, causal, evaluation, foundations, imbalanced, interpretability, modern,
                               optimization, pre_ml, production, supervised, trees, tuning, unsupervised)

QUIZ_GROUPS = [m.QUIZZES for m in (breiman, pre_ml, foundations, optimization, supervised, trees, evaluation, tuning,
                                   unsupervised, interpretability, causal, modern, production, imbalanced)]
