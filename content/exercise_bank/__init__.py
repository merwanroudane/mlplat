"""Exercise banks per curriculum group."""

from content.exercise_bank import (breiman, causal, evaluation, foundations, imbalanced, interpretability, modern,
                                   optimization, pre_ml, production, supervised, trees, tuning, unsupervised)

EXERCISE_GROUPS = [m.EXERCISES for m in (breiman, pre_ml, foundations, optimization, supervised, trees, evaluation, tuning,
                                         unsupervised, interpretability, causal, modern, production, imbalanced)]
