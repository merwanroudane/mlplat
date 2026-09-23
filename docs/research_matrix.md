# Research matrix

*Machine Learning Interactive Academy · Dr. Marwan Roudane*

Each time-sensitive item: **source · URL · version · access date · what was verified · module affected.**
Access date for every row: **2026-09-23**.

| source | URL | version | verified | modules affected |
|---|---|---|---|---|
| scikit-learn changelog 1.9 | https://scikit-learn.org/stable/whats_new/v1.9.html | 1.9.0/1.9.1 | SVC.probability deprecation, friedman_mse deprecation, GB criterion deprecation, TargetEncoder shuffle deprecation, n_alphas deprecation, missing values + monotonic constraints in trees | svm, decision_trees, boosting, leakage, regularization, random_forest |
| scikit-learn changelog 1.8 | https://scikit-learn.org/stable/whats_new/v1.8.html | 1.8 | LogisticRegression penalty→l1_ratio, temperature scaling, QDA solver/shrinkage, SGD eta0 > 0, Lasso gap-safe screening | logistic_regression, calibration, lda_qda, gradient_descent, advanced_optimizers |
| installed scikit-learn (inspect) | local `.venv` | 1.9.1 | every sklearn default in content/hyperparameters.py | all algorithm pages, encyclopedia |
| PyPI JSON API | https://pypi.org/pypi/{pkg}/json | see research_notes §1 | latest versions, python requirements | requirements.txt, about |
| XGBoost parameters | https://xgboost.readthedocs.io/en/stable/parameter.html | 3.2.0 installed | resolved defaults via booster config | xgboost |
| LightGBM parameters | https://lightgbm.readthedocs.io/en/stable/Parameters.html | 4.7.0 | defaults, aliases, eval_X/eval_y | lightgbm |
| CatBoost training parameters | https://catboost.ai/docs/en/references/training-parameters/ | 1.2.10 | resolved defaults via get_all_params | catboost |
| Optuna release notes | https://github.com/optuna/optuna/releases | 5.0.0 | new defaults (multivariate TPE, TPE for multi-objective), removed APIs | bayesian_optimization |
| DoubleML API | https://docs.doubleml.org/stable/api/api.html | 0.11.4 | class list, data classes, signatures, sensitivity, DiD | causal_ml group |
| EconML | https://www.pywhy.org/EconML/ | 0.17.0 | CausalForestDML and learners | hte |
| DoWhy | https://www.pywhy.org/dowhy/ | 0.14 | role in the ecosystem | doubleml_package |
| MLflow | https://mlflow.org/docs/latest/ | 3.16.1 (PyPI) | positioned as optional | mlops |
| Streamlit release notes | https://docs.streamlit.io/develop/quick-reference/release-notes | 1.64.0 | features used | whole app |
| Crossref / doi.org / arXiv | https://api.crossref.org, https://doi.org, https://arxiv.org | — | all references | references, every "Research" section |

## Curriculum mapping to reference courses and books

| platform group | CS229 | ISLP | ESL | Google MLCC | causal-ML literature |
|---|---|---|---|---|---|
| Before ML (readiness, splits, leakage, pipelines) | ML advice | Ch. 5 (resampling) | Ch. 7 | Data prep, generalization | — |
| Foundations (risk, bias-variance, loss, theory) | Learning theory | Ch. 2 | Ch. 2, 7 | Loss, generalization | — |
| Optimization | GD, Newton | — | Ch. 3.8, 10 | Gradient descent, learning rate | — |
| Supervised (linear, logistic, NB, LDA/QDA, kNN, SVM) | GLMs, generative, SVM, kernels | Ch. 3, 4, 9 | Ch. 3, 4, 12, 13 | Linear & logistic regression | — |
| Trees & ensembles | — | Ch. 8 | Ch. 9, 10, 15, 16 | — | — |
| Evaluation & tuning | Model selection | Ch. 5, 6 | Ch. 7 | Classification metrics, thresholds | — |
| Unsupervised | k-means, EM, PCA | Ch. 12 | Ch. 14 | Clustering | — |
| Interpretability | — | — | Ch. 10.13 (PDP) | Fairness, interpretability | — |
| Causal ML & DML | — | — | — | — | Chernozhukov et al. 2018; Belloni et al. 2014/2017; Semenova & Chernozhukov 2021; Callaway & Sant'Anna 2021; Chernozhukov et al. 2022 |
| Production | ML advice | — | — | Production ML systems | — |
