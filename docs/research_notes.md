# Research notes

*Machine Learning Interactive Academy · Dr. Marwan Roudane*

Reference date: **September 2026**. All checks below were performed on **2026-09-23**. Nothing time-sensitive in the
app (versions, defaults, API names, DOIs) comes from memory alone: each item was read from the installed package,
from PyPI, from official documentation, or from Crossref/arXiv on that date.

## 1. Package versions (PyPI JSON API, 2026-09-23)

| package | latest on PyPI | release date | requires-python | installed & tested here | note |
|---|---|---|---|---|---|
| scikit-learn | 1.9.1 | 2026-09-10 | >=3.11 | 1.9.1 | matches the brief's baseline |
| numpy | 2.5.3 | 2026-09-06 | >=3.12 | 2.4.6 | latest needs Python 3.12; 2.4.6 on 3.11 |
| scipy | 1.18.1 | 2026-08-21 | >=3.12 | 1.17.1 | as above |
| pandas | 3.0.6 | 2026-09-17 | >=3.11 | 3.0.6 | |
| statsmodels | 0.15.0 | 2026-08-27 | >=3.10 | 0.15.0 | |
| xgboost | 3.4.1 | 2026-08-15 | >=3.12 | 3.2.0 | 3.4.1 needs Python 3.12 |
| lightgbm | 4.7.0 | 2026-07-18 | >=3.10 | 4.7.0 | new `eval_X`/`eval_y` in `fit` |
| catboost | 1.2.10 | 2026-02-18 | — | 1.2.10 | |
| optuna | 5.0.0 | 2026-09-07 | >=3.9 | 5.0.0 | major release |
| shap | 0.52.0 | 2026-05-28 | >=3.12 | 0.51.0 | 0.52 needs Python 3.12 |
| imbalanced-learn | 0.14.2 | 2026-06-07 | >=3.10 | 0.14.2 | |
| DoubleML | 0.11.4 | 2026-08-10 | >=3.10 | 0.11.4 | |
| econml | 0.17.0 | 2026-07-31 | >=3.9 | 0.17.0 | installed alongside sklearn 1.9.1 without downgrade |
| dowhy | 0.14 | 2025-11-08 | >=3.9,<3.14 | not installed | referenced only |
| mlflow | 3.16.1 | 2026-09-16 | >=3.10 | not installed | referenced only (optional) |
| streamlit | 1.64.0 | 2026-09-15 | >=3.10 | 1.64.0 | |
| plotly | 7.1.0 | 2026-09-15 | >=3.8 | 7.1.0 | |

Development/test interpreter: **Python 3.11.0** (Windows). Streamlit Community Cloud lets the author pick the Python
version at deploy time; see `docs/deployment.md`.

## 2. API facts verified on the installed code

| fact | how verified | where used |
|---|---|---|
| `LogisticRegression(penalty=...)` deprecated in 1.8, removed in 1.10; default `l1_ratio=0.0`; `C=np.inf` = no penalty | signature + warning text: *"'penalty' was deprecated in version 1.8 and will be removed in 1.10"*; 1.8 changelog | logistic_regression, hyperparameters, playground |
| `LogisticRegression.n_jobs` deprecated (removed 1.10) | 1.8 changelog | logistic_regression |
| `SVC(probability=...)` deprecated in 1.9, removed in 1.11; replacement `CalibratedClassifierCV(SVC(), ensemble=False)` | warning text on 1.9.1 | svm, calibration |
| `CalibratedClassifierCV(method=...)` ∈ {sigmoid, isotonic, temperature}; temperature added in 1.8 | `_parameter_constraints` | calibration |
| `TunedThresholdClassifierCV(scoring='balanced_accuracy', thresholds=100, ...)` and `FixedThresholdClassifier` | signature | threshold_tuning, imbalanced, projects |
| Tree criteria: classifier {gini, entropy, log_loss}; regressor {squared_error, absolute_error, poisson}; `friedman_mse` deprecated (1.9) | `_parameter_constraints` + 1.9 changelog | decision_trees |
| `GradientBoosting*.criterion` deprecated (1.9) | signature default `'deprecated'` | boosting |
| `n_alphas` deprecated in `lasso_path`/`enet_path` (1.9): pass `alphas` as int or array | 1.9 changelog | regularization |
| `TargetEncoder(shuffle, random_state)` deprecated (1.9) | 1.9 changelog | leakage |
| `StratifiedGroupKFold`, `TimeSeriesSplit(gap=...)`, `HDBSCAN` present | module listing | cross_validation, dbscan |
| `HalvingGridSearchCV/HalvingRandomSearchCV` still require `sklearn.experimental.enable_halving_search_cv` | import test | hpo_foundations |
| `HDBSCAN(copy='warn')` default (future default change) | signature | dbscan |
| `QuadraticDiscriminantAnalysis` gained `solver`, `shrinkage`, `covariance_estimator` (1.8) | signature | lda_qda |
| `early_stopping='auto'` in HGB activates only when n > 10000 | docstring | hist_gradient_boosting |
| XGBoost resolved defaults: eta 0.3, max_depth 6, min_child_weight 1, gamma 0, subsample 1, colsample_* 1, lambda 1, alpha 0, 100 rounds, hist updater | `get_booster().save_config()` | xgboost, tests |
| LightGBM sklearn defaults (num_leaves 31, max_depth −1, min_child_samples 20, …); `eval_set` deprecated in favour of `eval_X`/`eval_y` (4.7) | signature + `LGBMDeprecationWarning` | lightgbm |
| CatBoost resolved defaults: iterations 1000, depth 6, l2_leaf_reg 3, random_strength 1, border_count 254, learning_rate data-dependent, bootstrap MVS | `get_all_params()` | catboost, tests |
| Optuna 5.0: samplers {TPE, Random, CmaEs, GP, QMC, NSGAII/III, Grid, BruteForce, PartialFixed}; pruners {Median, Hyperband, SuccessiveHalving, Percentile, Patient, Threshold, Wilcoxon, Nop}; keyword-only `create_study`; TPE `constant_liar=True`, `n_startup_trials=10` | module listing + signatures + release notes | bayesian_optimization |
| DoubleML 0.11.4 classes: PLR, PLIV, IRM, IIVM, APO(S), DID/DIDCS/DIDMulti, PLPR, LPLR, PQ, QTE, LPQ, CVAR, SSM, PolicyTree, BLP; data classes Data, ClusterData, PanelData, SSMData, RDDData, DIDData | `dir(doubleml)` | doubleml_package, dml_extensions |
| DoubleML signatures (`DoubleMLPLR(obj_dml_data, ml_l, ml_m, ml_g=None, n_folds=5, n_rep=1, score='partialling out')`, IRM `trimming_threshold=0.01`, `sensitivity_analysis(cf_y=0.03, cf_d=0.03, rho=1.0, level=0.95)`, `tune(...)`, `tune_ml_models(...)`, `gate`, `cate`, `DoubleMLPanelData(..., t_col, id_col)`, `DoubleMLDIDMulti(..., control_group='never_treated', anticipation_periods=0)`) | `inspect.signature` | causal pages, tests |
| EconML 0.17.0: `CausalForestDML`, `LinearDML`, `DRLearner`, `SLearner/TLearner/XLearner` | module listing | hte |
| Streamlit 1.64.0 features used: `st.navigation`/`st.Page`, `st.fragment(run_every=...)`, horizontal containers, `st.pills`, `st.segmented_control`, `width="stretch"`, `st.mermaid_chart`, `AppTest` | installed docs (developing-with-streamlit skill) | whole app |

## 3. Correctness cross-checks run as tests

- From-scratch **DML-PLR equals `DoubleMLPLR`** on the same folds to 1e-10 (θ) and 1e-8 relative (SE) — `tests/test_causal.py`.
- IRM (AIPW), PLIV and IIVM recover the known truth of the teaching DGPs.
- Metrics implemented from formulas match scikit-learn/SciPy — `tests/test_metrics.py`.
- Coordinate-descent Lasso equals `sklearn.linear_model.Lasso` to 1e-6 — `tests/test_optimization.py`.
- From-scratch Linear/Logistic GD, kNN, K-Means, PCA agree with library versions — `tests/test_models_scratch.py`.
- Every hyperparameter default in `content/hyperparameters.py` equals the installed signature/resolved config — `tests/test_hyperparameters.py` (119 checks).

## 4. References verification procedure

1. DOI → `curl -I https://doi.org/<doi>` must return 302/200.
2. Crossref `https://api.crossref.org/works/<doi>` → first author, year and title compared with the intended paper.
3. arXiv ids → title read from `https://arxiv.org/abs/<id>`.
4. URLs → HTTP 200 with redirects followed.

Three candidate DOIs were **rejected** because Crossref showed a different paper (they were never used). The Elkan
(2001) IJCAI PDF URL failed and was replaced by the DBLP record.

## 5. Curriculum benchmarks consulted

Stanford CS229 (course site), ISLP (James et al., 2023), ESL (Hastie et al., 2009), Google ML Crash Course, and the
causal-ML literature (Chernozhukov et al., 2018 and follow-ups). See `docs/research_matrix.md`.

## 6. Topics added from research (brief §89)

| topic | why it matters | source | prerequisites | place |
|---|---|---|---|---|
| Conformal prediction | Distribution-free prediction intervals with finite-sample marginal coverage for any model; now standard in modern ML uncertainty teaching | Vovk et al. (2005); Lei et al. (2018); Angelopoulos & Bates (2021) | data splits, regression metrics | Evaluation group (`conformal_prediction`) |
| Meta-learners and DR-learner | Modern CATE estimation beyond causal forests | Künzel et al. (2019); Nie & Wager (2021); Kennedy (2023) | IRM | `hte` |
| Sensitivity analysis for omitted confounders (RV) | Credibility of observational DML results | Chernozhukov et al. (2022, *Long story short*) | DoubleML package | `dml_extensions` |
| Staggered DiD with ATT(g,t) instead of TWFE | TWFE bias with heterogeneous dynamic effects | Callaway & Sant'Anna (2021); Sant'Anna & Zhao (2020); Chang (2020) | IRM, CV | `panel_did` |

## 7. Wording decisions

- The term "Dynamic DML" is **not** used; dynamic effects are presented only through the documented ATT(g, t)/event-study framework.
- Feature importance is never presented as a causal effect; pages carry an explicit caution.

## 8. Version 1.1 additions: Breiman's philosophy and imbalanced classification (2026-09-23)

**References.** 24 new entries, checked on 2026-09-23. The 21 DOIs below were resolved on Crossref, matching author,
year, title, venue, volume, issue and pages. The one arXiv id and the two URLs were checked as listed at the end:

- **Breiman:** Two Cultures `10.1214/ss/1009213726`; CART CRC reprint `10.1201/9781315139470`; ACE
  `10.1080/01621459.1985.10478157`; nonnegative garrote `10.1080/00401706.1995.10484371`; stacked regressions
  `10.1007/BF00117832`; arcing `10.1214/aos/1024691079`.
- **Rashomon / Occam discussion:** Efron 2020 `10.1080/01621459.2020.1762613`; Rudin 2019 `10.1038/s42256-019-0048-x`;
  Semenova et al. 2022 `10.1145/3531146.3533232`.
- **Imbalanced learning:** He & Garcia 2009 `10.1109/TKDE.2008.239`; Borderline-SMOTE `10.1007/11538059_91`; ADASYN
  `10.1109/IJCNN.2008.4633969`; Tomek 1976 `10.1109/TSMC.1976.4309452`; Wilson 1972 `10.1109/TSMC.1972.4309137`;
  Hart 1968 `10.1109/TIT.1968.1054155`; Batista et al. 2004 `10.1145/1007730.1007735`; EasyEnsemble
  `10.1109/TSMCB.2008.2007853`; RUSBoost `10.1109/TSMCA.2009.2029559`; Saerens et al. 2002
  `10.1162/089976602753284446`; van den Goorbergh et al. 2022 `10.1093/jamia/ocac093`.
- **arXiv:** focal loss `arXiv:1708.02002`, with the title matched on arxiv.org.
- **URLs checked for HTTP 200 and title:** JMLR Fisher et al. 2019 (v20/18-760), JMLR Lemaître et al. 2017 (v18/16-365),
  and Berkeley TR 666 (Chen, Liaw & Breiman 2004).

**imbalanced-learn 0.14.2, verified on the installed code.**

- Every sampler and ensemble signature was introspected. The defaults in `content/hyperparameters.py` are tested.
  Notable ones: BalancedRandomForestClassifier uses `sampling_strategy='all'`, `replacement=True`, `bootstrap=False`;
  RUSBoostClassifier uses `learning_rate=1.0`, with `estimator=None` resolving to a depth-1 stump.
- Samplers that expose `sample_indices_` are RandomOver/UnderSampler, NearMiss and the cleaning methods. SMOTE-family
  samplers, ClusterCentroids and the hybrids do not. `utils.imbalance.resample_roles` therefore matches rows exactly,
  which works for every family.
- NaN support: BalancedRandomForest and BalancedBagging (with the default tree) accept NaN. EasyEnsemble and RUSBoost
  (AdaBoost) do not.
- RUSBoost with default stumps raised "BaseClassifier ... worse than random" on the `imbalanced` dataset, at depth 1 and
  at depth 2. Depth-3 trees with `learning_rate=0.1` were stable, and the factory and page say so.

**Boosting libraries.** The imbalance parameters were verified by fitting: XGBoost `scale_pos_weight`, LightGBM
`is_unbalance` and `class_weight`, and CatBoost `auto_class_weights='Balanced'`.

**Lab text is written from computed results.** Pages state conclusions from the numbers they compute, not fixed claims.

- The Two Cultures lab at λ = 1.5 gives Hosmer–Lemeshow p = 0.84 (no rejection) while the RF–logistic AUC gap is +0.20.
- In the extreme-rarity lab, IsolationForest was clearly worse than the supervised models, and the page says so.
- In the cost lab, the theoretical t* was about 9% more costly than the empirical optimum. The page explains that the
  model is only calibrated in the large.
