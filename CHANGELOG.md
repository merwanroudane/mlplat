# Changelog

## 1.1.0 — 2026-09-23

- **New opening group: Leo Breiman's Philosophy (3 modules).** It is placed at the very start of the curriculum, and
  "Start from the beginning" now opens it.
  - `breiman_two_cultures`:
    - The black box.
    - Data-modeling vs algorithmic-modeling culture, and Breiman's critique.
    - The Cox / Efron / Hoadley / Parzen discussion.
    - Common misreadings.
    - Modern synthesis: Shmueli, Mullainathan & Spiess, Efron 2020, DML.
    - Two Cultures Lab: Hosmer–Lemeshow vs out-of-sample AUC.
  - `breiman_three_lessons`:
    - Rashomon subset-selection lab.
    - Bootstrap tree instability.
    - Occam accuracy-vs-size lab with a readable tree.
    - Bellman random-features and noise-dimension labs.
    - The modern Rashomon-set / Rudin debate.
  - `breiman_legacy`:
    - Timeline: CART, ACE, garrote, bagging, stacking, arcing, random forests, balanced RF.
    - Lineage graph.
    - Instability & Bagging Lab.
    - OOB vs CV.
    - Permutation importance tied back to the Two Cultures lab.
- **New group: Imbalanced Classification (10 modules).** The former `imbalanced` page became the track overview.
  New modules:
  - `imb_nature`: IR, absolute rarity, overlap, small disjuncts, the Bayes rule; geometry lab.
  - `imb_metrics`: prevalence lab (ROC vs PR), MCC, G-mean, F-beta, bootstrap CIs.
  - `imb_cost_sensitive`: Elkan threshold, class_weight as a prior shift, sample_weight, boosting-library equivalents,
    focal loss; cost lab.
  - `imb_oversampling`: ROS, shrinkage, SMOTE, Borderline-1/2, SVMSMOTE, ADASYN, SMOTENC/SMOTEN.
  - `imb_undersampling`: NearMiss, ClusterCentroids, Tomek, ENN/RENN/AllKNN, CNN, OSS, NCR, IHT, SMOTEENN/SMOTETomek.
  - `imb_ensembles`: BalancedRandomForest, BalancedBagging, EasyEnsemble, RUSBoost vs weighted RF/HGB.
  - `imb_probabilities`: prior-shift correction, calibration after resampling, EM prevalence estimation.
  - `imb_multiclass_extreme`: per-class sampling strategies, macro metrics, anomaly-detection framing.
  - `imb_workflow`: 10-step guide and a benchmark lab with a downloadable report.
- New `utils/breiman.py`, `utils/imbalance.py` and `components/sampler_viz.py`, plus
  `tests/test_breiman_imbalance.py` (42 tests).
- Content additions:
  - 4 algorithm profiles (imbalanced-learn ensembles).
  - 25 verified hyperparameter entries: 23 for imbalanced-learn, plus RF/HGB `class_weight`.
  - 24 verified references.
  - 24 glossary terms.
  - 2 new learning paths (Imbalanced, Breiman).
- The Algorithm Selector suggests balanced ensembles only when the data are imbalanced.

## 1.0.0 — 2026-09-23

First release of the Machine Learning Interactive Academy (Dr. Marwan Roudane).

- 101 pages in 16 groups driven by a single curriculum registry; 7 learning paths; 14-station roadmap.
- 27-item algorithm template with profiles, verified hyperparameter tables and a teacher's template map.
- Labs:
  - feature space, readiness checker, split, leakage, pipeline
  - loss, gradient descent, optimizer race, bias–variance animation, regularization paths
  - kNN, logistic boundary, SVM margin animation, tree split + pruning
  - random forest animation, boosting stages, boosting comparison
  - CV animator, metric explorer, threshold, calibration, imbalance, conformal coverage
  - search strategies, Optuna HPO
  - K-Means animation, DBSCAN animation, EM animation, PCA, embeddings, isolation animation
  - feature importance, PDP/ICE, SHAP
  - drift simulator, fairness audit
  - Q-learning gridworld, MLP, active learning, walk-forward, text classifier
  - parameter playground, model comparison
- Causal ML & DML: 14 modules, with DML estimators implemented from the formulas and validated against DoubleML 0.11.4.
  - Cross-fitting animation and an orthogonality sensitivity lab.
  - PLR / PLIV / IRM / IIVM labs.
  - CATE / GATE / BLP and a causal forest.
  - Staggered DiD (DoubleMLDIDMulti).
  - Sensitivity analysis and cluster-robust inference.
  - DML Monte Carlo lab.
- 10 end-to-end projects and 2 capstones with downloadable reports.
- Research-driven additions: conformal prediction, meta-learners/DR-learner, OVB sensitivity, ATT(g,t).
- API updates reflected:
  - LogisticRegression `l1_ratio` (penalty deprecated 1.8).
  - `SVC.probability` deprecated (1.9).
  - Temperature scaling (1.8).
  - `friedman_mse` deprecated (1.9).
  - LightGBM `eval_X` / `eval_y` (4.7).
  - Optuna 5.0 defaults.
- Tests, generated docs, research notes and deployment guide.
