# Architecture

*Machine Learning Interactive Academy · Dr. Marwan Roudane*

```text
streamlit_app.py          entry point: page config, CSS, Plotly template, state, sidebar (level, reduced motion), st.navigation
config.py                 identity (author, names, DSplat URL), compute limits, levels — the only place names live
.streamlit/config.toml    DSplat design system (light theme only), fonts
assets/css/theme.css      RTL/LTR rules and component styles (only what native theming cannot do)

core/
  curriculum.py           Module registry (single source of truth): 113 pages, 18 groups, roadmap, learning paths
  navigation.py           builds st.navigation from the registry
  page.py                 page_header / page_footer template (objectives, prerequisites, quiz, exercises, next, identity)
  state.py                session state: progress, quiz scores, bookmarks, experiment log, JSON export/import
  theme.py                CSS injection + Plotly template
  registry.py             optional-dependency registry: optional(), missing_notice()

components/               reusable UI
  animation.py            stepper(): Play/Pause/Prev/Next/Reset/Speed via st.fragment(run_every) + reduced motion
  algorithm_profile.py    27-item algorithm template, hyperparameter tables from metadata
  parameter_lab.py        Parameter Playground (controls → boundary → train/val → runtime → complexity)
  decision_boundary.py, model_compare.py, metric_explorer.py, dataset_viewer.py
  callouts.py, cards.py, formulas.py, code_lab.py (safe: runs predefined functions), quiz.py, diagrams.py
  sampler_viz.py          Sampler Visualizer: what an imbalanced-learn sampler adds, duplicates and removes

content/                  data, not UI
  hyperparameters.py      central defaults (checked by tests), verified versions
  algorithm_metadata.py   explorer/selector/profile data
  references.py           verified DOIs/arXiv/URLs + official docs
  glossary.py, quiz_bank/*, exercise_bank/*, quizzes.py, exercises.py

utils/                    pure Python/NumPy/scikit-learn logic (unit-tested, no Streamlit UI)
  datasets.py             15 synthetic datasets with ground truth
  metrics.py              metrics and losses from formulas
  optimization.py         GD/SGD/momentum/Adam/Newton/coordinate descent
  models.py               from-scratch models + playground factory
  causal.py               DML from formulas: PLR, PLIV, IRM, IIVM, DR-learner, GATE, Monte Carlo
  breiman.py              two-cultures data, Hosmer–Lemeshow, Rashomon subsets, tree instability, random features
  imbalance.py            imbalance metrics, costs, prior shift + EM, sampler/ensemble factories, repeated stratified CV
  tuning.py, inspection.py, preprocessing.py (readiness checker), plotting.py, validation.py, report.py

app_pages/<group>/*.py    one script per page (direct scripts, no wrapper functions)
tests/                    registry, content, datasets, metrics, optimizers, models, causal, hyperparameters,
                          render smoke test, click-every-button test, projects, optional-deps-missing
scripts/build_docs.py     regenerates curriculum/datasets/algorithm_matrix/hyperparameters/references docs
```

## Design decisions

- **Single source of truth.** Navigation, search, progress, docs and tests read `core/curriculum.py`; defaults live only in
  `content/hyperparameters.py`; author names only in `config.py` (a test enforces this).
- **Safety.** No user code is executed: code labs display code and run a predefined function. Uploads are limited to the
  progress JSON (parsed with `json`, never pickle). Models are never loaded from files.
- **Performance for Community Cloud.** Expensive work is cached (`st.cache_data` for data/results, `st.cache_resource` for
  fitted models) with `max_entries`; heavy labs run behind buttons; capstone stages compute on demand; limits in `config.py`
  (`MAX_SAMPLES_LAB`, `MAX_TREES_LAB`, `MAX_OPTUNA_TRIALS`, `MAX_MC_REPS`); `n_jobs=1` everywhere.
- **Optional libraries.** `core/registry.optional()` returns `None` if a package is missing; pages then show theory, reference
  code and an install hint. `tests/test_optional_deps.py` simulates every optional package missing.
- **RTL/LTR.** Arabic prose is RTL; code, math, charts, tables and sliders are forced LTR in `theme.css`.
- **Accessibility.** Classes use colour + marker shape; reduced-motion toggle disables autoplay; light theme with AA contrast.

## Adding content

- **Module:** add a `Module(...)` to `core/curriculum.py`, create `app_pages/<group>/<id>.py` starting with
  `page_header("<id>")` and ending with `page_footer("<id>", ...)`, add quizzes/exercises in the group banks, run
  `python scripts/build_docs.py` and `python -m pytest`.
- **Lab:** put the computation in `utils/` (tested), cache it, and render it in the page; add a "سجّل" button calling
  `core.state.log_experiment` if it produces a result worth tracking.
- **Algorithm:** add an `Algorithm(...)` to `content/algorithm_metadata.py` and its hyperparameters to
  `content/hyperparameters.py` (the test will tell you if a default is wrong); call `algorithm_profile(id)` and
  `hyperparameter_table(EstimatorName)` on the page.
