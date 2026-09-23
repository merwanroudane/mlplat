# Deployment

*Machine Learning Interactive Academy · Dr. Marwan Roudane*

## Streamlit Community Cloud

1. Push the repository to GitHub (a new repository, e.g. `MLplat`; DSplat is not modified).
2. On https://share.streamlit.io create an app: main file `streamlit_app.py`.
3. **Advanced settings → Python version: 3.11 or 3.12.** The pinned requirements were tested on 3.11; on 3.12 pip will
   also resolve (newer numpy/xgboost/shap wheels become available — re-run the tests locally on 3.12 if you choose it).
4. Dependencies: `requirements.txt` (core + optional libraries). For the smallest footprint use `requirements-core.txt`
   instead (rename it to `requirements.txt`): every page still renders; optional labs show theory + install hints.
5. No secrets are needed. `.streamlit/secrets.toml` is git-ignored.

## Resource considerations

- All data are generated in memory (no files). Heavy computations are cached and bounded by `config.py`.
- The largest optional wheels are CatBoost and XGBoost; if the build is slow or memory-limited, drop them first — the
  pages degrade gracefully (`tests/test_optional_deps.py`).
- Simulations (DML Monte Carlo, Naive vs DML) run only on button press with capped replications.

## Local run

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Tests

```bash
pip install pytest
python -m pytest                   # all suites (the click-every-button suite takes several minutes)
python -m pytest tests/test_hyperparameters.py   # re-verify defaults after any upgrade
```

## Upgrading libraries

1. Change the pin in `requirements.txt`.
2. Run `python -m pytest tests/test_hyperparameters.py` — any changed default fails the test; update
   `content/hyperparameters.py` (value, `VERIFIED_VERSIONS`, `CHECKED`) and the affected page text.
3. Run the full test suite and `python scripts/build_docs.py`.
