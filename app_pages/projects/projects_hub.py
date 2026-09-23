import time

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import (average_precision_score, brier_score_loss, mean_absolute_error, r2_score, roc_auc_score,
                             root_mean_squared_error, silhouette_score)
from sklearn.model_selection import (StratifiedKFold, TimeSeriesSplit, TunedThresholdClassifierCV, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from components.callouts import mistakes, why
from config import RANDOM_SEED
from core.page import page_footer, page_header
from core.registry import available, optional
from utils import causal as C
from utils.datasets import load_dataset, xy
from utils.preprocessing import ordinal_categoricals, tabular_preprocessor
from utils.report import build_report

page_header("projects_hub")

PROJECTS = {
    "p1": ("1. Regression project", "توقع هدف عددي بخط أساس ونماذج خطية وشجرية ومقاييس وبواقي.", "regression"),
    "p2": ("2. Binary classification", "Pipeline كامل، مقارنة، معايرة، واختيار عتبة.", "mixed"),
    "p3": ("3. Imbalanced classification", "كشف احتيال: PR-AUC، عتبة مضبوطة بالتكلفة داخل CV.", "imbalanced"),
    "p4": ("4. Tree boosting benchmark", "HGB مقابل XGBoost/LightGBM/CatBoost بنفس الطيات.", "mixed"),
    "p5": ("5. Customer segmentation", "تجميع عملاء بـK-Means مع اختيار k وملفات العناقيد.", "mixed"),
    "p6": ("6. Anomaly detection", "Isolation Forest مع تقييم على شذوذ معروف.", "anomalies"),
    "p7": ("7. Text classification", "TF-IDF + Logistic مع تحليل الأخطاء.", "text"),
    "p8": ("8. Time-series ML", "خصائص Lag وWalk-forward ومقارنة Seasonal naive.", "timeseries"),
    "p9": ("9. DML causal-effect project", "ATE لمعالجة ثنائية بـDML-IRM مع تحليل حساسية.", "causal"),
    "p10": ("10. Panel / DiD DML project", "ATT(g, t) وEvent study بـDoubleMLDIDMulti.", "panel"),
}
pid = st.selectbox("اختر المشروع", list(PROJECTS), format_func=lambda k: PROJECTS[k][0], key="proj_pick")
title, desc, ds = PROJECTS[pid]
st.markdown(f"### {title}\n{desc}")


# ---------------------------------------------------------------- runners
@st.cache_data(show_spinner="يشغّل المشروع كاملًا…", max_entries=12)
def run_project(pid: str) -> dict:
    t0 = time.perf_counter()
    out: dict = {"steps": [], "tables": {}, "figs": {}}
    if pid == "p1":
        X, y = xy("regression")
        Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.25, random_state=RANDOM_SEED)
        models = {"Dummy (mean)": DummyRegressor(), "RidgeCV": make_pipeline(StandardScaler(), RidgeCV(np.logspace(-3, 3, 20))),
                  "RandomForest": RandomForestRegressor(300, min_samples_leaf=3, random_state=0, n_jobs=1),
                  "HistGradientBoosting": HistGradientBoostingRegressor(random_state=0)}
        rows = []
        for n, m in models.items():
            cv = cross_val_score(m, Xa, ya, cv=5, scoring="neg_mean_absolute_error").mean()
            m.fit(Xa, ya)
            p = m.predict(Xb)
            rows.append({"model": n, "CV MAE (train)": -cv, "test MAE": mean_absolute_error(yb, p), "test RMSE": root_mean_squared_error(yb, p),
                         "test R²": r2_score(yb, p)})
        out["tables"]["Results"] = pd.DataFrame(rows)
        best = models["HistGradientBoosting"]
        pi = permutation_importance(best, Xb, yb, n_repeats=5, random_state=0)
        out["tables"]["Permutation importance (test)"] = pd.DataFrame({"feature": X.columns, "importance": pi.importances_mean}).sort_values(
            "importance", ascending=False)
        out["steps"] = ["Question: predict y from x1..x8 (MAE in y units).", "Split: 75/25 random (i.i.d. data); model selection by 5-fold CV on train.",
                        "Baseline: mean predictor.", "Candidates: Ridge (linear) vs RF and HGB (nonlinear).",
                        "Interpretation: permutation importance recovers x1–x4 and the nonlinear x4."]
    elif pid == "p2":
        X, y = xy("mixed")
        Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.25, random_state=RANDOM_SEED, stratify=y)
        lr = make_pipeline(tabular_preprocessor(), LogisticRegression(max_iter=3000))
        hgb = HistGradientBoostingClassifier(random_state=0)
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        rows = []
        for n, m, Xt, Xv in (("Logistic (pipeline)", lr, Xa, Xb), ("HGB (native cats + NaN)", hgb, ordinal_categoricals(Xa), ordinal_categoricals(Xb))):
            s = cross_val_score(m, Xt, ya, cv=cv, scoring="roc_auc").mean()
            m.fit(Xt, ya)
            p = m.predict_proba(Xv)[:, 1]
            rows.append({"model": n, "CV AUC": s, "test AUC": roc_auc_score(yb, p), "test Brier": brier_score_loss(yb, p)})
        out["tables"]["Results"] = pd.DataFrame(rows)
        tuned = TunedThresholdClassifierCV(make_pipeline(tabular_preprocessor(), LogisticRegression(max_iter=3000)), scoring="balanced_accuracy",
                                           cv=cv).fit(Xa, ya)
        out["tables"]["Threshold"] = pd.DataFrame([{"tuned threshold (balanced accuracy, inner CV)": tuned.best_threshold_,
                                                    "test balanced accuracy": float(np.mean([(tuned.predict(Xb)[yb == c] == c).mean() for c in (0, 1)]))}])
        out["steps"] = ["Question: approve loan applications (probability + decision).", "Readiness: missing income/credit score, high-cardinality city.",
                        "Pipeline: impute + scale + one-hot(min_frequency) for Logistic; native handling for HGB.",
                        "Calibration check with Brier; threshold tuned by TunedThresholdClassifierCV inside CV (test untouched)."]
    elif pid == "p3":
        X, y = xy("imbalanced")
        Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y)
        from sklearn.metrics import confusion_matrix, make_scorer

        def neg_cost(yt, yp):
            tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
            return -(1 * fp + 20 * fn) / len(yt)

        base = HistGradientBoostingClassifier(random_state=0)
        tuned = TunedThresholdClassifierCV(base, scoring=make_scorer(neg_cost), cv=StratifiedKFold(5, shuffle=True, random_state=0)).fit(Xa, ya)
        fixed = HistGradientBoostingClassifier(random_state=0).fit(Xa, ya)
        rows = []
        for n, m in (("threshold 0.5", fixed), (f"cost-tuned threshold {tuned.best_threshold_:.3f}", tuned)):
            pr = m.predict(Xb)
            tn, fp, fn, tp = confusion_matrix(yb, pr).ravel()
            rows.append({"decision rule": n, "TP": tp, "FP": fp, "FN": fn, "cost / case (FN = 20, FP = 1)": (fp + 20 * fn) / len(yb)})
        p = fixed.predict_proba(Xb)[:, 1]
        out["tables"]["Ranking quality"] = pd.DataFrame([{"ROC-AUC": roc_auc_score(yb, p), "PR-AUC (AP)": average_precision_score(yb, p),
                                                          "prevalence": yb.mean()}])
        out["tables"]["Decisions"] = pd.DataFrame(rows)
        out["steps"] = ["Question: flag fraud; a missed fraud costs 20× a false alarm.", "Metric: PR-AUC (rare positives) + expected cost.",
                        "Model: HGB; threshold tuned for cost inside CV.", "Result: lower expected cost at the tuned threshold."]
    elif pid == "p4":
        X, y = xy("mixed")
        Xc = ordinal_categoricals(X)
        cv = StratifiedKFold(3, shuffle=True, random_state=0)
        rows = []
        models = {"HistGradientBoosting": HistGradientBoostingClassifier(random_state=0)}
        if available("xgboost"):
            import xgboost as xgb
            models["XGBoost"] = xgb.XGBClassifier(n_estimators=200, learning_rate=0.1, enable_categorical=True, tree_method="hist", n_jobs=1)
        if available("lightgbm"):
            import lightgbm as lgb
            models["LightGBM"] = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1, verbose=-1, n_jobs=1)
        for n, m in models.items():
            t = time.perf_counter()
            s = cross_val_score(m, Xc, y, cv=cv, scoring="roc_auc")
            rows.append({"library": n, "ROC-AUC": s.mean(), "SD": s.std(), "seconds": time.perf_counter() - t})
        out["tables"]["Benchmark (shared folds)"] = pd.DataFrame(rows)
        out["steps"] = ["Same data, same folds, comparable budgets.", "Report mean ± SD and time; no absolute winner.",
                        "Full lab (with CatBoost and memory): Boosting Comparison page."]
    elif pid == "p5":
        df = load_dataset("mixed")
        num = df[["age", "income", "loan_amount", "credit_score", "n_accounts"]].dropna()
        Z = StandardScaler().fit_transform(np.log1p(num.clip(lower=0)))
        ks = range(2, 8)
        sil = [silhouette_score(Z, KMeans(k, n_init="auto", random_state=0).fit_predict(Z), sample_size=1500, random_state=0) for k in ks]
        k_best = list(ks)[int(np.argmax(sil))]
        labels = KMeans(k_best, n_init="auto", random_state=0).fit_predict(Z)
        prof = num.assign(segment=labels).groupby("segment").agg(["mean"]).round(1)
        prof.columns = [c[0] for c in prof.columns]
        prof["size"] = pd.Series(labels).value_counts().sort_index().to_numpy()
        out["tables"]["Silhouette by k"] = pd.DataFrame({"k": list(ks), "silhouette": sil})
        out["tables"][f"Segment profiles (k = {k_best})"] = prof.reset_index()
        out["steps"] = ["Features: log-transformed and standardised numeric attributes.", "k chosen by silhouette (plus business sense).",
                        "Profiles: mean attributes per segment — name them with domain experts."]
    elif pid == "p6":
        df = load_dataset("anomalies")
        X = StandardScaler().fit_transform(df[["x1", "x2"]])
        iso = IsolationForest(n_estimators=300, contamination=0.05, random_state=0).fit(X)
        score = -iso.score_samples(X)
        top = np.argsort(-score)[:25]
        out["tables"]["Evaluation (ground truth known only in simulation)"] = pd.DataFrame([{
            "average precision": average_precision_score(df["is_anomaly"], score), "precision@25": df["is_anomaly"].to_numpy()[top].mean(),
            "flagged at contamination 0.05": int((iso.predict(X) == -1).sum())}])
        out["steps"] = ["Unsupervised scoring with Isolation Forest.", "Review the top-k for investigation (Precision@k).",
                        "Distinguish data errors from genuine anomalies before action."]
    elif pid == "p7":
        df = load_dataset("text")
        Xa, Xb, ya, yb = train_test_split(df["text"], df["label"], test_size=0.3, random_state=RANDOM_SEED, stratify=df["label"])
        pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LogisticRegression(max_iter=2000)).fit(Xa, ya)
        pr = pipe.predict(Xb)
        err = pd.DataFrame({"text": Xb, "true": yb, "pred": pr})
        out["tables"]["Results"] = pd.DataFrame([{"test accuracy": (pr == yb).mean(), "errors": int((pr != yb).sum())}])
        out["tables"]["Error analysis (misclassified)"] = err[err.true != err.pred].head(12)
        out["steps"] = ["TF-IDF (1,2)-grams inside a Pipeline.", "Logistic regression baseline.",
                        "Error analysis: negations and mixed-sentiment sentences dominate the mistakes."]
    elif pid == "p8":
        ts = load_dataset("timeseries")
        s = ts["sales"]
        F = pd.DataFrame({f"lag_{k}": s.shift(k) for k in (1, 7, 14)})
        F["roll7"] = s.shift(1).rolling(7).mean()
        F["dow"] = ts["date"].dt.dayofweek
        F["t"] = np.arange(len(ts))
        F["promo"] = ts["promo"]
        d = pd.concat([F, s.rename("y")], axis=1).dropna()
        tss = TimeSeriesSplit(5, test_size=60)
        hgb = -cross_val_score(HistGradientBoostingRegressor(random_state=0), d.drop(columns="y"), d["y"], cv=tss,
                               scoring="neg_mean_absolute_error").mean()
        snaive = np.mean([np.mean(np.abs(d["y"].to_numpy()[te] - d["lag_7"].to_numpy()[te])) for _, te in tss.split(d)])
        out["tables"]["Walk-forward MAE"] = pd.DataFrame([{"HistGradientBoosting": hgb, "Seasonal naive": snaive,
                                                           "improvement": 1 - hgb / snaive}])
        out["steps"] = ["Features: lags ≥ 1, shifted rolling mean, calendar, known promotions.",
                        "Validation: 5 walk-forward windows of 60 days.", "Benchmark: seasonal naive (last week)."]
    elif pid == "p9":
        df = load_dataset("causal")
        X, y, d = df.filter(like="x").to_numpy(), df["y"].to_numpy(), df["d"].to_numpy()
        g = RandomForestRegressor(200, min_samples_leaf=5, random_state=0, n_jobs=1)
        m = RandomForestClassifier(200, min_samples_leaf=5, random_state=0, n_jobs=1)
        res = C.dml_irm(X, y, d, g, m, n_folds=5)
        rows = [{"estimator": "naive difference", "estimate": y[d == 1].mean() - y[d == 0].mean(), "95% CI": "—"},
                {"estimator": "OLS with X", "estimate": C.naive_ols(X, y, d).theta, "95% CI": "linear adjustment"},
                {"estimator": "DML-IRM (ATE)", "estimate": res.theta, "95% CI": f"[{res.ci[0]:.3f}, {res.ci[1]:.3f}]"}]
        out["tables"]["Estimates"] = pd.DataFrame(rows)
        if optional("doubleml") is not None:
            import doubleml as dml
            data = dml.DoubleMLData(df.drop(columns=["tau", "true_ps"]), "y", "d", [f"x{i}" for i in range(1, 6)])
            irm = dml.DoubleMLIRM(data, g, m, n_folds=5).fit()
            irm.sensitivity_analysis(cf_y=0.03, cf_d=0.03)
            out["tables"]["Sensitivity (cf_y = cf_d = 0.03)"] = pd.DataFrame([{"theta lower": float(np.ravel(irm.sensitivity_params["theta"]["lower"])[0]),
                                                                              "theta upper": float(np.ravel(irm.sensitivity_params["theta"]["upper"])[0]),
                                                                              "RV": float(np.ravel(irm.sensitivity_params["rv"])[0])}])
        out["steps"] = ["Causal question: effect of binary treatment d on y.", "Assumptions: unconfoundedness given x1..x5, overlap (checked).",
                        "Nuisances: random forests; 5-fold cross-fitting; AIPW score.", f"True ATE in the DGP ≈ {df['tau'].mean():.3f} (teaching only).",
                        "Sensitivity analysis reported."]
    else:
        if optional("doubleml") is None:
            out["steps"] = ["DoubleML not installed: see Panel DiD page for the reference code."]
        else:
            import doubleml as dml
            from doubleml.did import DoubleMLDIDMulti
            from sklearn.linear_model import LinearRegression
            p = load_dataset("panel").copy()
            p["g"] = p["g"].astype(float).replace(0, np.inf)
            data = dml.DoubleMLPanelData(p.drop(columns=["d", "true_att"]), y_col="y", d_cols="g", t_col="t", id_col="id", x_cols=["x1", "x2"])
            did = DoubleMLDIDMulti(data, ml_g=LinearRegression(), ml_m=LogisticRegression(), n_folds=5).fit()
            es = did.aggregate("eventstudy").aggregated_frameworks.summary.reset_index().rename(columns={"index": "event time"})
            out["tables"]["Event study"] = es.round(3)
            out["steps"] = ["Staggered adoption (g = 4, 6, never).", "ATT(g, t) with never-treated controls and DR-DiD scores.",
                            "Aggregated to event time; pre-periods test parallel trends.", "Truth: 1 + 0.5·e after treatment."]
    out["seconds"] = time.perf_counter() - t0
    return out


if st.button("شغّل المشروع", key="proj_run", type="primary", icon=":material/play_arrow:"):
    st.session_state["proj_ran"] = st.session_state.get("proj_ran", set()) | {pid}
if pid in st.session_state.get("proj_ran", set()):
    res = run_project(pid)
    st.markdown("#### الخطوات")
    st.markdown("\n".join(f"{i + 1}. {s}" for i, s in enumerate(res["steps"])))
    for name, tab in res["tables"].items():
        st.markdown(f"#### {name}")
        st.dataframe(tab, hide_index=True, width="stretch")
    st.caption(f"زمن التشغيل: {res['seconds']:.1f} s (يُخزَّن مؤقتًا).")
    report = build_report(title, [("Description", desc), ("Steps", res["steps"])] + list(res["tables"].items()))
    st.download_button("تنزيل تقرير المشروع (Markdown)", report, f"{pid}_report.md", "text/markdown", icon=":material/download:")
else:
    st.info("اضغط «شغّل المشروع» لتنفيذ كل الخطوات على بيانات المنصة وتوليد تقرير قابل للتنزيل.", icon=":material/play_circle:")
why("كل مشروع يتبع الهيكل نفسه: سؤال ← جاهزية ← تقسيم ← خط أساس ← نموذج ← تقييم ← تفسير ← تقرير.",
    "التكرار على مسائل مختلفة يرسّخ الإجراء أكثر من أي خوارزمية بعينها.")
mistakes(["تخطي خط الأساس.", "تقييم على بيانات استُخدمت في القرارات.", "تقرير بلا إصدارات وبذور."])
page_footer("projects_hub", takeaways=["عشرة مشاريع تغطي كل أنواع المسائل في المنصة.", "الإجراء الموحد أهم من الخوارزمية.",
                                       "كل تقرير يحمل الإصدارات للمراجعة."])
