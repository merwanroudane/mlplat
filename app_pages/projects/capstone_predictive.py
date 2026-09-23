import numpy as np
import pandas as pd
import streamlit as st
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, TunedThresholdClassifierCV, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from scipy.stats import loguniform, randint

from components.callouts import why
from config import RANDOM_SEED
from core.page import page_footer, page_header
from core.state import log_experiment
from utils.datasets import load_dataset, xy
from utils.plotting import bars, plot
from utils.preprocessing import ordinal_categoricals, readiness_frame, readiness_report, tabular_preprocessor
from utils.report import build_report
from utils.validation import package_versions

page_header("capstone_predictive")
st.markdown("مشروع نهائي موجَّه على بيانات القروض المختلطة. أكمل المراحل الـ13 بالترتيب؛ كل مرحلة تتطلب قرارًا مكتوبًا منك، والمنصة "
            "تنفذ الحسابات. في النهاية يُولَّد تقرير كامل قابل للتنزيل.")
STAGES = ["1 Problem", "2 Data readiness", "3 Split", "4 Baseline", "5 Pipeline", "6 Candidate models", "7 HPO", "8 Evaluation",
          "9 Threshold / calibration", "10 Error analysis", "11 Interpretation", "12 Reproducibility", "13 Final report"]
done = st.session_state.setdefault("capA_done", set())
st.progress(len(done) / len(STAGES), text=f"{len(done)} / {len(STAGES)} مراحل مكتملة")
X, y = xy("mixed")
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=RANDOM_SEED, stratify=y)
cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_SEED)
notes = st.session_state.setdefault("capA_notes", {})


HEAVY = {3, 6, 7, 8, 9, 10}  # stages with model training: computed on demand


def stage(i: int, body) -> None:
    with st.expander(STAGES[i], expanded=False, icon=":material/check_circle:" if STAGES[i] in done else ":material/radio_button_unchecked:"):
        if i in HEAVY and not st.session_state.get(f"capA_run_{i}"):
            if st.button("شغّل حسابات هذه المرحلة", key=f"capA_btn_{i}", icon=":material/play_arrow:"):
                st.session_state[f"capA_run_{i}"] = True
                st.rerun()
        else:
            body()
        notes[STAGES[i]] = st.text_area("قرارك وتبريره (يدخل التقرير)", value=notes.get(STAGES[i], ""), key=f"capA_note_{i}", height=80)
        if st.toggle("أكملت هذه المرحلة", value=STAGES[i] in done, key=f"capA_t_{i}"):
            done.add(STAGES[i])
        else:
            done.discard(STAGES[i])


@st.cache_data(show_spinner="يحسب…")
def _baselines():
    rows = []
    for n, m, Xa in (("Dummy (prior)", DummyClassifier(strategy="prior"), Xtr),
                     ("Logistic pipeline", make_pipeline(tabular_preprocessor(), LogisticRegression(max_iter=3000)), Xtr),
                     ("HGB (native)", HistGradientBoostingClassifier(random_state=0), ordinal_categoricals(Xtr))):
        s = cross_val_score(m, Xa, ytr, cv=cv, scoring="roc_auc")
        rows.append({"model": n, "CV ROC-AUC": s.mean(), "± SD": s.std()})
    return pd.DataFrame(rows)


@st.cache_resource(show_spinner="RandomizedSearchCV (20 محاولة × 5 طيات)…")
def _hpo():
    rs = RandomizedSearchCV(HistGradientBoostingClassifier(random_state=0, early_stopping=False),
                            {"learning_rate": loguniform(0.01, 0.3), "max_leaf_nodes": randint(8, 64), "min_samples_leaf": randint(5, 80),
                             "l2_regularization": loguniform(1e-3, 10)}, n_iter=20, cv=cv, scoring="roc_auc", random_state=RANDOM_SEED)
    rs.fit(ordinal_categoricals(Xtr), ytr)
    return rs


def s1():
    st.markdown("**المطلوب:** صياغة T وE وP، ووحدة التحليل، وأفق التنبؤ، وكلفة الأخطاء. مثال: «رفض قرض سيُسدَّد (FP) يكلف 1، "
                "والموافقة على قرض سيتعثر (FN) تكلف 5».")


def s2():
    st.dataframe(readiness_frame(readiness_report(load_dataset("mixed"), "approved")), hide_index=True, width="stretch")


def s3():
    st.markdown(f"تدريب {len(Xtr)} / اختبار {len(Xte)} (طبقي، بذرة {RANDOM_SEED}). **الاختبار مقفل حتى المرحلة 8.** "
                "كل القرارات بـ5-fold CV على التدريب.")


def s4():
    st.dataframe(_baselines().round(4), hide_index=True, width="stretch")


def s5():
    st.code("make_pipeline(ColumnTransformer([num: impute+scale, cat: impute+one-hot(min_frequency=0.01)]), LogisticRegression())\n"
            "HistGradientBoostingClassifier(categorical_features='from_dtype')  # native NaN & categories", language="python")


def s6():
    st.markdown("المرشحان: Logistic (خطي قابل للتفسير) وHGB (غير خطي). النتائج في جدول المرحلة 4.")


def s7():
    rs = _hpo()
    st.markdown(f"أفضل CV ROC-AUC = **{rs.best_score_:.4f}** (متفائل: اختير من 20 محاولة). المعاملات: `{rs.best_params_}`")


def s8():
    rs = _hpo()
    p = rs.best_estimator_.predict_proba(ordinal_categoricals(Xte))[:, 1]
    st.session_state["capA_test"] = {"auc": roc_auc_score(yte, p), "brier": brier_score_loss(yte, p)}
    st.metric("Test ROC-AUC (مرة واحدة)", f"{roc_auc_score(yte, p):.4f}")
    st.metric("Test Brier", f"{brier_score_loss(yte, p):.4f}")


def s9():
    rs = _hpo()
    from sklearn.metrics import make_scorer

    def neg_cost(a, b):
        tn, fp, fn, tp = confusion_matrix(a, b, labels=[0, 1]).ravel()
        return -(fp + 5 * fn) / len(a)
    cal = CalibratedClassifierCV(HistGradientBoostingClassifier(random_state=0, early_stopping=False, **rs.best_params_), method="sigmoid", cv=5)
    tuned = TunedThresholdClassifierCV(cal, scoring=make_scorer(neg_cost), cv=3).fit(ordinal_categoricals(Xtr), ytr)
    pr = tuned.predict(ordinal_categoricals(Xte))
    tn, fp, fn, tp = confusion_matrix(yte, pr).ravel()
    st.session_state["capA_thr"] = {"threshold": float(tuned.best_threshold_), "cost_per_case": (fp + 5 * fn) / len(yte)}
    st.markdown(f"عتبة مضبوطة داخل CV = **{tuned.best_threshold_:.3f}** (النظرية مع FN = 5·FP: 1/6 ≈ 0.167). تكلفة الاختبار لكل حالة = "
                f"{(fp + 5 * fn) / len(yte):.3f}.")
    frac, mean_p = calibration_curve(yte, tuned.estimator_.predict_proba(ordinal_categoricals(Xte))[:, 1], n_bins=8)
    st.dataframe(pd.DataFrame({"mean predicted": mean_p, "observed": frac}).round(3), hide_index=True)


def s10():
    rs = _hpo()
    p = rs.best_estimator_.predict_proba(ordinal_categoricals(Xte))[:, 1]
    err = Xte.assign(y=yte.to_numpy(), p=p, error=np.abs(yte.to_numpy() - p))
    by_emp = err.groupby("employment", observed=True)["error"].mean().sort_values(ascending=False)
    plot(bars(by_emp.index, by_emp.values, title="Mean absolute probability error by employment type"), height=280)
    st.dataframe(err.sort_values("error", ascending=False).head(8), hide_index=True, width="stretch")


def s11():
    rs = _hpo()
    pi = permutation_importance(rs.best_estimator_, ordinal_categoricals(Xte), yte, n_repeats=5, random_state=0, scoring="roc_auc")
    o = np.argsort(-pi.importances_mean)
    plot(bars(np.array(X.columns)[o], pi.importances_mean[o], title="Permutation importance (test, ROC-AUC drop)", horizontal=True), height=320)
    st.caption("تذكير: أهمية للنموذج لا أثر سببي.")


def s12():
    st.json(package_versions(("scikit-learn", "numpy", "pandas", "scipy")))
    st.markdown(f"البذرة: {RANDOM_SEED} · التقسيم: 75/25 طبقي · CV: StratifiedKFold(5, shuffle) · HPO: 20 محاولة Random")


def s13():
    test = st.session_state.get("capA_test", {})
    thr = st.session_state.get("capA_thr", {})
    rep = build_report("Capstone A — Predictive ML: loan approval", [
        ("Stage notes", {k: v or "—" for k, v in notes.items()}),
        ("Baselines & candidates (CV)", _baselines().round(4)),
        ("Held-out test (used once)", test or "run stage 8"),
        ("Decision threshold & calibration", thr or "run stage 9"),
        ("Completed stages", sorted(done)),
    ])
    st.download_button("تنزيل التقرير النهائي", rep, "capstone_predictive_report.md", "text/markdown", icon=":material/download:",
                       type="primary")
    if st.button("سجّل في سجل التجارب", key="capA_log", icon=":material/history:"):
        log_experiment("Capstone A", "HGB tuned + calibrated + threshold", {"hpo_trials": 20}, {**test, **thr} or {"none": 0.0},
                       seed=RANDOM_SEED, dataset="mixed", split="75/25 stratified + CV(5)")
        st.toast("سُجّل.", icon=":material/check:")


for i, fn in enumerate([s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13]):
    stage(i, fn)
why("اتبع الترتيب: الاختبار يُستخدم مرة واحدة في المرحلة 8.", "أي عودة للتعديل بعد رؤيته تجعل الرقم متفائلًا؛ صرّح بذلك إن حدث.")
page_footer("capstone_predictive", takeaways=["13 مرحلة من السؤال إلى التقرير.", "الاختبار مقفل حتى النهاية.", "القرارات موثقة ومبررة."])
