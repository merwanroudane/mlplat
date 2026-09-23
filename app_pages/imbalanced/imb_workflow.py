import time

import pandas as pd
import streamlit as st
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TunedThresholdClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from components.callouts import mistakes, researcher_note, warning, why
from components.diagrams import mermaid
from content.references import cite
from core.page import page_footer, page_header
from core.registry import optional
from core.state import at_least, log_experiment
from utils.datasets import xy
from utils.imbalance import cv_evaluate, make_balanced_ensemble, make_sampler
from utils.plotting import bars, plot

page_header("imb_workflow")
imb = optional("imblearn")

st.markdown("## دليل القرار في 10 خطوات")
mermaid("""flowchart TD
  A[1. Define the decision and a cost matrix] --> B[2. Measure prevalence, IR and the absolute number of positives]
  B --> C[3. Stratified or grouped/temporal split; test set keeps the real prevalence]
  C --> D[4. Baselines: majority class + strong plain model]
  D --> E[5. Choose metrics: PR-AUC + cost or precision/recall at the operating point]
  E --> F{6. Probabilities used as risks?}
  F -- yes --> G[Keep calibration: threshold moving, no resampling, or correct/calibrate]
  F -- no --> H[Try class_weight / resampling / balanced ensembles as hyperparameters]
  G --> I[7. Tune the threshold with inner CV]
  H --> I
  I --> J[8. Repeated stratified CV; compare against the plain + tuned-threshold baseline]
  J --> K[9. Final test once; bootstrap intervals]
  K --> L[10. Report prevalence, operating point, costs, calibration; monitor prior shift]
""")
warning("الخطوة الأكثر إهمالًا هي الرابعة: خط أساس «نموذج قوي عادي + عتبة مضبوطة». معظم الادعاءات بتفوق طريقة إعادة عيّنة "
        "تختفي أمامه.")

st.markdown("## مختبر المقارنة · Imbalanced Benchmark Lab")
st.caption("بيانات الاحتيال (3000 صف، ≈ 5%). اختر النماذج والاستراتيجيات؛ كل تركيبة تُقيَّم على الطيات نفسها بـRepeated "
           "stratified CV، والإعادة داخل الطيات فقط.")
MODELS = ["LogisticRegression", "RandomForest", "HistGradientBoosting"]
STRATS = ["none", "class_weight='balanced'", "tuned threshold (F1, inner CV)"]
if imb is not None:
    STRATS += ["RandomUnderSampler", "SMOTE", "BorderlineSMOTE-1", "SMOTEENN", "BalancedRandomForest (model itself)"]
c1, c2 = st.columns(2)
models = c1.multiselect("النماذج", MODELS, default=["LogisticRegression", "HistGradientBoosting"], key="wf_models")
strats = c2.multiselect("الاستراتيجيات", STRATS, default=["none", "class_weight='balanced'", "tuned threshold (F1, inner CV)"],
                        key="wf_strats")
c3, c4 = st.columns(2)
repeats = c3.segmented_control("تكرارات CV", [1, 2, 3], default=1, key="wf_rep") or 1
metric = c4.selectbox("مقياس الترتيب في الرسم", ["PR-AUC (AP)", "MCC", "F2", "balanced acc", "recall", "Brier"], key="wf_metric")


def _base(name: str, weighted: bool):
    if name == "LogisticRegression":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced" if weighted else None))
    if name == "RandomForest":
        return RandomForestClassifier(100, min_samples_leaf=3, class_weight="balanced_subsample" if weighted else None,
                                      random_state=0, n_jobs=1)
    return HistGradientBoostingClassifier(class_weight="balanced" if weighted else None, random_state=0)


def _build(model: str, strat: str):
    if strat == "none":
        return _base(model, False)
    if strat == "class_weight='balanced'":
        return _base(model, True)
    if strat.startswith("tuned threshold"):
        return TunedThresholdClassifierCV(_base(model, False), scoring="f1", cv=3)
    if strat.startswith("BalancedRandomForest"):
        return make_balanced_ensemble("BalancedRandomForestClassifier", n_estimators=100)
    from imblearn.pipeline import make_pipeline as imb_pipe
    base = _base(model, False)
    steps = list(base.named_steps.values()) if hasattr(base, "named_steps") else [base]
    return imb_pipe(*steps[:-1], make_sampler(strat), steps[-1])


@st.cache_data(show_spinner="تقييم التركيبات على الطيات نفسها…", max_entries=16)
def _run(models: tuple, strats: tuple, repeats: int) -> pd.DataFrame:
    X, y = xy("imbalanced")
    rows = []
    done = set()
    for mname in models:
        for s in strats:
            label = "BalancedRandomForest" if s.startswith("BalancedRandomForest") else f"{mname} · {s}"
            if label in done:
                continue
            done.add(label)
            t = time.perf_counter()
            r = cv_evaluate(_build(mname, s), X, y, n_splits=5, n_repeats=repeats)
            summ = r.drop(columns="fold").agg(["mean", "std"])
            rows.append({"combination": label, **{k: summ.loc["mean", k] for k in summ.columns},
                         **{f"{k} SD": summ.loc["std", k] for k in ("PR-AUC (AP)", "MCC", "F2", "recall")},
                         "seconds": time.perf_counter() - t})
    return pd.DataFrame(rows)


if st.button("شغّل المقارنة", key="wf_run", type="primary", icon=":material/play_arrow:", disabled=not (models and strats)):
    st.session_state["wf_done"] = True
if st.session_state.get("wf_done") and models and strats:
    res = _run(tuple(models), tuple(strats), repeats)
    show = ["combination", "PR-AUC (AP)", "ROC-AUC", "precision", "recall", "F1", "F2", "MCC", "balanced acc", "G-mean",
            "Brier", "mean p", "seconds"]
    st.dataframe(res[show].sort_values("PR-AUC (AP)", ascending=False).round(3), hide_index=True, width="stretch")
    err = res[f"{metric} SD"] if f"{metric} SD" in res else None
    plot(bars(res["combination"], res[metric], title=f"{metric} (mean over folds, ± SD where available)", horizontal=True,
              errors=err), height=max(300, 40 * len(res) + 120))
    best = res.sort_values("PR-AUC (AP)", ascending=False).iloc[0]
    report = (f"# Imbalanced benchmark report\n\n- Data: imbalanced (n = 3000, prevalence ≈ {xy('imbalanced')[1].mean():.3f})\n"
              f"- Protocol: repeated stratified 5-fold CV × {repeats}; resampling inside training folds only\n"
              f"- Best ranking (PR-AUC): {best['combination']} = {best['PR-AUC (AP)']:.3f} "
              f"(SD {best['PR-AUC (AP) SD']:.3f})\n"
              f"- At its operating point: precision {best['precision']:.3f}, recall {best['recall']:.3f}, "
              f"MCC {best['MCC']:.3f}\n- Calibration: mean predicted p {best['mean p']:.3f}, Brier {best['Brier']:.4f}\n\n"
              + "```text\n" + res[show].round(4).to_string(index=False) + "\n```\n")
    c1, c2 = st.columns(2)
    c1.download_button("تنزيل التقرير (Markdown)", report, file_name="imbalanced_benchmark.md", icon=":material/download:",
                       key="wf_dl")
    if c2.button("سجّل التجربة", key="wf_log", icon=":material/bookmark_add:"):
        log_experiment("Imbalanced Benchmark Lab", ", ".join(models), {"strategies": ", ".join(strats), "repeats": repeats},
                       {r["combination"]: round(r["PR-AUC (AP)"], 4) for _, r in res.iterrows()}, seed=0,
                       dataset="imbalanced", split=f"repeated stratified 5-fold × {repeats}")
        st.toast("سُجّلت.", icon=":material/check:")
    st.caption("اقرأ الجدول بثلاثة أسئلة: من يرتّب أفضل (PR-AUC)؟ من يعطي أفضل نقطة تشغيل (MCC، F2)؟ من يحافظ على المعايرة "
               "(mean p ≈ الانتشار، Brier منخفض)؟ نادرًا ما يفوز واحد في الثلاثة.")

st.markdown("## قالب تقرير")
st.code("""Prevalence (train / test / expected in deployment): 5.1% / 5.0% / ~5%
Split: stratified 5-fold × 3 (resampling inside folds only); final test untouched
Ranking: PR-AUC 0.58 [95% bootstrap CI 0.49–0.66]  (baseline = prevalence 0.05)
Operating point: threshold 0.18 chosen by inner CV to minimise cost (C_FP = 1, C_FN = 20)
  → precision 0.41, recall 0.72, alerts per 1000 cases: 88
Calibration: plain model, Brier 0.036, calibration slope 0.97 (no resampling used)
Compared with: class_weight, SMOTE, BalancedRandomForest — none improved PR-AUC beyond the CI""", language="text")

if at_least("research"):
    researcher_note([
        "ثبّت الطيات عبر كل التركيبات (نفس random_state) لتصبح الفروق مقترنة؛ قارن الفروق المقترنة لا المتوسطات المستقلة.",
        "ضبط sampling_strategy وk_neighbors وclass_weight كمعاملات فائقة يتطلب Nested CV للحصول على تقدير غير متحيز.",
        "أبلغ عن زمن التدريب: بعض الطرق (EasyEnsemble، SVMSMOTE) تضاعف التكلفة دون مكسب.",
    ])
why("اعتبر كل علاج لعدم التوازن معاملًا فائقًا يُقارن بإنصاف، لا خطوة إلزامية.",
    "بهذا تتجنب «SMOTE بالعادة» وتختار ما يثبت فائدته على بياناتك وبالمقياس الذي يهمك.")
st.markdown("### المراجع")
st.markdown(cite("he2009", "elkan2001", "vandengoorbergh2022", "lemaitre2017"))
mistakes(["مقارنة كل طريقة على تقسيم مختلف.", "إسقاط خط الأساس «عادي + عتبة».", "اختيار الفائز بمقياس واحد بلا عتبة واضحة.",
          "لمس Test أكثر من مرة."])
page_footer("imb_workflow",
            takeaways=["10 خطوات: من مصفوفة التكلفة إلى مراقبة الانتشار.", "كل علاج = معامل فائق يُقارن على الطيات نفسها.",
                       "التقرير: الانتشار، الترتيب، نقطة التشغيل، المعايرة، وعدم اليقين."])
