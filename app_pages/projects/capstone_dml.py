import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LassoCV, LogisticRegressionCV

from components.callouts import causal_caution
from components.diagrams import mermaid
from config import RANDOM_SEED
from core.page import page_footer, page_header
from core.registry import optional
from core.state import log_experiment
from core.theme import PALETTE
from utils import causal as C
from utils.datasets import load_dataset
from utils.plotting import interval_plot, plot
from utils.report import build_report

page_header("capstone_dml")
st.markdown("مشروع سببي نهائي: «ما أثر برنامج (d) على النتيجة (y)؟» على بيانات المنصة السببية. 11 مرحلة، كل منها تتطلب قرارًا "
            "مكتوبًا، والتقرير النهائي يجمعها.")
STAGES = ["1 Causal question", "2 Treatment", "3 Outcome", "4 Covariates", "5 Identification assumptions", "6 Nuisance learners",
          "7 Cross-fitting", "8 DML estimate", "9 Inference", "10 Sensitivity / limitations", "11 Final report"]
done = st.session_state.setdefault("capB_done", set())
notes = st.session_state.setdefault("capB_notes", {})
st.progress(len(done) / len(STAGES), text=f"{len(done)} / {len(STAGES)} مراحل مكتملة")
df = load_dataset("causal")
X, y, d = df.filter(like="x").to_numpy(), df["y"].to_numpy(), df["d"].to_numpy()


HEAVY = {4, 7, 9}  # stages with model training: computed on demand


def stage(i: int, body) -> None:
    with st.expander(STAGES[i], icon=":material/check_circle:" if STAGES[i] in done else ":material/radio_button_unchecked:"):
        if i in HEAVY and not st.session_state.get(f"capB_run_{i}"):
            if st.button("شغّل حسابات هذه المرحلة", key=f"capB_btn_{i}", icon=":material/play_arrow:"):
                st.session_state[f"capB_run_{i}"] = True
                st.rerun()
        else:
            body()
        notes[STAGES[i]] = st.text_area("قرارك وتبريره", value=notes.get(STAGES[i], ""), key=f"capB_note_{i}", height=80)
        if st.toggle("أكملت هذه المرحلة", value=STAGES[i] in done, key=f"capB_t_{i}"):
            done.add(STAGES[i])
        else:
            done.discard(STAGES[i])


learner = st.session_state.get("capB_learner", "Random Forest")


@st.cache_data(show_spinner="DML-IRM مع Cross-fitting…", max_entries=8)
def _estimate(learner: str, n_folds: int, trim: float):
    if learner == "Random Forest":
        g, m = RandomForestRegressor(200, min_samples_leaf=5, random_state=0, n_jobs=1), RandomForestClassifier(200, min_samples_leaf=5,
                                                                                                                 random_state=0, n_jobs=1)
    else:
        g, m = LassoCV(cv=3, random_state=0), LogisticRegressionCV(cv=3, max_iter=3000, l1_ratios=(0.0,), use_legacy_attributes=False)
    ate = C.dml_irm(X, y, d, g, m, n_folds=n_folds, score="ATE", trim=trim, seed=RANDOM_SEED)
    att = C.dml_irm(X, y, d, g, m, n_folds=n_folds, score="ATT", trim=trim, seed=RANDOM_SEED)
    return ate, att


def s1():
    st.markdown("صُغ السؤال بصيغة تدخّل: «ماذا لو شارك الجميع مقابل لا أحد (ATE)؟» أو «ما أثره على المشاركين (ATT)؟».")


def s2():
    st.markdown(f"d ثنائي: {d.mean():.1%} معالَجون. عرّف متى يُعتبر الشخص «معالَجًا» ومتى بدأت المعالجة.")


def s3():
    st.markdown("y مستمرة تُقاس **بعد** المعالجة. تحقق أن قياسها لا يتأثر باختيار المشاركة (نفس الأداة لكل المجموعات).")


def s4():
    st.markdown("x1..x5 مقاسة **قبل** المعالجة. لا تضف متغيرات بعدها.")
    mermaid("flowchart LR\n  X[x1..x5 pre-treatment] --> D[d] & Y[y]\n  D --> Y")


def s5():
    st.markdown("- Unconfoundedness: لا مربكات غير مقاسة (حجة من المجال).\n- Overlap: افحص درجات الميل أدناه.\n- SUTVA: لا تداخل بين الأفراد.")
    ate, _ = _estimate(learner, 5, 0.01)
    mh = ate.extra["m_hat"]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=mh[d == 1], name="treated", opacity=0.6, nbinsx=40, marker_color=PALETTE["coral"]))
    fig.add_trace(go.Histogram(x=mh[d == 0], name="control", opacity=0.6, nbinsx=40, marker_color=PALETTE["sky"]))
    fig.update_layout(barmode="overlay", title="Overlap check: cross-fitted propensity scores", height=280)
    plot(fig)


def s6():
    st.session_state["capB_learner"] = st.segmented_control("متعلمو الإزعاج", ["Random Forest", "Lasso / Logistic"],
                                                            default=learner, key="capB_l", required=True)
    st.caption("اختر قبل رؤية التقدير، وبرر من طبيعة العلاقات المتوقعة.")


def s7():
    st.markdown("5 طيات Cross-fitting، DML2، قص درجات الميل عند 0.01. (n_rep = 1 هنا للسرعة؛ استخدم أكثر في التقرير الحقيقي.)")


def s8():
    ate, att = _estimate(st.session_state.get("capB_learner", "Random Forest"), 5, 0.01)
    st.session_state["capB_res"] = {"ATE": ate.theta, "ATE_se": ate.se, "ATT": att.theta, "ATT_se": att.se}
    plot(interval_plot(["naive difference", "DML-IRM ATE", "DML-IRM ATT"], [y[d == 1].mean() - y[d == 0].mean(), ate.theta, att.theta],
                       [np.nan, *ate.ci[:1], *att.ci[:1]], [np.nan, ate.ci[1], att.ci[1]], title="Estimates with 95% CIs"))


def s9():
    r = st.session_state.get("capB_res")
    if r:
        st.markdown(f"ATE = {r['ATE']:.3f} (SE {r['ATE_se']:.3f})؛ ATT = {r['ATT']:.3f} (SE {r['ATT_se']:.3f}). الفترات تقاربية طبيعية "
                    "صالحة تحت الافتراضات وشروط معدل الإزعاج.")
    else:
        st.info("شغّل المرحلة 8 أولًا.")


def s10():
    dml = optional("doubleml")
    if dml is None:
        st.info("DoubleML غير مثبتة: ناقش الحساسية نصيًا.")
        return
    data = dml.DoubleMLData(df.drop(columns=["tau", "true_ps"]), "y", "d", [f"x{i}" for i in range(1, 6)])
    irm = dml.DoubleMLIRM(data, RandomForestRegressor(150, min_samples_leaf=5, random_state=0, n_jobs=1),
                          RandomForestClassifier(150, min_samples_leaf=5, random_state=0, n_jobs=1), n_folds=5).fit()
    irm.sensitivity_analysis(cf_y=0.03, cf_d=0.03)
    sp = irm.sensitivity_params
    st.session_state["capB_sens"] = {"theta_lower": float(np.ravel(sp["theta"]["lower"])[0]), "theta_upper": float(np.ravel(sp["theta"]["upper"])[0]),
                                     "RV": float(np.ravel(sp["rv"])[0])}
    st.json(st.session_state["capB_sens"])
    causal_caution("اذكر القيود: المربكات غير المقاسة المحتملة، تعميم النتيجة، وجودة القياس.")


def s11():
    rep = build_report("Capstone B — Causal / DML: programme effect", [
        ("Stage notes", {k: v or "—" for k, v in notes.items()}),
        ("Estimates", st.session_state.get("capB_res", "run stage 8")),
        ("Sensitivity (cf_y = cf_d = 0.03)", st.session_state.get("capB_sens", "run stage 10")),
        ("Design", {"model": "IRM (AIPW score)", "cross-fitting": "5 folds, DML2", "trimming": 0.01,
                    "learners": st.session_state.get("capB_learner", "Random Forest")}),
        ("Completed stages", sorted(done)),
    ])
    st.download_button("تنزيل التقرير النهائي", rep, "capstone_dml_report.md", "text/markdown", icon=":material/download:", type="primary")
    if st.button("سجّل في سجل التجارب", key="capB_log", icon=":material/history:") and st.session_state.get("capB_res"):
        log_experiment("Capstone B", "DML-IRM", {"learners": st.session_state.get("capB_learner"), "n_folds": 5, "trim": 0.01},
                       st.session_state["capB_res"], seed=RANDOM_SEED, dataset="causal", split="5-fold cross-fitting")
        st.toast("سُجّل.", icon=":material/check:")


for i, fn in enumerate([s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11]):
    stage(i, fn)
page_footer("capstone_dml", takeaways=["السؤال السببي والافتراضات قبل الخوارزمية.", "DML للتقدير؛ الحساسية للمصداقية.",
                                       "التقرير يوثق كل قرار."])
