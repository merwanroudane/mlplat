import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from components.callouts import causal_caution, definition, intuition, mistakes, researcher_note, warning, why
from components.formulas import formula
from content.references import cite
from core.page import page_footer, page_header, page_link
from core.registry import missing_notice, optional
from core.state import at_least
from core.theme import SEQUENCE
from utils.datasets import xy
from utils.imbalance import make_sampler, prior_shift, saerens_em
from utils.plotting import plot

page_header("imb_probabilities")
imb = optional("imblearn")

st.markdown("كل علاج لعدم التوازن (Resampling أو class_weight) يغيّر **الانتشار الذي يراه النموذج** أثناء التدريب. النتيجة: "
            "احتمالات تقدّر P(y=1|x) لعالم متوازن لا وجود له. إن استُخدمت الاحتمالات كمخاطر (طب، ائتمان، تسعير) فهذه مشكلة جوهرية.")
formula(r"\frac{p_s}{1-p_s} = \frac{p}{1-p}\cdot\frac{\pi_s/(1-\pi_s)}{\pi/(1-\pi)}"
        r"\quad\Longleftrightarrow\quad p = \frac{p_s\,\pi/\pi_s}{p_s\,\pi/\pi_s + (1-p_s)(1-\pi)/(1-\pi_s)}",
        title="تصحيح الأولوية (Prior-shift correction)",
        symbols={"p": "الاحتمال تحت الانتشار الحقيقي π", "p_s": "الاحتمال من نموذج دُرّب تحت انتشار π_s",
                 r"\pi_s": "الانتشار بعد الموازنة (0.5 لموازنة كاملة أو class_weight='balanced')"},
        intuition="الموازنة تضرب نسبة الأرجحية في ثابت؛ نقسم عليه لنعود. يفترض أن P(x|y) لم يتغير (صحيح لـRandomUnderSampler "
                  "وclass_weight تقريبًا؛ تقريبي فقط لـSMOTE لأنه يولّد x جديدة).",
        example="p_s = 0.6، π_s = 0.5، π = 0.05 ⇒ p = 0.6·0.1 / (0.6·0.1 + 0.4·1.9) = 0.073.")

st.markdown("## مختبر المعايرة بعد الموازنة")


@st.cache_data(show_spinner="تنبؤات خارج الطية لأربع استراتيجيات…")
def _oof(has_imb: bool):
    X, y = xy("imbalanced")
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    lr = lambda **k: LogisticRegression(max_iter=2000, **k)  # noqa: E731
    models = {"plain": make_pipeline(StandardScaler(), lr()),
              "class_weight='balanced'": make_pipeline(StandardScaler(), lr(class_weight="balanced"))}
    if has_imb:
        from imblearn.pipeline import make_pipeline as imb_pipe
        models["RandomUnderSampler"] = imb_pipe(StandardScaler(), make_sampler("RandomUnderSampler"), lr())
        models["SMOTE"] = imb_pipe(StandardScaler(), make_sampler("SMOTE"), lr())
    probs = {k: cross_val_predict(m, X, y, cv=cv, method="predict_proba")[:, 1] for k, m in models.items()}
    return y.to_numpy(), probs


y, probs = _oof(imb is not None)
pi = y.mean()
rows, fig = [], go.Figure()
fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect", line=dict(dash="dot", color="#5C677D")))
for i, (name, p) in enumerate(probs.items()):
    variants = {name: p} if name == "plain" else {name: p, f"{name} + prior correction": prior_shift(p, pi, 0.5)}
    for j, (vname, pv) in enumerate(variants.items()):
        rows.append({"model": vname, "mean p": pv.mean(), "Brier": brier_score_loss(y, pv), "log loss": log_loss(y, pv),
                     "ROC-AUC": roc_auc_score(y, pv)})
        frac, mean_pred = calibration_curve(y, pv, n_bins=10, strategy="quantile")
        fig.add_trace(go.Scatter(x=mean_pred, y=frac, mode="lines+markers", name=vname,
                                 line=dict(color=SEQUENCE[i % len(SEQUENCE)], dash="dash" if j else "solid")))
fig.update_layout(title="Reliability diagram (quantile bins, out-of-fold)", xaxis_title="mean predicted probability",
                  yaxis_title="observed frequency", height=440)
plot(fig)
st.dataframe(pd.DataFrame(rows).round(4), hide_index=True, width="stretch")
st.caption(f"الانتشار الحقيقي = {pi:.3f}. الخطوط المتصلة للنماذج الموازنة فوق الخط القطري بكثير (مبالغة في المخاطر)؛ "
           "بعد التصحيح (متقطعة) تعود قريبًا منه، وBrier وlog loss ينخفضان. ROC-AUC لا يتغير: التصحيح دالة رتيبة لا تغيّر الترتيب.")
if imb is None:
    missing_notice("imblearn")
warning("التصحيح يعيد المعايرة لكنه لا يلغي أثر الموازنة على الترتيب: إن لم تحسّن الموازنة PR-AUC أصلًا، فالنموذج العادي مع "
        "عتبة مضبوطة أبسط وأفضل. هذا ما وجدته دراسة van den Goorbergh et al. (2022) على نماذج المخاطر السريرية.")
definition("البديل: معايرة بعد الموازنة",
           "بدل الصيغة يمكن لفّ الـPipeline المتوازن في CalibratedClassifierCV (sigmoid أو isotonic) على بيانات بالانتشار "
           "الحقيقي. مفيد حين لا ينطبق افتراض ثبات P(x|y) (مثل SMOTE أو الحذف الموجّه).")
st.code("""from sklearn.calibration import CalibratedClassifierCV
balanced = make_pipeline(StandardScaler(), SMOTE(random_state=0), LogisticRegression(max_iter=2000))  # imblearn pipeline
calibrated = CalibratedClassifierCV(balanced, method="sigmoid", cv=5)   # calibration folds keep the real prevalence
calibrated.fit(X_train, y_train)""", language="python")
page_link("calibration")

st.markdown("## مختبر تحوّل الانتشار: تقدير انتشار مجهول بلا تسميات")
st.markdown("في الإنتاج قد يتغير الانتشار (موسم احتيال، وباء). لدينا نموذج معاير لانتشار التدريب (≈ 5%) وبيانات جديدة **بلا "
            "تسميات**. خوارزمية EM لـSaerens et al. (2002) تقدّر الانتشار الجديد وتصحح الاحتمالات معًا.")
formula(r"\hat\pi^{(s+1)} = \frac{1}{N}\sum_{i=1}^{N} \frac{p_i\,\hat\pi^{(s)}/\pi_{tr}}{p_i\,\hat\pi^{(s)}/\pi_{tr} + "
        r"(1-p_i)(1-\hat\pi^{(s)})/(1-\pi_{tr})}",
        symbols={r"\pi_{tr}": "الانتشار في التدريب", "p_i": "احتمال النموذج للحالة i", r"\hat\pi^{(s)}": "تقدير الانتشار الجديد"})
new_pi = st.select_slider("الانتشار الحقيقي في بيانات النشر (مخفي عن الخوارزمية)", [0.01, 0.02, 0.05, 0.1, 0.2, 0.3],
                          value=0.2, key="em_pi")


@st.cache_data(show_spinner="20 سحبًا عشوائيًا لبيانات النشر…", max_entries=10)
def _em(new_pi: float) -> tuple[pd.DataFrame, int]:
    X, y = xy("imbalanced")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=0, stratify=y)
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X_tr, y_tr)
    p_all = m.predict_proba(X_te)[:, 1]
    yv = y_te.to_numpy()
    pos, neg = np.flatnonzero(yv == 1), np.flatnonzero(yv == 0)
    n_pos = len(pos)
    n_neg = int(round(n_pos * (1 - new_pi) / new_pi))
    if n_neg > len(neg):  # low prevalence: keep all negatives and subsample positives instead
        n_neg = len(neg)
        n_pos = max(int(round(n_neg * new_pi / (1 - new_pi))), 3)
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(20):
        idx = np.r_[rng.choice(pos, n_pos, replace=False), rng.choice(neg, n_neg, replace=False)]
        p = p_all[idx]
        rows.append({"truth": yv[idx].mean(), "mean p (no correction)": p.mean(), "classify & count (p ≥ 0.5)": (p >= 0.5).mean(),
                     "EM (Saerens et al., 2002)": saerens_em(p, y_tr.mean())[0]})
    return pd.DataFrame(rows), n_pos + n_neg


draws, n_new = _em(new_pi)
methods = ["mean p (no correction)", "classify & count (p ≥ 0.5)", "EM (Saerens et al., 2002)"]
summary = pd.DataFrame([{"method": k, "mean estimate": draws[k].mean(), "SD across draws": draws[k].std(),
                         "MAE vs truth": (draws[k] - draws["truth"]).abs().mean()} for k in methods])
c1, c2 = st.columns([1.3, 1])
with c1:
    plot(go.Figure(go.Bar(x=methods, y=summary["mean estimate"], error_y=dict(type="data", array=summary["SD across draws"]),
                          marker_color=SEQUENCE[1:4], text=[f"{v:.3f}" for v in summary["mean estimate"]], textposition="auto"))
         .add_hline(y=draws["truth"].mean(), line=dict(dash="dot", color="#212529"), annotation_text="true prevalence")
         .update_layout(title=f"Estimated prevalence ({n_new} unlabeled cases per draw, 20 draws)", yaxis_title="prevalence",
                        height=380))
with c2:
    st.dataframe(summary.round(3), hide_index=True, width="stretch")
best = summary.sort_values("MAE vs truth").iloc[0]["method"]
st.caption(f"الأقل خطأ في المتوسط هنا: **{best}**. لاحظ الانحراف المعياري: مع بضع عشرات من الموجبات في كل سحب، تقدير "
           "الانتشار نفسه غير مؤكد — أي تحديث تلقائي للعتبة يجب أن يأخذ ذلك في الحسبان.")
intuition("متوسط الاحتمالات «يسحب» التقدير نحو انتشار التدريب لأن النموذج يحمل أولويته؛ Classify & count يعتمد على العتبة. "
          "EM يعيد التوازن بين الأولوية والأدلة في البيانات الجديدة. الشرط: ثبات P(x|y) (Label shift) واحتمالات معايرة في التدريب.")
causal_caution("تحوّل الانتشار ليس تحوّل المفهوم (Concept drift): إن تغيّرت P(y|x) نفسها (أساليب احتيال جديدة)، فلا EM ولا "
               "التصحيح يكفيان؛ يلزم إعادة تسمية وتدريب.")

if at_least("advanced"):
    st.markdown("## متقدم: متى يكون التصحيح دقيقًا؟")
    st.markdown("- **RandomUnderSampler:** يحذف عشوائيًا ⇒ P(x|y) ثابت ⇒ التصحيح صحيح نظريًا.\n"
                "- **class_weight:** يعادل تغيير الأولوية في الخسارة ⇒ صحيح للنموذج الصحيح المواصفة، تقريبي مع التنظيم.\n"
                "- **SMOTE / ADASYN / NearMiss / التنظيف:** تغيّر P(x|y) ⇒ التصحيح تقريبي؛ استخدم المعايرة التجريبية.")
if at_least("research"):
    researcher_note([
        "تقدير الانتشار في بيانات بلا تسميات مجال بحثي باسم Quantification؛ EM أحد أقدم وأقوى طرقه تحت Label shift.",
        "في النماذج السريرية، أبلغ عن معايرة (calibration-in-the-large، الميل) إلى جانب التمييز.",
        "اختبر حساسية القرار لخطأ تقدير الانتشار قبل تحديث العتبة تلقائيًا في الإنتاج.",
    ])
why("إن كانت الاحتمالات ستُقرأ كأرقام (مخاطر، أسعار)، لا توازن — أو صحّح وعاير بعد الموازنة.",
    "العتبة يمكن تحريكها في أي وقت؛ الاحتمالات المشوهة تضلل كل من يستخدمها لاحقًا.")
st.markdown("### المراجع")
st.markdown(cite("saerens2002", "vandengoorbergh2022", "elkan2001", "niculescu2005"))
mistakes(["قراءة احتمالات نموذج متوازن كمخاطر حقيقية.", "تصحيح الأولوية مع π_s خاطئة (مثلًا نسيان أن الموازنة كانت جزئية).",
          "افتراض أن التصحيح يحسن ROC-AUC.", "الخلط بين تحوّل الانتشار وتحوّل المفهوم."])
page_footer("imb_probabilities",
            takeaways=["الموازنة ترفع الاحتمالات: صحّحها بصيغة الأولوية أو عاير.", "التصحيح لا يغيّر الترتيب.",
                       "EM يقدّر الانتشار الجديد من بيانات بلا تسميات تحت Label shift."])
