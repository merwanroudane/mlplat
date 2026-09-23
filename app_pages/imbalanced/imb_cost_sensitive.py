import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from components.callouts import definition, mistakes, researcher_note, warning, why
from components.cards import comparison_table
from components.formulas import formula
from content.references import cite
from core.page import page_footer, page_header, page_link
from core.state import at_least, log_experiment
from core.theme import PALETTE
from utils.datasets import xy
from utils.imbalance import balanced_class_weight, cost_optimal_threshold, expected_cost, focal_loss
from utils.plotting import lines, plot

page_header("imb_cost_sensitive")

st.markdown("## مصفوفة التكلفة")
st.markdown("السؤال الصحيح ليس «كم نسبة الأقلية؟» بل «**كم يكلّف كل نوع خطأ؟**». تفويت احتيال بـ5000 دولار ليس كإزعاج عميل "
            "بإنذار كاذب يكلّف مكالمة بـ5 دولارات.")
comparison_table([
    {"": "الحقيقة: سالب (0)", "التنبؤ: سالب": "C_TN (عادة 0)", "التنبؤ: موجب": "C_FP — إنذار كاذب"},
    {"": "الحقيقة: موجب (1)", "التنبؤ: سالب": "C_FN — حالة فائتة", "التنبؤ: موجب": "C_TP (عادة 0 أو تكلفة المعالجة)"},
])
formula(r"t^\star = \frac{C_{FP} - C_{TN}}{(C_{FP} - C_{TN}) + (C_{FN} - C_{TP})}\ \ \overset{C_{TP}=C_{TN}=0}{=}\ \ "
        r"\frac{C_{FP}}{C_{FP}+C_{FN}}",
        title="العتبة المثلى (Elkan, 2001)",
        symbols={"t^\\star": "تنبأ بموجب إذا P(y=1|x) ≥ t*", "C_{FP}, C_{FN}": "تكلفة الإنذار الكاذب والتفويت"},
        intuition="نتنبأ بموجب حين تكون التكلفة المتوقعة للتنبؤ بموجب أقل: (1−p)·C_FP < p·C_FN.",
        example="C_FP = 1، C_FN = 19 ⇒ t* = 1/20 = 0.05. مع احتمالات معايرة، أي حالة احتمالها ≥ 5% تستحق المراجعة.")
warning("الصيغة تفترض **احتمالات معايرة** للانتشار الحقيقي. بعد SMOTE أو class_weight لم تعد الاحتمالات كذلك، ويجب "
        "إما تصحيحها (صفحة الاحتمالات) أو ضبط العتبة تجريبيًا بالتحقق.")

st.markdown("## ثلاث طرق لإدخال التكلفة")
comparison_table([
    {"الطريقة": "تحريك العتبة (Threshold moving)", "أين": "بعد التدريب", "الأداة": "t* نظريًا، أو TunedThresholdClassifierCV",
     "الاحتمالات": "تبقى معايرة"},
    {"الطريقة": "إعادة الوزن (Reweighting)", "أين": "أثناء التدريب", "الأداة": "class_weight، sample_weight، scale_pos_weight",
     "الاحتمالات": "تُزاح نحو الأقلية"},
    {"الطريقة": "إعادة العيّنة (Resampling)", "أين": "قبل التدريب (داخل الطيات)", "الأداة": "imbalanced-learn",
     "الاحتمالات": "تُزاح نحو الأقلية"},
])
formula(r"\hat\beta = \arg\min_\beta\ \sum_{i=1}^{n} w_{y_i}\,\ell\big(y_i, f_\beta(x_i)\big) + \text{penalty},\qquad "
        r"w_c^{\text{balanced}} = \frac{n}{K\,n_c}",
        title="class_weight = خسارة موزونة",
        symbols={"w_c": "وزن الفئة c", "K": "عدد الفئات", "n_c": "عدد أمثلة الفئة c"},
        intuition="كأن كل مثال نادر مكرر w₁/w₀ مرة. في الانحدار اللوجستي يعادل ذلك تقريبًا إزاحة الحد الثابت بـlog(w₁/w₀).")
X, y = xy("imbalanced")
w = balanced_class_weight(y)
st.markdown(f"على بيانات الاحتيال: w₀ = **{w[0]:.3f}**، w₁ = **{w[1]:.3f}**، ونسبتهما {w[1] / w[0]:.1f} = IR.")


@st.cache_data(show_spinner="تنبؤات خارج الطية لنموذجين…")
def _oof():
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    plain = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    bal = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    p0 = cross_val_predict(plain, X, y, cv=cv, method="predict_proba")[:, 1]
    p1 = cross_val_predict(bal, X, y, cv=cv, method="predict_proba")[:, 1]
    i0 = plain.fit(X, y)[-1].intercept_[0]
    i1 = bal.fit(X, y)[-1].intercept_[0]
    return p0, p1, i0, i1


p_plain, p_bal, int_plain, int_bal = _oof()
c1, c2, c3 = st.columns(3)
c1.metric("الحد الثابت: عادي", f"{int_plain:.2f}", border=True)
c2.metric("الحد الثابت: balanced", f"{int_bal:.2f}", border=True)
c3.metric("الفرق مقابل log(w₁/w₀)", f"{int_bal - int_plain:.2f} / {np.log(w[1] / w[0]):.2f}", border=True)
st.caption("الإزاحة قريبة من log(w₁/w₀) لكن ليست مساوية تمامًا: التنظيم (C) والمواصفة الخاطئة يغيّران باقي المعاملات أيضًا.")

st.markdown("## مختبر التكلفة · Cost Lab")
c1, c2 = st.columns(2)
cfp = c1.number_input("C_FP (تكلفة إنذار كاذب)", 0.1, 100.0, 1.0, 0.5, key="cost_fp")
cfn = c2.number_input("C_FN (تكلفة حالة فائتة)", 0.1, 500.0, 20.0, 1.0, key="cost_fn")
grid = np.linspace(0.01, 0.99, 99)
cost_plain = expected_cost(y, p_plain, grid, cfp, cfn)
cost_bal = expected_cost(y, p_bal, grid, cfp, cfn)
t_star = cost_optimal_threshold(cfp, cfn)
fig = go.Figure()
fig.add_trace(go.Scatter(x=grid, y=cost_plain, name="plain LR", line=dict(color=PALETTE["sky"], width=3)))
fig.add_trace(go.Scatter(x=grid, y=cost_bal, name="LR class_weight='balanced'", line=dict(color=PALETTE["coral"], width=3,
                                                                                       dash="dash")))
fig.add_vline(x=t_star, line=dict(color=PALETTE["sky"], dash="dot"), annotation_text=f"t* = {t_star:.3f}")
fig.add_vline(x=0.5, line=dict(color=PALETTE["muted"], dash="dot"), annotation_text="0.5")
fig.update_layout(title="Expected cost per case vs threshold (out-of-fold predictions)", xaxis_title="threshold",
                  yaxis_title="average cost per case", height=420)
plot(fig)
best_plain, best_bal = grid[cost_plain.argmin()], grid[cost_bal.argmin()]
rows = pd.DataFrame([
    {"النموذج / العتبة": "plain @ 0.5", "التكلفة لكل حالة": expected_cost(y, p_plain, [0.5], cfp, cfn)[0]},
    {"النموذج / العتبة": f"plain @ t* = {t_star:.3f} (نظري)", "التكلفة لكل حالة": expected_cost(y, p_plain, [t_star], cfp, cfn)[0]},
    {"النموذج / العتبة": f"plain @ أفضل عتبة تجريبية {best_plain:.2f}", "التكلفة لكل حالة": cost_plain.min()},
    {"النموذج / العتبة": "balanced @ 0.5", "التكلفة لكل حالة": expected_cost(y, p_bal, [0.5], cfp, cfn)[0]},
    {"النموذج / العتبة": f"balanced @ أفضل عتبة تجريبية {best_bal:.2f}", "التكلفة لكل حالة": cost_bal.min()},
])
st.dataframe(rows.round(4), hide_index=True, width="stretch")
cost_at_tstar = expected_cost(y, p_plain, [t_star], cfp, cfn)[0]
regret = cost_at_tstar - cost_plain.min()
st.markdown(
    f"- **أدنى تكلفة:** عادي {cost_plain.min():.3f} مقابل موزون {cost_bal.min():.3f} — متقاربتان: إعادة الوزن تحرك نقطة "
    "التشغيل ولا تضيف معلومات.\n"
    f"- **t* النظرية مقابل الأفضل تجريبيًا:** t* = {t_star:.3f} تكلف {cost_at_tstar:.3f}، أي أعلى بـ{regret:.3f} من الأفضل "
    f"تجريبيًا (عند {best_plain:.2f}). "
    + ("الفرق صغير: احتمالات النموذج قريبة من المعايرة في منطقة القرار."
       if regret < 0.05 * cost_plain.min() else
       "الفرق ملحوظ: متوسط الاحتمالات يطابق الانتشار (معايرة إجمالية) لكن الاحتمالات ليست معايرة في كل المناطق — "
       "النموذج الخطي مواصفته ناقصة هنا. لذلك تُضبط العتبة عمليًا بالتحقق الداخلي (TunedThresholdClassifierCV) أو "
       "بعد معايرة أفضل، وتبقى t* نقطة انطلاق نظرية.")
)
if st.button("سجّل التجربة", key="cost_log", icon=":material/bookmark_add:"):
    log_experiment("Cost Lab", "LogisticRegression plain vs balanced", {"C_FP": cfp, "C_FN": cfn},
                   {"t*": round(t_star, 4), "min cost plain": round(float(cost_plain.min()), 4),
                    "min cost balanced": round(float(cost_bal.min()), 4)}, seed=0, dataset="imbalanced", split="5-fold OOF")
    st.toast("سُجّلت.", icon=":material/check:")
st.code("""from sklearn.metrics import make_scorer
from sklearn.model_selection import TunedThresholdClassifierCV

def neg_cost(y_true, y_pred, c_fp=1.0, c_fn=20.0):
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    return -(c_fp * fp + c_fn * fn) / len(y_true)

model = TunedThresholdClassifierCV(pipe, scoring=make_scorer(neg_cost), cv=5)   # threshold chosen by inner CV
model.fit(X_train, y_train)
model.best_threshold_""", language="python")

st.markdown("## أوزان لكل مثال: sample_weight")
st.markdown("حين تختلف التكلفة من حالة لأخرى (مبلغ المعاملة مثلًا)، مرّر وزنًا لكل صف بدل وزن لكل فئة:")
st.code("""w = np.where(y_train == 1, amount_train, 1.0)          # a missed fraud costs its amount
model.fit(X_train, y_train, logisticregression__sample_weight=w)   # routed to the pipeline step""", language="python")
definition("مكافئات في مكتبات التعزيز",
           "XGBoost: `scale_pos_weight` (≈ n_neg / n_pos كنقطة بداية)؛ LightGBM: `is_unbalance=True` أو `class_weight`؛ "
           "CatBoost: `auto_class_weights='Balanced'` أو `'SqrtBalanced'`. كلها إعادة وزن ⇒ تزيح الاحتمالات.")

st.markdown("## Focal loss")
formula(r"\mathrm{FL}(p_t) = -\alpha_t\,(1-p_t)^{\gamma}\,\log p_t",
        title="Focal loss (Lin et al., 2017)",
        symbols={"p_t": "الاحتمال المعطى للفئة الصحيحة", r"\gamma": "معامل التركيز (γ = 0 ⇒ Log loss)",
                 r"\alpha_t": "وزن الفئة"},
        intuition="الحد (1 − p_t)^γ يقلّص خسارة الأمثلة السهلة المصنفة جيدًا (غالبًا الأغلبية) فيتركز التدريب على الصعبة.")
pt = np.linspace(0.01, 0.99, 99)
series = {f"γ = {g}": [focal_loss(np.array([1]), np.array([v]), gamma=g) for v in pt] for g in (0, 0.5, 1, 2, 5)}
plot(lines(pt, series, title="Focal loss vs probability of the true class", xaxis="p_t", yaxis="loss"), height=340)
st.caption("غير متاحة كخيار في scikit-learn؛ تُستخدم كدالة هدف مخصصة في التعلم العميق أو في XGBoost/LightGBM. تغيّر الاحتمالات أيضًا.")
page_link("threshold_tuning")

if at_least("advanced"):
    st.markdown("## متقدم: لماذا تعادل إعادة الوزن تغيير الأولوية؟")
    formula(r"\frac{P_w(y=1\mid x)}{P_w(y=0\mid x)} = \frac{w_1}{w_0}\cdot\frac{P(y=1\mid x)}{P(y=0\mid x)}",
            intuition="الأوزان تضرب نسبة الأرجحية الحقيقية في ثابت (للنموذج الصحيح المواصفة)، وهذا مكافئ لتحريك العتبة إلى "
                      "w₀/(w₀ + w₁) على الاحتمال الأصلي.")
if at_least("research"):
    researcher_note([
        "Elkan (2001): مصفوفة التكلفة 2×2 (بشروط معقولة) تُختزل إلى عتبة واحدة على الاحتمال، وتغيير نسبة السالبة في "
        "التدريب يعادل تغيير هذه العتبة — فإعادة العيّنة ليست ضرورية إن كانت الاحتمالات جيدة.",
        "عندما تُستخدم الاحتمالات كمخاطر (طب، ائتمان)، فضّل نموذجًا معايرًا + عتبة على إعادة الوزن.",
        "مع تكاليف غير مؤكدة، أبلغ عن منحنى التكلفة عبر مدى من نسب C_FN/C_FP بدل رقم واحد.",
    ])
why("ابدأ بمصفوفة تكلفة ولو تقريبية، ثم احسب t*.", "هذا يحوّل «عدم التوازن» من مشكلة تقنية إلى قرار أعمال شفاف.")
st.markdown("### المراجع")
st.markdown(cite("elkan2001", "lin2017focal", "he2009"))
mistakes(["استخدام 0.5 مع تكاليف غير متماثلة.", "حساب t* على احتمالات مُعاد وزنها.",
          "ضبط العتبة على Test.", "افتراض أن class_weight يضيف معلومات جديدة."])
page_footer("imb_cost_sensitive",
            takeaways=["t* = C_FP/(C_FP + C_FN) على احتمالات معايرة.", "class_weight = خسارة موزونة = إزاحة الأولوية.",
                       "أدنى تكلفة متقاربة بإعادة الوزن أو بتحريك العتبة؛ الثاني يحافظ على المعايرة.",
                       "sample_weight للتكاليف الفردية؛ Focal loss للتعلم العميق."])
