import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression

from components.callouts import intuition, mistakes, researcher_note, warning, why
from components.cards import comparison_table
from components.formulas import formula
from content.references import cite
from core.page import page_footer, page_header
from core.state import at_least
from core.theme import PALETTE
from utils.plotting import plot

page_header("fairness")

st.markdown("## مصادر التحيز")
comparison_table([
    {"المصدر": "Representation bias", "الوصف": "مجموعة ممثلة تمثيلًا ناقصًا في البيانات", "مثال": "بيانات طبية من فئة عمرية واحدة"},
    {"المصدر": "Measurement bias", "الوصف": "المتغير يُقاس بدقة مختلفة بين المجموعات", "مثال": "اعتقالات كبديل لـ«الجريمة»"},
    {"المصدر": "Label bias", "الوصف": "التسميات نفسها نتاج قرارات متحيزة", "مثال": "«الأداء الوظيفي» من تقييمات مديرين"},
    {"المصدر": "Historical bias", "الوصف": "البيانات دقيقة لكنها تعكس ظلمًا تاريخيًا", "مثال": "أنماط إقراض سابقة"},
    {"المصدر": "Deployment bias", "الوصف": "استخدام النموذج في سياق لم يُصمم له", "مثال": "نموذج مدينة على مدينة أخرى"},
])
formula(r"\text{DP gap} = \big|P(\hat Y=1\mid A=a) - P(\hat Y=1\mid A=b)\big|,\qquad \text{EO gap} = \big|P(\hat Y=1\mid Y=1,A=a) - P(\hat Y=1\mid Y=1,A=b)\big|",
        title="Demographic parity and equal opportunity (Hardt et al., 2016)",
        symbols={"A": "السمة الحساسة (مجموعة)", "DP": "معدل القرار الموجب متساوٍ", "EO": "معدل الموجب الحقيقي (Recall) متساوٍ"})

st.markdown("## Fairness Audit Lab")
st.caption("بيانات اصطناعية لقرار قبول: المجموعتان A وB تختلفان في الانتشار الأساسي وفي جودة قياس خاصية (Measurement bias).")
c1, c2, c3 = st.columns(3)
base_gap = c1.slider("فرق الانتشار الأساسي", 0.0, 1.5, 0.6, 0.1, key="fa_gap")
noise_b = c2.slider("ضجيج القياس للمجموعة B", 0.0, 2.0, 1.0, 0.1, key="fa_noise")
use_group = c3.toggle("أدخل السمة الحساسة كخاصية", value=False, key="fa_use")
rng = np.random.default_rng(0)
n = 6000
g = rng.integers(0, 2, n)  # 0 = A, 1 = B
skill = rng.normal(size=n)
y = (rng.random(n) < 1 / (1 + np.exp(-(1.5 * skill - base_gap * g)))).astype(int)
measured = skill + rng.normal(scale=np.where(g == 1, noise_b, 0.3))
other = rng.normal(size=n) + 0.3 * y
Xf = np.c_[measured, other] if not use_group else np.c_[measured, other, g]
model = LogisticRegression().fit(Xf[: n // 2], y[: n // 2])
p = model.predict_proba(Xf[n // 2:])[:, 1]
gt, yt = g[n // 2:], y[n // 2:]
common_t = st.slider("العتبة المشتركة", 0.1, 0.9, 0.5, 0.05, key="fa_t")


def rates(t_a, t_b):
    pred = np.where(gt == 0, p >= t_a, p >= t_b).astype(int)
    out = []
    for grp, name in ((0, "A"), (1, "B")):
        m = gt == grp
        tp = np.sum((pred == 1) & (yt == 1) & m)
        out.append({"group": name, "n": int(m.sum()), "base rate": yt[m].mean(), "selection rate": pred[m].mean(),
                    "TPR (recall)": tp / max(np.sum((yt == 1) & m), 1),
                    "FPR": np.sum((pred == 1) & (yt == 0) & m) / max(np.sum((yt == 0) & m), 1),
                    "precision": tp / max(np.sum((pred == 1) & m), 1), "accuracy": np.mean(pred[m] == yt[m])})
    return pd.DataFrame(out)


r = rates(common_t, common_t)
st.dataframe(r.round(3), hide_index=True, width="stretch")
with st.container(horizontal=True):
    st.metric("Demographic parity gap", f"{abs(r['selection rate'][0] - r['selection rate'][1]):.3f}", border=True)
    st.metric("Equal opportunity gap (TPR)", f"{abs(r['TPR (recall)'][0] - r['TPR (recall)'][1]):.3f}", border=True)
    st.metric("FPR gap", f"{abs(r['FPR'][0] - r['FPR'][1]):.3f}", border=True)
fig = go.Figure()
for grp, name, color in ((0, "A", PALETTE["sky"]), (1, "B", PALETTE["coral"])):
    fig.add_trace(go.Histogram(x=p[gt == grp], name=f"group {name}", opacity=0.6, nbinsx=40, marker_color=color))
fig.add_vline(x=common_t, line=dict(color="#212529", dash="dash"))
fig.update_layout(barmode="overlay", title="Score distributions by group", xaxis_title="P(y=1)", height=300)
plot(fig)
intuition("ضجيج القياس الأكبر للمجموعة B يجعل درجاتها أقرب إلى المتوسط ⇒ حتى مع عتبة واحدة يختلف Recall بين المجموعتين. حذف "
          "السمة الحساسة («Fairness through unawareness») لا يزيل التفاوت لأن الخصائص الأخرى تحمل أثرها.")
warning("**لا توجد مقياس واحد يمثل «الحل الأخلاقي».** حين يختلف الانتشار الأساسي بين المجموعات، يستحيل رياضيًا تحقيق المعايرة ومساواة "
        "FPR وFNR معًا (Kleinberg et al., 2016؛ Chouldechova, 2017). اختيار المقياس قرار معياري يجب إشراك أصحاب المصلحة فيه.")
comparison_table([
    {"المعيار": "Demographic parity", "يطلب": "معدلات قبول متساوية", "متى يُطرح": "حين تُعتبر الفروق الأساسية نتاج ظلم"},
    {"المعيار": "Equal opportunity", "يطلب": "Recall متساوٍ", "متى يُطرح": "حين تكون الفرص المستحقة هي الأهم"},
    {"المعيار": "Equalized odds", "يطلب": "TPR وFPR متساويان", "متى يُطرح": "حين تهم الأخطاء بنوعيها"},
    {"المعيار": "Calibration within groups", "يطلب": "p = 0.7 تعني 70% في كل مجموعة", "متى يُطرح": "حين تُستخدم الدرجات مباشرة"},
])
why("قيّم الأداء حسب المجموعات دائمًا، حتى لو لم تستهدف معيار عدالة معين.", "المتوسط العام يخفي فشلًا في مجموعة صغيرة.")

st.markdown("## الخصوصية وتقليل البيانات")
st.markdown("- **Data minimization:** اجمع واحتفظ فقط بما يلزم للغرض المعلن.\n- **لا تُرسل بيانات المستخدم** إلى خدمات خارجية دون موافقة "
            "(هذه المنصة لا تفعل ذلك إطلاقًا).\n- **النماذج تتسرب:** قد تكشف النماذج معلومات عن بيانات تدريبها (Membership inference).\n"
            "- **Model cards:** وثّق الاستخدام المقصود والقيود والأداء حسب المجموعات (Mitchell et al., 2019).")
if at_least("research"):
    researcher_note(["Hardt, Price & Srebro (2016) لـEqualized odds/Equal opportunity وتعديل العتبات بعد التدريب.",
                     "Chouldechova (2017) وKleinberg et al. (2016) لنتائج الاستحالة.",
                     "العدالة السببية (Counterfactual fairness) تربط هذا الموضوع بمسار Causal ML."])
    st.markdown(cite("hardt2016", "chouldechova2017", "kleinberg2016", "mitchell2019"))
mistakes(["الاعتماد على «حذف السمة الحساسة».", "مقياس عدالة واحد كحل نهائي.", "تقييم أداء كلي دون تقسيم المجموعات."])
page_footer("fairness",
            takeaways=["التحيز يدخل من التمثيل والقياس والتسميات والتاريخ والنشر.", "مقاييس العدالة متعارضة رياضيًا؛ الاختيار معياري.",
                       "قيّم حسب المجموعات ووثّق في Model card."])
