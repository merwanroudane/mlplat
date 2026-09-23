import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from components.callouts import causal_caution, intuition, mistakes, researcher_note, warning, why
from components.cards import comparison_table
from components.formulas import formula
from content.references import cite
from core.page import page_footer, page_header
from core.registry import installed_version, missing_notice, optional
from core.state import at_least
from core.theme import PALETTE
from utils.datasets import xy
from utils.inspection import shapley_exact
from utils.plotting import bars, plot

page_header("shap")

formula(r"\phi_j = \sum_{S\subseteq F\setminus\{j\}} \frac{|S|!\,(p-|S|-1)!}{p!}\Big[v(S\cup\{j\}) - v(S)\Big],\qquad "
        r"\hat f(x) = \phi_0 + \sum_j\phi_j",
        title="Shapley values (Lundberg & Lee, 2017)",
        symbols={"v(S)": "قيمة التحالف: متوسط التنبؤ حين نعرف خصائص S فقط (والبقية من الخلفية)",
                 r"\phi_0": "القيمة الأساسية = متوسط التنبؤ على الخلفية", r"\phi_j": "مساهمة الخاصية j لهذه الملاحظة"},
        intuition="من نظرية الألعاب: وزّع «الربح» (التنبؤ − المتوسط) على الخصائص بعدالة، بمتوسط مساهمتها الحدية عبر كل ترتيبات الدخول.",
        example="التنبؤ 12، المتوسط 10: φ = (x1: +3، x2: −1.5، x3: +0.5) ⇒ 10 + 3 − 1.5 + 0.5 = 12 (خاصية الجمع).")

X, y = xy("regression")
Xs = X[["x1", "x2", "x3", "x4", "x5"]]


@st.cache_resource(show_spinner="يدرّب النموذج…")
def _models():
    Xn = Xs.to_numpy()  # numpy in, numpy out: shapley_exact passes arrays
    return RandomForestRegressor(200, min_samples_leaf=3, random_state=0, n_jobs=1).fit(Xn, y), LinearRegression().fit(Xn, y)


rf, lin = _models()

st.markdown("## Shapley بالتعريف (تعداد كل التحالفات)")
i = st.slider("اختر ملاحظة", 0, len(Xs) - 1, 7, key="shap_i")
bg_n = st.select_slider("حجم بيانات الخلفية", [10, 50, 100, 300], value=50, key="shap_bg")
bg = Xs.sample(bg_n, random_state=0).to_numpy()
x = Xs.iloc[i].to_numpy()
phi = shapley_exact(rf.predict, x, bg)
base = rf.predict(bg).mean()
pred = rf.predict(x[None])[0]
order = np.argsort(-np.abs(phi))
fig = go.Figure(go.Waterfall(orientation="h", y=[f"{Xs.columns[j]} = {x[j]:.2f}" for j in order][::-1],
                             x=list(phi[order][::-1]), base=base, connector=dict(line=dict(color="#ADB5BD")),
                             increasing=dict(marker=dict(color=PALETTE["coral"])), decreasing=dict(marker=dict(color=PALETTE["sky"]))))
fig.update_layout(title=f"Waterfall: base value {base:.2f} → prediction {pred:.2f} (sum check: {base + phi.sum():.2f})", height=380)
plot(fig)
st.caption(f"حُسبت بتعداد 2⁵ = 32 تحالفًا و{bg_n} صف خلفية (utils/inspection.shapley_exact). خاصية الجمع تتحقق بدقة الآلة.")
lin_phi = lin.coef_ * (x - bg.mean(0))
st.markdown("**تحقق للنموذج الخطي:** قيم SHAP (خلفية مستقلة) = β_j·(x_j − E[x_j]): "
            + ", ".join(f"`{c}`: {v:+.2f}" for c, v in zip(Xs.columns, lin_phi)))

st.markdown("## SHAP Lab: الحزمة الرسمية")
shap = optional("shap")
if shap is None:
    missing_notice("shap")
else:
    st.caption(f"shap {installed_version('shap')} مثبتة — TreeExplainer يحسب قيم Shapley الدقيقة للأشجار في زمن متعدد الحدود.")

    @st.cache_data(show_spinner="TreeSHAP لـ500 ملاحظة…")
    def _tree_shap():
        ex = shap.TreeExplainer(rf)
        sv = ex.shap_values(Xs.iloc[:500].to_numpy())
        return np.asarray(sv), float(np.ravel(ex.expected_value)[0])

    sv, ev = _tree_shap()
    mean_abs = np.abs(sv).mean(0)
    c1, c2 = st.columns(2)
    with c1:
        o = np.argsort(-mean_abs)
        plot(bars(Xs.columns[o], mean_abs[o], title="Global: mean |SHAP|", horizontal=True, color=PALETTE["purple"]), height=320)
    with c2:
        fig = go.Figure()
        for k, col in enumerate(Xs.columns):
            v = Xs[col].iloc[:500].to_numpy()
            fig.add_trace(go.Scatter(x=sv[:, k], y=np.full(500, k) + np.random.default_rng(k).uniform(-0.3, 0.3, 500), mode="markers",
                                     marker=dict(size=4, color=v, colorscale=[[0, "#1C7ED6"], [1, "#F76707"]], showscale=k == 0,
                                                 colorbar=dict(title="feature value")), showlegend=False))
        fig.update_layout(title="Beeswarm: each dot is one row", yaxis=dict(tickvals=list(range(5)), ticktext=list(Xs.columns)),
                          xaxis_title="SHAP value", height=320)
        plot(fig)
    st.caption(f"expected_value = {ev:.3f}. لاحظ x4: قيم SHAP تتوزع على الجانبين لأن علاقته جيبية؛ الأهمية العامة وحدها تخفي الاتجاه.")
st.code("""import shap
explainer = shap.TreeExplainer(model)                 # exact & fast for tree ensembles
sv = explainer(X_sample)                              # Explanation object
shap.plots.waterfall(sv[0]); shap.plots.beeswarm(sv)  # local and global views""", language="python")

st.markdown("## التحذيرات")
comparison_table([
    {"التحذير": "Background data", "لماذا": "φ_j تقارن بخط أساس = الخلفية؛ تغيير الخلفية يغيّر القيم", "ماذا تفعل": "اختر خلفية ممثلة وأبلغ عنها"},
    {"التحذير": "Correlated features", "لماذا": "Interventional SHAP يولّد تركيبات غير واقعية؛ Observational يوزّع الفضل بين المترابطات",
     "ماذا تفعل": "فسّر المجموعات لا الخصائص المفردة"},
    {"التحذير": "Model ≠ world", "لماذا": "SHAP يفسّر f̂ لا آلية توليد البيانات", "ماذا تفعل": "لا تستنتج تدخلات"},
    {"التحذير": "Output scale", "لماذا": "للمصنفات قد تكون على log-odds لا الاحتمال", "ماذا تفعل": "تحقق من model_output"},
])
warning("**Feature importance ≠ causal effect.** قيمة SHAP كبيرة لـ«عدد زيارات المستشفى» لا تعني أن تقليل الزيارات يحسّن الصحة.")
causal_caution("للأثر السببي انتقل إلى مسار Causal ML وDML حيث نحدد المعالجة والمربكات صراحة.")

st.markdown("## LIME: نظرة عامة")
intuition("LIME (Ribeiro et al., 2016) يلائم نموذجًا خطيًا بسيطًا محليًا حول الملاحظة بعينات مضطربة. سريع ومرن، لكن تفسيراته "
          "غير مستقرة (تعتمد على عرض النواة وعينات الاضطراب) ولا تحقق خاصية الجمع؛ SHAP أكثر اتساقًا نظريًا.")
why("استخدم SHAP للنماذج الشجرية (TreeExplainer)، واحذر KernelExplainer مع خصائص كثيرة (مكلف وتقريبي).",
    "TreeSHAP دقيق ومتعدد الحدود؛ KernelSHAP يقدّر بالمحاكاة ويتطلب خلفية صغيرة.")

if at_least("research"):
    researcher_note(["Lundberg & Lee (2017) وحّدوا طرقًا عدة تحت SHAP بخصائص الدقة المحلية والاتساق.",
                     "Janzing et al. (2020): Interventional مقابل Observational Shapley — اختيار له دلالة سببية ضمنية.",
                     "أبلغ عن: الخلفية، نوع Explainer، مقياس المخرج، وعدد العينات."])
    st.markdown(cite("lundberg2017", "ribeiro2016"))
mistakes(["تفسير SHAP كأثر سببي.", "خلفية غير ممثلة (أو صف واحد).", "تفسير خصائص مترابطة منفردة.", "KernelSHAP على مئات الخصائص."])
page_footer("shap",
            takeaways=["Shapley يوزّع (التنبؤ − الأساس) على الخصائص بعدالة وجمعية.", "النتائج تعتمد على الخلفية وترابط الخصائص.",
                       "تفسير للنموذج لا للعالم."])
