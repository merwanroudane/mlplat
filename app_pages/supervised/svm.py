import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from components.algorithm_profile import algorithm_profile, hyperparameter_table, template_checklist
from components.animation import stepper
from components.callouts import intuition, mistakes, researcher_note, warning, why
from components.cards import comparison_table
from components.formulas import formula
from components.parameter_lab import parameter_playground
from content.references import cite
from core.page import page_footer, page_header
from core.state import at_least
from core.theme import PALETTE
from utils.datasets import toy_2d
from utils.plotting import grid_for, heatmap, plot, scatter_classes

page_header("svm")
algorithm_profile("svm")

st.markdown("## الهامش الأقصى")
formula(r"\min_{w,b}\ \tfrac12\|w\|^2\quad\text{s.t.}\quad y_i(w^\top x_i+b)\ge 1\ \ \forall i",
        title="Hard-margin SVM (separable data)",
        symbols={r"w^\top x + b = 0": "المستوى الفاصل", r"2/\|w\|": "عرض الهامش", "y_i ∈ {−1, +1}": "التسمية"},
        intuition="من بين كل الخطوط التي تفصل الفئتين، اختر الذي يترك أعرض «طريق» فارغ. تصغير ‖w‖ = توسيع الهامش.")
formula(r"\min_{w,b}\ \tfrac12\|w\|^2 + C\sum_i \xi_i,\quad y_i(w^\top x_i+b)\ge 1-\xi_i,\ \xi_i\ge 0"
        r"\ \ \Longleftrightarrow\ \ \min\ \tfrac12\|w\|^2 + C\sum_i\max\big(0, 1-y_if(x_i)\big)",
        title="Soft-margin SVM = hinge loss + L2 penalty",
        symbols={r"\xi_i": "متغير الارتخاء: كم تنتهك النقطة الهامش", "C": "ثمن الانتهاك (مقلوب قوة التنظيم)"},
        intuition="C كبير: لا تسامح ⇒ هامش ضيق يلاحق كل نقطة. C صغير: تسامح ⇒ هامش واسع وحدود أبسط.",
        example="نقطة على الجانب الصحيح وخارج الهامش: ξ = 0، hinge = 0، لا تؤثر في الحل إطلاقًا.")

st.markdown("## Animation: الهامش مع تغيّر C")
st.caption("SVM خطي على بيانات شبه قابلة للفصل. تابع كيف يضيق الهامش ويتناقص عدد المتجهات الداعمة مع زيادة C.")
Xl, yl = toy_2d("linear", n=120, noise=0.5, seed=3)
Xl = StandardScaler().fit_transform(Xl)
Cs = [0.005, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 100]


def _margin_frame(i: int) -> None:
    C = Cs[i]
    m = SVC(kernel="linear", C=C).fit(Xl, yl)
    w, b = m.coef_[0], m.intercept_[0]
    xs = np.linspace(Xl[:, 0].min() - 0.5, Xl[:, 0].max() + 0.5, 50)
    fig = scatter_classes(Xl, yl, highlight=np.isin(np.arange(len(yl)), m.support_))
    for off, dash, name in ((0, "solid", "boundary"), (1, "dash", "margin +1"), (-1, "dash", "margin −1")):
        fig.add_trace(go.Scatter(x=xs, y=(-(w[0] * xs + b) + off) / w[1], mode="lines", name=name,
                                 line=dict(color="#212529" if off == 0 else PALETTE["purple"], dash=dash, width=2.5 if off == 0 else 1.8)))
    fig.update_layout(title=f"C = {C} · margin width 2/‖w‖ = {2 / np.linalg.norm(w):.2f} · support vectors = {len(m.support_)}",
                      yaxis=dict(range=[Xl[:, 1].min() - 0.5, Xl[:, 1].max() + 0.5]), height=430)
    plot(fig)


stepper("svm_margin", len(Cs), _margin_frame, labels=[f"C = {c}" for c in Cs])
intuition("النقاط المحاطة بدائرة هي المتجهات الداعمة: وحدها تحدد الحد. احذف أي نقطة أخرى ولن يتغير شيء — لهذا SVM "
          "«متفرق» في النقاط.")

st.markdown("## النوى وKernel trick")
formula(r"f(x) = \sum_{i\in SV}\alpha_i y_i\,K(x_i, x) + b,\qquad K_{\text{RBF}}(x,x') = e^{-\gamma\|x-x'\|^2}",
        title="Kernel SVM (dual form)",
        symbols={r"\alpha_i": "معاملات ثنائية (غير صفرية للمتجهات الداعمة فقط)", "K": "النواة = جداء داخلي في فضاء ضمني"},
        intuition="المسألة الثنائية لا تحتاج x إلا عبر جداءات xᵢᵀxⱼ؛ نستبدلها بـK(xᵢ, xⱼ) فنعمل ضمنيًا في فضاء عالي "
                  "(أو لانهائي) الأبعاد دون حسابه.",
        example="Polynomial: K = (γ xᵀx' + coef0)^degree؛ RBF ≈ فضاء لانهائي؛ sigmoid: tanh(γ xᵀx' + coef0).")
comparison_table([
    {"النواة": "linear", "الحد": "مستوى", "المعاملات": "C", "متى": "p كبير (نصوص)، n كبير ⇒ LinearSVC"},
    {"النواة": "poly", "الحد": "كثير حدود", "المعاملات": "C, degree, gamma, coef0", "متى": "تفاعلات بدرجة معروفة"},
    {"النواة": "rbf", "الحد": "مرن محلي", "المعاملات": "C, gamma", "متى": "الخيار الافتراضي للبيانات الكثيفة المتوسطة"},
    {"النواة": "sigmoid", "الحد": "يشبه شبكة بطبقة", "المعاملات": "C, gamma, coef0", "متى": "نادر؛ ليس نواة موجبة دائمًا"},
])

st.markdown("## مختبر SVM: النواة وC وgamma")
c1, c2, c3, c4 = st.columns(4)
kind = c1.segmented_control("البيانات", ["moons", "circles", "xor"], default="circles", key="svm_kind", required=True)
kernel = c2.selectbox("kernel", ["rbf", "poly", "linear", "sigmoid"], key="svm_kernel")
C = c3.select_slider("C", [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0], value=1.0, key="svm_C")
gamma = c4.select_slider("gamma", ["scale", 0.01, 0.1, 1.0, 10.0, 100.0], value="scale", key="svm_gamma")
X, y = toy_2d(kind, n=240, noise=0.3)
model = make_pipeline(StandardScaler(), SVC(kernel=kernel, C=C, gamma=gamma, degree=3)).fit(X, y)
svc = model[-1]
xx, yy, grid = grid_for(X)
Z = model.decision_function(grid).reshape(xx.shape)
fig = go.Figure(go.Contour(x=xx[0], y=yy[:, 0], z=np.clip(Z, -3, 3), colorscale=[[0, "#D0EBFF"], [0.5, "#FCFCFF"], [1, "#FFE8CC"]],
                           contours=dict(start=-1, end=1, size=1, coloring="fill"), showscale=False, hoverinfo="skip"))
fig.add_trace(go.Contour(x=xx[0], y=yy[:, 0], z=Z, contours=dict(start=-1, end=1, size=1, coloring="none", showlabels=True),
                         line=dict(color="#212529", width=2), showscale=False, hoverinfo="skip"))
sv_mask = np.zeros(len(y), bool)
sv_mask[svc.support_] = True
scatter_classes(X, y, fig, highlight=sv_mask)
fig.update_layout(title=f"{kernel} kernel · C={C}, gamma={gamma} · support vectors {len(svc.support_)} / {len(y)} · "
                        f"train acc {model.score(X, y):.3f}", height=460)
plot(fig)
if kernel == "rbf" and gamma in (10.0, 100.0):
    warning("gamma كبير: كل نقطة تؤثر في محيط ضيق جدًا ⇒ «جزر» حول النقاط = Overfitting.")

st.markdown("## ضبط (C, gamma) معًا")


@st.cache_data(show_spinner="Grid search 6×6 × 5 folds…")
def _grid(kind: str):
    X, y = toy_2d(kind, n=240, noise=0.3)
    Cs_ = np.logspace(-2, 3, 6)
    gs_ = np.logspace(-2, 3, 6)
    gs = GridSearchCV(make_pipeline(StandardScaler(), SVC()), {"svc__C": Cs_, "svc__gamma": gs_},
                      cv=StratifiedKFold(5, shuffle=True, random_state=0)).fit(X, y)
    Z = gs.cv_results_["mean_test_score"].reshape(len(Cs_), len(gs_))
    return Cs_, gs_, Z, gs.best_params_


Cs_, gs_, Zg, best = _grid(kind)
fig = heatmap(Zg, [f"{g:g}" for g in gs_], [f"{c:g}" for c in Cs_], title=f"5-fold CV accuracy · best {best}",
              colorscale=[[0, "#FFF4E6"], [1, "#1C7ED6"]], text_fmt=".2f")
fig.update_layout(xaxis_title="gamma", yaxis_title="C", height=380)
plot(fig)
why("ابحث في (C, gamma) على شبكة لوغاريتمية مشتركة.", "المعاملان متفاعلان بقوة: C كبير مع gamma صغير قد يعادل C صغيرًا مع "
    "gamma كبير. القطر المائل في الخريطة الحرارية يعكس هذا التفاعل.")

st.markdown("## الاحتمالات: تغيير في scikit-learn 1.9")
warning("`SVC(probability=True)` **مهجور منذ 1.9** ويُزال في 1.11. البديل الموصى به رسميًا: "
        "`CalibratedClassifierCV(SVC(), ensemble=False)`. تحقق مباشر من رسالة التحذير في 1.9.1.",
        title=":material/update: تحديث API")
st.code("""from sklearn.calibration import CalibratedClassifierCV
svm_proba = CalibratedClassifierCV(make_pipeline(StandardScaler(), SVC(C=1.0, gamma="scale")),
                                   method="sigmoid", ensemble=False, cv=5).fit(X, y)
svm_proba.predict_proba(X_new)          # Platt-scaled probabilities""", language="python")
cal = CalibratedClassifierCV(make_pipeline(StandardScaler(), SVC()), method="sigmoid", ensemble=False, cv=5).fit(X, y)
st.caption(f"مثال: P(y=1) لأول 5 نقاط = {np.round(cal.predict_proba(X[:5])[:, 1], 3).tolist()}")

st.markdown("## ساحة المعاملات")
parameter_playground("svc", key="svm_pg", dataset="moons")
hyperparameter_table("SVC")

if at_least("advanced"):
    st.markdown("## متقدم: الثنائية والتعقيد")
    st.latex(r"\max_\alpha \sum_i\alpha_i - \tfrac12\sum_{i,j}\alpha_i\alpha_j y_iy_jK(x_i,x_j)\quad\text{s.t. } 0\le\alpha_i\le C,\ \sum_i\alpha_iy_i=0")
    st.markdown("- libsvm (SVC) يحل الثنائية بـSMO: بين O(n²) وO(n³) زمنًا وO(n²) ذاكرة لمصفوفة النواة (cache_size).\n"
                "- لـn كبير: LinearSVC (liblinear) أو SGDClassifier(loss='hinge') أو تقريب النواة (Nystroem, RBFSampler).\n"
                "- **SVR:** نفس الفكرة للانحدار بخسارة ε-insensitive.")
    lin = LinearSVC(C=1.0, max_iter=10000).fit(Xl, yl)
    st.caption(f"LinearSVC على بيانات الـAnimation: w = {np.round(lin.coef_[0], 3).tolist()} (يستخدم squared hinge افتراضيًا).")
if at_least("research"):
    researcher_note(["نظرية الهامش تعطي حدود تعميم لا تعتمد على البعد (Vapnik): سر أداء SVM مع p ≫ n في النصوص.",
                     "SVM ليس نموذجًا احتماليًا؛ Platt scaling يضيف طبقة معايرة يجب أن تُدرَّب على بيانات منفصلة.",
                     "Representer theorem: الحل الأمثل في RKHS هو تركيبة خطية من K(xᵢ, ·) — أساس كل طرق النوى."])
    st.markdown(cite("cortes1995", "esl", "uml"))
template_checklist({3: "صيغة الهامش الصلب واللين", 5: "Animation الهامش", 6: "Hinge + L2", 7: "SMO (متقدم)",
                    8: "support_vectors_, dual_coef_", 9: "جدول المعاملات الفائقة", 15: "قسم متقدم",
                    19: "خريطة (C, gamma) الحرارية", 22: "SVC / LinearSVC", 23: "مختبر النوى", 24: "ساحة المعاملات"})
mistakes(["SVM دون قياس.", "gamma كبير جدًا ⇒ حفظ.", "استخدام probability=True في كود جديد.", "SVC على مئات الآلاف من الصفوف."])
page_footer("svm",
            takeaways=["SVM = أعرض هامش؛ الحد تحدده المتجهات الداعمة.", "Soft margin = Hinge + L2؛ C يوازن بينهما.",
                       "النوى تعطي حدودًا غير خطية؛ اضبط C وgamma معًا."])
