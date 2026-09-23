import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz, export_text

from components.algorithm_profile import algorithm_profile, hyperparameter_table, template_checklist
from components.callouts import intuition, mistakes, researcher_note, why
from components.cards import comparison_table
from components.decision_boundary import boundary_chart
from components.diagrams import graphviz
from components.formulas import formula
from content.references import cite
from core.page import page_footer, page_header
from core.state import at_least
from core.theme import PALETTE
from utils.datasets import toy_2d, xy
from utils.plotting import lines, plot

page_header("decision_trees")
algorithm_profile("decision_tree")

st.markdown("## المفردات")
comparison_table([
    {"المصطلح": "Root", "المعنى": "العقدة الأولى (كل البيانات)"},
    {"المصطلح": "Node / split", "المعنى": "سؤال: هل x_j ≤ t؟"},
    {"المصطلح": "Leaf", "المعنى": "عقدة نهائية تعطي التنبؤ (أغلبية/متوسط)"},
    {"المصطلح": "Impurity", "المعنى": "درجة خلط الفئات في العقدة"},
    {"المصطلح": "Information gain", "المعنى": "انخفاض الشوائب الموزون بعد التقسيم"},
    {"المصطلح": "Recursive partitioning", "المعنى": "كرر البحث داخل كل ابن حتى شرط توقف"},
])
formula(r"G = 1-\sum_k p_k^2,\qquad H = -\sum_k p_k\log_2 p_k,\qquad \Delta = I(\text{parent}) - \tfrac{n_L}{n}I(L) - \tfrac{n_R}{n}I(R)",
        title="Gini, entropy and information gain",
        symbols={"p_k": "نسبة الفئة k في العقدة", "I": "مقياس الشوائب (Gini أو Entropy)", "n_L, n_R": "أحجام الابنين"},
        intuition="العقدة النقية (فئة واحدة) شوائبها 0. نختار التقسيم الذي يجعل الابنين أنقى ما يمكن في المتوسط الموزون.",
        example="عقدة 50/50: Gini = 0.5، Entropy = 1. عقدة 90/10: Gini = 0.18، Entropy = 0.47.")

st.markdown("## مختبر التقسيم (Tree Split Lab)")
st.caption("خاصية واحدة وفئتان. حرّك العتبة t وشاهد شوائب الابنين والكسب؛ الخوارزمية تجرّب كل العتبات وتختار الأعلى كسبًا.")
rng = np.random.default_rng(2)
x = np.r_[rng.normal(2, 1, 60), rng.normal(5, 1.2, 60)]
yb = np.r_[np.zeros(60, int), np.ones(60, int)]
crit = st.segmented_control("المعيار", ["gini", "entropy"], default="gini", key="dt_crit", required=True)


def imp(labels):
    if len(labels) == 0:
        return 0.0
    p = np.bincount(labels, minlength=2) / len(labels)
    return float(1 - (p ** 2).sum()) if crit == "gini" else float(-(p[p > 0] * np.log2(p[p > 0])).sum())


def gain(t):
    L, R = yb[x <= t], yb[x > t]
    return imp(yb) - len(L) / len(yb) * imp(L) - len(R) / len(yb) * imp(R), imp(L), imp(R), len(L), len(R)


ts = np.sort(x)[:-1] + np.diff(np.sort(x)) / 2
gains = np.array([gain(t)[0] for t in ts])
best_t = float(ts[np.argmax(gains)])
t = st.slider("العتبة t", float(x.min()), float(x.max()), 3.0, 0.05, key="dt_t")
g, iL, iR, nL, nR = gain(t)
c1, c2 = st.columns([1.6, 1])
with c1:
    fig = go.Figure()
    for k, color, sym in ((0, PALETTE["sky"], "circle"), (1, PALETTE["coral"], "diamond")):
        fig.add_trace(go.Scatter(x=x[yb == k], y=rng.uniform(-0.3, 0.3, (yb == k).sum()) + 1, mode="markers", name=f"class {k}",
                                 marker=dict(color=color, symbol=sym, size=8)))
    fig.add_trace(go.Scatter(x=ts, y=gains / gains.max() * 0.6 - 0.2, mode="lines", name="gain (scaled)",
                             line=dict(color=PALETTE["purple"], width=2.5), yaxis="y"))
    fig.add_vline(x=t, line=dict(color="#212529", width=2), annotation_text=f"t = {t:.2f}")
    fig.add_vline(x=best_t, line=dict(color=PALETTE["teal"], dash="dash"), annotation_text="best", annotation_position="bottom right")
    fig.update_layout(height=340, yaxis=dict(visible=False), title="Candidate thresholds and their information gain")
    plot(fig)
with c2:
    st.metric(f"{crit} (parent)", f"{imp(yb):.3f}")
    st.metric(f"left (n={nL})", f"{iL:.3f}")
    st.metric(f"right (n={nR})", f"{iR:.3f}")
    st.metric("Information gain", f"{g:.4f}", f"best = {gains.max():.4f} at t = {best_t:.2f}", delta_color="off")

st.markdown("## الشجرة الكاملة وحدود القرار")
c1, c2, c3 = st.columns(3)
kind = c1.segmented_control("البيانات", ["moons", "circles", "xor"], default="moons", key="dt_kind", required=True)
depth = c2.slider("max_depth (0 = بلا حد)", 0, 15, 3, key="dt_depth")
leaf = c3.slider("min_samples_leaf", 1, 40, 1, key="dt_leaf")
X, y = toy_2d(kind, n=300, noise=0.3)
tree = DecisionTreeClassifier(max_depth=depth or None, min_samples_leaf=leaf, criterion=crit, random_state=0).fit(X, y)
cv = cross_val_score(DecisionTreeClassifier(max_depth=depth or None, min_samples_leaf=leaf, criterion=crit, random_state=0), X, y, cv=5).mean()
boundary_chart(tree, X, y, title=f"depth {tree.get_depth()}, {tree.get_n_leaves()} leaves · train {tree.score(X, y):.3f} · CV {cv:.3f}",
               show_proba=False)
intuition("الحدود دائمًا محاذية للمحاور (مستطيلات)؛ لذلك تحتاج الأشجار تقسيمات كثيرة لتقريب حد مائل بسيط.")
if tree.get_n_leaves() <= 16:
    dot = export_graphviz(tree, feature_names=["x1", "x2"], class_names=["0", "1"], filled=True, rounded=True,
                          impurity=True, proportion=False, precision=2)
    graphviz(dot)
else:
    with st.expander("الشجرة كنص (كبيرة للرسم)"):
        st.code(export_text(tree, feature_names=["x1", "x2"], max_depth=6), language="text")

st.markdown("## مختبر التقليم (Pruning Lab): Minimal cost-complexity")
formula(r"R_\alpha(T) = R(T) + \alpha\,|T|", title="Cost-complexity criterion",
        symbols={"R(T)": "خطأ/شوائب أوراق الشجرة T", "|T|": "عدد الأوراق", r"\alpha": "ccp_alpha: ثمن كل ورقة إضافية"},
        intuition="نمّي شجرة كاملة ثم احذف الفروع التي لا يبرر تحسينها ثمنها. كل α يعطي شجرة جزئية؛ نختار α بـCV.")
Xd, yd = xy("classification")
Xa, Xb, ya, yb2 = train_test_split(Xd, yd, test_size=0.3, random_state=0, stratify=yd)


@st.cache_data(show_spinner="يحسب مسار التقليم…")
def _pruning():
    path = DecisionTreeClassifier(random_state=0).cost_complexity_pruning_path(Xa, ya)
    alphas = path.ccp_alphas[:-1][::max(1, len(path.ccp_alphas) // 40)]
    tr, va, leaves = [], [], []
    for a in alphas:
        m = DecisionTreeClassifier(random_state=0, ccp_alpha=a).fit(Xa, ya)
        tr.append(m.score(Xa, ya))
        va.append(m.score(Xb, yb2))
        leaves.append(m.get_n_leaves())
    return alphas, tr, va, leaves


alphas, tr, va, leaves = _pruning()
fig = lines(alphas, {"train accuracy": tr, "held-out accuracy": va}, title="Accuracy vs ccp_alpha", xaxis="ccp_alpha",
            yaxis="accuracy", markers=True)
best_a = alphas[int(np.argmax(va))]
fig.add_vline(x=best_a, line=dict(color=PALETTE["teal"], dash="dash"), annotation_text=f"best α ≈ {best_a:.4f}")
plot(fig, height=340)
st.caption(f"عدد الأوراق ينخفض من {leaves[0]} (α = 0) إلى {leaves[-1]}. للاختيار الصارم استخدم CV بدل مجموعة واحدة.")
why("قيّد نمو الشجرة (max_depth، min_samples_leaf) أو قلّمها (ccp_alpha).", "الشجرة الكاملة تحفظ التدريب: تباين هائل.")

st.markdown("## أشجار الانحدار")
formula(r"\text{MSE}(node) = \frac{1}{n_m}\sum_{i\in m}(y_i-\bar y_m)^2", title="Regression criterion ('squared_error')",
        intuition="الورقة تتنبأ بمتوسط y فيها؛ التقسيم يقلل التباين داخل الأبناء.")
comparison_table([
    {"criterion (Regressor)": "squared_error (افتراضي)", "الورقة تتنبأ بـ": "المتوسط", "ملاحظة": "friedman_mse مهجور منذ 1.9 (مطابق)"},
    {"criterion (Regressor)": "absolute_error", "الورقة تتنبأ بـ": "الوسيط", "ملاحظة": "O(n log n) منذ 1.8؛ يدعم NaN منذ 1.9"},
    {"criterion (Regressor)": "poisson", "الورقة تتنبأ بـ": "المتوسط (عدّ)", "ملاحظة": "لبيانات العدّ غير السالبة"},
])
xs = np.sort(np.random.default_rng(0).uniform(0, 1, 120))
ys = np.sin(2 * np.pi * xs) + np.random.default_rng(1).normal(scale=0.3, size=120)
d_reg = st.slider("max_depth (انحدار)", 1, 10, 3, key="dt_rdepth")
reg = DecisionTreeRegressor(max_depth=d_reg).fit(xs[:, None], ys)
g = np.linspace(0, 1, 500)
fig = go.Figure(go.Scatter(x=xs, y=ys, mode="markers", marker=dict(color=PALETTE["muted"], opacity=0.6), name="data"))
fig.add_trace(go.Scatter(x=g, y=reg.predict(g[:, None]), name=f"tree depth {d_reg}", line=dict(color=PALETTE["purple"], width=3, shape="hv")))
fig.update_layout(height=320, title="A regression tree is a piecewise-constant function")
plot(fig)

hyperparameter_table("DecisionTreeClassifier")

if at_least("advanced"):
    st.markdown("## متقدم: التعقيد والقيم المفقودة والرتابة")
    st.markdown("- **التعقيد:** فرز كل خاصية O(n log n)، والبحث في كل العتبات؛ العمق النموذجي O(log n).\n"
                "- **القيم المفقودة:** مع splitter='best' تُرسل NaN إلى الابن الذي يعظّم الكسب؛ وفي التنبؤ إلى الابن الأكثر عينات "
                "إن لم تُرَ NaN في التدريب.\n"
                "- **monotonic_cst:** يفرض رتابة (مثلًا: السعر المتوقع لا ينقص مع المساحة) — مفيد للتفسير والعدالة.\n"
                "- **الفئات:** DecisionTreeClassifier لا يدعم الفئات أصليًا؛ استخدم OrdinalEncoder أو HistGradientBoosting.")
if at_least("research"):
    researcher_note(["عدم الاستقرار: تغيير صغير في البيانات قد يغير التقسيم الأول وبالتالي الشجرة كلها — لذلك Bagging.",
                     "Impurity-based importance منحاز للخصائص عالية الكاردينالية والمستمرة (Strobl et al., 2007).",
                     "Causal trees (Athey & Imbens, 2016) تعدّل معيار التقسيم لتقدير آثار معالجة غير متجانسة بصدق (Honesty)."])
    st.markdown(cite("quinlan1986", "esl", "athey2016"))
template_checklist({3: "صيغ Gini/Entropy/Gain", 5: "حدود محاذية للمحاور", 6: "الشوائب الموزونة", 7: "Recursive partitioning",
                    8: "بنية الشجرة (Graphviz)", 9: "جدول المعاملات الفائقة", 14: "قسم متقدم", 15: "قسم متقدم",
                    19: "مسار التقليم", 20: "مسار الجذر إلى الورقة", 23: "Tree Split Lab", 24: "منزلقات العمق/الأوراق"})
mistakes(["شجرة بلا قيود.", "تفسير Impurity importance كأهمية حقيقية.", "توقع استقراء خارج مدى التدريب."])
page_footer("decision_trees",
            takeaways=["الشجرة = تقسيمات ثنائية جشعة تقلل الشوائب.", "العمق وحجم الورقة وccp_alpha تتحكم في السعة.",
                       "قابلة للتفسير لكنها عالية التباين."])
