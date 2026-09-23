import numpy as np
import streamlit as st
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from components.algorithm_profile import algorithm_profile, hyperparameter_table, template_checklist
from components.callouts import intuition, mistakes, researcher_note, why
from components.code_lab import code_lab
from components.decision_boundary import boundary_chart
from components.formulas import formula
from content.references import cite
from core.page import page_footer, page_header
from core.state import at_least
from utils.datasets import toy_2d, xy
from utils.models import KNNScratch
from utils.plotting import lines, plot

page_header("knn")
algorithm_profile("knn")

formula(r"\hat y(x) = \operatorname{mode}\{y_i : i\in N_k(x)\},\qquad \hat p_c(x) = \frac1k\sum_{i\in N_k(x)}\mathbb 1[y_i=c]",
        title="k-NN prediction",
        symbols={"N_k(x)": "فهارس أقرب k نقطة تدريب إلى x", "d(x, x_i)": "دالة المسافة (Minkowski بأس p)"},
        intuition="لا تدريب حقيقي: النموذج هو البيانات نفسها. كل العمل يحدث وقت التنبؤ.",
        example="k = 5، الجيران (1, 1, 0, 1, 0) ⇒ ŷ = 1 باحتمال 3/5.")

st.markdown("## مختبر kNN: k والمسافة والأوزان")
c1, c2, c3, c4 = st.columns(4)
kind = c1.segmented_control("البيانات", ["moons", "circles", "xor"], default="moons", key="knn_kind", required=True)
k = c2.slider("k", 1, 75, 5, key="knn_k")
p = c3.segmented_control("المسافة", ["Manhattan (p=1)", "Euclidean (p=2)", "Chebyshev (p=∞)"], default="Euclidean (p=2)",
                         key="knn_p", required=True)
weights = c4.segmented_control("الأوزان", ["uniform", "distance"], default="uniform", key="knn_w", required=True)
X, y = toy_2d(kind, n=250, noise=0.3)
pmap = {"Manhattan (p=1)": ("minkowski", 1), "Euclidean (p=2)": ("minkowski", 2), "Chebyshev (p=∞)": ("chebyshev", 2)}
metric, pp = pmap[p]
model = KNeighborsClassifier(n_neighbors=k, weights=weights, metric=metric, p=pp).fit(X, y)
cv = cross_val_score(KNeighborsClassifier(n_neighbors=k, weights=weights, metric=metric, p=pp), X, y, cv=5).mean()
boundary_chart(model, X, y, title=f"k = {k}, {p}, weights = {weights} · train {model.score(X, y):.3f} · CV {cv:.3f}")
intuition("k = 1 يحفظ التدريب (دقة تدريب 100%) بحدود متعرجة؛ k كبير جدًا يقترب من التنبؤ بالأغلبية. المسافة تغيّر شكل "
          "«الجوار»: ماسة (Manhattan)، دائرة (Euclidean)، مربع (Chebyshev).")


@st.cache_data(show_spinner=False)
def _k_curve(kind: str):
    X, y = toy_2d(kind, n=250, noise=0.3)
    ks = np.arange(1, 80, 2)
    tr = [KNeighborsClassifier(k).fit(X, y).score(X, y) for k in ks]
    va = [cross_val_score(KNeighborsClassifier(k), X, y, cv=5).mean() for k in ks]
    return ks, tr, va


ks, tr, va = _k_curve(kind)
fig = lines(ks, {"train": tr, "5-fold CV": va}, title="Validation curve for k (reversed x-axis: complexity grows to the right)",
            xaxis="k", yaxis="accuracy", markers=True)
fig.update_xaxes(autorange="reversed")
plot(fig, height=320)

st.markdown("## القياس إلزامي")
Xm, ym = xy("classification")
Xm = Xm.copy()
Xm["f0"] = Xm["f0"] * 1000  # one feature in 'grams', others in 'kilograms'
raw = cross_val_score(KNeighborsClassifier(), Xm, ym, cv=5).mean()
scaled = cross_val_score(make_pipeline(StandardScaler(), KNeighborsClassifier()), Xm, ym, cv=5).mean()
c1, c2 = st.columns(2)
c1.metric("دون قياس (f0 × 1000)", f"{raw:.3f}")
c2.metric("مع StandardScaler داخل Pipeline", f"{scaled:.3f}", f"{scaled - raw:+.3f}")
why("ضع StandardScaler قبل kNN داخل Pipeline دائمًا.", "خاصية بمقياس أكبر تسيطر على المسافة وحدها؛ القياس يعطي كل خاصية صوتًا متساويًا.")

st.markdown("## لعنة الأبعاد")
st.caption("مع زيادة الأبعاد تتقارب المسافة إلى أقرب جار وإلى أبعد جار: يفقد مفهوم «القرب» معناه.")


@st.cache_data(show_spinner=False)
def _curse():
    rng = np.random.default_rng(0)
    dims = [1, 2, 5, 10, 20, 50, 100, 500]
    ratio = []
    for d in dims:
        P = rng.uniform(size=(500, d))
        q = rng.uniform(size=d)
        dist = np.linalg.norm(P - q, axis=1)
        ratio.append(dist.min() / dist.max())
    return dims, ratio


dims, ratio = _curse()
fig = lines(dims, {"nearest / farthest distance": ratio}, title="Distance concentration in high dimensions", xaxis="dimension",
            yaxis="ratio", log_x=True, markers=True)
plot(fig, height=300)

st.markdown("## من الصفر مقابل scikit-learn")
Xs, ys = toy_2d("moons", n=200, noise=0.3)


def _code(p):
    return ("# utils/models.py — brute-force kNN\n"
            "d = (np.abs(X_query[:, None, :] - X_train[None]) ** p).sum(2) ** (1/p)   # all pairwise distances\n"
            "idx = np.argsort(d, axis=1)[:, :k]                                     # k nearest\n"
            "proba = mean(y_train[idx] == c)                                         # vote share per class\n\n"
            f"scratch = KNNScratch(n_neighbors={p['k']}, p={p['p']}).fit(X, y)\n"
            f"sk = KNeighborsClassifier(n_neighbors={p['k']}, p={p['p']}).fit(X, y)\n"
            "agreement = (scratch.predict(X_test) == sk.predict(X_test)).mean()")


def _run(k, p):
    grid = np.random.default_rng(1).uniform(-1.5, 2.5, size=(400, 2))
    a = KNNScratch(n_neighbors=k, p=p).fit(Xs, ys).predict(grid)
    b = KNeighborsClassifier(n_neighbors=k, p=p).fit(Xs, ys).predict(grid)
    return f"اتفاق التنفيذين على 400 نقطة جديدة: **{(a == b).mean():.1%}** (الفروق الممكنة فقط عند تعادل المسافات أو الأصوات)."


code_lab("knn_code", "Scratch kNN vs KNeighborsClassifier", _code, _run,
         lambda: {"k": st.slider("k", 1, 25, 5, key="knnc_k"), "p": st.segmented_control("p", [1, 2], default=2, key="knnc_p",
                                                                                        required=True)})
hyperparameter_table("KNeighborsClassifier")

if at_least("advanced"):
    st.markdown("## متقدم: التعقيد وبنى البحث")
    st.markdown("- **Brute force:** O(n·p) لكل استعلام، بلا تدريب.\n- **KD-tree:** سريع في أبعاد منخفضة (p ≲ 20)، يتدهور بعدها.\n"
                "- **Ball tree:** أفضل مع مقاييس عامة وأبعاد أعلى.\n- للبيانات الضخمة: بحث تقريبي (ANN) مثل HNSW خارج scikit-learn.\n"
                "- **kNN Regression:** KNeighborsRegressor يتوسط قيم الجيران؛ نفس الاعتبارات.")
if at_least("research"):
    researcher_note(["Cover & Hart (1967): خطأ 1-NN تقاربيًا ≤ ضعف خطأ Bayes الأمثل — نتيجة لافتة لطريقة بهذه البساطة.",
                     "kNN مع k_n → ∞ وk_n/n → 0 متسق عالميًا (Stone, 1977)."])
    st.markdown(cite("cover1967", "esl"))
template_checklist({3: "صيغة التصويت", 5: "مختبر الحد", 7: "لا تدريب (Lazy)", 8: "لا معاملات: يخزن البيانات",
                    9: "جدول المعاملات الفائقة", 12: "قسم «القياس إلزامي»", 15: "قسم متقدم", 19: "Validation curve",
                    21: "KNNScratch", 22: "KNeighborsClassifier", 23: "مختبر kNN", 24: "منزلقات k/المسافة/الأوزان"})
mistakes(["نسيان القياس.", "k زوجي في تصنيف ثنائي (تعادلات).", "kNN على مئات الأبعاد دون تقليل أبعاد.",
          "الظن أن kNN لا يحتاج ضبطًا."])
page_footer("knn",
            takeaways=["kNN يخزن البيانات ويتنبأ بالجوار.", "k يتحكم في التحيز/التباين؛ المسافة تحدد شكل الجوار.",
                       "القياس إلزامي، والأبعاد العالية تُفقد القرب معناه."])
