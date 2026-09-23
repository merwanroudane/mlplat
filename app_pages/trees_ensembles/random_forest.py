import numpy as np
import streamlit as st
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

from components.algorithm_profile import algorithm_profile, hyperparameter_table, template_checklist
from components.animation import stepper
from components.callouts import intuition, mistakes, researcher_note, why
from components.cards import comparison_table
from components.decision_boundary import boundary_chart
from components.parameter_lab import parameter_playground
from config import MAX_TREES_LAB
from content.references import cite
from core.page import page_footer, page_header
from core.state import at_least
from core.theme import PALETTE
from utils.datasets import toy_2d, xy
from utils.plotting import bars, decision_boundary, lines, plot

page_header("random_forest")
algorithm_profile("random_forest")

st.markdown("## مصدرا العشوائية")
comparison_table([
    {"المصدر": "Bootstrap sample", "المعامل": "bootstrap=True, max_samples", "الأثر": "كل شجرة ترى ~63.2% من الصفوف الفريدة"},
    {"المصدر": "Random feature subset عند كل تقسيم", "المعامل": "max_features ('sqrt' للتصنيف)", "الأثر": "يقلل ارتباط الأشجار ρ"},
])
intuition("لو استخدمت كل الأشجار كل الخصائص، لاختارت كلها الخاصية الأقوى في الجذر فتصبح متشابهة (ρ مرتفع). حرمان كل تقسيم "
          "من معظم الخصائص يجبر الأشجار على التنوع.")

st.markdown("## Animation: من البيانات إلى التجميع")
X, y = toy_2d("moons", n=200, noise=0.35, seed=4)
rng = np.random.default_rng(0)
boots = [rng.integers(0, len(y), len(y)) for _ in range(6)]
trees = [DecisionTreeClassifier(max_features=1, random_state=i).fit(X[b], y[b]) for i, b in enumerate(boots)]
steps = ["البيانات الأصلية"] + [f"شجرة {i + 1} على عينة Bootstrap {i + 1}" for i in range(6)] + ["التصويت: الغابة"]


class _Vote:
    def __init__(self, ts):
        self.ts = ts

    def predict_proba(self, Z):
        p = np.mean([t.predict_proba(Z)[:, 1] for t in self.ts], axis=0)
        return np.c_[1 - p, p]

    def predict(self, Z):
        return (self.predict_proba(Z)[:, 1] > 0.5).astype(int)


def _rf_frame(i: int) -> None:
    if i == 0:
        from utils.plotting import scatter_classes
        plot(scatter_classes(X, y).update_layout(title="Original data (n = 200)", height=400))
        st.caption("كل شجرة ستحصل على عينة Bootstrap: سحب 200 صف مع الإرجاع.")
    elif i <= 6:
        b = boots[i - 1]
        counts = np.bincount(b, minlength=len(y))
        oob = counts == 0
        fig = decision_boundary(trees[i - 1], X, y, title=f"Tree {i}: {len(np.unique(b))} unique rows, {oob.sum()} out-of-bag",
                                show_proba=False, highlight=counts > 1)
        plot(fig, height=400)
        st.caption("النقاط المحاطة بدائرة سُحبت أكثر من مرة. الشجرة عميقة ومتعرجة (تباين عالٍ).")
    else:
        plot(decision_boundary(_Vote(trees), X, y, title="Average of 6 trees: smoother boundary"), height=400)
        st.caption("التصويت على 6 أشجار فقط ينعّم الحدود بوضوح؛ الغابة الحقيقية تستخدم 100+.")


stepper("rf_anim", len(steps), _rf_frame, labels=steps)

st.markdown("## مختبر Random Forest")
Xd, yd = xy("classification")
Xa, Xb, ya, yb = train_test_split(Xd, yd, test_size=0.3, random_state=0, stratify=yd)
c1, c2, c3, c4 = st.columns(4)
n_est = c1.slider("n_estimators", 1, MAX_TREES_LAB, 100, key="rf_n")
mf = c2.selectbox("max_features", ["sqrt", "log2", 0.5, 1.0], key="rf_mf", format_func=str)
depth = c3.slider("max_depth (0 = None)", 0, 20, 0, key="rf_depth")
leaf = c4.slider("min_samples_leaf", 1, 30, 1, key="rf_leaf")


@st.cache_data(show_spinner="يدرّب الغابة…", max_entries=32)
def _rf_lab(n_est, mf, depth, leaf):
    rf = RandomForestClassifier(n_estimators=n_est, max_features=mf, max_depth=depth or None, min_samples_leaf=leaf,
                                oob_score=n_est >= 20, random_state=0, n_jobs=1).fit(Xa, ya)
    return (rf.score(Xa, ya), rf.oob_score_ if n_est >= 20 else None, rf.score(Xb, yb),
            float(np.mean([t.get_n_leaves() for t in rf.estimators_])))


tr_acc, oob_acc, ho_acc, mean_leaves = _rf_lab(n_est, mf, depth, leaf)
with st.container(horizontal=True):
    st.metric("Train accuracy", f"{tr_acc:.3f}", border=True)
    st.metric("OOB accuracy", f"{oob_acc:.3f}" if oob_acc is not None else "n_estimators < 20", border=True)
    st.metric("Held-out accuracy", f"{ho_acc:.3f}", border=True)
    st.metric("Mean leaves / tree", f"{mean_leaves:.0f}", border=True)


@st.cache_data(show_spinner="يرسم منحنى عدد الأشجار…")
def _n_curve(mf, depth, leaf):
    rf = RandomForestClassifier(n_estimators=MAX_TREES_LAB, max_features=mf, max_depth=depth or None, min_samples_leaf=leaf,
                                random_state=0, n_jobs=1).fit(Xa, ya)
    probs = np.cumsum([t.predict_proba(Xb.to_numpy())[:, 1] for t in rf.estimators_], axis=0)
    ns = np.arange(1, MAX_TREES_LAB + 1)
    acc = [((probs[k - 1] / k > 0.5).astype(int) == yb.to_numpy()).mean() for k in ns]
    return ns, acc


ns, acc = _n_curve(mf, depth, leaf)
plot(lines(ns, {"held-out accuracy": acc}, title="More trees never overfit — the curve stabilises", xaxis="number of trees",
           yaxis="accuracy"), height=300)
why("n_estimators ليس معامل Overfitting في RF.", "زيادة الأشجار تخفض التباين حتى الاستقرار؛ القيد الوحيد هو الزمن والذاكرة. "
    "المعاملات التي تحدد التحيز/التباين هي max_features والعمق وحجم الورقة.")

st.markdown("## الأهمية: Impurity (MDI) مقابل Permutation")


@st.cache_data(show_spinner="يحسب الأهميات…")
def _importances():
    Xn = Xa.copy()
    Xn["random_id"] = np.arange(len(Xn))  # unique high-cardinality noise feature
    Xbn = Xb.copy()
    Xbn["random_id"] = np.arange(len(Xbn)) + 10_000
    rfi = RandomForestClassifier(n_estimators=150, random_state=0, n_jobs=1).fit(Xn, ya)
    pi = permutation_importance(rfi, Xbn, yb, n_repeats=5, random_state=0)
    return list(Xn.columns), rfi.feature_importances_, pi.importances_mean, pi.importances_std


cols, mdi, pim, pis = _importances()
cols = np.array(cols)
c1, c2 = st.columns(2)
with c1:
    o = np.argsort(-mdi)
    plot(bars(cols[o], mdi[o], title="Impurity importance (training)", horizontal=True, color=PALETTE["coral"]), height=380)
with c2:
    o = np.argsort(-pim)
    plot(bars(cols[o], pim[o], errors=pis[o], title="Permutation importance (held-out)", horizontal=True,
              color=PALETTE["teal"]), height=380)
st.markdown("لاحظ `random_id`: خاصية ضجيج فريدة لكل صف تحصل على **أهمية MDI ملحوظة** لأن الأشجار العميقة تستطيع التقسيم عليها "
            "لحفظ التدريب، بينما أهميتها بالتبديل على بيانات منفصلة ≈ 0.")

st.markdown("## Extra Trees: عشوائية أكثر")
Xm, ym = toy_2d("moons", n=300, noise=0.35)


@st.cache_resource(show_spinner="يدرّب RF وExtra Trees…")
def _rf_vs_et():
    m1 = RandomForestClassifier(200, random_state=0, n_jobs=1).fit(Xm, ym)
    m2 = ExtraTreesClassifier(200, random_state=0, n_jobs=1).fit(Xm, ym)
    return m1, m2, cross_val_score(m1, Xm, ym, cv=5).mean(), cross_val_score(m2, Xm, ym, cv=5).mean()


m1, m2, cv1, cv2 = _rf_vs_et()
c1, c2 = st.columns(2)
with c1:
    boundary_chart(m1, Xm, ym, title=f"Random Forest · CV {cv1:.3f}", height=360)
with c2:
    boundary_chart(m2, Xm, ym, title=f"Extra Trees · CV {cv2:.3f}", height=360)
st.caption("Extra Trees يختار العتبة عشوائيًا (لا أفضلها) ولا يستخدم Bootstrap افتراضيًا ⇒ حدود أنعم وتدريب أسرع وتباين أقل، "
           "مع تحيز أعلى قليلًا.")

st.markdown("## ساحة المعاملات")
parameter_playground("rf", key="rf_pg", dataset="moons")
hyperparameter_table("RandomForestClassifier")

if at_least("advanced"):
    st.markdown("## متقدم: OOB وتعقيد الغابة")
    st.markdown("- احتمال ألا يُسحب صف في Bootstrap = (1 − 1/n)ⁿ → e⁻¹ ≈ 36.8% ⇒ كل صف خارج الحقيبة لنحو ثلث الأشجار؛ OOB "
                "يتنبأ له بها فقط — تقدير شبه مجاني قريب من CV.\n- **التعقيد:** O(B · n log n · max_features · depth).\n"
                "- **القيم المفقودة:** مدعومة منذ 1.4 في الغابات؛ **monotonic_cst** مدعوم أيضًا.\n"
                "- **sample_weight:** منذ 1.9 يُستخدم للسحب مع الإرجاع (Bootstrap مرجّح).")
if at_least("research"):
    researcher_note(["Breiman (2001): خطأ الغابة ≤ ρ̄(1 − s²)/s² حيث s قوة الأشجار وρ̄ ارتباطها — التصميم كله موازنة بينهما.",
                     "Wager & Athey (2018): غابات «صادقة» (Honest) تعطي تقديرات CATE طبيعية تقاربيًا مع فترات ثقة — أساس "
                     "CausalForestDML.",
                     "الغابات متعلمات إزعاج شائعة في DML لأنها مرنة ولا تحتاج قياسًا."])
    st.markdown(cite("breiman2001", "geurts2006", "wager2018"))
template_checklist({5: "Animation الغابة", 7: "Bootstrap + تقسيمات عشوائية", 8: "estimators_", 9: "جدول المعاملات الفائقة",
                    19: "OOB ومنحنى عدد الأشجار", 20: "MDI مقابل Permutation", 22: "RandomForestClassifier / ExtraTrees",
                    23: "مختبر RF", 24: "ساحة المعاملات"})
mistakes(["ضبط n_estimators كأنه يسبب Overfitting.", "الاعتماد على MDI للخصائص عالية الكاردينالية.",
          "الظن أن OOB يغني عن Test نهائي."])
page_footer("random_forest",
            takeaways=["RF = Bagging + خصائص عشوائية عند كل تقسيم.", "الأشجار الإضافية لا تسبب Overfitting.",
                       "OOB تقدير مجاني؛ Permutation importance أصدق من MDI."])
