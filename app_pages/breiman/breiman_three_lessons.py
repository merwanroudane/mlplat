import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from components.callouts import definition, intuition, mistakes, researcher_note, warning, why
from components.cards import comparison_table
from components.formulas import formula
from content.references import cite
from core.page import page_footer, page_header, page_link
from core.state import at_least
from core.theme import PALETTE
from utils.breiman import random_feature_curve, rashomon_data, rashomon_subsets, tree_instability, two_cultures_data
from utils.datasets import toy_2d
from utils.plotting import bars, lines, plot

page_header("breiman_three_lessons")

st.markdown(
    "في الجزء الأخير من «The Two Cultures» يلخّص Breiman ما تعلّمه من النمذجة الخوارزمية في ثلاثة دروس، سمّاها بأسماء "
    "مستعارة من خارج الإحصاء: **Rashomon** (فيلم Kurosawa عام 1950 الذي يروي فيه أربعة شهود الحادثة نفسها بأربع روايات "
    "متناقضة ومقنعة)، **Occam** (مبدأ البساطة)، و**Bellman** (صاحب مصطلح «لعنة الأبعاد»). لكل درس هنا مختبر يعيد إنتاجه."
)
comparison_table([
    {"الدرس": "Rashomon", "الفكرة": "نماذج كثيرة مختلفة بدقة متقاربة جدًا", "النتيجة العملية": "لا تفسّر نموذجًا واحدًا كأنه الحقيقة"},
    {"الدرس": "Occam", "الفكرة": "الدقة والبساطة (قابلية التفسير) تتعارضان غالبًا", "النتيجة العملية": "استخرج المعلومات من النموذج الدقيق"},
    {"الدرس": "Bellman", "الفكرة": "الأبعاد الكثيرة قد تكون نعمة لا لعنة", "النتيجة العملية": "لا تقلّص الخصائص تلقائيًا؛ دع التحقق يحكم"},
])

# ======================================================================= Rashomon
st.markdown("## 1. تأثير راشومون · The Rashomon Effect")
definition("Rashomon effect",
           "وجود **عدد كبير من النماذج المختلفة** — بمتغيرات أو بنى مختلفة — تحقق **الدقة نفسها تقريبًا**. كل منها يروي "
           "«قصة» مختلفة عن أي المتغيرات مهمة.")
st.markdown("مثال Breiman: انحدار خطي بـ30 متغيرًا؛ نبحث عن أفضل 5 متغيرات. قد نجد عدة مجموعات مختلفة من 5 متغيرات "
            "تختلف مجاميع مربعات بواقيها بأقل من 1%. أي مجموعة «تفسّر» y؟ هنا نسخة مصغّرة يمكنك التحكم فيها.")
st.markdown("### مختبر راشومون (1): اختيار المجموعات الجزئية")
c1, c2 = st.columns(2)
k = c1.segmented_control("عدد المتغيرات في كل نموذج k", [3, 4, 5], default=4, key="rs_k") or 4
tol = c2.select_slider("التسامح في RSS (نسبة مئوية فوق الأفضل)", [0.5, 1.0, 2.0, 3.0, 5.0], value=1.0, key="rs_tol")
X_r, y_r = rashomon_data()


@st.cache_data(show_spinner="تجربة كل المجموعات الجزئية…", max_entries=12)
def _rashomon(k: int, tol: float) -> pd.DataFrame:
    return rashomon_subsets(X_r, y_r, k=k, tol=tol / 100)


rs = _rashomon(k, tol)
st.caption(f"البيانات: n = 200، 12 متغيرًا بارتباط متبادل ρ = 0.85؛ الحقيقة: y = 0.5·(x1 + … + x6) + ضجيج. "
           f"من أصل كل تركيبات {k} متغيرات، هذه التي يقع RSS لها ضمن {tol}% من الأفضل:")
st.dataframe(rs.round(3), hide_index=True, width="stretch")
freq = pd.Series([v for row in rs["variables"] for v in row.split(", ")]).value_counts()
c1, c2 = st.columns([1.2, 1])
with c1:
    plot(bars(freq.index.tolist(), freq.to_numpy(dtype=float), title="How often each variable appears in the Rashomon set",
              text_fmt=".0f"), height=320)
with c2:
    st.metric("نماذج «متساوية» تقريبًا", len(rs), border=True)
    st.metric("متغيرات مختلفة تظهر فيها", len(freq), border=True)
    st.metric("مدى R²", f"{rs['R2'].min():.3f} – {rs['R2'].max():.3f}", border=True)
intuition("حين تترابط المتغيرات بقوة، يستطيع كل منها أن ينوب عن الآخر. البيانات لا تملك معلومات كافية لتقول «x2 وليس x5» — "
          "فأي نموذج يختار واحدًا فقط يروي قصة من قصص راشومون.")

st.markdown("### مختبر راشومون (2): عدم استقرار الشجرة الواحدة")
st.caption("نحوّل y إلى فئتين (فوق/تحت الوسيط) ونعيد تدريب شجرة بعمق 3 على 40 عينة Bootstrap. ما المتغير في الجذر؟")


@st.cache_data(show_spinner=False)
def _instab():
    yc = (y_r > y_r.median()).astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X_r, yc, test_size=0.3, random_state=0, stratify=yc)
    ti = tree_instability(X_tr, y_tr, X_te, y_te, n_boot=40)
    rf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=1).fit(X_tr, y_tr)
    return ti, rf.score(X_te, y_te)


ti, rf_acc = _instab()
roots = ti["root variable"].value_counts()
c1, c2 = st.columns([1.2, 1])
with c1:
    plot(bars(roots.index.tolist(), roots.to_numpy(dtype=float), title="Root split variable across 40 bootstrap trees",
              text_fmt=".0f", color=PALETTE["purple"]), height=320)
with c2:
    st.metric("متغيرات مختلفة في الجذر", len(roots), border=True)
    st.metric("دقة الأشجار المفردة (مدى)", f"{ti['test accuracy'].min():.2f} – {ti['test accuracy'].max():.2f}", border=True)
    st.metric("دقة Random Forest (300 شجرة)", f"{rf_acc:.2f}", border=True)
st.markdown("**علاج Breiman:** بدل اختيار قصة واحدة، **اجمع القصص**. الغابة تمتص عدم الاستقرار (دقة أعلى وأثبت)، ثم نسأل الغابة "
            "نفسها عن أهمية المتغيرات. هذا هو الطريق من Rashomon إلى Bagging وRandom Forests.")

# ======================================================================= Occam
st.markdown("## 2. معضلة أوكام · Occam's Dilemma")
definition("Occam's dilemma (Breiman)",
           "في التنبؤ، كثيرًا ما تتعارض **الدقة** مع **البساطة وقابلية التفسير**: النماذج الأدق (الغابات، الشبكات) أصعب "
           "قراءة، والنماذج السهلة القراءة (شجرة صغيرة، انحدار خطي) أقل دقة حين تكون الآلية معقدة.")
st.markdown("موقف Breiman: لا تتخلَّ عن الدقة لأجل البساطة؛ **اختر النموذج الدقيق ثم استخرج منه المعلومات** (أهمية المتغيرات، "
            "الاعتماد الجزئي). مختبر: نماذج بتعقيد متزايد على بيانات الثقافتين (λ = 1.5) بـ5-fold CV.")


def _size(m) -> int:
    """Number of coefficients (linear) or total number of leaves (trees and ensembles)."""
    if hasattr(m, "tree_"):
        return int(m.get_n_leaves())
    if hasattr(m, "estimators_"):
        return int(sum(t.get_n_leaves() for t in m.estimators_))
    if hasattr(m, "n_iter_"):  # HistGradientBoosting: exact count from the (private) predictors, else the upper bound
        try:
            return int(sum(p[0].get_n_leaf_nodes() for p in m._predictors))
        except (AttributeError, IndexError, TypeError):
            return int(m.n_iter_ * m.max_leaf_nodes)
    return int(m[-1].coef_.size + 1)


@st.cache_data(show_spinner="5-fold CV لستة نماذج…")
def _occam():
    X, y = two_cultures_data(n=1500, nonlinearity=1.5)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    models = {
        "Logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "Tree depth 2": DecisionTreeClassifier(max_depth=2, random_state=0),
        "Tree depth 4": DecisionTreeClassifier(max_depth=4, random_state=0),
        "Tree depth 8": DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, random_state=0),
        "Random Forest (300 trees)": RandomForestClassifier(300, min_samples_leaf=5, random_state=0, n_jobs=1),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=0),
    }
    rows = []
    for name, m in models.items():
        s = cross_val_score(m, X, y, cv=cv, scoring="roc_auc")
        rows.append({"model": name, "size (leaves or coefficients)": _size(m.fit(X, y)), "CV AUC": s.mean(), "SD": s.std()})
    small = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y)
    return pd.DataFrame(rows), export_text(small, feature_names=list(X.columns))


occ, tree_txt = _occam()
fig = go.Figure(go.Scatter(x=occ["size (leaves or coefficients)"], y=occ["CV AUC"], mode="markers+text", text=occ["model"],
                           textposition="top center", marker=dict(size=13, color=PALETTE["sky"]),
                           error_y=dict(type="data", array=occ["SD"])))
fig.update_layout(title="Accuracy vs size: Occam's dilemma", xaxis_type="log", xaxis_title="model size (log scale)",
                  yaxis_title="5-fold CV ROC-AUC", height=420)
plot(fig)
c1, c2 = st.columns(2)
with c1:
    st.markdown("**أبسط نموذج قابل للقراءة (شجرة بعمق 2):**")
    st.code(tree_txt, language="text")
with c2:
    st.dataframe(occ.round(3), hide_index=True, width="stretch")
warning("النقاش الحديث أعقد من «الدقة أو التفسير»: **Rudin (2019)** تجادل بأنه في البيانات الجدولية ذات الخصائص المعنوية "
        "كثيرًا ما يوجد نموذج قابل للتفسير بدقة مماثلة، وأن شرح صندوق أسود في القرارات عالية المخاطر أخطر من استخدام نموذج "
        "مفهوم. و**Semenova, Rudin & Parr (2022)** يربطون ذلك براشومون: حين تكون مجموعة النماذج الجيدة كبيرة، يُرجَّح أن "
        "تحتوي نموذجًا بسيطًا. معضلة أوكام إذن **ليست قانونًا**؛ هي سؤال تجيب عنه بياناتك.",
        title="تحديث بعد Breiman")

# ======================================================================= Bellman
st.markdown("## 3. بيلمان: الأبعاد نعمة؟ · Bellman")
st.markdown("«لعنة الأبعاد» (Bellman) تعني أن البيانات تتناثر في الفضاءات عالية الأبعاد فتصعب التقديرات المحلية. تقليديًا "
            "نقلّص المتغيرات قبل النمذجة. Breiman رأى الوجه الآخر: **إضافة خصائص** (تحويلات، تفاعلات، ميزات عشوائية) قد "
            "تحمل معلومات تجعل المسألة أسهل — كما تفعل SVM بالنواة، والغابات حين تختار من خصائص كثيرة.")
formula(r"z(x) = \sqrt{2/D}\,\big[\cos(w_1^\top x + b_1),\dots,\cos(w_D^\top x + b_D)\big],\quad w_j\sim\mathcal N(0, 2\gamma I)",
        title="Random Fourier features (RBFSampler)",
        symbols={"D": "عدد الخصائص المضافة", r"\gamma": "معامل نواة RBF"},
        intuition="نرفع نقطتين في المستوى إلى D بُعد؛ في الفضاء الجديد يصبح الفصل الخطي ممكنًا.")
st.markdown("### مختبر بيلمان (1): مصنّف خطي + خصائص عشوائية على دوائر متداخلة")
gamma = st.segmented_control("γ", [0.5, 1.0, 2.0], default=1.0, key="bell_gamma") or 1.0


@st.cache_data(show_spinner=False, max_entries=6)
def _bellman(gamma: float) -> pd.DataFrame:
    X, y = toy_2d("circles", n=600, noise=0.15, seed=0)
    return random_feature_curve(X[:400], y[:400], X[400:], y[400:], gamma=gamma)


bc = _bellman(gamma)
plot(lines(bc["label"], {"train accuracy": bc["train accuracy"], "test accuracy": bc["test accuracy"]},
           title="Linear classifier accuracy as dimensions are added", xaxis="features", yaxis="accuracy", markers=True),
     height=360)
st.caption("مع خاصيتين فقط لا يستطيع أي حد خطي فصل دائرة داخل دائرة (دقة قرب الصدفة). بإضافة عشرات الخصائص يصبح "
           "الفصل الخطي شبه تام على بيانات الاختبار أيضًا.")

st.markdown("### مختبر بيلمان (2): لكن الأبعاد **العشوائية** ليست كلها نعمة")
st.caption("نضيف خصائص ضجيج نقي (لا علاقة لها بـy) إلى بيانات الثقافتين. من يتأذى؟")


@st.cache_data(show_spinner="إضافة خصائص ضجيج…")
def _noise_dims() -> pd.DataFrame:
    X, y = two_cultures_data(n=1000, nonlinearity=1.5)
    rng = np.random.default_rng(0)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    rows = []
    for d in (0, 10, 50, 200):
        Xa = np.c_[X.to_numpy(), rng.normal(size=(len(y), d))] if d else X.to_numpy()
        knn = cross_val_score(make_pipeline(StandardScaler(), KNeighborsClassifier(25)), Xa, y, cv=cv, scoring="roc_auc").mean()
        rf = cross_val_score(RandomForestClassifier(200, min_samples_leaf=5, random_state=0, n_jobs=1), Xa, y, cv=cv,
                             scoring="roc_auc").mean()
        rows.append({"noise features": d, "kNN (k=25) AUC": knn, "Random Forest AUC": rf})
    return pd.DataFrame(rows)


nd = _noise_dims()
plot(lines(nd["noise features"].astype(str), {"kNN (k=25)": nd["kNN (k=25) AUC"], "Random Forest": nd["Random Forest AUC"]},
           title="Adding pure-noise dimensions", xaxis="number of noise features added", yaxis="5-fold CV ROC-AUC",
           markers=True), height=340)
st.markdown("الخلاصة المتوازنة: الأبعاد التي **تحمل معلومات** (تحويلات، تفاعلات) نعمة؛ الأبعاد **العشوائية** تضر الطرق المعتمدة "
            "على المسافات (kNN) أكثر بكثير من الغابات التي تختار أفضل تقسيم من بين الخصائص. لعنة Bellman ونعمة Breiman "
            "وجهان لعملة واحدة.")

st.markdown("## ما الذي بقي وما الذي تغيّر؟")
comparison_table([
    {"الدرس": "Rashomon", "ما بقي": "تعدد النماذج الجيدة حقيقة متكررة", "تطور حديث":
     "Rashomon set، الاعتماد على المتغير عبر فئة نماذج كاملة (Fisher, Rudin & Dominici, 2019)، Predictive multiplicity"},
    {"الدرس": "Occam", "ما بقي": "النماذج المرنة تتفوق حين تكون الآلية معقدة",
     "تطور حديث": "نماذج قابلة للتفسير تنافسية على الجداول (Rudin, 2019)؛ SHAP وPDP لاستخراج المعلومات"},
    {"الدرس": "Bellman", "ما بقي": "الخصائص الغنية تفيد", "تطور حديث": "التعلم العميق يتعلم التمثيلات بدل تصميمها يدويًا"},
])
c1, c2 = st.columns(2)
with c1:
    page_link("random_forest")
with c2:
    page_link("model_inspection")
if at_least("advanced"):
    st.markdown("## متقدم: تعريف مجموعة راشومون")
    formula(r"\mathcal R(\varepsilon) = \{\, f \in \mathcal F : \hat L(f) \le \hat L(f^\star) + \varepsilon \,\}",
            symbols={r"\mathcal F": "فئة النماذج", r"\hat L": "الخسارة التجريبية", "f^\\star": "أفضل نموذج في الفئة",
                     r"\varepsilon": "التسامح"},
            intuition="المختبر الأول يحسب هذه المجموعة حرفيًا لفئة «انحدار بـk متغيرات» مع ε نسبية.")
if at_least("research"):
    researcher_note([
        "عند التقرير عن أهمية متغير، أبلغ عن مداها عبر مجموعة راشومون (Model class reliance) لا عن نموذج واحد.",
        "Semenova et al. (2022): نسبة حجم مجموعة راشومون إلى فئة النماذج مؤشر على وجود نموذج بسيط دقيق.",
        "اختبار الاستقرار (إعادة التدريب على Bootstrap) تشخيص رخيص يجب أن يسبق أي تفسير لشجرة أو لاختيار متغيرات.",
    ])
why("قبل أن تقول «المتغير x هو الأهم»، أعد التدريب على عينات أخرى وجرّب نماذج أخرى متقاربة الدقة.",
    "إن تغيّر الجواب، فالبيانات لا تدعم قصة واحدة؛ هذا هو درس راشومون.")
st.markdown("### المراجع")
st.markdown(cite("breiman2001two", "fisher2019", "rudin2019", "semenova2022", "breiman2001"))
mistakes(["تفسير المتغيرات المختارة في نموذج واحد كأنها «السبب».",
          "التضحية بدقة كبيرة لأجل نموذج بسيط دون قياس الفرق.",
          "حذف الخصائص تلقائيًا خوفًا من الأبعاد دون تحقق.",
          "إضافة مئات الخصائص العشوائية لنموذج يعتمد على المسافة."])
page_footer("breiman_three_lessons",
            takeaways=["Rashomon: نماذج كثيرة متقاربة الدقة بقصص مختلفة.",
                       "Occam: الدقة والبساطة قد تتعارضان — لكن تحقق، فأحيانًا لا تتعارضان.",
                       "Bellman: الخصائص الغنية بالمعلومات نعمة، والضجيج لعنة خصوصًا للطرق المعتمدة على المسافة."])
