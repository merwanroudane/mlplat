import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import (GridSearchCV, GroupKFold, KFold, RepeatedKFold, StratifiedGroupKFold,
                                     StratifiedKFold, TimeSeriesSplit, cross_val_score)
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from components.animation import stepper
from components.callouts import intuition, mistakes, researcher_note, why
from components.cards import comparison_table
from components.diagrams import mermaid
from content.references import cite
from core.page import page_footer, page_header, page_link
from core.state import at_least
from core.theme import PALETTE
from utils.datasets import load_dataset, xy
from utils.plotting import bars, plot

page_header("cross_validation")

st.markdown("## الفكرة")
intuition("بدل تقسيم واحد متقلب، دوّر دور «التحقق» على k أجزاء؛ كل ملاحظة تُستخدم للتحقق مرة واحدة وللتدريب k−1 مرة. "
          "المتوسط أثبت، والانحراف المعياري يعطي فكرة عن التقلب.")

st.markdown("## CV Animator: كيف تتحرك الطيات؟")
N = 40
rng = np.random.default_rng(0)
y = (rng.random(N) < 0.25).astype(int)
groups = np.repeat(np.arange(10), 4)
rng.shuffle(groups)
scheme = st.segmented_control("المُقسِّم", ["KFold", "StratifiedKFold", "GroupKFold", "StratifiedGroupKFold", "TimeSeriesSplit",
                                             "TimeSeriesSplit (gap=2)", "RepeatedKFold"], default="KFold", key="cv_scheme", required=True)
K = st.slider("k", 2, 8, 5, key="cv_k")
splitters = {
    "KFold": KFold(K, shuffle=True, random_state=0), "StratifiedKFold": StratifiedKFold(K, shuffle=True, random_state=0),
    "GroupKFold": GroupKFold(K), "StratifiedGroupKFold": StratifiedGroupKFold(K, shuffle=True, random_state=0),
    "TimeSeriesSplit": TimeSeriesSplit(K), "TimeSeriesSplit (gap=2)": TimeSeriesSplit(K, gap=2),
    "RepeatedKFold": RepeatedKFold(n_splits=K, n_repeats=2, random_state=0),
}
splits = list(splitters[scheme].split(np.zeros((N, 1)), y, groups if "Group" in scheme else None))


def _frame(i: int) -> None:
    fig = go.Figure()
    for f, (tr, te) in enumerate(splits[: i + 1]):
        role = np.full(N, "unused", dtype=object)
        role[tr], role[te] = "train", "validation"
        colors = [PALETTE["sky"] if r == "train" else PALETTE["coral"] if r == "validation" else "#E9ECEF" for r in role]
        symbols = ["square" if r == "train" else "diamond" if r == "validation" else "x-thin" for r in role]
        fig.add_trace(go.Scatter(x=np.arange(N), y=[f] * N, mode="markers+text", showlegend=False,
                                 text=[str(groups[j]) if "Group" in scheme else "" for j in range(N)], textposition="middle center",
                                 textfont=dict(size=8, color="white"),
                                 marker=dict(symbol=symbols, size=15, color=colors, line=dict(width=[2 if y[j] else 0 for j in range(N)],
                                                                                              color="#212529"))))
    fig.update_layout(height=140 + 42 * len(splits), xaxis=dict(title="observation index" + (" (label = group id)" if "Group" in scheme else "")),
                      yaxis=dict(title="split", autorange="reversed", dtick=1, range=[-0.5, len(splits) - 0.5]), margin=dict(t=20))
    plot(fig)
    tr, te = splits[i]
    txt = f"Split {i + 1}: train = {len(tr)}, validation = {len(te)}, positives in validation = {y[te].mean():.0%}"
    if "Group" in scheme:
        txt += f" · groups in validation {sorted(set(groups[te]))} — none of them in training"
    st.caption(txt + " · ■ train  ◆ validation  × unused · black outline = positive class")


stepper(f"cv_{scheme}_{K}", len(splits), _frame, labels=[f"split {i + 1}" for i in range(len(splits))])

comparison_table([
    {"المُقسِّم": "KFold", "متى": "ملاحظات مستقلة متجانسة (انحدار)", "انتبه": "shuffle=True إن كانت البيانات مرتبة"},
    {"المُقسِّم": "StratifiedKFold", "متى": "تصنيف، خاصة مع عدم التوازن", "انتبه": "لا يحل مشكلة المجموعات"},
    {"المُقسِّم": "RepeatedKFold / RepeatedStratifiedKFold", "متى": "تقدير أثبت مع بيانات قليلة", "انتبه": "الطيات غير مستقلة"},
    {"المُقسِّم": "GroupKFold", "متى": "كيانات متكررة؛ التعميم على كيانات جديدة", "انتبه": "أحجام طيات غير متساوية"},
    {"المُقسِّم": "StratifiedGroupKFold", "متى": "مجموعات + عدم توازن (متاح في الإصدار الحالي)", "انتبه": "الطبقية تقريبية"},
    {"المُقسِّم": "TimeSeriesSplit (+gap)", "متى": "التنبؤ بالمستقبل", "انتبه": "gap يمنع التسرب القريب؛ max_train_size للنافذة المنزلقة"},
    {"المُقسِّم": "LeaveOneOut", "متى": "n صغير جدًا (مفهوميًا)", "انتبه": "تباين عالٍ ومكلف؛ نادرًا الأفضل"},
])

st.markdown("## مختبر: المُقسِّم يغيّر الرقم الذي تبلغ عنه")
dfp = load_dataset("panel")


@st.cache_data(show_spinner="يقارن المقسِّمات على بيانات Panel…")
def _panel_cv():
    d = dfp.copy()
    d["lag_y"] = d.groupby("id")["y"].shift(1)
    d = d.dropna()
    X = d[["x1", "x2", "t", "lag_y", "id"]].to_numpy()   # id included on purpose: a model can memorise units
    target = d["y"].to_numpy()
    from sklearn.ensemble import RandomForestRegressor
    m = RandomForestRegressor(n_estimators=100, min_samples_leaf=3, random_state=0, n_jobs=1)
    rows = []
    for name, cv, g in (("KFold (random rows)", KFold(5, shuffle=True, random_state=0), None),
                        ("GroupKFold (new units)", GroupKFold(5), d["id"].to_numpy()),
                        ("TimeSeriesSplit (future periods)", None, None)):
        if cv is None:
            order = np.argsort(d["t"].to_numpy(), kind="stable")
            s = cross_val_score(m, X[order], target[order], cv=TimeSeriesSplit(4), scoring="r2")
        else:
            s = cross_val_score(m, X, target, cv=cv, groups=g, scoring="r2")
        rows.append({"scheme": name, "R²": s.mean(), "± SD": s.std()})
    return pd.DataFrame(rows)


res = _panel_cv()
plot(bars(res["scheme"], res["R²"], errors=res["± SD"], title="Same model, same data, three CV schemes", color=PALETTE["purple"]),
     height=320)
why("اختر المُقسِّم من سؤال التعميم، لا من الرقم الأعلى.",
    "على من سيُطبق النموذج؟ صفوف جديدة من الكيانات نفسها ⇒ KFold. كيانات جديدة ⇒ GroupKFold. فترات مستقبلية ⇒ TimeSeriesSplit.")

st.markdown("## Nested CV: تقييم الإجراء كاملًا مع الضبط")
mermaid("""
flowchart LR
  D[(data)] --> O1[outer fold 1 test] & O2[outer fold 2 test] & O3[...]
  O1 --> I1[inner CV on outer-train<br/>choose hyperparameters] --> F1[refit on outer-train] --> S1[score on outer test]
""")
Xc, yc = xy("classification")


@st.cache_data(show_spinner="Nested CV: 5 طيات خارجية × 3 داخلية × 16 تركيبة…")
def _nested():
    grid = {"svc__C": [0.1, 1, 10, 100], "svc__gamma": [0.001, 0.01, 0.1, 1]}
    inner = StratifiedKFold(3, shuffle=True, random_state=1)
    outer = StratifiedKFold(5, shuffle=True, random_state=2)
    gs = GridSearchCV(make_pipeline(StandardScaler(), SVC()), grid, cv=inner)
    non_nested = gs.fit(Xc, yc).best_score_
    nested = cross_val_score(gs, Xc, yc, cv=outer)
    return non_nested, nested


if st.button("شغّل Nested vs non-nested", key="cv_nested", icon=":material/play_arrow:"):
    st.session_state["cv_nested_on"] = True
if st.session_state.get("cv_nested_on"):
    non_nested, nested = _nested()
    c1, c2 = st.columns(2)
    c1.metric("Non-nested (best_score_)", f"{non_nested:.4f}", "متفائل", delta_color="off")
    c2.metric("Nested CV", f"{nested.mean():.4f}", f"± {nested.std():.4f}", delta_color="off")
    st.caption("best_score_ هو أقصى 16 تقديرًا متقلبًا ⇒ متحيز للأعلى. Nested يقيّم «GridSearch + SVC» كإجراء واحد على بيانات لم "
               "يرها الضبط. الفرق يكبر مع شبكات أكبر وبيانات أقل (Varma & Simon, 2006).")

st.markdown("## Cross-Validation مقابل Cross-Fitting في DML")
comparison_table([
    {"": "الهدف", "Cross-validation": "تقدير أداء التعميم / اختيار نموذج", "Cross-fitting (DML)": "تقدير معلمة سببية θ بلا تحيز الإفراط"},
    {"": "ما يُحسب على الطية المحجوزة", "Cross-validation": "مقياس (AUC، MSE)", "Cross-fitting (DML)": "تنبؤات الإزعاج ℓ̂(X)، m̂(X) خارج الطية"},
    {"": "ماذا بعد", "Cross-validation": "متوسط الدرجات", "Cross-fitting (DML)": "تُجمع التنبؤات لكل الملاحظات ثم تُحل الدرجة المتعامدة لـθ"},
    {"": "المخرج", "Cross-validation": "رقم أداء", "Cross-fitting (DML)": "θ̂ + خطأ معياري + فترة ثقة"},
    {"": "لماذا الطيات؟", "Cross-validation": "تقييم على بيانات غير مرئية", "Cross-fitting (DML)": "كسر الاعتماد بين أخطاء الإزعاج والملاحظة"},
    {"": "يمكن الجمع؟", "Cross-validation": "CV داخلية لضبط متعلمي الإزعاج داخل كل طية", "Cross-fitting (DML)": "الطيات الخارجية للتقدير"},
])
page_link("cross_fitting", "انتقل إلى Cross-Fitting في مسار DML", ":material/sync_alt:")

if at_least("advanced"):
    st.markdown("## متقدم: اختيار k")
    st.markdown("- k صغير (2–3): تحيز متشائم (تدريب على بيانات أقل).\n- k كبير/LOO: تحيز أقل وتباين أعلى وتكلفة أكبر.\n"
                "- 5 أو 10 توازن عملي (ISLP §5.1.4). مع بيانات قليلة: RepeatedStratifiedKFold.\n"
                "- LOO مع kNN أو نماذج خطية يمكن حسابه بكفاءة (RidgeCV يستخدم GCV/LOO تحليليًا).")
if at_least("research"):
    researcher_note(["الانحراف المعياري عبر الطيات يستهين بعدم اليقين لأن مجموعات التدريب متداخلة؛ Nadeau & Bengio (2003) "
                     "اقترحا تصحيحًا للاختبارات المزدوجة.",
                     "أبلغ عن: المُقسِّم، k، التكرارات، البذور، ومستوى التجميع (Group)؛ هذه جزء من تعريف النتيجة."])
    st.markdown(cite("varma2006", "nadeau2003", "islp"))
mistakes(["KFold عشوائي لبيانات زمنية أو مجمّعة.", "الإبلاغ عن best_score_ كأداء نهائي.", "shuffle=False لبيانات مرتبة حسب الهدف.",
          "الخلط بين Cross-validation وCross-fitting."])
page_footer("cross_validation",
            takeaways=["المُقسِّم يجب أن يحاكي سؤال التعميم.", "Nested CV يقيّم الإجراء كاملًا مع الضبط.",
                       "Cross-fitting يستخدم الطيات لتقدير θ لا لتقييم الأداء."])
