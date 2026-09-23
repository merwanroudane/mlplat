import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from components.callouts import intuition, mistakes, researcher_note, why
from components.cards import comparison_table
from components.diagrams import graphviz
from components.formulas import formula
from content.references import cite
from core.page import page_footer, page_header, page_link
from core.state import at_least, log_experiment
from core.theme import PALETTE
from utils.breiman import instability, two_cultures_data
from utils.plotting import bars, plot

page_header("breiman_legacy")

st.markdown("لم يكتفِ Breiman بالنقد الفلسفي؛ بنى معظم الأدوات التي جعلت «الثقافة الخوارزمية» ممكنة داخل الإحصاء. "
            "هذه الصفحة خريطة لإرثه، ولكل عنصر صفحته التفصيلية في المنصة.")

st.markdown("## الخط الزمني")
comparison_table([
    {"السنة": "1984", "الإسهام": "CART (مع Friedman وOlshen وStone)", "المشكلة": "نموذج مرن قابل للقراءة لأي نوع مخرجات",
     "اليوم في scikit-learn": "DecisionTreeClassifier/Regressor؛ ccp_alpha = Cost-complexity pruning"},
    {"السنة": "1985", "الإسهام": "ACE (مع Friedman)", "المشكلة": "إيجاد تحويلات مثلى للمتغيرات والاستجابة",
     "اليوم في scikit-learn": "سلف نماذج GAM والتحويلات المتعلَّمة"},
    {"السنة": "1995", "الإسهام": "Nonnegative garrote", "المشكلة": "اختيار المتغيرات بانكماش مستمر بدل Subset selection",
     "اليوم في scikit-learn": "سلف مباشر للـLasso (Tibshirani, 1996)"},
    {"السنة": "1996", "الإسهام": "Bagging", "المشكلة": "عدم استقرار الإجراءات مثل الأشجار",
     "اليوم في scikit-learn": "BaggingClassifier/Regressor"},
    {"السنة": "1996", "الإسهام": "Stacked regressions", "المشكلة": "دمج نماذج مختلفة بأوزان متعلَّمة (غير سالبة)",
     "اليوم في scikit-learn": "StackingClassifier/Regressor"},
    {"السنة": "1998", "الإسهام": "Arcing classifiers", "المشكلة": "فهم لماذا ينجح Boosting (إعادة الوزن التكيفية)",
     "اليوم في scikit-learn": "AdaBoost؛ مهّد للنظرة الإحصائية للتعزيز"},
    {"السنة": "2001", "الإسهام": "Random Forests", "المشكلة": "خفض تباين الأشجار بعشوائية مزدوجة",
     "اليوم في scikit-learn": "RandomForestClassifier، oob_score، permutation_importance"},
    {"السنة": "2001", "الإسهام": "The Two Cultures", "المشكلة": "فلسفة: التنبؤ كمعيار تحقق", "اليوم في scikit-learn": "ثقافة CV كلها"},
    {"السنة": "2004", "الإسهام": "Balanced Random Forest (مع Chen وLiaw)", "المشكلة": "الغابات مع بيانات غير متوازنة",
     "اليوم في scikit-learn": "imblearn.ensemble.BalancedRandomForestClassifier"},
])
st.markdown("## شجرة النسب: من CART إلى اليوم")
graphviz("""
digraph G {
  rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#E7F5FF", color="#1971C2", fontname="Helvetica"];
  CART [label="CART (1984)"]; BAG [label="Bagging (1996)"]; RF [label="Random Forests (2001)"];
  ARC [label="Arcing (1998)"]; GB [label="Gradient boosting (Friedman 2001)"]; STACK [label="Stacking (1996)"];
  GAR [label="Nonnegative garrote (1995)"]; LASSO [label="Lasso (1996)"]; BRF [label="Balanced RF (2004)"];
  CF [label="Causal / generalized forests"]; DML [label="Double ML nuisance learners"]; XGB [label="XGBoost / LightGBM / CatBoost"];
  CART -> BAG -> RF; RF -> BRF; RF -> CF; RF -> DML; CART -> ARC -> GB -> XGB; GB -> DML; GAR -> LASSO -> DML; STACK -> DML;
}
""")
c1, c2, c3 = st.columns(3)
with c1:
    page_link("decision_trees")
with c2:
    page_link("ensemble_learning")
with c3:
    page_link("random_forest")

st.markdown("## الفكرة المركزية: الإجراءات غير المستقرة")
st.markdown("في ورقة Bagging (1996) لاحظ Breiman أن التجميع بالـBootstrap يحسّن الدقة كثيرًا **حين يكون الإجراء غير مستقر** "
            "— أي حين يغيّر تغيير صغير في بيانات التدريب النموذجَ كثيرًا (الأشجار، اختيار المتغيرات) — وقد لا يفيد أو "
            "يضر قليلًا مع الإجراءات المستقرة (مثل kNN بـk كبير أو الانحدار الخطي).")
formula(r"\hat f_{\text{bag}}(x) = \frac{1}{B}\sum_{b=1}^{B} \hat f^{*b}(x),\qquad "
        r"\operatorname{Var}\big(\hat f_{\text{bag}}\big) = \rho\,\sigma^2 + \frac{1-\rho}{B}\,\sigma^2",
        symbols={r"\hat f^{*b}": "نموذج مدرَّب على عينة Bootstrap رقم b", r"\sigma^2": "تباين النموذج المفرد",
                 r"\rho": "الارتباط بين نماذج العينات المختلفة"},
        intuition="التجميع يقتل الحد الثاني فقط. إن كان النموذج مستقرًا فتباينه صغير أصلًا ولا يبقى ما نكسبه؛ وRandom Forests "
                  "تخفض ρ نفسه بالخصائص العشوائية.")
st.markdown("### مختبر عدم الاستقرار والتجميع · Instability & Bagging Lab")
st.caption("لكل إجراء: (1) عدم الاستقرار = احتمال أن يختلف تنبؤ نموذجين دُرّبا على عينتي Bootstrap مختلفتين عند نقطة اختبار؛ "
           "(2) خطأ الاختبار للنموذج المفرد و(3) لـBagging بـ50 نموذجًا.")


@st.cache_data(show_spinner="تدريب 3 إجراءات × (25 + 50) نموذجًا…")
def _bagging_lab() -> pd.DataFrame:
    X, y = two_cultures_data(n=1500, nonlinearity=1.5)
    Xv, yv = X.to_numpy(), y.to_numpy()
    X_tr, X_te, y_tr, y_te = Xv[:900], Xv[900:], yv[:900], yv[900:]
    learners = {"Decision tree (unpruned)": DecisionTreeClassifier(random_state=0),
                "kNN (k=15)": make_pipeline(StandardScaler(), KNeighborsClassifier(15)),
                "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))}
    rows = []
    for name, est in learners.items():
        single = 1 - est.fit(X_tr, y_tr).score(X_te, y_te)
        bag = 1 - BaggingClassifier(est, n_estimators=50, random_state=0, n_jobs=1).fit(X_tr, y_tr).score(X_te, y_te)
        rows.append({"procedure": name, "instability": instability(est, X_tr, y_tr, X_te, n_models=25),
                     "single error": single, "bagged error": bag, "gain": single - bag})
    return pd.DataFrame(rows)


if st.button("شغّل المختبر", key="bag_run", type="primary", icon=":material/play_arrow:"):
    st.session_state["bag_done"] = True
if st.session_state.get("bag_done"):
    bl = _bagging_lab()
    st.dataframe(bl.round(3), hide_index=True, width="stretch")
    plot(bars(bl["procedure"], bl["gain"], title="Error reduction from bagging (single − bagged)",
              color=[PALETTE["coral"], PALETTE["sky"], PALETTE["teal"]]), height=320)
    top = bl.sort_values("instability", ascending=False).iloc[0]
    st.markdown(f"الأكثر عدم استقرار: **{top['procedure']}** (عدم استقرار = {top['instability']:.2f})، وهو الأكثر استفادة من "
                "التجميع. الانحدار اللوجستي مستقر فيبقى خطؤه كما هو تقريبًا — ولا يستطيع Bagging إصلاح تحيّزه (نموذج خطي "
                "لآلية غير خطية).")
    if st.button("سجّل التجربة", key="bag_log", icon=":material/bookmark_add:"):
        log_experiment("Instability & Bagging Lab", "tree / kNN / logistic ± bagging", {"B": 50},
                       {r["procedure"]: round(r["gain"], 4) for _, r in bl.iterrows()}, seed=0, dataset="two_cultures",
                       split="900/600")
        st.toast("سُجّلت.", icon=":material/check:")
intuition("Bagging يقلّل التباين لا التحيّز. لذلك فهو دواء للأشجار العميقة (تحيز منخفض، تباين عالٍ) وليس للنماذج البسيطة.")

st.markdown("## هدايا Random Forests المجانية: OOB والأهمية بالتبديل")
st.markdown("كل شجرة لا ترى نحو 36.8% من الصفوف (خارج عينة Bootstrap). تنبؤ الغابة على كل صف باستخدام الأشجار التي لم تره فقط "
            "يعطي تقدير **Out-of-bag** للخطأ دون بيانات تحقق منفصلة. وبتبديل عمود واحد عشوائيًا وقياس تدهور الأداء نحصل على "
            "**أهمية بالتبديل** — فكرة Breiman لاستخراج «المعلومات» من صندوق أسود.")
formula(r"P(\text{row } i \notin \text{bootstrap}) = \left(1-\tfrac1n\right)^n \xrightarrow[n\to\infty]{} e^{-1}\approx 0.368")


@st.cache_data(show_spinner="OOB مقابل CV والأهمية بالتبديل…")
def _oob_and_importance():
    X, y = two_cultures_data(n=1500, nonlinearity=1.5)
    rf = RandomForestClassifier(300, min_samples_leaf=5, oob_score=True, random_state=0, n_jobs=1).fit(X, y)
    cv = cross_val_score(RandomForestClassifier(300, min_samples_leaf=5, random_state=0, n_jobs=1), X, y,
                         cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4, random_state=0, stratify=y)
    rf2 = RandomForestClassifier(300, min_samples_leaf=5, random_state=0, n_jobs=1).fit(X_tr, y_tr)
    pi = permutation_importance(rf2, X_te, y_te, n_repeats=10, random_state=0, scoring="roc_auc")
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X_tr, y_tr)
    imp = pd.DataFrame({"variable": X.columns, "RF permutation importance (ΔAUC)": pi.importances_mean,
                        "SD": pi.importances_std, "|logistic coef| (standardized)": np.abs(lr[-1].coef_[0])})
    return rf.oob_score_, cv, imp


oob, cvacc, imp = _oob_and_importance()
c1, c2 = st.columns(2)
c1.metric("OOB accuracy", f"{oob:.3f}", border=True)
c2.metric("5-fold CV accuracy", f"{cvacc:.3f}", border=True)
st.caption("التقديران متقاربان؛ OOB يأتي مجانًا من تدريب واحد.")
plot(bars(imp["variable"], imp["RF permutation importance (ΔAUC)"], errors=imp["SD"],
          title="Random forest permutation importance on test data"), height=320)
st.dataframe(imp.round(3), hide_index=True, width="stretch")
st.markdown("قارن بمختبر الثقافتين: معاملات Logistic لـx3 وx4 وx5 صغيرة، بينما الأهمية بالتبديل للغابة تكشف أن x5 (الحد "
            "التربيعي) وx3 وx4 (التفاعل) مهمة. **هذه هي «المعلومات» التي وعد بها Breiman من النموذج الخوارزمي.**")
page_link("model_inspection")

if at_least("advanced"):
    st.markdown("## متقدم: خطأ التعميم للغابة (Breiman, 2001)")
    formula(r"PE^* \le \frac{\bar\rho\,(1-s^2)}{s^2}",
            symbols={"PE^*": "خطأ التعميم", r"\bar\rho": "متوسط الارتباط بين الأشجار", "s": "قوة (Strength) الأشجار الفردية"},
            intuition="غابة جيدة = أشجار قوية + ارتباط منخفض بينها. max_features يتحكم في هذه المفاضلة.")
if at_least("research"):
    researcher_note([
        "Arcing (1998): جادل Breiman بأن نجاح Boosting يعود إلى إعادة الوزن التكيفية نفسها، واقترح arc-x4؛ النقاش مع "
        "Friedman, Hastie & Tibshirani قاد إلى تفسير التعزيز كتحسين تدرجي لدالة خسارة.",
        "الأهمية المبنية على الشوائب (MDI) منحازة للمتغيرات كثيرة القيم؛ الأهمية بالتبديل على بيانات تحقق أكثر أمانًا، "
        "لكنها تتأثر بترابط المتغيرات (انظر درس راشومون).",
        "الغابات السببية وDML يستعملان أدوات Breiman لتقدير دوال الإزعاج وآثار غير متجانسة.",
    ])
why("حين تستخدم RandomForestClassifier أو BaggingClassifier أو StackingClassifier، أنت تستخدم أفكار Breiman مباشرة.",
    "فهم الدافع (عدم الاستقرار، الارتباط، التحقق التنبؤي) يجعلك تضبط المعاملات بفهم لا بالتجربة العمياء.")
st.markdown("### المراجع")
st.markdown(cite("breiman1984", "breiman1985ace", "breiman1995garrote", "breiman1996", "breiman1996stack",
                 "breiman1998arcing", "breiman2001", "chen2004brf"))
mistakes(["توقع أن يحسّن Bagging نموذجًا خطيًا متحيزًا.", "الاعتماد على Impurity importance بدل الأهمية بالتبديل.",
          "تجاهل OOB وتقسيم بيانات صغيرة لتحقق منفصل بلا داعٍ.", "نسيان أن BaggingClassifier يقلّل التباين لا التحيّز."])
page_footer("breiman_legacy",
            takeaways=["CART → Bagging → Random Forests: سلسلة واحدة لحل عدم الاستقرار.",
                       "Bagging يفيد الإجراءات غير المستقرة أساسًا.",
                       "OOB والأهمية بالتبديل: التحقق والمعلومات من داخل الغابة.",
                       "Garrote → Lasso، وArcing → Boosting، وStacking → Super learners وDML."])
