import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import BaggingClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from components.callouts import intuition, mistakes, researcher_note, why
from components.cards import comparison_table
from components.decision_boundary import boundary_chart
from components.diagrams import mermaid
from components.formulas import formula
from content.references import cite
from core.page import page_footer, page_header
from core.state import at_least
from utils.datasets import toy_2d, xy
from utils.plotting import lines, plot

page_header("ensemble_learning")

formula(r"\operatorname{Var}\Big(\frac1B\sum_{b=1}^B \hat f_b\Big) = \rho\,\sigma^2 + \frac{1-\rho}{B}\sigma^2",
        title="Why averaging helps",
        symbols={r"\sigma^2": "تباين نموذج واحد", r"\rho": "الارتباط بين النماذج", "B": "عدد النماذج"},
        intuition="مع B كبير يختفي الحد الثاني ويبقى ρσ²: التجميع يفيد بقدر ما تكون النماذج **مختلفة**. لذلك نحقن العشوائية.",
        example="σ² = 1، ρ = 0.3، B = 100 ⇒ التباين = 0.3 + 0.007 = 0.307 بدل 1.")
comparison_table([
    {"الأسلوب": "Bagging", "الفكرة": "نماذج متوازية على عينات Bootstrap ثم متوسط/تصويت", "يخفض": "التباين",
     "مثال": "BaggingClassifier، Random Forest"},
    {"الأسلوب": "Random subspace", "الفكرة": "كل نموذج يرى خصائص عشوائية", "يخفض": "الارتباط ρ", "مثال": "max_features في RF"},
    {"الأسلوب": "Voting", "الفكرة": "نماذج مختلفة النوع تصوّت (hard) أو تتوسط الاحتمالات (soft)", "يخفض": "التباين/الأخطاء المستقلة",
     "مثال": "VotingClassifier"},
    {"الأسلوب": "Stacking", "الفكرة": "نموذج فوقي يتعلم كيف يجمع تنبؤات خارج الطية", "يخفض": "كلاهما", "مثال": "StackingClassifier"},
    {"الأسلوب": "Boosting", "الفكرة": "نماذج تسلسلية يصحح كل منها أخطاء ما قبله", "يخفض": "التحيز أساسًا",
     "مثال": "AdaBoost، Gradient Boosting"},
])
mermaid("""
flowchart LR
  subgraph Bagging
    D1[(data)] --> B1[bootstrap 1] --> M1[model]
    D1 --> B2[bootstrap 2] --> M2[model]
    D1 --> B3[bootstrap B] --> M3[model]
    M1 & M2 & M3 --> AVG((average / vote))
  end
  subgraph Boosting
    D2[(data)] --> F1[model 1] --> R1[residuals / weights] --> F2[model 2] --> R2[...] --> FM[model M]
    F1 & F2 & FM --> SUM((weighted sum))
  end
""")

st.markdown("## مختبر: Bagging لشجرة عميقة")
c1, c2 = st.columns(2)
B = c1.slider("عدد النماذج B", 1, 200, 50, key="ens_B")
kind = c2.segmented_control("البيانات", ["moons", "circles"], default="moons", key="ens_kind", required=True)
X, y = toy_2d(kind, n=300, noise=0.35)
single = DecisionTreeClassifier(random_state=0).fit(X, y)
bag = BaggingClassifier(DecisionTreeClassifier(), n_estimators=B, random_state=0, n_jobs=1).fit(X, y)
c1, c2 = st.columns(2)
with c1:
    boundary_chart(single, X, y, title=f"Single deep tree · CV {cross_val_score(DecisionTreeClassifier(random_state=0), X, y, cv=5).mean():.3f}",
                   height=380, show_proba=False)
with c2:
    cv_bag = cross_val_score(BaggingClassifier(DecisionTreeClassifier(), n_estimators=B, random_state=0), X, y, cv=5).mean()
    boundary_chart(bag, X, y, title=f"Bagging of {B} trees · CV {cv_bag:.3f}", height=380)
intuition("كل شجرة عميقة تحفظ عينتها؛ متوسطها ينعّم الحدود ويلغي جزءًا كبيرًا من الضجيج — تحيز منخفض + تباين أقل.")

st.markdown("## Voting وStacking على بيانات حقيقية")
Xc, yc = xy("classification")
base = [("lr", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
        ("knn", make_pipeline(StandardScaler(), KNeighborsClassifier(15))),
        ("nb", GaussianNB()),
        ("tree", DecisionTreeClassifier(max_depth=6, random_state=0))]


@st.cache_data(show_spinner="يقيّم 7 نماذج بـ5 طيات…")
def _compare():
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    rows = []
    for name, m in base:
        s = cross_val_score(m, Xc, yc, cv=cv, scoring="roc_auc")
        rows.append({"model": name, "ROC-AUC": s.mean(), "± SD": s.std()})
    for name, m in (("Voting (hard)", VotingClassifier(base, voting="hard")),
                    ("Voting (soft)", VotingClassifier(base, voting="soft")),
                    ("Stacking (meta = logistic)", StackingClassifier(base, final_estimator=LogisticRegression(), cv=5))):
        scoring = "accuracy" if "hard" in name else "roc_auc"
        s = cross_val_score(m, Xc, yc, cv=cv, scoring=scoring)
        rows.append({"model": name + (" [accuracy]" if scoring == "accuracy" else ""), "ROC-AUC": s.mean(), "± SD": s.std()})
    return pd.DataFrame(rows)


if st.button("شغّل المقارنة", key="ens_run", icon=":material/play_arrow:"):
    st.session_state["ens_done"] = True
if st.session_state.get("ens_done"):
    st.dataframe(_compare().round(4), hide_index=True, width="stretch")
    st.caption("Hard voting لا يعطي احتمالات فقيس بالدقة. Stacking يدرّب النموذج الفوقي على تنبؤات **خارج الطية** للنماذج الأساسية "
               "(cv=5 داخليًا) — وإلا لتعلّم من تنبؤات متفائلة (تسرب).")
why("اجمع نماذج مختلفة الطبيعة.", "التجميع يفيد حين تكون الأخطاء غير مترابطة؛ خمس نسخ من النموذج نفسه لا تضيف شيئًا.")

if at_least("advanced"):
    st.markdown("## متقدم: متى يفشل التجميع؟")
    st.markdown("- نماذج أساسية متحيزة بالطريقة نفسها ⇒ المتوسط متحيز أيضًا.\n- Bagging نموذج مستقر (Logistic) ⇒ فائدة ضئيلة.\n"
                "- Stacking مع بيانات قليلة ⇒ النموذج الفوقي يفرط في الملاءمة.")
    rhos = np.linspace(0, 1, 21)
    plot(lines(rhos, {"B = 10": rhos + (1 - rhos) / 10, "B = 100": rhos + (1 - rhos) / 100},
               title="Variance of the average (σ² = 1) vs correlation ρ", xaxis="ρ", yaxis="variance"), height=300)
if at_least("research"):
    researcher_note(["Super Learner (van der Laan et al., 2007) = Stacking مع ضمانات Oracle؛ شائع كمتعلم إزعاج في TMLE وDML.",
                     "Bagging يقلل التباين للمتعلمين غير المستقرين (Breiman, 1996)."])
    st.markdown(cite("breiman1996", "esl"))
mistakes(["Stacking دون تنبؤات خارج الطية.", "تجميع نماذج متطابقة تقريبًا.", "Hard voting ثم استخدام احتمالات."])
page_footer("ensemble_learning",
            takeaways=["التجميع يخفض التباين بقدر عدم ترابط النماذج.", "Bagging/Voting متوازيان؛ Boosting تسلسلي يخفض التحيز.",
                       "Stacking يتعلم الجمع من تنبؤات خارج الطية."])
