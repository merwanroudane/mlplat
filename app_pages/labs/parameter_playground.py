import streamlit as st

from components.callouts import intuition, mistakes
from components.parameter_lab import parameter_playground
from core.page import page_footer, page_header

page_header("parameter_playground")
st.markdown("اختر خوارزمية وبيانات ثنائية الأبعاد، وحرّك المعاملات الفائقة: يُعاد التدريب فورًا وتظهر حدود القرار ودرجتا التدريب "
            "والتحقق (30% محجوزة، النقاط المحاطة بإطار) والزمن والتعقيد. القيم الابتدائية لكل منزلق هي **القيمة الافتراضية المتحقق منها** "
            "في scikit-learn 1.9.1 (عدا بنية MLP).")
parameter_playground(None, key="pg_main", dataset="moons", allow_model_choice=True)
intuition("ابحث عن ثلاث حالات لكل خوارزمية: (1) Underfitting: الدرجتان منخفضتان؛ (2) Overfitting: تدريب ≈ 1 وتحقق أقل بوضوح؛ "
          "(3) المنطقة الجيدة بينهما. ثم غيّر الضجيج وn وراقب كيف تتحرك المنطقة الجيدة.")
st.markdown("### أفكار للتجريب")
st.markdown("- **kNN:** k = 1 مقابل k = 50 على moons بضجيج 0.4.\n- **SVM:** gamma = 10 وC = 1000 (جزر) مقابل gamma = 'scale'.\n"
            "- **Decision tree:** max_depth = 0 (بلا حد) ثم زد ccp_alpha.\n- **Random forest:** لاحظ أن زيادة n_estimators لا تسبب Overfitting.\n"
            "- **Logistic:** على circles يفشل خطيًا مهما كان C — السعة لا المعامل.\n- **MLP:** وحدة واحدة مقابل 64 وحدة.")
mistakes(["الحكم من تقسيم واحد 70/30 (استخدم CV للقرار النهائي).", "ضبط المعاملات بالنظر إلى درجة التحقق نفسها مرارًا (تسرب HPO)."])
page_footer("parameter_playground", takeaways=["كل معامل فائق يحرك النموذج على محور التحيز/التباين.",
                                               "الفجوة بين التدريب والتحقق مؤشر Overfitting.", "السعة المناسبة تعتمد على n والضجيج."])
