import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.animation import stepper
from components.callouts import intuition, mistakes, researcher_note
from components.cards import comparison_table
from components.diagrams import mermaid
from components.formulas import formula
from content.references import cite
from core.page import page_footer, page_header
from core.state import at_least
from utils.plotting import lines, plot

page_header("rl_bridge")

mermaid("""
flowchart LR
  A[Agent · policy π] -->|action a_t| E[Environment]
  E -->|state s_t+1| A
  E -->|reward r_t+1| A
""")
comparison_table([
    {"المفهوم": "State s", "المعنى": "وصف الوضع الحالي"}, {"المفهوم": "Action a", "المعنى": "ما يفعله الوكيل"},
    {"المفهوم": "Reward r", "المعنى": "إشارة فورية بالجودة"}, {"المفهوم": "Policy π(a|s)", "المعنى": "قاعدة اختيار الفعل"},
    {"المفهوم": "Return G_t", "المعنى": "مجموع المكافآت المستقبلية المخصومة Σγᵏr_{t+k+1}"},
    {"المفهوم": "Value / Q", "المعنى": "العائد المتوقع من حالة (أو حالة وفعل) باتباع السياسة"},
    {"المفهوم": "MDP", "المعنى": "(S, A, P, R, γ): الانتقال يعتمد على الحالة والفعل الحاليين فقط (خاصية ماركوف)"},
    {"المفهوم": "Exploration vs exploitation", "المعنى": "جرّب أفعالًا جديدة مقابل استغل الأفضل المعروف"},
])
formula(r"Q(s,a) \leftarrow Q(s,a) + \alpha\Big[r + \gamma\max_{a'}Q(s',a') - Q(s,a)\Big]", title="Q-learning (Watkins & Dayan, 1992)",
        symbols={r"\alpha": "معدل التعلّم", r"\gamma": "معامل الخصم", r"r + \gamma\max Q(s',\cdot)": "الهدف (TD target)"},
        intuition="قدّر قيمة كل (حالة، فعل) من التجربة، وحدّثها نحو «المكافأة الآن + أفضل ما يمكن بعدها».")

st.markdown("## Gridworld Q-learning Lab")
st.caption("شبكة 5×5: ابدأ من الزاوية العليا اليسرى (S)، الهدف (G) +10، الحفر (H) −10، كل خطوة −0.1.")
c1, c2, c3 = st.columns(3)
eps = c1.slider("ε (استكشاف)", 0.0, 1.0, 0.2, 0.05, key="rl_eps")
gamma = c2.slider("γ (خصم)", 0.5, 0.99, 0.95, 0.01, key="rl_gamma")
episodes = c3.select_slider("عدد الحلقات", [50, 200, 500, 1000], value=500, key="rl_ep")
N = 5
GOAL, HOLES = (4, 4), {(1, 1), (2, 3), (3, 1), (1, 3)}
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right


def _step(s, a):
    r, c = s[0] + MOVES[a][0], s[1] + MOVES[a][1]
    ns = (min(max(r, 0), N - 1), min(max(c, 0), N - 1))
    if ns == GOAL:
        return ns, 10.0, True
    if ns in HOLES:
        return ns, -10.0, True
    return ns, -0.1, False


@st.cache_data(show_spinner="يدرّب الوكيل…", max_entries=16)
def _train(eps, gamma, episodes, alpha=0.2, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.zeros((N, N, 4))
    returns, snapshots = [], []
    for ep in range(episodes):
        s, G, done, steps = (0, 0), 0.0, False, 0
        while not done and steps < 100:
            a = rng.integers(4) if rng.random() < eps else int(np.argmax(Q[s] + rng.normal(scale=1e-6, size=4)))
            ns, r, done = _step(s, a)
            target = r + (0 if done else gamma * Q[ns].max())
            Q[s][a] += alpha * (target - Q[s][a])
            s, G, steps = ns, G + r, steps + 1
        returns.append(G)
        if ep in (0, episodes // 10, episodes // 4, episodes // 2, episodes - 1):
            snapshots.append((ep + 1, Q.copy()))
    return np.array(returns), snapshots


returns, snaps = _train(eps, gamma, episodes)
ARROWS = ["↑", "↓", "←", "→"]


def _frame(i: int) -> None:
    ep, Q = snaps[i]
    V = Q.max(axis=2)
    text = [["G" if (r, c) == GOAL else "H" if (r, c) in HOLES else ARROWS[int(np.argmax(Q[r, c]))] for c in range(N)] for r in range(N)]
    text[0][0] = "S " + text[0][0]
    fig = go.Figure(go.Heatmap(z=V, text=text, texttemplate="%{text}", textfont=dict(size=20),
                               colorscale=[[0, "#FFE8CC"], [0.5, "#FCFCFF"], [1, "#C3FAE8"]], showscale=True,
                               colorbar=dict(title="max Q")))
    fig.update_layout(title=f"Greedy policy and state values after {ep} episodes", yaxis=dict(autorange="reversed", visible=False),
                      xaxis=dict(visible=False), height=400)
    plot(fig)


stepper(f"rl_{eps}_{gamma}_{episodes}", len(snaps), _frame, labels=[f"episode {s[0]}" for s in snaps])
w = max(1, episodes // 25)
smooth = np.convolve(returns, np.ones(w) / w, mode="valid")
plot(lines(np.arange(len(smooth)), {"moving-average return": smooth}, title="Learning curve", xaxis="episode", yaxis="return"), height=280)
intuition("ε = 0: لا استكشاف ⇒ قد يعلق الوكيل في مسار سيئ. ε = 1: عشوائي تمامًا. ε صغير موجب يوازن. γ قريب من 1 يجعل الوكيل "
          "«بعيد النظر» فيفضّل مسارًا أطول آمنًا على مخاطرة قريبة.")
st.markdown("## صلته بتعلّم الآلة الموجّه")
comparison_table([
    {"": "Supervised", "البيانات": "أزواج (x, y) ثابتة", "الإشارة": "الإجابة الصحيحة", "التحدي": "التعميم"},
    {"": "Reinforcement", "البيانات": "يولّدها الوكيل بأفعاله", "الإشارة": "مكافأة متأخرة", "التحدي": "Credit assignment والاستكشاف"},
    {"": "Contextual bandits", "البيانات": "سياق ← فعل ← مكافأة فورية", "الإشارة": "مكافأة للفعل المختار فقط", "التحدي": "جسر بين الاثنين (توصيات، تسعير)"},
])
if at_least("research"):
    researcher_note(["Sutton & Barto (2018) المرجع الأساسي؛ Deep RL (DQN، Policy gradients) خارج نطاق المنصة عمدًا.",
                     "تقييم السياسات من بيانات تاريخية (Off-policy evaluation) يستخدم درجات مزدوجة المتانة — قريبة من DML."])
    st.markdown(cite("watkins1992", "sutton2018"))
mistakes(["ε = 0 من البداية.", "تقييم الوكيل على بيئة التدريب نفسها فقط.", "مكافأة مصممة تُستغل بطرق غير مقصودة (Reward hacking)."])
page_footer("rl_bridge",
            takeaways=["RL: وكيل يتعلم سياسة من مكافآت متأخرة.", "Q-learning يحدّث القيم نحو هدف TD.",
                       "الاستكشاف والاستغلال مقايضة أساسية."])
