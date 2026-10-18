import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="VillainNet Arena | Comic Character Alignment AI",
    page_icon="🥊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# Custom Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Bungee&family=Inter:wght@400;600;800&display=swap');

    .main-header {
        font-family: 'Bungee', cursive, sans-serif;
        font-size: 2.2rem;
        letter-spacing: 1px;
        background: linear-gradient(90deg, #ff4b4b, #ffa116, #00d2ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    .verdict-box {
        border-radius: 10px;
        padding: 16px;
        margin-top: 10px;
        font-weight: 600;
        text-align: center;
    }
    .correct {
        background-color: rgba(34, 197, 94, 0.15);
        border: 2px solid #22c55e;
        color: #4ade80;
    }
    .incorrect {
        background-color: rgba(239, 68, 68, 0.15);
        border: 2px solid #ef4444;
        color: #f87171;
    }
    .stat-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .badge-bad { background: rgba(239, 68, 68, 0.18); color: #f87171; border: 1px solid #ef4444; }
    .badge-good { background: rgba(34, 197, 94, 0.18); color: #4ade80; border: 1px solid #22c55e; }
    .badge-neutral { background: rgba(245, 158, 11, 0.18); color: #fbbf24; border: 1px solid #f59e0b; }

    .hero-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 10px;
        text-align: center;
        transition: transform 0.2s, border-color 0.2s;
    }
    .hero-card:hover {
        border-color: #58a6ff;
        transform: translateY(-2px);
    }
    .hero-img {
        width: 100%;
        height: 140px;
        object-fit: cover;
        border-radius: 8px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Verified Character Roster (Accurate Dataset Attributes + Permanent Art)
# ---------------------------------------------------------
ROSTER = {
    "spiderman": {
        "name": "Spider-Man (Peter Parker)",
        "universe": "Marvel Comics",
        "img": "https://cdn.jsdelivr.net/gh/shammadahmed/assets@main/spiderman.jpg",
        "fallback_svg": "🕷️",
        "publisher": "Marvel", "sex": "Male Characters", "identity": "Secret Identity",
        "alive": "Living Characters", "eye": "Hazel Eyes", "hair": "Brown Hair",
        "appearances": 4043, "year": 1962, "actual_align": "Good Characters"
    },
    "joker": {
        "name": "The Joker",
        "universe": "DC Comics",
        "img": "https://cdn.jsdelivr.net/gh/shammadahmed/assets@main/joker.jpg",
        "fallback_svg": "🃏",
        "publisher": "DC", "sex": "Male Characters", "identity": "Secret Identity",
        "alive": "Living Characters", "eye": "Green Eyes", "hair": "Green Hair",
        "appearances": 519, "year": 1940, "actual_align": "Bad Characters"
    },
    "batman": {
        "name": "Batman (Bruce Wayne)",
        "universe": "DC Comics",
        "img": "https://cdn.jsdelivr.net/gh/shammadahmed/assets@main/batman.jpg",
        "fallback_svg": "🦇",
        "publisher": "DC", "sex": "Male Characters", "identity": "Secret Identity",
        "alive": "Living Characters", "eye": "Blue Eyes", "hair": "Black Hair",
        "appearances": 3098, "year": 1939, "actual_align": "Good Characters"
    },
    "goblin": {
        "name": "Green Goblin (Norman Osborn)",
        "universe": "Marvel Comics",
        "img": "https://cdn.jsdelivr.net/gh/shammadahmed/assets@main/goblin.jpg",
        "fallback_svg": "🎃",
        "publisher": "Marvel", "sex": "Male Characters", "identity": "Secret Identity",
        "alive": "Living Characters", "eye": "Blue Eyes", "hair": "Auburn Hair",
        "appearances": 574, "year": 1964, "actual_align": "Bad Characters"
    },
    "magneto": {
        "name": "Magneto (Max Eisenhardt)",
        "universe": "Marvel Comics",
        "img": "https://cdn.jsdelivr.net/gh/shammadahmed/assets@main/magneto.jpg",
        "fallback_svg": "🧲",
        "publisher": "Marvel", "sex": "Male Characters", "identity": "Public Identity",
        "alive": "Living Characters", "eye": "Grey Eyes", "hair": "White Hair",
        "appearances": 1408, "year": 1963, "actual_align": "Neutral Characters"
    },
    "thanos": {
        "name": "Thanos (The Mad Titan)",
        "universe": "Marvel Comics",
        "img": "https://cdn.jsdelivr.net/gh/shammadahmed/assets@main/thanos.jpg",
        "fallback_svg": "🧤",
        "publisher": "Marvel", "sex": "Male Characters", "identity": "Public Identity",
        "alive": "Deceased Characters", "eye": "Red Eyes", "hair": "No Hair",
        "appearances": 330, "year": 1973, "actual_align": "Bad Characters"
    }
}

CLASSES = ['Bad Characters', 'Good Characters', 'Neutral Characters']

# ---------------------------------------------------------
# Neural Network Definition
# ---------------------------------------------------------
class VillainNet(nn.Module):
    def __init__(self, in_features, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# ---------------------------------------------------------
# Robust Training Pipeline
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Training Neural Network on Comic Data...")
def load_and_train():
    dc = pd.read_csv("dc-wikia-data.csv")
    ml = pd.read_csv("marvel-wikia-data.csv")

    for d in (dc, ml):
        d.columns = d.columns.str.upper()
    dc['PUBLISHER'], ml['PUBLISHER'] = 'DC', 'Marvel'

    df = pd.concat([dc, ml], ignore_index=True)
    df = df[df['ALIGN'].isin(CLASSES)]
    df = df.drop(columns=['PAGE_ID', 'URLSLUG'], errors='ignore')

    # Fixed regex: 'Characters' with 'a'
    def tidy(col, keep):
        s = df[col].fillna('Other').str.replace(' Characters| Eyes| Hair| Identity', '', regex=True)
        top_cats = s.value_counts().index[:keep]
        return s.where(s.isin(top_cats), 'Other')

    clean_df = pd.DataFrame({
        'publisher': df['PUBLISHER'],
        'sex': tidy('SEX', 2),
        'identity': tidy('ID', 3),
        'alive': tidy('ALIVE', 2),
        'eye': tidy('EYE', 5),
        'hair': tidy('HAIR', 5)
    })

    X = pd.get_dummies(clean_df, dtype=float)
    X['appearance'] = np.log1p(df['APPEARANCES'].fillna(1))
    median_year = float(df['YEAR'].median())
    X['year'] = df['YEAR'].fillna(median_year)
    y = df['ALIGN'].map({c: i for i, c in enumerate(CLASSES)})

    feature_cols = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.long)
    X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_t = torch.tensor(y_val.values, dtype=torch.long)

    torch.manual_seed(42)
    model = VillainNet(in_features=len(feature_cols))

    counts = torch.bincount(y_train_t).float()
    weights = counts.max() / counts
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.008)

    best_val, best_state = float('inf'), None
    for _ in range(80):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(X_train_t), y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            vloss = loss_fn(model(X_val_t), y_val_t).item()
        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    return model, scaler, feature_cols, median_year

if "arena_tests" not in st.session_state:
    st.session_state.arena_tests = []
if "selected_key" not in st.session_state:
    st.session_state.selected_key = "joker"

try:
    model, scaler, feature_cols, median_year = load_and_train()
except Exception as e:
    st.error(f"Dataset load error: {e}. Ensure `dc-wikia-data.csv` and `marvel-wikia-data.csv` exist.")
    st.stop()

# ---------------------------------------------------------
# App Header
# ---------------------------------------------------------
st.markdown('<div class="main-header">🥊 VILLAIN-NET ARENA</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Challenge the neural network against iconic heroes, anti-heroes, and supervillains.</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# Character Roster Selector
# ---------------------------------------------------------
st.markdown("#### 🌟 Select a Challenger")
roster_cols = st.columns(len(ROSTER))

for idx, (key, data) in enumerate(ROSTER.items()):
    with roster_cols[idx]:
        is_sel = (st.session_state.selected_key == key)
        st.markdown(f"""
        <div class="hero-card" style="border-color:{'#ff4b4b' if is_sel else '#30363d'};">
            <div style="font-size: 2.2rem; margin-bottom: 6px;">{data['fallback_svg']}</div>
            <div style="font-weight:700; font-size:0.9rem;">{data['name'].split(' (')[0]}</div>
            <div style="color:#8b949e; font-size:0.75rem;">{data['universe']}</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Select", key=f"sel_{key}", use_container_width=True, type="primary" if is_sel else "secondary"):
            st.session_state.selected_key = key
            st.rerun()

current = ROSTER[st.session_state.selected_key]
st.write("---")

# ---------------------------------------------------------
# Profile & Execution
# ---------------------------------------------------------
col_left, col_right = st.columns([1.1, 1.3], gap="large")

with col_left:
    st.markdown("### 🧬 Character Profile")
    with st.form("character_form"):
        char_name = st.text_input("Name", value=current["name"])
        
        c_truth, c_pub = st.columns(2)
        with c_truth:
            actual_align = st.selectbox(
                "Known Ground Truth",
                CLASSES,
                index=CLASSES.index(current["actual_align"])
            )
        with c_pub:
            publisher = st.radio(
                "Universe",
                ["DC", "Marvel"],
                index=0 if current["publisher"] == "DC" else 1,
                horizontal=True
            )

        c1, c2 = st.columns(2)
        with c1:
            clean_sex = current["sex"].replace(" Characters", "")
            sex = st.selectbox("Sex", ["Male", "Female", "Other"], index=["Male", "Female", "Other"].index(clean_sex if clean_sex in ["Male", "Female"] else "Other"))
            
            clean_id = current["identity"].replace(" Identity", "")
            id_opts = ["Secret", "Public", "No Dual", "Other"]
            identity = st.selectbox("Identity Type", id_opts, index=id_opts.index(clean_id if clean_id in id_opts else "Other"))

            clean_alive = current["alive"].replace(" Characters", "")
            alive_opts = ["Living", "Deceased", "Other"]
            alive = st.selectbox("Vital Status", alive_opts, index=alive_opts.index(clean_alive if clean_alive in alive_opts else "Other"))

        with c2:
            clean_eye = current["eye"].replace(" Eyes", "")
            eye_opts = ["Blue", "Brown", "Black", "Green", "Red", "Other"]
            eye = st.selectbox("Eye Color", eye_opts, index=eye_opts.index(clean_eye if clean_eye in eye_opts else "Other"))

            clean_hair = current["hair"].replace(" Hair", "")
            hair_opts = ["Black", "Brown", "Blond", "Red", "White", "Other"]
            hair = st.selectbox("Hair Color", hair_opts, index=hair_opts.index(clean_hair if clean_hair in hair_opts else "Other"))

            appearances = st.number_input("Comic Appearances", min_value=1, max_value=6000, value=int(current["appearances"]))
            year = st.slider("Debut Year", 1935, 2026, int(current["year"]))

        predict_btn = st.form_submit_button("⚡ Run Neural Net Prediction", type="primary", use_container_width=True)

with col_right:
    st.markdown("### 📊 Classification Output")

    if predict_btn:
        input_df = pd.DataFrame(0.0, index=[0], columns=feature_cols)
        input_df['appearance'] = np.log1p(appearances)
        input_df['year'] = float(year)

        for col in [
            f"publisher_{publisher}",
            f"sex_{sex}",
            f"identity_{identity}",
            f"alive_{alive}",
            f"eye_{eye}",
            f"hair_{hair}"
        ]:
            if col in input_df.columns:
                input_df[col] = 1.0

        scaled = scaler.transform(input_df)
        tensor_input = torch.tensor(scaled, dtype=torch.float32)

        with torch.no_grad():
            logits = model(tensor_input)
            probs = torch.softmax(logits, dim=1).numpy()[0]
            pred_idx = int(torch.argmax(logits, dim=1).item())

        pred_class = CLASSES[pred_idx]
        badge_type = "badge-bad" if "Bad" in pred_class else ("badge-good" if "Good" in pred_class else "badge-neutral")
        is_correct = (pred_class == actual_align)

        st.session_state.arena_tests.insert(0, {
            "Character": char_name,
            "Actual": actual_align,
            "Predicted": pred_class,
            "Confidence": f"{probs[pred_idx] * 100:.1f}%",
            "Result": "✅ Correct" if is_correct else "❌ Missed"
        })

        with st.container(border=True):
            if is_correct:
                st.markdown(f"""
                <div class="verdict-box correct">
                    🎉 CORRECT PREDICTION!<br>
                    The neural net recognized <b>{char_name}</b> as <u>{pred_class}</u>.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="verdict-box incorrect">
                    ❌ MISSED PREDICTION!<br>
                    Expected <b>{actual_align}</b>, but model predicted <b>{pred_class}</b>.
                </div>
                """, unsafe_allow_html=True)

            st.write("")
            st.markdown(f"""
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <span style="font-size:0.85rem; color:#8b949e; text-transform:uppercase;">Model Verdict</span>
                        <h3 style="margin:4px 0 0 0;">{pred_class}</h3>
                    </div>
                    <div class="stat-badge {badge_type}">{probs[pred_idx]*100:.1f}% Confidence</div>
                </div>
                <hr style="border-color:#30363d; margin: 15px 0;">
            """, unsafe_allow_html=True)

            st.caption("Class Probability Distribution")
            for label, prob in zip(CLASSES, probs):
                col_label, col_bar, col_pct = st.columns([1.2, 3, 0.8])
                with col_label:
                    st.write(f"**{label.split()[0]}**")
                with col_bar:
                    st.progress(float(prob))
                with col_pct:
                    st.write(f"{prob * 100:.1f}%")
    else:
        st.info("Select a challenger above, review their stats, and click **⚡ Run Neural Net Prediction**.")

# ---------------------------------------------------------
# Scorecard
# ---------------------------------------------------------
if st.session_state.arena_tests:
    st.write("---")
    st.markdown("### 🏆 Arena Scorecard")
    test_df = pd.DataFrame(st.session_state.arena_tests)
    total_tests = len(test_df)
    correct_tests = len(test_df[test_df["Result"] == "✅ Correct"])
    accuracy = (correct_tests / total_tests) * 100

    m1, m2, m3 = st.columns(3)
    m1.metric("Characters Tested", total_tests)
    m2.metric("Correct Predictions", correct_tests)
    m3.metric("Arena Accuracy", f"{accuracy:.1f}%")

    st.dataframe(test_df, use_container_width=True, hide_index=True)

    csv_data = test_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export Scorecard (CSV)",
        data=csv_data,
        file_name="arena_scorecard.csv",
        mime="text/csv"
    )