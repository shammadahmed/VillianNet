import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import json
import os

st.set_page_config(
    page_title="VillainNet | Character Alignment Arena",
    page_icon="🥊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# Custom Styling (Comic Visual Cards & Badges)
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
        letter-spacing: 0.5px;
    }
    .badge-bad { background: rgba(239, 68, 68, 0.18); color: #f87171; border: 1px solid #ef4444; }
    .badge-good { background: rgba(34, 197, 94, 0.18); color: #4ade80; border: 1px solid #22c55e; }
    .badge-neutral { background: rgba(245, 158, 11, 0.18); color: #fbbf24; border: 1px solid #f59e0b; }
    
    .char-avatar-container {
        display: flex;
        align-items: center;
        gap: 16px;
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 1rem;
    }
    .char-avatar-img {
        width: 80px;
        height: 80px;
        border-radius: 50%;
        object-fit: cover;
        border: 2px solid #00d2ff;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Preset Roster with Hosted Imagery & Lore Details
# ---------------------------------------------------------
PRESET_CHARACTERS = {
    "invincible": {
        "name": "Invincible (Mark Grayson)",
        "tag": "Image Comics",
        "image": "https://images.unsplash.com/photo-1607604276583-eef5d076aa5f?w=400&auto=format&fit=crop&q=80",
        "publisher": "Marvel", "sex": "Male", "identity": "Secret",
        "alive": "Living", "eye": "Blue", "hair": "Black",
        "appearances": 144, "year": 2003, "actual_align": "Good Characters"
    },
    "joker": {
        "name": "The Joker",
        "tag": "DC Comics",
        "image": "https://images.unsplash.com/photo-1579783902614-a3fb3927b675?w=400&auto=format&fit=crop&q=80",
        "publisher": "DC", "sex": "Male", "identity": "Secret",
        "alive": "Living", "eye": "Green", "hair": "Other",
        "appearances": 3000, "year": 1940, "actual_align": "Bad Characters"
    },
    "homelander": {
        "name": "Homelander (John)",
        "tag": "The Boys / Dynamite",
        "image": "https://images.unsplash.com/photo-1534447677768-be436bb09401?w=400&auto=format&fit=crop&q=80",
        "publisher": "DC", "sex": "Male", "identity": "Public",
        "alive": "Living", "eye": "Blue", "hair": "Blonde",
        "appearances": 72, "year": 2006, "actual_align": "Bad Characters"
    },
    "magneto": {
        "name": "Magneto (Max Eisenhardt)",
        "tag": "Marvel Comics",
        "image": "https://images.unsplash.com/photo-1568602471122-7832951cc4c5?w=400&auto=format&fit=crop&q=80",
        "publisher": "Marvel", "sex": "Male", "identity": "Public",
        "alive": "Living", "eye": "Other", "hair": "White",
        "appearances": 1400, "year": 1963, "actual_align": "Neutral Characters"
    },
    "spawn": {
        "name": "Spawn (Al Simmons)",
        "tag": "Image Comics",
        "image": "https://images.unsplash.com/photo-1509198397868-475647b2a1e5?w=400&auto=format&fit=crop&q=80",
        "publisher": "DC", "sex": "Male", "identity": "Secret",
        "alive": "Deceased", "eye": "Green", "hair": "Other",
        "appearances": 350, "year": 1992, "actual_align": "Neutral Characters"
    }
}

CLASSES = ['Bad Characters', 'Good Characters', 'Neutral Characters']

# ---------------------------------------------------------
# Neural Network Architecture
# ---------------------------------------------------------
class EnhancedVillainNet(nn.Module):
    def __init__(self, in_features, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# ---------------------------------------------------------
# Pre-Trained Model Loader
# ---------------------------------------------------------
@st.cache_resource
def load_saved_model():
    if not (os.path.exists("model.pt") and os.path.exists("scaler.pkl") and os.path.exists("meta.json")):
        from train import train_and_export
        train_and_export()

    with open("meta.json", "r") as f:
        meta = json.load(f)

    feature_cols = meta["features"]
    median_year = meta["median_year"]

    scaler = joblib.load("scaler.pkl")
    model = EnhancedVillainNet(in_features=len(feature_cols))
    model.load_state_dict(torch.load("model.pt", map_location=torch.device('cpu')))
    model.eval()

    return model, scaler, feature_cols, median_year

if "arena_tests" not in st.session_state:
    st.session_state.arena_tests = []
if "active_char_key" not in st.session_state:
    st.session_state.active_char_key = "invincible"

model, scaler, feature_cols, median_year = load_saved_model()

# ---------------------------------------------------------
# App Header
# ---------------------------------------------------------
st.markdown('<div class="main-header">🥊 VILLAIN-NET ARENA</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Select a hero or villain from the roster below to challenge the neural network\'s alignment classifier.</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# Visual Preset Character Cards
# ---------------------------------------------------------
st.markdown("#### 🌟 Select a Challenger")
cols = st.columns(len(PRESET_CHARACTERS))

for idx, (key, data) in enumerate(PRESET_CHARACTERS.items()):
    with cols[idx]:
        st.image(data["image"], use_container_width=True)
        is_active = (st.session_state.active_char_key == key)
        btn_label = f"✓ {data['name'].split(' (')[0]}" if is_active else data['name'].split(' (')[0]
        
        if st.button(btn_label, key=f"btn_{key}", use_container_width=True, type="primary" if is_active else "secondary"):
            st.session_state.active_char_key = key
            st.rerun()

current_preset = PRESET_CHARACTERS[st.session_state.active_char_key]

st.markdown("---")

# ---------------------------------------------------------
# Character Profile & Arena Prediction
# ---------------------------------------------------------
col_left, col_right = st.columns([1.1, 1.3], gap="large")

with col_left:
    st.markdown("### 🧬 Character Profile")
    
    # Active character avatar summary card
    st.markdown(f"""
    <div class="char-avatar-container">
        <img src="{current_preset['image']}" class="char-avatar-img">
        <div>
            <h3 style="margin:0;">{current_preset['name']}</h3>
            <span style="color:#8b949e; font-size:0.85rem;">{current_preset['tag']} • Known: <b>{current_preset['actual_align']}</b></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.form("character_form"):
        char_name = st.text_input("Display Name", value=current_preset["name"])
        
        c_truth, c_pub = st.columns(2)
        with c_truth:
            actual_align = st.selectbox(
                "Known True Alignment",
                CLASSES,
                index=CLASSES.index(current_preset["actual_align"])
            )
        with c_pub:
            pub_idx = 0 if current_preset["publisher"] == "DC" else 1
            publisher = st.radio("Publisher Style", ["DC", "Marvel"], index=pub_idx, horizontal=True)

        c1, c2 = st.columns(2)
        with c1:
            sex_options = ["Male", "Female", "Other"]
            sex = st.selectbox("Sex", sex_options, index=sex_options.index(current_preset.get("sex", "Male")))

            id_options = ["Secret", "Public", "No Dual", "Other"]
            identity = st.selectbox("Identity Type", id_options, index=id_options.index(current_preset.get("identity", "Secret")))

            alive_options = ["Living", "Deceased", "Other"]
            alive = st.selectbox("Vital Status", alive_options, index=alive_options.index(current_preset.get("alive", "Living")))

        with c2:
            eye_options = ["Blue", "Brown", "Black", "Green", "Red", "Other"]
            eye = st.selectbox("Eye Color", eye_options, index=eye_options.index(current_preset.get("eye", "Blue")))

            hair_options = ["Black", "Brown", "Blonde", "Red", "White", "Other"]
            hair = st.selectbox("Hair Color", hair_options, index=hair_options.index(current_preset.get("hair", "Black")))

            appearances = st.number_input(
                "Comic Appearances",
                min_value=1,
                max_value=6000,
                value=int(current_preset["appearances"]),
                step=5
            )

            year = st.slider(
                "Debut Year",
                min_value=1935,
                max_value=2026,
                value=int(current_preset["year"])
            )

        predict_btn = st.form_submit_button("⚡ Run Neural Net Prediction", type="primary", use_container_width=True)

with col_right:
    st.markdown("### 📊 Classification & Evaluation")

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

        # Record test in session history
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
                    ❌ INCORRECT VERDICT!<br>
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
        st.info("Pick a character preset above or adjust the traits in the form, then click **⚡ Run Neural Net Prediction**.")

# ---------------------------------------------------------
# Arena Scorecard & History Export
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