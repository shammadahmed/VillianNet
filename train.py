import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import json
import os

st.set_page_config(
    page_title="VillainNet | Comic Character Alignment AI",
    page_icon="🦸",
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
</style>
""", unsafe_allow_html=True)

CLASSES = ['Bad Characters', 'Good Characters', 'Neutral Characters']

# ---------------------------------------------------------
# Model Architecture
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
# Instant Artifact Loader
# ---------------------------------------------------------
@st.cache_resource
def load_saved_model():
    if not (os.path.exists("model.pt") and os.path.exists("scaler.pkl") and os.path.exists("meta.json")):
        # Auto-train fallback if artifacts aren't committed
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

if "history" not in st.session_state:
    st.session_state.history = []

model, scaler, feature_cols, median_year = load_saved_model()

# ---------------------------------------------------------
# Header & UI
# ---------------------------------------------------------
st.markdown('<div class="main-header">⚡ VILLAIN-NET ANALYZER</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Pre-Trained Deep Neural Net Character Alignment Classifier</div>', unsafe_allow_html=True)

col_left, col_right = st.columns([1.1, 1.3], gap="large")

with col_left:
    st.markdown("### 🧬 Character Profile")
    with st.form("character_form"):
        c1, c2 = st.columns(2)
        with c1:
            publisher = st.radio("Publisher", ["DC", "Marvel"], horizontal=True)
            sex = st.selectbox("Sex", ["Male Characters", "Female Characters", "Other"])
            identity = st.selectbox("Identity Type", ["Secret Identity", "Public Identity", "No Dual Identity", "Other"])
            alive = st.selectbox("Vital Status", ["Living Characters", "Deceased Characters", "Other"])
        with c2:
            eye = st.selectbox("Eye Color", ["Blue Eyes", "Brown Eyes", "Black Eyes", "Green Eyes", "Red Eyes", "Other"])
            hair = st.selectbox("Hair Color", ["Black Hair", "Brown Hair", "Blonde Hair", "Red Hair", "White Hair", "Other"])
            appearances = st.number_input("Comic Appearances", min_value=1, max_value=6000, value=50, step=5)
            year = st.slider("First Debuted (Year)", min_value=1935, max_value=2026, value=int(median_year))

        predict_btn = st.form_submit_button("⚡ Classify Alignment", type="primary", use_container_width=True)

with col_right:
    st.markdown("### 📊 Classification Output")

    if predict_btn:
        clean_sex = sex.replace(" Characters", "")
        clean_id = identity.replace(" Identity", "")
        clean_alive = alive.replace(" Characters", "")
        clean_eye = eye.replace(" Eyes", "")
        clean_hair = hair.replace(" Hair", "")

        input_df = pd.DataFrame(0.0, index=[0], columns=feature_cols)
        input_df['appearance'] = np.log1p(appearances)
        input_df['year'] = float(year)

        for col in [
            f"publisher_{publisher}",
            f"sex_{clean_sex}",
            f"identity_{clean_id}",
            f"alive_{clean_alive}",
            f"eye_{clean_eye}",
            f"hair_{clean_hair}"
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

        st.session_state.history.insert(0, {
            "Publisher": publisher,
            "Debuted": year,
            "Appearances": appearances,
            "Predicted": pred_class,
            "Confidence": f"{probs[pred_idx] * 100:.1f}%"
        })

        with st.container(border=True):
            st.markdown(f"""
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <span style="font-size:0.85rem; color:#8b949e; text-transform:uppercase;">Model Verdict</span>
                        <h2 style="margin:4px 0 0 0;">{pred_class}</h2>
                    </div>
                    <div class="stat-badge {badge_type}">{probs[pred_idx]*100:.1f}% Confidence</div>
                </div>
                <hr style="border-color:#30363d; margin: 15px 0;">
            """, unsafe_allow_html=True)

            st.caption("Probability Distribution")
            for label, prob in zip(CLASSES, probs):
                col_label, col_bar, col_pct = st.columns([1.2, 3, 0.8])
                with col_label:
                    st.write(f"**{label.split()[0]}**")
                with col_bar:
                    st.progress(float(prob))
                with col_pct:
                    st.write(f"{prob * 100:.1f}%")
    else:
        st.info("Adjust the attributes in the profile form and click **⚡ Classify Alignment** to run the prediction.")

# ---------------------------------------------------------
# Prediction Log & Export
# ---------------------------------------------------------
if st.session_state.history:
    st.write("---")
    c_hist1, c_hist2 = st.columns([3, 1])
    with c_hist1:
        st.markdown("#### 📜 Session Prediction History")
    with c_hist2:
        hist_df = pd.DataFrame(st.session_state.history)
        st.download_button(
            label="📥 Export Predictions to CSV",
            data=hist_df.to_csv(index=False).encode('utf-8'),
            file_name="character_alignment_predictions.csv",
            mime="text/csv",
            use_container_width=True
        )
    st.dataframe(hist_df, use_container_width=True, hide_index=True)