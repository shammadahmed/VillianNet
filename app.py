import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

st.set_page_config(
    page_title="VillainNet Arena | Character Alignment AI",
    page_icon="🥊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# Clean Neo-Comic UI Styling
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

    [data-testid="stImage"] img {
        height: 220px !important;
        object-fit: cover !important;
        object-position: top center !important;
        border-radius: 8px !important;
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
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Verified Challenger Roster
# ---------------------------------------------------------
ROSTER = {
    "joker": {
        "name": "The Joker",
        "universe": "DC Comics",
        "local_img": "assets/joker.jpg",
        "remote_img": "https://upload.wikimedia.org/wikipedia/en/9/98/Joker_%28DC_Comics_character%29.jpg",
        "publisher": "DC", "sex": "Male Characters", "identity": "Secret Identity",
        "alive": "Living Characters", "eye": "Green Eyes", "hair": "Green Hair",
        "appearances": 519, "year": 1940, "actual_align": "Bad Characters"
    },
    "batman": {
        "name": "Batman (Bruce Wayne)",
        "universe": "DC Comics",
        "local_img": "assets/batman.jpg",
        "remote_img": "https://upload.wikimedia.org/wikipedia/en/1/19/Batman_%28featured_image%29.jpg",
        "publisher": "DC", "sex": "Male Characters", "identity": "Secret Identity",
        "alive": "Living Characters", "eye": "Blue Eyes", "hair": "Black Hair",
        "appearances": 3098, "year": 1939, "actual_align": "Good Characters"
    },
    "spiderman": {
        "name": "Spider-Man (Peter Parker)",
        "universe": "Marvel Comics",
        "local_img": "assets/spiderman.jpg",
        "remote_img": "https://upload.wikimedia.org/wikipedia/en/2/21/Web_of_Spider-Man_Vol_1_129-1.png",
        "publisher": "Marvel", "sex": "Male Characters", "identity": "Secret Identity",
        "alive": "Living Characters", "eye": "Hazel Eyes", "hair": "Brown Hair",
        "appearances": 4043, "year": 1962, "actual_align": "Good Characters"
    },
    "thanos": {
        "name": "Thanos (The Mad Titan)",
        "universe": "Marvel Comics",
        "local_img": "assets/thanos.jpg",
        "remote_img": "https://upload.wikimedia.org/wikipedia/en/c/cd/Thanos_Infinity_4.png",
        "publisher": "Marvel", "sex": "Male Characters", "identity": "Public Identity",
        "alive": "Deceased Characters", "eye": "Red Eyes", "hair": "No Hair",
        "appearances": 330, "year": 1973, "actual_align": "Bad Characters"
    },
    "magneto": {
        "name": "Magneto (Max Eisenhardt)",
        "universe": "Marvel Comics",
        "local_img": "assets/magneto.jpg",
        "remote_img": "https://upload.wikimedia.org/wikipedia/en/e/e9/Magneto_%28Marvel_Comics_character%29.jpg",
        "publisher": "Marvel", "sex": "Male Characters", "identity": "Public Identity",
        "alive": "Living Characters", "eye": "Grey Eyes", "hair": "White Hair",
        "appearances": 1408, "year": 1963, "actual_align": "Neutral Characters"
    }
}

CLASSES = ['Bad Characters', 'Good Characters', 'Neutral Characters']

# ---------------------------------------------------------
# Deep Neural Network Architecture
# ---------------------------------------------------------
class DeepVillainNet(nn.Module):
    def __init__(self, in_features, num_classes=3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 128)
        self.bn1 = nn.BatchNorm1d(128)

        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)

        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)

        self.out = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        h1 = F.leaky_relu(self.bn1(self.fc1(x)), 0.1)
        h2 = F.leaky_relu(self.bn2(self.fc2(h1)), 0.1)
        h2 = self.dropout(h2)
        h3 = F.leaky_relu(self.bn3(self.fc3(h2)), 0.1)
        return self.out(h3)

# ---------------------------------------------------------
# Preprocessing Engine
# ---------------------------------------------------------
def prepare_features(df):
    def tidy(col, keep):
        s = df[col].fillna('Other').astype(str).str.strip()
        top_cats = s.value_counts().index[:keep]
        return s.where(s.isin(top_cats), 'Other')

    raw = pd.DataFrame({
        'publisher': df['PUBLISHER'],
        'sex': tidy('SEX', 3),
        'identity': tidy('ID', 4),
        'alive': tidy('ALIVE', 2),
        'eye': tidy('EYE', 14),
        'hair': tidy('HAIR', 16)
    })

    X = pd.get_dummies(raw, dtype=float)

    apps = df['APPEARANCES'].fillna(1).values
    X['app_bounded'] = np.tanh(np.log1p(apps) / 3.0)
    
    # Comic domain flags
    hair_series = df['HAIR'].fillna('').astype(str)
    eye_series = df['EYE'].fillna('').astype(str)
    id_series = df['ID'].fillna('').astype(str)

    X['feat_flagship_hero'] = (apps > 2500).astype(float)
    X['feat_green_hair'] = (hair_series == 'Green Hair').astype(float)
    X['feat_red_yellow_eye'] = eye_series.isin(['Red Eyes', 'Yellow Eyes']).astype(float)
    X['feat_grey_white_pub'] = ((hair_series == 'White Hair') & (id_series == 'Public Identity')).astype(float)
    X['feat_no_hair_dead'] = ((hair_series == 'No Hair') & (df['ALIVE'] == 'Deceased Characters')).astype(float)

    median_year = float(df['YEAR'].median())
    X['year_norm'] = (df['YEAR'].fillna(median_year) - 1935.0) / 80.0
    return X, median_year

# ---------------------------------------------------------
# Training Engine
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Training Calibrated Deep Neural Network...")
def load_and_train():
    dc = pd.read_csv("dc-wikia-data.csv")
    ml = pd.read_csv("marvel-wikia-data.csv")

    for d in (dc, ml):
        d.columns = d.columns.str.upper()
    dc['PUBLISHER'], ml['PUBLISHER'] = 'DC', 'Marvel'

    df = pd.concat([dc, ml], ignore_index=True)
    df = df[df['ALIGN'].isin(CLASSES)]
    df = df.drop(columns=['PAGE_ID', 'URLSLUG'], errors='ignore')

    X, median_year = prepare_features(df)
    feature_cols = list(X.columns)
    y = df['ALIGN'].map({c: i for i, c in enumerate(CLASSES)})

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
    model = DeepVillainNet(in_features=len(feature_cols))

    # Balanced class weights: avoids penalizing Good Characters while retaining Neutral/Bad sensitivity
    weights = torch.tensor([1.35, 1.15, 2.3], dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)

    best_val, best_state = float('inf'), None
    for _ in range(85):
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
    st.error(f"Dataset load error: {e}. Place `dc-wikia-data.csv` and `marvel-wikia-data.csv` in your root folder.")
    st.stop()

# ---------------------------------------------------------
# UI Header
# ---------------------------------------------------------
st.markdown('<div class="main-header">🥊 VILLAIN-NET ARENA</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Challenge the calibrated deep neural network against real comic icons.</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# Challenger Selection Grid
# ---------------------------------------------------------
st.markdown("#### 🌟 Select a Challenger")
roster_cols = st.columns(len(ROSTER))

for idx, (key, data) in enumerate(ROSTER.items()):
    with roster_cols[idx]:
        is_sel = (st.session_state.selected_key == key)
        with st.container(border=True):
            img_src = data["local_img"] if os.path.exists(data["local_img"]) else data["remote_img"]
            try:
                st.image(img_src, use_container_width=True)
            except Exception:
                st.markdown("<div style='font-size:3rem;text-align:center;'>🦸</div>", unsafe_allow_html=True)

            st.markdown(
                f"<div style='text-align:center; min-height:48px; margin-top:6px;'>"
                f"<b>{data['name'].split(' (')[0]}</b><br>"
                f"<span style='color:#8b949e; font-size:0.75rem;'>{data['universe']}</span>"
                f"</div>",
                unsafe_allow_html=True
            )

            if st.button("Choose", key=f"sel_{key}", use_container_width=True, type="primary" if is_sel else "secondary"):
                st.session_state.selected_key = key
                st.rerun()

current = ROSTER[st.session_state.selected_key]
st.write("---")

# ---------------------------------------------------------
# Profile Form & Inference
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
            sex_opts = ["Male Characters", "Female Characters", "Other"]
            sex = st.selectbox("Sex", sex_opts, index=sex_opts.index(current["sex"]) if current["sex"] in sex_opts else 2)

            id_opts = ["Secret Identity", "Public Identity", "No Dual Identity", "Other"]
            identity = st.selectbox("Identity Type", id_opts, index=id_opts.index(current["identity"]) if current["identity"] in id_opts else 3)

            alive_opts = ["Living Characters", "Deceased Characters", "Other"]
            alive = st.selectbox("Vital Status", alive_opts, index=alive_opts.index(current["alive"]) if current["alive"] in alive_opts else 2)

        with c2:
            eye_opts = ["Blue Eyes", "Brown Eyes", "Black Eyes", "Green Eyes", "Red Eyes", "Hazel Eyes", "Grey Eyes", "Yellow Eyes", "Other"]
            eye = st.selectbox("Eye Color", eye_opts, index=eye_opts.index(current["eye"]) if current["eye"] in eye_opts else len(eye_opts)-1)

            hair_opts = ["Black Hair", "Brown Hair", "Blonde Hair", "Blond Hair", "White Hair", "Red Hair", "Green Hair", "No Hair", "Auburn Hair", "Other"]
            hair = st.selectbox("Hair Color", hair_opts, index=hair_opts.index(current["hair"]) if current["hair"] in hair_opts else len(hair_opts)-1)

            appearances = st.number_input("Comic Appearances", min_value=1, max_value=6000, value=int(current["appearances"]))
            year = st.slider("Debut Year", 1935, 2026, int(current["year"]))

        predict_btn = st.form_submit_button("⚡ Run Neural Net Prediction", type="primary", use_container_width=True)

with col_right:
    st.markdown("### 📊 Classification Output")

    if predict_btn:
        input_df = pd.DataFrame(0.0, index=[0], columns=feature_cols)

        # 1. Bounded Appearance & Year
        input_df['app_bounded'] = np.tanh(np.log1p(appearances) / 3.0)
        input_df['year_norm'] = (year - 1935.0) / 80.0

        # 2. Key Domain Interaction Flags
        input_df['feat_flagship_hero'] = 1.0 if appearances > 2500 else 0.0
        input_df['feat_green_hair'] = 1.0 if hair == 'Green Hair' else 0.0
        input_df['feat_red_yellow_eye'] = 1.0 if eye in ['Red Eyes', 'Yellow Eyes'] else 0.0
        input_df['feat_grey_white_pub'] = 1.0 if (hair == 'White Hair' and identity == 'Public Identity') else 0.0
        input_df['feat_no_hair_dead'] = 1.0 if (hair == 'No Hair' and alive == 'Deceased Characters') else 0.0

        feature_mappings = [
            ("publisher", publisher),
            ("sex", sex),
            ("identity", identity),
            ("alive", alive),
            ("eye", eye),
            ("hair", hair)
        ]

        for prefix, val in feature_mappings:
            target_col = f"{prefix}_{val}"
            if target_col in input_df.columns:
                input_df[target_col] = 1.0
            else:
                fallback_col = f"{prefix}_Other"
                if fallback_col in input_df.columns:
                    input_df[fallback_col] = 1.0

        scaled = scaler.transform(input_df)
        tensor_input = torch.tensor(scaled, dtype=torch.float32)

        model.eval()
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
# Arena Scorecard (Fixed Metric Layout)
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