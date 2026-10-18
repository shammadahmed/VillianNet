import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="VillainNet Arena | Neural Net Challenge",
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
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Preset Non-Dataset / Third-Party Characters
# ---------------------------------------------------------
PRESET_CHARACTERS = {
    "-- Select a Preset Character or Build Custom --": None,
    "Homelander (The Boys / Dynamite)": {
        "publisher": "Marvel", "sex": "Male Characters", "identity": "Public Identity",
        "alive": "Living Characters", "eye": "Blue Eyes", "hair": "Blonde Hair",
        "appearances": 72, "year": 2006, "actual_align": "Bad Characters"
    },
    "Invincible / Mark Grayson (Image Comics)": {
        "publisher": "Marvel", "sex": "Male Characters", "identity": "Secret Identity",
        "alive": "Living Characters", "eye": "Blue Eyes", "hair": "Black Hair",
        "appearances": 144, "year": 2003, "actual_align": "Good Characters"
    },
    "Spawn / Al Simmons (Image Comics)": {
        "publisher": "DC", "sex": "Male Characters", "identity": "Secret Identity",
        "alive": "Deceased Characters", "eye": "Green Eyes", "hair": "Other",
        "appearances": 350, "year": 1992, "actual_align": "Neutral Characters"
    },
    "Hellboy (Dark Horse)": {
        "publisher": "DC", "sex": "Male Characters", "identity": "Public Identity",
        "alive": "Living Characters", "eye": "Red Eyes", "hair": "Black Hair",
        "appearances": 120, "year": 1993, "actual_align": "Good Characters"
    },
    "Omni-Man / Nolan Grayson (Image Comics)": {
        "publisher": "Marvel", "sex": "Male Characters", "identity": "Public Identity",
        "alive": "Living Characters", "eye": "Brown Eyes", "hair": "Black Hair",
        "appearances": 95, "year": 2003, "actual_align": "Neutral Characters"
    }
}

# ---------------------------------------------------------
# Neural Network Architecture
# ---------------------------------------------------------
class EnhancedVillainNet(nn.Module):
    def __init__(self, in_features=26, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x)

CLASSES = ['Bad Characters', 'Good Characters', 'Neutral Characters']

# ---------------------------------------------------------
# Data Engine & Training
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Training PyTorch Neural Net on Comic Archives...")
def load_and_train():
    dc = pd.read_csv("dc-wikia-data.csv")
    ml = pd.read_csv("marvel-wikia-data.csv")
    
    for d in (dc, ml):
        d.columns = d.columns.str.upper()
    dc['PUBLISHER'], ml['PUBLISHER'] = 'DC', 'Marvel'
    
    df = pd.concat([dc, ml], ignore_index=True)
    df = df[df['ALIGN'].isin(CLASSES)]
    df = df.drop(columns=['PAGE_ID', 'URLSLUG'], errors='ignore')

    def tidy(col, keep):
        s = df[col].fillna('Unknown').str.replace(' Chracters| Eyes| Hair| Identity', '', regex=True)
        return s.where(s.isin(s.value_counts().index[:keep]), 'Other')

    X = pd.get_dummies(pd.DataFrame({
        'publisher': df['PUBLISHER'],
        'sex': tidy('SEX', 2),
        'identity': tidy('ID', 3),
        'alive': tidy('ALIVE', 2),
        'eye': tidy('EYE', 5),
        'hair': tidy('HAIR', 5)
    }), dtype=float)

    X['appearance'] = np.log1p(df['APPEARANCES']).fillna(1)
    median_year = df['YEAR'].median()
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

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=256, shuffle=True)

    torch.manual_seed(42)
    model = EnhancedVillainNet(in_features=len(feature_cols))

    counts = torch.bincount(y_train_t).float()
    weights = counts.sum() / (3 * counts)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)

    best_val, best_state = float('inf'), None
    for _ in range(80):
        model.train()
        for bx, by in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(bx), by)
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

# Session state tracking for accuracy game
if "arena_tests" not in st.session_state:
    st.session_state.arena_tests = []

try:
    model, scaler, feature_cols, median_year = load_and_train()
except Exception as e:
    st.error(f"Dataset error: {e}. Place `dc-wikia-data.csv` and `marvel-wikia-data.csv` in the directory.")
    st.stop()

# ---------------------------------------------------------
# App Interface
# ---------------------------------------------------------
st.markdown('<div class="main-header">🥊 VILLAIN-NET ARENA: TEST THE AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Feed external, custom, or crossover characters to see if the neural network correctly determines their alignment.</div>', unsafe_allow_html=True)

# Preset Selector
selected_preset = st.selectbox("🎯 Quick Load an External Character:", list(PRESET_CHARACTERS.keys()))
default_vals = PRESET_CHARACTERS.get(selected_preset) or {
    "name": "Omni-Man",
    "publisher": "Marvel",
    "sex": "Male Characters",
    "identity": "Secret Identity",
    "alive": "Living Characters",
    "eye": "Blue Eyes",
    "hair": "Black Hair",
    "appearances": 50,
    "year": 2000,
    "actual_align": "Good Characters"
}

col_form, col_arena = st.columns([1.1, 1.2], gap="large")

with col_form:
    st.markdown("### 📋 Character Input & Ground Truth")
    with st.form("challenger_form"):
        char_name = st.text_input("Character Name", value=selected_preset.split(" (")[0] if default_vals else "Invincible")
        
        c_align1, c_align2 = st.columns(2)
        with c_align1:
            actual_alignment = st.selectbox(
                "Known True Alignment (Ground Truth)",
                CLASSES,
                index=CLASSES.index(default_vals.get("actual_align", "Good Characters"))
            )
        with c_align2:
            publisher = st.radio(
                "Closest Universe Style",
                ["DC", "Marvel"],
                index=0 if default_vals.get("publisher") == "DC" else 1,
                horizontal=True
            )

        c1, c2 = st.columns(2)
        with c1:
            sex_opts = ["Male Characters", "Female Characters", "Other"]
            sex = st.selectbox("Sex", sex_opts, index=sex_opts.index(default_vals.get("sex", "Male Characters")))
            
            id_opts = ["Secret Identity", "Public Identity", "No Dual Identity", "Other"]
            identity = st.selectbox("Identity Type", id_opts, index=id_opts.index(default_vals.get("identity", "Secret Identity")))
            
            alive_opts = ["Living Characters", "Deceased Characters", "Other"]
            alive = st.selectbox("Vital Status", alive_opts, index=alive_opts.index(default_vals.get("alive", "Living Characters")))

        with c2:
            eye_opts = ["Blue Eyes", "Brown Eyes", "Black Eyes", "Green Eyes", "Red Eyes", "Other"]
            eye = st.selectbox("Eye Color", eye_opts, index=eye_opts.index(default_vals.get("eye", "Blue Eyes")))
            
            hair_opts = ["Black Hair", "Brown Hair", "Blonde Hair", "Red Hair", "White Hair", "Other"]
            hair = st.selectbox("Hair Color", hair_opts, index=hair_opts.index(default_vals.get("hair", "Black Hair")))
            
            appearances = st.number_input("Comic Appearances Count", min_value=1, max_value=6000, value=int(default_vals.get("appearances", 50)))
            year = st.slider("Debut Year", 1935, 2026, int(default_vals.get("year", 2000)))

        submit_btn = st.form_submit_button("⚡ Run Neural Net Prediction", type="primary", use_container_width=True)

# ---------------------------------------------------------
# Evaluation & Ground-Truth Verification
# ---------------------------------------------------------
with col_arena:
    st.markdown("### 🎯 Model Evaluation")

    if submit_btn:
        clean_sex = sex.replace(" Characters", "")
        clean_id = identity.replace(" Identity", "")
        clean_alive = alive.replace(" Characters", "")
        clean_eye = eye.replace(" Eyes", "")
        clean_hair = hair.replace(" Hair", "")

        input_df = pd.DataFrame(0.0, index=[0], columns=feature_cols)
        input_df['appearance'] = np.log1p(appearances)
        input_df['year'] = float(year)

        for col in [f"publisher_{publisher}", f"sex_{clean_sex}", f"identity_{clean_id}",
                    f"alive_{clean_alive}", f"eye_{clean_eye}", f"hair_{clean_hair}"]:
            if col in input_df.columns:
                input_df[col] = 1.0

        scaled = scaler.transform(input_df)
        tensor_input = torch.tensor(scaled, dtype=torch.float32)

        model.eval()
        with torch.no_grad():
            logits = model(tensor_input)
            probs = torch.softmax(logits, dim=1).numpy()[0]
            pred_idx = int(torch.argmax(logits, dim=1).item())

        predicted_class = CLASSES[pred_idx]
        is_correct = (predicted_class == actual_alignment)

        # Log round in state
        st.session_state.arena_tests.insert(0, {
            "Character": char_name,
            "Actual": actual_alignment,
            "Predicted": predicted_class,
            "Confidence": f"{probs[pred_idx]*100:.1f}%",
            "Result": "✅ Correct" if is_correct else "❌ Missed"
        })

        with st.container(border=True):
            if is_correct:
                st.markdown(f"""
                <div class="verdict-box correct">
                    🎉 CORRECT PREDICTION!<br>
                    The neural net recognized <b>{char_name}</b> as <u>{predicted_class}</u>.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="verdict-box incorrect">
                    ❌ INCORRECT VERDICT!<br>
                    Expected <b>{actual_alignment}</b>, but model predicted <b>{predicted_class}</b>.
                </div>
                """, unsafe_allow_html=True)

            st.write("")
            st.caption("Class Probability Distribution")
            for label, prob in zip(CLASSES, probs):
                c_lbl, c_bar, c_val = st.columns([1.3, 3, 0.8])
                with c_lbl:
                    st.write(f"**{label.split()[0]}**")
                with c_bar:
                    st.progress(float(prob))
                with c_val:
                    st.write(f"{prob * 100:.1f}%")
    else:
        st.info("Select or build an external character, state their known alignment, and click **⚡ Run Neural Net Prediction**.")

# ---------------------------------------------------------
# Scorecard & Session Tracking
# ---------------------------------------------------------
if st.session_state.arena_tests:
    st.write("---")
    st.markdown("### 🏆 Arena Scorecard")
    
    test_df = pd.DataFrame(st.session_state.arena_tests)
    total_tests = len(test_df)
    correct_tests = len(test_df[test_df["Result"] == "✅ Correct"])
    arena_accuracy = (correct_tests / total_tests) * 100

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Challengers Tested", total_tests)
    m2.metric("Model Hits", correct_tests)
    m3.metric("Arena Accuracy", f"{arena_accuracy:.1f}%")

    st.dataframe(test_df, use_container_width=True, hide_index=True)

    csv_data = test_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export Arena Results (CSV)",
        data=csv_data,
        file_name="model_arena_evaluation.csv",
        mime="text/csv"
    )