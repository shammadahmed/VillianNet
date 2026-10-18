import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Comic Character Alignment Predictor", page_icon="🦸", layout="wide")

# ---------------------------------------------------------
# 1. Model Definition & Pipeline Setup
# ---------------------------------------------------------
class VillainNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(26, 32)
        self.output = nn.Linear(32, 3)

    def forward(self, x):
        return self.output(torch.relu(self.hidden(x)))

CLASSES = ['Bad Characters', 'Good Characters', 'Neutral Characters']

@st.cache_resource(show_spinner="Training VillainNet model...")
def load_and_train(dc_path: str = "dc-wikia-data.csv", ml_path: str = "marvel-wikia-data.csv"):
    # Load and concatenate datasets
    dc = pd.read_csv(dc_path)
    ml = pd.read_csv(ml_path)
    
    for d in (dc, ml):
        d.columns = d.columns.str.upper()
    dc['PUBLISHER'], ml['PUBLISHER'] = 'DC', 'Marvel'
    
    df = pd.concat([dc, ml], ignore_index=True)
    df = df[df['ALIGN'].isin(CLASSES)]
    df = df.drop(columns=['PAGE_ID', 'URLSLUG'], errors='ignore')

    # Preprocessing helpers
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

    # Train / Val / Test Splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.long)
    X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_t = torch.tensor(y_val.values, dtype=torch.long)

    # Training
    torch.manual_seed(42)
    model = VillainNet()
    counts = torch.bincount(y_train_t).float()
    weights = counts.sum() / (3 * counts)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    best_val, best_state, patience, bad = float('inf'), None, 50, 0
    for _ in range(1000):
        model.train()
        loss = loss_fn(model(X_train_t), y_train_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val_t), y_val_t).item()
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()

    return model, scaler, feature_cols, median_year

# ---------------------------------------------------------
# 2. UI Layout
# ---------------------------------------------------------
st.title("🦸 Comic Character Alignment Predictor")
st.markdown("Predict whether a comic character is **Good**, **Bad**, or **Neutral** based on physical traits and publishing attributes using the trained `VillainNet` model.")

# File selection / Data source sidebar
st.sidebar.header("Data Paths")
dc_csv = st.sidebar.text_input("DC CSV Path", value="dc-wikia-data.csv")
marvel_csv = st.sidebar.text_input("Marvel CSV Path", value="marvel-wikia-data.csv")

try:
    model, scaler, feature_cols, median_year = load_and_train(dc_csv, marvel_csv)
    st.sidebar.success("Model loaded and cached successfully!")
except Exception as e:
    st.error(f"Error loading datasets or training model: {e}")
    st.info("Make sure `dc-wikia-data.csv` and `marvel-wikia-data.csv` are in the working directory or enter the valid file paths in the sidebar.")
    st.stop()

# Form Inputs
col1, col2 = st.columns(2)

with col1:
    publisher = st.selectbox("Publisher", ["DC", "Marvel"])
    sex = st.selectbox("Sex", ["Male Characters", "Female Characters", "Other"])
    identity = st.selectbox("Identity", ["Secret Identity", "Public Identity", "No Dual Identity", "Other"])
    alive = st.selectbox("Status", ["Living Characters", "Deceased Characters", "Other"])

with col2:
    eye = st.selectbox("Eye Color", ["Blue Eyes", "Brown Eyes", "Black Eyes", "Green Eyes", "Red Eyes", "Other"])
    hair = st.selectbox("Hair Color", ["Black Hair", "Brown Hair", "Blonde Hair", "Red Hair", "White Hair", "Other"])
    appearances = st.number_input("Appearances Count", min_value=1, max_value=5000, value=25, step=1)
    year = st.number_input("First Appearance Year", min_value=1930, max_value=2026, value=int(median_year), step=1)

# ---------------------------------------------------------
# 3. Inference
# ---------------------------------------------------------
if st.button("Predict Alignment", type="primary"):
    # Normalize string inputs to match notebook tidy rules
    clean_sex = sex.replace(" Characters", "")
    clean_id = identity.replace(" Identity", "")
    clean_alive = alive.replace(" Characters", "")
    clean_eye = eye.replace(" Eyes", "")
    clean_hair = hair.replace(" Hair", "")

    # Construct one-hot feature vector matching trained columns exactly
    input_df = pd.DataFrame(0.0, index=[0], columns=feature_cols)

    input_df['appearance'] = np.log1p(appearances)
    input_df['year'] = float(year)

    for col in [f"publisher_{publisher}", f"sex_{clean_sex}", f"identity_{clean_id}",
                f"alive_{clean_alive}", f"eye_{clean_eye}", f"hair_{clean_hair}"]:
        if col in input_df.columns:
            input_df[col] = 1.0

    # Scale and predict
    scaled_features = scaler.transform(input_df)
    tensor_input = torch.tensor(scaled_features, dtype=torch.float32)

    with torch.no_grad():
        logits = model(tensor_input)
        probs = torch.softmax(logits, dim=1).numpy()[0]
        pred_idx = int(torch.argmax(logits, dim=1).item())

    pred_label = CLASSES[pred_idx]

    st.subheader(f"Prediction: **{pred_label}**")
    
    # Probability breakdown
    prob_df = pd.DataFrame({
        "Alignment": CLASSES,
        "Probability": probs
    }).set_index("Alignment")

    st.bar_chart(prob_df)