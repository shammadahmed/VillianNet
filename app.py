import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="VillainNet | Comic Character Alignment AI",
    page_icon="⚡",
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

# ---------------------------------------------------------
# Model Definition & Data Engine
# ---------------------------------------------------------
class VillainNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(26, 32)
        self.output = nn.Linear(32, 3)

    def forward(self, x):
        return self.output(torch.relu(self.hidden(x)))

CLASSES = ['Bad Character', 'Good Character', 'Neutral Character']

@st.cache_resource(show_spinner="Training VillainNet neural network...")
def load_and_train():
    dc = pd.read_csv("dc-wikia-data.csv")
    ml = pd.read_csv("marvel-wikia-data.csv")
    
    for d in (dc, ml):
        d.columns = d.columns.str.upper()
    dc['PUBLISHER'], ml['PUBLISHER'] = 'DC', 'Marvel'
    
    df = pd.concat([dc, ml], ignore_index=True)
    df =