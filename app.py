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