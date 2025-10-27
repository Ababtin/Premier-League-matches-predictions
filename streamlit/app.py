import streamlit as st
import requests
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date
import os
from pathlib import Path

# =========================
# Config
# =========================
API_URL = os.getenv('API_URL', 'http://localhost:8000')
API_URL = "http://localhost:8000"
ASSETS_DIR = Path(__file__).parent / "assets" / "logos"  # assets/logos/*.png
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# High-res PNG fallbacks (transparent if possible)
# You can replace any of these with your own PNGs in assets/logos to override.
PNG_FALLBACKS = {
    "Arsenal": "https://upload.wikimedia.org/wikipedia/en/thumb/5/53/Arsenal_FC.svg/240px-Arsenal_FC.svg.png",
    "Aston Villa": "https://upload.wikimedia.org/wikipedia/en/thumb/9/9f/Aston_Villa_FC_crest_%282016%29.svg/240px-Aston_Villa_FC_crest_%282016%29.svg.png",
    "Bournemouth": "https://upload.wikimedia.org/wikipedia/en/thumb/e/e5/AFC_Bournemouth_%282013%29.svg/240px-AFC_Bournemouth_%282013%29.svg.png",
    "Brentford": "https://upload.wikimedia.org/wikipedia/en/thumb/2/2a/Brentford_FC_crest.svg/240px-Brentford_FC_crest.svg.png",
    "Brighton": "https://upload.wikimedia.org/wikipedia/en/thumb/f/fd/Brighton_%26_Hove_Albion_logo.svg/240px-Brighton_%26_Hove_Albion_logo.svg.png",
    "Burnley": "https://upload.wikimedia.org/wikipedia/en/thumb/6/62/Burnley_F.C._Logo.svg/240px-Burnley_F.C._Logo.svg.png",
    "Chelsea": "https://upload.wikimedia.org/wikipedia/en/thumb/c/cc/Chelsea_FC.svg/240px-Chelsea_FC.svg.png",
    "Crystal Palace": "https://upload.wikimedia.org/wikipedia/en/thumb/0/0c/Crystal_Palace_FC_logo.svg/240px-Crystal_Palace_FC_logo.svg.png",
    "Everton": "https://upload.wikimedia.org/wikipedia/en/thumb/7/7c/Everton_FC_logo.svg/240px-Everton_FC_logo.svg.png",
    "Fulham": "https://upload.wikimedia.org/wikipedia/en/thumb/3/3e/Fulham_FC_%28shield%29.svg/240px-Fulham_FC_%29shield%29.svg.png",
    "Leeds United": "https://upload.wikimedia.org/wikipedia/en/thumb/8/81/Leeds_United_F.C._logo.svg/240px-Leeds_United_F.C._logo.svg.png",
    "Leicester City": "https://upload.wikimedia.org/wikipedia/en/thumb/2/2d/Leicester_City_crest.svg/240px-Leicester_City_crest.svg.png",
    "Liverpool": "https://upload.wikimedia.org/wikipedia/en/thumb/0/0c/Liverpool_FC.svg/240px-Liverpool_FC.svg.png",
    "Luton Town": "https://upload.wikimedia.org/wikipedia/en/thumb/5/5a/Luton_Town_FC.svg/240px-Luton_Town_FC.svg.png",
    "Man City": "https://upload.wikimedia.org/wikipedia/en/thumb/e/eb/Manchester_City_FC_badge.svg/240px-Manchester_City_FC_badge.svg.png",
    "Man United": "https://upload.wikimedia.org/wikipedia/en/thumb/7/7a/Manchester_United_FC_crest.svg/240px-Manchester_United_FC_crest.svg.png",
    "Newcastle": "https://upload.wikimedia.org/wikipedia/en/thumb/5/56/Newcastle_United_Logo.svg/240px-Newcastle_United_Logo.svg.png",
    "Norwich": "https://upload.wikimedia.org/wikipedia/en/thumb/6/6c/Norwich_City.svg/240px-Norwich_City.svg.png",
    "Nottingham Forest": "https://upload.wikimedia.org/wikipedia/en/thumb/7/79/Nottingham_Forest_logo.svg/240px-Nottingham_Forest_logo.svg.png",
    "Sheffield United": "https://upload.wikimedia.org/wikipedia/en/thumb/9/9c/Sheffield_United_FC_logo.svg/240px-Sheffield_United_FC_logo.svg.png",
    "Southampton": "https://upload.wikimedia.org/wikipedia/en/thumb/c/c9/FC_Southampton.svg/240px-FC_Southampton.svg.png",
    "Stoke": "https://upload.wikimedia.org/wikipedia/en/thumb/2/29/Stoke_City_FC.svg/240px-Stoke_City_FC.svg.png",
    "Sunderland": "https://upload.wikimedia.org/wikipedia/en/thumb/7/77/Sunderland.svg/240px-Sunderland.svg.png",
    "Swansea": "https://upload.wikimedia.org/wikipedia/en/thumb/a/ab/Swansea_City_AFC_logo.svg/240px-Swansea_City_AFC_logo.svg.png",
    "Tottenham": "https://upload.wikimedia.org/wikipedia/en/thumb/b/b4/Tottenham_Hotspur.svg/240px-Tottenham_Hotspur.svg.png",
    "Watford": "https://upload.wikimedia.org/wikipedia/en/thumb/e/e2/Watford.svg/240px-Watford.svg.png",
    "West Brom": "https://upload.wikimedia.org/wikipedia/en/thumb/8/8b/West_Bromwich_Albion.svg/240px-West_Bromwich_Albion.svg.png",
    "West Ham": "https://upload.wikimedia.org/wikipedia/en/thumb/c/c2/West_Ham_United_FC_logo.svg/240px-West_Ham_United_FC_logo.svg.png",
    "Wigan": "https://upload.wikimedia.org/wikipedia/en/thumb/4/43/Wigan_Athletic.svg/240px-Wigan_Athletic.svg.png",
    "Wolves": "https://upload.wikimedia.org/wikipedia/en/thumb/f/fc/Wolverhampton_Wanderers.svg/240px-Wolverhampton_Wanderers.svg.png",
}

PREMIER_LEAGUE_LOGO = "premier-league.png"   # place in assets/logos/
PREMIER_LEAGUE_LOGO_FALLBACK = "https://upload.wikimedia.org/wikipedia/en/thumb/f/f2/Premier_League_Logo.svg/240px-Premier_League_Logo.svg.png"

def local_logo_path(team: str) -> Path:
    # normalize file name: "manchester united" -> "manchester-united.png"
    fname = team.lower().replace("&", "and").replace(" ", "-") + ".png"
    return ASSETS_DIR / fname

def get_logo_src(team: str) -> str:
    """Return local PNG if present, else high-res PNG URL."""
    p = local_logo_path(team)
    if p.exists():
        return str(p.as_posix())
    return PNG_FALLBACKS.get(team, PREMIER_LEAGUE_LOGO_FALLBACK)

def get_pl_logo_src() -> str:
    p = ASSETS_DIR / PREMIER_LEAGUE_LOGO
    return str(p.as_posix()) if p.exists() else PREMIER_LEAGUE_LOGO_FALLBACK

# =========================
# Page config & styles
# =========================
st.set_page_config(
    page_title="Premier League Match Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .team-vs {
        font-size: 2rem;
        font-weight: bold;
        color: #e9eef5;
        text-align: center;
        margin: 1rem 0;
    }
    /* Top-left PL logo */
    .pl-logo {
        position: fixed;
        top: 10px;
        left: 16px;
        width: 60px;
        z-index: 9999;
        filter: drop-shadow(0 0 6px rgba(0,0,0,0.35));
    }
    /* Subtle lift for all images */
    img {
        image-rendering: -webkit-optimize-contrast;
        image-rendering: crisp-edges;
        filter: drop-shadow(0 0 4px rgba(0,0,0,0.25));
    }
</style>
""", unsafe_allow_html=True)

# Top-left Premier League logo (local if present)
st.markdown(
    f"<img class='pl-logo' src='{get_pl_logo_src()}' />",
    unsafe_allow_html=True
)

# =========================
# API helpers
# =========================
def get_teams():
    try:
        response = requests.get(f"{API_URL}/teams")
        if response.status_code == 200:
            return response.json()['teams']
        else:
            st.error(f"Failed to fetch teams: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
        return []

def check_api_health():
    try:
        response = requests.get(f"{API_URL}/health")
        return response.status_code == 200
    except:
        return False

def predict_match(home_team, away_team, match_date):
    try:
        payload = {
            "home_team": home_team,
            "away_team": away_team,
            "match_date": match_date.strftime('%Y-%m-%d')
        }
        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Prediction failed: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None

# =========================
# Charts
# =========================
def create_probability_chart(probabilities):
    outcomes = list(probabilities.keys())
    probs = [probabilities[o] * 100 for o in outcomes]
    colors = ['#ff6b6b', '#feca57', '#48dbfb']
    fig = go.Figure(data=[
        go.Bar(x=outcomes, y=probs, marker_color=colors, text=[f'{p:.1f}%' for p in probs], textposition='auto')
    ])
    fig.update_layout(
        title="Match Outcome Probabilities",
        xaxis_title="Outcome", yaxis_title="Probability (%)",
        showlegend=False, height=400,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def create_confidence_gauge(confidence):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence * 100,
        domain={'x':[0,1], 'y':[0,1]},
        title={'text':"Prediction Confidence"},
        delta={'reference':50},
        gauge={
            'axis': {'range':[None,100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range':[0,25], 'color':"lightgray"},
                {'range':[25,50], 'color':"gray"},
                {'range':[50,75], 'color':"lightgreen"},
                {'range':[75,100], 'color':"green"},
            ],
            'threshold': {'line':{'color':"red",'width':4}, 'thickness':0.75, 'value':90}
        }
    ))
    fig.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)')
    return fig

# =========================
# App
# =========================
def main():
    st.markdown('<h1 class="main-header">⚽ Premier League Match Predictor</h1>', unsafe_allow_html=True)

    if not check_api_health():
        st.error("🚨 API is not available! Please make sure the FastAPI server is running.")
        st.info("Run: `docker-compose up` or `python api/main.py`")
        return
    st.success("✅ Connected to API successfully!")

    st.sidebar.header("🎯 Match Prediction Settings")
    teams = get_teams()
    if not teams:
        st.error("Could not load teams from API")
        return

    col1, col2 = st.sidebar.columns(2)
    with col1:
        home_team = st.selectbox("🏠 Home Team", teams, index=teams.index("Arsenal") if "Arsenal" in teams else 0)
        st.image(get_logo_src(home_team), width=64)
    with col2:
        away_team = st.selectbox("✈️ Away Team", teams, index=teams.index("Chelsea") if "Chelsea" in teams else (1 if len(teams)>1 else 0))
        st.image(get_logo_src(away_team), width=64)

    match_date = st.sidebar.date_input("📅 Match Date", value=date.today(), min_value=date(2000,1,1), max_value=date(2030,12,31))
    predict_button = st.sidebar.button("🎯 Predict Match!", type="primary")

    if home_team == away_team:
        st.warning("⚠️ Please select different teams for home and away.")
        return

    # VS banner with crisp PNGs
    st.markdown(f"""
    <div style='text-align:center; margin-top:8px;'>
        <img src='{get_logo_src(home_team)}' width='96' style='vertical-align:middle; margin-right:16px;'>
        <b style='font-size:2rem;'>{home_team}</b>
        <span style='font-size:2rem; margin:0 20px;'>🆚</span>
        <b style='font-size:2rem;'>{away_team}</b>
        <img src='{get_logo_src(away_team)}' width='96' style='vertical-align:middle; margin-left:16px;'>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"**📅 Match Date:** {match_date.strftime('%B %d, %Y')}")

    if predict_button:
        with st.spinner("🤖 Making prediction..."):
            prediction = predict_match(home_team, away_team, match_date)
            if prediction:
                st.session_state.prediction = prediction

    if 'prediction' in st.session_state:
        pred = st.session_state.prediction
        st.markdown(f"""
        <div class="prediction-card">
            <h2>🎯 Prediction Result</h2>
            <h1>{pred['prediction']}</h1>
            <p>Confidence: {pred['confidence']*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        probs = pred['probabilities']
        with c1:
            st.metric("🏠 Home Win", f"{probs['Home Win']*100:.1f}%",
                      delta=f"{(probs['Home Win']-0.33)*100:+.1f}%" if probs['Home Win'] != 0.33 else None)
        with c2:
            st.metric("🤝 Draw", f"{probs['Draw']*100:.1f}%",
                      delta=f"{(probs['Draw']-0.33)*100:+.1f}%" if probs['Draw'] != 0.33 else None)
        with c3:
            st.metric("✈️ Away Win", f"{probs['Away Win']*100:.1f}%",
                      delta=f"{(probs['Away Win']-0.33)*100:+.1f}%" if probs['Away Win'] != 0.33 else None)

        g1, g2 = st.columns(2)
        with g1:
            st.plotly_chart(create_probability_chart(probs), use_container_width=True)
        with g2:
            st.plotly_chart(create_confidence_gauge(pred['confidence']), use_container_width=True)

        with st.expander("📊 Detailed Prediction Information"):
            st.json(pred)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 About the Model")
    st.sidebar.info("""
    This predictor uses a Random Forest model trained on:
    - 32+ engineered features
    - Historical Premier League data
    - Team statistics, Elo ratings, form, etc.
    """)

    st.sidebar.markdown("### 🔗 API Endpoints")
    st.sidebar.code(f"""
    Health: {API_URL}/health
    Teams: {API_URL}/teams
    Predict: {API_URL}/predict
    Docs: {API_URL}/docs
    """)

if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import joblib
import json
import numpy as np
from datetime import datetime

@st.cache_resource
def load_model():
    try:
        # Load model directly in Streamlit
        model = joblib.load('../models/rf_model.pkl')
        le = joblib.load('../models/label_encoder.pkl')
        with open('../models/feature_names.json', 'r') as f:
            feature_names = json.load(f)
        return model, le, feature_names
    except:
        st.error("Model files not found!")
        return None, None, None

def predict_match_local(home_team, away_team, model, le, feature_names):
    # Your prediction logic here (copy from your API)
    # ... prediction code ...
    return {"prediction": "Home Win", "probabilities": {"Home": 0.6, "Draw": 0.3, "Away": 0.1}}

# Main app
st.title("⚽ Premier League Match Predictor")

model, le, feature_names = load_model()

if model is not None:
    home_team = st.selectbox("Home Team", ["Arsenal", "Chelsea", "Liverpool"])
    away_team = st.selectbox("Away Team", ["Arsenal", "Chelsea", "Liverpool"])

    if st.button("Predict Match"):
        result = predict_match_local(home_team, away_team, model, le, feature_names)
        st.success(f"Prediction: {result['prediction']}")
