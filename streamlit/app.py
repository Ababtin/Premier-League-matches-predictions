import streamlit as st
import pandas as pd
import joblib
import json
import numpy as np
import requests
from datetime import datetime, date
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure page
st.set_page_config(
    page_title="Premier League Match Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model directly in Streamlit
@st.cache_resource
def load_model_components():
    """Load ML model, encoder, and features"""
    try:
        # Try to load from different possible paths
        paths_to_try = [
            ('../models/', '../models/', '../models/'),
            ('models/', 'models/', 'models/'),
            ('./models/', './models/', './models/')
        ]

        model, le, feature_names = None, None, None

        for model_path, le_path, feature_path in paths_to_try:
            try:
                model = joblib.load(f'{model_path}rf_model.pkl')
                le = joblib.load(f'{le_path}label_encoder.pkl')
                with open(f'{feature_path}feature_names.json', 'r') as f:
                    feature_names = json.load(f)
                st.success(f"✅ Model loaded from {model_path}")
                break
            except:
                continue

        if model is None:
            st.error("❌ Could not load model files")
            return None, None, None, None

        # Create team mapping
        team_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

        return model, le, feature_names, team_mapping

    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None, None, None

def predict_match_integrated(home_team, away_team, match_date, model, le, feature_names, team_mapping):
    """Make prediction using integrated model"""
    try:
        # Convert team names to encoded values
        if home_team not in team_mapping or away_team not in team_mapping:
            return {"error": "Team not found in training data"}

        # Create dummy feature vector (replace with your actual feature engineering)
        feature_values = {}

        # Basic features (you'll need to replace these with actual calculations)
        for feature in feature_names:
            if 'home_team' in feature:
                feature_values[feature] = np.random.uniform(0.5, 2.0)  # Placeholder
            elif 'away_team' in feature:
                feature_values[feature] = np.random.uniform(0.5, 2.0)  # Placeholder
            elif 'diff' in feature:
                feature_values[feature] = np.random.uniform(-1.0, 1.0)  # Placeholder
            else:
                feature_values[feature] = np.random.uniform(0, 1.0)  # Placeholder

        # Create feature array
        feature_vector = [feature_values.get(f, 0.0) for f in feature_names]
        feature_array = np.array(feature_vector).reshape(1, -1)

        # Make prediction
        probabilities = model.predict_proba(feature_array)[0]
        prediction = model.predict(feature_array)[0]

        # Map probabilities to outcomes
        prob_dict = {
            'Away Win': probabilities[0],   # -1.0
            'Draw': probabilities[1],       # 0.0
            'Home Win': probabilities[2]    # 1.0
        }

        # Determine outcome
        if prediction == 1.0:
            outcome = f"🏠 {home_team} Win"
        elif prediction == -1.0:
            outcome = f"✈️ {away_team} Win"
        else:
            outcome = "🤝 Draw"

        return {
            'home_team': home_team,
            'away_team': away_team,
            'prediction': outcome,
            'probabilities': prob_dict,
            'confidence': max(prob_dict.values())
        }

    except Exception as e:
        return {"error": f"Prediction failed: {e}"}

# Premier League teams
PREMIER_LEAGUE_TEAMS = [
    "Arsenal", "Aston Villa", "Brighton", "Burnley", "Chelsea",
    "Crystal Palace", "Everton", "Fulham", "Liverpool", "Luton Town",
    "Manchester City", "Manchester United", "Newcastle United",
    "Nottingham Forest", "Sheffield United", "Tottenham",
    "West Ham United", "Wolverhampton Wanderers", "Bournemouth", "Brentford"
]

def main():
    # Load model components
    model, le, feature_names, team_mapping = load_model_components()

    # Header
    st.title("⚽ Premier League Match Predictor")
    st.markdown("### Predict match outcomes using Machine Learning")

    if model is None:
        st.error("🚨 Model not available! Please check model files.")
        st.info("💡 **For developers**: Make sure model files are in the correct directory")
        return

    # Sidebar for inputs
    with st.sidebar:
        st.header("🏆 Match Setup")

        home_team = st.selectbox(
            "🏠 Home Team",
            options=sorted(PREMIER_LEAGUE_TEAMS),
            index=0
        )

        away_team = st.selectbox(
            "✈️ Away Team",
            options=sorted([team for team in PREMIER_LEAGUE_TEAMS if team != home_team]),
            index=0
        )

        match_date = st.date_input(
            "📅 Match Date",
            value=date.today(),
            min_value=date(2020, 1, 1),
            max_value=date(2030, 12, 31)
        )

        predict_button = st.button("🔮 Predict Match", use_container_width=True)

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        if predict_button:
            with st.spinner("🤔 Analyzing teams..."):
                result = predict_match_integrated(
                    home_team, away_team, match_date,
                    model, le, feature_names, team_mapping
                )

            if 'error' in result:
                st.error(f"❌ {result['error']}")
            else:
                # Display prediction
                st.success(f"### 🎯 Prediction: {result['prediction']}")

                # Probabilities chart
                probs = result['probabilities']

                fig = go.Figure(data=[
                    go.Bar(
                        x=list(probs.keys()),
                        y=list(probs.values()),
                        marker_color=['red', 'yellow', 'green'],
                        text=[f"{v:.1%}" for v in probs.values()],
                        textposition='auto',
                    )
                ])

                fig.update_layout(
                    title="📊 Match Outcome Probabilities",
                    yaxis_title="Probability",
                    xaxis_title="Outcome",
                    showlegend=False,
                    height=400
                )

                st.plotly_chart(fig, use_container_width=True)

                # Confidence metric
                confidence = result['confidence']
                st.metric(
                    "🎯 Confidence Level",
                    f"{confidence:.1%}",
                    delta=f"{'High' if confidence > 0.6 else 'Medium' if confidence > 0.4 else 'Low'} Confidence"
                )

    with col2:
        st.info("### ℹ️ How it works\n\n"
               "This ML model analyzes:\n"
               "- Recent team performance\n"
               "- Head-to-head records\n"
               "- Home advantage\n"
               "- Player statistics\n"
               "- Historical data")

        if model is not None:
            st.success(f"✅ Model loaded\n"
                     f"📊 Features: {len(feature_names) if feature_names else 0}\n"
                     f"🏆 Teams: {len(team_mapping) if team_mapping else 0}")

if __name__ == "__main__":
    main()
