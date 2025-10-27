"""
Match predictor class that loads the trained model and makes predictions
"""
import joblib
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import os
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MatchPredictor:
    """Handles loading the trained model and making predictions"""

    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.feature_names = None
        self.team_mapping = None
        self.historical_data = None
        self._loaded = False

        # Define model paths - go up to project root
        current_dir = Path(__file__).parent  # api directory
        app_dir = current_dir.parent         # app directory
        project_dir = app_dir.parent         # project directory
        root_dir = project_dir.parent        # root directory

        self.model_dir = root_dir / 'models'
        self.data_dir = root_dir / 'data'

        logger.info(f"🔍 Model directory: {self.model_dir}")
        logger.info(f"🔍 Data directory: {self.data_dir}")

    def load_model(self):
        """Load the trained model and preprocessors"""
        try:
            # Load the trained model
            model_path = self.model_dir / 'rf_model.pkl'
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            self.model = joblib.load(model_path)
            logger.info(f"✅ Model loaded from {model_path}")

            # Load label encoder
            le_path = self.model_dir / 'label_encoder.pkl'
            if not le_path.exists():
                raise FileNotFoundError(f"Label encoder not found: {le_path}")

            self.label_encoder = joblib.load(le_path)
            logger.info(f"✅ Label encoder loaded from {le_path}")

            # Load feature names
            features_path = self.model_dir / 'feature_names.json'
            if not features_path.exists():
                raise FileNotFoundError(f"Feature names not found: {features_path}")

            with open(features_path, 'r') as f:
                self.feature_names = json.load(f)
            logger.info(f"✅ Feature names loaded: {len(self.feature_names)} features")
            logger.info(f"🔍 Feature names: {self.feature_names[:10]}...")

            # Create team mapping
            self.team_mapping = dict(zip(
                self.label_encoder.classes_,
                self.label_encoder.transform(self.label_encoder.classes_)
            ))
            logger.info(f"✅ Team mapping created for {len(self.team_mapping)} teams")

            # Load and preprocess historical data
            data_path = self.data_dir / 'results.csv'
            if not data_path.exists():
                raise FileNotFoundError(f"Historical data not found: {data_path}")

            # Load and preprocess data (same as training)
            df = pd.read_csv(data_path, encoding='latin-1')
            logger.info(f"📊 Raw data loaded: {len(df)} matches")

            # Apply same preprocessing as training
            df['DateTime'] = pd.to_datetime(df['DateTime']).dt.date
            df['DateTime'] = pd.to_datetime(df['DateTime'])
            df.drop(columns=['Referee'], inplace=True, errors='ignore')
            df.drop(columns=['HTR'], inplace=True, errors='ignore')
            df['Season'] = df['Season'].str.split('-').str[0].astype(int)
            df = df[~((df['Season'] >= 1993) & (df['Season'] < 2000))]
            df.reset_index(drop=True, inplace=True)
            df['FTR'] = df['FTR'].map({'H': 1, 'A': -1, 'D': 0})
            df['FTR'] = df['FTR'].astype('float64')

            # Encode team names using SAME label encoder from training
            df['HomeTeam'] = self.label_encoder.transform(df['HomeTeam'])
            df['AwayTeam'] = self.label_encoder.transform(df['AwayTeam'])

            self.historical_data = df
            logger.info(f"✅ Historical data preprocessed: {len(df)} matches")

            self._loaded = True
            logger.info("🎉 All components loaded successfully!")

        except Exception as e:
            logger.error(f"❌ Error loading model components: {e}")
            raise

    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._loaded

    def get_available_teams(self) -> List[str]:
        """Get list of available teams"""
        if not self._loaded:
            raise ValueError("Model not loaded")
        return list(self.label_encoder.classes_)

    def _calculate_real_features(self, home_team_encoded: int, away_team_encoded: int, match_date_dt, window: int = 10) -> Dict[str, float]:
        """Calculate REAL features using the same logic as your notebook"""
        logger.info(f"🔍 Calculating real features for teams {home_team_encoded} vs {away_team_encoded}")

        # Get historical data before match date
        recent_data = self.historical_data[self.historical_data['DateTime'] < match_date_dt].copy()

        if len(recent_data) == 0:
            logger.warning("⚠️ No historical data - using defaults")
            return self._get_default_features()

        # Initialize feature dictionary
        features = {}

        # Get team matches
        home_matches = recent_data[
            (recent_data['HomeTeam'] == home_team_encoded) |
            (recent_data['AwayTeam'] == home_team_encoded)
        ].copy()

        away_matches = recent_data[
            (recent_data['HomeTeam'] == away_team_encoded) |
            (recent_data['AwayTeam'] == away_team_encoded)
        ].copy()

        logger.info(f"🔍 Home team matches: {len(home_matches)}, Away team matches: {len(away_matches)}")

        # 1. AVERAGE GOALS FEATURES
        home_goals_avg = self._calculate_avg_goals(home_matches, home_team_encoded, window)
        away_goals_avg = self._calculate_avg_goals(away_matches, away_team_encoded, window)

        features['home_team_avg_goals_last10'] = home_goals_avg
        features['away_team_avg_goals_last10'] = away_goals_avg

        # 2. CONVERSION RATE FEATURES
        home_conversion = self._calculate_conversion_rate(home_matches, home_team_encoded, window)
        away_conversion = self._calculate_conversion_rate(away_matches, away_team_encoded, window)

        features['home_conversion_rate_last10'] = home_conversion
        features['away_conversion_rate_last10'] = away_conversion

        # 3. WIN RATE FEATURES
        home_winrate = self._calculate_winrate(home_matches, home_team_encoded, window)
        away_winrate = self._calculate_winrate(away_matches, away_team_encoded, window)

        features['home_team_winrate_last10'] = home_winrate
        features['away_team_winrate_last10'] = away_winrate

        # 4. GOALS CONCEDED FEATURES
        home_conceded = self._calculate_goals_conceded(home_matches, home_team_encoded, window)
        away_conceded = self._calculate_goals_conceded(away_matches, away_team_encoded, window)

        features['home_team_avg_goals_conceded_last10'] = home_conceded
        features['away_team_avg_goals_conceded_last10'] = away_conceded

        # 5. CORNERS FEATURES
        home_corners = self._calculate_avg_corners(home_matches, home_team_encoded, window)
        away_corners = self._calculate_avg_corners(away_matches, away_team_encoded, window)

        features['home_team_corners_last10'] = home_corners
        features['away_team_corners_last10'] = away_corners

        # 6. FOULS FEATURES
        home_fouls = self._calculate_avg_fouls(home_matches, home_team_encoded, window)
        away_fouls = self._calculate_avg_fouls(away_matches, away_team_encoded, window)

        features['home_team_fouls_last10'] = home_fouls
        features['away_team_fouls_last10'] = away_fouls

        # 7. CARDS FEATURES
        home_yellow, home_red = self._calculate_avg_cards(home_matches, home_team_encoded, window)
        away_yellow, away_red = self._calculate_avg_cards(away_matches, away_team_encoded, window)

        features['home_team_yellow_cards_last10'] = home_yellow
        features['home_team_red_cards_last10'] = home_red
        features['away_team_yellow_cards_last10'] = away_yellow
        features['away_team_red_cards_last10'] = away_red

        # 8. HOME ADVANTAGE FEATURE
        home_advantage = self._calculate_home_advantage(recent_data, home_team_encoded)
        features['HomeTeam_HomeAdvantage'] = home_advantage

        # 9. FORM FEATURES (last 5 matches)
        home_form, home_goal_diff = self._calculate_recent_form(home_matches, home_team_encoded, 5)
        away_form, away_goal_diff = self._calculate_recent_form(away_matches, away_team_encoded, 5)

        features['home_team_form_last5'] = home_form
        features['away_team_form_last5'] = away_form
        features['home_team_goals_diff_last5'] = home_goal_diff
        features['away_team_goals_diff_last5'] = away_goal_diff

        # 10. DIFFERENCE FEATURES
        features['goals_diff_last10'] = home_goals_avg - away_goals_avg
        features['concede_diff_last10'] = home_conceded - away_conceded
        features['conversion_diff_last10'] = home_conversion - away_conversion
        features['winrate_diff_last10'] = home_winrate - away_winrate
        features['form_diff_last5'] = home_form - away_form

        # 11. ELO RATING FEATURES
        home_elo, away_elo, elo_diff, win_prob = self._calculate_elo_features(
            recent_data, home_team_encoded, away_team_encoded, match_date_dt
        )

        features['home_elo_rating'] = home_elo
        features['away_elo_rating'] = away_elo
        features['elo_difference'] = elo_diff
        features['home_win_probability'] = win_prob

        # 12. HEAD-TO-HEAD FEATURES
        h2h_home, h2h_away, h2h_draw = self._calculate_h2h_features(
            recent_data, home_team_encoded, away_team_encoded, 5
        )

        features['h2h_home_win_ratio'] = h2h_home
        features['h2h_away_win_ratio'] = h2h_away
        features['h2h_draw_ratio'] = h2h_draw

        logger.info(f"🔍 Calculated {len(features)} real features")
        return features

    def _calculate_avg_goals(self, matches, team_encoded, window):
        """Calculate average goals scored by team"""
        if len(matches) == 0:
            return 1.5

        goals = []
        for _, match in matches.tail(window).iterrows():
            if match['HomeTeam'] == team_encoded:
                goals.append(match['FTHG'])
            else:
                goals.append(match['FTAG'])

        return np.mean(goals) if goals else 1.5

    def _calculate_conversion_rate(self, matches, team_encoded, window):
        """Calculate conversion rate (goals/shots)"""
        if len(matches) == 0:
            return 0.15

        total_goals = 0
        total_shots = 0

        for _, match in matches.tail(window).iterrows():
            if match['HomeTeam'] == team_encoded:
                total_goals += match['FTHG']
                total_shots += match['HS']
            else:
                total_goals += match['FTAG']
                total_shots += match['AS']

        return total_goals / total_shots if total_shots > 0 else 0.15

    def _calculate_winrate(self, matches, team_encoded, window):
        """Calculate win rate"""
        if len(matches) == 0:
            return 0.4

        wins = 0
        total = 0

        for _, match in matches.tail(window).iterrows():
            total += 1
            if match['HomeTeam'] == team_encoded and match['FTR'] == 1.0:
                wins += 1
            elif match['AwayTeam'] == team_encoded and match['FTR'] == -1.0:
                wins += 1

        return wins / total if total > 0 else 0.4

    def _calculate_goals_conceded(self, matches, team_encoded, window):
        """Calculate average goals conceded"""
        if len(matches) == 0:
            return 1.2

        conceded = []
        for _, match in matches.tail(window).iterrows():
            if match['HomeTeam'] == team_encoded:
                conceded.append(match['FTAG'])  # Home team concedes away goals
            else:
                conceded.append(match['FTHG'])  # Away team concedes home goals

        return np.mean(conceded) if conceded else 1.2

    def _calculate_avg_corners(self, matches, team_encoded, window):
        """Calculate average corners"""
        if len(matches) == 0:
            return 5.0

        corners = []
        for _, match in matches.tail(window).iterrows():
            if match['HomeTeam'] == team_encoded:
                corners.append(match['HC'])
            else:
                corners.append(match['AC'])

        return np.mean(corners) if corners else 5.0

    def _calculate_avg_fouls(self, matches, team_encoded, window):
        """Calculate average fouls"""
        if len(matches) == 0:
            return 12.0

        fouls = []
        for _, match in matches.tail(window).iterrows():
            if match['HomeTeam'] == team_encoded:
                fouls.append(match['HF'])
            else:
                fouls.append(match['AF'])

        return np.mean(fouls) if fouls else 12.0

    def _calculate_avg_cards(self, matches, team_encoded, window):
        """Calculate average yellow and red cards"""
        if len(matches) == 0:
            return 2.0, 0.1

        yellow_cards = []
        red_cards = []

        for _, match in matches.tail(window).iterrows():
            if match['HomeTeam'] == team_encoded:
                yellow_cards.append(match['HY'])
                red_cards.append(match['HR'])
            else:
                yellow_cards.append(match['AY'])
                red_cards.append(match['AR'])

        avg_yellow = np.mean(yellow_cards) if yellow_cards else 2.0
        avg_red = np.mean(red_cards) if red_cards else 0.1

        return avg_yellow, avg_red

    def _calculate_home_advantage(self, recent_data, team_encoded):
        """Calculate home advantage for team"""
        home_matches = recent_data[recent_data['HomeTeam'] == team_encoded]
        away_matches = recent_data[recent_data['AwayTeam'] == team_encoded]

        if len(home_matches) == 0 or len(away_matches) == 0:
            return 0.2

        # Calculate points at home vs away
        home_points = []
        for _, match in home_matches.iterrows():
            if match['FTR'] == 1.0:
                home_points.append(3)
            elif match['FTR'] == 0.0:
                home_points.append(1)
            else:
                home_points.append(0)

        away_points = []
        for _, match in away_matches.iterrows():
            if match['FTR'] == -1.0:
                away_points.append(3)
            elif match['FTR'] == 0.0:
                away_points.append(1)
            else:
                away_points.append(0)

        home_avg = np.mean(home_points) if home_points else 1.5
        away_avg = np.mean(away_points) if away_points else 1.5

        return home_avg - away_avg

    def _calculate_recent_form(self, matches, team_encoded, window):
        """Calculate recent form (points and goal difference)"""
        if len(matches) == 0:
            return 1.5, 0.0

        points = []
        goal_diffs = []

        for _, match in matches.tail(window).iterrows():
            if match['HomeTeam'] == team_encoded:
                # Home match
                goals_for = match['FTHG']
                goals_against = match['FTAG']
                if match['FTR'] == 1.0:
                    points.append(3)
                elif match['FTR'] == 0.0:
                    points.append(1)
                else:
                    points.append(0)
            else:
                # Away match
                goals_for = match['FTAG']
                goals_against = match['FTHG']
                if match['FTR'] == -1.0:
                    points.append(3)
                elif match['FTR'] == 0.0:
                    points.append(1)
                else:
                    points.append(0)

            goal_diffs.append(goals_for - goals_against)

        avg_points = np.mean(points) if points else 1.5
        avg_goal_diff = np.mean(goal_diffs) if goal_diffs else 0.0

        return avg_points, avg_goal_diff

    def _calculate_elo_features(self, recent_data, home_team_encoded, away_team_encoded, match_date):
        """Calculate ELO ratings (simplified)"""
        # For simplicity, we'll use a basic ELO calculation
        # In reality, you'd implement the full ELO system from your notebook

        home_elo = 1500.0
        away_elo = 1500.0

        # Get recent performance to adjust ELO
        home_recent = recent_data[
            (recent_data['HomeTeam'] == home_team_encoded) |
            (recent_data['AwayTeam'] == home_team_encoded)
        ].tail(10)

        away_recent = recent_data[
            (recent_data['HomeTeam'] == away_team_encoded) |
            (recent_data['AwayTeam'] == away_team_encoded)
        ].tail(10)

        # Adjust ELO based on recent wins
        for _, match in home_recent.iterrows():
            if match['HomeTeam'] == home_team_encoded and match['FTR'] == 1.0:
                home_elo += 20
            elif match['AwayTeam'] == home_team_encoded and match['FTR'] == -1.0:
                home_elo += 20
            elif match['FTR'] == 0.0:
                home_elo += 5
            else:
                home_elo -= 15

        for _, match in away_recent.iterrows():
            if match['HomeTeam'] == away_team_encoded and match['FTR'] == 1.0:
                away_elo += 20
            elif match['AwayTeam'] == away_team_encoded and match['FTR'] == -1.0:
                away_elo += 20
            elif match['FTR'] == 0.0:
                away_elo += 5
            else:
                away_elo -= 15

        elo_diff = home_elo - away_elo
        win_prob = 1 / (1 + 10**(-elo_diff / 400))

        return home_elo, away_elo, elo_diff, win_prob

    def _calculate_h2h_features(self, recent_data, home_team_encoded, away_team_encoded, window):
        """Calculate head-to-head features"""
        h2h_matches = recent_data[
            ((recent_data['HomeTeam'] == home_team_encoded) & (recent_data['AwayTeam'] == away_team_encoded)) |
            ((recent_data['HomeTeam'] == away_team_encoded) & (recent_data['AwayTeam'] == home_team_encoded))
        ].tail(window)

        if len(h2h_matches) == 0:
            return 0.33, 0.33, 0.33

        home_wins = 0
        away_wins = 0
        draws = 0

        for _, match in h2h_matches.iterrows():
            if match['HomeTeam'] == home_team_encoded:
                if match['FTR'] == 1.0:
                    home_wins += 1
                elif match['FTR'] == -1.0:
                    away_wins += 1
                else:
                    draws += 1
            else:  # away_team_encoded is home
                if match['FTR'] == 1.0:
                    away_wins += 1
                elif match['FTR'] == -1.0:
                    home_wins += 1
                else:
                    draws += 1

        total = len(h2h_matches)
        return home_wins / total, away_wins / total, draws / total

    def predict_match(self, home_team: str, away_team: str, match_date: str) -> Dict:
        """Make a prediction for a match with debug logging"""
        logger.info(f"🔍 DEBUG: Predicting {home_team} vs {away_team} on {match_date}")

        if not self._loaded:
            raise ValueError("Model not loaded")

        # Validate teams
        available_teams = list(self.label_encoder.classes_)
        if home_team not in available_teams:
            logger.error(f"❌ Home team '{home_team}' not found in training data")
            logger.info(f"Available teams: {available_teams[:10]}...")
            raise ValueError(f"Home team '{home_team}' not found in training data")

        if away_team not in available_teams:
            logger.error(f"❌ Away team '{away_team}' not found in training data")
            raise ValueError(f"Away team '{away_team}' not found in training data")

        # Validate date format
        try:
            match_date_dt = pd.to_datetime(match_date)
        except ValueError:
            raise ValueError("Invalid date format. Use YYYY-MM-DD")

        # Encode team names
        home_team_encoded = self.label_encoder.transform([home_team])[0]
        away_team_encoded = self.label_encoder.transform([away_team])[0]
        logger.info(f"🔍 Encoded teams: {home_team}={home_team_encoded}, {away_team}={away_team_encoded}")

                # Check if we have engineered features or just basic features
        if len(self.feature_names) <= 14:
            logger.warning("⚠️ Model has basic features only. Need to retrain with engineered features!")
            # For basic features, create simple prediction based on recent averages
            recent_data = self.historical_data[self.historical_data['DateTime'] < match_date_dt].copy()

            if len(recent_data) == 0:
                # No data - equal probabilities
                feature_vector = [home_team_encoded, away_team_encoded, 1.5, 1.3, 12, 10, 4, 3, 12, 13, 5, 5, 2, 2]
            else:
                # Get basic stats for prediction
                home_recent = recent_data[
                    (recent_data['HomeTeam'] == home_team_encoded) |
                    (recent_data['AwayTeam'] == home_team_encoded)
                ].tail(10)

                away_recent = recent_data[
                    (recent_data['HomeTeam'] == away_team_encoded) |
                    (recent_data['AwayTeam'] == away_team_encoded)
                ].tail(10)

                # Calculate basic averages
                home_goals_avg = 1.5
                away_goals_avg = 1.3

                if len(home_recent) > 0:
                    home_goals = []
                    for _, match in home_recent.iterrows():
                        if match['HomeTeam'] == home_team_encoded:
                            home_goals.append(match['FTHG'])
                        else:
                            home_goals.append(match['FTAG'])
                    home_goals_avg = np.mean(home_goals) if home_goals else 1.5

                if len(away_recent) > 0:
                    away_goals = []
                    for _, match in away_recent.iterrows():
                        if match['HomeTeam'] == away_team_encoded:
                            away_goals.append(match['FTHG'])
                        else:
                            away_goals.append(match['FTAG'])
                    away_goals_avg = np.mean(away_goals) if away_goals else 1.3

                # Create feature vector matching model expectations
                feature_vector = [
                    home_team_encoded,     # HomeTeam
                    away_team_encoded,     # AwayTeam
                    home_goals_avg,        # FTHG (avg goals home team scores)
                    away_goals_avg,        # FTAG (avg goals away team scores)
                    home_goals_avg * 8,    # HS (estimated shots)
                    away_goals_avg * 8,    # AS (estimated shots)
                    home_goals_avg * 3,    # HST (estimated shots on target)
                    away_goals_avg * 3,    # AST (estimated shots on target)
                    12.0,                  # HF (avg fouls home)
                    12.5,                  # AF (avg fouls away)
                    5.2,                   # HC (avg corners home)
                    4.8,                   # AC (avg corners away)
                    2.0,                   # HY (avg yellow cards home)
                    2.1                    # AY (avg yellow cards away)
                ]

            logger.info(f"🔍 Using basic features with averages: FTHG={feature_vector[2]:.2f}, FTAG={feature_vector[3]:.2f}")

        else:
            # Model has engineered features - calculate them properly
            features_dict = self._calculate_real_features(home_team_encoded, away_team_encoded, match_date_dt)

            # Create feature vector in correct order
            feature_vector = []
            for feature_name in self.feature_names:
                if feature_name in features_dict:
                    feature_vector.append(features_dict[feature_name])
                else:
                    logger.warning(f"⚠️ Missing feature: {feature_name}, using default 0.0")
                    feature_vector.append(0.0)        # Ensure correct length
        if len(feature_vector) != len(self.feature_names):
            logger.warning(f"⚠️ Feature vector length mismatch. Expected: {len(self.feature_names)}, Got: {len(feature_vector)}")
            # Pad or truncate to correct length
            if len(feature_vector) < len(self.feature_names):
                feature_vector.extend([1.0] * (len(self.feature_names) - len(feature_vector)))
            else:
                feature_vector = feature_vector[:len(self.feature_names)]

        logger.info(f"🔍 Feature vector (first 10): {feature_vector[:10]}")
        logger.info(f"🔍 Feature vector length: {len(feature_vector)}")

        # Create feature array in correct order
        feature_array = np.array(feature_vector).reshape(1, -1)

        # Make prediction
        try:
            probabilities = self.model.predict_proba(feature_array)[0]
            prediction_encoded = self.model.predict(feature_array)[0]

            logger.info(f"🔍 Raw probabilities: {probabilities}")
            logger.info(f"🔍 Raw prediction: {prediction_encoded}")

            # Convert prediction back to original format
            if prediction_encoded == 1.0:
                outcome = f"🏠 {home_team} Win"
            elif prediction_encoded == -1.0:
                outcome = f"✈️ {away_team} Win"
            else:
                outcome = "🤝 Draw"

            # Create probability dictionary
            # Classes should be ordered as [-1.0, 0.0, 1.0] (Away, Draw, Home)
            prob_dict = {
                'Away Win': float(probabilities[0]),
                'Draw': float(probabilities[1]),
                'Home Win': float(probabilities[2])
            }

            # Calculate confidence (highest probability)
            confidence = float(max(probabilities))

            logger.info(f"🎯 Final prediction: {outcome}")
            logger.info(f"🎯 Probabilities: {prob_dict}")
            logger.info(f"🎯 Confidence: {confidence:.3f}")

            return {
                'home_team': home_team,
                'away_team': away_team,
                'match_date': match_date,
                'prediction': outcome,
                'probabilities': prob_dict,
                'confidence': confidence,
                'api_version': '1.0.0'
            }

        except Exception as e:
            logger.error(f"❌ Prediction error: {e}")
            raise
