# Premier League Match Predictor API

FastAPI application for predicting Premier League match outcomes using machine learning.

## Features

- **Match Prediction**: Predict outcomes for Premier League matches
- **Team Information**: Get list of available teams
- **REST API**: Easy-to-use HTTP endpoints
- **Model Persistence**: Trained model saved and loaded automatically

## Installation

1. Install required packages:
```bash
pip install fastapi uvicorn scikit-learn pandas numpy joblib
```

2. Make sure your trained model files are in the `../models/` directory:
   - `rf_model.pkl` - Trained Random Forest model
   - `label_encoder.pkl` - Label encoder for team names
   - `feature_names.json` - List of feature names

## Running the API

Start the FastAPI server:

```bash
cd api
python main.py
```

Or using uvicorn directly:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

### GET /
Root endpoint with API information

### GET /health
Health check endpoint

### GET /teams
Get list of available teams

### POST /predict
Predict match outcome

**Request Body:**
```json
{
    "home_team": "Arsenal",
    "away_team": "Chelsea",
    "match_date": "2024-01-15"
}
```

**Response:**
```json
{
    "home_team": "Arsenal",
    "away_team": "Chelsea",
    "match_date": "2024-01-15",
    "prediction": "🏠 Arsenal Win",
    "probabilities": {
        "Home Win": 0.45,
        "Draw": 0.28,
        "Away Win": 0.27
    },
    "confidence": 0.45,
    "model_version": "1.0.0"
}
```

## Interactive Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Testing the API

You can test the API using curl:

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "home_team": "Arsenal",
       "away_team": "Chelsea",
       "match_date": "2024-01-15"
     }'
```

## File Structure

```
api/
├── main.py          # FastAPI application
├── schemas.py       # Pydantic models
├── predictor.py     # Model loading and prediction logic
└── README.md        # This file
```
