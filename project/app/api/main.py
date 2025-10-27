"""
FastAPI application for Premier League match prediction
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List
import uvicorn
import os
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the parent directory to Python path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.predictor import MatchPredictor
from api.schemas import MatchRequest, PredictionResponse
from contextlib import asynccontextmanager

# Initialize the predictor
predictor = MatchPredictor()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events"""
    # Startup
    try:
        predictor.load_model()
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise
    yield
    # Shutdown (cleanup if needed)
    pass

# Initialize FastAPI app
app = FastAPI(
    title="Premier League Match Predictor",
    description="API for predicting Premier League match outcomes using machine learning",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Premier League Match Predictor API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "teams": "/teams"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": predictor.is_loaded()
    }

@app.get("/teams")
async def get_teams():
    """Get list of available teams"""
    try:
        teams = predictor.get_available_teams()
        return {"teams": teams}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=PredictionResponse)
async def predict_match(match_request: MatchRequest):
    """Predict match outcome"""
    try:
        logger.info(f"🎯 Prediction request: {match_request.home_team} vs {match_request.away_team} on {match_request.match_date}")

        if not predictor.is_loaded():
            logger.error("❌ Model not loaded")
            raise HTTPException(status_code=503, detail="Model not loaded")

        prediction = predictor.predict_match(
            home_team=match_request.home_team,
            away_team=match_request.away_team,
            match_date=match_request.match_date
        )

        logger.info(f"🎯 Prediction result: {prediction['prediction']} (confidence: {prediction['confidence']:.3f})")
        return prediction

    except ValueError as e:
        logger.error(f"❌ Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
