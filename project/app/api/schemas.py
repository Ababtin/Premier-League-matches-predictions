"""
Pydantic schemas for API request/response models
"""
from pydantic import BaseModel, Field
from typing import Dict, Optional
from datetime import datetime

class MatchRequest(BaseModel):
    """Request model for match prediction"""
    home_team: str = Field(..., description="Home team name")
    away_team: str = Field(..., description="Away team name")
    match_date: str = Field(..., description="Match date in YYYY-MM-DD format")

    class Config:
        json_schema_extra = {
            "example": {
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "match_date": "2024-01-15"
            }
        }

class PredictionResponse(BaseModel):
    """Response model for match prediction"""
    home_team: str
    away_team: str
    match_date: str
    prediction: str
    probabilities: Dict[str, float]
    confidence: float
    api_version: str = "1.0.0"

    class Config:
        json_schema_extra = {
            "example": {
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
                "api_version": "1.0.0"
            }
        }
