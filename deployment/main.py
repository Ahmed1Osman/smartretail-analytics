
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from datetime import date, datetime
import json
import os

app = FastAPI(title="SmartRetail Analytics API", version="1.0.0")

# Load model on startup
model = None
feature_columns = []
metadata = {}

@app.on_event("startup")
async def load_model():
    global model, feature_columns, metadata
    try:
        model = joblib.load("sales_forecasting_model.joblib")
        
        if os.path.exists("feature_columns.json"):
            with open("feature_columns.json") as f:
                feature_columns = json.load(f)
        
        if os.path.exists("model_metadata.json"):
            with open("model_metadata.json") as f:
                metadata = json.load(f)
                
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")

class PredictionRequest(BaseModel):
    start_date: str
    end_date: str

class PredictionResponse(BaseModel):
    date: str
    predicted_revenue: float
    model_version: str

@app.get("/")
async def root():
    return {
        "message": "SmartRetail Analytics API",
        "status": "running",
        "model_loaded": model is not None,
        "documentation": "/docs"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if model else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "model_performance": metadata.get("performance_metrics", {})
    }

@app.post("/predict", response_model=list[PredictionResponse])
async def predict_sales(request: PredictionRequest):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Generate date range
        dates = pd.date_range(request.start_date, request.end_date, freq='D')
        
        # Create simple features (time-based)
        features = []
        for date_val in dates:
            feature_vector = [
                date_val.year - 2020,  # Normalized year
                date_val.month,
                date_val.day,
                date_val.dayofweek,
                date_val.dayofyear,
                1 if date_val.dayofweek >= 5 else 0,  # Weekend
            ]
            
            # Pad to match training features
            while len(feature_vector) < len(feature_columns):
                feature_vector.append(0)
            
            features.append(feature_vector[:len(feature_columns)])
        
        # Make predictions
        predictions = model.predict(np.array(features))
        
        # Format response
        results = []
        for i, date_val in enumerate(dates):
            results.append(PredictionResponse(
                date=date_val.strftime("%Y-%m-%d"),
                predicted_revenue=max(0, float(predictions[i])),
                model_version=metadata.get("model_version", "1.0.0")
            ))
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/info")
async def model_info():
    return {
        "model_type": metadata.get("model_type", "Unknown"),
        "training_date": metadata.get("training_date"),
        "performance": metadata.get("performance_metrics", {}),
        "feature_count": len(feature_columns)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
