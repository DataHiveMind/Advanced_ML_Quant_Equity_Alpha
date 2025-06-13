from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI(title="Quant Alpha Model API")

MODEL_PATH = "models/lgbm_best.pkl"  # Adjust as needed
EQUITY_CURVE_PATH = "results/equity_curve.csv"  # Adjust as needed


class PredictRequest(BaseModel):
    features: dict


@app.get("/predict/{ticker}")
def predict_ticker(ticker: str, features: dict = Query(...)):
    """
    Get model prediction for a single ticker and feature set.
    """
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=404, detail="Model not found")
    model = joblib.load(MODEL_PATH)
    X = pd.DataFrame([features])
    try:
        pred = model.predict(X)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")
    return {"ticker": ticker, "prediction": float(np.ravel(pred)[0])}


@app.post("/predict_batch/")
def predict_batch(request: PredictRequest):
    """
    Get model predictions for a batch of feature sets.
    """
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=404, detail="Model not found")
    model = joblib.load(MODEL_PATH)
    X = pd.DataFrame(request.features)
    try:
        preds = model.predict(X)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")
    return {"predictions": list(np.ravel(preds))}


@app.get("/strategy_status")
def strategy_status():
    """
    Return a simple status for the strategy (placeholder).
    """
    return {
        "status": "running",
        "message": "Strategy is active and monitoring signals.",
    }


@app.get("/equity_curve")
def get_equity_curve():
    """
    Return the simulated equity curve as a list.
    """
    if not os.path.exists(EQUITY_CURVE_PATH):
        raise HTTPException(status_code=404, detail="Equity curve not found")
    equity_curve = pd.read_csv(EQUITY_CURVE_PATH)
    return {"equity_curve": equity_curve["equity"].tolist()}
