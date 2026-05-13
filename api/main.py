"""
FastAPI prediction server for Sentiment-Enhanced Demand Forecasting.

Endpoints:
    GET  /health              — health check
    GET  /sentiment           — current Fear & Greed + latest news sentiment
    POST /predict             — next-day BTC price direction prediction
    GET  /predict/latest      — prediction using live-fetched data

Usage:
    uvicorn api.main:app --reload --port 8000
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAVED_DIR = PROJECT_ROOT / "models" / "saved"

EXCLUDE_FROM_FEATURES = {"close", "open", "high", "low", "volume", "sma_50", "sma_200", "target"}

models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = SAVED_DIR / "btc_usd_xgb_all_features_model.joblib"
    if model_path.exists():
        models["xgb"] = joblib.load(model_path)
        logger.info("Loaded XGBoost model")
    else:
        logger.warning("XGBoost model not found at %s", model_path)

    results_path = SAVED_DIR / "btc_usd_xgb_results.json"
    if results_path.exists():
        import json
        with open(results_path) as f:
            models["xgb_meta"] = json.load(f)
    yield
    models.clear()


app = FastAPI(
    title="Sentiment-Enhanced Demand Forecaster",
    description="Predict BTC price direction using market data + social sentiment",
    version="0.1.0",
    lifespan=lifespan,
)


class PredictionRequest(BaseModel):
    daily_return: float
    momentum_5d: float
    momentum_10d: float
    momentum_20d: float
    volatility_20d: float
    sma_crossover: int
    volume_sma_20: float
    fng_value: int
    fng_normalized: float
    fng_normalized_lag_1: float
    fng_normalized_lag_3: float
    fng_normalized_lag_7: float
    fng_sma_7: float
    fng_sma_14: float
    fng_momentum_5d: float
    fng_momentum_10d: float
    extreme_fear: int
    extreme_greed: int
    volume_fng_ratio: float
    price_sentiment_divergence: float
    news_sentiment_mean: float = 0.0
    news_sentiment_std: float = 0.0
    news_volume: float = 0.0
    news_positive_ratio: float = 0.0
    news_negative_ratio: float = 0.0
    news_sentiment_mean_lag_1: float = 0.0
    news_sentiment_mean_lag_3: float = 0.0
    news_sentiment_mean_lag_7: float = 0.0
    news_sentiment_sma_7: float = 0.0


class PredictionResponse(BaseModel):
    direction: str
    probability_up: float
    probability_down: float
    confidence: float
    model: str
    timestamp: str


class SentimentResponse(BaseModel):
    fng_value: int
    fng_classification: str
    fng_normalized: float
    timestamp: str


def _fetch_live_sentiment() -> dict:
    """Fetch current Fear & Greed Index."""
    try:
        resp = requests.get(
            "https://api.alternative.me/fng/",
            params={"limit": 14, "format": "json"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()["data"]

        current = data[0]
        values = [int(d["value"]) for d in data]
        normalized = [(v - 50) / 50 for v in values]

        return {
            "fng_value": int(current["value"]),
            "fng_classification": current["value_classification"],
            "fng_normalized": normalized[0],
            "fng_normalized_lag_1": normalized[1] if len(normalized) > 1 else normalized[0],
            "fng_normalized_lag_3": normalized[3] if len(normalized) > 3 else normalized[0],
            "fng_normalized_lag_7": normalized[7] if len(normalized) > 7 else normalized[0],
            "fng_sma_7": np.mean(normalized[:7]),
            "fng_sma_14": np.mean(normalized[:14]),
            "fng_momentum_5d": normalized[0] - normalized[5] if len(normalized) > 5 else 0,
            "fng_momentum_10d": normalized[0] - normalized[10] if len(normalized) > 10 else 0,
            "extreme_fear": 1 if int(current["value"]) <= 20 else 0,
            "extreme_greed": 1 if int(current["value"]) >= 80 else 0,
            "history": values,
        }
    except Exception as e:
        logger.error("Failed to fetch Fear & Greed: %s", e)
        raise HTTPException(status_code=503, detail="Sentiment API unavailable")


def _fetch_live_market(ticker: str = "BTC-USD", days: int = 220) -> dict:
    """Fetch recent market data and compute features."""
    try:
        end = datetime.now()
        start = end - timedelta(days=days)
        df = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                         end=end.strftime("%Y-%m-%d"), auto_adjust=True, progress=False)

        if df.empty:
            raise ValueError("No data returned")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]

        df["sma_50"] = df["close"].rolling(50).mean()
        df["sma_200"] = df["close"].rolling(200).mean()
        df["daily_return"] = df["close"].pct_change()
        for p in [5, 10, 20]:
            df[f"momentum_{p}d"] = df["close"].pct_change(periods=p)
        df["volatility_20d"] = df["daily_return"].rolling(20).std()
        df["sma_crossover"] = (df["sma_50"] > df["sma_200"]).astype(int)
        df["volume_sma_20"] = df["volume"].rolling(20).mean()

        latest = df.iloc[-1]
        return {
            "daily_return": float(latest["daily_return"]),
            "momentum_5d": float(latest["momentum_5d"]),
            "momentum_10d": float(latest["momentum_10d"]),
            "momentum_20d": float(latest["momentum_20d"]),
            "volatility_20d": float(latest["volatility_20d"]),
            "sma_crossover": int(latest["sma_crossover"]),
            "volume_sma_20": float(latest["volume_sma_20"]),
            "close": float(latest["close"]),
            "date": str(df.index[-1].date()),
        }
    except Exception as e:
        logger.error("Failed to fetch market data: %s", e)
        raise HTTPException(status_code=503, detail="Market data unavailable")


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": "xgb" in models,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/sentiment", response_model=SentimentResponse)
async def get_sentiment():
    """Get current Crypto Fear & Greed Index."""
    data = _fetch_live_sentiment()
    return SentimentResponse(
        fng_value=data["fng_value"],
        fng_classification=data["fng_classification"],
        fng_normalized=round(data["fng_normalized"], 4),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(req: PredictionRequest):
    """Predict next-day BTC price direction from provided features."""
    if "xgb" not in models:
        raise HTTPException(status_code=503, detail="Model not loaded")

    feature_names = sorted(set(req.model_fields.keys()))
    features = np.array([[getattr(req, f) for f in feature_names]])

    prob = models["xgb"].predict_proba(features)[0]
    direction = "Up" if prob[1] > 0.5 else "Down"

    return PredictionResponse(
        direction=direction,
        probability_up=round(float(prob[1]), 4),
        probability_down=round(float(prob[0]), 4),
        confidence=round(float(max(prob)), 4),
        model="xgboost_all_features",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/predict/latest", response_model=PredictionResponse)
async def predict_latest():
    """Predict next-day direction using live-fetched market + sentiment data."""
    if "xgb" not in models:
        raise HTTPException(status_code=503, detail="Model not loaded")

    market = _fetch_live_market()
    sentiment = _fetch_live_sentiment()

    fng_abs = abs(sentiment["fng_normalized"]) or 0.01
    volume_fng_ratio = market["volume_sma_20"] / (fng_abs * 1e10)
    price_sentiment_divergence = market["momentum_5d"] - sentiment["fng_normalized"]

    feature_dict = {
        **{k: market[k] for k in [
            "daily_return", "momentum_5d", "momentum_10d", "momentum_20d",
            "volatility_20d", "sma_crossover", "volume_sma_20",
        ]},
        **{k: sentiment[k] for k in [
            "fng_value", "fng_normalized", "fng_normalized_lag_1",
            "fng_normalized_lag_3", "fng_normalized_lag_7",
            "fng_sma_7", "fng_sma_14", "fng_momentum_5d",
            "fng_momentum_10d", "extreme_fear", "extreme_greed",
        ]},
        "volume_fng_ratio": volume_fng_ratio,
        "price_sentiment_divergence": price_sentiment_divergence,
        "news_sentiment_mean": 0.0,
        "news_sentiment_std": 0.0,
        "news_volume": 0.0,
        "news_positive_ratio": 0.0,
        "news_negative_ratio": 0.0,
        "news_sentiment_mean_lag_1": 0.0,
        "news_sentiment_mean_lag_3": 0.0,
        "news_sentiment_mean_lag_7": 0.0,
        "news_sentiment_sma_7": 0.0,
    }

    feature_names = sorted(feature_dict.keys())
    features = np.array([[feature_dict[f] for f in feature_names]])

    prob = models["xgb"].predict_proba(features)[0]
    direction = "Up" if prob[1] > 0.5 else "Down"

    return PredictionResponse(
        direction=direction,
        probability_up=round(float(prob[1]), 4),
        probability_down=round(float(prob[0]), 4),
        confidence=round(float(max(prob)), 4),
        model="xgboost_all_features",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
