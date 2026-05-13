# Sentiment-Enhanced Demand Forecasting

Predict next-day BTC price direction by combining traditional market data with real-time social sentiment signals. Built as a full ML pipeline — from data ingestion through model training to interactive deployment.

## Core Idea

Two parallel data pipelines (quantitative + qualitative) merge on a shared daily timestamp. A hybrid modeling approach compares a tree-based baseline (XGBoost) against a sequence-aware deep learning model (LSTM) to measure the incremental value of sentiment features.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3+-red)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-orange)

## Results

| Model | F1 | Accuracy | AUC-ROC |
|-------|:--:|:--------:|:-------:|
| XGBoost + Sentiment | 0.659 | 0.492 | **0.519** |
| XGBoost (price-only) | 0.659 | 0.492 | 0.411 |
| LSTM + Sentiment | 0.659 | 0.492 | **0.555** |
| LSTM (price-only) | 0.654 | 0.486 | 0.540 |

**Key finding:** Sentiment features improve AUC-ROC by **+10.8%** (XGBoost) and **+1.5%** (LSTM), demonstrating that social sentiment carries predictive signal for crypto price direction beyond what technical indicators alone provide.

## Architecture

```
                    ┌─────────────────┐
                    │   Yahoo Finance  │
                    │   (BTC-USD)      │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐     ┌──────────────────┐
                    │  Quantitative   │     │  Fear & Greed    │
                    │  Pipeline       │     │  Index API       │
                    │  OHLCV + SMA +  │     │                  │
                    │  Momentum +     │     └────────┬─────────┘
                    │  Volatility     │              │
                    └────────┬────────┘     ┌────────▼─────────┐
                             │              │  News RSS Feeds   │
                             │              │  + FinBERT NLP    │
                             │              └────────┬─────────┘
                             │                       │
                    ┌────────▼───────────────────────▼─────────┐
                    │         Feature Engineering               │
                    │  Lag features, rolling averages,          │
                    │  interaction features, target variable    │
                    └────────────────────┬─────────────────────┘
                                        │
                    ┌───────────────────┬┴───────────────────┐
                    │                   │                     │
              ┌─────▼─────┐     ┌──────▼──────┐    ┌────────▼────────┐
              │  XGBoost  │     │    LSTM     │    │   Evaluation    │
              │  Baseline │     │   Advanced  │    │   + Error       │
              │  (Optuna) │     │  (PyTorch)  │    │   Analysis      │
              └─────┬─────┘     └──────┬──────┘    └────────┬────────┘
                    │                  │                     │
              ┌─────▼──────────────────▼─────────────────────▼────┐
              │                  Deployment                        │
              │   FastAPI (/predict, /sentiment)                   │
              │   Streamlit Dashboard                              │
              └───────────────────────────────────────────────────┘
```

## Project Structure

```
├── api/main.py                  # FastAPI prediction server
├── app/dashboard.py             # Streamlit interactive dashboard
├── configs/
│   ├── xgboost_params.yaml      # XGBoost hyperparameters
│   └── lstm_params.yaml         # LSTM hyperparameters
├── data/
│   ├── fetch_quantitative.py    # Yahoo Finance OHLCV + technicals
│   ├── fetch_sentiment.py       # Fear & Greed API + news RSS scraper
│   ├── raw/                     # Raw parquet files (gitignored)
│   └── processed/               # Feature-engineered datasets (gitignored)
├── evaluation/
│   └── evaluate.py              # Model evaluation + error analysis
├── features/
│   └── build_features.py        # Merge quant + sentiment, lag/rolling features
├── models/
│   ├── train_xgboost.py         # XGBoost with Optuna tuning + TimeSeriesSplit
│   ├── train_lstm.py            # PyTorch LSTM with early stopping
│   └── saved/                   # Trained model artifacts (gitignored)
├── nlp/
│   └── sentiment_pipeline.py    # FinBERT inference on news headlines
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory data analysis
│   ├── 02_model_comparison.ipynb # ROC curves, feature importance, uplift
│   └── 03_error_analysis.ipynb  # Worst predictions, Black Swan case study
├── Dockerfile
├── pyproject.toml
└── requirements.txt
```

## Quick Start

```bash
# Clone and set up
git clone https://github.com/KsaraSami/Sentiment-Enhanced-Demand-Forecasting.git
cd Sentiment-Enhanced-Demand-Forecasting
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the full pipeline
python -m data.fetch_quantitative --tickers BTC-USD --years 3
python -m data.fetch_sentiment
python -m nlp.sentiment_pipeline          # requires torch + transformers
python -m features.build_features
python -m models.train_xgboost            # ~30s with Optuna tuning
python -m models.train_lstm               # ~20s on CPU
python -m evaluation.evaluate

# Launch the app
uvicorn api.main:app --port 8000 &        # API server
streamlit run app/dashboard.py            # Dashboard at localhost:8501
```

## Data Sources

| Source | Type | Coverage |
|--------|------|----------|
| **Yahoo Finance** | BTC-USD daily OHLCV | 3+ years historical |
| **Alternative.me** | Crypto Fear & Greed Index (0-100) | 8+ years (since Feb 2018) |
| **CoinTelegraph / CoinDesk** | News headlines via RSS | Recent (last 1-2 days per fetch) |
| **FinBERT** (ProsusAI/finbert) | Sentiment scores from headlines | Scored on fetch |

## Feature Engineering

**36 features** organized into three categories:

- **Technical** (7): daily return, 5/10/20-day momentum, 20-day volatility, SMA crossover, volume SMA
- **Sentiment** (22): Fear & Greed value + normalized + 1/3/7-day lags + 7/14-day SMA + 5/10-day momentum + extreme flags + news sentiment stats + volume-sentiment ratio + price-sentiment divergence
- **Target**: next-day price direction (1 = Up, 0 = Down) — naturally balanced at ~51/49%

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server health check |
| `/sentiment` | GET | Current Fear & Greed Index |
| `/predict` | POST | Predict from provided features |
| `/predict/latest` | GET | Live prediction using fetched data |

## Tech Stack

| Category | Tools |
|----------|-------|
| **Data** | yfinance, pandas, pyarrow, requests |
| **NLP** | Hugging Face Transformers (FinBERT) |
| **ML/DL** | scikit-learn, XGBoost, PyTorch |
| **Tuning** | Optuna (Bayesian hyperparameter optimization) |
| **API** | FastAPI, uvicorn |
| **Frontend** | Streamlit |
| **Deployment** | Docker |

## License

MIT
