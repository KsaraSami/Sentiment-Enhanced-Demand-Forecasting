# Sentiment-Enhanced Demand Forecasting

Crypto price direction forecasting using market data + social sentiment.
Asset focus: BTC-USD (primary), ETH-USD (secondary).

## Quick Start

```bash
source .venv/bin/activate

# 1. Fetch data
python -m data.fetch_quantitative --tickers BTC-USD ETH-USD --years 3
python -m data.fetch_sentiment

# 2. Process sentiment (FinBERT — requires torch + transformers)
python -m nlp.sentiment_pipeline

# 3. Build features
python -m features.build_features

# 4. Train models
python -m models.train_xgboost          # with Optuna tuning
python -m models.train_lstm

# 5. Evaluate
python -m evaluation.evaluate

# 6. Run app
uvicorn api.main:app --port 8000 &      # API server
streamlit run app/dashboard.py           # Dashboard
```

## Project Layout

- `data/` — ingestion scripts and raw/processed parquet files
- `nlp/` — sentiment extraction pipeline (FinBERT)
- `features/` — feature engineering (merges quant + sentiment)
- `models/` — training scripts (XGBoost baseline, LSTM advanced)
- `evaluation/` — evaluation and error analysis
- `notebooks/` — EDA, model comparison, error analysis
- `api/` — FastAPI prediction server
- `app/` — Streamlit dashboard
- `configs/` — model hyperparameter YAML files

## Conventions

- Python 3.10+, virtual env in `.venv/`
- Data stored as Parquet files (raw/ and processed/)
- Run scripts as modules: `python -m data.fetch_quantitative`
- Linting: ruff (config in pyproject.toml)
- Time-series validation only — never random splits (use `TimeSeriesSplit`)

## Current State

All 5 phases complete. Full pipeline: data ingestion → feature engineering → model training → evaluation → deployment. Models trained on 894 days of BTC-USD data. GTX 1070 GPU not compatible with installed PyTorch — LSTM trains on CPU (fast enough for this dataset size).
