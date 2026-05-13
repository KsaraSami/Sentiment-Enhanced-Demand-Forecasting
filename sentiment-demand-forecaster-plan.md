# Sentiment-Enhanced Demand Forecasting System — Project Plan

## Table of Contents

- [Project Overview](#project-overview)
- [Phase 1: Data Architecture](#phase-1-data-architecture)
  - [1.1 Quantitative Pipeline](#11-quantitative-pipeline-the-what)
  - [1.2 Qualitative Pipeline](#12-qualitative-pipeline-the-why)
- [Phase 2: Feature Engineering](#phase-2-feature-engineering)
- [Phase 3: Modeling Strategy](#phase-3-modeling-strategy)
  - [Model A: Baseline (XGBoost / LightGBM)](#model-a-baseline-xgboost--lightgbm)
  - [Model B: Advanced (LSTM / Temporal Fusion Transformer)](#model-b-advanced-lstm--temporal-fusion-transformer)
- [Phase 4: Evaluation & Error Analysis](#phase-4-evaluation--error-analysis)
- [Phase 5: Deployment](#phase-5-deployment)
- [Tech Stack Summary](#tech-stack-summary)
- [Suggested Timeline](#suggested-timeline)
- [CV Description Template](#cv-description-template)

---

## Project Overview

**Problem statement:** Predict demand (price direction or magnitude) for an asset or product by combining traditional quantitative market data with real-time social sentiment — solving the "Cold Start" problem where historical data alone cannot capture viral trends or sudden shifts in public perception.

**Core idea:** Two parallel data pipelines (quantitative + qualitative) merge on a shared daily timestamp. A hybrid modeling approach compares a tree-based baseline against a sequence-aware deep learning model to measure the incremental value of sentiment features.

---

## Phase 1: Data Architecture

Build two independent pipelines that converge on a shared daily timestamp index.

### 1.1 Quantitative Pipeline (The "What")

**Objective:** Collect and store historical market/sales data with standard technical indicators.

| Item | Detail |
|------|--------|
| **Data sources** | `yfinance` (stocks), `ccxt` (crypto), or a Kaggle retail dataset (Walmart/Amazon) |
| **Core features** | Open, Close, High, Low, Volume |
| **Derived features** | SMA-50, SMA-200, daily returns, volatility (rolling std dev) |
| **Storage** | Parquet files (efficient columnar format for time-series) or local SQLite |
| **Granularity** | Daily |

**Deliverables:**
- Data ingestion script (`data/fetch_quantitative.py`)
- Parquet or SQLite output with clean, deduplicated daily records
- Basic validation checks (missing dates, null values, outlier detection)

### 1.2 Qualitative Pipeline (The "Why")

**Objective:** Extract daily sentiment signals from social media or news sources.

| Item | Detail |
|------|--------|
| **Data sources** | Reddit via `PRAW`, or news headlines via a news API |
| **NLP model** | FinBERT (finance domain) or Twitter-RoBERTa (general consumer sentiment) via Hugging Face |
| **Output features** | `sentiment_score` (−1.0 to +1.0), `sentiment_volume` (post/article count) |
| **Storage** | Same Parquet/SQLite store, keyed by date |

**Deliverables:**
- Scraping/collection script (`data/fetch_sentiment.py`)
- NLP inference pipeline (`nlp/sentiment_pipeline.py`)
- Daily aggregated sentiment table merged into the main dataset

---

## Phase 2: Feature Engineering

This phase bridges the quantitative and qualitative data into a single model-ready dataset.

**Key transformations:**

| Feature | Rationale |
|---------|-----------|
| `sentiment_lag_1`, `sentiment_lag_3`, `sentiment_lag_7` | Sentiment today typically affects price tomorrow or later — lagged features capture this delay |
| `sentiment_sma_7` | 7-day rolling average smooths noise from trolls, bots, or one-off spikes |
| `volume_sentiment_ratio` | Trade volume relative to sentiment volume — detects "hype vs. action" divergence |
| `price_momentum` | Rate of change over 5/10/20 days |
| `volatility_20d` | Rolling 20-day standard deviation of returns |

**Target variable options:**

| Target | Type | Notes |
|--------|------|-------|
| Next-day price (`y_t+1`) | Regression | Harder to evaluate, but more granular |
| Directional movement (1 = Up, 0 = Down) | Classification | More robust for a portfolio project — recommended |

**Deliverables:**
- Feature engineering script (`features/build_features.py`)
- Merged, feature-rich dataset saved to `data/processed/`
- Exploratory data analysis notebook (`notebooks/01_eda.ipynb`) with correlation heatmaps and distribution plots

---

## Phase 3: Modeling Strategy

Compare two model families to demonstrate critical thinking and measure the value of sentiment.

### Model A: Baseline (XGBoost / LightGBM)

| Aspect | Detail |
|--------|--------|
| **Why** | Tree-based models are state-of-the-art for tabular data; fast to train and interpret |
| **Input** | All engineered features as flat columns |
| **Key output** | Feature importance plots showing whether sentiment features rank high |
| **Hyperparameter tuning** | Optuna or GridSearchCV with `TimeSeriesSplit` |

### Model B: Advanced (LSTM / Temporal Fusion Transformer)

| Aspect | Detail |
|--------|--------|
| **Why** | Sequence-aware models capture temporal dependencies that tree models miss |
| **Architecture** | Hidden state updated by both numerical inputs and sentiment embeddings: `h_t = LSTM(x_t, s_t, h_{t-1})` |
| **Framework** | PyTorch or PyTorch Lightning |
| **Key output** | Attention weights or hidden-state analysis showing how the model weighs sentiment over time |

**Deliverables:**
- Model training scripts (`models/train_xgboost.py`, `models/train_lstm.py`)
- Hyperparameter configs (`configs/`)
- Saved model artifacts (`models/saved/`)
- Comparison notebook (`notebooks/02_model_comparison.ipynb`)

---

## Phase 4: Evaluation & Error Analysis

**Validation strategy:** Use `TimeSeriesSplit` from scikit-learn — never random splits on time-series data, to avoid future-leaking.

**Metrics:**

| Task | Primary Metric | Secondary |
|------|---------------|-----------|
| Classification (direction) | F1-Score | Accuracy, AUC-ROC |
| Regression (price) | MAE | RMSE, MAPE |

**Error analysis checklist:**
- Identify days with largest prediction errors
- Cross-reference with real-world events (earnings calls, regulatory news, viral posts)
- Document at least one "Black Swan" failure case and explain why the model missed it
- Compare "with sentiment" vs. "without sentiment" model performance to quantify the uplift

**Deliverables:**
- Evaluation script (`evaluation/evaluate.py`)
- Error analysis notebook (`notebooks/03_error_analysis.ipynb`)
- Performance comparison table (with vs. without sentiment features)

---

## Phase 5: Deployment

Make the project interactive and accessible.

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Backend API** | FastAPI | Serve predictions, expose `/predict` and `/sentiment` endpoints |
| **Frontend dashboard** | Streamlit | Interactive UI with two core charts |
| **Hosting** | Streamlit Community Cloud or Hugging Face Spaces | Free-tier deployment |

**Dashboard charts:**
1. **Price vs. Sentiment trend line** — overlay historical price with the rolling sentiment score
2. **7-day forecast** — show predicted direction or price with confidence intervals

**Deliverables:**
- FastAPI app (`api/main.py`)
- Streamlit app (`app/dashboard.py`)
- Deployment configs (`Dockerfile`, `requirements.txt`, Streamlit/HF Space config)
- Live URL to share with recruiters

---

## Tech Stack Summary

| Category | Tools |
|----------|-------|
| **Language** | Python 3.10+ |
| **Data** | yfinance, ccxt, PRAW, pandas, parquet/SQLite |
| **NLP** | Hugging Face Transformers (FinBERT / Twitter-RoBERTa) |
| **ML/DL** | scikit-learn, XGBoost/LightGBM, PyTorch |
| **Evaluation** | scikit-learn (TimeSeriesSplit, metrics), matplotlib, seaborn |
| **API** | FastAPI, uvicorn |
| **Frontend** | Streamlit |
| **Deployment** | Docker, Streamlit Community Cloud / Hugging Face Spaces |
| **Version control** | Git + GitHub |

---

## Suggested Timeline

| Week | Phase | Key Milestones |
|------|-------|----------------|
| 1 | Data Architecture | Quantitative pipeline working, initial data in Parquet |
| 2 | Data Architecture | Sentiment pipeline working, NLP model tested, daily scores generated |
| 3 | Feature Engineering | Merged dataset, lagged/rolling features, EDA notebook complete |
| 4 | Modeling (Baseline) | XGBoost trained, feature importance plotted, baseline metrics logged |
| 5 | Modeling (Advanced) | LSTM/TFT trained, comparison notebook complete |
| 6 | Evaluation | Error analysis done, Black Swan case documented, uplift quantified |
| 7 | Deployment | FastAPI + Streamlit running locally, dashboard charts working |
| 8 | Polish & Deploy | Deployed to cloud, README finalized, CV bullet points written |

---

## CV Description Template

> **Sentiment-Augmented Demand Forecaster** | Python, PyTorch, Transformers, XGBoost
>
> - Developed a hybrid forecasting engine merging 2+ years of market data with 50k+ scraped social media posts.
> - Implemented a FinBERT pipeline to extract daily market sentiment, improving price direction prediction accuracy by X% over baseline models.
> - Engineered a multi-stage data pipeline with lagged features and rolling sentiment averages to capture delayed market reactions.
> - Deployed an interactive Streamlit dashboard for real-time visualization of "Hype vs. Value" metrics.

---

## Project Structure

```
sentiment-demand-forecaster/
├── README.md
├── requirements.txt
├── Dockerfile
├── configs/
│   ├── xgboost_params.yaml
│   └── lstm_params.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   ├── fetch_quantitative.py
│   └── fetch_sentiment.py
├── nlp/
│   └── sentiment_pipeline.py
├── features/
│   └── build_features.py
├── models/
│   ├── train_xgboost.py
│   ├── train_lstm.py
│   └── saved/
├── evaluation/
│   └── evaluate.py
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_model_comparison.ipynb
│   └── 03_error_analysis.ipynb
├── api/
│   └── main.py
└── app/
    └── dashboard.py
```
