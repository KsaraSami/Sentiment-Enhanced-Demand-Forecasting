"""
Streamlit dashboard for Sentiment-Enhanced Demand Forecasting.

Charts:
  1. Price vs Sentiment trend line (historical overlay)
  2. Model predictions with confidence (test period)
  3. Live prediction using current market + sentiment data

Usage:
    streamlit run app/dashboard.py
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EVAL_DIR = PROJECT_ROOT / "evaluation"
SAVED_DIR = PROJECT_ROOT / "models" / "saved"

st.set_page_config(
    page_title="BTC Sentiment Forecaster",
    page_icon="📊",
    layout="wide",
)


@st.cache_data(ttl=3600)
def load_model_data():
    df = pd.read_parquet(PROCESSED_DIR / "btc_usd_model_ready.parquet")
    with open(EVAL_DIR / "btc_usd_evaluation.json") as f:
        eval_data = json.load(f)
    return df, eval_data


@st.cache_data(ttl=300)
def fetch_live_fng():
    try:
        resp = requests.get(
            "https://api.alternative.me/fng/",
            params={"limit": 1, "format": "json"},
            timeout=10,
        )
        resp.raise_for_status()
        entry = resp.json()["data"][0]
        return {
            "value": int(entry["value"]),
            "classification": entry["value_classification"],
        }
    except Exception:
        return None


@st.cache_data(ttl=300)
def fetch_live_price():
    try:
        df = yf.download("BTC-USD", period="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return float(df["Close"].iloc[-1])
    except Exception:
        return None


def fng_color(value: int) -> str:
    if value <= 20:
        return "🔴"
    elif value <= 40:
        return "🟠"
    elif value <= 60:
        return "🟡"
    elif value <= 80:
        return "🟢"
    else:
        return "🔵"


def render_header():
    st.title("BTC Sentiment-Enhanced Demand Forecaster")
    st.markdown("Predicting next-day BTC price direction using market data + social sentiment")

    col1, col2, col3 = st.columns(3)

    price = fetch_live_price()
    fng = fetch_live_fng()

    with col1:
        if price:
            st.metric("BTC Price", f"${price:,.0f}")
        else:
            st.metric("BTC Price", "Unavailable")

    with col2:
        if fng:
            st.metric(
                "Fear & Greed",
                f"{fng_color(fng['value'])} {fng['value']}",
                fng["classification"],
            )
        else:
            st.metric("Fear & Greed", "Unavailable")

    with col3:
        st.metric("Model", "LSTM + Sentiment", "Best AUC-ROC")


def render_price_vs_sentiment(df: pd.DataFrame):
    st.subheader("Price vs. Sentiment Trend")

    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(14, 5))

    color_price = "#2c3e50"
    color_fng = "#e67e22"

    ax1.plot(df.index, df["close"], color=color_price, linewidth=1, label="BTC Close")
    ax1.set_ylabel("BTC Price (USD)", color=color_price)
    ax1.tick_params(axis="y", labelcolor=color_price)

    ax2 = ax1.twinx()
    ax2.fill_between(df.index, df["fng_value"], alpha=0.25, color=color_fng)
    ax2.plot(df.index, df["fng_sma_14"], color=color_fng, linewidth=1, alpha=0.8, label="FnG SMA-14")
    ax2.axhline(y=20, color="red", linestyle="--", alpha=0.3, linewidth=0.7)
    ax2.axhline(y=80, color="green", linestyle="--", alpha=0.3, linewidth=0.7)
    ax2.set_ylabel("Fear & Greed Index", color=color_fng)
    ax2.tick_params(axis="y", labelcolor=color_fng)
    ax2.set_ylim(0, 100)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax1.set_title("Historical BTC Price vs. Crypto Fear & Greed Index")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()


def render_predictions(eval_data: dict, df: pd.DataFrame):
    st.subheader("Model Predictions (Test Period)")

    model_choice = st.selectbox(
        "Select model",
        list(eval_data.keys()),
        format_func=lambda x: x.replace("_", " ").title(),
    )

    data = eval_data[model_choice]
    dates = pd.to_datetime(data["dates"])
    y_true = np.array(data["y_true"])
    y_prob = np.array(data["y_prob"])
    y_pred = (y_prob > 0.5).astype(int)
    correct = y_true == y_pred

    min_len = min(len(dates), len(y_true))

    col1, col2, col3, col4 = st.columns(4)
    m = data["metrics"]
    col1.metric("F1 Score", f"{m['f1']:.4f}")
    col2.metric("Accuracy", f"{m['accuracy']:.4f}")
    col3.metric("AUC-ROC", f"{m['auc_roc']:.4f}")
    col4.metric("Test Days", str(min_len))

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Price + predictions
    test_dates = dates[:min_len]
    test_prices = df.loc[test_dates, "close"] if all(d in df.index for d in test_dates) else None

    if test_prices is not None:
        axes[0].plot(test_dates, test_prices.values, color="#2c3e50", linewidth=1)
    axes[0].set_ylabel("BTC Price (USD)")
    axes[0].set_title(f"Predictions — {model_choice.replace('_', ' ').title()}")

    # Probability plot
    axes[1].fill_between(test_dates, 0.5, y_prob[:min_len],
                         where=y_prob[:min_len] >= 0.5, alpha=0.4, color="#2ecc71", label="Predict Up")
    axes[1].fill_between(test_dates, 0.5, y_prob[:min_len],
                         where=y_prob[:min_len] < 0.5, alpha=0.4, color="#e74c3c", label="Predict Down")
    axes[1].axhline(y=0.5, color="black", linewidth=0.5)
    axes[1].set_ylabel("P(Up)")
    axes[1].legend()

    # Mark incorrect predictions
    wrong_mask = ~correct[:min_len]
    if wrong_mask.any():
        axes[1].scatter(test_dates[wrong_mask], y_prob[:min_len][wrong_mask],
                        color="red", marker="x", s=30, zorder=5, label="Wrong")

    fig.tight_layout()
    st.pyplot(fig)
    plt.close()


def render_sentiment_uplift(eval_data: dict):
    st.subheader("Sentiment Uplift Analysis")

    import matplotlib.pyplot as plt

    uplift_rows = []
    for model_type, prefix in [("XGBoost", "xgb"), ("LSTM", "lstm")]:
        with_key = f"{prefix}_all_features"
        without_key = f"{prefix}_no_sentiment"
        if with_key in eval_data and without_key in eval_data:
            for metric in ["f1", "accuracy", "auc_roc"]:
                w = eval_data[with_key]["metrics"][metric]
                wo = eval_data[without_key]["metrics"][metric]
                uplift_rows.append({
                    "Model": model_type,
                    "Metric": metric.upper().replace("_", "-"),
                    "With Sentiment": w,
                    "Without": wo,
                    "Uplift": w - wo,
                })

    if uplift_rows:
        uplift_df = pd.DataFrame(uplift_rows)

        fig, ax = plt.subplots(figsize=(10, 4))
        x = np.arange(len(uplift_df))
        colors = ["#2ecc71" if u >= 0 else "#e74c3c" for u in uplift_df["Uplift"]]
        ax.bar(x, uplift_df["Uplift"], color=colors, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r['Model']}\n{r['Metric']}" for _, r in uplift_df.iterrows()], fontsize=9)
        ax.axhline(y=0, color="black", linewidth=0.8)
        ax.set_ylabel("Uplift (with - without sentiment)")
        ax.set_title("Does Sentiment Improve Predictions?")

        for i, val in enumerate(uplift_df["Uplift"]):
            ax.text(i, val, f"{val:+.4f}", ha="center",
                    va="bottom" if val >= 0 else "top", fontsize=8)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.dataframe(uplift_df.set_index(["Model", "Metric"]), use_container_width=True)


def render_live_prediction():
    st.subheader("Live Prediction")
    st.markdown("Fetches current market data and sentiment to predict tomorrow's BTC direction.")

    if st.button("Get Live Prediction", type="primary"):
        with st.spinner("Fetching data and running prediction..."):
            try:
                resp = requests.get("http://localhost:8000/predict/latest", timeout=15)
                if resp.status_code == 200:
                    pred = resp.json()
                    direction = pred["direction"]
                    color = "green" if direction == "Up" else "red"

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Predicted Direction", f"{'📈' if direction == 'Up' else '📉'} {direction}")
                    col2.metric("P(Up)", f"{pred['probability_up']:.1%}")
                    col3.metric("Confidence", f"{pred['confidence']:.1%}")

                    st.info(f"Model: {pred['model']} | Timestamp: {pred['timestamp']}")
                else:
                    st.error(f"API returned {resp.status_code}. Is the FastAPI server running?")
            except requests.ConnectionError:
                st.warning(
                    "Could not connect to the API server. "
                    "Start it with: `uvicorn api.main:app --port 8000`"
                )


def main():
    render_header()
    st.divider()

    df, eval_data = load_model_data()

    tab1, tab2, tab3, tab4 = st.tabs([
        "Price vs Sentiment",
        "Model Predictions",
        "Sentiment Uplift",
        "Live Prediction",
    ])

    with tab1:
        render_price_vs_sentiment(df)

    with tab2:
        render_predictions(eval_data, df)

    with tab3:
        render_sentiment_uplift(eval_data)

    with tab4:
        render_live_prediction()

    st.divider()
    st.caption(
        "Built as a portfolio project demonstrating ML pipeline skills: "
        "data engineering, NLP (FinBERT), tree-based & deep learning models, "
        "and interactive deployment."
    )


if __name__ == "__main__":
    main()
