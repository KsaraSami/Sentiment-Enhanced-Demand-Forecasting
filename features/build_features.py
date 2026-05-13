"""
Merge quantitative and sentiment data into a single model-ready dataset.

Builds lag features, rolling averages, interaction features, and the
classification target (next-day price direction).

Usage:
    python -m features.build_features                          # BTC-USD (default)
    python -m features.build_features --ticker ETH-USD
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def load_quantitative(ticker: str) -> pd.DataFrame:
    safe = ticker.lower().replace("-", "_")
    path = PROCESSED_DIR / f"{safe}_features.parquet"
    df = pd.read_parquet(path)
    logger.info("Loaded quantitative: %d rows from %s", len(df), path.name)
    return df


def load_sentiment() -> pd.DataFrame:
    path = PROCESSED_DIR / "sentiment_features.parquet"
    df = pd.read_parquet(path)
    logger.info("Loaded sentiment: %d rows from %s", len(df), path.name)
    return df


def merge_datasets(quant: pd.DataFrame, sentiment: pd.DataFrame) -> pd.DataFrame:
    """Inner-join on date index so we only keep days with both price and sentiment."""
    merged = quant.join(sentiment, how="inner")
    logger.info("Merged dataset: %d rows, %d columns", len(merged), len(merged.columns))
    return merged


def add_sentiment_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Sentiment today affects price tomorrow — create lagged features."""
    df = df.copy()

    for col in ["fng_normalized", "news_sentiment_mean"]:
        if col not in df.columns:
            continue
        for lag in [1, 3, 7]:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)

    # Rolling averages to smooth noise
    if "fng_normalized" in df.columns:
        df["fng_sma_7"] = df["fng_normalized"].rolling(7).mean()
        df["fng_sma_14"] = df["fng_normalized"].rolling(14).mean()

    if "news_sentiment_mean" in df.columns:
        df["news_sentiment_sma_7"] = df["news_sentiment_mean"].rolling(7).mean()

    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-domain features that capture divergence between sentiment and market action."""
    df = df.copy()

    # Volume-sentiment ratio: high trade volume + low sentiment = potential divergence
    if "volume_sma_20" in df.columns and "fng_normalized" in df.columns:
        fng_abs = df["fng_normalized"].abs().replace(0, 0.01)
        df["volume_fng_ratio"] = df["volume_sma_20"] / (fng_abs * 1e10)

    # Price-sentiment divergence: price going up but sentiment going down (or vice versa)
    if "momentum_5d" in df.columns and "fng_normalized" in df.columns:
        df["price_sentiment_divergence"] = df["momentum_5d"] - df["fng_normalized"]

    # Sentiment momentum: rate of change in Fear & Greed
    if "fng_normalized" in df.columns:
        df["fng_momentum_5d"] = df["fng_normalized"].diff(5)
        df["fng_momentum_10d"] = df["fng_normalized"].diff(10)

    # Extreme sentiment flags
    if "fng_value" in df.columns:
        df["extreme_fear"] = (df["fng_value"] <= 20).astype(int)
        df["extreme_greed"] = (df["fng_value"] >= 80).astype(int)

    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Classification target: 1 if next-day close > today's close, else 0."""
    df = df.copy()
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    return df


def drop_non_numeric_and_leaky(df: pd.DataFrame) -> pd.DataFrame:
    """Remove string columns and features that would leak future info."""
    drop_cols = [
        "fng_classification",  # string — already encoded as fng_value/fng_normalized
    ]
    existing = [c for c in drop_cols if c in df.columns]
    if existing:
        df = df.drop(columns=existing)
    return df


def handle_sparse_news_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN in news columns with 0 — news data is sparse (RSS only covers recent days)."""
    df = df.copy()
    news_cols = [c for c in df.columns if c.startswith("news_")]
    if news_cols:
        filled = df[news_cols].isna().all(axis=0).sum()
        df[news_cols] = df[news_cols].fillna(0)
        logger.info("Filled NaN in %d news columns with 0 (sparse RSS coverage)", len(news_cols))
    return df


def run(ticker: str) -> None:
    quant = load_quantitative(ticker)
    sentiment = load_sentiment()

    df = merge_datasets(quant, sentiment)
    df = add_sentiment_lag_features(df)
    df = add_interaction_features(df)
    df = add_target(df)
    df = drop_non_numeric_and_leaky(df)
    df = handle_sparse_news_features(df)

    # Drop the last row (no target) and warmup rows (NaN from core features)
    df = df.iloc[:-1]
    rows_before = len(df)
    core_cols = [c for c in df.columns if not c.startswith("news_")]
    df_clean = df.dropna(subset=core_cols)
    rows_dropped = rows_before - len(df_clean)
    logger.info("Dropped %d warmup/NaN rows (%.1f%%)", rows_dropped, rows_dropped / rows_before * 100)

    safe = ticker.lower().replace("-", "_")
    output_path = PROCESSED_DIR / f"{safe}_model_ready.parquet"
    df_clean.to_parquet(output_path)

    logger.info(
        "Saved → %s (%d rows, %d features + target)",
        output_path, len(df_clean), len(df_clean.columns) - 1,
    )

    # Summary
    target_dist = df_clean["target"].value_counts(normalize=True)
    logger.info(
        "Target balance: Up=%.1f%% Down=%.1f%%",
        target_dist.get(1, 0) * 100,
        target_dist.get(0, 0) * 100,
    )

    print(f"\nFeature list ({len(df_clean.columns)} total):")
    for col in sorted(df_clean.columns):
        print(f"  {col}")


def main():
    parser = argparse.ArgumentParser(description="Build model-ready features")
    parser.add_argument("--ticker", default="BTC-USD", help="Ticker symbol (default: BTC-USD)")
    args = parser.parse_args()
    run(args.ticker)


if __name__ == "__main__":
    main()
