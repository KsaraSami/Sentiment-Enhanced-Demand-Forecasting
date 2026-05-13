"""
Fetch and process daily crypto market data from Yahoo Finance.

Usage:
    python -m data.fetch_quantitative                     # defaults: BTC-USD, 3 years
    python -m data.fetch_quantitative --tickers ETH-USD   # single ticker
    python -m data.fetch_quantitative --tickers BTC-USD ETH-USD --years 5
"""

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parent / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent / "processed"


def fetch_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download daily OHLCV data from Yahoo Finance."""
    logger.info("Fetching %s from %s to %s", ticker, start, end)
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for {ticker}")

    # yfinance may return MultiIndex columns for single tickers — flatten
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index.name = "date"
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    logger.info("Fetched %d rows for %s", len(df), ticker)
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived technical features from OHLCV data."""
    df = df.copy()

    # Simple Moving Averages
    df["sma_50"] = df["close"].rolling(window=50).mean()
    df["sma_200"] = df["close"].rolling(window=200).mean()

    # Daily returns (percentage)
    df["daily_return"] = df["close"].pct_change()

    # Price momentum (rate of change over N days)
    for period in [5, 10, 20]:
        df[f"momentum_{period}d"] = df["close"].pct_change(periods=period)

    # Volatility — rolling 20-day standard deviation of daily returns
    df["volatility_20d"] = df["daily_return"].rolling(window=20).std()

    # SMA crossover signal (1 when 50-day > 200-day)
    df["sma_crossover"] = (df["sma_50"] > df["sma_200"]).astype(int)

    # Volume moving average (for later ratio features)
    df["volume_sma_20"] = df["volume"].rolling(window=20).mean()

    return df


def validate(df: pd.DataFrame, ticker: str) -> None:
    """Run basic data quality checks and log warnings."""
    total = len(df)

    # Check for null values
    null_counts = df.isnull().sum()
    nulls_after_warmup = df.iloc[200:].isnull().sum()
    if nulls_after_warmup.any():
        cols = nulls_after_warmup[nulls_after_warmup > 0]
        logger.warning(
            "[%s] Null values after 200-day warmup period:\n%s", ticker, cols.to_string()
        )

    # Check for duplicate dates
    dupes = df.index.duplicated().sum()
    if dupes > 0:
        logger.warning("[%s] %d duplicate dates found — dropping", ticker, dupes)

    # Check for gaps (missing trading days)
    date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")
    missing_days = len(date_range) - total
    # Crypto trades 365 days/year, so gaps are real missing data
    gap_pct = missing_days / len(date_range) * 100
    if gap_pct > 5:
        logger.warning("[%s] %.1f%% of days missing (%d gaps)", ticker, gap_pct, missing_days)

    # Outlier detection on daily returns (beyond ±3 std)
    returns = df["daily_return"].dropna()
    mean, std = returns.mean(), returns.std()
    outliers = returns[returns.abs() > mean + 3 * std]
    if len(outliers) > 0:
        logger.info(
            "[%s] %d outlier return days detected (>3σ). Largest: %.2f%%",
            ticker,
            len(outliers),
            outliers.abs().max() * 100,
        )

    logger.info(
        "[%s] Validation complete — %d rows, %d nulls in warmup zone, %d post-warmup nulls",
        ticker,
        total,
        null_counts.sum(),
        nulls_after_warmup.sum(),
    )


def run(tickers: list[str], years: int) -> None:
    """Main pipeline: fetch → enrich → validate → save."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=365 * years)).strftime("%Y-%m-%d")

    for ticker in tickers:
        safe_name = ticker.lower().replace("-", "_")

        # Fetch raw OHLCV
        raw_df = fetch_ohlcv(ticker, start, end)
        raw_df = raw_df[~raw_df.index.duplicated(keep="first")]

        # Save raw
        raw_path = RAW_DIR / f"{safe_name}_ohlcv.parquet"
        raw_df.to_parquet(raw_path)
        logger.info("Saved raw data → %s", raw_path)

        # Enrich with technical indicators
        enriched_df = add_technical_indicators(raw_df)

        # Validate
        validate(enriched_df, ticker)

        # Save processed
        processed_path = PROCESSED_DIR / f"{safe_name}_features.parquet"
        enriched_df.to_parquet(processed_path)
        logger.info("Saved processed data → %s (%d rows, %d columns)",
                     processed_path, len(enriched_df), len(enriched_df.columns))


def main():
    parser = argparse.ArgumentParser(description="Fetch crypto market data")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=["BTC-USD"],
        help="Yahoo Finance ticker symbols (default: BTC-USD)",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=3,
        help="Years of historical data to fetch (default: 3)",
    )
    args = parser.parse_args()
    run(args.tickers, args.years)


if __name__ == "__main__":
    main()
