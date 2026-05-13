"""
Fetch sentiment data from two sources:
  1. Alternative.me Crypto Fear & Greed Index (full history)
  2. Crypto news headlines from RSS feeds (recent, for FinBERT pipeline)

Usage:
    python -m data.fetch_sentiment                # fetch both sources
    python -m data.fetch_sentiment --source fng    # Fear & Greed only
    python -m data.fetch_sentiment --source news   # news headlines only
"""

import argparse
import logging
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parent / "raw"

FNG_API_URL = "https://api.alternative.me/fng/"

NEWS_RSS_FEEDS = {
    "cointelegraph": "https://cointelegraph.com/rss",
    "coindesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
}

HEADERS = {
    "User-Agent": "SentimentDemandForecaster/0.1 (academic project)"
}


# ---------------------------------------------------------------------------
# Fear & Greed Index
# ---------------------------------------------------------------------------

def fetch_fear_greed() -> pd.DataFrame:
    """Fetch full history of the Crypto Fear & Greed Index."""
    logger.info("Fetching Fear & Greed Index (full history)...")

    resp = requests.get(FNG_API_URL, params={"limit": 0, "format": "json"}, timeout=30)
    resp.raise_for_status()
    data = resp.json()["data"]

    records = []
    for entry in data:
        records.append({
            "date": datetime.fromtimestamp(int(entry["timestamp"]), tz=timezone.utc).date(),
            "fng_value": int(entry["value"]),
            "fng_classification": entry["value_classification"],
        })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset="date").sort_values("date").set_index("date")

    logger.info(
        "Fear & Greed: %d days, %s → %s",
        len(df), df.index.min().date(), df.index.max().date(),
    )
    return df


# ---------------------------------------------------------------------------
# News Headlines (RSS)
# ---------------------------------------------------------------------------

def _parse_rss(xml_text: str, source: str) -> list[dict]:
    """Extract headlines from RSS XML."""
    root = ET.fromstring(xml_text)
    items = []

    for item in root.iter("item"):
        title_el = item.find("title")
        pub_el = item.find("pubDate")
        desc_el = item.find("description")

        if title_el is None or pub_el is None:
            continue

        try:
            pub_date = pd.to_datetime(pub_el.text.strip(), utc=True)
        except (ValueError, TypeError):
            continue

        items.append({
            "date": pub_date.normalize(),
            "published_at": pub_date,
            "source": source,
            "title": (title_el.text or "").strip(),
            "description": (desc_el.text or "").strip()[:500] if desc_el is not None else "",
        })

    return items


def fetch_news_headlines() -> pd.DataFrame:
    """Scrape recent crypto headlines from RSS feeds."""
    logger.info("Fetching news headlines from %d RSS feeds...", len(NEWS_RSS_FEEDS))
    all_items = []

    for source, url in NEWS_RSS_FEEDS.items():
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            items = _parse_rss(resp.text, source)
            logger.info("  %s: %d headlines", source, len(items))
            all_items.extend(items)
        except Exception as e:
            logger.warning("  %s: failed — %s", source, e)

        time.sleep(1)  # polite delay between feeds

    if not all_items:
        logger.warning("No headlines fetched from any source")
        return pd.DataFrame()

    df = pd.DataFrame(all_items)
    df = df.drop_duplicates(subset=["title"]).sort_values("published_at")
    df["date"] = pd.to_datetime(df["date"].dt.date)
    df = df.set_index("date")

    logger.info(
        "Headlines total: %d unique, spanning %s → %s",
        len(df), df.index.min().date(), df.index.max().date(),
    )
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(source: str = "all") -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if source in ("all", "fng"):
        fng_df = fetch_fear_greed()
        path = RAW_DIR / "fear_greed.parquet"
        fng_df.to_parquet(path)
        logger.info("Saved → %s", path)

    if source in ("all", "news"):
        news_df = fetch_news_headlines()
        if not news_df.empty:
            path = RAW_DIR / "news_headlines.parquet"
            news_df.to_parquet(path)
            logger.info("Saved → %s", path)


def main():
    parser = argparse.ArgumentParser(description="Fetch sentiment data")
    parser.add_argument(
        "--source",
        choices=["all", "fng", "news"],
        default="all",
        help="Which source to fetch (default: all)",
    )
    args = parser.parse_args()
    run(args.source)


if __name__ == "__main__":
    main()
