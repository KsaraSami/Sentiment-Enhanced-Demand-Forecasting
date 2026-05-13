"""
FinBERT sentiment inference pipeline for crypto news headlines.

Loads headlines from data/raw/news_headlines.parquet, scores each with FinBERT,
aggregates to daily sentiment, and merges with the Fear & Greed Index.

Usage:
    python -m nlp.sentiment_pipeline                    # process + merge
    python -m nlp.sentiment_pipeline --skip-finbert     # merge FnG only (no GPU needed)
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

FINBERT_MODEL = "ProsusAI/finbert"


def load_finbert():
    """Load FinBERT model and tokenizer (lazy import to keep startup fast)."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    logger.info("Loading FinBERT model: %s", FINBERT_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
    model.eval()
    return tokenizer, model


def score_headlines(headlines: list[str], tokenizer, model, batch_size: int = 16) -> list[dict]:
    """Run FinBERT inference on a list of headlines. Returns sentiment scores."""
    import torch

    label_map = {0: "positive", 1: "negative", 2: "neutral"}
    results = []

    for i in range(0, len(headlines), batch_size):
        batch = headlines[i : i + batch_size]
        inputs = tokenizer(
            batch, padding=True, truncation=True, max_length=128, return_tensors="pt"
        )

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        for j, prob in enumerate(probs):
            pos, neg, neu = prob[0].item(), prob[1].item(), prob[2].item()
            # Sentiment score: positive - negative (range: -1 to +1)
            score = pos - neg
            predicted_label = label_map[prob.argmax().item()]

            results.append({
                "sentiment_score": round(score, 4),
                "prob_positive": round(pos, 4),
                "prob_negative": round(neg, 4),
                "prob_neutral": round(neu, 4),
                "sentiment_label": predicted_label,
            })

        if (i // batch_size) % 10 == 0:
            logger.info("  Scored %d / %d headlines", min(i + batch_size, len(headlines)), len(headlines))

    return results


def run_finbert_pipeline() -> pd.DataFrame | None:
    """Score news headlines with FinBERT and return per-headline results."""
    headlines_path = RAW_DIR / "news_headlines.parquet"
    if not headlines_path.exists():
        logger.warning("No headlines file found at %s — run fetch_sentiment first", headlines_path)
        return None

    df = pd.read_parquet(headlines_path)
    logger.info("Loaded %d headlines for scoring", len(df))

    tokenizer, model = load_finbert()
    scores = score_headlines(df["title"].tolist(), tokenizer, model)
    scores_df = pd.DataFrame(scores)

    scored = pd.concat([df.reset_index(), scores_df], axis=1)
    scored["date"] = pd.to_datetime(scored["date"])
    scored = scored.set_index("date")

    scored_path = RAW_DIR / "news_scored.parquet"
    scored.to_parquet(scored_path)
    logger.info("Saved scored headlines → %s", scored_path)

    return scored


def aggregate_daily_news_sentiment(scored_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-headline scores to daily sentiment features."""
    daily = scored_df.groupby(scored_df.index).agg(
        news_sentiment_mean=("sentiment_score", "mean"),
        news_sentiment_std=("sentiment_score", "std"),
        news_volume=("sentiment_score", "count"),
        news_positive_ratio=("sentiment_label", lambda x: (x == "positive").mean()),
        news_negative_ratio=("sentiment_label", lambda x: (x == "negative").mean()),
    )
    daily["news_sentiment_std"] = daily["news_sentiment_std"].fillna(0)
    return daily


def load_fear_greed() -> pd.DataFrame | None:
    """Load the Fear & Greed Index from raw data."""
    path = RAW_DIR / "fear_greed.parquet"
    if not path.exists():
        logger.warning("No Fear & Greed file found at %s — run fetch_sentiment first", path)
        return None

    df = pd.read_parquet(path)
    # Normalize to -1 to +1 range (original is 0-100)
    df["fng_normalized"] = (df["fng_value"] - 50) / 50
    logger.info("Loaded Fear & Greed Index: %d days", len(df))
    return df


def merge_sentiment_features(
    fng_df: pd.DataFrame | None,
    news_daily_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Merge Fear & Greed and news sentiment into a single daily table."""
    frames = []

    if fng_df is not None:
        frames.append(fng_df)
    if news_daily_df is not None:
        frames.append(news_daily_df)

    if not frames:
        raise ValueError("No sentiment data available to merge")

    if len(frames) == 1:
        return frames[0]

    merged = frames[0].join(frames[1], how="outer")
    logger.info("Merged sentiment features: %d days, %d columns", len(merged), len(merged.columns))
    return merged


def run(skip_finbert: bool = False) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    fng_df = load_fear_greed()

    news_daily_df = None
    if not skip_finbert:
        scored_df = run_finbert_pipeline()
        if scored_df is not None:
            news_daily_df = aggregate_daily_news_sentiment(scored_df)
    else:
        logger.info("Skipping FinBERT (--skip-finbert flag)")

    merged = merge_sentiment_features(fng_df, news_daily_df)

    output_path = PROCESSED_DIR / "sentiment_features.parquet"
    merged.to_parquet(output_path)
    logger.info("Saved → %s (%d rows, %d columns)", output_path, len(merged), len(merged.columns))


def main():
    parser = argparse.ArgumentParser(description="Run FinBERT sentiment pipeline")
    parser.add_argument(
        "--skip-finbert",
        action="store_true",
        help="Skip FinBERT inference — just merge Fear & Greed data",
    )
    args = parser.parse_args()
    run(skip_finbert=args.skip_finbert)


if __name__ == "__main__":
    main()
