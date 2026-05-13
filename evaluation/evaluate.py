"""
Evaluate all trained models on the held-out test set.

Loads XGBoost and LSTM models, generates predictions with confidence scores,
identifies worst prediction days, and saves a unified evaluation report.

Usage:
    python -m evaluation.evaluate
    python -m evaluation.evaluate --ticker ETH-USD
"""

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SAVED_DIR = PROJECT_ROOT / "models" / "saved"
EVAL_DIR = PROJECT_ROOT / "evaluation"

EXCLUDE_FROM_FEATURES = {"close", "open", "high", "low", "volume", "sma_50", "sma_200", "target"}

SENTIMENT_FEATURES = {
    "fng_value", "fng_normalized", "fng_normalized_lag_1", "fng_normalized_lag_3",
    "fng_normalized_lag_7", "fng_sma_7", "fng_sma_14", "fng_momentum_5d",
    "fng_momentum_10d", "extreme_fear", "extreme_greed",
    "news_sentiment_mean", "news_sentiment_std", "news_volume",
    "news_positive_ratio", "news_negative_ratio",
    "news_sentiment_mean_lag_1", "news_sentiment_mean_lag_3",
    "news_sentiment_mean_lag_7", "news_sentiment_sma_7",
    "volume_fng_ratio", "price_sentiment_divergence",
}


def load_data(ticker: str) -> pd.DataFrame:
    safe = ticker.lower().replace("-", "_")
    return pd.read_parquet(PROCESSED_DIR / f"{safe}_model_ready.parquet")


def get_test_split(df: pd.DataFrame, test_ratio: float = 0.2):
    split_idx = int(len(df) * (1 - test_ratio))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def get_feature_lists(df: pd.DataFrame) -> dict[str, list[str]]:
    all_features = sorted(set(df.columns) - EXCLUDE_FROM_FEATURES)
    no_sentiment = sorted(set(all_features) - SENTIMENT_FEATURES)
    return {"all_features": all_features, "no_sentiment": no_sentiment}


# ---------------------------------------------------------------------------
# XGBoost evaluation
# ---------------------------------------------------------------------------

def evaluate_xgboost(df_test: pd.DataFrame, features: list[str], model_path: Path) -> dict:
    model = joblib.load(model_path)
    X = df_test[features]
    y = df_test["target"]

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    fpr, tpr, _ = roc_curve(y, y_prob)

    return {
        "y_true": y.values,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "dates": df_test.index,
        "fpr": fpr,
        "tpr": tpr,
        "metrics": {
            "f1": f1_score(y, y_pred),
            "accuracy": accuracy_score(y, y_pred),
            "auc_roc": roc_auc_score(y, y_prob),
            "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
            "report": classification_report(y, y_pred, target_names=["Down", "Up"], output_dict=True),
        },
    }


# ---------------------------------------------------------------------------
# LSTM evaluation
# ---------------------------------------------------------------------------

def evaluate_lstm(df: pd.DataFrame, features: list[str], model_path: Path, test_ratio: float = 0.2) -> dict:
    from models.train_lstm import LSTMClassifier, create_sequences

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    seq_len = config["model"]["sequence_length"]

    X_raw = df[features].values
    y_all = df["target"].values
    split_idx = int(len(X_raw) * (1 - test_ratio))

    scaler = StandardScaler()
    X_raw[:split_idx] = scaler.fit_transform(X_raw[:split_idx])
    X_raw[split_idx:] = scaler.transform(X_raw[split_idx:])

    X_seq, y_seq = create_sequences(X_raw, y_all, seq_len)
    seq_split = split_idx - seq_len
    X_test, y_test = X_seq[seq_split:], y_seq[seq_split:]

    test_dates = df.index[split_idx:]

    model = LSTMClassifier(
        input_size=len(features),
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        logits = model(torch.FloatTensor(X_test))
        y_prob = torch.sigmoid(logits).numpy()
        y_pred = (y_prob > 0.5).astype(int)

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    return {
        "y_true": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "dates": test_dates,
        "fpr": fpr,
        "tpr": tpr,
        "metrics": {
            "f1": f1_score(y_test, y_pred),
            "accuracy": accuracy_score(y_test, y_pred),
            "auc_roc": roc_auc_score(y_test, y_prob),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "report": classification_report(y_test, y_pred, target_names=["Down", "Up"], output_dict=True),
        },
    }


# ---------------------------------------------------------------------------
# Error analysis helpers
# ---------------------------------------------------------------------------

def find_worst_predictions(dates, y_true, y_prob, df_full: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Find days where the model was most confidently wrong."""
    confidence = np.where(y_true == 1, y_prob, 1 - y_prob)
    error_score = np.where(y_true != (y_prob > 0.5).astype(int), 1 - confidence, 0)

    min_len = min(len(dates), len(error_score))
    dates = dates[:min_len]
    error_score = error_score[:min_len]
    y_true = y_true[:min_len]
    y_prob = y_prob[:min_len]

    worst_idx = np.argsort(error_score)[-top_n:][::-1]

    records = []
    for idx in worst_idx:
        if error_score[idx] == 0:
            continue
        date = dates[idx]
        row = df_full.loc[date] if date in df_full.index else {}
        records.append({
            "date": date,
            "actual": "Up" if y_true[idx] == 1 else "Down",
            "predicted_prob_up": round(float(y_prob[idx]), 4),
            "confidence_error": round(float(error_score[idx]), 4),
            "daily_return": round(float(row.get("daily_return", 0)) * 100, 2) if isinstance(row, pd.Series) else None,
            "fng_value": int(row.get("fng_value", 0)) if isinstance(row, pd.Series) else None,
            "volatility_20d": round(float(row.get("volatility_20d", 0)) * 100, 2) if isinstance(row, pd.Series) else None,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(ticker: str) -> None:
    safe = ticker.lower().replace("-", "_")
    df = load_data(ticker)
    _, df_test = get_test_split(df)
    feature_lists = get_feature_lists(df)

    logger.info("Test set: %d rows (%s → %s)", len(df_test), df_test.index.min().date(), df_test.index.max().date())

    all_results = {}
    model_configs = [
        ("xgb_all_features", "xgboost", "all_features"),
        ("xgb_no_sentiment", "xgboost", "no_sentiment"),
        ("lstm_all_features", "lstm", "all_features"),
        ("lstm_no_sentiment", "lstm", "no_sentiment"),
    ]

    for name, model_type, feat_key in model_configs:
        features = feature_lists[feat_key]
        model_path = SAVED_DIR / f"{safe}_{name}_model.{'joblib' if model_type == 'xgboost' else 'pt'}"

        if not model_path.exists():
            logger.warning("Model not found: %s — skipping", model_path)
            continue

        logger.info("Evaluating %s (%d features)...", name, len(features))

        if model_type == "xgboost":
            result = evaluate_xgboost(df_test, features, model_path)
        else:
            result = evaluate_lstm(df, features, model_path)

        worst = find_worst_predictions(result["dates"], result["y_true"], result["y_prob"], df)
        result["worst_predictions"] = worst

        all_results[name] = result

        logger.info(
            "  %s — F1: %.4f | Acc: %.4f | AUC: %.4f",
            name, result["metrics"]["f1"], result["metrics"]["accuracy"], result["metrics"]["auc_roc"],
        )

    # Save evaluation data for notebooks
    eval_data = {}
    for name, res in all_results.items():
        eval_data[name] = {
            "metrics": res["metrics"],
            "y_true": res["y_true"].tolist(),
            "y_pred": res["y_pred"].tolist(),
            "y_prob": res["y_prob"].tolist(),
            "dates": [str(d.date()) if hasattr(d, "date") else str(d) for d in res["dates"]],
            "fpr": res["fpr"].tolist(),
            "tpr": res["tpr"].tolist(),
            "worst_predictions": res["worst_predictions"].to_dict("records"),
        }

    output_path = EVAL_DIR / f"{safe}_evaluation.json"
    with open(output_path, "w") as f:
        json.dump(eval_data, f, indent=2, default=str)
    logger.info("Saved evaluation → %s", output_path)

    # Print summary table
    print("\n" + "=" * 75)
    print(f"{'Model':<25s} {'F1':>8s} {'Accuracy':>10s} {'AUC-ROC':>10s}")
    print("-" * 75)
    for name, res in all_results.items():
        m = res["metrics"]
        print(f"{name:<25s} {m['f1']:>8.4f} {m['accuracy']:>10.4f} {m['auc_roc']:>10.4f}")
    print("=" * 75)

    # Uplift summary
    print("\nSentiment Uplift:")
    for model_type in ["xgb", "lstm"]:
        with_key = f"{model_type}_all_features"
        without_key = f"{model_type}_no_sentiment"
        if with_key in all_results and without_key in all_results:
            for metric in ["f1", "accuracy", "auc_roc"]:
                w = all_results[with_key]["metrics"][metric]
                wo = all_results[without_key]["metrics"][metric]
                print(f"  {model_type.upper()} {metric}: {w:.4f} vs {wo:.4f} (Δ={w-wo:+.4f})")

    # Show worst predictions for best model
    best_name = max(all_results, key=lambda k: all_results[k]["metrics"]["auc_roc"])
    worst = all_results[best_name]["worst_predictions"]
    if not worst.empty:
        print(f"\nWorst predictions ({best_name}):")
        print(worst.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Evaluate all models")
    parser.add_argument("--ticker", default="BTC-USD")
    args = parser.parse_args()
    run(args.ticker)


if __name__ == "__main__":
    main()
