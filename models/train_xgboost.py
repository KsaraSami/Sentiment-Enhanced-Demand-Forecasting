"""
Train XGBoost classifier for next-day BTC price direction.

Runs two experiments:
  1. All features (with sentiment)
  2. Price-only features (without sentiment) — to measure sentiment uplift

Uses TimeSeriesSplit for validation and Optuna for hyperparameter tuning.

Usage:
    python -m models.train_xgboost                        # full pipeline
    python -m models.train_xgboost --skip-tuning           # skip Optuna, use config defaults
    python -m models.train_xgboost --ticker ETH-USD
"""

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CONFIGS_DIR = PROJECT_ROOT / "configs"
SAVED_DIR = PROJECT_ROOT / "models" / "saved"

# Raw price columns leak non-stationary scale information across time splits
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
    path = PROCESSED_DIR / f"{safe}_model_ready.parquet"
    df = pd.read_parquet(path)
    logger.info("Loaded %d rows from %s", len(df), path.name)
    return df


def get_feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    """Return feature lists for 'all' and 'no_sentiment' experiments."""
    all_features = sorted(set(df.columns) - EXCLUDE_FROM_FEATURES)
    no_sentiment = sorted(set(all_features) - SENTIMENT_FEATURES)
    return {"all_features": all_features, "no_sentiment": no_sentiment}


def load_config() -> dict:
    path = CONFIGS_DIR / "xgboost_params.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def evaluate_cv(model_cls, params: dict, X: pd.DataFrame, y: pd.Series, n_splits: int) -> dict:
    """Evaluate with TimeSeriesSplit and return aggregate metrics."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = model_cls(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

        fold_metrics.append({
            "fold": fold,
            "f1": f1_score(y_val, y_pred),
            "accuracy": accuracy_score(y_val, y_pred),
            "auc_roc": roc_auc_score(y_val, y_prob),
        })

    metrics_df = pd.DataFrame(fold_metrics)
    return {
        "f1_mean": metrics_df["f1"].mean(),
        "f1_std": metrics_df["f1"].std(),
        "accuracy_mean": metrics_df["accuracy"].mean(),
        "accuracy_std": metrics_df["accuracy"].std(),
        "auc_roc_mean": metrics_df["auc_roc"].mean(),
        "auc_roc_std": metrics_df["auc_roc"].std(),
        "per_fold": fold_metrics,
    }


def tune_with_optuna(X: pd.DataFrame, y: pd.Series, n_splits: int, n_trials: int = 50) -> dict:
    """Use Optuna to find optimal hyperparameters."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "early_stopping_rounds": 50,
            "random_state": 42,
        }

        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = XGBClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            y_pred = model.predict(X_val)
            scores.append(f1_score(y_val, y_pred))

        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info("Optuna best F1: %.4f", study.best_value)
    logger.info("Optuna best params: %s", study.best_params)

    best = study.best_params
    best["objective"] = "binary:logistic"
    best["eval_metric"] = "logloss"
    best["early_stopping_rounds"] = 50
    best["random_state"] = 42
    return best


def train_final_model(
    X: pd.DataFrame, y: pd.Series, params: dict, test_ratio: float = 0.2
) -> tuple:
    """Train on train split, evaluate on held-out test set (last N%)."""
    split_idx = int(len(X) * (1 - test_ratio))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, target_names=["Down", "Up"], output_dict=True)
    test_metrics = {
        "f1": f1_score(y_test, y_pred),
        "accuracy": accuracy_score(y_test, y_pred),
        "auc_roc": roc_auc_score(y_test, y_prob),
        "classification_report": report,
        "test_size": len(X_test),
        "train_size": len(X_train),
    }

    return model, test_metrics, X_test.index


def get_feature_importance(model, feature_names: list[str], top_n: int = 20) -> pd.DataFrame:
    importances = model.feature_importances_
    fi = pd.DataFrame({"feature": feature_names, "importance": importances})
    fi = fi.sort_values("importance", ascending=False).head(top_n)
    return fi


def run(ticker: str, skip_tuning: bool = False) -> None:
    SAVED_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data(ticker)
    feature_sets = get_feature_sets(df)
    config = load_config()
    y = df["target"]
    n_splits = config["training"]["n_splits"]

    results = {}

    for experiment_name, features in feature_sets.items():
        logger.info("=" * 60)
        logger.info("Experiment: %s (%d features)", experiment_name, len(features))
        logger.info("=" * 60)

        X = df[features]

        # Hyperparameter tuning
        if skip_tuning:
            params = {
                **config["model"],
                "random_state": config["training"]["random_state"],
            }
            logger.info("Using config defaults (--skip-tuning)")
        else:
            logger.info("Running Optuna tuning (50 trials)...")
            params = tune_with_optuna(X, y, n_splits)

        # Cross-validation evaluation
        logger.info("Running %d-fold TimeSeriesSplit CV...", n_splits)
        cv_metrics = evaluate_cv(XGBClassifier, params, X, y, n_splits)
        logger.info(
            "CV results — F1: %.4f ± %.4f | Accuracy: %.4f ± %.4f | AUC: %.4f ± %.4f",
            cv_metrics["f1_mean"], cv_metrics["f1_std"],
            cv_metrics["accuracy_mean"], cv_metrics["accuracy_std"],
            cv_metrics["auc_roc_mean"], cv_metrics["auc_roc_std"],
        )

        # Train final model on chronological train/test split
        model, test_metrics, test_dates = train_final_model(
            X, y, params, config["training"]["test_size"]
        )
        logger.info(
            "Test set — F1: %.4f | Accuracy: %.4f | AUC: %.4f",
            test_metrics["f1"], test_metrics["accuracy"], test_metrics["auc_roc"],
        )

        # Feature importance
        fi = get_feature_importance(model, features)
        logger.info("Top 10 features:\n%s", fi.head(10).to_string(index=False))

        # Save artifacts
        safe = ticker.lower().replace("-", "_")
        prefix = f"{safe}_xgb_{experiment_name}"

        joblib.dump(model, SAVED_DIR / f"{prefix}_model.joblib")
        fi.to_csv(SAVED_DIR / f"{prefix}_feature_importance.csv", index=False)

        results[experiment_name] = {
            "features_used": features,
            "n_features": len(features),
            "params": {k: v for k, v in params.items() if not callable(v)},
            "cv_metrics": {k: v for k, v in cv_metrics.items() if k != "per_fold"},
            "test_metrics": {k: v for k, v in test_metrics.items() if k != "classification_report"},
            "top_features": fi.head(10).to_dict("records"),
        }

    # Save comparison results
    results_path = SAVED_DIR / f"{safe}_xgb_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Saved results → %s", results_path)

    # Print uplift summary
    if "all_features" in results and "no_sentiment" in results:
        print("\n" + "=" * 60)
        print("SENTIMENT UPLIFT ANALYSIS")
        print("=" * 60)
        for metric in ["f1", "accuracy", "auc_roc"]:
            cv_key = f"{metric}_mean"
            all_val = results["all_features"]["cv_metrics"][cv_key]
            no_val = results["no_sentiment"]["cv_metrics"][cv_key]
            diff = all_val - no_val
            print(f"  {metric.upper():>10s} (CV):  with={all_val:.4f}  without={no_val:.4f}  Δ={diff:+.4f}")

        print()
        for metric in ["f1", "accuracy", "auc_roc"]:
            all_val = results["all_features"]["test_metrics"][metric]
            no_val = results["no_sentiment"]["test_metrics"][metric]
            diff = all_val - no_val
            print(f"  {metric.upper():>10s} (test): with={all_val:.4f}  without={no_val:.4f}  Δ={diff:+.4f}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost baseline")
    parser.add_argument("--ticker", default="BTC-USD")
    parser.add_argument("--skip-tuning", action="store_true", help="Skip Optuna, use YAML defaults")
    args = parser.parse_args()
    run(args.ticker, args.skip_tuning)


if __name__ == "__main__":
    main()
