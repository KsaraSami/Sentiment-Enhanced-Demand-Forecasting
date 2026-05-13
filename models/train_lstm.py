"""
Train an LSTM classifier for next-day BTC price direction.

Takes the same model-ready dataset as XGBoost but reshapes it into
sequences (lookback windows) for the recurrent model.

Runs two experiments (with/without sentiment) to measure uplift.

Usage:
    python -m models.train_lstm
    python -m models.train_lstm --ticker ETH-USD
    python -m models.train_lstm --epochs 50 --sequence-length 20
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CONFIGS_DIR = PROJECT_ROOT / "configs"
SAVED_DIR = PROJECT_ROOT / "models" / "saved"

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

def _get_device() -> torch.device:
    if torch.cuda.is_available():
        try:
            cc = torch.cuda.get_device_capability()
            if cc[0] >= 7 and cc[1] >= 5:
                return torch.device("cuda")
        except Exception:
            pass
    return torch.device("cpu")


DEVICE = _get_device()


class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.head(last_hidden).squeeze(-1)


def create_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    """Reshape flat features into overlapping sequences for LSTM input."""
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len : i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


def load_config() -> dict:
    with open(CONFIGS_DIR / "lstm_params.yaml") as f:
        return yaml.safe_load(f)


def load_data(ticker: str) -> pd.DataFrame:
    safe = ticker.lower().replace("-", "_")
    path = PROCESSED_DIR / f"{safe}_model_ready.parquet"
    df = pd.read_parquet(path)
    logger.info("Loaded %d rows from %s", len(df), path.name)
    return df


def get_feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    all_features = sorted(set(df.columns) - EXCLUDE_FROM_FEATURES)
    no_sentiment = sorted(set(all_features) - SENTIMENT_FEATURES)
    return {"all_features": all_features, "no_sentiment": no_sentiment}


def train_model(
    X_train_seq: np.ndarray,
    y_train_seq: np.ndarray,
    X_val_seq: np.ndarray,
    y_val_seq: np.ndarray,
    config: dict,
    input_size: int,
) -> tuple[LSTMClassifier, list[dict]]:
    """Train LSTM with early stopping, return best model and training history."""
    mc = config["model"]
    tc = config["training"]

    model = LSTMClassifier(
        input_size=input_size,
        hidden_size=mc["hidden_size"],
        num_layers=mc["num_layers"],
        dropout=mc["dropout"],
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tc["learning_rate"], weight_decay=tc["weight_decay"]
    )
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    X_train_t = torch.FloatTensor(X_train_seq).to(DEVICE)
    y_train_t = torch.FloatTensor(y_train_seq).to(DEVICE)
    X_val_t = torch.FloatTensor(X_val_seq).to(DEVICE)
    y_val_t = torch.FloatTensor(y_val_seq).to(DEVICE)

    train_ds = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=tc["batch_size"], shuffle=False  # preserve temporal order
    )

    history = []
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(tc["epochs"]):
        # Train
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # Validate
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = criterion(val_logits, y_val_t).item()
            val_preds = (torch.sigmoid(val_logits) > 0.5).cpu().numpy().astype(int)
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            val_f1 = f1_score(y_val_seq, val_preds)
            val_acc = accuracy_score(y_val_seq, val_preds)

        scheduler.step(val_loss)
        train_loss = np.mean(train_losses)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_f1": val_f1,
            "val_accuracy": val_acc,
        })

        if epoch % 10 == 0 or epoch == tc["epochs"] - 1:
            logger.info(
                "  Epoch %3d — train_loss: %.4f  val_loss: %.4f  val_f1: %.4f  val_acc: %.4f",
                epoch, train_loss, val_loss, val_f1, val_acc,
            )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= tc["patience"]:
                logger.info("  Early stopping at epoch %d", epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def evaluate_model(model: LSTMClassifier, X_seq: np.ndarray, y_seq: np.ndarray) -> dict:
    model.eval()
    X_t = torch.FloatTensor(X_seq).to(DEVICE)

    with torch.no_grad():
        logits = model(X_t)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs > 0.5).astype(int)

    return {
        "f1": f1_score(y_seq, preds),
        "accuracy": accuracy_score(y_seq, preds),
        "auc_roc": roc_auc_score(y_seq, probs),
    }


def run(ticker: str, epochs: int | None = None, sequence_length: int | None = None) -> None:
    SAVED_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)

    df = load_data(ticker)
    feature_sets = get_feature_sets(df)
    config = load_config()

    if epochs is not None:
        config["training"]["epochs"] = epochs
    if sequence_length is not None:
        config["model"]["sequence_length"] = sequence_length

    seq_len = config["model"]["sequence_length"]
    test_ratio = config["training"]["test_size"]
    y_all = df["target"].values

    results = {}

    for experiment_name, features in feature_sets.items():
        logger.info("=" * 60)
        logger.info("Experiment: %s (%d features, seq_len=%d)", experiment_name, len(features), seq_len)
        logger.info("=" * 60)

        X_raw = df[features].values

        # Chronological train/test split BEFORE scaling (no data leakage)
        split_idx = int(len(X_raw) * (1 - test_ratio))

        scaler = StandardScaler()
        X_raw[:split_idx] = scaler.fit_transform(X_raw[:split_idx])
        X_raw[split_idx:] = scaler.transform(X_raw[split_idx:])

        # Create sequences
        X_seq, y_seq = create_sequences(X_raw, y_all, seq_len)

        # Adjust split index for sequence creation offset
        seq_split = split_idx - seq_len
        X_train, X_test = X_seq[:seq_split], X_seq[seq_split:]
        y_train, y_test = y_seq[:seq_split], y_seq[seq_split:]

        # Use last 15% of training data as validation for early stopping
        val_split = int(len(X_train) * 0.85)
        X_tr, X_val = X_train[:val_split], X_train[val_split:]
        y_tr, y_val = y_train[:val_split], y_train[val_split:]

        logger.info(
            "Splits — train: %d, val: %d, test: %d",
            len(X_tr), len(X_val), len(X_test),
        )

        # Train
        model, history = train_model(X_tr, y_tr, X_val, y_val, config, input_size=len(features))

        # Evaluate on test set
        test_metrics = evaluate_model(model, X_test, y_test)
        logger.info(
            "Test — F1: %.4f | Accuracy: %.4f | AUC: %.4f",
            test_metrics["f1"], test_metrics["accuracy"], test_metrics["auc_roc"],
        )

        # Save model
        safe = ticker.lower().replace("-", "_")
        prefix = f"{safe}_lstm_{experiment_name}"
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": config,
            "features": features,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
        }, SAVED_DIR / f"{prefix}_model.pt")

        results[experiment_name] = {
            "n_features": len(features),
            "sequence_length": seq_len,
            "test_metrics": test_metrics,
            "best_val_loss": min(h["val_loss"] for h in history),
            "epochs_trained": len(history),
            "train_size": len(X_tr),
            "val_size": len(X_val),
            "test_size": len(X_test),
        }

    # Save results
    results_path = SAVED_DIR / f"{safe}_lstm_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Saved results → %s", results_path)

    # Uplift summary
    if "all_features" in results and "no_sentiment" in results:
        print("\n" + "=" * 60)
        print("LSTM SENTIMENT UPLIFT ANALYSIS")
        print("=" * 60)
        for metric in ["f1", "accuracy", "auc_roc"]:
            all_val = results["all_features"]["test_metrics"][metric]
            no_val = results["no_sentiment"]["test_metrics"][metric]
            diff = all_val - no_val
            print(f"  {metric.upper():>10s}: with={all_val:.4f}  without={no_val:.4f}  Δ={diff:+.4f}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Train LSTM classifier")
    parser.add_argument("--ticker", default="BTC-USD")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--sequence-length", type=int, default=None)
    args = parser.parse_args()
    run(args.ticker, args.epochs, args.sequence_length)


if __name__ == "__main__":
    main()
