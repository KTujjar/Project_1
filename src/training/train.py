"""
End-to-end training script.

Trains both the EWMA statistical detector and the LSTM Autoencoder on the
univariate dataset, saves all artifacts to artifacts/.

Usage:
    python src/training/train.py
    python src/training/train.py --dataset multivariate --epochs 50 --window 50
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

# Allow running as `python src/training/train.py` from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.ingestion.features import Normalizer, sliding_windows, time_split
from src.models.lstm_autoencoder import AnomalyDetector
from src.models.statistical import EWMADetector


ARTIFACT_DIR = "artifacts"


def load_dataset(dataset: str) -> tuple[pd.DataFrame, list[str]]:
    path = f"data/sample/{dataset}.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python data/generate_data.py` first."
        )
    df = pd.read_csv(path, parse_dates=["timestamp"])
    feature_cols = (
        ["value"] if dataset == "univariate" else ["signal_0", "signal_1", "signal_2"]
    )
    return df, feature_cols


def evaluate(labels: np.ndarray, predictions: np.ndarray) -> dict:
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    try:
        roc_auc = roc_auc_score(labels, predictions)
    except ValueError:
        roc_auc = float("nan")
    return {"precision": precision, "recall": recall, "f1": f1, "roc_auc": roc_auc}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["univariate", "multivariate"], default="univariate")
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ewma-span", type=int, default=20)
    parser.add_argument("--ewma-sigma", type=float, default=3.0)
    args = parser.parse_args()

    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Dataset: {args.dataset}  |  Window: {args.window}  |  Epochs: {args.epochs}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Load and split
    # ------------------------------------------------------------------
    df, feature_cols = load_dataset(args.dataset)
    print(f"Loaded {len(df):,} rows, {df['is_anomaly'].sum()} anomalies ({df['is_anomaly'].mean():.2%})")

    train_df, val_df, test_df = time_split(df)
    print(f"Split: train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}\n")

    # ------------------------------------------------------------------
    # Normalize (fit on train only — no leakage)
    # ------------------------------------------------------------------
    normalizer = Normalizer()
    train_norm = normalizer.fit_transform(train_df, feature_cols)
    val_norm = normalizer.transform(val_df)
    test_norm = normalizer.transform(test_df)
    normalizer.save(os.path.join(ARTIFACT_DIR, f"{args.dataset}_scaler.pkl"))

    # ------------------------------------------------------------------
    # Statistical baseline
    # ------------------------------------------------------------------
    print("--- Training EWMA Detector ---")
    t0 = time.time()
    ewma = EWMADetector(span=args.ewma_span, threshold_sigma=args.ewma_sigma)
    ewma.fit(train_norm, feature_cols)
    _, test_flags = ewma.predict(test_norm)
    ewma_time = time.time() - t0

    test_labels = test_df["is_anomaly"].to_numpy()
    ewma_metrics = evaluate(test_labels, test_flags.astype(int))
    ewma.save(os.path.join(ARTIFACT_DIR, f"{args.dataset}_ewma.pkl"))

    print(f"  Train time  : {ewma_time*1000:.1f} ms")
    print(f"  Precision   : {ewma_metrics['precision']:.4f}")
    print(f"  Recall      : {ewma_metrics['recall']:.4f}")
    print(f"  F1          : {ewma_metrics['f1']:.4f}")
    print(f"  ROC-AUC     : {ewma_metrics['roc_auc']:.4f}\n")

    # ------------------------------------------------------------------
    # LSTM Autoencoder
    # ------------------------------------------------------------------
    print("--- Training LSTM Autoencoder ---")

    # Build windows (train on normal only — unsupervised)
    train_normal = train_norm[train_norm["is_anomaly"] == 0]
    val_normal = val_norm[val_norm["is_anomaly"] == 0]

    train_windows, _ = sliding_windows(train_normal, feature_cols, args.window)
    val_windows, _ = sliding_windows(val_normal, feature_cols, args.window)
    test_windows, test_window_starts = sliding_windows(test_norm, feature_cols, args.window)

    print(f"  train windows: {len(train_windows):,}  val windows: {len(val_windows):,}  test windows: {len(test_windows):,}")

    t0 = time.time()
    lstm_detector = AnomalyDetector(
        n_features=len(feature_cols),
        window_size=args.window,
    )
    history = lstm_detector.fit(
        train_windows,
        val_windows,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    lstm_train_time = time.time() - t0

    # Map window-level predictions back to row-level labels
    # A row is anomalous if ANY window covering it is flagged
    _, window_flags = lstm_detector.predict(test_windows)
    row_flags = np.zeros(len(test_df), dtype=int)
    for i, start in enumerate(test_window_starts):
        if window_flags[i]:
            row_flags[start : start + args.window] = 1

    # Align with test labels (some rows at the tail may not be covered by windows)
    aligned_len = min(len(test_labels), len(row_flags))
    lstm_metrics = evaluate(test_labels[:aligned_len], row_flags[:aligned_len])

    lstm_detector.save(os.path.join(ARTIFACT_DIR, f"{args.dataset}_lstm.pt"))

    print(f"\n  Train time  : {lstm_train_time:.1f} s")
    print(f"  Precision   : {lstm_metrics['precision']:.4f}")
    print(f"  Recall      : {lstm_metrics['recall']:.4f}")
    print(f"  F1          : {lstm_metrics['f1']:.4f}")
    print(f"  ROC-AUC     : {lstm_metrics['roc_auc']:.4f}\n")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'F1':>8} {'Precision':>10} {'Recall':>8} {'ROC-AUC':>9}")
    print(f"{'-'*60}")
    print(
        f"{'EWMA (statistical)':<20} {ewma_metrics['f1']:>8.4f} "
        f"{ewma_metrics['precision']:>10.4f} {ewma_metrics['recall']:>8.4f} "
        f"{ewma_metrics['roc_auc']:>9.4f}"
    )
    print(
        f"{'LSTM Autoencoder':<20} {lstm_metrics['f1']:>8.4f} "
        f"{lstm_metrics['precision']:>10.4f} {lstm_metrics['recall']:>8.4f} "
        f"{lstm_metrics['roc_auc']:>9.4f}"
    )
    print(f"{'='*60}\n")

    # Save metrics for notebooks
    metrics_out = {
        "dataset": args.dataset,
        "window_size": args.window,
        "ewma": ewma_metrics,
        "lstm": lstm_metrics,
        "training_history": history,
    }
    metrics_path = os.path.join(ARTIFACT_DIR, f"{args.dataset}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"Metrics saved to {metrics_path}")
    print(f"Artifacts saved to {ARTIFACT_DIR}/")


if __name__ == "__main__":
    main()
