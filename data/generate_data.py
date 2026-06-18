"""
Synthetic time-series generator with ground-truth anomaly labels.

Produces two datasets:
  - univariate: single signal, best for testing the statistical baseline
  - multivariate: 3 correlated signals, where the LSTM has an advantage

Usage:
    python data/generate_data.py
    python data/generate_data.py --rows 50000 --anomaly-rate 0.03 --seed 99
"""

import argparse
import os
import numpy as np
import pandas as pd


def _base_signal(n: int, freq: float, noise_std: float, rng: np.random.Generator) -> np.ndarray:
    t = np.linspace(0, 4 * np.pi, n)
    return np.sin(freq * t) + rng.normal(0, noise_std, n)


def _inject_point_anomalies(signal: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sudden spike 4–7x the signal std."""
    out = signal.copy()
    std = signal.std()
    indices = np.where(mask)[0]
    directions = rng.choice([-1, 1], size=len(indices))
    magnitudes = rng.uniform(4, 7, size=len(indices))
    out[indices] += directions * magnitudes * std
    return out


def _inject_contextual_anomalies(signal: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Value that is normal in magnitude but placed at the wrong phase."""
    out = signal.copy()
    indices = np.where(mask)[0]
    # Swap values with a distant point (different phase)
    swap_offsets = rng.integers(len(signal) // 3, len(signal) // 2, size=len(indices))
    for i, offset in zip(indices, swap_offsets):
        partner = (i + offset) % len(signal)
        out[i] = signal[partner]
    return out


def _inject_collective_anomalies(signal: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Short subsequences that drift away from the base pattern."""
    out = signal.copy()
    std = signal.std()
    indices = np.where(mask)[0]
    i = 0
    while i < len(indices):
        start = indices[i]
        # extend the anomaly run up to 8 consecutive points
        run_len = min(rng.integers(3, 9), len(signal) - start)
        drift = rng.choice([-1, 1]) * rng.uniform(2.5, 4.0) * std
        out[start : start + run_len] += drift
        # mark the whole run in mask (already done by caller)
        i += run_len
    return out


def generate_univariate(
    n_rows: int,
    anomaly_rate: float,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    signal = _base_signal(n_rows, freq=1.0, noise_std=0.15, rng=rng)

    n_anomalies = int(n_rows * anomaly_rate)
    anomaly_types = ["point", "contextual", "collective"]
    # divide anomaly budget evenly across types
    per_type = n_anomalies // len(anomaly_types)

    labels = np.zeros(n_rows, dtype=int)
    anomaly_type_col = np.full(n_rows, "normal", dtype=object)

    # --- point ---
    point_idx = rng.choice(n_rows, size=per_type, replace=False)
    point_mask = np.zeros(n_rows, dtype=bool)
    point_mask[point_idx] = True
    signal = _inject_point_anomalies(signal, point_mask, rng)
    labels[point_mask] = 1
    anomaly_type_col[point_mask] = "point"

    # --- contextual ---
    remaining = np.where(labels == 0)[0]
    ctx_idx = rng.choice(remaining, size=per_type, replace=False)
    ctx_mask = np.zeros(n_rows, dtype=bool)
    ctx_mask[ctx_idx] = True
    signal = _inject_contextual_anomalies(signal, ctx_mask, rng)
    labels[ctx_mask] = 1
    anomaly_type_col[ctx_mask] = "contextual"

    # --- collective ---
    remaining = np.where(labels == 0)[0]
    col_idx = rng.choice(remaining, size=per_type, replace=False)
    col_idx.sort()
    col_mask = np.zeros(n_rows, dtype=bool)
    col_mask[col_idx] = True
    signal = _inject_collective_anomalies(signal, col_mask, rng)
    labels[col_mask] = 1
    anomaly_type_col[col_mask] = "collective"

    timestamps = pd.date_range("2024-01-01", periods=n_rows, freq="1min")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "value": signal,
            "is_anomaly": labels,
            "anomaly_type": anomaly_type_col,
        }
    )


def generate_multivariate(
    n_rows: int,
    anomaly_rate: float,
    seed: int,
) -> pd.DataFrame:
    """
    3 correlated signals:
      signal_0: base
      signal_1: signal_0 with phase shift + independent noise
      signal_2: signal_0 * 0.7 + signal_1 * 0.3 + noise

    Anomalies break the correlation structure — the LSTM detects these
    because it learns the cross-signal relationship during training.
    """
    rng = np.random.default_rng(seed)
    s0 = _base_signal(n_rows, freq=1.0, noise_std=0.10, rng=rng)
    phase_shift = int(n_rows * 0.05)
    s1 = np.roll(s0, phase_shift) + rng.normal(0, 0.12, n_rows)
    s2 = 0.7 * s0 + 0.3 * s1 + rng.normal(0, 0.08, n_rows)

    n_anomalies = int(n_rows * anomaly_rate)
    per_type = n_anomalies // 2

    labels = np.zeros(n_rows, dtype=int)
    anomaly_type_col = np.full(n_rows, "normal", dtype=object)

    # Correlation-breaking anomaly: spike only one signal
    point_idx = rng.choice(n_rows, size=per_type, replace=False)
    point_mask = np.zeros(n_rows, dtype=bool)
    point_mask[point_idx] = True
    target_signal = rng.integers(0, 3)
    signals = [s0, s1, s2]
    signals[target_signal] = _inject_point_anomalies(signals[target_signal], point_mask, rng)
    s0, s1, s2 = signals
    labels[point_mask] = 1
    anomaly_type_col[point_mask] = "correlation_break_point"

    # Phase inversion: invert correlation direction on a subsequence
    remaining = np.where(labels == 0)[0]
    col_idx = rng.choice(remaining, size=per_type, replace=False)
    col_idx.sort()
    col_mask = np.zeros(n_rows, dtype=bool)
    col_mask[col_idx] = True
    # invert s1 relationship with s0 during anomaly windows
    s1[col_mask] = -s0[col_mask] + rng.normal(0, 0.1, col_mask.sum())
    labels[col_mask] = 1
    anomaly_type_col[col_mask] = "correlation_break_collective"

    timestamps = pd.date_range("2024-01-01", periods=n_rows, freq="1min")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "signal_0": s0,
            "signal_1": s1,
            "signal_2": s2,
            "is_anomaly": labels,
            "anomaly_type": anomaly_type_col,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic anomaly detection datasets")
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--anomaly-rate", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="data/sample")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Generating univariate dataset ({args.rows:,} rows, {args.anomaly_rate:.1%} anomalies)...")
    uni = generate_univariate(args.rows, args.anomaly_rate, args.seed)
    out_uni = os.path.join(args.out_dir, "univariate.csv")
    uni.to_csv(out_uni, index=False)
    anomaly_count = uni["is_anomaly"].sum()
    print(f"  Saved {out_uni}  ({anomaly_count:,} anomalies = {anomaly_count/args.rows:.2%})")
    print(f"  Breakdown: {uni.groupby('anomaly_type').size().to_dict()}")

    print(f"\nGenerating multivariate dataset ({args.rows:,} rows, {args.anomaly_rate:.1%} anomalies)...")
    multi = generate_multivariate(args.rows, args.anomaly_rate, args.seed)
    out_multi = os.path.join(args.out_dir, "multivariate.csv")
    multi.to_csv(out_multi, index=False)
    anomaly_count = multi["is_anomaly"].sum()
    print(f"  Saved {out_multi}  ({anomaly_count:,} anomalies = {anomaly_count/args.rows:.2%})")
    print(f"  Breakdown: {multi.groupby('anomaly_type').size().to_dict()}")


if __name__ == "__main__":
    main()
