"""
Feature engineering for anomaly detection.

Provides three building blocks used by both the statistical detector and LSTM:
  - rolling_zscore: per-series rolling standardization
  - sliding_windows: converts a DataFrame into overlapping (N, window, features) arrays
  - Normalizer: fit-on-train MinMaxScaler wrapper that can be persisted to disk
"""

import os
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

"""
Instead of comparing a value to the entire dataset's average. This function 
is used to compare the value to only the last 50 values
"""
def rolling_zscore(series: pd.Series, window: int = 50) -> pd.Series:
    """
    Rolling z-score: (x - rolling_mean) / rolling_std.
    First `window` values are NaN — drop or fill before use.
    """
    roll = series.rolling(window=window, min_periods=window)
    mean = roll.mean()
    std = roll.std().replace(0, 1e-8)
    return (series - mean) / std

"""Instead of looking at individual rows of data 
this function takes a magnifying glass of size 50 and slides it acaross the strip
one row at a time cutting out each view as a seperate chunk.
"""
def sliding_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_size: int = 50,
    step: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slice a DataFrame into overlapping windows.

    Returns
    -------
    windows : ndarray of shape (N, window_size, n_features)
    start_indices : ndarray of shape (N,) — row index of the first point in each window
    """
    values = df[feature_cols].to_numpy(dtype=np.float32)
    n = len(values)
    indices = range(0, n - window_size + 1, step)
    windows = np.stack([values[i : i + window_size] for i in indices])
    start_indices = np.array(list(indices))
    return windows, start_indices

"""
Normalizes values into the range 0.0 to 1.0 so the values are
not on wildly different scales.
"""
class Normalizer:
    """MinMaxScaler wrapper that saves/loads its state alongside model artifacts."""

    def __init__(self, feature_range: Tuple[float, float] = (0.0, 1.0)) -> None:
        self._scaler = MinMaxScaler(feature_range=feature_range)
        self.is_fit = False

    def fit(self, df: pd.DataFrame, feature_cols: list[str]) -> "Normalizer":
        self._scaler.fit(df[feature_cols].to_numpy())
        self.feature_cols = feature_cols
        self.is_fit = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out[self.feature_cols] = self._scaler.transform(df[self.feature_cols].to_numpy())
        return out

    def fit_transform(self, df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        self.fit(df, feature_cols)
        return self.transform(df)

    def inverse_transform(self, arr: np.ndarray) -> np.ndarray:
        return self._scaler.inverse_transform(arr)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"scaler": self._scaler, "feature_cols": self.feature_cols}, path)

    @classmethod
    def load(cls, path: str) -> "Normalizer":
        data = joblib.load(path)
        obj = cls()
        obj._scaler = data["scaler"]
        obj.feature_cols = data["feature_cols"]
        obj.is_fit = True
        return obj


"""
Splitting data into three buckets:
    training - what the model learns from
    validation - used during training to check if the model is overfitting
    test - locked away until the very end; the final honest score
"""
def time_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological train/val/test split — never shuffle time-series data.
    Returns (train, val, test) DataFrames.
    """
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]
