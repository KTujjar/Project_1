"""
EWMA (Exponentially Weighted Moving Average) anomaly detector with 3-sigma control limits.

This is the classical statistical baseline. It adapts to slow drift via exponential
weighting — which is why it out-performs a plain rolling z-score on infrastructure
metrics that drift over time.

Usage:
    detector = EWMADetector(span=20, threshold_sigma=3.0)
    detector.fit(train_series)
    scores, flags = detector.predict(test_series)
"""

import os
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm

"""model that statistically finds anomalies based on previous data"""
class EWMADetector:
    """
    Per-feature EWMA detector. Works on both univariate (single column) and
    multivariate DataFrames — each feature is scored independently and the
    final anomaly flag is set if ANY feature exceeds the threshold.

    Parameters
    ----------
    span : int
        EWMA span (controls decay rate). Larger = slower adaptation.
        Roughly equivalent to a rolling window of `span` points.
    threshold_sigma : float
        Number of standard deviations for the control limit.
        3.0 gives ~0.27% false positive rate for Gaussian data.
    """

    def __init__(self, span: int = 20, threshold_sigma: float = 3.0) -> None:
        self.span = span
        self.threshold_sigma = threshold_sigma
        self._mu: np.ndarray | None = None
        self._sigma: np.ndarray | None = None
        self._upper: np.ndarray | None = None
        self._lower: np.ndarray | None = None
        self.feature_cols: list[str] = []

    def fit(self, df: pd.DataFrame, feature_cols: list[str]) -> "EWMADetector":
        """
        Estimate mu and sigma from training data (expected to be mostly normal).
        Control limits are set using SciPy's norm.ppf for the given sigma level.
        """
        self.feature_cols = feature_cols
        values = df[feature_cols].to_numpy(dtype=np.float64)

        # Use the EWMA of the training set to estimate baseline mu and sigma
        ewm_mean = df[feature_cols].ewm(span=self.span).mean().to_numpy()
        residuals = values - ewm_mean

        self._mu = residuals.mean(axis=0)
        self._sigma = residuals.std(axis=0) + 1e-8

        # One-sided bound at threshold_sigma (two-tailed control limits)
        tail_prob = norm.cdf(-self.threshold_sigma)
        z = norm.ppf(1 - tail_prob)
        self._upper = self._mu + z * self._sigma
        self._lower = self._mu - z * self._sigma
        return self

    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Score each row of `df`.

        Returns
        -------
        scores : ndarray shape (N, n_features) — standardized residuals
        is_anomaly : ndarray shape (N,) bool — True if any feature exceeds limits
        """
        if self._mu is None:
            raise RuntimeError("Call fit() before predict()")

        values = df[self.feature_cols].to_numpy(dtype=np.float64)
        ewm_mean = df[self.feature_cols].ewm(span=self.span).mean().to_numpy()
        residuals = values - ewm_mean

        # Normalize residuals into z-score space
        scores = (residuals - self._mu) / self._sigma

        above = residuals > self._upper
        below = residuals < self._lower
        is_anomaly = (above | below).any(axis=1)
        return scores, is_anomaly

    def anomaly_score(self, df: pd.DataFrame) -> np.ndarray:
        """Return max absolute z-score across features per row (scalar anomaly score)."""
        scores, _ = self.predict(df)
        return np.abs(scores).max(axis=1)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "EWMADetector":
        return joblib.load(path)
