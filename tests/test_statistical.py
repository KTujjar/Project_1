import numpy as np
import pandas as pd
import pytest

from src.models.statistical import EWMADetector


def _make_df(values, col="value"):
    return pd.DataFrame({col: values})


def normal_series(n=500, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, n)


# ------------------------------------------------------------------
# Basic fit/predict
# ------------------------------------------------------------------

def test_fit_predict_returns_correct_shapes():
    df = _make_df(normal_series())
    detector = EWMADetector(span=20, threshold_sigma=3.0)
    detector.fit(df, ["value"])
    scores, flags = detector.predict(df)
    assert scores.shape == (len(df), 1)
    assert flags.shape == (len(df),)


def test_anomalies_score_higher_than_normal():
    """Injected spikes must produce higher anomaly scores than the normal baseline."""
    rng = np.random.default_rng(42)
    values = rng.normal(0, 1, 500)
    anomaly_indices = [200, 300, 400]
    anomaly_values = values.copy()
    anomaly_values[anomaly_indices] = 20.0   # obvious spikes

    train_df = _make_df(values[:350])
    test_normal = _make_df(values[350:])
    test_anomaly_df = _make_df(anomaly_values[350:])

    detector = EWMADetector(span=20, threshold_sigma=3.0)
    detector.fit(train_df, ["value"])

    normal_scores = detector.anomaly_score(test_normal)
    anomaly_scores = detector.anomaly_score(test_anomaly_df)

    assert anomaly_scores.max() > normal_scores.max()


def test_predict_before_fit_raises():
    detector = EWMADetector()
    with pytest.raises(RuntimeError):
        detector.predict(_make_df([1.0, 2.0]))


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------

def test_constant_signal_no_crash():
    """A perfectly constant signal has zero std — must not divide by zero."""
    df = _make_df([1.0] * 200)
    detector = EWMADetector(span=20, threshold_sigma=3.0)
    detector.fit(df, ["value"])
    scores, flags = detector.predict(df)
    assert not np.any(np.isnan(scores))


def test_multivariate_any_feature_flags():
    """Flag is raised if ANY feature exceeds its limit."""
    rng = np.random.default_rng(0)
    s0 = rng.normal(0, 1, 300)
    s1 = rng.normal(0, 1, 300)
    s1[250] = 50.0   # spike only in s1

    df = pd.DataFrame({"a": s0, "b": s1})
    detector = EWMADetector(span=20, threshold_sigma=3.0)
    detector.fit(df.iloc[:200], ["a", "b"])
    _, flags = detector.predict(df.iloc[200:])
    assert flags[50]   # index 250 - 200 = 50


# ------------------------------------------------------------------
# Persistence
# ------------------------------------------------------------------

def test_save_load_roundtrip(tmp_path):
    df = _make_df(normal_series())
    detector = EWMADetector(span=20, threshold_sigma=3.0)
    detector.fit(df, ["value"])
    _, flags_before = detector.predict(df)

    path = str(tmp_path / "ewma.pkl")
    detector.save(path)
    loaded = EWMADetector.load(path)
    _, flags_after = loaded.predict(df)

    np.testing.assert_array_equal(flags_before, flags_after)
