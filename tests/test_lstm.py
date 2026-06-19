import numpy as np
import pytest
import torch

from src.models.lstm_autoencoder import AnomalyDetector, LSTMAutoencoder


# ------------------------------------------------------------------
# Architecture / forward pass
# ------------------------------------------------------------------

def test_output_shape_univariate():
    model = LSTMAutoencoder(n_features=1)
    x = torch.randn(8, 50, 1)   # batch=8, seq=50, features=1
    out = model(x)
    assert out.shape == x.shape


def test_output_shape_multivariate():
    model = LSTMAutoencoder(n_features=3)
    x = torch.randn(4, 50, 3)
    out = model(x)
    assert out.shape == x.shape


def test_reconstruction_error_positive():
    """Reconstruction errors must be non-negative."""
    detector = AnomalyDetector(n_features=1, window_size=20)
    windows = np.random.randn(10, 20, 1).astype(np.float32)
    errors = detector.anomaly_score(windows)
    assert (errors >= 0).all()


# ------------------------------------------------------------------
# Threshold logic
# ------------------------------------------------------------------

def test_predict_before_fit_raises():
    detector = AnomalyDetector(n_features=1)
    with pytest.raises(RuntimeError):
        detector.predict(np.zeros((1, 50, 1), dtype=np.float32))


def test_anomaly_score_shape():
    detector = AnomalyDetector(n_features=1, window_size=10)
    windows = np.zeros((5, 10, 1), dtype=np.float32)
    scores = detector.anomaly_score(windows)
    assert scores.shape == (5,)


def test_obvious_anomaly_scores_higher():
    """
    Train on zeros; a window of large values should score much higher
    than another window of zeros.
    """
    detector = AnomalyDetector(n_features=1, window_size=20, threshold_percentile=95.0)

    normal_windows = np.zeros((200, 20, 1), dtype=np.float32)
    # quick training — just 2 epochs to establish a threshold
    detector.fit(normal_windows, normal_windows, epochs=2, batch_size=32)

    anomaly_window = np.ones((1, 20, 1), dtype=np.float32) * 10.0
    normal_window = np.zeros((1, 20, 1), dtype=np.float32)

    anomaly_score = detector.anomaly_score(anomaly_window)[0]
    normal_score = detector.anomaly_score(normal_window)[0]
    assert anomaly_score > normal_score


# ------------------------------------------------------------------
# Persistence
# ------------------------------------------------------------------

def test_save_load_roundtrip(tmp_path):
    detector = AnomalyDetector(n_features=1, window_size=10)
    normal = np.zeros((50, 10, 1), dtype=np.float32)
    detector.fit(normal, normal, epochs=1, batch_size=16)

    path = str(tmp_path / "lstm.pt")
    detector.save(path)
    loaded = AnomalyDetector.load(path)

    assert loaded.threshold == detector.threshold
    assert loaded.n_features == detector.n_features

    windows = np.random.randn(3, 10, 1).astype(np.float32)
    scores_orig, _ = detector.predict(windows)
    scores_loaded, _ = loaded.predict(windows)
    np.testing.assert_allclose(scores_orig, scores_loaded, rtol=1e-5)
