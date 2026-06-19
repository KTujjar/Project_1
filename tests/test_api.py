import warnings
import numpy as np
import pytest

warnings.filterwarnings("ignore", category=DeprecationWarning)

from fastapi.testclient import TestClient
from src.serving.api import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ------------------------------------------------------------------
# Health / readiness
# ------------------------------------------------------------------

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_ready(client):
    r = client.get("/ready")
    # 200 if artifacts exist, 503 if not — both are valid depending on environment
    assert r.status_code in (200, 503)


# ------------------------------------------------------------------
# /predict schema
# ------------------------------------------------------------------

def test_predict_response_schema(client):
    window = np.linspace(0, 1, 50).tolist()
    r = client.post("/predict", json={"windows": [window], "n_features": 1, "model": "both"})
    # skip payload checks if models aren't loaded
    if r.status_code == 503:
        pytest.skip("Artifacts not loaded in this environment")
    assert r.status_code == 200
    data = r.json()
    assert "ewma" in data
    assert "lstm" in data
    assert "latency_ms" in data
    assert isinstance(data["ewma"]["scores"], list)
    assert isinstance(data["ewma"]["is_anomaly"], list)
    assert isinstance(data["lstm"]["scores"], list)
    assert isinstance(data["lstm"]["is_anomaly"], list)
    assert data["latency_ms"] > 0


def test_predict_ewma_only(client):
    window = [0.5] * 50
    r = client.post("/predict", json={"windows": [window], "n_features": 1, "model": "ewma"})
    if r.status_code == 503:
        pytest.skip("Artifacts not loaded")
    assert r.status_code == 200
    data = r.json()
    assert data["ewma"] is not None
    assert data["lstm"] is None


def test_predict_lstm_only(client):
    window = [0.5] * 50
    r = client.post("/predict", json={"windows": [window], "n_features": 1, "model": "lstm"})
    if r.status_code == 503:
        pytest.skip("Artifacts not loaded")
    assert r.status_code == 200
    data = r.json()
    assert data["lstm"] is not None
    assert data["ewma"] is None


def test_predict_batch(client):
    """Send multiple windows in one request."""
    windows = [np.linspace(0, 1, 50).tolist() for _ in range(4)]
    r = client.post("/predict", json={"windows": windows, "n_features": 1, "model": "lstm"})
    if r.status_code == 503:
        pytest.skip("Artifacts not loaded")
    assert r.status_code == 200
    data = r.json()
    assert len(data["lstm"]["scores"]) == 4


# ------------------------------------------------------------------
# /metrics
# ------------------------------------------------------------------

def test_metrics_format(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "anomaly_requests_total" in r.text
    assert "anomaly_detections_total" in r.text
    assert "anomaly_latency_ms_avg" in r.text
