"""
FastAPI inference service.

Endpoints:
  POST /predict  — run one or both detectors on a batch of windows
  GET  /health   — liveness probe (always 200 if process is alive)
  GET  /ready    — readiness probe (200 only if models are loaded)
  GET  /metrics  — Prometheus-style plain-text counters

Startup loads both the EWMA detector and LSTM Autoencoder from artifacts/.
The service is stateless — model weights are baked in at image build time.
"""

import os
import time
from contextlib import asynccontextmanager
from typing import Literal, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from src.ingestion.features import Normalizer
from src.models.lstm_autoencoder import AnomalyDetector as LSTMDetector
from src.models.statistical import EWMADetector


ARTIFACT_DIR = os.environ.get("ARTIFACT_DIR", "artifacts")
DATASET = os.environ.get("DATASET", "univariate")

_state: dict = {
    "ewma": None,
    "lstm": None,
    "normalizer": None,
    "ready": False,
    "request_count": 0,
    "anomaly_count": 0,
    "total_latency_ms": 0.0,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts once at startup — keeps cold-start fast."""
    ewma_path = os.path.join(ARTIFACT_DIR, f"{DATASET}_ewma.pkl")
    lstm_path = os.path.join(ARTIFACT_DIR, f"{DATASET}_lstm.pt")
    scaler_path = os.path.join(ARTIFACT_DIR, f"{DATASET}_scaler.pkl")

    missing = [p for p in [ewma_path, lstm_path, scaler_path] if not os.path.exists(p)]
    if missing:
        print(f"WARNING: artifact(s) not found: {missing}. /ready will return 503.")
    else:
        _state["ewma"] = EWMADetector.load(ewma_path)
        _state["lstm"] = LSTMDetector.load(lstm_path)
        _state["normalizer"] = Normalizer.load(scaler_path)
        _state["ready"] = True
        print(f"Models loaded from {ARTIFACT_DIR}/ (dataset={DATASET})")

    yield


app = FastAPI(
    title="Anomaly Detection API",
    description="Dual-model anomaly detection: EWMA statistical baseline + LSTM Autoencoder",
    version="1.0.0",
    lifespan=lifespan,
)


# ------------------------------------------------------------------
# Request / Response schemas
# ------------------------------------------------------------------

class PredictRequest(BaseModel):
    windows: list[list[float]] = Field(
        ...,
        description=(
            "Batch of windows. Each window is a flat list of floats. "
            "Univariate: length = window_size. "
            "Multivariate: length = window_size * n_features (row-major)."
        ),
    )
    n_features: int = Field(1, description="Number of features per timestep")
    model: Literal["ewma", "lstm", "both"] = Field("both")


class ModelResult(BaseModel):
    scores: list[float]
    is_anomaly: list[bool]


class PredictResponse(BaseModel):
    ewma: Optional[ModelResult] = None
    lstm: Optional[ModelResult] = None
    latency_ms: float


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@app.get("/health")
def health():
    """Liveness probe — always 200 if the process is running."""
    return {"status": "ok"}


@app.get("/ready")
def ready():
    """Readiness probe — 503 until model artifacts are loaded."""
    if not _state["ready"]:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return {"status": "ready", "dataset": DATASET}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Run one or both detectors on a batch of windows.

    Example body:
        {"windows": [[0.1, 0.2, ..., 0.5]], "n_features": 1, "model": "both"}
    """
    if not _state["ready"]:
        raise HTTPException(status_code=503, detail="Models not loaded — run training first")

    t_start = time.perf_counter()

    # Reshape flat windows → (batch, seq_len, n_features)
    raw = np.array(req.windows, dtype=np.float32)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)

    batch_size = raw.shape[0]
    window_size = raw.shape[1] // req.n_features
    windows = raw.reshape(batch_size, window_size, req.n_features)

    response = PredictResponse(latency_ms=0.0)

    if req.model in ("ewma", "both"):
        feature_cols = _state["ewma"].feature_cols
        scores_out: list[float] = []
        flags_out: list[bool] = []
        for window in windows:
            df_w = pd.DataFrame(window, columns=feature_cols)
            scores, flags = _state["ewma"].predict(df_w)
            scores_out.append(float(np.abs(scores).max()))
            flags_out.append(bool(flags.any()))
        response.ewma = ModelResult(scores=scores_out, is_anomaly=flags_out)

    if req.model in ("lstm", "both"):
        scores_arr, flags_arr = _state["lstm"].predict(windows)
        response.lstm = ModelResult(
            scores=[float(s) for s in scores_arr],
            is_anomaly=[bool(f) for f in flags_arr],
        )

    latency_ms = (time.perf_counter() - t_start) * 1000
    response.latency_ms = round(latency_ms, 3)

    _state["request_count"] += 1
    any_anomaly = (
        (response.ewma and any(response.ewma.is_anomaly))
        or (response.lstm and any(response.lstm.is_anomaly))
    )
    if any_anomaly:
        _state["anomaly_count"] += 1
    _state["total_latency_ms"] += latency_ms

    return response


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    """Prometheus-compatible plain-text counters."""
    count = _state["request_count"]
    avg_latency = _state["total_latency_ms"] / count if count > 0 else 0.0
    lines = [
        "# HELP anomaly_requests_total Total prediction requests",
        "# TYPE anomaly_requests_total counter",
        f"anomaly_requests_total {count}",
        "# HELP anomaly_detections_total Requests where at least one anomaly was flagged",
        "# TYPE anomaly_detections_total counter",
        f"anomaly_detections_total {_state['anomaly_count']}",
        "# HELP anomaly_latency_ms_avg Average prediction latency in milliseconds",
        "# TYPE anomaly_latency_ms_avg gauge",
        f"anomaly_latency_ms_avg {avg_latency:.3f}",
    ]
    return "\n".join(lines) + "\n"
