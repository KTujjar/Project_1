# Real-Time Anomaly Detection Platform

A production-grade streaming anomaly detection service that compares a classical statistical detector (EWMA + 3-sigma) against a PyTorch LSTM Autoencoder, deployed as a containerized API on Kubernetes.

## Architecture

```
Kafka Stream → Feature Engineering (Pandas) → Dual Detectors → FastAPI → K8s
                                                  ├── EWMA (SciPy)
                                                  └── LSTM Autoencoder (PyTorch)
```

## Key Finding

> The EWMA statistical baseline matches or exceeds the LSTM on univariate signals. The LSTM wins on multivariate correlated anomalies where temporal cross-signal patterns matter. Use both: EWMA as a fast first-pass alert, LSTM for high-value signals where false negatives are costly.

See `notebooks/03_comparison.ipynb` for the full analysis.

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate synthetic data
```bash
python data/generate_data.py
```

### 3. Train the LSTM model
```bash
python src/training/train.py
```

### 4. Run the API locally
```bash
uvicorn src.serving.api:app --reload --port 8080
```

### 5. Test the API
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"windows": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]], "model": "both"}'
```

### 6. Run with Docker (includes Kafka)
```bash
docker-compose up --build
```

### 7. Deploy to Kubernetes
```bash
minikube start
eval $(minikube docker-env)
docker build -t anomaly-api:latest .
kubectl apply -f k8s/
kubectl port-forward svc/anomaly-api 8080:8080
```

## Project Structure

```
├── data/
│   ├── generate_data.py      # Synthetic dataset generator
│   └── sample/               # Generated CSVs (gitignored)
├── notebooks/
│   ├── 01_eda_and_baseline.ipynb
│   ├── 02_lstm_autoencoder.ipynb
│   └── 03_comparison.ipynb   # Head-to-head results
├── src/
│   ├── ingestion/
│   │   ├── features.py       # Pandas feature engineering
│   │   └── kafka_consumer.py # Kafka consumer loop
│   ├── models/
│   │   ├── statistical.py    # EWMA + 3-sigma detector
│   │   └── lstm_autoencoder.py  # PyTorch LSTM Autoencoder
│   ├── serving/
│   │   └── api.py            # FastAPI inference service
│   └── training/
│       └── train.py          # End-to-end training script
├── k8s/                      # Kubernetes manifests
├── tests/                    # pytest suite
├── Dockerfile
└── docker-compose.yml
```

## Running Tests

```bash
pytest tests/ -v
```

## Tech Stack

- **PyTorch** — LSTM Autoencoder for temporal anomaly detection
- **SciPy / Pandas** — Statistical baseline and feature engineering
- **FastAPI** — Async inference API
- **Kafka** — Streaming ingestion
- **Docker** — Containerization
- **Kubernetes** — Deployment with HPA auto-scaling
- **Jupyter** — Experimentation and comparison reports
