"""
Kafka consumer that feeds live time-series data through both detectors.

Reads JSON messages from a Kafka topic, builds sliding windows using Pandas,
and runs both the EWMA and LSTM detectors. Anomaly flags are printed and can
be forwarded to any downstream sink (alerting, database, another topic).

Message format expected on the topic:
    {"timestamp": "2024-01-01T00:00:00", "value": 1.23}          # univariate
    {"timestamp": "...", "signal_0": 1.2, "signal_1": 0.9, ...}  # multivariate

Usage:
    python -m src.ingestion.kafka_consumer \
        --topic sensor-metrics \
        --bootstrap-servers localhost:9092 \
        --dataset univariate
"""

import argparse
import json
import os
import sys
import time
from collections import deque

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.ingestion.features import Normalizer, sliding_windows
from src.models.lstm_autoencoder import AnomalyDetector as LSTMDetector
from src.models.statistical import EWMADetector

ARTIFACT_DIR = "artifacts"
WINDOW_SIZE = 50


def load_artifacts(dataset: str):
    ewma = EWMADetector.load(os.path.join(ARTIFACT_DIR, f"{dataset}_ewma.pkl"))
    lstm = LSTMDetector.load(os.path.join(ARTIFACT_DIR, f"{dataset}_lstm.pt"))
    normalizer = Normalizer.load(os.path.join(ARTIFACT_DIR, f"{dataset}_scaler.pkl"))
    return ewma, lstm, normalizer


def run_consumer(topic: str, bootstrap_servers: str, dataset: str) -> None:
    from kafka import KafkaConsumer

    print(f"Loading artifacts for dataset='{dataset}'...")
    ewma, lstm, normalizer = load_artifacts(dataset)
    feature_cols = ewma.feature_cols
    print(f"  Features: {feature_cols}")
    print(f"  Connecting to Kafka at {bootstrap_servers}, topic='{topic}'")

    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="latest",
        enable_auto_commit=True,
        group_id="anomaly-detector",
    )

    # Rolling buffer — keeps the last WINDOW_SIZE rows for windowed inference
    buffer: deque = deque(maxlen=WINDOW_SIZE)

    print(f"Listening on topic '{topic}'... (Ctrl-C to stop)\n")

    for message in consumer:
        row = message.value
        timestamp = row.get("timestamp", "unknown")

        # Extract feature values in the expected order
        values = {col: row.get(col) for col in feature_cols}
        if any(v is None for v in values.values()):
            print(f"[SKIP] Missing fields in message: {row}")
            continue

        buffer.append(values)

        if len(buffer) < WINDOW_SIZE:
            continue   # not enough data yet to fill a window

        # Build a DataFrame from the rolling buffer
        df = pd.DataFrame(list(buffer))
        df_norm = normalizer.transform(df)

        # EWMA: score the most recent point
        _, ewma_flags = ewma.predict(df_norm)
        ewma_anomaly = bool(ewma_flags[-1])
        ewma_score = ewma.anomaly_score(df_norm)[-1]

        # LSTM: score the whole window
        windows, _ = sliding_windows(df_norm, feature_cols, window_size=WINDOW_SIZE, step=WINDOW_SIZE)
        lstm_scores, lstm_flags = lstm.predict(windows)
        lstm_anomaly = bool(lstm_flags[0])
        lstm_score = float(lstm_scores[0])

        status = "ANOMALY" if (ewma_anomaly or lstm_anomaly) else "normal"
        print(
            f"[{timestamp}]  {status:8s}  "
            f"ewma={ewma_score:.4f}({'!' if ewma_anomaly else ' '})  "
            f"lstm={lstm_score:.6f}({'!' if lstm_anomaly else ' '})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Kafka anomaly detection consumer")
    parser.add_argument("--topic", default="sensor-metrics")
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--dataset", choices=["univariate", "multivariate"], default="univariate")
    args = parser.parse_args()

    run_consumer(args.topic, args.bootstrap_servers, args.dataset)


if __name__ == "__main__":
    main()
