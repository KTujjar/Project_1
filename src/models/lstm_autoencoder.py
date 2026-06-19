"""
LSTM Autoencoder for unsupervised anomaly detection.

Architecture:
    Input  (batch, seq_len, n_features)
      → LSTM encoder  (hidden=64, layers=2, dropout=0.2)
      → repeat last hidden state across seq_len
      → LSTM decoder  (hidden=64, layers=2)
      → Linear(64 → n_features)
    Output (batch, seq_len, n_features)

Anomaly score = mean MSE reconstruction error over the window.
Threshold     = 95th percentile of reconstruction error on held-out NORMAL val data.
"""

import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

"""LSTM neural network model that trains based on data to find anomalies"""
class LSTMAutoencoder(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_layer = nn.Linear(hidden_size, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        _, (h_n, c_n) = self.encoder(x)

        # Repeat the final encoder hidden state across seq_len for the decoder input
        seq_len = x.size(1)
        # Take last layer hidden state: (batch, hidden_size)
        decoder_input = h_n[-1].unsqueeze(1).repeat(1, seq_len, 1)

        decoded, _ = self.decoder(decoder_input, (h_n, c_n))
        reconstruction = self.output_layer(decoded)
        return reconstruction


class AnomalyDetector:
    """
    Wraps LSTMAutoencoder with fit / predict / save / load — mirrors the
    EWMADetector API so the two are interchangeable in serving and tests.
    """

    def __init__(
        self,
        n_features: int,
        window_size: int = 50,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        threshold_percentile: float = 95.0,
        device: Optional[str] = None,
    ) -> None:
        self.n_features = n_features
        self.window_size = window_size
        self.threshold_percentile = threshold_percentile
        self.threshold: Optional[float] = None

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = LSTMAutoencoder(n_features, hidden_size, num_layers, dropout).to(self.device)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        train_windows: np.ndarray,
        val_normal_windows: np.ndarray,
        epochs: int = 30,
        batch_size: int = 128,
        lr: float = 1e-3,
        patience: int = 5,
    ) -> dict:
        """
        Train on normal windows only (unsupervised).
        Sets self.threshold from val_normal_windows reconstruction errors.

        Returns a history dict with train_loss and val_loss per epoch.
        """
        train_tensor = torch.tensor(train_windows, dtype=torch.float32)
        val_tensor = torch.tensor(val_normal_windows, dtype=torch.float32)

        train_loader = DataLoader(
            TensorDataset(train_tensor), batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(val_tensor), batch_size=batch_size, shuffle=False
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        history = {"train_loss": [], "val_loss": []}
        best_val = float("inf")
        patience_counter = 0
        best_state = None

        self.model.train()
        for epoch in range(epochs):
            train_loss = self._run_epoch(train_loader, optimizer, criterion, train=True)
            val_loss = self._run_epoch(val_loader, optimizer, criterion, train=False)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1

            print(f"Epoch {epoch+1:03d}/{epochs}  train={train_loss:.6f}  val={val_loss:.6f}")

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if best_state:
            self.model.load_state_dict(best_state)

        # Set threshold from val reconstruction errors on normal data
        errors = self._reconstruction_errors(val_normal_windows)
        self.threshold = float(np.percentile(errors, self.threshold_percentile))
        print(f"Anomaly threshold set at {self.threshold:.6f} ({self.threshold_percentile}th pct)")

        return history

    def _run_epoch(self, loader, optimizer, criterion, train: bool) -> float:
        self.model.train(train)
        total_loss = 0.0
        with torch.set_grad_enabled(train):
            for (batch,) in loader:
                batch = batch.to(self.device)
                reconstruction = self.model(batch)
                loss = criterion(reconstruction, batch)
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                total_loss += loss.item() * len(batch)
        return total_loss / len(loader.dataset)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _reconstruction_errors(self, windows: np.ndarray) -> np.ndarray:
        """Per-window mean squared reconstruction error."""
        self.model.eval()
        tensor = torch.tensor(windows, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            reconstruction = self.model(tensor)
        errors = ((tensor - reconstruction) ** 2).mean(dim=(1, 2)).cpu().numpy()
        return errors

    def predict(self, windows: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns
        -------
        scores : ndarray (N,) — reconstruction errors (higher = more anomalous)
        is_anomaly : ndarray (N,) bool
        """
        if self.threshold is None:
            raise RuntimeError("Call fit() or load() before predict()")
        scores = self._reconstruction_errors(windows)
        is_anomaly = scores > self.threshold
        return scores, is_anomaly

    def anomaly_score(self, windows: np.ndarray) -> np.ndarray:
        return self._reconstruction_errors(windows)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "threshold": self.threshold,
                "n_features": self.n_features,
                "window_size": self.window_size,
                "hidden_size": self.model.hidden_size,
                "num_layers": self.model.num_layers,
                "threshold_percentile": self.threshold_percentile,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "AnomalyDetector":
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        obj = cls(
            n_features=checkpoint["n_features"],
            window_size=checkpoint["window_size"],
            hidden_size=checkpoint["hidden_size"],
            num_layers=checkpoint["num_layers"],
            threshold_percentile=checkpoint["threshold_percentile"],
            device=device,
        )
        obj.model.load_state_dict(checkpoint["model_state"])
        obj.threshold = checkpoint["threshold"]
        return obj
