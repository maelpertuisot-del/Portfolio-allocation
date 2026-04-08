"""
lstm_model.py
=============
LSTM-based return prediction model.

Architecture
------------
Input  → LSTM layers → Dropout → Dense → Output (scalar return)

The LSTM captures temporal dependencies in sequences of features,
which standard ML models (RF, GBM) cannot exploit directly.

Key concepts from course (Cours 6 — Deep Learning):
- RNN/LSTM: hidden state retains memory across timesteps
- Vanishing gradient: mitigated by LSTM gates (forget, input, output)
- Dropout: regularisation to prevent overfitting
- Walk-forward training: temporal ordering always preserved

Usage
-----
    from src.lstm_model import LSTMReturnPredictor
    model = LSTMReturnPredictor(seq_len=21, hidden_size=64)
    model.fit(X_train_seq, y_train)
    preds = model.predict(X_test_seq)

Course: AI in Finance — Nicolas de Roux & Mohamed El Fakir
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    log.warning("PyTorch not found. LSTMReturnPredictor will be unavailable.")


# ---------------------------------------------------------------------------
# PyTorch LSTM module
# ---------------------------------------------------------------------------

if HAS_TORCH:
    class _LSTMNet(nn.Module):
        """
        Multi-layer LSTM with dropout.

        Architecture:
        - num_layers LSTM layers of size hidden_size
        - Dropout between layers (regularisation)
        - Linear output head (regression)
        """

        def __init__(
            self,
            input_size:  int,
            hidden_size: int = 64,
            num_layers:  int = 2,
            dropout:     float = 0.2,
        ):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size  = input_size,
                hidden_size = hidden_size,
                num_layers  = num_layers,
                dropout     = dropout if num_layers > 1 else 0.0,
                batch_first = True,
            )
            self.dropout = nn.Dropout(dropout)
            self.fc      = nn.Linear(hidden_size, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (batch, seq_len, input_size)
            out, _ = self.lstm(x)
            # Take last timestep output
            out = self.dropout(out[:, -1, :])
            return self.fc(out).squeeze(-1)


# ---------------------------------------------------------------------------
# sklearn-compatible wrapper
# ---------------------------------------------------------------------------

class LSTMReturnPredictor:
    """
    sklearn-compatible LSTM for return prediction.

    Parameters
    ----------
    seq_len     : int   — number of past days in each input sequence
    hidden_size : int   — LSTM hidden state dimension
    num_layers  : int   — number of stacked LSTM layers
    dropout     : float — dropout rate (regularisation)
    lr          : float — Adam learning rate
    epochs      : int   — training epochs
    batch_size  : int   — mini-batch size
    patience    : int   — early-stopping patience (epochs without improvement)
    device      : str   — 'cpu' or 'cuda'
    """

    def __init__(
        self,
        seq_len:     int   = 21,
        hidden_size: int   = 64,
        num_layers:  int   = 2,
        dropout:     float = 0.2,
        lr:          float = 1e-3,
        epochs:      int   = 50,
        batch_size:  int   = 128,
        patience:    int   = 10,
        device:      str   = "cpu",
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for LSTMReturnPredictor. "
                              "Install with: pip install torch")
        self.seq_len     = seq_len
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.dropout     = dropout
        self.lr          = lr
        self.epochs      = epochs
        self.batch_size  = batch_size
        self.patience    = patience
        self.device      = torch.device(device)
        self.model_      = None
        self.scaler_mean_= None
        self.scaler_std_ = None

    # ── Internal helpers ──────────────────────────────────────────────────── #

    def _make_sequences(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Convert flat feature matrix to (samples, seq_len, features) tensor.

        For each sample i, the input sequence is X[i-seq_len : i].
        """
        n, f = X.shape
        seqs, targets = [], []

        for i in range(self.seq_len, n):
            seqs.append(X[i - self.seq_len: i])
            if y is not None:
                targets.append(y[i])

        seqs = np.array(seqs, dtype=np.float32)
        if y is not None:
            targets = np.array(targets, dtype=np.float32)
            return seqs, targets
        return seqs, None

    def _normalise(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit:
            self.scaler_mean_ = X.mean(axis=0, keepdims=True)
            self.scaler_std_  = X.std(axis=0, keepdims=True) + 1e-8
        return (X - self.scaler_mean_) / self.scaler_std_

    # ── Public API ────────────────────────────────────────────────────────── #

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose: bool = False,
    ) -> "LSTMReturnPredictor":
        """
        Train the LSTM on (X, y) pairs.

        X : (T × F) feature matrix (chronological order)
        y : (T,)    target returns

        Training protocol
        -----------------
        - Normalise features (fit on train set only)
        - Build sequences of length seq_len
        - Mini-batch Adam with early stopping on train loss
        """
        X_norm = self._normalise(X, fit=True)
        X_seq, y_seq = self._make_sequences(X_norm, y)

        n_train = int(0.85 * len(X_seq))
        Xtr, ytr = X_seq[:n_train],   y_seq[:n_train]
        Xva, yva = X_seq[n_train:],   y_seq[n_train:]

        train_ds = TensorDataset(
            torch.tensor(Xtr), torch.tensor(ytr)
        )
        val_ds = TensorDataset(
            torch.tensor(Xva), torch.tensor(yva)
        )
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=False)

        self.model_ = _LSTMNet(
            input_size  = X.shape[1],
            hidden_size = self.hidden_size,
            num_layers  = self.num_layers,
            dropout     = self.dropout,
        ).to(self.device)

        optimiser = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, patience=5, factor=0.5, verbose=False
        )

        best_val_loss = float("inf")
        patience_cnt  = 0

        for epoch in range(1, self.epochs + 1):
            # ── Training ─────────────────────────────────────────────────── #
            self.model_.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimiser.zero_grad()
                pred = self.model_(xb)
                loss = criterion(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                optimiser.step()
                train_loss += loss.item() * len(xb)
            train_loss /= len(Xtr)

            # ── Validation ───────────────────────────────────────────────── #
            self.model_.eval()
            with torch.no_grad():
                Xva_t = torch.tensor(Xva).to(self.device)
                yva_t = torch.tensor(yva).to(self.device)
                val_loss = criterion(self.model_(Xva_t), yva_t).item()

            scheduler.step(val_loss)

            if verbose and epoch % 10 == 0:
                log.info(f"  Epoch {epoch:3d}/{self.epochs}"
                         f"  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

            # ── Early stopping ───────────────────────────────────────────── #
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                patience_cnt  = 0
                self._best_state = {k: v.clone()
                                    for k, v in self.model_.state_dict().items()}
            else:
                patience_cnt += 1
                if patience_cnt >= self.patience:
                    log.info(f"  Early stopping at epoch {epoch}")
                    break

        # Restore best weights
        if hasattr(self, "_best_state"):
            self.model_.load_state_dict(self._best_state)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict returns for a feature matrix X."""
        if self.model_ is None:
            raise RuntimeError("Call fit() before predict().")

        X_norm = self._normalise(X, fit=False)
        X_seq, _ = self._make_sequences(X_norm)

        self.model_.eval()
        with torch.no_grad():
            Xt   = torch.tensor(X_seq).to(self.device)
            preds = self.model_(Xt).cpu().numpy()

        # Pad the first seq_len entries with NaN
        full = np.full(len(X), np.nan)
        full[self.seq_len:] = preds
        return full

    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).predict(X)


# ---------------------------------------------------------------------------
# Convenience: build LSTM sequences from panel DataFrame
# ---------------------------------------------------------------------------

def build_lstm_sequences(
    prices: pd.DataFrame,
    horizon: int = 5,
    seq_len: int = 21,
) -> Tuple[np.ndarray, np.ndarray, pd.Index]:
    """
    Build LSTM-ready (N_samples, seq_len, n_features) arrays
    from a price DataFrame.

    Returns
    -------
    X_seq : np.ndarray  (samples, seq_len, features)
    y     : np.ndarray  (samples,)   forward log-returns
    index : pd.DatetimeIndex         corresponding dates
    """
    from .features import build_feature_dict, log_returns

    feats  = build_feature_dict(prices)
    fwd    = log_returns(prices, horizon).shift(-horizon)

    # Use cross-sectional mean features (market-level signal)
    # For stock-level LSTM, iterate per ticker
    feat_matrix = pd.DataFrame({k: v.mean(axis=1) for k, v in feats.items()})
    target      = fwd.mean(axis=1)

    feat_matrix = feat_matrix.dropna()
    target      = target.reindex(feat_matrix.index).dropna()
    common      = feat_matrix.index.intersection(target.index)
    feat_matrix = feat_matrix.loc[common]
    target      = target.loc[common]

    X_arr = feat_matrix.values.astype(np.float32)
    y_arr = target.values.astype(np.float32)

    seqs, ys = [], []
    for i in range(seq_len, len(X_arr)):
        seqs.append(X_arr[i - seq_len: i])
        ys.append(y_arr[i])

    return (
        np.array(seqs,  dtype=np.float32),
        np.array(ys,    dtype=np.float32),
        common[seq_len:],
    )
