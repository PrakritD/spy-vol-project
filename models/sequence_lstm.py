"""LSTM on intraday microstructure sequences -> next-day RV regime.

UNUSED IN SHIPPED PIPELINE. Designed for the intraday microstructure feature
group (OBI, signed flow, microprice) computed from ARCX SPY tbbo, which was
not pulled (would have pushed beyond the free $100 Databento credit). This
file is kept as design documentation; `configs/experiment.yaml` does not
reference it and it is not loaded by `models/factory.py`. The class does NOT
conform to the standard Model protocol (takes a `dict[date -> ndarray]`
input rather than a DataFrame).

Sequence input: 1-minute features for one trading session, shape (T, F).
Target: same binary y_next as the daily-feature models.

If/when intraday data is added in a future iteration, the integration
points are: features/microstructure.py (already implemented, reads tbbo
DBN), and a small adapter to translate to the standard Model protocol.

Inputs to fit():
    X: dict mapping date -> np.ndarray of shape (T_d, F)
    y: pd.Series indexed by date with 0/1 labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class LSTMConfig:
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.2
    lr: float = 1e-3
    epochs: int = 30
    batch_size: int = 32
    device: str = "cpu"


def _lazy_torch():
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset
        return torch, nn, DataLoader, Dataset
    except ImportError as e:
        raise ImportError("torch is required for the LSTM model: pip install torch") from e


@dataclass
class SequenceLSTM:
    name: str = "lstm_intraday"
    cfg: LSTMConfig = field(default_factory=LSTMConfig)
    _model: object | None = None
    _input_dim: int | None = None

    def _build_model(self, input_dim: int):
        torch, nn, _, _ = _lazy_torch()

        class Net(nn.Module):
            def __init__(self, in_dim, hidden, layers, dropout):
                super().__init__()
                self.lstm = nn.LSTM(in_dim, hidden, num_layers=layers,
                                    batch_first=True, dropout=dropout if layers > 1 else 0.0)
                self.head = nn.Sequential(
                    nn.Linear(hidden, hidden // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden // 2, 1),
                )

            def forward(self, x, lengths):
                packed = nn.utils.rnn.pack_padded_sequence(
                    x, lengths.cpu(), batch_first=True, enforce_sorted=False)
                _, (h, _) = self.lstm(packed)
                return self.head(h[-1]).squeeze(-1)

        return Net(input_dim, self.cfg.hidden_size, self.cfg.num_layers, self.cfg.dropout)

    def fit(self, X: dict[pd.Timestamp, np.ndarray], y: pd.Series) -> "SequenceLSTM":
        torch, nn, DataLoader, Dataset = _lazy_torch()
        dates = [d for d in y.index if d in X and X[d].shape[0] > 0]
        if not dates:
            raise ValueError("no usable sequences in training data")
        input_dim = X[dates[0]].shape[1]
        self._input_dim = input_dim
        model = self._build_model(input_dim).to(self.cfg.device)
        opt = torch.optim.Adam(model.parameters(), lr=self.cfg.lr)
        loss_fn = nn.BCEWithLogitsLoss()

        seqs = [torch.tensor(X[d], dtype=torch.float32) for d in dates]
        lengths = torch.tensor([s.shape[0] for s in seqs])
        labels = torch.tensor([int(y.loc[d]) for d in dates], dtype=torch.float32)
        padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True)

        idx = np.arange(len(dates))
        for _ in range(self.cfg.epochs):
            np.random.shuffle(idx)
            for start in range(0, len(idx), self.cfg.batch_size):
                batch = idx[start:start + self.cfg.batch_size]
                xb = padded[batch].to(self.cfg.device)
                lb = lengths[batch]
                yb = labels[batch].to(self.cfg.device)
                opt.zero_grad()
                logits = model(xb, lb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()
        self._model = model
        self._padded_dim = padded.shape[1]
        return self

    def predict_proba(self, X: dict[pd.Timestamp, np.ndarray]) -> pd.Series:
        torch, nn, _, _ = _lazy_torch()
        if self._model is None:
            raise RuntimeError("model not fit")
        self._model.eval()
        dates = sorted([d for d in X if X[d].shape[0] > 0])
        seqs = [torch.tensor(X[d], dtype=torch.float32) for d in dates]
        lengths = torch.tensor([s.shape[0] for s in seqs])
        padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True).to(self.cfg.device)
        with torch.no_grad():
            logits = self._model(padded, lengths)
            probs = torch.sigmoid(logits).cpu().numpy()
        return pd.Series(probs, index=dates, name="p_hat")
