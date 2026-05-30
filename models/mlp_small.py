"""Small 2-layer MLP — sample-size-honest neural baseline.

Architecture: 2 hidden layers of 16 units, L2 weight decay (alpha=1e-3),
sklearn's `MLPClassifier` with early-stopping on a 15% validation split
inside the training fold (the validation fold is time-ordered tail of the
training set, NOT a random split — preserves walk-forward semantics).

Parameter count: ~500 — within the VC-bound for N_eff ≈ 300. Heavier NNs
overfit at this sample size; the small MLP is here as a sanity check that
NN ≈ classical at this scale, not as an attempt to beat XGBoost.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class SmallMLPModel:
    name: str = "mlp_small"
    hidden_sizes: tuple[int, ...] = (16, 16)
    alpha: float = 1e-3
    max_iter: int = 500
    random_state: int = 13
    _pipe: Pipeline | None = field(default=None, init=False)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SmallMLPModel":
        self._pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=tuple(self.hidden_sizes),
                alpha=self.alpha,
                max_iter=self.max_iter,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=15,
                random_state=self.random_state,
            )),
        ])
        self._pipe.fit(X, y.astype(int))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self._pipe is None:
            raise RuntimeError("model not fit")
        return self._pipe.predict_proba(X)[:, 1]
