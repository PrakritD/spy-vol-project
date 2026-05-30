"""Gaussian-process classifier as a Bayesian last-layer.

The architecture is conceptually a Bayesian last-layer over an RBF feature
map: kernel = ConstantKernel * RBF (length-scale learned from data). The
Laplace approximation in sklearn's `GaussianProcessClassifier` gives a
posterior over the latent function, producing **calibrated** probabilities
with implicit uncertainty intervals.

This is the project's "creative architecture" choice. Even if mean AUC
matches XGBoost, the calibrated probability surface is the deliverable —
downstream `backtest/sizing.py` can use it to size positions inversely to
model variance. Strictly small-sample-honest: GP scales as O(N^3) at fit
time, fine at N_eff ≈ 300 (~few seconds per fold).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class BayesianHeadModel:
    name: str = "bayesian_head"
    length_scale: float = 1.0
    n_restarts: int = 1
    random_state: int = 13
    _pipe: Pipeline | None = field(default=None, init=False)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BayesianHeadModel":
        kernel = ConstantKernel(1.0) * RBF(length_scale=self.length_scale)
        self._pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("gp", GaussianProcessClassifier(
                kernel=kernel,
                n_restarts_optimizer=self.n_restarts,
                max_iter_predict=100,
                random_state=self.random_state,
            )),
        ])
        self._pipe.fit(X, y.astype(int))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self._pipe is None:
            raise RuntimeError("model not fit")
        return self._pipe.predict_proba(X)[:, 1]
