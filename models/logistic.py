"""Logistic regression baseline. Used for the VIX-only benchmark."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


@dataclass
class LogisticModel:
    name: str = "logistic"
    C: float = 1.0
    _pipe: Pipeline | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LogisticModel":
        self._pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(C=self.C, max_iter=2000, solver="lbfgs")),
        ])
        self._pipe.fit(X, y.astype(int))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self._pipe is None:
            raise RuntimeError("model not fit")
        return self._pipe.predict_proba(X)[:, 1]
