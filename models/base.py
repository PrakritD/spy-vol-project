"""Common model protocol used by the walk-forward harness."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd


class Model(Protocol):
    name: str

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Model": ...

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return P(y=1) for each row in X."""
        ...
