"""Logistic regression with pre-registered hand-crafted interactions.

The interactions are picked from theory BEFORE seeing any CV result and
locked in code — they are not tunable. This protects against the kind of
post-hoc interaction-hunting that inflates apparent edge.

The four interactions (and their mechanisms):
    vix_zscore × gex_net         high-VIX + short-dealer-gamma = vol amplifier
    term_9d_30d × vix_chg_5d     backwardation under shock = regime break
    vix_zscore²                  vol stress is non-linear (variance not linear in z)
    gex_net × vix_chg_1d         dealer hedging pressure x same-day VIX move
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Pre-registered (a, b) interactions. b=None means quadratic.
_INTERACTIONS: tuple[tuple[str, str | None], ...] = (
    ("vix_zscore_lag1", "gex_net_lag1"),
    ("term_9d_30d_lag1", "vix_chg_5d_lag1"),
    ("vix_zscore_lag1", None),
    ("gex_net_lag1", "vix_chg_1d_lag1"),
)


def _engineer(X: pd.DataFrame) -> pd.DataFrame:
    Xf = X.copy()
    for a, b in _INTERACTIONS:
        if b is None:
            if a in Xf.columns:
                Xf[f"{a}_sq"] = Xf[a].astype(float) ** 2
        else:
            if a in Xf.columns and b in Xf.columns:
                Xf[f"{a}_x_{b}"] = Xf[a].astype(float) * Xf[b].astype(float)
    return Xf


@dataclass
class LogisticInteractionsModel:
    name: str = "logistic_interactions"
    C: float = 1.0
    _pipe: Pipeline | None = field(default=None, init=False)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LogisticInteractionsModel":
        Xf = _engineer(X)
        self._pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(C=self.C, max_iter=2000, solver="lbfgs")),
        ])
        self._pipe.fit(Xf, y.astype(int))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self._pipe is None:
            raise RuntimeError("model not fit")
        Xf = _engineer(X)
        return self._pipe.predict_proba(Xf)[:, 1]
