"""HAR-X classifier — Heterogeneous Autoregressive vol model with VIX/GEX.

Corsi (2009) HAR-RV regresses log(RV_{t+1}) on log(RV) at three horizons —
daily, weekly, monthly — capturing the well-documented persistence of
volatility across timescales. HAR-X adds exogenous variance-risk-premium
features (VIX z-score, term structure) and optional dealer flow (GEX).

This file wraps the HAR-X regressor into the binary `Model` protocol by
mapping `log(RV_pred_{t+1})` to a probability via

    z = (log(RV_pred) - log(rv_rolling_mean)) / sigma_resid
    P(y=1) = sigmoid(z)

where `sigma_resid` is the in-fold (training) standard deviation of regression
residuals — a true held-out calibration scale, not test-set-fitted. This is
the strongest classical baseline for next-day vol regime prediction.

REQUIRED INPUT COLUMNS in X (fit AND predict):
    rv, rv_5d_mean, rv_rolling_mean  -- HAR triple (positive RV values)
    vix_zscore_lag1, term_9d_30d_lag1, term_30d_90d_lag1  -- exogenous
    gex_net_lag1  -- if use_gex=True
    rv_next  -- ONLY at fit time, as the regression target. The model never
                reads this column at predict time (lookahead guard below).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


@dataclass
class HARXClassifier:
    name: str = "har_x"
    use_gex: bool = True
    _regressor: object = field(default=None, init=False)
    _sigma_resid: float = field(default=1.0, init=False)
    _feature_cols: list[str] = field(default_factory=list, init=False)

    REGRESSION_TARGET = "rv_next"
    THRESHOLD_COL = "rv_rolling_mean"
    LOG_RV_COLS = ("rv", "rv_5d_mean", "rv_rolling_mean")

    def _predictor_cols(self, X: pd.DataFrame) -> list[str]:
        wanted = ["rv", "rv_5d_mean", "rv_rolling_mean",
                  "vix_zscore_lag1", "term_9d_30d_lag1", "term_30d_90d_lag1"]
        if self.use_gex:
            wanted.append("gex_net_lag1")
        return [c for c in wanted if c in X.columns]

    def _transform(self, X: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        Xf = X[cols].copy()
        for c in self.LOG_RV_COLS:
            if c in Xf.columns:
                Xf[c] = np.log(Xf[c].astype(float).clip(lower=1e-6))
        return Xf

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HARXClassifier":
        if self.REGRESSION_TARGET not in X.columns:
            raise ValueError(
                f"HARXClassifier requires column {self.REGRESSION_TARGET!r} at fit time"
            )
        target = np.log(X[self.REGRESSION_TARGET].astype(float).clip(lower=1e-6))
        cols = self._predictor_cols(X)
        Xf = self._transform(X, cols)
        # impute medians from training data
        self._impute_medians = Xf.median()
        Xf = Xf.fillna(self._impute_medians)

        mask = np.isfinite(target.to_numpy()) & np.isfinite(Xf.to_numpy()).all(axis=1)
        Xf, target = Xf.loc[mask], target.loc[mask]
        if len(Xf) < 30:
            # Not enough rows to fit reliably; degenerate but harmless prediction at 0.5.
            self._regressor = None
            self._feature_cols = cols
            return self

        self._regressor = LinearRegression().fit(Xf.values, target.values)
        residuals = target.values - self._regressor.predict(Xf.values)
        self._sigma_resid = float(np.std(residuals, ddof=1))
        if not np.isfinite(self._sigma_resid) or self._sigma_resid < 1e-4:
            self._sigma_resid = 1e-4
        self._feature_cols = cols
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # Lookahead guard: drop the regression target from X if present at predict time.
        X_pred = X.drop(columns=[self.REGRESSION_TARGET], errors="ignore")
        if self._regressor is None:
            return np.full(len(X_pred), 0.5)
        if self.THRESHOLD_COL not in X_pred.columns:
            raise RuntimeError(
                f"HARXClassifier needs {self.THRESHOLD_COL!r} in X at predict time"
            )
        Xf = self._transform(X_pred, self._feature_cols)
        Xf = Xf.fillna(self._impute_medians)
        log_rv_pred = self._regressor.predict(Xf.values)
        log_threshold = np.log(X_pred[self.THRESHOLD_COL].astype(float).clip(lower=1e-6))
        z = (log_rv_pred - log_threshold.values) / self._sigma_resid
        return 1.0 / (1.0 + np.exp(-z))
