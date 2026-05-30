"""XGBoost classifier with per-fold isotonic calibration.

Calibration is fit on a held-out slice (last 20 %) of each training fold,
never on test data. This preserves the walk-forward guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


@dataclass
class XGBCalibratedModel:
    name: str = "xgb_calibrated"
    n_estimators: int = 400
    max_depth: int = 4
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    calibration_holdout_frac: float = 0.2
    random_state: int = 13
    _pipe: Pipeline | None = field(default=None, init=False)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBCalibratedModel":
        y = y.astype(int).to_numpy()
        n = len(X)
        n_cal = max(int(n * self.calibration_holdout_frac), 50) if n > 200 else 0

        base = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            eval_metric="logloss",
            tree_method="hist",
        )

        if n_cal == 0:
            # Too little data: skip calibration, return raw XGB.
            self._pipe = Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("xgb", base),
            ])
            self._pipe.fit(X, y)
            return self

        # Time-ordered calibration: train on first (1-frac), calibrate on last frac.
        train_X, train_y = X.iloc[:-n_cal], y[:-n_cal]
        cal_X, cal_y = X.iloc[-n_cal:], y[-n_cal:]

        impute = SimpleImputer(strategy="median").fit(train_X)
        base.fit(impute.transform(train_X), train_y)

        # sklearn 1.6+ replaced cv="prefit" with FrozenEstimator. Wrap the
        # fitted base so the calibrator treats it as already-trained and only
        # fits the isotonic mapping on the time-ordered held-out cal slice.
        calibrator = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic")
        calibrator.fit(impute.transform(cal_X), cal_y)

        self._pipe = Pipeline([("impute", impute)])
        # We can't drop a fitted calibrator into a sklearn pipeline cleanly with cv=prefit,
        # so we stash it directly.
        self._calibrator = calibrator
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self._pipe is None:
            raise RuntimeError("model not fit")
        impute = self._pipe.named_steps["impute"]
        Xi = impute.transform(X)
        if hasattr(self, "_calibrator"):
            return self._calibrator.predict_proba(Xi)[:, 1]
        # uncalibrated fallback
        xgb = self._pipe.named_steps["xgb"]
        return xgb.predict_proba(Xi)[:, 1]
