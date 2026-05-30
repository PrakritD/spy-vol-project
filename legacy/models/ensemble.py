"""Bayesian model averaging across the six pre-registered model variants.

Per-day weights:

    w_i(t) ∝ exp(-eta * mean(log_loss_i over last `lookback` days))

renormalised to sum to 1 across the models. The ensemble prediction is then
the weighted mean of the per-model p_hat. For the first `lookback` days,
weights fall back to uniform (no history to score on).

This is post-processing on the long-format predictions emitted by
`backtest/walk_forward.run()`, not a model class that itself implements
fit/predict_proba. Cheaper and decouples cleanly from the walk-forward
harness.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EnsembleConfig:
    eta: float = 1.0          # temperature on log-loss; higher = more concentrated weights
    lookback: int = 30        # days of OOS history used to score each model
    eps: float = 1e-6         # clip on p_hat to keep log-loss finite
    uniform_warmup: bool = True   # use uniform weights until `lookback` days observed


def _logloss(y: np.ndarray, p: np.ndarray, eps: float) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def bayesian_average(
    preds_long: pd.DataFrame,
    cfg: EnsembleConfig | None = None,
    name: str = "ensemble_bma",
) -> pd.DataFrame:
    """Combine per-model predictions into a single ensemble probability stream.

    Args:
        preds_long: long-format frame with columns (date, y_true, p_hat, model_name)
                    as produced by `backtest.walk_forward.run()`.
        cfg:        EnsembleConfig.
        name:       value to use in the output `model_name` column.

    Returns:
        Same long-format schema, with one row per date for the ensemble.
        Also returns the per-day weight matrix as an attribute via attrs.
    """
    cfg = cfg or EnsembleConfig()
    df = preds_long[["date", "y_true", "p_hat", "model_name"]].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values(["date", "model_name"]).reset_index(drop=True)

    p_wide = df.pivot(index="date", columns="model_name", values="p_hat").sort_index()
    y_wide = df.pivot(index="date", columns="model_name", values="y_true").sort_index()
    # y_true is the same across models on a given date; collapse to a series
    y_series = y_wide.iloc[:, 0]

    models = list(p_wide.columns)
    n_days = len(p_wide)
    n_models = len(models)

    # Per-model per-day log-loss
    ll = pd.DataFrame(index=p_wide.index, columns=models, dtype=float)
    for m in models:
        ll[m] = _logloss(y_series.to_numpy().astype(float),
                          p_wide[m].to_numpy().astype(float), cfg.eps)

    # Rolling mean of log-loss over the previous `lookback` days (causal: shift(1) first)
    ll_shifted = ll.shift(1)
    rolling_mean_ll = ll_shifted.rolling(cfg.lookback, min_periods=1).mean()

    # Convert to weights via softmin (exp of negative mean log-loss)
    weights = pd.DataFrame(index=p_wide.index, columns=models, dtype=float)
    for t in range(n_days):
        if cfg.uniform_warmup and t < cfg.lookback:
            weights.iloc[t] = 1.0 / n_models
        else:
            row = rolling_mean_ll.iloc[t].to_numpy()
            if not np.all(np.isfinite(row)):
                weights.iloc[t] = 1.0 / n_models
                continue
            # subtract min for numerical stability before exp
            adj = -cfg.eta * (row - row.min())
            w = np.exp(adj)
            weights.iloc[t] = w / w.sum()

    ensemble_p = (p_wide * weights).sum(axis=1)
    out = pd.DataFrame({
        "date": ensemble_p.index,
        "y_true": y_series.values.astype(int),
        "p_hat": ensemble_p.values,
        "model_name": name,
    }).reset_index(drop=True)
    out.attrs["weights"] = weights
    return out
