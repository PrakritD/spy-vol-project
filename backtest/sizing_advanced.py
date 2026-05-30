"""Vol-target, Kelly-with-uncertainty, and Mean-CVaR sizing rules.

All return a sizing function with the SizingSpec interface — fn: ndarray ->
ndarray in [0, 1]. The LFS execution mirrors the magnitude on the short side.

Three rules:

1. Vol-target: position scales inversely to a forecast of next-day VXX
   vol so the strategy's annualised volatility hits a constant target.
   Standard risk-parity practice.

2. Kelly with posterior-uncertainty shrinkage: half-Kelly multiplied by
   (1 - shrinkage * sqrt(p_var)) where p_var comes from the Bayesian
   head's posterior variance or the ensemble's weight entropy.

3. Mean-CVaR: solve daily — maximise expected return s.t. historical CVaR
   95% of the resulting return series stays under a target. Falls back to
   half-Kelly if cvxpy is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VolTargetConfig:
    target_ann_vol: float = 0.10
    vol_floor: float = 0.05          # don't oversize when realised vol is tiny
    max_position: float = 1.0


def vol_target_sizing_fn(predicted_ann_vol: pd.Series,
                          cfg: VolTargetConfig | None = None) -> Callable[[np.ndarray], np.ndarray]:
    """Return a sizing function indexed implicitly by panel order.

    Usage at the SizingSpec layer:
        spec = SizingSpec(name='vol_target', fn=vol_target_sizing_fn(predicted_ann_vol))

    Args:
        predicted_ann_vol: pd.Series (date-indexed) of next-day annualised vol
                            of VXX (e.g. rolling 21d RV, or GARCH(1,1) forecast).
        cfg: VolTargetConfig.

    The returned fn takes p_hat values in order; it looks up the matching vol
    via positional alignment. Caller is responsible for passing p_hat in the
    same date order as predicted_ann_vol.
    """
    cfg = cfg or VolTargetConfig()
    vols = predicted_ann_vol.fillna(cfg.target_ann_vol).clip(lower=cfg.vol_floor).to_numpy()

    def fn(p_hat: np.ndarray) -> np.ndarray:
        p = np.asarray(p_hat, dtype=float)
        # confidence component: linear (2p - 1)
        conf = np.clip(2.0 * p - 1.0, 0.0, 1.0)
        # vol-target scaling: cfg.target_ann_vol / forecast_vol
        n = min(len(p), len(vols))
        scaled = np.zeros_like(p)
        scaled[:n] = conf[:n] * (cfg.target_ann_vol / vols[:n])
        return np.clip(scaled, 0.0, cfg.max_position)
    return fn


def kelly_with_uncertainty_fn(
    b: float,
    p_var: pd.Series | None = None,
    shrinkage: float = 1.0,
    frac: float = 0.5,
) -> Callable[[np.ndarray], np.ndarray]:
    """Half-Kelly with optional shrinkage by posterior-uncertainty.

    Args:
        b: realised win/loss payoff ratio from training fold.
        p_var: optional date-aligned pd.Series of posterior variance on p_hat
               (from Bayesian head or ensemble entropy). If None, no shrinkage.
        shrinkage: how strongly to dampen by uncertainty; 1.0 = full effect.
        frac: Kelly fraction (0.5 = half-Kelly, the standard cut).
    """
    shrink_arr = None
    if p_var is not None:
        shrink_arr = np.clip(1.0 - shrinkage * np.sqrt(p_var.fillna(0).to_numpy()), 0.0, 1.0)

    def fn(p_hat: np.ndarray) -> np.ndarray:
        p = np.asarray(p_hat, dtype=float)
        if not np.isfinite(b) or b <= 0:
            return np.zeros_like(p)
        f_star = p - (1.0 - p) / b
        out = frac * f_star
        if shrink_arr is not None:
            n = min(len(p), len(shrink_arr))
            out = out.copy()
            out[:n] = out[:n] * shrink_arr[:n]
        return np.clip(out, 0.0, 1.0)
    return fn


@dataclass(frozen=True)
class MeanCVaRConfig:
    cvar_target: float = 0.05         # daily CVaR-95 ceiling (5% = 5% expected tail loss/day)
    candidate_sizes: tuple = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)
    alpha: float = 0.05               # CVaR confidence level


def mean_cvar_sizing_fn(
    training_returns: pd.Series,
    cfg: MeanCVaRConfig | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a sizing function that maps p_hat -> Mean-CVaR-optimal size.

    Algorithmically (closed form, no cvxpy needed):
    For each candidate size s in cfg.candidate_sizes, compute the historical
    CVaR-95 of `s * training_returns`. Pick the largest s with CVaR <= target.
    Then scale by confidence (2p - 1).

    The closed-form bypass works because CVaR is positively homogeneous:
    CVaR(s * R) = s * CVaR(R). So we only need to compute CVaR(R) once.
    """
    cfg = cfg or MeanCVaRConfig()
    r = pd.Series(training_returns).dropna().to_numpy()
    if len(r) < 30:
        # Insufficient history: fall back to half-Kelly with b from training.
        wins = r[r > 0]
        losses = r[r < 0]
        if wins.size > 0 and losses.size > 0:
            b = float(wins.mean() / abs(losses.mean()))
        else:
            b = 1.0
        return kelly_with_uncertainty_fn(b)

    # CVaR-95 of a unit-size strategy (positive number = the expected tail loss)
    cutoff = np.quantile(r, cfg.alpha)
    tail = r[r <= cutoff]
    cvar_unit = float(-tail.mean()) if tail.size else 0.0

    if cvar_unit <= 0:
        max_size = 1.0
    else:
        max_size = cfg.cvar_target / cvar_unit
        max_size = min(1.0, max_size)

    def fn(p_hat: np.ndarray) -> np.ndarray:
        p = np.asarray(p_hat, dtype=float)
        conf = np.clip(2.0 * p - 1.0, 0.0, 1.0)
        return conf * max_size
    return fn
