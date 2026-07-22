"""Vectorized Black-Scholes IV inversion: a faster, numerically-validated drop-in for bulk
panel builds.

`features.gex.implied_vol` is a per-row Brent-bracketed scalar solve; timed on this project's
full OPRA panel (~670K contract-days) it does not scale linearly with sample size (measured
1.0-1.1 ms/row on several 3-5K random samples, but 20.7 ms/row on a 50K sample -- consistent
with per-call Python/GC overhead compounding across hundreds of thousands of scipy calls
rather than the root-finding itself being slow), putting the full panel at hours rather than
minutes. This module solves the SAME equation with vectorized Newton-Raphson (all contracts
updated together each iteration, ~40 iterations total instead of ~40 iterations PER contract),
validated element-wise against `features.gex.implied_vol` before being trusted for anything
(see tests/test_fast_iv.py).

`features/gex.py`'s scalar implied_vol/compute_contract_greeks are left completely untouched
-- they back the already-published FINDINGS.md gamma numbers, and nothing here calls or
imports them. This module is used only where the FULL panel needs greeks computed once for
bulk feature-building (features/assemble.py), not by the existing gamma pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def _bs_price_vec(S, K, Tsafe, r, q, sigma, is_call):
    sqrtT = np.sqrt(Tsafe)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * Tsafe) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return np.where(
        is_call,
        S * np.exp(-q * Tsafe) * norm.cdf(d1) - K * np.exp(-r * Tsafe) * norm.cdf(d2),
        K * np.exp(-r * Tsafe) * norm.cdf(-d2) - S * np.exp(-q * Tsafe) * norm.cdf(-d1),
    )


def implied_vol_vectorized(price: np.ndarray, S: np.ndarray, K: np.ndarray, T: np.ndarray,
                           r: np.ndarray, q: np.ndarray, is_call: np.ndarray,
                           lo: float = 1e-4, hi: float = 5.0,
                           tol: float = 1e-7, max_iter: int = 60) -> np.ndarray:
    """Vectorized bisection BS IV, all elements updated together per iteration. NaN where
    price is below intrinsic, non-finite, non-positive, T<=0, or unbracketed by [lo, hi] --
    the SAME guard rails (including the bracket check) features.gex.implied_vol applies
    per-row, so a contract this returns a value for is one the scalar Brent solver could
    bracket too. Bisection over Newton: guaranteed monotone convergence given a valid
    bracket, no oscillation risk near the tiny-vega (short-dated, deep ITM/OTM) contracts
    where Newton's step can overshoot; max_iter=60 gives ~60 halvings of the initial
    [1e-4, 5.0] bracket, far more than the ~26 needed for 1e-7 precision on sigma itself."""
    price = np.asarray(price, float)
    S = np.asarray(S, float)
    K = np.asarray(K, float)
    T = np.asarray(T, float)
    r = np.asarray(r, float)
    q = np.asarray(q, float)
    is_call = np.asarray(is_call, bool)

    intrinsic = np.where(is_call, np.maximum(S - K, 0.0), np.maximum(K - S, 0.0))
    valid = (T > 0) & np.isfinite(price) & (price > 0) & (price >= intrinsic - 1e-6)

    Tsafe = np.where(T > 0, T, 1.0)   # placeholder so sqrt/log never warn; masked out by `valid`
    lo_arr = np.full_like(S, lo)
    hi_arr = np.full_like(S, hi)
    f_lo = _bs_price_vec(S, K, Tsafe, r, q, lo_arr, is_call) - price
    f_hi = _bs_price_vec(S, K, Tsafe, r, q, hi_arr, is_call) - price
    valid &= (f_lo * f_hi <= 0)   # reject any contract [lo, hi] does not bracket

    for _ in range(max_iter):
        mid = 0.5 * (lo_arr + hi_arr)
        f_mid = _bs_price_vec(S, K, Tsafe, r, q, mid, is_call) - price
        go_right = (f_lo * f_mid) > 0                 # root lives in [mid, hi]
        lo_arr = np.where(go_right, mid, lo_arr)
        f_lo = np.where(go_right, f_mid, f_lo)
        hi_arr = np.where(go_right, hi_arr, mid)
        if valid.any() and np.nanmax((hi_arr - lo_arr)[valid]) < tol:
            break

    sigma = np.where(valid, 0.5 * (lo_arr + hi_arr), np.nan)
    return sigma


def compute_contract_greeks_fast(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized equivalent of features.gex.compute_contract_greeks (iv/delta/gamma columns
    added), for bulk builds. Same required columns:
        date, expiry, strike, option_type ('C'|'P'), price, open_interest, spot, r, q
    """
    out = df.copy()
    out["dte_years"] = (pd.to_datetime(out["expiry"]) - pd.to_datetime(out["date"])).dt.days / 365.0
    out["is_call"] = out["option_type"].str.upper().eq("C")

    S = out["spot"].to_numpy()
    K = out["strike"].to_numpy()
    T = out["dte_years"].to_numpy()
    r = out["r"].to_numpy()
    q = out["q"].to_numpy()
    price = out["price"].to_numpy()
    is_call = out["is_call"].to_numpy()

    iv = implied_vol_vectorized(price, S, K, T, r, q, is_call)
    Tsafe = np.where(T > 0, T, 1.0)
    sigma = np.where(np.isnan(iv), 1.0, iv)   # placeholder; masked out below
    sqrtT = np.sqrt(Tsafe)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * Tsafe) / (sigma * sqrtT)
    pdf = norm.pdf(d1)
    delta = np.exp(-q * Tsafe) * np.where(is_call, norm.cdf(d1), norm.cdf(d1) - 1.0)
    gamma = np.exp(-q * Tsafe) * pdf / (S * sigma * sqrtT)
    nanmask = np.isnan(iv)
    delta = np.where(nanmask, np.nan, delta)
    gamma = np.where(nanmask, np.nan, gamma)

    out["iv"] = iv
    out["delta"] = delta
    out["gamma"] = gamma
    return out
