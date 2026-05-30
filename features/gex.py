"""Gamma exposure (GEX) feature pipeline.

Inputs (joined upstream):
    instrument_id, date, strike, expiry, option_type ('C'|'P'),
    open_interest, settlement (or mid), spot, r, q, multiplier

Pipeline per (date):
    1. Filter to live SPY contracts (expiry > date).
    2. Back out IV from settlement price via Brent-bracketed BS inversion.
    3. Compute delta + gamma.
    4. Filter |delta| in [0.05, 0.95] and DTE in [7, 60] days.
    5. Aggregate to daily net GEX = sum_calls(gamma * OI * S^2 * mult * 0.01)
                                 - sum_puts(gamma * OI * S^2 * mult * 0.01).

The 0.01 scales to "$ per 1% spot move". Sign convention assumes dealers long calls,
short puts — the standard practitioner simplification. Document this in the report.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm


@dataclass(frozen=True)
class GexConfig:
    delta_lo: float = 0.05
    delta_hi: float = 0.95
    dte_lo_days: int = 7
    dte_hi_days: int = 60
    multiplier: int = 100


def bs_price(S: float, K: float, T: float, r: float, sigma: float, q: float, is_call: bool) -> float:
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0) if is_call else max(K - S, 0.0)
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    if is_call:
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def bs_delta_gamma(S: float, K: float, T: float, r: float, sigma: float, q: float, is_call: bool) -> tuple[float, float]:
    if T <= 0 or sigma <= 0:
        return (0.0, 0.0)
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    pdf = norm.pdf(d1)
    delta = np.exp(-q * T) * (norm.cdf(d1) if is_call else norm.cdf(d1) - 1.0)
    gamma = np.exp(-q * T) * pdf / (S * sigma * sqrtT)
    return (delta, gamma)


def implied_vol(price: float, S: float, K: float, T: float, r: float, q: float, is_call: bool,
                lo: float = 1e-4, hi: float = 5.0) -> float:
    if T <= 0 or not np.isfinite(price) or price <= 0:
        return np.nan
    intrinsic = max(S - K, 0.0) if is_call else max(K - S, 0.0)
    # Time value must be non-negative; allow a small numerical fudge.
    if price < intrinsic - 1e-6:
        return np.nan
    f_lo = bs_price(S, K, T, r, lo, q, is_call) - price
    f_hi = bs_price(S, K, T, r, hi, q, is_call) - price
    if f_lo * f_hi > 0:
        return np.nan
    try:
        return brentq(lambda s: bs_price(S, K, T, r, s, q, is_call) - price,
                      lo, hi, xtol=1e-6, maxiter=100)
    except (ValueError, RuntimeError):
        return np.nan


def _annualise_dte(date: pd.Timestamp, expiry: pd.Timestamp) -> float:
    return max((expiry - date).days, 0) / 365.0


def compute_contract_greeks(df: pd.DataFrame, cfg: GexConfig) -> pd.DataFrame:
    """Expand an input contract-day frame with iv/delta/gamma columns.

    Required input columns:
        date, expiry, strike, option_type ('C'|'P'), price, open_interest, spot, r, q
    """
    required = {"date", "expiry", "strike", "option_type", "price",
                "open_interest", "spot", "r", "q"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    out = df.copy()
    out["dte_years"] = [_annualise_dte(d, e) for d, e in zip(out["date"], out["expiry"])]
    out["is_call"] = out["option_type"].str.upper().eq("C")

    iv = np.empty(len(out))
    delta = np.empty(len(out))
    gamma = np.empty(len(out))
    for i, row in enumerate(out.itertuples(index=False)):
        S, K, T = row.spot, row.strike, row.dte_years
        iv[i] = implied_vol(row.price, S, K, T, row.r, row.q, row.is_call)
        if np.isnan(iv[i]):
            delta[i] = np.nan
            gamma[i] = np.nan
        else:
            d, g = bs_delta_gamma(S, K, T, row.r, iv[i], row.q, row.is_call)
            delta[i] = d
            gamma[i] = g
    out["iv"] = iv
    out["delta"] = delta
    out["gamma"] = gamma
    return out


def filter_contracts(df: pd.DataFrame, cfg: GexConfig) -> pd.DataFrame:
    dte_days = (df["dte_years"] * 365).round().astype("Int64")
    abs_delta = df["delta"].abs()
    mask = (
        df["iv"].notna()
        & dte_days.between(cfg.dte_lo_days, cfg.dte_hi_days, inclusive="both")
        & abs_delta.between(cfg.delta_lo, cfg.delta_hi, inclusive="both")
    )
    return df.loc[mask].copy()


def aggregate_gex(df: pd.DataFrame, cfg: GexConfig) -> pd.DataFrame:
    """Aggregate per-contract gamma to daily net GEX.

    Output columns:
        date, gex_net, gex_calls, gex_puts, gex_regime, n_contracts
    Regime tags: 'pin' for gex_net > +1bn, 'expansion' for < -1bn, else 'neutral'.
    """
    df = df.copy()
    df["contract_gex"] = (
        df["gamma"] * df["open_interest"] * df["spot"] ** 2 * cfg.multiplier * 0.01
    )
    sign = np.where(df["is_call"], 1.0, -1.0)
    df["signed_gex"] = sign * df["contract_gex"]

    calls_mask = df["is_call"]
    g_all = df.groupby("date", sort=True)
    g_calls = df.loc[calls_mask].groupby("date", sort=True)
    g_puts = df.loc[~calls_mask].groupby("date", sort=True)
    out = pd.DataFrame({
        "gex_net": g_all["signed_gex"].sum(),
        "gex_calls": g_calls["contract_gex"].sum(),
        "gex_puts": g_puts["contract_gex"].sum(),
        "n_contracts": g_all.size(),
    })
    # date may be missing from one side on days with only calls or only puts.
    out[["gex_calls", "gex_puts"]] = out[["gex_calls", "gex_puts"]].fillna(0.0)
    out = out.reset_index()

    one_bn = 1e9
    out["gex_regime"] = pd.cut(
        out["gex_net"],
        bins=[-np.inf, -one_bn, one_bn, np.inf],
        labels=["expansion", "neutral", "pin"],
    )
    return out


def run(df: pd.DataFrame, cfg: GexConfig | None = None) -> pd.DataFrame:
    cfg = cfg or GexConfig()
    with_greeks = compute_contract_greeks(df, cfg)
    filtered = filter_contracts(with_greeks, cfg)
    return aggregate_gex(filtered, cfg)
