"""Put-call skew / smile-shape feature pipeline.

Inputs (same options_panel schema gex.py reads):
    instrument_id, date, strike, expiry, option_type ('C'|'P'),
    open_interest, price, spot, r, q

Pipeline per date:
    1. Back out IV/delta via the SAME Brent-bracketed BS inversion gex.py uses
       (features.gex.compute_contract_greeks), filtered to DTE in [7, 60] (the
       repo's existing monthly-expiry-band convention).
    2. Per date, find the single live contract nearest each target delta:
       -0.25 (25-delta put), +0.25 (25-delta call), +0.50 (ATM call, the level
       reference).
    3. skew_25d = IV(25-delta put) - IV(25-delta call): a risk reversal.
       Positive means downside protection priced richer than upside, the usual
       equity smile direction. skew_25d_norm divides by atm_iv (comparable
       across vol regimes, since the raw spread scales with the level).

This is a level per date, not an aggregate like GEX's dealer-exposure sum, so
there is no notion of long/short sign convention to document here.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from features.gex import GexConfig, compute_contract_greeks


@dataclass(frozen=True)
class SkewConfig:
    dte_lo_days: int = 7
    dte_hi_days: int = 60
    put_target_delta: float = -0.25
    call_target_delta: float = 0.25
    atm_target_delta: float = 0.50


def _nearest_by_delta(df: pd.DataFrame, target: float) -> pd.DataFrame:
    """Per date, the single contract whose delta is nearest `target`."""
    d = df.copy()
    d["_dist"] = (d["delta"] - target).abs()
    idx = d.groupby("date")["_dist"].idxmin()
    return d.loc[idx, ["date", "iv"]]


def from_greeks(g: pd.DataFrame, cfg: SkewConfig | None = None) -> pd.DataFrame:
    """Same aggregation as `run`, but takes an ALREADY-computed contract-greeks frame
    (features.gex.compute_contract_greeks output, unfiltered) -- lets a caller that also
    needs GEX on the same options_panel invert IV once and reuse it for both, instead of
    paying the Brent-inversion cost twice (it dominates runtime: ~1ms/row over hundreds of
    thousands of contract-days)."""
    cfg = cfg or SkewConfig()
    dte_days = (g["dte_years"] * 365).round().astype("Int64")
    g = g[g["iv"].notna() & dte_days.between(cfg.dte_lo_days, cfg.dte_hi_days)].copy()

    puts = g[~g["is_call"]]
    calls = g[g["is_call"]]

    put25 = _nearest_by_delta(puts, cfg.put_target_delta).rename(columns={"iv": "put25_iv"})
    call25 = _nearest_by_delta(calls, cfg.call_target_delta).rename(columns={"iv": "call25_iv"})
    atm = _nearest_by_delta(calls, cfg.atm_target_delta).rename(columns={"iv": "atm_iv"})

    out = (put25.merge(call25, on="date", how="outer")
                .merge(atm, on="date", how="outer")
                .sort_values("date").reset_index(drop=True))
    out["skew_25d"] = out["put25_iv"] - out["call25_iv"]
    out["skew_25d_norm"] = out["skew_25d"] / out["atm_iv"]
    return out


def run(df: pd.DataFrame, cfg: SkewConfig | None = None) -> pd.DataFrame:
    """Self-contained entry point (computes its own greeks): standalone use and tests."""
    cfg = cfg or SkewConfig()
    gcfg = GexConfig(dte_lo_days=cfg.dte_lo_days, dte_hi_days=cfg.dte_hi_days)
    g = compute_contract_greeks(df, gcfg)
    return from_greeks(g, cfg)
