"""IV-surface / put-skew figure for FINDINGS.md.

Reads data/processed/options_panel.parquet directly (not the assembled features panel) and
inverts IV/delta ONCE via the vectorized solver (features/fast_iv.py, validated against
features.gex's original scalar Brent inversion to ~1e-7 agreement on real data -- see
tests/test_fast_iv.py), reused for both the snapshot smile and the skew time series. The
scalar solver alone takes hours over the full ~670K-row panel; see fast_iv.py's docstring.

Panel A: the smile on the most recent trading day in the window (IV vs delta, calls and
puts separately) -- the actual shape STRATEGY.md has never shown.
Panel B: ATM IV term structure (by DTE bucket) on that same day.
Panel C: the 25-delta risk-reversal (put IV - call IV) over the full window, so the smile
snapshot in Panel A is seen in the context of its own time series, not presented as typical.

Produces analysis/figures/iv_surface.png.
"""
from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = __file__.rsplit("/analysis/", 1)[0]
sys.path.insert(0, REPO)
from features.fast_iv import compute_contract_greeks_fast  # noqa: E402
from features.skew import SkewConfig, from_greeks  # noqa: E402

OPT = f"{REPO}/data/processed/options_panel.parquet"
FIG = f"{REPO}/analysis/figures"
DTE_LO, DTE_HI = 7, 60


def main():
    import os
    os.makedirs(FIG, exist_ok=True)

    opt = pd.read_parquet(OPT)
    opt["date"] = pd.to_datetime(opt["date"])
    greeks = compute_contract_greeks_fast(opt)
    dte_days = (greeks["dte_years"] * 365).round().astype("Int64")
    g = greeks[greeks["iv"].notna() & dte_days.between(DTE_LO, DTE_HI)].copy()
    g["dte_days"] = dte_days[g.index]

    snap_date = g["date"].max()
    snap = g[g["date"] == snap_date].copy()
    print(f"snapshot date: {snap_date.date()}  ({len(snap)} live contracts, DTE {DTE_LO}-{DTE_HI})")

    skew_ts = from_greeks(greeks, SkewConfig(dte_lo_days=DTE_LO, dte_hi_days=DTE_HI))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))

    # abs(delta) in [0.05, 0.95]: same band GEX already filters to (features/gex.py's
    # GexConfig defaults). Deep ITM/OTM prints beyond this are thin, noisy tape and would
    # dominate the y-axis without showing more real skew than the liquid band already does.
    smile = snap[snap["delta"].abs().between(0.05, 0.95)]
    ax = axes[0]
    calls = smile[smile["is_call"]].sort_values("delta")
    puts = smile[~smile["is_call"]].sort_values("delta")
    ax.plot(calls["delta"], calls["iv"] * 100, "o-", color="#2c6fbb", ms=3, label="calls", alpha=0.8)
    ax.plot(puts["delta"], puts["iv"] * 100, "o-", color="#c0392b", ms=3, label="puts", alpha=0.8)
    ax.set_xlabel("delta")
    ax.set_ylabel("implied vol (%)")
    ax.set_title(f"Smile, {snap_date.date()}\n(DTE {DTE_LO}-{DTE_HI}d, |delta| 0.05-0.95)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1]
    bins = [7, 14, 21, 30, 45, 60]
    labels = ["7-14", "14-21", "21-30", "30-45", "45-60"]
    snap["dte_bucket"] = pd.cut(snap["dte_days"], bins=bins, labels=labels, include_lowest=True)
    atm = snap[(snap["delta"] - 0.5).abs() < 0.10]
    atm_by_bucket = atm.groupby("dte_bucket", observed=True)["iv"].mean() * 100
    ax.bar(range(len(atm_by_bucket)), atm_by_bucket.to_numpy(), color="#7f8c8d")
    ax.set_xticks(range(len(atm_by_bucket))); ax.set_xticklabels(atm_by_bucket.index, fontsize=8)
    ax.set_xlabel("days to expiry"); ax.set_ylabel("ATM implied vol (%)")
    ax.set_title(f"ATM term structure, {snap_date.date()}", fontsize=10)
    ax.grid(alpha=0.25, axis="y")

    ax = axes[2]
    ts = skew_ts.dropna(subset=["skew_25d"])
    ax.plot(ts["date"], ts["skew_25d"] * 100, color="#8e44ad", lw=1.1)
    ax.axhline(0, color="black", lw=0.6)
    ax.axvline(snap_date, color="gray", lw=0.8, ls="--", alpha=0.7)
    ax.set_ylabel("25-delta skew, vol pts\n(put IV - call IV)")
    ax.set_title("25-delta risk reversal over the window", fontsize=10)
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()

    fig.suptitle("SPY options: smile, term structure, and skew (OPRA statistics, 2024-08 -> 2026-04)",
                fontsize=11, y=1.03)
    fig.tight_layout()
    fig.savefig(f"{FIG}/iv_surface.png", dpi=150, bbox_inches="tight")
    print(f"saved {FIG}/iv_surface.png")


if __name__ == "__main__":
    main()
