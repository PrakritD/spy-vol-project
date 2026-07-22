"""Illustrative tail-floor pricing: what a 30-day, 20%-OTM VIX call would have cost.

Groundwork for STRATEGY.md Sec.7's proposed convex left-tail floor (a VIX-call ladder sized as
negative carry), demonstrating `black76.py` on the real constant-maturity futures curve built
in `vix_futures_curve.py`. This is NOT a fitted strategy and NOT a real quote: no VIX-options
market data is ingested in this project (SPY OPRA options back FINDINGS.md's IV-surface work;
VIX options are a separate, unpaid-for feed), so sigma is a realized-vol proxy -- the trailing
60-day annualized vol of the constant-maturity index's own daily returns -- not a market
implied vol. Real VIX-option IV runs persistently above this kind of realized-vol proxy (the
same variance risk premium the whole project is about), so every price here is a lower bound
on what the real hedge would have cost, not an estimate of it.

Run: python analysis/black76_tail_floor_demo.py
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

REPO = __file__.rsplit("/analysis/", 1)[0]
sys.path.insert(0, REPO + "/analysis")
import black76 as B          # noqa: E402
import vix_futures_curve as V  # noqa: E402

OTM_PCT = 1.20          # 20% out-of-the-money strike
T_DAYS = 30
SIGMA_WINDOW = 60       # trailing days for the realized-vol proxy
CRISIS_DATES = {
    "2008 GFC (pre-Lehman)": "2008-09-12",
    "2010 flash crash (day before)": "2010-05-05",
    "2011 debt-ceiling downgrade (day before)": "2011-08-04",
}


def main() -> int:
    p = V.load_futures_panel()
    curve = V.build_curve(p)
    d = curve[(curve["date"] >= V.WINDOW_START) & (curve["date"] <= V.WINDOW_END)].copy()
    d = d.dropna(subset=["cm_level", "index_ret"]).reset_index(drop=True)

    rf = pd.read_parquet(f"{REPO}/data/raw/fred/dgs3mo_deep.parquet")[["date", "dgs3mo"]]
    rf["date"] = pd.to_datetime(rf["date"])
    d = d.merge(rf, on="date", how="left")
    d["dgs3mo"] = d["dgs3mo"].ffill()
    r = (d["dgs3mo"].fillna(0) / 100.0).to_numpy()

    sigma = d["index_ret"].rolling(SIGMA_WINDOW, min_periods=20).std().to_numpy() * np.sqrt(252)
    f = d["cm_level"].to_numpy()
    k = f * OTM_PCT
    t = np.full(len(d), T_DAYS / 365)

    valid = ~np.isnan(sigma)
    price = np.full(len(d), np.nan)
    price[valid] = B.call_price(f[valid], k[valid], t[valid], r[valid], sigma[valid])
    d["call_price"] = price
    d["call_price_pct_of_f"] = d["call_price"] / f

    stats = d["call_price_pct_of_f"].dropna()
    print(f"20%-OTM, 30-day VIX call, realized-vol proxy, {V.WINDOW_START}..{V.WINDOW_END}:")
    print(f"  n={len(stats)}  mean={stats.mean()*100:.2f}%  median={stats.median()*100:.2f}%  "
          f"min={stats.min()*100:.2f}%  max={stats.max()*100:.2f}%  of forward level")

    crisis_rows = {}
    for label, date_str in CRISIS_DATES.items():
        row = d[d["date"] <= date_str].tail(1)
        if row.empty:
            continue
        pct = float(row["call_price_pct_of_f"].iloc[0])
        crisis_rows[label] = {"date": str(row["date"].iloc[0].date()), "call_price_pct_of_f": pct}
        print(f"  {label} ({row['date'].iloc[0].date()}): {pct*100:.2f}% of forward")

    out = {
        "window": [V.WINDOW_START, V.WINDOW_END],
        "otm_pct": OTM_PCT, "t_days": T_DAYS, "sigma_window": SIGMA_WINDOW,
        "call_price_pct_of_f": {
            "n": int(len(stats)), "mean": float(stats.mean()), "median": float(stats.median()),
            "min": float(stats.min()), "max": float(stats.max()),
        },
        "crisis_snapshots": crisis_rows,
        "caveat": ("sigma is a realized-vol proxy, not a market implied vol; no VIX-options "
                  "quotes are ingested in this project, so every price here is a lower bound "
                  "on the real hedge cost, not an estimate of it"),
    }
    with open(f"{REPO}/analysis/black76_tail_floor_demo_results.json", "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print("\nsaved analysis/black76_tail_floor_demo_results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
