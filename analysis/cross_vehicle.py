"""Cross-vehicle generalization for the VRP-carry rule (STRATEGY.md §5).

Runs the IDENTICAL zero-parameter contango rule (short when VIX < VIX3M, flat otherwise,
`strategy_two_sleeve.contango_flag`) on three other vol ETPs, unmodified, at the same fixed
0.20 notional and the same CostCfg cost/borrow defaults as the VIXY headline:

  - VXX   the relaunched VIX short-term futures ETN (2018-01-25 -> only; same product family
          as VIXY, so shorted the same way). Its window overlaps only VIXY's post-2018,
          lower-Sharpe sub-period, so it is also compared against VIXY over the IDENTICAL dates.
  - SVXY  ProShares Short VIX Short-Term Futures ETF: -1x through 2018-02-27, then ProShares'
          public deleverage to -0.5x. Already inverse, so the rule goes LONG SVXY in contango
          (not short), and pays no borrow (borrow_ann=0: this is a long position in a fund the
          book owns outright, not a short sale of a hard-to-borrow name). Reported pre/post the
          deleverage date as well as pooled, since pooling across a leverage change would blend
          two different exposures into one number.
  - UVXY  ProShares Ultra VIX Short-Term Futures ETF, 2x leveraged long-vol; shorted the same
          way as VIXY/VXX.

Notional and costs are held fixed rather than normalized per vehicle, because the question is
"what happens if the identical rule and the identical size is run on a different vehicle,
unmodified" (e.g. UVXY's 2x leverage is expected to show up as worse decay/tail, not be sized
away).

Run: python analysis/cross_vehicle.py
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

REPO = __file__.rsplit("/analysis/", 1)[0]
sys.path.insert(0, REPO + "/analysis")
import strategy_two_sleeve as S  # noqa: E402

# Identical to strategy_two_sleeve.main()'s own `need` list, so the VIXY reference row here
# reproduces the committed headline exactly (same row set -> same Sharpe/Calmar/maxDD).
NEED = ["vixy_ret", "spy_ret", "vixy_vol21", "t_30_90", "t_9_30", "vix_z", "vvix_z",
        "gex_neg", "amihud_z", "rf_d"]
SVXY_DELEVER_DATE = "2018-02-28"   # ProShares -1x -> -0.5x effective date (public record)
COST = S.CostCfg()


def load_vehicle_returns(d: pd.DataFrame, ticker: str) -> pd.DataFrame:
    v = pd.read_parquet(f"{REPO}/data/raw/deep/{ticker}.parquet")[["date", "adj_close"]]
    v["date"] = pd.to_datetime(v["date"])
    col = f"{ticker.lower()}_ret"
    v[col] = v["adj_close"].pct_change()
    return d.merge(v[["date", col]], on="date", how="left")


def prepare_panel() -> pd.DataFrame:
    """Headline signal panel (contango flag, t_30_90, rf_d) plus each vehicle's own return,
    computed on that vehicle's full price history before any row is dropped, so a vehicle's
    later inception (VXX, 2018-01-25) restricts the EVALUATION window only, not the signal."""
    d = S.build_signals(S.load_panel())
    d = d.dropna(subset=NEED).reset_index(drop=True)
    for t in ["VXX", "SVXY", "UVXY"]:
        d = load_vehicle_returns(d, t)
    return d


def leg_returns(d: pd.DataFrame, ret_col: str, sign: float, borrow_ann: float) -> tuple[np.ndarray, np.ndarray]:
    """Signed contango-filtered position on the given vehicle's return column, at the fixed
    headline notional; `sign` is +1 for a long-in-contango vehicle (SVXY) and -1 for a
    short-in-contango vehicle (VXX, UVXY, matching VIXY). Returns (dates, excess_return),
    restricted to dates where the vehicle's return is available."""
    flag = S.contango_flag(d)
    pos = sign * S.NOTIONAL * flag
    r = S.sleeve_excess(pos, d[ret_col].to_numpy(), d["rf_d"].to_numpy(), COST.vixy_bps, borrow_ann)
    valid = ~d[ret_col].isna().to_numpy()
    return d["date"].to_numpy()[valid], r[valid]


def block_metrics(dates: np.ndarray, r: np.ndarray, lo: str | None, hi: str | None, label: str) -> dict:
    dts = pd.to_datetime(dates)
    sel = np.ones(len(dts), dtype=bool)
    if lo is not None:
        sel &= dts >= pd.Timestamp(lo)
    if hi is not None:
        sel &= dts < pd.Timestamp(hi)
    m = S.metrics(r[sel], dates[sel], label)
    m["window"] = [str(dts[sel].min().date()) if sel.any() else None,
                   str(dts[sel].max().date()) if sel.any() else None]
    return m


def main() -> int:
    d = prepare_panel()
    print(f"panel: {len(d)} rows  {d['date'].min().date()} -> {d['date'].max().date()}  "
          f"(notional {S.NOTIONAL}, costs {COST.vixy_bps:.0f}bps)\n")

    results: dict = {}

    # --- VIXY headline, for reference, over its own full window ---
    dates_v, r_v = leg_returns(d, "vixy_ret", -1.0, COST.borrow_ann)
    results["vixy_headline"] = block_metrics(dates_v, r_v, None, None, "VIXY (headline)")

    # --- VXX: short, 2018-> only; also VIXY over the identical dates for a fair comparison ---
    dates_x, r_x = leg_returns(d, "vxx_ret", -1.0, COST.borrow_ann)
    results["vxx"] = block_metrics(dates_x, r_x, None, None, "VXX (short, 2018->)")
    vxx_lo = str(pd.to_datetime(dates_x).min().date())
    results["vixy_same_window_as_vxx"] = block_metrics(
        dates_v, r_v, vxx_lo, None, "VIXY (same window as VXX)")

    # --- SVXY: long, no borrow; pooled + pre/post the public deleverage ---
    dates_s, r_s = leg_returns(d, "svxy_ret", +1.0, 0.0)
    results["svxy_pooled"] = block_metrics(dates_s, r_s, None, None, "SVXY (long, pooled)")
    results["svxy_pre_delever"] = block_metrics(
        dates_s, r_s, None, SVXY_DELEVER_DATE, "SVXY (long, -1x, pre 2018-02-28)")
    results["svxy_post_delever"] = block_metrics(
        dates_s, r_s, SVXY_DELEVER_DATE, None, "SVXY (long, -0.5x, post 2018-02-28)")

    # --- UVXY: short, 2x leverage, same window as VIXY headline ---
    dates_u, r_u = leg_returns(d, "uvxy_ret", -1.0, COST.borrow_ann)
    results["uvxy"] = block_metrics(dates_u, r_u, None, None, "UVXY (short, 2x)")

    print(f"{'vehicle':<32s} {'window':<23s} {'n':>5s} {'Sharpe':>7s} {'Calmar':>7s} {'maxDD%':>7s} {'CAGR%':>7s}")
    for k, m in results.items():
        w = f"{m['window'][0]}..{m['window'][1]}" if m["window"][0] else "n/a"
        print(f"{m['name']:<32s} {w:<23s} {m['n']:>5d} {m['sharpe']:>+7.2f} {m['calmar']:>+7.2f} "
              f"{m['maxdd']*100:>+7.1f} {m['cagr']*100:>+7.2f}")

    out = {k: {kk: vv for kk, vv in m.items()} for k, m in results.items()}
    with open(f"{REPO}/analysis/cross_vehicle_results.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("\nsaved analysis/cross_vehicle_results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
