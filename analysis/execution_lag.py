"""Execution-lag sensitivity for the VRP-carry strategy.

The headline backtest forms the contango signal from the t-1 CLOSE of VIX/VIX3M and assumes
the VIXY fill happens at that same t-1 close. That fill is optimistic: the VIX family
disseminates until 4:15pm ET while VIXY stops trading at 4:00pm, so the 4:15 print that
defines the signal cannot be traded at the 4:00 close. This script prices the gap with
three strictly-causal fill assumptions for the SAME position series:

  (i)   t-1 close  — the headline; reproduced here as the baseline row.
  (ii)  t open     — the position decided from the t-1 close signal is entered at the day-t
                     open. On days the position changes, the OLD position carries the
                     overnight close(t-1)->open(t) leg and the NEW position earns
                     open(t)->close(t); on no-trade days P&L is close-to-close, identical
                     to the headline. VIXY's raw open is adjusted by that day's
                     adj_close/close ratio so both legs live on the adjusted series.
  (iii) t close    — one full extra day of lag: the whole position series shifts one day.

Costs and borrow are the headline CostCfg defaults throughout. Event-window P&L is shown
for Volmageddon and the COVID crash under (i) and (iii), because the lag binds exactly
while the curve is inverting. A flip-fragility count proxies how often the 4:00pm contango
sign could plausibly differ from the 4:15pm one: flips where |VIX/VIX3M - 1| at the t-1
close is under 0.01.

Run:  python analysis/execution_lag.py
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

REPO = __file__.rsplit("/analysis/", 1)[0]
sys.path.insert(0, REPO + "/analysis")
import strategy_two_sleeve as S  # noqa: E402

NEED = ["vixy_ret", "spy_ret", "vixy_vol21", "t_30_90", "t_9_30", "vix_z", "vvix_z",
        "gex_neg", "amihud_z", "rf_d"]

EVENTS = [("volmageddon", "2018-02-01", "2018-02-15"),
          ("covid_crash", "2020-02-19", "2020-03-31")]

FRAGILE_BAND = 0.01   # |VIX/VIX3M - 1| at t-1 close under this => 4:00pm sign plausibly differs


# ------------------------------------------------------------------ data ----
def prepare_panel() -> pd.DataFrame:
    """Headline panel plus the split-consistent adjusted VIXY open for the open-fill leg."""
    d = S.build_signals(S.load_panel())
    d = d.dropna(subset=NEED).reset_index(drop=True)
    v = pd.read_parquet(f"{REPO}/data/raw/deep/VIXY.parquet")[["date", "open", "close", "adj_close"]]
    v["date"] = pd.to_datetime(v["date"])
    v["vixy_open_adj"] = v["open"] * (v["adj_close"] / v["close"])
    d = d.merge(v[["date", "vixy_open_adj"]], on="date", how="left")
    if d["vixy_open_adj"].isna().any():
        raise ValueError("missing VIXY open on panel dates; open-fill P&L would be wrong")
    return d


# ------------------------------------------------------------- fill P&L ----
def open_fill_excess(pos: np.ndarray, d: pd.DataFrame, cost: S.CostCfg) -> np.ndarray:
    """Excess return when the day-t target position is FILLED AT THE DAY-t OPEN.

    Causal: pos[t] is decided from t-1 close information (unchanged from the headline);
    only the fill moves from the t-1 close to the t open. On no-trade days the holding and
    P&L are identical to the headline (close-to-close on pos[t]). On trade days the day
    splits into two legs: the previous position rides overnight close(t-1)->open(t), the
    new position earns intraday open(t)->close(t). rf and borrow are charged on the
    leg-weighted exposure (each leg treated as half a day)."""
    adj = d["vixy_adj"].to_numpy()
    op = d["vixy_open_adj"].to_numpy()
    rf = d["rf_d"].to_numpy()
    cc = np.nan_to_num(d["vixy_ret"].to_numpy(), nan=0.0)          # close(t-1)->close(t)
    prev_adj = np.concatenate([[np.nan], adj[:-1]])
    on_ret = np.nan_to_num(op / prev_adj - 1.0, nan=0.0)           # close(t-1)->open(t)
    id_ret = np.nan_to_num(adj / op - 1.0, nan=0.0)                # open(t)->close(t)

    pos_prev = np.concatenate([[0.0], pos[:-1]])
    trade = pos != pos_prev
    held_ret = np.where(trade, pos_prev * on_ret + pos * id_ret, pos * cc)
    exposure = np.where(trade, 0.5 * (pos_prev + pos), pos)        # leg-weighted signed notional
    r = held_ret - exposure * rf
    r = r - np.abs(pos - pos_prev) * (cost.vixy_bps / 1e4)
    short_notional = np.where(trade,
                              0.5 * (np.clip(-pos_prev, 0, None) + np.clip(-pos, 0, None)),
                              np.clip(-pos, 0, None))
    r = r - short_notional * (cost.borrow_ann / S.ANN)
    return r


def event_pnl(r: np.ndarray, dates: np.ndarray, lo: str, hi: str) -> float:
    dts = pd.to_datetime(dates)
    sel = np.asarray((dts >= pd.Timestamp(lo)) & (dts <= pd.Timestamp(hi)))
    return float(np.prod(1.0 + np.nan_to_num(r[sel], nan=0.0)) - 1.0)


# ------------------------------------------------------------------ main ----
def main():
    d = prepare_panel()
    dates = d["date"].to_numpy()
    vret = d["vixy_ret"].to_numpy()
    rf = d["rf_d"].to_numpy()
    cost = S.CostCfg()
    years = len(d) / S.ANN

    pos = S.carry_positions(d)                                     # headline contango-filtered carry
    pos_lag = np.concatenate([[0.0], pos[:-1]])                    # one extra day of lag

    fills = {
        "t-1 close (headline)": S.sleeve_excess(pos, vret, rf, cost.vixy_bps, cost.borrow_ann),
        "t open": open_fill_excess(pos, d, cost),
        "t close (+1 day lag)": S.sleeve_excess(pos_lag, vret, rf, cost.vixy_bps, cost.borrow_ann),
    }
    M = {k: S.metrics(r, dates, k) for k, r in fills.items()}
    base = M["t-1 close (headline)"]

    print(f"panel: {len(d)} rows  {d['date'].min().date()} -> {d['date'].max().date()}  "
          f"(costs {cost.vixy_bps:.0f}bps, borrow {cost.borrow_ann*100:.0f}%/yr)\n")
    print("=" * 108)
    print("EXECUTION-LAG SENSITIVITY — contango-filtered short-VIXY carry under three fill assumptions")
    print("=" * 108)
    print(f"  {'fill':<24s} {'Sharpe':>7s} {'Calmar':>7s} {'maxDD%':>7s} {'CAGR%':>7s}   "
          f"{'dSharpe':>8s} {'dCalmar':>8s} {'dCAGR%':>7s}")
    for k in fills:
        m = M[k]
        print(f"  {k:<24s} {m['sharpe']:>+7.2f} {m['calmar']:>+7.2f} {m['maxdd']*100:>+7.1f} "
              f"{m['cagr']*100:>+7.2f}   {m['sharpe']-base['sharpe']:>+8.2f} "
              f"{m['calmar']-base['calmar']:>+8.2f} {(m['cagr']-base['cagr'])*100:>+7.2f}")

    # --- event windows: where the lag actually binds (curve inverting) ---
    print("\n--- EVENT-WINDOW P&L, (i) t-1 close vs (iii) t close +1d lag ---")
    ev = {}
    for name, lo, hi in EVENTS:
        p1 = event_pnl(fills["t-1 close (headline)"], dates, lo, hi)
        p3 = event_pnl(fills["t close (+1 day lag)"], dates, lo, hi)
        pv = event_pnl(vret, dates, lo, hi)
        ev[name] = {"window": [lo, hi], "pnl_t1close": p1, "pnl_tclose_lag1": p3,
                    "lag_cost": p3 - p1, "long_vixy": pv}
        print(f"  {name:<12s} {lo}..{hi}  (i) {p1*100:+6.2f}%  (iii) {p3*100:+6.2f}%  "
              f"lag cost {(p3-p1)*100:+5.2f}pp   [long-VIXY {pv*100:+6.0f}%]")

    # --- flip frequency + fragility (proxy for 4:00 vs 4:15 sign disagreement) ---
    flag = S.contango_flag(d)
    flip = np.zeros(len(d), dtype=bool)
    flip[1:] = flag[1:] != flag[:-1]
    ratio_dev = np.abs(d["t_30_90"].to_numpy() - 1.0)              # |VIX/VIX3M - 1| at t-1 close
    fragile = flip & (ratio_dev < FRAGILE_BAND)
    n_flips, n_frag = int(flip.sum()), int(fragile.sum())
    print("\n--- SIGNAL FLIPS AND 4:00pm/4:15pm FRAGILITY ---")
    print(f"  flips: {n_flips} over {years:.1f}y  (~{n_flips/years:.1f}/yr)")
    print(f"  fragile flips (|VIX/VIX3M - 1| < {FRAGILE_BAND:.2f} at t-1 close): {n_frag} "
          f"({n_frag/n_flips*100:.0f}% of flips, ~{n_frag/years:.1f}/yr) — days the 4:00pm "
          f"contango sign could plausibly differ from the 4:15pm print the backtest uses")

    out = {
        "window": [str(d["date"].min().date()), str(d["date"].max().date())], "n": int(len(d)),
        "cost_cfg": {"vixy_bps": cost.vixy_bps, "borrow_ann": cost.borrow_ann},
        "fills": {k: {kk: M[k][kk] for kk in ("sharpe", "calmar", "maxdd", "cagr", "ann_vol", "n")}
                  for k in fills},
        "deltas_vs_headline": {k: {"sharpe": M[k]["sharpe"] - base["sharpe"],
                                   "calmar": M[k]["calmar"] - base["calmar"],
                                   "maxdd": M[k]["maxdd"] - base["maxdd"],
                                   "cagr": M[k]["cagr"] - base["cagr"]}
                               for k in fills if k != "t-1 close (headline)"},
        "events": ev,
        "flips": {"total": n_flips, "per_year": n_flips / years,
                  "fragile_band": FRAGILE_BAND, "fragile": n_frag,
                  "fragile_share": n_frag / n_flips, "fragile_per_year": n_frag / years},
    }
    with open(f"{REPO}/analysis/execution_lag_results.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("\nsaved analysis/execution_lag_results.json")


if __name__ == "__main__":
    main()
