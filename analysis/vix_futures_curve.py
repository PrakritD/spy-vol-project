"""Constant-maturity VIX futures short: the ETF-free implementation of the VRP-carry rule.

Builds a genuine short position directly on the front two VX futures contracts, held at a
constant 30-calendar-day weighted maturity -- the same roll-weight construction the S&P 500
VIX Short-Term Futures Index (what VXX/VIXY track) uses -- then applies the IDENTICAL
zero-parameter contango rule (short when VIX < VIX3M, `strategy_two_sleeve.contango_flag`)
used everywhere else in this project. Unlike VIXY/VXX/UVXY, this sleeve never touches an ETP
wrapper: the "short" is a real futures short funded by margin, so `borrow_ann=0` here is not
a modeling choice, it is the mechanics (contrast with `cross_vehicle.py`'s ETP-side "why pay
borrow" comparison).

Construction (causal by design, weights and contract identity always taken from t-1, matching
this repo's shift(1) convention):
  - On every trade date, rank live contracts by days-to-expiry; front/second = ranks 1/2.
  - w1 = clip((dte2 - 30) / (dte2 - dte1), 0, 1); w2 = 1 - w1 (dte computed from t-1, so the
    weight applied to day t's return is always known before day t opens).
  - index_return_t = w1_{t-1} * r1_t + w2_{t-1} * r2_t, where r1_t/r2_t are the SAME two
    contracts' (yesterday's front/second) settle returns from t-1 to t. On a roll's final day
    the outgoing contract stops trading (no r_t): its already-tiny t-1 weight is treated as
    contributing 0 rather than looked up, since the contract has already converged to spot by
    its last settle.

Data: `ingest/vix_futures_pull.py`'s free CBOE per-contract archive. That module's docstring
has the full coverage audit; the short version is that the only window simultaneously
gap-free, full-year, and correctly scaled against spot VIX is 2008-01 through 2013-12. The
curve is built over every contract on disk (2004-2018) so the boundary is visible in the
printed diagnostics; the reported backtest is restricted to the clean window.

Run: python analysis/vix_futures_curve.py
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

REPO = __file__.rsplit("/analysis/", 1)[0]
sys.path.insert(0, REPO + "/analysis")
import strategy_two_sleeve as S  # noqa: E402

TARGET_MATURITY_DAYS = 30
WINDOW_START = "2008-01-01"   # first date settle-vs-spot-VIX scale check passes (ingest audit)
WINDOW_END = "2013-12-31"     # last full, gap-free contract-year in the free archive
COST = S.CostCfg()


def load_futures_panel() -> pd.DataFrame:
    p = pd.read_parquet(f"{REPO}/data/raw/vix_futures/vix_futures_panel.parquet")
    p = p[p["settle"] > 0][["trade_date", "contract_code", "expiry_date", "settle"]].copy()
    p["trade_date"] = pd.to_datetime(p["trade_date"])
    p["expiry_date"] = pd.to_datetime(p["expiry_date"])
    return p.sort_values(["trade_date", "expiry_date"]).reset_index(drop=True)


def front_second(p: pd.DataFrame) -> pd.DataFrame:
    """For each trade date, the two nearest-to-expiry live contracts and their settle."""
    p = p.copy()
    p["rank"] = p.groupby("trade_date")["expiry_date"].rank(method="first").astype(int)
    fs = p[p["rank"] <= 2]
    wide = fs.pivot(index="trade_date", columns="rank",
                     values=["contract_code", "expiry_date", "settle"])
    wide.columns = [f"{a}{b}" for a, b in wide.columns]
    wide = wide.dropna(subset=["contract_code1", "contract_code2"]).reset_index()
    return wide.sort_values("trade_date").reset_index(drop=True)


def load_term_structure_panel() -> pd.DataFrame:
    """Minimal VIX/VIX3M/rf panel, independent of strategy_two_sleeve.load_panel()'s VIXY/GEX
    inner joins (those bind the panel to 2011-05-> for columns this study never uses; using
    them here would silently drop the 2008-2010 crisis years from a study that has real data
    for them). Reproduces build_signals' exact t_30_90/rf_d formulas so the signal is
    identical, just computed on an unrestricted date range."""
    vix = pd.read_csv(f"{REPO}/data/raw/cboe_vix.csv")
    vix["date"] = pd.to_datetime(vix["DATE"])
    vix = vix[["date", "CLOSE"]].rename(columns={"CLOSE": "vix"})
    vix3m = pd.read_parquet(f"{REPO}/data/raw/deep/VIX3M.parquet")[["date", "close"]]
    vix3m = vix3m.rename(columns={"close": "vix3m"})
    vix3m["date"] = pd.to_datetime(vix3m["date"])
    rf = pd.read_parquet(f"{REPO}/data/raw/fred/dgs3mo_deep.parquet")[["date", "dgs3mo"]]
    rf["date"] = pd.to_datetime(rf["date"])

    d = (vix.merge(vix3m, on="date", how="inner")
            .merge(rf, on="date", how="left")
            .sort_values("date").reset_index(drop=True))
    d["dgs3mo"] = d["dgs3mo"].ffill()
    d["vix3m"] = d["vix3m"].ffill()
    d["t_30_90"] = (d["vix"] / d["vix3m"]).shift(1)      # identical formula to build_signals
    d["rf_d"] = (d["dgs3mo"].fillna(0) / 100.0) / S.ANN
    return d


def build_curve(p: pd.DataFrame) -> pd.DataFrame:
    """Daily constant-maturity index return, causal (weights/contract identity from t-1)."""
    wide = front_second(p)
    wide["dte1"] = (wide["expiry_date1"] - wide["trade_date"]).dt.days
    wide["dte2"] = (wide["expiry_date2"] - wide["trade_date"]).dt.days
    denom = (wide["dte2"] - wide["dte1"]).replace(0, np.nan)
    wide["w1"] = ((wide["dte2"] - TARGET_MATURITY_DAYS) / denom).clip(0, 1)
    wide["w2"] = 1 - wide["w1"]

    settle_lookup = {(row.trade_date, row.contract_code): row.settle
                      for row in p.itertuples(index=False)}

    prev_code1 = wide["contract_code1"].shift(1).to_numpy()
    prev_code2 = wide["contract_code2"].shift(1).to_numpy()
    prev_settle1 = wide["settle1"].shift(1).to_numpy()
    prev_settle2 = wide["settle2"].shift(1).to_numpy()
    prev_w1 = wide["w1"].shift(1).to_numpy()
    prev_w2 = wide["w2"].shift(1).to_numpy()

    index_ret = np.full(len(wide), np.nan)
    for i in range(1, len(wide)):
        t = wide["trade_date"].iloc[i]
        r1 = settle_lookup.get((t, prev_code1[i]))
        r2 = settle_lookup.get((t, prev_code2[i]))
        r1 = (r1 / prev_settle1[i] - 1.0) if r1 is not None and prev_settle1[i] else 0.0
        r2 = (r2 / prev_settle2[i] - 1.0) if r2 is not None and prev_settle2[i] else 0.0
        index_ret[i] = prev_w1[i] * r1 + prev_w2[i] * r2

    wide["index_ret"] = index_ret
    return wide[["trade_date", "contract_code1", "contract_code2", "dte1", "dte2",
                 "w1", "w2", "index_ret"]].rename(columns={"trade_date": "date"})


def main() -> int:
    p = load_futures_panel()
    curve = build_curve(p)
    print(f"curve built: {len(curve)} dates, {curve['date'].min().date()} -> "
          f"{curve['date'].max().date()} (full available range, unscoped)")

    term = load_term_structure_panel()
    d = term.merge(curve[["date", "index_ret", "w1", "dte1", "dte2"]], on="date", how="inner")
    d = d.dropna(subset=["index_ret", "t_30_90", "rf_d"]).reset_index(drop=True)

    mask = (d["date"] >= WINDOW_START) & (d["date"] <= WINDOW_END)
    dw = d[mask].reset_index(drop=True)
    print(f"clean window: {len(dw)} rows, {dw['date'].min().date()} -> {dw['date'].max().date()}")

    # --- sanity: overlap with VIXY (available 2011->) should correlate closely; both track
    # the same front-two-contract roll, just via different wrappers. ---
    vixy = pd.read_parquet(f"{REPO}/data/raw/deep/VIXY.parquet")[["date", "adj_close"]]
    vixy["date"] = pd.to_datetime(vixy["date"])
    vixy["vixy_ret"] = vixy["adj_close"].pct_change()
    dw = dw.merge(vixy[["date", "vixy_ret"]], on="date", how="left")
    overlap = dw[dw["date"] >= "2011-01-03"].copy()
    corr = np.nan
    ok_n = 0
    if len(overlap) > 30:
        vixy_ret = overlap["vixy_ret"].to_numpy()
        idx_ret = overlap["index_ret"].to_numpy()
        ok = ~(np.isnan(vixy_ret) | np.isnan(idx_ret))
        ok_n = int(ok.sum())
        if ok_n > 30:
            corr = float(np.corrcoef(vixy_ret[ok], idx_ret[ok])[0, 1])
    print(f"index_ret vs vixy_ret daily-return corr over 2011-{WINDOW_END[:4]} overlap: {corr:.4f} "
          f"(n={ok_n}; expect > 0.9, same underlying roll)")

    flag = S.contango_flag(dw)
    pos = -1.0 * S.NOTIONAL * flag
    r = S.sleeve_excess(pos, dw["index_ret"].to_numpy(), dw["rf_d"].to_numpy(),
                        COST.vixy_bps, 0.0)
    m = S.metrics(r, dw["date"].to_numpy(), "VX constant-maturity short (2008-2013, no borrow)")
    print(f"\n{m['name']}: n={m['n']} Sharpe={m['sharpe']:+.2f} Calmar={m.get('calmar', float('nan')):+.2f} "
          f"maxDD={m.get('maxdd', float('nan'))*100:+.1f}% CAGR={m.get('cagr', float('nan'))*100:+.2f}%")

    # Same window, VIXY sleeve, for a direct apples-to-apples comparison (same dates, same
    # notional/costs, only the vehicle differs: real futures short vs ETP short).
    dw_vixy = dw.dropna(subset=["vixy_ret"])
    flag_v = S.contango_flag(dw_vixy)
    pos_v = -1.0 * S.NOTIONAL * flag_v
    r_v = S.sleeve_excess(pos_v, dw_vixy["vixy_ret"].to_numpy(), dw_vixy["rf_d"].to_numpy(),
                          COST.vixy_bps, COST.borrow_ann)
    m_v = S.metrics(r_v, dw_vixy["date"].to_numpy(), "VIXY (2011-2013 subset, no 2008-2010 data)")
    print(f"{m_v['name']}: n={m_v['n']} Sharpe={m_v['sharpe']:+.2f} "
          f"Calmar={m_v.get('calmar', float('nan')):+.2f} "
          f"maxDD={m_v.get('maxdd', float('nan'))*100:+.1f}% CAGR={m_v.get('cagr', float('nan'))*100:+.2f}%")

    # Futures short restricted to the IDENTICAL 2011-2013 dates VIXY has, for a true
    # apples-to-apples comparison isolating the vehicle (real futures vs ETP), not the window.
    r_matched = r[dw["date"].isin(dw_vixy["date"]).to_numpy()]
    m_matched = S.metrics(r_matched, dw_vixy["date"].to_numpy(),
                          "VX constant-maturity short (2011-2013, matched to VIXY dates)")
    print(f"{m_matched['name']}: n={m_matched['n']} Sharpe={m_matched['sharpe']:+.2f} "
          f"Calmar={m_matched.get('calmar', float('nan')):+.2f} "
          f"maxDD={m_matched.get('maxdd', float('nan'))*100:+.1f}%")

    out = {
        "window": [WINDOW_START, WINDOW_END],
        "vixy_vs_futures_corr_2011_overlap": corr,
        "constant_maturity_short": m,
        "vixy_same_window": m_v,
        "constant_maturity_short_matched_to_vixy_dates": m_matched,
        "coverage_note": ("free CBOE per-contract archive; gap-free/full-year/correctly-"
                          "scaled only 2008-01..2013-12 -- see ingest/vix_futures_pull.py"),
    }
    with open(f"{REPO}/analysis/vix_futures_results.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("\nsaved analysis/vix_futures_results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
