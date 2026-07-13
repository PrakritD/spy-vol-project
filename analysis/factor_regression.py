"""Factor-regression defense for the VRP-carry strategy.

The carry sleeve's excess-return stream is +0.61 correlated to SPY, which invites the
attack "this is just ~0.4 SPY beta with no significant alpha." This script quantifies
the defense: (1) CAPM alpha with a Newey-West HAC t-stat, (2) a Fama-French 5+momentum
regression, (3) state-dependent betas showing the contango filter has the book flat in
the worst states, and (4) a co-drawdown table over every SPY drawdown deeper than 10%.

Everything here is a descriptive regression on realized P&L; nothing is fit for trading.

Run: python analysis/factor_regression.py
Writes: analysis/factor_regression_results.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
ANN = 252


# ---------------------------------------------------------------- loading ----
def load_carry() -> pd.DataFrame:
    """Carry excess returns recovered from the shipped equity curve (base 1.0)."""
    eq = pd.read_parquet(REPO_ROOT / "analysis" / "strategy_equity.parquet")
    eq = eq.sort_values("date").reset_index(drop=True)
    prev = eq["carry"].shift(1).fillna(1.0)  # equity starts from 1.0
    out = pd.DataFrame({
        "date": pd.to_datetime(eq["date"]),
        "carry": eq["carry"].to_numpy() / prev.to_numpy() - 1.0,
        "inmkt": eq["inmkt"].to_numpy(),
    })
    return out


def load_spy_excess() -> pd.DataFrame:
    """SPY total returns from adjusted close, net of the daily 3m T-bill rate."""
    spy = pd.read_parquet(REPO_ROOT / "data" / "raw" / "deep" / "SPY.parquet")
    spy = spy.sort_values("date")[["date", "adj_close"]].reset_index(drop=True)
    spy["date"] = pd.to_datetime(spy["date"])
    spy["spy_ret"] = spy["adj_close"].pct_change()
    rf = pd.read_parquet(REPO_ROOT / "data" / "raw" / "fred" / "dgs3mo_deep.parquet")
    rf["date"] = pd.to_datetime(rf["date"])
    spy = spy.merge(rf, on="date", how="left")
    spy["rf_d"] = (spy["dgs3mo"].ffill().fillna(0) / 100.0) / ANN
    spy["spy_excess"] = spy["spy_ret"] - spy["rf_d"]
    return spy[["date", "adj_close", "spy_ret", "rf_d", "spy_excess"]]


_KF_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"


def _kf_daily_csv(name: str, cols: list[str]) -> pd.DataFrame:
    """Download a Ken French daily-CSV zip and parse the yyyymmdd rows positionally
    (the files' prose headers/footers trip pandas_datareader's date parser)."""
    import io
    import urllib.request
    import zipfile

    req = urllib.request.Request(_KF_BASE + name + "_CSV.zip",
                                 headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        buf = io.BytesIO(resp.read())
    with zipfile.ZipFile(buf) as zf:
        text = zf.read(zf.namelist()[0]).decode("latin-1")
    rows = []
    for ln in text.splitlines():
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) > len(cols) and len(parts[0]) == 8 and parts[0].isdigit():
            rows.append(parts[: 1 + len(cols)])
    df = pd.DataFrame(rows, columns=["date"] + cols)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    for c in cols:
        df[c] = pd.to_numeric(df[c])
    return df.set_index("date")


def load_ff6(start: str, end: str) -> pd.DataFrame | None:
    """FF5 (2x3, daily) + momentum; pandas_datareader first, then a direct
    download of the same Ken French files. Returns decimals, not percent."""
    ff5 = mom = None
    try:
        from pandas_datareader import data as pdr
        for _ in range(2):
            try:
                ff5 = pdr.DataReader("F-F_Research_Data_5_Factors_2x3_daily",
                                     "famafrench", start=start, end=end)[0]
                mom = pdr.DataReader("F-F_Momentum_Factor_daily",
                                     "famafrench", start=start, end=end)[0]
                break
            except Exception:
                ff5 = mom = None
    except ImportError:
        pass
    if ff5 is None or mom is None:  # direct-download fallback, same source files
        try:
            ff5 = _kf_daily_csv("F-F_Research_Data_5_Factors_2x3_daily",
                                ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"])
            mom = _kf_daily_csv("F-F_Momentum_Factor_daily", ["Mom"])
        except Exception as e:
            print(f"  Fama-French download failed (reader + direct): {e}")
            return None
    ff = ff5.join(mom, how="inner") / 100.0
    ff.columns = [c.strip() for c in ff.columns]
    ff.index = pd.to_datetime(ff.index)
    ff = ff.loc[(ff.index >= pd.Timestamp(start)) & (ff.index <= pd.Timestamp(end))]
    return ff.reset_index().rename(columns={"Date": "date", "index": "date"})


# -------------------------------------------------------------- inference ----
def hac_ols(y: np.ndarray, X: np.ndarray, lag: int | None = None) -> dict:
    """OLS with Newey-West HAC standard errors (Bartlett kernel, lag ~ n^(1/3))."""
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    n, k = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    if lag is None:
        lag = max(1, int(round(n ** (1 / 3))))
    Xe = X * e[:, None]
    S = Xe.T @ Xe / n
    for j in range(1, lag + 1):
        G = Xe[j:].T @ Xe[:-j] / n
        S += (1 - j / (lag + 1)) * (G + G.T)
    XtX_inv = np.linalg.inv(X.T @ X / n)
    V = XtX_inv @ S @ XtX_inv / n
    se = np.sqrt(np.diag(V))
    ss_res = float(e @ e)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"beta": beta, "se": se, "t": beta / se, "n": n, "lag": lag,
            "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan}


def capm(y: np.ndarray, mkt: np.ndarray) -> dict:
    X = np.column_stack([np.ones(len(y)), mkt])
    f = hac_ols(y, X)
    return {"n": f["n"], "nw_lag": f["lag"], "r2": round(f["r2"], 4),
            "beta": round(float(f["beta"][1]), 4),
            "beta_t": round(float(f["t"][1]), 2),
            "alpha_daily": float(f["beta"][0]),
            "alpha_ann": round(float(f["beta"][0]) * ANN, 4),
            "alpha_t": round(float(f["t"][0]), 2)}


# ------------------------------------------------------------- drawdowns ----
def spy_drawdown_episodes(df: pd.DataFrame, min_depth: float = 0.10) -> list[dict]:
    """Every SPY total-return drawdown deeper than min_depth, with the strategy's
    own move (and worst internal drawdown, and time in market) over the same dates."""
    tr = (1.0 + df["spy_ret"]).cumprod().to_numpy()
    strat = (1.0 + df["carry"]).cumprod().to_numpy()
    dates = df["date"].to_numpy()
    inmkt = df["inmkt"].to_numpy()
    peak = np.maximum.accumulate(tr)
    under = tr / peak - 1.0
    episodes: list[dict] = []
    i, n = 0, len(tr)
    while i < n:
        if under[i] < 0:
            start = i  # first day below the running peak; peak day is start-1
            j = i
            while j < n and under[j] < 0:
                j += 1
            seg = slice(start, j)
            depth = under[seg].min()
            if depth <= -min_depth:
                p = max(start - 1, 0)                      # SPY peak day
                t = start + int(np.argmin(under[seg]))     # SPY trough day
                s_seg = strat[p:t + 1] / strat[p]
                episodes.append({
                    "spy_peak": str(pd.Timestamp(dates[p]).date()),
                    "spy_trough": str(pd.Timestamp(dates[t]).date()),
                    "days_peak_to_trough": int(t - p),
                    "spy_dd": round(float(depth), 4),
                    "strategy_same_dates": round(float(strat[t] / strat[p] - 1.0), 4),
                    "strategy_worst_dd_within": round(float((s_seg / np.maximum.accumulate(s_seg) - 1.0).min()), 4),
                    "pct_days_in_market": round(float(inmkt[p:t + 1].mean()), 3),
                })
            i = j
        else:
            i += 1
    return episodes


# ------------------------------------------------------------------ main ----
def main() -> None:
    carry = load_carry()
    spy = load_spy_excess()
    df = carry.merge(spy, on="date", how="inner").dropna(
        subset=["carry", "spy_excess"]).reset_index(drop=True)
    y = df["carry"].to_numpy()
    m = df["spy_excess"].to_numpy()
    window = f"{df['date'].min().date()} -> {df['date'].max().date()}"
    results: dict = {"window": window, "n_days": len(df),
                     "corr_carry_spy": round(float(np.corrcoef(y, m)[0, 1]), 3)}

    # 1. CAPM ---------------------------------------------------------------
    results["capm"] = capm(y, m)
    c = results["capm"]
    print(f"panel: {len(df)} days  {window}  corr(carry, SPY)={results['corr_carry_spy']:+.2f}\n")
    print(f"[1] CAPM (NW lag {c['nw_lag']}):  beta={c['beta']:+.3f} (t={c['beta_t']:+.1f})  "
          f"alpha={c['alpha_ann']*100:+.2f}%/yr (t={c['alpha_t']:+.2f})  R2={c['r2']:.3f}")

    # 2. Fama-French 5 + momentum --------------------------------------------
    ff = load_ff6(str(df["date"].min().date()), str(df["date"].max().date()))
    if ff is not None:
        fdf = df.merge(ff, on="date", how="inner").dropna()
        cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]
        X = np.column_stack([np.ones(len(fdf))] + [fdf[c].to_numpy() for c in cols])
        f = hac_ols(fdf["carry"].to_numpy(), X)
        results["ff6"] = {
            "n": f["n"], "nw_lag": f["lag"], "r2": round(f["r2"], 4),
            "alpha_ann": round(float(f["beta"][0]) * ANN, 4),
            "alpha_t": round(float(f["t"][0]), 2),
            "loadings": {c: {"beta": round(float(b), 4), "t": round(float(t), 2)}
                         for c, b, t in zip(cols, f["beta"][1:], f["t"][1:])},
        }
        g = results["ff6"]
        loads = "  ".join(f"{c}={v['beta']:+.3f}(t={v['t']:+.1f})"
                          for c, v in g["loadings"].items())
        print(f"[2] FF5+Mom (n={g['n']}):  alpha={g['alpha_ann']*100:+.2f}%/yr "
              f"(t={g['alpha_t']:+.2f})  R2={g['r2']:.3f}\n    {loads}")
    else:
        results["ff6"] = None
        print("[2] FF5+Mom: famafrench download unavailable; skipped.")

    # 3. State-dependent beta -------------------------------------------------
    states = {
        "spy_down_days": m < 0,
        "spy_up_days": m >= 0,
        "in_market": df["inmkt"].to_numpy() == 1,
        "flat": df["inmkt"].to_numpy() == 0,
        "spy_worst_decile": m <= np.quantile(m, 0.10),
    }
    results["state_beta"] = {}
    print("[3] state-dependent beta:")
    for name, sel in states.items():
        r = capm(y[sel], m[sel])
        r["mean_spy_excess_daily"] = round(float(m[sel].mean()), 5)
        r["mean_carry_daily"] = round(float(y[sel].mean()), 5)
        r["inmkt_share"] = round(float(df["inmkt"].to_numpy()[sel].mean()), 3)
        results["state_beta"][name] = r
        print(f"    {name:<18s} n={r['n']:>4d}  beta={r['beta']:+.3f} (t={r['beta_t']:+.1f})  "
              f"intercept={r['alpha_ann']*100:+.1f}%/yr (t={r['alpha_t']:+.2f})  "
              f"in-mkt={r['inmkt_share']*100:.0f}%")

    # 4. Co-drawdown table -----------------------------------------------------
    episodes = spy_drawdown_episodes(df)
    results["co_drawdowns"] = episodes
    print("\n[4] SPY total-return drawdowns deeper than 10% (strategy over the same dates):")
    print(f"    {'peak':<12s}{'trough':<12s}{'SPY dd':>8s}{'strat same-dates':>18s}"
          f"{'strat worst-in-ep':>19s}{'% days in-mkt':>15s}")
    for e in episodes:
        print(f"    {e['spy_peak']:<12s}{e['spy_trough']:<12s}{e['spy_dd']*100:+7.1f}%"
              f"{e['strategy_same_dates']*100:+17.1f}%{e['strategy_worst_dd_within']*100:+18.1f}%"
              f"{e['pct_days_in_market']*100:>14.0f}%")

    out = REPO_ROOT / "analysis" / "factor_regression_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
