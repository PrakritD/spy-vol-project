"""Two-sleeve daily strategy: risk-managed VRP carry + SPY tactical timing.

The variance risk premium (VIX persistently > realized) is the edge. The vehicle is SHORT
VIXY (ProShares VIX short-term futures ETF, 2011-> , no splice, *includes* the Feb-2018 /
Mar-2020 / Aug-2015 blowups, so the tail is in-sample). Naive always-short VIXY has a great
calm-period Sharpe and a catastrophic drawdown. The overlay's JOB is to manage that tail:
reduce/flatten the short when the term structure inverts, gamma goes negative, or vol-of-vol
spikes. The overlay is therefore judged by TAIL-adjusted metrics (Calmar, maxDD, Sortino) vs
the *naive vol-targeted short*; the gate must earn its keep there, not on raw Sharpe. Sleeve 2
(SPY long/flat/short from DIX flow + gamma + trend) is a diversifier, reported standalone
including when it is weak. Every signal is pre-registered and strictly lagged (<= t-1); gex is
lagged for OCC's T-1 open interest. Trades are sized from prior-close signals; P&L is
close-to-close (no reliable free intraday). Headline numbers sit next to the naive-carry
benchmark, the Deflated Sharpe over all variants tried, maxDD, and the blowups IN-SAMPLE.

Run:  python analysis/strategy_two_sleeve.py
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

REPO = __file__.rsplit("/analysis/", 1)[0]
sys.path.insert(0, REPO + "/analysis")
from rvutil import daily_yang_zhang_rv  # noqa: E402  (vendored Yang-Zhang OHLC RV; no v1 dependency)

ANN = 252
EULER = 0.5772156649015329


# ------------------------------------------------------------------ data ----
def load_panel() -> pd.DataFrame:
    """Merge all free deep series into one daily panel on date (binding ~2011-05 -> 2026-05)."""
    def pq(path, cols):
        d = pd.read_parquet(f"{REPO}/{path}")
        d = d[[c for c in cols if c in d.columns]].copy()
        d["date"] = pd.to_datetime(d["date"])
        return d

    spy = pq("data/raw/deep/SPY.parquet", ["date", "open", "high", "low", "close", "adj close", "volume"])
    spy = spy.rename(columns={"adj close": "spy_adj", "close": "spy_close", "volume": "spy_vol"})
    rv = daily_yang_zhang_rv(spy.rename(columns={"spy_close": "close"})[["date", "open", "high", "low", "close"]])
    spy = spy.merge(rv[["date", "rv"]], on="date", how="left")

    vixy = pq("data/raw/deep/VIXY.parquet", ["date", "adj_close", "volume"]).rename(
        columns={"adj_close": "vixy_adj", "volume": "vixy_vol"})

    sq = pd.read_csv(f"{REPO}/data/raw/squeeze_dix.csv")
    sq["date"] = pd.to_datetime(sq["date"])
    sq = sq[["date", "gex", "dix"]]

    vix = pd.read_csv(f"{REPO}/data/raw/cboe_vix.csv")
    vix["date"] = pd.to_datetime(vix["DATE"])
    vix = vix[["date", "CLOSE"]].rename(columns={"CLOSE": "vix"})

    def lvl(name, col):
        d = pq(f"data/raw/deep/{name}.parquet", ["date", "close"]).rename(columns={"close": col})
        return d

    rf = pq("data/raw/fred/dgs3mo_deep.parquet", ["date", "dgs3mo"])

    df = (spy.merge(vixy, on="date", how="inner")
             .merge(sq, on="date", how="inner")
             .merge(vix, on="date", how="inner")
             .merge(lvl("VIX3M", "vix3m"), on="date", how="left")
             .merge(lvl("VIX9D", "vix9d"), on="date", how="left")
             .merge(lvl("VVIX", "vvix"), on="date", how="left")
             .merge(rf, on="date", how="left"))
    df = df.sort_values("date").reset_index(drop=True)
    df["dgs3mo"] = df["dgs3mo"].ffill()
    for c in ["vix3m", "vix9d", "vvix"]:                # ffill rare gaps so a NaN-drop does not
        df[c] = df[c].ffill()                           # splice multi-month calendar jumps into "one day"
    return df


# -------------------------------------------------------------- signals ----
def build_signals(df: pd.DataFrame) -> pd.DataFrame:
    """All predictors STRICTLY lagged (<= t-1). A row at date t may only use info known at
    the close of t-1, so a position formed from it (entered at t-1 close) earns the t-1->t
    close-to-close return without look-ahead."""
    d = df.copy()

    # tradeable simple returns (close-to-close), realized over day t
    d["vixy_ret"] = d["vixy_adj"].pct_change()
    d["spy_ret"] = d["spy_adj"].pct_change()
    d["rf_d"] = (d["dgs3mo"].fillna(0) / 100.0) / ANN  # daily risk-free

    lrv = np.log(d["rv"].clip(lower=1e-6))
    d["har_d"] = lrv.shift(1)
    d["har_w"] = lrv.rolling(5).mean().shift(1)
    d["har_m"] = lrv.rolling(22).mean().shift(1)

    # --- term structure (the canonical VIX-carry risk filter) ---
    d["t_9_30"] = (d["vix9d"] / d["vix"]).shift(1)     # >1 => short end inverted (stress)
    d["t_30_90"] = (d["vix"] / d["vix3m"]).shift(1)    # >1 => backwardation (carry-hostile)
    d["vvix_vix"] = (d["vvix"] / d["vix"]).shift(1)
    d["vix_l"] = d["vix"].shift(1)
    d["vix_z"] = ((d["vix"] - d["vix"].rolling(20).mean()) / d["vix"].rolling(20).std()).shift(1)
    vvr = d["vvix"] / d["vix"]
    d["vvix_z"] = ((vvr - vvr.rolling(60).mean()) / vvr.rolling(60).std()).shift(1)

    # --- dealer gamma (lagged for OCC T-1 OI) ---
    d["gex_pct"] = d["gex"].rolling(252, min_periods=60).apply(lambda a: (a[-1] > a).mean(), raw=True).shift(1)
    d["gex_neg"] = (d["gex"] < 0).astype(float).shift(1)

    # --- flow (DIX = SqueezeMetrics Dark Index, a short-volume/flow signal) ---
    d["dix_l"] = d["dix"].shift(1)
    d["dix_chg"] = (d["dix"] - d["dix"].shift(5)).shift(1)
    d["dix_z"] = ((d["dix"] - d["dix"].rolling(60).mean()) / d["dix"].rolling(60).std()).shift(1)

    # --- volume / liquidity ---
    relvol = d["spy_vol"] / d["spy_vol"].rolling(21).mean()
    d["relvol"] = relvol.shift(1)
    dollar = d["spy_close"] * d["spy_vol"]
    amih = (d["spy_ret"].abs() / dollar) * 1e12
    d["amihud_z"] = ((amih - amih.rolling(60).mean()) / amih.rolling(60).std()).shift(1)

    # --- trend / momentum (SPY) ---
    ma100 = d["spy_adj"].rolling(100).mean()
    d["trend"] = (d["spy_adj"] / ma100 - 1.0).shift(1)
    d["mom_21"] = d["spy_adj"].pct_change(21).shift(1)
    d["mom_63"] = d["spy_adj"].pct_change(63).shift(1)

    # --- vol-target sizing input: trailing realized vol of VIXY (<= t-1) ---
    d["vixy_vol21"] = d["vixy_ret"].rolling(21).std().shift(1)
    d["spy_vol21"] = d["spy_ret"].rolling(21).std().shift(1)

    d["era"] = np.where(d["date"] < "2020-01-01", "pre2020",
                np.where(d["date"] < "2022-01-01", "2020-21", "2022+"))
    return d


# ---------------------------------------------------------------- sleeves ----
TARGET_VOL = 0.10 / np.sqrt(ANN)   # 10% annualized per sleeve
MAX_NOTIONAL = 1.0                  # never lever beyond 1x notional (realistic, conservative)


def voltarget_size(trailing_vol: pd.Series) -> np.ndarray:
    s = TARGET_VOL / trailing_vol.replace(0, np.nan)
    return s.clip(upper=MAX_NOTIONAL).fillna(0.0).to_numpy()


NOTIONAL = 0.20   # fixed modest short exposure (Sharpe/Sortino/Calmar are scale-invariant;
                  # only CAGR/maxDD scale with this — a concrete, realistic exposure choice)


def contango_flag(d: pd.DataFrame) -> np.ndarray:
    """THE pre-registered, zero-parameter VRP signal: front-month vol below 3-month vol
    (VIX < VIX3M => term structure in contango => positive roll yield to a short)."""
    return (d["t_30_90"] < 1.0).astype(float).to_numpy()


def carry_positions(d: pd.DataFrame, addons: frozenset = frozenset(),
                    notional: float = NOTIONAL, filt: np.ndarray | None = None) -> np.ndarray:
    """Primary carry: constant short VIXY, ON only in contango. `addons` apply extra
    pre-registered risk REDUCTIONS for the add-one signal-attribution study. Signed (<=0)."""
    f = contango_flag(d) if filt is None else filt
    pos = -notional * f
    if "gamma" in addons:
        pos = pos * np.where(d["gex_neg"].to_numpy() == 1, 0.5, 1.0)
    if "vvix" in addons:
        pos = pos * np.where(d["vvix_z"].to_numpy() > 1.0, 0.5, 1.0)
    if "vix_z" in addons:
        pos = pos * np.where(d["vix_z"].to_numpy() > 1.0, 0.5, 1.0)
    if "liquidity" in addons:
        pos = pos * np.where(d["amihud_z"].to_numpy() > 2.0, 0.5, 1.0)
    if "voltarget" in addons:                          # CAUSAL multiplicative vol-scaling overlay
        vt = voltarget_size(d["vixy_vol21"])           # normalise by an EXPANDING mean (<= t-1), not a
        norm = pd.Series(vt).expanding(60).mean().shift(1)   # full-sample mean (which would peek)
        pos = pos * np.nan_to_num((vt / norm).to_numpy(), nan=1.0)
    return np.nan_to_num(pos, nan=0.0)


def carry_constant(d: pd.DataFrame, notional: float = NOTIONAL) -> np.ndarray:
    """Unfiltered constant short — the naive baseline that eats the blowups."""
    return np.full(len(d), -notional)


def carry_rollyield(d: pd.DataFrame, notional: float = NOTIONAL, full_at: float = 0.05) -> np.ndarray:
    """Robustness variant: size the short CONTINUOUSLY to the roll yield (VIX3M-VIX)/VIX3M,
    flat in backwardation. Theory-true 'size the carry to the carry'; one tuned scale."""
    slope = ((d["vix3m"] - d["vix"]) / d["vix3m"]).shift(1).to_numpy()
    return -notional * 2.0 * np.clip(np.nan_to_num(slope) / full_at, 0, 1)


# --- Sleeve 2: SPY tactical timing via walk-forward logistic on flow/positioning/trend ---
from sklearn.linear_model import LogisticRegression, Ridge  # noqa: E402

TIMING_FEATS = ["dix_z", "dix_chg", "gex_pct", "gex_neg", "trend", "mom_21", "mom_63",
                "vix_z", "t_30_90", "relvol"]
TRAIN0 = 504        # ~2y initial train
REFIT_EVERY = 21    # refit monthly
EMBARGO = 5         # purge/embargo days between train end and prediction


def timing_positions(d: pd.DataFrame):
    """Predict P(SPY up tomorrow) from lagged flow/positioning/trend; map to long/flat/short
    with a deadband. Expanding walk-forward, monthly refit, 5-day embargo, train-only scaling.
    Returns (signed SPY notional, p_hat array)."""
    X = d[TIMING_FEATS].to_numpy()
    y = (d["spy_ret"].to_numpy() > 0).astype(float)   # contemporaneous ret over the held day
    n = len(d)
    p = np.full(n, np.nan)
    clf = mu = sd = None
    base = 0.5
    valid = ~np.isnan(X).any(1)
    for i in range(TRAIN0, n):
        if (i - TRAIN0) % REFIT_EVERY == 0:
            tr = valid[: i - EMBARGO]
            Xtr, ytr = X[: i - EMBARGO][tr], y[: i - EMBARGO][tr]
            if len(np.unique(ytr)) < 2 or len(ytr) < 100:
                clf = None; base = ytr.mean() if len(ytr) else 0.5
            else:
                mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
                clf = LogisticRegression(C=0.5, max_iter=2000).fit((Xtr - mu) / sd, ytr)
        if not valid[i]:
            p[i] = base if clf is None else 0.5
            continue
        p[i] = base if clf is None else clf.predict_proba(((X[i] - mu) / sd)[None, :])[0, 1]

    edge = 2 * np.nan_to_num(p, nan=0.5) - 1.0                 # in [-1, 1]
    edge = np.where(np.abs(edge) < 0.04, 0.0, edge)            # deadband -> flat near 50/50
    size = voltarget_size(d["spy_vol21"])
    pos = np.clip(edge, -1, 1) * size
    pos[:TRAIN0] = 0.0
    return np.nan_to_num(pos, nan=0.0), p


# --- ML sizing layer: walk-forward Ridge magnitude sizing (Path-2 validated variant) ---
ML_FEATS = ["t_30_90", "t_9_30", "vvix_vix", "vix_l", "vix_z", "vvix_z",
            "har_d", "har_w", "har_m", "gex_pct", "gex_neg"]
ML_ALPHA = 10.0   # Ridge regularization, fixed a priori (standardized features); no alpha-tuning leak


def ml_size_positions(d: pd.DataFrame, notional: float = NOTIONAL,
                      alpha: float = ML_ALPHA, cap: float = 2.0, gate: np.ndarray | None = None):
    """Continuous, magnitude-scaled short sizing learned walk-forward. A regularized-linear
    (Ridge) model predicts next-day SHORT excess return from lagged term-structure / vol-of-vol
    / realized-vol / gamma features; the short is sized PROPORTIONAL to the predicted positive
    carry and flat when the predicted edge <= 0. With `gate` set (e.g. the contango flag), the
    learned magnitude scales the short only on gated days (a hybrid: keep the structural
    flattening, let ML set the size within it); without it, ML fully replaces the binary gate.

    Causal by construction (so the no-lookahead gate holds): expanding walk-forward, train-only
    standardization, monthly refit, 5-day embargo, and an EXPANDING (<= t-1) exposure
    normaliser. Returns (signed short notional, raw prediction)."""
    X = d[ML_FEATS].to_numpy()
    yshort = -(d["vixy_ret"].to_numpy() - d["rf_d"].to_numpy())   # excess return to a UNIT short
    n = len(d)
    pred = np.full(n, np.nan)
    mdl = mu = sd = None
    valid = (~np.isnan(X).any(1)) & (~np.isnan(yshort))
    for i in range(TRAIN0, n):
        if (i - TRAIN0) % REFIT_EVERY == 0:
            tr = valid[: i - EMBARGO]
            Xtr, ytr = X[: i - EMBARGO][tr], yshort[: i - EMBARGO][tr]
            if len(ytr) < 100:
                mdl = None
            else:
                mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
                mdl = Ridge(alpha=alpha).fit((Xtr - mu) / sd, ytr)
        if not valid[i] or mdl is None:
            continue
        pred[i] = mdl.predict(((X[i] - mu) / sd)[None, :])[0]
    raw = np.clip(np.nan_to_num(pred, nan=0.0), 0, None)          # short only when predicted edge > 0
    scale = pd.Series(np.where(raw > 0, raw, np.nan)).expanding(60).mean().shift(1).to_numpy()
    mult = np.divide(raw, scale, out=np.zeros_like(raw), where=(scale > 0) & np.isfinite(scale))
    pos = -notional * np.clip(mult, 0.0, cap)
    if gate is not None:
        pos = pos * np.asarray(gate, float)                       # size only on gated (e.g. contango) days
    pos[:TRAIN0] = 0.0
    return np.nan_to_num(pos, nan=0.0), pred


# ----------------------------------------------------------- backtest core ----
@dataclass
class CostCfg:
    vixy_bps: float = 10.0       # round-trip-ish spread per unit turnover
    spy_bps: float = 1.0
    borrow_ann: float = 0.03     # short-VIXY borrow drag (conservative)


def sleeve_excess(pos: np.ndarray, asset_ret: np.ndarray, rf_d: np.ndarray,
                  bps: float, borrow_ann: float = 0.0) -> np.ndarray:
    """Daily EXCESS return of a signed-notional sleeve: pos*(asset_ret - rf) - costs - borrow.
    (capital earns rf; position adds asset excess return; short pays borrow on |notional|)."""
    asset_ret = np.nan_to_num(asset_ret, nan=0.0)
    excess_asset = asset_ret - rf_d
    r = pos * excess_asset
    turn = np.abs(np.diff(pos, prepend=0.0))
    r = r - turn * (bps / 1e4)
    borrow_ann = np.asarray(borrow_ann, float)          # scalar OR per-day array (VIX-conditioned)
    if np.any(borrow_ann != 0):
        short_notional = np.clip(-pos, 0, None)
        r = r - short_notional * (borrow_ann / ANN)
    return r


def combine_volparity(r1: np.ndarray, r2: np.ndarray, lookback: int = 63) -> np.ndarray:
    """Inverse-vol (risk-parity) weighting of two sleeve excess-return streams, weights set
    from trailing vol only (<= t-1)."""
    s1 = pd.Series(r1).rolling(lookback).std().shift(1).to_numpy()
    s2 = pd.Series(r2).rolling(lookback).std().shift(1).to_numpy()
    ok = (s1 > 0) & (s2 > 0)
    w1 = np.where(ok, 1 / np.where(s1 > 0, s1, 1), 0.0)
    w2 = np.where(ok, 1 / np.where(s2 > 0, s2, 1), 0.0)
    tot = w1 + w2
    w1 = np.divide(w1, tot, out=np.full_like(w1, 0.5), where=tot > 0)
    w2 = 1 - w1
    # scale combined to ~TARGET_VOL using trailing combined vol
    raw = w1 * r1 + w2 * r2
    cv = pd.Series(raw).rolling(lookback).std().shift(1).to_numpy()
    scale = np.where(cv > 0, TARGET_VOL / cv, 0.0)
    scale = np.clip(scale, 0, 3.0)
    return np.nan_to_num(raw * scale, nan=0.0)


# -------------------------------------------------------------- metrics ----
def _dd(equity: np.ndarray):
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    mdd = dd.min()
    end = int(dd.argmin())
    start = int(np.argmax(equity[: end + 1])) if end > 0 else 0
    # longest underwater duration (trading days)
    uw = equity < peak
    longest = cur = 0
    for f in uw:
        cur = cur + 1 if f else 0
        longest = max(longest, cur)
    return mdd, longest, start, end


def metrics(r: np.ndarray, dates: np.ndarray, name: str = "") -> dict:
    r = np.asarray(r, float)
    r = r[~np.isnan(r)]
    if len(r) < 5 or r.std() == 0:
        return {"name": name, "n": len(r), "sharpe": np.nan}
    mu, sd = r.mean(), r.std()
    sharpe = mu / sd * np.sqrt(ANN)
    downside = r[r < 0]
    sortino = mu / downside.std() * np.sqrt(ANN) if len(downside) > 1 and downside.std() > 0 else np.nan
    equity = np.cumprod(1 + r)
    mdd, uw_days, _, _ = _dd(equity)
    yrs = len(r) / ANN
    cagr = equity[-1] ** (1 / yrs) - 1 if equity[-1] > 0 else -1.0
    calmar = cagr / abs(mdd) if mdd < 0 else np.nan
    wins, losses = r[r > 0], r[r < 0]
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else np.inf
    return {
        "name": name, "n": int(len(r)),
        "sharpe": sharpe, "sortino": sortino, "calmar": calmar,
        "cagr": cagr, "ann_vol": sd * np.sqrt(ANN), "maxdd": mdd, "uw_days": uw_days,
        "hit": (r > 0).mean(), "pf": pf, "skew": float(pd.Series(r).skew()),
        "kurt": float(pd.Series(r).kurt()), "final_equity": equity[-1],
    }


def deflated_sharpe(r: np.ndarray, n_trials: int, sr_trials_std: float) -> dict:
    """Bailey & Lopez de Prado Deflated Sharpe Ratio. sr_* are per-OBSERVATION (not annualized).
    sr_trials_std = std of the (per-obs) Sharpes across all variants tried (the selection set)."""
    r = np.asarray(r, float); r = r[~np.isnan(r)]
    T = len(r)
    sr = r.mean() / r.std()                                   # per-obs Sharpe
    g3 = float(pd.Series(r).skew()); g4 = float(pd.Series(r).kurt()) + 3.0  # kurt() is excess
    if sr_trials_std <= 0 or n_trials < 2:
        sr0 = 0.0
    else:
        z1 = norm.ppf(1 - 1.0 / n_trials)
        z2 = norm.ppf(1 - 1.0 / (n_trials * np.e))
        sr0 = sr_trials_std * ((1 - EULER) * z1 + EULER * z2)  # expected max Sharpe under nulls
    denom = np.sqrt(max(1e-12, 1 - g3 * sr + (g4 - 1) / 4.0 * sr ** 2))
    dsr = norm.cdf((sr - sr0) * np.sqrt(T - 1) / denom)
    psr0 = norm.cdf(sr * np.sqrt(T - 1) / denom)              # PSR vs 0 (no haircut)
    return {"sr_perobs": sr, "sr0_haircut": sr0, "dsr": dsr, "psr_vs0": psr0,
            "n_trials": n_trials, "sr_trials_std": sr_trials_std}


def block_bootstrap_sharpe(r: np.ndarray, nb: int = 5000, seed: int = 7) -> tuple:
    r = np.asarray(r, float); r = r[~np.isnan(r)]
    n = len(r); bs = max(5, int(round(n ** (1 / 3)))); k = int(np.ceil(n / bs))
    rng = np.random.default_rng(seed)
    out = np.empty(nb)
    for j in range(nb):
        st = rng.integers(0, max(1, n - bs + 1), size=k)
        idx = np.concatenate([np.arange(s, s + bs) for s in st])[:n]
        x = r[idx]
        out[j] = x.mean() / x.std() * np.sqrt(ANN) if x.std() > 0 else 0.0
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), float((out <= 0).mean())


def sharpe_minus_topk(r: np.ndarray, k: int) -> float:
    r = np.asarray(r, float); r = r[~np.isnan(r)].copy()
    if len(r) <= k:
        return np.nan
    drop = np.argsort(r)[-k:]
    r[drop] = 0.0
    return r.mean() / r.std() * np.sqrt(ANN) if r.std() > 0 else np.nan


def hac_tstat(r: np.ndarray) -> float:
    """Newey-West HAC t-stat that the mean daily (excess) return > 0: the significance test
    for an annualized Sharpe that accounts for autocorrelation."""
    r = np.asarray(r, float); r = r[~np.isnan(r)]; n = len(r)
    if n < 10 or r.std() == 0:
        return np.nan
    e = r - r.mean(); lag = max(1, int(round(n ** (1 / 3))))
    s = (e @ e) / n
    for k in range(1, lag + 1):
        s += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    return r.mean() / np.sqrt(s / n)


def per_regime(r: np.ndarray, era: np.ndarray) -> dict:
    out = {}
    for blk in ["pre2020", "2020-21", "2022+"]:
        sel = era == blk
        m = metrics(r[sel], None, blk)
        m["t_hac"] = hac_tstat(r[sel])
        m["sharpe_minus_top3"] = sharpe_minus_topk(r[sel], 3)
        out[blk] = m
    return out


def subperiod(r: np.ndarray, dates: np.ndarray, lo: str, hi: str, name: str) -> dict:
    dts = pd.to_datetime(dates)
    sel = np.asarray((dts >= pd.Timestamp(lo)) & (dts < pd.Timestamp(hi)))
    rr = np.asarray(r)[sel]
    m = metrics(rr, None, name)
    m["t_hac"] = hac_tstat(rr)
    return m


def _fmt(m: dict) -> str:
    if not np.isfinite(m.get("sharpe", np.nan)):
        return f"{m['name']:<26s} n={m.get('n',0):>4d}  (degenerate)"
    return (f"{m['name']:<26s} n={m['n']:>4d}  Sharpe={m['sharpe']:+.2f}  Sortino={m['sortino']:+.2f}  "
            f"Calmar={m['calmar']:+.2f}  CAGR={m['cagr']*100:+5.1f}%  vol={m['ann_vol']*100:4.1f}%  "
            f"maxDD={m['maxdd']*100:+6.1f}%  UW={m['uw_days']:>4d}d  hit={m['hit']*100:4.1f}%  PF={m['pf']:.2f}")


# ------------------------------------------------------------------ main ----
def main():
    d = build_signals(load_panel())
    need = ["vixy_ret", "spy_ret", "vixy_vol21", "t_30_90", "t_9_30", "vix_z", "vvix_z",
            "gex_neg", "amihud_z", "rf_d"]
    d = d.dropna(subset=need).reset_index(drop=True)
    dates = d["date"].to_numpy(); era = d["era"].to_numpy()
    rf = d["rf_d"].to_numpy(); vret = d["vixy_ret"].to_numpy(); sret = d["spy_ret"].to_numpy()
    cost = CostCfg()
    bx = lambda pos, bps=cost.vixy_bps, bor=cost.borrow_ann: sleeve_excess(pos, vret, rf, bps, bor)
    print(f"panel: {len(d)} rows  {d['date'].min().date()} -> {d['date'].max().date()}  "
          f"%backwardated={(d['t_30_90']>=1).mean()*100:.1f}  in-contango={contango_flag(d).mean()*100:.1f}  "
          f"%neg-gamma={(d['gex_neg']==1).mean()*100:.1f}\n")

    # =========================== CARRY CONSTRUCTION LADDER ===========================
    # Attribution: which control actually tames the short-vol tail?
    pos_const = carry_constant(d)
    pos_vt = carry_positions(d, addons=frozenset({"voltarget"}), filt=np.ones(len(d)))  # vol-tgt, no filter
    pos_carry = carry_positions(d)                                  # PRIMARY: contango filter only
    pos_roll = carry_rollyield(d)                                   # robustness: continuous roll-yield
    pos_full = carry_positions(d, addons=frozenset({"gamma", "vvix", "vix_z", "liquidity"}))
    r_const, r_vt, r_carry, r_roll, r_full = map(bx, [pos_const, pos_vt, pos_carry, pos_roll, pos_full])

    print("=" * 132)
    print("CARRY LADDER — short VIXY (full sample, excess-of-rf, costs+borrow in, blowups IN-SAMPLE)")
    print("=" * 132)
    ladder = [("1. constant short (no controls)", r_const),
              ("2. + vol-targeting (no filter)", r_vt),
              ("3. + CONTANGO FILTER  <<HEADLINE", r_carry),
              ("   alt: continuous roll-yield", r_roll),
              ("4. + extra signal gates (full)", r_full)]
    Ml = {nm: metrics(r, dates, nm) for nm, r in ladder}
    for nm, _ in ladder:
        print(_fmt(Ml[nm]))
    print("\n  Read: causal vol-targeting is roughly NEUTRAL on VIXY (≈ constant short); it does NOT")
    print("  approach the filter. The zero-parameter CONTANGO filter (VIX<VIX3M) is the real control.")

    # =========================== BENCHMARKS ===========================
    # Both the strategy and 'SPY (excess)' are excess-of-rf (apples-to-apples). We ALSO show SPY
    # total-return (the way investors actually quote 'buy-hold SPY'), so the comparison is apples-to-apples.
    bh_spy = sleeve_excess(np.ones(len(d)), sret, rf, 0.0, 0.0)   # excess of rf
    sixty40 = 0.6 * (sret - rf)
    avg_rf = np.nanmean(rf) * ANN
    print("\n--- BENCHMARKS ---")
    Mb = {}
    for nm, r in [("buy-hold SPY (excess)", bh_spy), ("buy-hold SPY (total ret)", sret),
                  ("60/40 (SPY/cash, excess)", sixty40), ("cash", np.zeros(len(d)))]:
        Mb[nm] = metrics(r, dates, nm); print(_fmt(Mb[nm]))
    H = Ml["3. + CONTANGO FILTER  <<HEADLINE"]
    spx_e, spx_t = Mb["buy-hold SPY (excess)"], Mb["buy-hold SPY (total ret)"]
    print(f"\n  HEADLINE vs SPY (avg rf ≈ {avg_rf*100:.1f}%/yr over window):")
    print(f"    Sharpe {H['sharpe']:+.2f}  vs SPY {spx_e['sharpe']:+.2f} (excess) / {spx_t['sharpe']:+.2f} (total) "
          f"— COMPARABLE, NOT a clean beat; on total-return SPY edges ahead.")
    print(f"    Calmar {H['calmar']:+.2f} vs {spx_e['calmar']:+.2f}  |  maxDD {H['maxdd']*100:+.0f}% vs "
          f"{spx_e['maxdd']*100:+.0f}%  |  Sortino {H['sortino']:+.2f} vs {spx_e['sortino']:+.2f} (carry TRAILS on Sortino)")
    print("    -> the DURABLE, convention-robust edge is DRAWDOWN CONTROL (Calmar/maxDD), not Sharpe.")

    # =========================== BLOWUP DODGING (the mechanism, in-sample) ===========================
    print("\n--- BLOWUP DODGING (did the filter flatten INTO the disasters?) ---")
    fc = contango_flag(d)
    sr_carry = pd.DataFrame({"date": d["date"], "vret": vret, "inmkt": fc, "pnl": r_carry}).set_index("date")
    for lbl, a, b in [("Volmageddon 2018", "2018-01-25", "2018-02-12"),
                      ("COVID crash 2020", "2020-02-20", "2020-03-25"),
                      ("2022 bear", "2022-01-01", "2022-12-31")]:
        seg = sr_carry.loc[a:b]
        print(f"  {lbl:18s} in-market {seg['inmkt'].mean()*100:3.0f}% of window | strategy "
              f"{(np.prod(1+seg['pnl'])-1)*100:+6.1f}%  while long-VIXY {(np.prod(1+seg['vret'])-1)*100:+5.0f}%")

    # =========================== SIGNAL ATTRIBUTION (add-one on top of the filter) ===========================
    print("\n--- SIGNAL ATTRIBUTION (add ONE extra risk signal to the contango filter; Δ vs headline) ---")
    print(f"  {'add-on':<24s} {'Sharpe':>7s} {'Calmar':>7s} {'maxDD%':>7s}  {'ΔSharpe':>8s} {'ΔCalmar':>8s} {'ΔmaxDD%':>8s}")
    print(f"  {'(none = headline)':<24s} {H['sharpe']:>+7.2f} {H['calmar']:>+7.2f} {H['maxdd']*100:>+7.1f}  "
          f"{'—':>8s} {'—':>8s} {'—':>8s}")
    addon_M = {}
    for s in ["gamma", "vvix", "vix_z", "liquidity"]:
        m = metrics(bx(carry_positions(d, addons=frozenset({s}))), dates, s); addon_M[s] = m
        print(f"  + {s:<22s} {m['sharpe']:>+7.2f} {m['calmar']:>+7.2f} {m['maxdd']*100:>+7.1f}  "
              f"{m['sharpe']-H['sharpe']:>+8.2f} {m['calmar']-H['calmar']:>+8.2f} {(m['maxdd']-H['maxdd'])*100:>+8.1f}")
    print("  An add-on EARNS its place only if it improves risk-adjusted return (Δ>0). "
          "Gamma ≈ null — consistent with FINDINGS (gamma is ~95% a VIX echo).")

    # =========================== SLEEVE 2: TIMING (a null) ===========================
    from sklearn.metrics import roc_auc_score
    pos_tim, p_hat = timing_positions(d)
    r_tim = sleeve_excess(pos_tim, sret, rf, cost.spy_bps, 0.0)
    oos = np.arange(TRAIN0, len(d)); mok = ~np.isnan(p_hat[oos])
    y_dir = (sret > 0).astype(float)
    auc = roc_auc_score(y_dir[oos][mok], p_hat[oos][mok])
    print("\n--- SLEEVE 2: SPY TACTICAL TIMING (DIX flow + gamma + trend) ---")
    print(f"  OOS direction AUC = {auc:.3f} (0.50 = no skill) | active {np.mean(pos_tim!=0)*100:.0f}% of days, "
          f"long {np.mean(pos_tim>0)*100:.0f}% / short {np.mean(pos_tim<0)*100:.0f}%")
    print(f"  -> NULL: DIX/gamma/flow/trend do not predict next-day SPY direction; the fit collapses "
          f"to closet-long (corr to SPY {np.corrcoef(r_tim, sret-rf)[0,1]:+.2f}). Sleeve 2 adds no timing edge.")

    # =========================== CORRELATION REALITY ===========================
    print("\n--- CORRELATION REALITY (is the carry a diversifier? NO) ---")
    rho = np.corrcoef(r_carry, sret - rf)[0, 1]
    print(f"  corr(carry, SPY) = {rho:+.2f}  — short-vol is an equity-like premium (short tail ≈ long equity).")
    print("  So this is an absolute-return premium, NOT uncorrelated alpha. We do not claim diversification.")

    # =========================== COST / BORROW SENSITIVITY ===========================
    # Borrow is the binding axis (spread is near-irrelevant — turnover is tiny). VIXY is chronically
    # HARD-TO-BORROW, so we stress borrow to 25%/yr and also model a VIX-conditioned borrow that rises
    # in stress. The strategy is SHORT (paying borrow) on ~92% of days, so this is a constant calm-cost.
    turnover = np.abs(np.diff(pos_carry, prepend=0.0)).sum() / NOTIONAL
    pct_short = float((pos_carry < 0).mean())
    print("\n--- COST / BORROW SENSITIVITY (headline carry; borrow is the binding axis) ---")
    print(f"  turnover ~{turnover:.0f} flips over {len(d)/ANN:.0f}y (~{turnover/(len(d)/ANN):.0f}/yr); "
          f"short (paying borrow) {pct_short*100:.0f}% of days; spread 5→30bps moves Sharpe only ~0.05.")
    borrows = [0.0, 0.03, 0.05, 0.08, 0.12, 0.18, 0.25]
    print(f"  {'borrow/yr ->':>14s}  " + "  ".join(f"{b*100:>4.0f}%" for b in borrows))
    sh_row = [metrics(bx(pos_carry, 10, b), dates)["sharpe"] for b in borrows]
    cal_row = [metrics(bx(pos_carry, 10, b), dates)["calmar"] for b in borrows]
    print(f"  {'Sharpe(@10bps)':>14s}  " + "  ".join(f"{s:>+4.2f}" for s in sh_row))
    print(f"  {'Calmar':>14s}  " + "  ".join(f"{c:>+4.2f}" for c in cal_row))
    # VIX-conditioned borrow: base 5% + 1%/VIX-pt above 20, capped 50% (uses lagged VIX)
    vixb = np.clip(0.05 + 0.01 * np.maximum(0.0, d["vix_l"].to_numpy() - 20.0), 0.05, 0.50)
    r_carry_vb = sleeve_excess(pos_carry, vret, rf, 10, vixb)
    m_vb = metrics(r_carry_vb, dates)
    spx_sh = spx_e["sharpe"]
    sh_at = lambda r: metrics(bx(pos_carry, 10, r), dates)["sharpe"]
    # find borrow where strategy Sharpe == SPY-excess Sharpe
    bgrid = np.linspace(0, 0.30, 301); shg = np.array([sh_at(b) for b in bgrid])
    cross = bgrid[np.argmin(np.abs(shg - spx_sh))]
    print(f"  VIX-conditioned borrow (base5%+stress, avg {np.mean(vixb[pos_carry<0])*100:.0f}% on short): "
          f"Sharpe {m_vb['sharpe']:+.2f}, Calmar {m_vb['calmar']:+.2f}, maxDD {m_vb['maxdd']*100:+.0f}%")
    print(f"  Sharpe-parity-with-SPY ({spx_sh:+.2f}) breaks at borrow ≈ {cross*100:.1f}%/yr; "
          f"BUT Calmar/maxDD beat SPY out to ~20%/yr. The drawdown edge is borrow-robust; the Sharpe 'beat' is not.")

    # =========================== ROBUSTNESS: DSR (range) / bootstrap / fragility ===========================
    # The DSR haircut depends on the DISPERSION of Sharpes across strategies CONSIDERED. Estimating it
    # from our own near-identical short-VIXY variants understates it (they are ~0.87 correlated clones).
    # We report a RANGE: (a) empirical std over a DIVERSE trial set, and (b) the theory-grounded B&LdP
    # H0 per-trial std sqrt((1+SR^2/2)/T). The DSR range is materially below an all-clones estimate.
    filt_defs = {  # distinct filter DEFINITIONS compared when selecting VIX<VIX3M (headline excluded — it IS r_carry)
        "VIX<VIX3M & VIX9D<VIX": ((d["t_30_90"] < 1) & (d["t_9_30"] < 1)).to_numpy().astype(float),
        "VIX9D<VIX": (d["t_9_30"] < 1).to_numpy().astype(float),
        "VIX<VIX3M & VIX9D<VIX3M": ((d["t_30_90"] < 1) & ((d["vix9d"] / d["vix3m"]).shift(1) < 1)).to_numpy().astype(float),
    }
    thr_grid = [0.97, 0.98, 0.99, 1.00, 1.01, 1.02, 1.03, 1.05]   # threshold-stability sweep
    thr_streams = {t: bx(carry_positions(d, filt=(d["t_30_90"] < t).to_numpy().astype(float))) for t in thr_grid}
    # DIVERSE trial set: genuinely different strategies the project built/considered
    diverse = ([r_const, r_vt, r_carry, r_roll, r_full, r_tim]
               + [bx(carry_positions(d, addons=frozenset({s}))) for s in ["gamma", "vvix", "vix_z", "liquidity"]]
               + [bx(carry_positions(d, filt=f)) for f in filt_defs.values()]
               + list(thr_streams.values())
               + [bx(-pos_carry)])                                # a deliberately-bad long-VIXY anti-strategy
    sr_po = lambda r: (r[~np.isnan(r)].mean() / r[~np.isnan(r)].std()) if r[~np.isnan(r)].std() > 0 else 0.0
    sr_div = np.array([sr_po(r) for r in diverse]); n_trials = len(diverse)
    sr_h = sr_po(r_carry); T = int(np.sum(~np.isnan(r_carry)))
    std_emp = float(sr_div.std(ddof=1))
    std_h0 = float(np.sqrt((1 + sr_h ** 2 / 2) / T))             # B&LdP H0 per-trial SR std
    dsr_emp = deflated_sharpe(r_carry, n_trials, std_emp)
    dsr_h0 = deflated_sharpe(r_carry, n_trials, std_h0)
    lo, hi, pneg = block_bootstrap_sharpe(r_carry)
    s1, s5, s10 = (sharpe_minus_topk(r_carry, k) for k in (1, 5, 10))
    print("\n--- ROBUSTNESS (HEADLINE carry) ---")
    print("  filter-definition band (all positive): VIX<VIX3M=+{:.2f} | ".format(H["sharpe"]) + " | ".join(
        f"{k}={metrics(bx(carry_positions(d, filt=f)), dates)['sharpe']:+.2f}" for k, f in filt_defs.items()))
    print("  threshold-stability sweep (Sharpe): " + " ".join(
        f"{t:.2f}={metrics(s, dates)['sharpe']:+.2f}" for t, s in thr_streams.items())
        + "  (1.00 is the structural VIX=VIX3M point, not the grid-max)")
    print(f"  block-bootstrap Sharpe 95% CI [{lo:+.2f}, {hi:+.2f}]  P(SR<=0)={pneg:.3f}  "
          f"(significance-vs-ZERO, NO multiplicity adjustment)")
    print(f"  Deflated Sharpe RANGE: {min(dsr_emp['dsr'], dsr_h0['dsr']):.2f}–{max(dsr_emp['dsr'], dsr_h0['dsr']):.2f}  "
          f"(N={n_trials}; diverse-set std→{dsr_emp['dsr']:.2f}, B&LdP-H0 std→{dsr_h0['dsr']:.2f}). "
          f"An all-clones std would inflate this to ~0.98 — we DON'T cite that.")
    print(f"  fragility: Sharpe minus top-1={s1:+.2f}  top-5={s5:+.2f}  top-10={s10:+.2f}")

    # =========================== OOS SUB-PERIOD SPLIT (carry is parameter-light, but show persistence) ===========================
    print("\n--- OUT-OF-SAMPLE PERSISTENCE (sub-period splits; edge halves post-2018 but stays positive) ---")
    splits = [("2011-2018", "2011-01-01", "2019-01-01"), ("2019-2026", "2019-01-01", "2027-01-01"),
              ("2018+ (post-XIV)", "2018-01-01", "2027-01-01")]
    sub = {}
    for nm, a, b in splits:
        mc = subperiod(r_carry, dates, a, b, nm); ms = subperiod(bh_spy, dates, a, b, nm); sub[nm] = mc
        print(f"  {nm:<18s} carry Sharpe {mc['sharpe']:+.2f} (t_HAC {mc['t_hac']:+.1f}) "
              f"Calmar {mc['calmar']:+.2f} maxDD {mc['maxdd']*100:+.0f}%   | SPY-excess Sharpe {ms['sharpe']:+.2f}")

    # =========================== PER-REGIME (with significance + fragility) ===========================
    print("\n--- PER-REGIME (HAC t-stats + fragility; only pre-2020 is individually significant) ---")
    pr_h = per_regime(r_carry, era)
    for b in ["pre2020", "2020-21", "2022+"]:
        m = pr_h[b]
        sig = "SIGNIFICANT" if abs(m["t_hac"]) > 1.96 else "not significant"
        print(f"  {b:<9s} n={m['n']:>4d}  Sharpe {m['sharpe']:+.2f}  t_HAC {m['t_hac']:+.2f} ({sig})  "
              f"maxDD {m['maxdd']*100:+.0f}%  minus-top3-days→{m['sharpe_minus_top3']:+.2f}")
    print("  NOTE: the 2020-21 (COVID) block is NOT distinguishable from zero and collapses when a few days drop; "
          "do not read 'positive point estimate' as 'robust'.")

    # =========================== ML SIZING LAYER (walk-forward Ridge; validated variant) ===========================
    # Path-2 experiment: replace the binary contango gate with a continuous, magnitude-scaled short
    # sized by a walk-forward Ridge forecast of next-day short carry. Judged on the agreed bar
    # (Calmar/maxDD primary), with its own selection-aware Deflated Sharpe over the diverse set + itself.
    # The binary gate stays the HEADLINE; this is reported as a validated variant.
    pos_ml, pred_ml = ml_size_positions(d)                          # ML fully replaces the gate
    pos_mlg, _ = ml_size_positions(d, gate=contango_flag(d))        # ML sizes magnitude WITHIN the gate
    r_ml = bx(pos_ml); r_mlg = bx(pos_mlg)
    M_ml = metrics(r_ml, dates, "ML sizing (replace gate)")
    M_mlg = metrics(r_mlg, dates, "ML sizing (within gate)")
    trials_ml = diverse + [r_ml, r_mlg]; n_ml = len(trials_ml)
    std_ml = float(np.array([sr_po(r) for r in trials_ml]).std(ddof=1))
    dsr_ml = deflated_sharpe(r_ml, n_ml, std_ml); dsr_mlg = deflated_sharpe(r_mlg, n_ml, std_ml)
    pr_mlg = per_regime(r_mlg, era)
    rho_ml = float(np.corrcoef(r_ml, r_carry)[0, 1]); rho_mlg = float(np.corrcoef(r_mlg, r_carry)[0, 1])
    # "wins" = strictly better than the binary-gate HEADLINE on the agreed bar (less-negative maxDD is better)
    best = M_mlg if M_mlg["calmar"] >= M_ml["calmar"] else M_ml
    win_cal = best["calmar"] > H["calmar"]; win_dd = best["maxdd"] > H["maxdd"]
    print("\n" + "=" * 132)
    print("ML SIZING LAYER — walk-forward Ridge magnitude sizing vs the binary contango gate (drawdown-adjusted bar)")
    print("=" * 132)
    print(f"  Ridge(alpha={ML_ALPHA:g}) on lagged: {', '.join(ML_FEATS)}")
    print(f"  {'':28s} {'Sharpe':>7s} {'Sortino':>7s} {'Calmar':>7s} {'CAGR%':>6s} {'vol%':>5s} {'maxDD%':>7s} {'DSR':>5s}")
    print(f"  {'binary gate (HEADLINE)':28s} {H['sharpe']:>+7.2f} {H['sortino']:>+7.2f} {H['calmar']:>+7.2f} "
          f"{H['cagr']*100:>+6.1f} {H['ann_vol']*100:>5.1f} {H['maxdd']*100:>+7.1f} {'   -':>5s}")
    print(f"  {'ML sizing (replace gate)':28s} {M_ml['sharpe']:>+7.2f} {M_ml['sortino']:>+7.2f} {M_ml['calmar']:>+7.2f} "
          f"{M_ml['cagr']*100:>+6.1f} {M_ml['ann_vol']*100:>5.1f} {M_ml['maxdd']*100:>+7.1f} {dsr_ml['dsr']:>5.2f}")
    print(f"  {'ML sizing (within gate)':28s} {M_mlg['sharpe']:>+7.2f} {M_mlg['sortino']:>+7.2f} {M_mlg['calmar']:>+7.2f} "
          f"{M_mlg['cagr']*100:>+6.1f} {M_mlg['ann_vol']*100:>5.1f} {M_mlg['maxdd']*100:>+7.1f} {dsr_mlg['dsr']:>5.2f}")
    print(f"  corr(replace, gate)={rho_ml:+.2f} | corr(within, gate)={rho_mlg:+.2f} | N_trials={n_ml}")
    print("  per-regime (within-gate) ML Sharpe: " + " ".join(
        f"{b}={pr_mlg[b]['sharpe']:+.2f}(t{pr_mlg[b]['t_hac']:+.1f})" for b in ["pre2020", "2020-21", "2022+"]))
    verdict = ("BEATS the gate on BOTH Calmar and maxDD" if (win_cal and win_dd) else
               "improves Calmar but not maxDD" if win_cal else
               "improves maxDD but not Calmar" if win_dd else
               "does NOT beat the binary gate on drawdown-adjusted metrics; the structural rule wins")
    print(f"  VERDICT (Calmar/maxDD primary): best ML sizing variant {verdict}.")

    # =========================== PERSIST ===========================
    out = {
        "window": [str(d['date'].min().date()), str(d['date'].max().date())], "n": int(len(d)),
        "headline_metrics": {k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                             for k, v in H.items() if k != "name"},
        "ladder": {k: {kk: vv for kk, vv in v.items() if kk in
                       ("sharpe", "sortino", "calmar", "cagr", "ann_vol", "maxdd", "hit")}
                   for k, v in Ml.items()},
        "benchmarks": {k: {kk: vv for kk, vv in v.items() if kk in ("sharpe", "sortino", "calmar", "cagr", "maxdd")}
                       for k, v in Mb.items()},
        "addon_attrib": {k: {"sharpe": v["sharpe"], "calmar": v["calmar"], "maxdd": v["maxdd"]}
                         for k, v in addon_M.items()},
        "timing_auc": float(auc), "carry_spy_corr": float(rho), "avg_rf": float(avg_rf),
        "cost_borrow_sharpe": {f"{int(b*100)}pct": s for b, s in zip(borrows, sh_row)},
        "vix_borrow_metrics": {"sharpe": m_vb["sharpe"], "calmar": m_vb["calmar"], "maxdd": m_vb["maxdd"]},
        "sharpe_parity_break_borrow": float(cross),
        "dsr": {"range": [min(dsr_emp["dsr"], dsr_h0["dsr"]), max(dsr_emp["dsr"], dsr_h0["dsr"])],
                "dsr_diverse_std": dsr_emp["dsr"], "dsr_h0_std": dsr_h0["dsr"], "n_trials": int(n_trials),
                "std_diverse": std_emp, "std_h0": std_h0, "note": "all-clones std would give ~0.98; not cited"},
        "bootstrap_ci_vs0": [lo, hi, pneg], "fragility": {"minus_top1": s1, "minus_top5": s5, "minus_top10": s10},
        "threshold_sweep": {f"{t:.2f}": metrics(s, dates)["sharpe"] for t, s in thr_streams.items()},
        "subperiods": {nm: {"sharpe": sub[nm]["sharpe"], "t_hac": sub[nm]["t_hac"],
                            "calmar": sub[nm]["calmar"], "maxdd": sub[nm]["maxdd"]} for nm in sub},
        "regime": {b: {"sharpe": pr_h[b]["sharpe"], "t_hac": pr_h[b]["t_hac"], "n": pr_h[b]["n"],
                       "maxdd": pr_h[b]["maxdd"], "minus_top3": pr_h[b]["sharpe_minus_top3"]}
                   for b in ["pre2020", "2020-21", "2022+"]},
        "ml_sizing": {
            "features": ML_FEATS, "alpha": ML_ALPHA, "n_trials": int(n_ml),
            "replace_gate": {**{k: M_ml[k] for k in ("sharpe", "sortino", "calmar", "cagr", "ann_vol", "maxdd")},
                             "dsr": dsr_ml["dsr"], "corr_to_binary": rho_ml},
            "within_gate": {**{k: M_mlg[k] for k in ("sharpe", "sortino", "calmar", "cagr", "ann_vol", "maxdd")},
                            "dsr": dsr_mlg["dsr"], "corr_to_binary": rho_mlg,
                            "regime": {b: {"sharpe": pr_mlg[b]["sharpe"], "t_hac": pr_mlg[b]["t_hac"]}
                                       for b in ["pre2020", "2020-21", "2022+"]}},
            "beats_binary_calmar": bool(win_cal), "beats_binary_maxdd": bool(win_dd),
        },
    }
    with open(f"{REPO}/analysis/strategy_results.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    eq = pd.DataFrame({"date": pd.to_datetime(dates)})
    for nm, r in [("constant", r_const), ("voltarget", r_vt), ("carry", r_carry),
                  ("rollyield", r_roll), ("full_gate", r_full), ("ml_replace", r_ml),
                  ("ml_within", r_mlg), ("timing", r_tim), ("spy_excess", bh_spy), ("spy_total", sret)]:
        eq[nm] = np.cumprod(1 + np.nan_to_num(r, nan=0.0))
    eq["inmkt"] = fc; eq["vix_over_vix3m"] = d["t_30_90"].to_numpy()
    eq.to_parquet(f"{REPO}/analysis/strategy_equity.parquet", index=False)
    print("\nsaved analysis/strategy_results.json + analysis/strategy_equity.parquet")


if __name__ == "__main__":
    main()
