"""Term-structure slope/curvature (PCA) as a continuous VRP-carry sizing signal.

STRATEGY.md Sec.7 flags this as the natural next step: replace the binary contango switch
with a signal sized to the PREDICTED MAGNITUDE of the roll, not just its sign, noting that "a
crude continuous version already under-performed the binary gate." This uses the real futures
curve (`vix_futures_curve.py`'s constant-maturity tenor levels, 30-180 calendar days,
gap-free/correctly-scaled only 2008-2013), not the 3-point VIX/VIX3M/VIX9D index proxy used
elsewhere in the repo.

Causal, walk-forward, same TRAIN0/REFIT_EVERY/EMBARGO convention as
`strategy_two_sleeve.ml_size_positions`/`timing_positions`: tenor levels are lagged one day
before anything touches them (this repo's shift(1) rule), then PCA loadings and the per-tenor
standardization are fit on an EXPANDING training window only, refit monthly with a 5-day
embargo, and projected onto the current (already-lagged) day to get that day's PC1 (level) /
PC2 (slope) / PC3 (curvature) scores. PC2's sign is fixed each refit by its train-window
correlation with contango depth (1 - t_30_90), so "higher PC2" always means "measured deeper
contango," not an arbitrary PCA sign.

The size multiplier (mirrors `ml_size_positions`' clip + expanding-mean-normalize pattern)
scales the EXISTING binary contango_flag gate; it does not replace it, since PC2 is defined
even on flat/backwardation days when the gate is already off.

Run: python analysis/vix_futures_term_pca.py
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

REPO = __file__.rsplit("/analysis/", 1)[0]
sys.path.insert(0, REPO + "/analysis")
import strategy_two_sleeve as S  # noqa: E402
import vix_futures_curve as V    # noqa: E402

TENOR_COLS = [f"cm_{m}" for m in V.DEFAULT_TENORS]
LAGGED_COLS = [c + "_l" for c in TENOR_COLS]
CAP = 2.0            # matches ml_size_positions' magnitude cap
PCA_MIN_TRAIN = 60    # expanding-mean warmup for the size normalizer, matches ml_size_positions


def build_panel() -> pd.DataFrame:
    p = V.load_futures_panel()
    curve = V.build_curve(p)
    term = V.build_term_structure(p)
    term_struct_signal = V.load_term_structure_panel()

    d = term_struct_signal.merge(curve[["date", "index_ret"]], on="date", how="inner")
    d = d.merge(term, on="date", how="inner")
    d = d[(d["date"] >= V.WINDOW_START) & (d["date"] <= V.WINDOW_END)]
    d = d.dropna(subset=["index_ret", "t_30_90", "rf_d"] + TENOR_COLS).reset_index(drop=True)
    for c in TENOR_COLS:                      # info known at the close BEFORE position formed
        d[c + "_l"] = d[c].shift(1)
    return d.dropna(subset=LAGGED_COLS).reset_index(drop=True)


def pc2_walkforward(d: pd.DataFrame) -> np.ndarray:
    """Causal PC2 (slope) score, oriented so higher = deeper measured contango. NaN before
    TRAIN0 (the same warmup strategy_two_sleeve's own walk-forward sleeves use)."""
    X = d[LAGGED_COLS].to_numpy()
    contango_depth = (1.0 - d["t_30_90"].to_numpy())   # positive & larger = deeper contango
    n = len(d)
    pc2 = np.full(n, np.nan)
    mu = sd = pca = sign = None
    for i in range(S.TRAIN0, n):
        if (i - S.TRAIN0) % S.REFIT_EVERY == 0:
            Xtr = X[: i - S.EMBARGO]
            mu, sd = Xtr.mean(0), Xtr.std(0)
            sd[sd == 0] = 1
            Xtr_z = (Xtr - mu) / sd
            pca = PCA(n_components=3, random_state=0).fit(Xtr_z)
            raw_tr = pca.transform(Xtr_z)[:, 1]
            corr = np.corrcoef(raw_tr, contango_depth[: i - S.EMBARGO])[0, 1]
            sign = 1.0 if corr >= 0 else -1.0
        xi = ((X[i] - mu) / sd).reshape(1, -1)
        pc2[i] = sign * pca.transform(xi)[0, 1]
    return pc2


def size_multiplier(pc2: np.ndarray, cap: float = CAP) -> np.ndarray:
    raw = np.clip(np.nan_to_num(pc2, nan=0.0), 0, None)
    scale = (pd.Series(np.where(raw > 0, raw, np.nan))
             .expanding(PCA_MIN_TRAIN).mean().shift(1).to_numpy())
    mult = np.divide(raw, scale, out=np.zeros_like(raw), where=(scale > 0) & np.isfinite(scale))
    return np.clip(mult, 0.0, cap)


def main() -> int:
    d = build_panel()
    print(f"panel: {len(d)} rows, {d['date'].min().date()} -> {d['date'].max().date()}")

    pc2 = pc2_walkforward(d)
    mult = size_multiplier(pc2)
    flag = S.contango_flag(d)
    warmup = S.TRAIN0
    dates = d["date"].to_numpy()
    idx_ret = d["index_ret"].to_numpy()
    rf = d["rf_d"].to_numpy()

    pos_binary = -1.0 * S.NOTIONAL * flag
    r_binary = S.sleeve_excess(pos_binary, idx_ret, rf, S.CostCfg().vixy_bps, 0.0)
    m_binary = S.metrics(r_binary[warmup:], dates[warmup:],
                         "binary gate (post-warmup window, matched to PCA)")

    pos_pca = -1.0 * S.NOTIONAL * flag * mult
    r_pca = S.sleeve_excess(pos_pca, idx_ret, rf, S.CostCfg().vixy_bps, 0.0)
    r_pca_test = r_pca[warmup:]
    dates_test = dates[warmup:]
    m_pca = S.metrics(r_pca_test, dates_test, "PCA slope-scaled gate")

    # Robustness on the PCA result itself, mirroring STRATEGY.md's own "few-days fragility"
    # and sub-period checks: a >0.5 Sharpe jump from one untuned specification (six tenors,
    # cap=2.0, no sweep across alternatives) deserves scrutiny before it is trusted at all.
    order = np.argsort(r_pca_test)
    fragility = {}
    for k in (1, 3, 5, 10):
        keep = np.ones(len(r_pca_test), dtype=bool)
        keep[order[-k:]] = False
        fragility[f"minus_top{k}"] = float(S.metrics(r_pca_test[keep], dates_test[keep], "")["sharpe"])
    mid = len(r_pca_test) // 2
    split = {
        "first_half": S.metrics(r_pca_test[:mid], dates_test[:mid], "first half"),
        "second_half": S.metrics(r_pca_test[mid:], dates_test[mid:], "second half"),
    }
    hac_t = float(S.hac_tstat(r_pca_test))

    print(f"\n{m_binary['name']}: n={m_binary['n']} Sharpe={m_binary['sharpe']:+.2f} "
          f"Calmar={m_binary.get('calmar', float('nan')):+.2f} "
          f"maxDD={m_binary.get('maxdd', float('nan'))*100:+.1f}%")
    print(f"{m_pca['name']}: n={m_pca['n']} Sharpe={m_pca['sharpe']:+.2f} "
          f"Calmar={m_pca.get('calmar', float('nan')):+.2f} "
          f"maxDD={m_pca.get('maxdd', float('nan'))*100:+.1f}%")
    print(f"HAC t-stat: {hac_t:+.2f}")
    print(f"fragility (Sharpe minus top-k days): {fragility}")
    print(f"split-sample Sharpe: first half {split['first_half']['sharpe']:+.2f}  "
          f"second half {split['second_half']['sharpe']:+.2f}")

    out = {
        "window": [V.WINDOW_START, V.WINDOW_END], "tenors": list(V.DEFAULT_TENORS),
        "train0": S.TRAIN0, "refit_every": S.REFIT_EVERY, "embargo": S.EMBARGO, "cap": CAP,
        "binary_gate": m_binary, "pca_slope_scaled_gate": m_pca,
        "pca_hac_tstat": hac_t, "pca_fragility": fragility, "pca_split_sample": split,
        "caveat": ("single untuned specification (tenors/cap/refit cadence chosen once, not "
                  "swept); no multiplicity-adjusted DSR was computed for it unlike the "
                  "flagship strategy's 22-variant deflated Sharpe -- read this as a proof of "
                  "concept on a narrow ~6-year window, not a validated headline result"),
    }
    with open(f"{REPO}/analysis/vix_futures_term_pca_results.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("\nsaved analysis/vix_futures_term_pca_results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
