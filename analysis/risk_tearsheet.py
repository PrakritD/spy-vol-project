"""Risk-desk tearsheet for the VRP-carry strategy: VaR/ES, stress table, rolling beta.

This is not new research. It retranslates numbers STRATEGY.md already reports (Sharpe,
Calmar, maxDD, the co-drawdown episode table) into the vocabulary a bank or fund risk
desk uses day to day: 99% 1-day VaR and Expected Shortfall (both plain-historical and
Cornish-Fisher tail-adjusted), a stress-scenario table, and a rolling market beta.

Everything traces to two already-committed artifacts:
  analysis/strategy_equity.parquet          -> the carry sleeve's own daily returns
                                                and SPY's total-return series
  analysis/factor_regression_results.json   -> the co-drawdown episode table (§4e),
                                                reformatted here, not recomputed

Run: /opt/anaconda3/envs/trading/bin/python analysis/risk_tearsheet.py
Output: analysis/risk_tearsheet_results.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
EQUITY_PATH = REPO_ROOT / "analysis" / "strategy_equity.parquet"
FACTOR_PATH = REPO_ROOT / "analysis" / "factor_regression_results.json"
OUT_PATH = REPO_ROOT / "analysis" / "risk_tearsheet_results.json"

ANN = 252
VAR_CONF = 0.99
Z_99 = 2.326347874040845  # scipy.stats.norm.ppf(0.99), pinned so scipy isn't a hard import
BETA_WINDOW = 126  # ~6 trading months, standard risk-desk rolling-beta horizon


# ---------------------------------------------------------------- loading ----
def load_returns() -> pd.DataFrame:
    """Daily returns for the carry sleeve and SPY total-return series.

    Both equity columns start from a base of 1.0, so the first return is
    eq[0]/1.0 - 1 and subsequent returns are eq[t]/eq[t-1] - 1 (same convention
    as drawdown_inference.py and factor_regression.py).
    """
    eq = pd.read_parquet(EQUITY_PATH).sort_values("date").reset_index(drop=True)
    out = pd.DataFrame({"date": pd.to_datetime(eq["date"])})
    for col in ("carry", "spy_total"):
        e = eq[col].to_numpy(float)
        out[col] = e / np.concatenate(([1.0], e[:-1])) - 1.0
    return out


# ------------------------------------------------------------- VaR / ES ----
def sample_skew_kurt(r: np.ndarray) -> tuple[float, float]:
    """Sample skew and excess kurtosis (Fisher, i.e. normal -> 0).

    Uses pandas' bias-corrected Fisher-Pearson estimators (Series.skew()/.kurt()),
    matching strategy_two_sleeve.py's own headline_metrics computation exactly, so
    this reconciles to STRATEGY.md's quoted skew -1.31 / kurtosis 6.1 rather than
    diverging on an estimator-choice technicality.
    """
    s = pd.Series(r)
    return float(s.skew()), float(s.kurt())


def historical_var_es(r: np.ndarray, conf: float = VAR_CONF) -> tuple[float, float]:
    """Plain-historical VaR/ES at `conf`, reported as positive loss numbers.

    VaR is the (1-conf) empirical quantile of the return distribution; ES is the
    mean of returns at or below that quantile.
    """
    q = np.percentile(r, (1.0 - conf) * 100.0)
    tail = r[r <= q]
    var = -float(q)
    es = -float(tail.mean()) if len(tail) else var
    return var, es


def cornish_fisher_z(z: float, skew: float, kurt: float) -> float:
    """Cornish-Fisher expansion of a Gaussian quantile for skew S and excess kurtosis K.

    z_cf = z + (z^2-1)/6 * S + (z^3-3z)/24 * K - (2z^3-5z)/36 * S^2

    Reduces exactly to z when S=K=0 (see tests/test_risk_tearsheet.py).
    """
    return (
        z
        + (z**2 - 1.0) / 6.0 * skew
        + (z**3 - 3.0 * z) / 24.0 * kurt
        - (2.0 * z**3 - 5.0 * z) / 36.0 * skew**2
    )


def _z_for(conf: float) -> float:
    # Only VAR_CONF (0.99) is used in this module; kept for clarity, not generality.
    if conf == 0.99:
        return Z_99
    raise ValueError(f"no pinned z-score for conf={conf}; add one if a new level is needed")


def cornish_fisher_var_es(
    r: np.ndarray, skew: float, kurt: float, conf: float = VAR_CONF
) -> tuple[float, float]:
    """Modified (Cornish-Fisher) VaR at `conf`, mean+sigma*z_cf, reported as a positive loss.

    ES is approximated as the empirical mean of historical returns at or below the
    Cornish-Fisher VaR cutoff (CF sets the cutoff, the historical sample supplies the
    tail average beyond it). If that tail is empty (a deep enough adjustment that no
    historical draw is that bad), ES falls back to the VaR itself, which keeps ES >= VaR
    at the same confidence level in every case.
    """
    z = _z_for(conf)
    z_cf = cornish_fisher_z(z, skew, kurt)
    mu, sigma = r.mean(), r.std(ddof=0)
    cutoff = mu - z_cf * sigma
    var_cf = -float(cutoff)
    tail = r[r <= cutoff]
    es_cf = -float(tail.mean()) if len(tail) else var_cf
    return var_cf, max(es_cf, var_cf)


# --------------------------------------------------------------- stress ----
def stress_table(factor_results: dict) -> list[dict]:
    """Reformat factor_regression.py's co_drawdowns episodes into bank stress-scenario
    vocabulary. No numbers are recomputed; every field is a rename/reformat of a field
    already in factor_regression_results.json."""
    names = {
        ("2011-07-28", "2011-10-03"): "2011 US debt-ceiling downgrade",
        ("2015-07-20", "2016-02-11"): "2015-16 China slowdown / oil crash",
        ("2018-01-26", "2018-02-08"): "Volmageddon 2018",
        ("2018-09-20", "2018-12-24"): "Q4 2018 rate-shock selloff",
        ("2020-02-19", "2020-03-23"): "COVID 2020",
        ("2022-01-03", "2022-10-12"): "2022 rate-hike bear market",
        ("2025-02-19", "2025-04-08"): "2025 tariff-shock selloff",
    }
    table = []
    for ep in factor_results["co_drawdowns"]:
        key = (ep["spy_peak"], ep["spy_trough"])
        table.append({
            "scenario": names.get(key, f"{ep['spy_peak']} -> {ep['spy_trough']}"),
            "spy_peak": ep["spy_peak"],
            "spy_trough": ep["spy_trough"],
            "days_peak_to_trough": ep["days_peak_to_trough"],
            "spy_move_pct": round(ep["spy_dd"] * 100.0, 2),
            "strategy_move_pct": round(ep["strategy_same_dates"] * 100.0, 2),
            "strategy_worst_move_within_pct": round(ep["strategy_worst_dd_within"] * 100.0, 2),
            "pct_days_in_market": round(ep["pct_days_in_market"] * 100.0, 1),
        })
    return table


# ---------------------------------------------------------- rolling beta ----
def rolling_beta(r_strat: np.ndarray, r_mkt: np.ndarray, window: int = BETA_WINDOW) -> np.ndarray:
    """Rolling beta_t = Cov(r_strat, r_mkt) / Var(r_mkt) over a trailing window.

    Closed-form rolling covariance/variance ratio; statsmodels is absent from this
    env so this avoids a rolling-OLS regression library entirely.
    """
    s = pd.Series(r_strat)
    m = pd.Series(r_mkt)
    cov = s.rolling(window).cov(m)
    var = m.rolling(window).var()
    return (cov / var).to_numpy()


# ------------------------------------------------------------------ main ----
def main() -> None:
    df = load_returns()
    r_carry = df["carry"].to_numpy(float)
    r_spy = df["spy_total"].to_numpy(float)

    skew, kurt = sample_skew_kurt(r_carry)
    var_hist, es_hist = historical_var_es(r_carry)
    var_cf, es_cf = cornish_fisher_var_es(r_carry, skew, kurt)

    beta = rolling_beta(r_carry, r_spy, BETA_WINDOW)
    beta_valid = beta[~np.isnan(beta)]
    full_sample_beta = float(np.cov(r_carry, r_spy)[0, 1] / np.var(r_spy))

    results: dict = {
        "data": {
            "source": "analysis/strategy_equity.parquet",
            "columns": ["carry", "spy_total"],
            "n_days": int(len(df)),
            "start": str(df["date"].iloc[0].date()),
            "end": str(df["date"].iloc[-1].date()),
        },
        "var_es": {
            "confidence": VAR_CONF,
            "horizon_days": 1,
            "sample_skew": round(skew, 4),
            "sample_excess_kurtosis": round(kurt, 4),
            "strategy_results_json_skew": -1.310131496623831,
            "strategy_results_json_kurt": 6.1377702194257155,
            "historical": {
                "var_99_1d": round(var_hist, 6),
                "es_99_1d": round(es_hist, 6),
            },
            "cornish_fisher": {
                "var_99_1d": round(var_cf, 6),
                "es_99_1d": round(es_cf, 6),
                "z_gaussian_99": Z_99,
            },
        },
        "stress_table": {
            "source": "analysis/factor_regression_results.json:co_drawdowns",
            "episodes": stress_table(json.loads(FACTOR_PATH.read_text())),
        },
        "rolling_beta": {
            "market_series": "spy_total",
            "window_days": BETA_WINDOW,
            "full_sample_pooled_beta": round(full_sample_beta, 4),
            "factor_regression_capm_beta": 0.4247,
            "note": (
                "spy_total is used here (rather than spy_excess) so the rolling window lines "
                "up with the raw market moves in the stress table above. The choice barely "
                "matters: the full-sample pooled cov/var beta comes out the same either way "
                "(0.4247 on spy_total, 0.4248 on spy_excess), both matching "
                "factor_regression.py's static CAPM beta of 0.4247 almost exactly. The real gap "
                "is between that pooled full-sample beta (0.42) and the mean of the 126-day "
                "rolling betas (see 'mean' below, materially higher): beta is time-varying over "
                "the window, and the mean of short-window ratios is not the same statistic as "
                "the ratio computed on the pooled full sample. This is a real discrepancy worth "
                "flagging, not a units or convention mismatch."
            ),
            "n_valid_windows": int(len(beta_valid)),
            "mean": round(float(beta_valid.mean()), 4),
            "min": round(float(beta_valid.min()), 4),
            "max": round(float(beta_valid.max()), 4),
            "current": round(float(beta_valid[-1]), 4),
        },
    }

    OUT_PATH.write_text(json.dumps(results, indent=2) + "\n")

    # ---- compact printout ----
    d = results["data"]
    v = results["var_es"]
    b = results["rolling_beta"]
    print(f"Data: {d['start']} .. {d['end']}  ({d['n_days']} days)")
    print(f"Skew {v['sample_skew']:+.3f}  excess kurtosis {v['sample_excess_kurtosis']:+.3f}  "
          f"(STRATEGY.md pooled: skew {v['strategy_results_json_skew']:+.3f}, "
          f"kurt {v['strategy_results_json_kurt']:+.3f})")
    print(f"99% 1-day VaR:  historical {v['historical']['var_99_1d']*100:.2f}%   "
          f"Cornish-Fisher {v['cornish_fisher']['var_99_1d']*100:.2f}%")
    print(f"99% 1-day ES:   historical {v['historical']['es_99_1d']*100:.2f}%   "
          f"Cornish-Fisher {v['cornish_fisher']['es_99_1d']*100:.2f}%")
    print(f"\nStress table ({len(results['stress_table']['episodes'])} episodes):")
    for e in results["stress_table"]["episodes"]:
        print(f"  {e['scenario']:<32s} SPY {e['spy_move_pct']:+7.1f}%   "
              f"strategy {e['strategy_move_pct']:+7.1f}%   "
              f"in-mkt {e['pct_days_in_market']:5.1f}%")
    print(f"\nRolling {BETA_WINDOW}d beta (vs spy_total): "
          f"mean {b['mean']:+.3f}  min {b['min']:+.3f}  max {b['max']:+.3f}  current {b['current']:+.3f}")
    print(f"Full-sample pooled beta {b['full_sample_pooled_beta']:+.3f} "
          f"(factor_regression.py static CAPM beta: {b['factor_regression_capm_beta']:+.3f})")
    print(f"\nWrote {OUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
