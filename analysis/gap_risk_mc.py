"""Regime-conditional Monte Carlo for the carry sleeve's residual gap risk.

STRATEGY.md Sec.6 states qualitatively that a true overnight gap through the daily contango
gate is the residual risk the rule cannot hedge with a once-a-day signal: the gate can only
flatten VIXY exposure at the next close, so a crisis that opens and gaps before the gate flips
flat still lands in the carry sleeve's return stream. This script turns that qualitative
statement into a number: given the carry sleeve's own historical realized dynamics, what is
P(maxDD worse than -20% / -25% / -30%) over a full backtest-length horizon?

This is a DIFFERENT question from `drawdown_inference.py`, which runs a PAIRED bootstrap of
carry vs SPY to support a comparative "shallower drawdown" claim. Here there is no comparison
series: this is a single-series stationary block bootstrap of the carry sleeve's own daily
return stream, in isolation, to characterize the dispersion of its own worst-case drawdown.

Method and regime-conditioning choice: the carry column already encodes the contango gate --
`inmkt` is 1 on days the gate is short VIXY and 0 (near-zero return) on flat days. Rather than
splitting the series into two separately-resampled in-market/flat sub-series (which would
require re-stitching regime transitions and risks pairing a resampled in-market return with a
resampled flat-day run that never actually followed it), this script runs a single Politis-
Romano stationary block bootstrap (geometric block lengths, wrap-around indexing, the same
construction as `drawdown_inference.py`) directly over the historical (date, return, inmkt)
sequence. Each block is a contiguous historical stretch, so it already carries whatever regime
dominated that stretch -- a block drawn from a crisis run stays a crisis run, a block drawn
from a long flat stretch stays flat. This keeps return and regime label paired exactly as they
occurred, which is what matters for tail drawdown risk: a flat day contributes ~0 return by
construction, so the in-market regime's own block structure (how crisis stretches cluster) is
the only structure that needs preserving, and contiguous-block resampling preserves it
directly without any extra bookkeeping.

Caveat (also written into the JSON): resampling scrambles the ordering of blocks across the
full path, so this is a dispersion/robustness statement about the rule's own historical
dynamics, not a forecast of the next realized drawdown. It answers "how bad has this rule's
own return stream been able to get, under exchangeable-block resampling of its own history,"
not "how bad will it get."

Run: /opt/anaconda3/envs/trading/bin/python analysis/gap_risk_mc.py
Output: analysis/gap_risk_mc_results.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
EQUITY_PATH = REPO_ROOT / "analysis" / "strategy_equity.parquet"
OUT_PATH = REPO_ROOT / "analysis" / "gap_risk_mc_results.json"

ANN = 252
N_DRAWS = 5000
SEED = 7
MAIN_BLOCK = 90
SENSITIVITY_BLOCKS = (30, 180)
DD_THRESHOLDS = (-0.20, -0.25, -0.30)
CHUNK = 500  # draws per vectorized chunk (memory guard)

CAVEAT = (
    "Bootstrap paths scramble the ordering of historical blocks, so resampled maxDD draws "
    "measure the dispersion of the carry sleeve's own drawdown risk under exchangeable-block "
    "resampling of its own history, not a forecast of the next realized drawdown."
)


def load_carry_series() -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Daily carry-sleeve returns and the contango-gate regime label from the equity curves.

    The carry column starts from a base of 1.0 (cumulative product form), so the first
    return is eq[0]/1.0 - 1 and subsequent returns are eq[t]/eq[t-1] - 1. `inmkt` is 1.0
    on days the gate is short VIXY and 0.0 on flat days; it is carried alongside the return
    series purely as a diagnostic (to confirm regime prevalence survives resampling), not
    used to drive the bootstrap itself -- see module docstring for why.
    """
    df = pd.read_parquet(EQUITY_PATH)
    dates = pd.DatetimeIndex(df["date"])
    eq = df["carry"].to_numpy(float)
    r = eq / np.concatenate(([1.0], eq[:-1])) - 1.0
    inmkt = df["inmkt"].to_numpy(float)
    return r, inmkt, dates


def stationary_bootstrap_indices(
    n: int, n_draws: int, mean_block: float, rng: np.random.Generator
) -> np.ndarray:
    """(n_draws, n) index matrix from the Politis-Romano stationary bootstrap.

    Each position restarts a block with prob 1/mean_block at a uniform start; otherwise it
    continues the previous index, wrapping around mod n. Fully vectorized: block starts are
    located with a running maximum over positions. Same construction as
    `drawdown_inference.py::stationary_bootstrap_indices`.
    """
    p = 1.0 / mean_block
    restart = rng.random((n_draws, n)) < p
    restart[:, 0] = True
    starts = rng.integers(0, n, size=(n_draws, n)).astype(np.int64)
    pos = np.arange(n, dtype=np.int64)
    # position (column) of the most recent restart, per draw
    start_pos = np.maximum.accumulate(np.where(restart, pos[None, :], -1), axis=1)
    offset = pos[None, :] - start_pos
    start_val = np.take_along_axis(starts, start_pos, axis=1)
    return (start_val + offset) % n


def path_metrics(r: np.ndarray) -> dict[str, np.ndarray]:
    """Vectorized Sharpe / maxDD / CAGR / Calmar for a (draws, n) return matrix."""
    n = r.shape[1]
    log_eq = np.cumsum(np.log1p(r), axis=1)
    dd_log = log_eq - np.maximum.accumulate(log_eq, axis=1)
    maxdd = np.expm1(dd_log.min(axis=1))
    cagr = np.expm1(log_eq[:, -1] * (ANN / n))
    sd = r.std(axis=1)
    sharpe = np.where(sd > 0, r.mean(axis=1) / np.where(sd > 0, sd, 1.0) * np.sqrt(ANN), 0.0)
    calmar = np.where(maxdd < 0, cagr / np.abs(maxdd), np.nan)
    return {"sharpe": sharpe, "maxdd": maxdd, "cagr": cagr, "calmar": calmar}


def sample_metrics(r: np.ndarray) -> dict[str, float]:
    m = path_metrics(r[None, :])
    return {k: float(v[0]) for k, v in m.items()}


def ci(x: np.ndarray) -> list[float]:
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def run_bootstrap(
    r: np.ndarray, inmkt: np.ndarray, mean_block: float, n_draws: int, seed: int
) -> dict:
    n = len(r)
    rng = np.random.default_rng([seed, int(mean_block)])
    maxdd_parts: list[np.ndarray] = []
    sharpe_parts: list[np.ndarray] = []
    calmar_parts: list[np.ndarray] = []
    inmkt_frac_parts: list[np.ndarray] = []
    for lo in range(0, n_draws, CHUNK):
        idx = stationary_bootstrap_indices(n, min(CHUNK, n_draws - lo), mean_block, rng)
        m = path_metrics(r[idx])
        maxdd_parts.append(m["maxdd"])
        sharpe_parts.append(m["sharpe"])
        calmar_parts.append(m["calmar"])
        inmkt_frac_parts.append(inmkt[idx].mean(axis=1))
    maxdd = np.concatenate(maxdd_parts)
    sharpe = np.concatenate(sharpe_parts)
    calmar = np.concatenate(calmar_parts)
    inmkt_frac = np.concatenate(inmkt_frac_parts)
    return {
        "mean_block_days": mean_block,
        "n_draws": n_draws,
        "maxdd_ci95": ci(maxdd),
        "maxdd_median": float(np.median(maxdd)),
        "maxdd_mean": float(np.mean(maxdd)),
        "sharpe_ci95": ci(sharpe),
        "calmar_ci95": ci(calmar),
        "mean_inmkt_fraction": float(np.mean(inmkt_frac)),
        "p_maxdd_worse_than_-20pct": float((maxdd < DD_THRESHOLDS[0]).mean()),
        "p_maxdd_worse_than_-25pct": float((maxdd < DD_THRESHOLDS[1]).mean()),
        "p_maxdd_worse_than_-30pct": float((maxdd < DD_THRESHOLDS[2]).mean()),
    }


def main() -> None:
    r, inmkt, dates = load_carry_series()
    point = sample_metrics(r)
    point["inmkt_fraction"] = float(np.mean(inmkt))

    results = {
        "method": (
            "Single-series Politis-Romano stationary bootstrap (geometric block lengths, "
            "wrap-around) over the carry sleeve's own historical (date, return, inmkt) "
            "sequence; blocks are contiguous historical stretches, so each resampled block "
            "keeps its historical return and contango-gate regime label paired, preserving "
            "in-market crisis clustering without splitting into separately-resampled regime "
            "sub-series. Answers P(maxDD worse than -20%/-25%/-30%) over a full "
            "backtest-length horizon, conditioned on the carry sleeve's own realized dynamics."
        ),
        "caveat": CAVEAT,
        "data": {
            "source": "analysis/strategy_equity.parquet",
            "columns": ["carry", "inmkt"],
            "n_days": int(len(r)),
            "start": str(dates[0].date()),
            "end": str(dates[-1].date()),
        },
        "seed": SEED,
        "point_estimate": point,
        "main": run_bootstrap(r, inmkt, MAIN_BLOCK, N_DRAWS, SEED),
        "sensitivity": {
            str(b): run_bootstrap(r, inmkt, b, N_DRAWS, SEED)
            for b in SENSITIVITY_BLOCKS
        },
    }

    OUT_PATH.write_text(json.dumps(results, indent=2) + "\n")

    # ---- compact printout ----
    print(f"Data: {results['data']['start']} .. {results['data']['end']}  "
          f"({results['data']['n_days']} days)  draws={N_DRAWS}  seed={SEED}")
    print(f"Point: carry maxDD {point['maxdd']:.1%}  Sharpe {point['sharpe']:.2f}  "
          f"Calmar {point['calmar']:.2f}  in-market fraction {point['inmkt_fraction']:.1%}")
    print()
    hdr = (f"{'block':>6} {'maxDD 95% CI':>18} {'median maxDD':>13} {'P(DD<-20%)':>11} "
           f"{'P(DD<-25%)':>11} {'P(DD<-30%)':>11} {'mean in-mkt frac':>17}")
    print(hdr)
    print("-" * len(hdr))
    for label, res in [("30", results["sensitivity"]["30"]),
                       ("90", results["main"]),
                       ("180", results["sensitivity"]["180"])]:
        dd = res["maxdd_ci95"]
        print(f"{label:>6} {f'[{dd[0]:.1%}, {dd[1]:.1%}]':>18} "
              f"{res['maxdd_median']:>13.1%} "
              f"{res['p_maxdd_worse_than_-20pct']:>11.3f} "
              f"{res['p_maxdd_worse_than_-25pct']:>11.3f} "
              f"{res['p_maxdd_worse_than_-30pct']:>11.3f} "
              f"{res['mean_inmkt_fraction']:>17.1%}")
    print()
    print(f"Caveat: {CAVEAT}")
    print(f"Wrote {OUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
