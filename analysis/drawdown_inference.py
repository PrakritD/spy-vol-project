"""Bootstrap inference for the drawdown edge of the VRP-carry strategy.

The strategy's pitch is drawdown control (Calmar, maxDD) rather than a Sharpe beat,
but the existing bootstrap (`block_bootstrap_sharpe`) only covers Sharpe and uses
~15-day blocks, too short to preserve multi-week crisis clustering. This script runs
a PAIRED stationary bootstrap (Politis-Romano 1994: geometric block lengths with
wrap-around indexing) on the carry and SPY excess-return streams, resampling the
same date indices for both series so every draw is a like-for-like comparison, and
puts confidence intervals on Calmar, dCalmar, and dmaxDD.

Caveat (also written into the JSON): bootstrap resampling scrambles the ordering of
regimes, so the maxDD of a resampled path is a dispersion measure of drawdown risk
under exchangeable-block assumptions, not a forecast of the next realized drawdown.

Run: /opt/anaconda3/envs/trading/bin/python analysis/drawdown_inference.py
Output: analysis/drawdown_inference_results.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
EQUITY_PATH = REPO_ROOT / "analysis" / "strategy_equity.parquet"
OUT_PATH = REPO_ROOT / "analysis" / "drawdown_inference_results.json"

ANN = 252
N_DRAWS = 5000
SEED = 7
MAIN_BLOCK = 90
SENSITIVITY_BLOCKS = (15, 180)
DD_THRESHOLD = -0.25
CHUNK = 500  # draws per vectorized chunk (memory guard)

CAVEAT = (
    "Bootstrap paths scramble regime ordering; resampled maxDD/Calmar draws measure "
    "the dispersion of drawdown risk under exchangeable-block resampling, not a "
    "forecast of the next realized drawdown."
)


def load_returns() -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Daily excess returns for the carry sleeve and SPY from the equity curves.

    Both equity columns start from a base of 1.0, so the first return is
    eq[0]/1.0 - 1 and subsequent returns are eq[t]/eq[t-1] - 1.
    """
    df = pd.read_parquet(EQUITY_PATH)
    dates = pd.DatetimeIndex(df["date"])
    out = []
    for col in ("carry", "spy_excess"):
        eq = df[col].to_numpy(float)
        r = eq / np.concatenate(([1.0], eq[:-1])) - 1.0
        out.append(r)
    return out[0], out[1], dates


def stationary_bootstrap_indices(
    n: int, n_draws: int, mean_block: float, rng: np.random.Generator
) -> np.ndarray:
    """(n_draws, n) index matrix from the Politis-Romano stationary bootstrap.

    Each position restarts a block with prob 1/mean_block at a uniform start;
    otherwise it continues the previous index, wrapping around mod n. Fully
    vectorized: block starts are located with a running maximum over positions.
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
    r_carry: np.ndarray, r_spy: np.ndarray, mean_block: float, n_draws: int, seed: int
) -> dict:
    n = len(r_carry)
    rng = np.random.default_rng([seed, int(mean_block)])
    acc: dict[str, list[np.ndarray]] = {}
    for lo in range(0, n_draws, CHUNK):
        idx = stationary_bootstrap_indices(n, min(CHUNK, n_draws - lo), mean_block, rng)
        mc = path_metrics(r_carry[idx])  # paired: same idx for both series
        ms = path_metrics(r_spy[idx])
        for k, v in mc.items():
            acc.setdefault(f"carry_{k}", []).append(v)
        for k, v in ms.items():
            acc.setdefault(f"spy_{k}", []).append(v)
    d = {k: np.concatenate(v) for k, v in acc.items()}
    d_calmar = d["carry_calmar"] - d["spy_calmar"]
    d_maxdd = d["carry_maxdd"] - d["spy_maxdd"]  # >0 means carry drawdown shallower
    d_sharpe = d["carry_sharpe"] - d["spy_sharpe"]
    return {
        "mean_block_days": mean_block,
        "n_draws": n_draws,
        "carry_calmar_ci95": ci(d["carry_calmar"]),
        "spy_calmar_ci95": ci(d["spy_calmar"]),
        "delta_calmar_ci95": ci(d_calmar),
        "delta_calmar_median": float(np.median(d_calmar)),
        "carry_maxdd_ci95": ci(d["carry_maxdd"]),
        "spy_maxdd_ci95": ci(d["spy_maxdd"]),
        "delta_maxdd_ci95": ci(d_maxdd),
        "delta_maxdd_median": float(np.median(d_maxdd)),
        "carry_sharpe_ci95": ci(d["carry_sharpe"]),
        "delta_sharpe_ci95": ci(d_sharpe),
        "p_carry_calmar_gt_spy": float((d_calmar > 0).mean()),
        "p_carry_maxdd_shallower": float((d_maxdd > 0).mean()),
        "p_carry_maxdd_worse_than_-25pct": float((d["carry_maxdd"] < DD_THRESHOLD).mean()),
    }


def main() -> None:
    r_carry, r_spy, dates = load_returns()
    point = {
        "carry": sample_metrics(r_carry),
        "spy_excess": sample_metrics(r_spy),
    }
    point["delta_calmar"] = point["carry"]["calmar"] - point["spy_excess"]["calmar"]
    point["delta_maxdd"] = point["carry"]["maxdd"] - point["spy_excess"]["maxdd"]

    results = {
        "method": (
            "Paired Politis-Romano stationary bootstrap (geometric block lengths, "
            "wrap-around); the same resampled date indices are applied to both the "
            "carry and SPY excess-return streams, so every draw is a paired comparison."
        ),
        "caveat": CAVEAT,
        "data": {
            "source": "analysis/strategy_equity.parquet",
            "columns": ["carry", "spy_excess"],
            "n_days": int(len(r_carry)),
            "start": str(dates[0].date()),
            "end": str(dates[-1].date()),
        },
        "seed": SEED,
        "point_estimates": point,
        "main": run_bootstrap(r_carry, r_spy, MAIN_BLOCK, N_DRAWS, SEED),
        "sensitivity": {
            str(b): run_bootstrap(r_carry, r_spy, b, N_DRAWS, SEED)
            for b in SENSITIVITY_BLOCKS
        },
    }

    OUT_PATH.write_text(json.dumps(results, indent=2) + "\n")

    # ---- compact printout ----
    pt_c, pt_s = point["carry"], point["spy_excess"]
    print(f"Data: {results['data']['start']} .. {results['data']['end']}  "
          f"({results['data']['n_days']} days)  draws={N_DRAWS}  seed={SEED}")
    print(f"Point: carry Calmar {pt_c['calmar']:.2f}  maxDD {pt_c['maxdd']:.1%}   "
          f"SPY Calmar {pt_s['calmar']:.2f}  maxDD {pt_s['maxdd']:.1%}")
    print()
    hdr = (f"{'block':>6} {'carry Calmar 95% CI':>22} {'dCalmar 95% CI':>20} "
           f"{'dmaxDD 95% CI':>20} {'P(dCal>0)':>10} {'P(shallower)':>13} {'P(DD<-25%)':>11}")
    print(hdr)
    print("-" * len(hdr))
    for label, res in [("90", results["main"]),
                       ("15", results["sensitivity"]["15"]),
                       ("180", results["sensitivity"]["180"])]:
        cc = res["carry_calmar_ci95"]
        dc = res["delta_calmar_ci95"]
        dm = res["delta_maxdd_ci95"]
        print(f"{label:>6} {f'[{cc[0]:.2f}, {cc[1]:.2f}]':>22} "
              f"{f'[{dc[0]:.2f}, {dc[1]:.2f}]':>20} "
              f"{f'[{dm[0]:+.1%}, {dm[1]:+.1%}]':>20} "
              f"{res['p_carry_calmar_gt_spy']:>10.3f} "
              f"{res['p_carry_maxdd_shallower']:>13.3f} "
              f"{res['p_carry_maxdd_worse_than_-25pct']:>11.3f}")
    print()
    print(f"Caveat: {CAVEAT}")
    print(f"Wrote {OUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
