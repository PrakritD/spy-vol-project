"""Cost × threshold sensitivity sweep.

For each model in `data/processed/walk_forward_preds.parquet`, re-run
`backtest.execution.backtest()` over a grid of (cost_bps, threshold) and
record the resulting Sharpe + max-drawdown. No model re-fitting — just
re-applies the sizing rule + cost model.

Output:
    report/_build/sensitivity_<model>.png      Sharpe heatmap (cost × threshold)
    report/_build/sensitivity_summary.csv      tall frame: (model, cost, threshold, sharpe, max_dd)

Demonstrates strategy fragility / robustness to execution-cost assumptions —
the textbook quant-shop "how does the Sharpe survive at 20 bps instead of 5?"
question, answered directly.

Run via:
    python -m backtest.sensitivity
    make sensitivity
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtest.execution import ExecConfig, backtest
from backtest.metrics import max_drawdown, sharpe
from backtest.sizing import SizingSpec


REPO_ROOT = Path(__file__).resolve().parents[1]
PRED_PATH = REPO_ROOT / "data" / "processed" / "walk_forward_preds.parquet"
PANEL_PATH = REPO_ROOT / "data" / "processed" / "features_panel.parquet"
BUILD = REPO_ROOT / "report" / "_build"
CSV_OUT = BUILD / "sensitivity_summary.csv"

COSTS_BPS = [5, 10, 20, 35, 50]
THRESHOLDS = [0.45, 0.50, 0.55, 0.60, 0.65]


def _setup_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 140,
        "font.size": 9.5, "axes.titlesize": 11, "axes.labelsize": 10,
        "axes.grid": False, "legend.frameon": False,
    })


def _threshold_sizing_fn(threshold: float):
    """Long-flat: size=1 when p_hat >= threshold, else size=0."""
    def fn(p_hat):
        return (np.asarray(p_hat) >= threshold).astype(float)
    return fn


def _run_one(preds: pd.DataFrame, vxx_prices: pd.DataFrame,
             vix_z: pd.Series, cost_bps: float, threshold: float) -> tuple[float, float]:
    cfg = ExecConfig(base_bps_per_side=cost_bps, extra_bps_high_vol=0.0)
    sizing = SizingSpec(name=f"thresh_{threshold}", fn=_threshold_sizing_fn(threshold))
    pnl = backtest(preds, vxx_prices, cfg=cfg, sizing=sizing, vix_zscore=vix_z)
    sr = sharpe(pnl["net_pnl"])
    dd = max_drawdown(pnl["equity"])
    return sr, dd


def _plot_heatmap(name: str, sharpe_grid: np.ndarray, dd_grid: np.ndarray) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    vmax = max(abs(np.nanmin(sharpe_grid)), abs(np.nanmax(sharpe_grid)), 0.5)
    ax = axes[0]
    im = ax.imshow(sharpe_grid, cmap="RdYlGn", vmin=-vmax, vmax=vmax,
                    aspect="auto", origin="lower")
    ax.set_xticks(range(len(COSTS_BPS)))
    ax.set_xticklabels([f"{c}" for c in COSTS_BPS])
    ax.set_xlabel("Cost (bps per side)")
    ax.set_yticks(range(len(THRESHOLDS)))
    ax.set_yticklabels([f"{t:.2f}" for t in THRESHOLDS])
    ax.set_ylabel("Long-entry threshold")
    ax.set_title(f"{name}: Sharpe heatmap (long-flat threshold sizing)")
    for i, t in enumerate(THRESHOLDS):
        for j, c in enumerate(COSTS_BPS):
            v = sharpe_grid[i, j]
            ax.text(j, i, f"{v:+.2f}" if np.isfinite(v) else "nan",
                    ha="center", va="center", fontsize=8,
                    color="black" if abs(v) < 0.5 * vmax else "white")
    fig.colorbar(im, ax=ax, label="Sharpe (annualised)", shrink=0.75)

    vmin = min(np.nanmin(dd_grid), -0.05)
    ax = axes[1]
    im2 = ax.imshow(dd_grid * 100, cmap="Reds_r", vmin=vmin * 100, vmax=0,
                     aspect="auto", origin="lower")
    ax.set_xticks(range(len(COSTS_BPS)))
    ax.set_xticklabels([f"{c}" for c in COSTS_BPS])
    ax.set_xlabel("Cost (bps per side)")
    ax.set_yticks(range(len(THRESHOLDS)))
    ax.set_yticklabels([f"{t:.2f}" for t in THRESHOLDS])
    ax.set_ylabel("Long-entry threshold")
    ax.set_title(f"{name}: Max drawdown (%)")
    for i, t in enumerate(THRESHOLDS):
        for j, c in enumerate(COSTS_BPS):
            v = dd_grid[i, j] * 100
            ax.text(j, i, f"{v:.1f}" if np.isfinite(v) else "nan",
                    ha="center", va="center", fontsize=8, color="white")
    fig.colorbar(im2, ax=ax, label="Max DD (%)", shrink=0.75)

    fig.tight_layout()
    out = BUILD / f"sensitivity_{name}.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def run() -> pd.DataFrame:
    _setup_style()
    BUILD.mkdir(parents=True, exist_ok=True)
    if not PRED_PATH.exists():
        raise FileNotFoundError(
            f"missing {PRED_PATH} — run `make backtest` first to produce predictions"
        )
    panel = pd.read_parquet(PANEL_PATH)
    panel["date"] = pd.to_datetime(panel["date"])
    preds_all = pd.read_parquet(PRED_PATH)
    preds_all["date"] = pd.to_datetime(preds_all["date"])

    vix_z = panel.set_index("date")["vix_zscore_lag1"]
    vxx_prices = panel[["date", "vxx_close"]]

    long_rows: list[dict] = []
    for name in preds_all["model_name"].unique():
        preds_m = preds_all[preds_all["model_name"] == name][["date", "p_hat"]]
        print(f"[sweep] {name:<24s} {len(preds_m)} OOS predictions")
        sharpe_grid = np.full((len(THRESHOLDS), len(COSTS_BPS)), np.nan)
        dd_grid = np.full((len(THRESHOLDS), len(COSTS_BPS)), np.nan)
        for i, t in enumerate(THRESHOLDS):
            for j, c in enumerate(COSTS_BPS):
                sr, dd = _run_one(preds_m, vxx_prices, vix_z, c, t)
                sharpe_grid[i, j] = sr
                dd_grid[i, j] = dd
                long_rows.append({"model": name, "cost_bps": c,
                                   "threshold": t, "sharpe": sr, "max_dd": dd})
        path = _plot_heatmap(name, sharpe_grid, dd_grid)
        print(f"        wrote {path.relative_to(REPO_ROOT)}")

    df = pd.DataFrame(long_rows)
    df.to_csv(CSV_OUT, index=False)
    print(f"\nwrote {CSV_OUT.relative_to(REPO_ROOT)} ({len(df)} rows)")
    return df


def main():
    run()


if __name__ == "__main__":
    main()
