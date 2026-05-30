"""In-sample threshold + asymmetry sweep for LFS execution.

Loads cached daily walk-forward predictions, ensembles them, then sweeps the
(p_long_threshold, p_short_threshold, asymmetry) grid. For each setting,
re-runs LFS execution (cheap — no model refit) and computes Sharpe on a
50/50 in-sample/out-of-sample split of the test window. Picks the in-sample
winner and reports its OOS performance.

The 50/50 split is the discipline against picking thresholds from final
results. We tune on the first half, validate on the second.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.execution_lfs import LFSConfig, backtest_lfs
from backtest.metrics import sharpe, sortino, max_drawdown
from backtest.sizing import SizingSpec
from backtest.sizing_advanced import VolTargetConfig, vol_target_sizing_fn
from models.ensemble import EnsembleConfig, bayesian_average


REPO_ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = REPO_ROOT / "data" / "processed" / "features_panel.parquet"


def split_inout(ens: pd.DataFrame, frac_in: float = 0.5):
    ens = ens.sort_values("date").reset_index(drop=True)
    cut = int(len(ens) * frac_in)
    return ens.iloc[:cut], ens.iloc[cut:]


def sweep(
    ens: pd.DataFrame,
    panel: pd.DataFrame,
    p_longs: list[float] = (0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.65),
    p_shorts: list[float] = (0.50, 0.48, 0.45, 0.42, 0.40, 0.38, 0.35),
    asymmetries: list[float] = (0.3, 0.5, 0.7, 1.0),
    cost_bps: float = 5.0,
):
    """Returns long-format frame: (p_long, p_short, asymmetry, sharpe_in, sortino_in, sharpe_out, ...)."""
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    vxx_prices = panel[["date", "vxx_close"]]
    vix_z = panel.set_index("date")["vix_zscore_lag1"]

    # Vol-target sizing (the variant that won at h=5)
    vxx_ret = panel.set_index("date")["vxx_close"].pct_change()
    vxx_vol_21d = vxx_ret.rolling(21, min_periods=10).std() * np.sqrt(252)
    vol_for_dates = vxx_vol_21d.loc[ens["date"]]

    sizing = SizingSpec(
        name="voltarget",
        fn=vol_target_sizing_fn(vol_for_dates, VolTargetConfig(target_ann_vol=0.10)),
    )

    ens_in, ens_out = split_inout(ens)

    rows = []
    for pl in p_longs:
        for ps in p_shorts:
            if ps >= pl:
                continue
            for a in asymmetries:
                cfg = LFSConfig(p_long_threshold=pl, p_short_threshold=ps,
                                 asymmetry=a, base_bps_per_side=cost_bps)
                pnl_in = backtest_lfs(
                    ens_in[["date", "p_hat"]].rename(columns={"p_hat": "p_hat"}),
                    vxx_prices, cfg=cfg, sizing_long=sizing, vix_zscore=vix_z,
                )
                pnl_out = backtest_lfs(
                    ens_out[["date", "p_hat"]].rename(columns={"p_hat": "p_hat"}),
                    vxx_prices, cfg=cfg, sizing_long=sizing, vix_zscore=vix_z,
                )
                rows.append({
                    "p_long": pl, "p_short": ps, "asymmetry": a,
                    "n_in": len(pnl_in), "n_out": len(pnl_out),
                    "sharpe_in": sharpe(pnl_in["net_pnl"]),
                    "sortino_in": sortino(pnl_in["net_pnl"]),
                    "ret_in": float(pnl_in["equity"].iloc[-1] - 1),
                    "sharpe_out": sharpe(pnl_out["net_pnl"]),
                    "sortino_out": sortino(pnl_out["net_pnl"]),
                    "ret_out": float(pnl_out["equity"].iloc[-1] - 1),
                    "max_dd_out": max_drawdown(pnl_out["equity"]),
                    "tim_out": float((pnl_out["size"].abs() > 1e-9).mean()),
                })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=Path, default=REPO_ROOT / "data" / "processed" / "walk_forward_preds_daily.parquet")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "processed" / "threshold_sweep.csv")
    args = ap.parse_args()

    panel = pd.read_parquet(PANEL_PATH)
    preds = pd.read_parquet(args.preds)
    ens = bayesian_average(preds, EnsembleConfig(eta=1.0, lookback=30))
    ens["date"] = pd.to_datetime(ens["date"]).dt.normalize()

    print(f"Sweep on {len(ens)} ensemble preds; split 50/50 in-sample/out-of-sample")
    res = sweep(ens, panel)
    res = res.sort_values("sharpe_in", ascending=False)
    res.to_csv(args.out, index=False)
    print(f"\nwrote {args.out.name}  ({len(res)} rows)")

    print("\n=== Top 5 by in-sample Sharpe ===")
    print(res.head(5).to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

    # Honest readout: best by IN-sample, see how it does OUT-of-sample
    best = res.iloc[0]
    print("\n=== Best by IS Sharpe (its OOS performance is the validated claim) ===")
    print(f"  thresholds: p_long={best['p_long']}, p_short={best['p_short']}, "
          f"asymmetry={best['asymmetry']}")
    print(f"  IS  Sharpe={best['sharpe_in']:+.2f}  ret={best['ret_in']:+.2%}")
    print(f"  OOS Sharpe={best['sharpe_out']:+.2f}  ret={best['ret_out']:+.2%}  "
          f"MaxDD={best['max_dd_out']:.1%}  TIM={best['tim_out']:.0%}")


if __name__ == "__main__":
    main()
