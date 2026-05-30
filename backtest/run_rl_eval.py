"""Train the PPO sizing policy and append its row to the LFS comparison table.

Reads:
    data/processed/features_panel.parquet
    data/processed/walk_forward_preds_daily.parquet  (from runner_v2)
    data/processed/profitable_summary.csv             (from runner_v2)

Writes:
    data/processed/profitable_summary_with_rl.csv      (appended row)
    data/processed/rl_pnl.parquet                      (daily P&L of the RL policy)
    report/_build/profitable/equity_overlay_rl.png     (re-rendered with RL row)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from backtest.metrics import trader_summary
from backtest.montecarlo import simulate
from models.ensemble import EnsembleConfig, bayesian_average
from models.rl_sizing import run_rl_pipeline


REPO_ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = REPO_ROOT / "data" / "processed" / "features_panel.parquet"
PRED_DAILY = REPO_ROOT / "data" / "processed" / "walk_forward_preds_daily.parquet"
PRED_MONTHLY = REPO_ROOT / "data" / "processed" / "walk_forward_preds.parquet"
SUMMARY_IN = REPO_ROOT / "data" / "processed" / "profitable_summary.csv"
SUMMARY_OUT = REPO_ROOT / "data" / "processed" / "profitable_summary_with_rl.csv"
PNL_OUT = REPO_ROOT / "data" / "processed" / "rl_pnl.parquet"
FIG_DIR = REPO_ROOT / "report" / "_build" / "profitable"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=30_000)
    ap.add_argument("--test-start", default="2025-12-01")
    ap.add_argument("--test-end", default="2026-04-30")
    args = ap.parse_args()

    panel = pd.read_parquet(PANEL_PATH)
    # For RL we need predictions covering BOTH the in-sample training window
    # (~340 days pre-test) AND the test window. Phase-1 monthly-refit
    # predictions cover the full OOS slice, so we use those for RL training.
    # Daily-refit predictions (Phase 2) only cover the 5mo test window.
    if PRED_MONTHLY.exists():
        preds = pd.read_parquet(PRED_MONTHLY)
        print(f"loaded monthly-refit preds ({len(preds)} rows, "
              f"{preds['model_name'].nunique()} models)")
    else:
        preds = pd.read_parquet(PRED_DAILY)
        print(f"loaded daily-refit preds (only test-window coverage) "
              f"({len(preds)} rows, {preds['model_name'].nunique()} models)")
    ensemble = bayesian_average(preds, EnsembleConfig(eta=1.0, lookback=30))
    print(f"ensemble preds: {len(ensemble)} dates from "
          f"{ensemble['date'].min().date()} to {ensemble['date'].max().date()}")

    rl = run_rl_pipeline(
        panel, ensemble,
        test_start=args.test_start, test_end=args.test_end,
        total_timesteps=args.timesteps,
    )
    rl_pnl = rl["pnl"].copy()
    print(f"  RL test produced {len(rl_pnl)} daily P&L rows")
    print(f"  total return : {(rl_pnl['equity'].iloc[-1] - 1) * 100:+.2f}%")
    print(f"  time in mkt  : {(rl_pnl['size'].abs() > 1e-9).mean():.1%}")
    print(f"  net long days: {(rl_pnl['size'] > 0).mean():.1%}")
    print(f"  net short    : {(rl_pnl['size'] < 0).mean():.1%}")

    rl_pnl.to_parquet(PNL_OUT, index=False)

    # Metrics + Monte Carlo
    trd = trader_summary(rl_pnl)
    r = rl_pnl["net_pnl"].dropna().values
    mc = simulate(r, n_paths=5000) if len(r) >= 30 else None

    row = {
        "variant": "rl_ppo",
        "sharpe_obs": trd["sharpe_net"],
        "sharpe_mc_p5": mc.sharpe_p5 if mc else float("nan"),
        "sharpe_mc_p50": mc.sharpe_p50 if mc else float("nan"),
        "sharpe_mc_p95": mc.sharpe_p95 if mc else float("nan"),
        "p_profitable": mc.p_profitable if mc else float("nan"),
        "max_drawdown": trd["max_drawdown"],
        "max_dd_obs": trd["max_drawdown"],
        "max_dd_mc_p95": mc.max_dd_p95 if mc else float("nan"),
        "total_return": trd["total_return"],
        "cagr": trd["cagr"],
        "time_in_market": trd["time_in_market"],
        "sharpe_net": trd["sharpe_net"],
        "psr_vs_zero": trd["psr_vs_zero"],
        "sortino": trd["sortino"],
        "n_paths": mc.n_paths if mc else 0,
    }
    print("\n=== RL headline ===")
    for k in ["sharpe_obs", "sharpe_mc_p5", "sharpe_mc_p50", "sharpe_mc_p95",
              "p_profitable", "max_drawdown", "total_return", "time_in_market"]:
        print(f"  {k:<20s} {row[k]}")

    # Append to existing summary
    summary = pd.read_csv(SUMMARY_IN).set_index("variant")
    row_df = pd.DataFrame([row]).set_index("variant")
    summary_out = pd.concat([summary, row_df]).sort_values("sharpe_obs", ascending=False)
    summary_out.to_csv(SUMMARY_OUT)
    print(f"\nwrote {SUMMARY_OUT.name}")

    # Refresh the equity overlay with RL line included
    try:
        import matplotlib.pyplot as plt
        long_pnl_path = REPO_ROOT / "data" / "processed" / "walk_forward_pnl_lfs.parquet"
        if long_pnl_path.exists():
            pnl_long = pd.read_parquet(long_pnl_path)
            plt.rcParams.update({
                "figure.dpi": 110, "savefig.dpi": 140, "font.size": 9.5,
                "axes.spines.top": False, "axes.spines.right": False,
                "grid.alpha": 0.25, "axes.grid": True, "legend.frameon": False,
            })
            fig, ax = plt.subplots(figsize=(11, 5.5))
            for v, grp in pnl_long.groupby("variant"):
                ax.plot(grp["date"], grp["equity"], linewidth=1.0, alpha=0.6, label=v)
            ax.plot(rl_pnl["date"], rl_pnl["equity"], color="black", linewidth=2.2, label="rl_ppo")
            ax.axhline(1.0, color="black", linewidth=0.4, alpha=0.5)
            ax.set_title("LFS variants vs RL PPO — equity curves (5-month OOS)")
            ax.set_ylabel("Equity")
            ax.legend(loc="best", ncol=2, fontsize=8)
            fig.tight_layout()
            out = FIG_DIR / "equity_overlay_rl.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out)
            plt.close(fig)
            print(f"wrote {out.relative_to(REPO_ROOT)}")
    except Exception as e:
        print(f"figure render failed: {e}")


if __name__ == "__main__":
    main()
