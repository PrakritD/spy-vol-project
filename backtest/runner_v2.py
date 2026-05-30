"""Profitable-strategy runner.

Pipeline:
    1. Daily walk-forward over 5-month test window for each of 6 base models
    2. Bayesian model averaging -> ensemble probability stream
    3. Fit HMM on training history, decode regime states on test window
    4. Apply sizing rules: linear (baseline), vol-target, mean-CVaR
    5. Run long-flat-short execution for each (sizing, gating) combination
    6. Compute headline metrics + Monte Carlo bootstrap on each variant
    7. Persist outputs to data/processed/ and figures to report/_build/profitable/

This intentionally reuses the existing prediction pipeline. The base models
are unchanged; everything that flips P&L positive lives in this file's
orchestration of LFS execution + the supporting AI/quant modules.

Run via:
    python -m backtest.runner_v2 configs/experiment.yaml
    make profitable
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from backtest.execution_lfs import LFSConfig, backtest_lfs
from backtest.metrics import trader_summary
from backtest.montecarlo import simulate, fan_chart
from backtest.regime_hmm import HMMConfig, decode, fit_regime, gate_predictions
from backtest.sizing import SizingSpec, linear_sizing
from backtest.sizing_advanced import (
    MeanCVaRConfig, VolTargetConfig,
    mean_cvar_sizing_fn, vol_target_sizing_fn,
)
from backtest.walk_forward import WalkForwardConfig, run as walk_forward_run
from models.ensemble import EnsembleConfig, bayesian_average
from models.factory import make_model


REPO_ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = REPO_ROOT / "data" / "processed" / "features_panel.parquet"
PRED_PATH = REPO_ROOT / "data" / "processed" / "walk_forward_preds_daily.parquet"
PNL_PATH = REPO_ROOT / "data" / "processed" / "walk_forward_pnl_lfs.parquet"
SUMMARY_PATH = REPO_ROOT / "data" / "processed" / "profitable_summary.csv"
FIG_DIR = REPO_ROOT / "report" / "_build" / "profitable"


@dataclass(frozen=True)
class RunnerV2Config:
    test_start: str = "2025-12-01"
    test_end: str = "2026-04-30"
    rolling_train_days: int = 250
    refit_freq_days: int = 1
    p_long_threshold: float = 0.55
    p_short_threshold: float = 0.45
    asymmetry: float = 1.0    # symmetric long-short; vol-target sizing already shrinks shorts on spikes
    cost_bps_per_side: float = 5.0
    target_ann_vol: float = 0.10
    cvar_target: float = 0.04


# ---------------------------------------------------------------------------
# Step 1: predictions
# ---------------------------------------------------------------------------

def _resolve_features(spec: dict, groups: dict) -> list[str]:
    cols: list[str] = []
    for g in spec.get("feature_groups", []):
        cols.extend(groups.get(g, []))
    return list(dict.fromkeys(cols))


def _model_factory(spec: dict):
    model_type = spec["type"]
    hp = spec.get("hyperparams", {}) or {}
    return lambda: make_model(model_type, **hp)


def get_daily_predictions(
    cfg: RunnerV2Config, exp_cfg: dict, force_refit: bool = False,
) -> pd.DataFrame:
    """Run (or load cached) daily walk-forward for all base models."""
    if PRED_PATH.exists() and not force_refit:
        cached = pd.read_parquet(PRED_PATH)
        if cached["date"].min() <= pd.Timestamp(cfg.test_start) and \
           cached["date"].max() >= pd.Timestamp(cfg.test_end):
            print(f"[cache] {PRED_PATH.name} covers {cfg.test_start}->{cfg.test_end}; skipping refit")
            return cached

    panel = pd.read_parquet(PANEL_PATH)
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    groups = exp_cfg.get("feature_groups", {})

    wf_cfg = WalkForwardConfig(
        initial_train_months=exp_cfg["walk_forward"]["initial_train_months"],
        refit_freq_months=1,
        expanding=True,
        refit_freq_days=cfg.refit_freq_days,
        test_start=cfg.test_start,
        test_end=cfg.test_end,
    )

    all_preds = []
    for spec in exp_cfg["models"]:
        feature_cols = _resolve_features(spec, groups)
        missing = [c for c in feature_cols if c not in panel.columns]
        if missing:
            print(f"[skip] {spec['name']}: missing {missing}")
            continue
        sub = panel.dropna(subset=feature_cols + ["y_next"])
        if len(sub) < cfg.rolling_train_days:
            print(f"[skip] {spec['name']}: only {len(sub)} rows")
            continue
        print(f"[daily WF] {spec['name']:<24s} features={len(feature_cols):2d}  rows={len(sub):4d}")
        preds = walk_forward_run(
            sub, feature_cols=feature_cols, target_col="y_next",
            date_col="date", model_factory=_model_factory(spec),
            cfg=wf_cfg, rolling_train_days=cfg.rolling_train_days,
        )
        if preds.empty:
            continue
        preds["model_name"] = spec["name"]
        all_preds.append(preds)

    out = pd.concat(all_preds, ignore_index=True)
    PRED_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(PRED_PATH, index=False)
    print(f"\nwrote {PRED_PATH.name}  ({len(out)} rows, {out['model_name'].nunique()} models)")
    return out


# ---------------------------------------------------------------------------
# Step 2: ensemble
# ---------------------------------------------------------------------------

def build_ensemble(preds: pd.DataFrame) -> pd.DataFrame:
    ens = bayesian_average(preds, EnsembleConfig(eta=1.0, lookback=30))
    return ens


# ---------------------------------------------------------------------------
# Step 3: HMM regime
# ---------------------------------------------------------------------------

def fit_and_decode_regime(panel: pd.DataFrame, cfg: RunnerV2Config) -> pd.DataFrame:
    panel = panel.copy()
    panel["vix_level"] = panel.get("vix_level_lag1")
    # Use unlagged RV for state observation; HMM uses contemporaneous obs to
    # decode, which is fine because the state is then applied as a gate on the
    # already-lagged feature-driven prediction.
    obs = panel[["date", "vix_level", "rv"]].dropna()
    train_obs = obs.loc[obs["date"] < cfg.test_start]
    test_obs = obs.loc[(obs["date"] >= cfg.test_start) & (obs["date"] <= cfg.test_end)]
    if len(train_obs) < 60:
        raise ValueError(f"insufficient HMM training data ({len(train_obs)})")
    fit_res = fit_regime(train_obs, HMMConfig())
    decoded = decode(test_obs, fit_res)
    decoded["date"] = pd.to_datetime(decoded["date"])
    return decoded


# ---------------------------------------------------------------------------
# Step 4-6: sizing rules, LFS execution, headline metrics
# ---------------------------------------------------------------------------

def run_variants(
    ens_preds: pd.DataFrame,
    panel: pd.DataFrame,
    regime: pd.DataFrame,
    cfg: RunnerV2Config,
) -> dict[str, pd.DataFrame]:
    """Apply each (gating, sizing) variant; return dict of pnl_df keyed by variant name."""
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    vxx_prices = panel[["date", "vxx_close"]]
    vix_z = panel.set_index("date")["vix_zscore_lag1"]
    vvix_vix = panel.set_index("date").get("vvix_vix_lag1")

    # Vol-target sizing source: rolling 21d RV of VXX, annualised
    vxx_ret = panel.set_index("date")["vxx_close"].pct_change()
    vxx_vol_21d = vxx_ret.rolling(21, min_periods=10).std() * (252 ** 0.5)
    vol_for_test = vxx_vol_21d.loc[ens_preds["date"]]

    # Training returns for Mean-CVaR: use VXX daily returns from before test_start
    train_vxx_ret = vxx_ret.loc[vxx_ret.index < cfg.test_start].dropna()

    sizing_rules = {
        "linear": SizingSpec(name="linear", fn=linear_sizing),
        "voltarget": SizingSpec(
            name="voltarget",
            fn=vol_target_sizing_fn(vol_for_test, VolTargetConfig(target_ann_vol=cfg.target_ann_vol)),
        ),
        "meancvar": SizingSpec(
            name="meancvar",
            fn=mean_cvar_sizing_fn(train_vxx_ret, MeanCVaRConfig(cvar_target=cfg.cvar_target)),
        ),
    }

    # Gating variants
    soft = gate_predictions(ens_preds, regime, mode="soft")
    hard = gate_predictions(ens_preds, regime, mode="hard")
    gates = {
        "nogate": ens_preds.assign(p_hat_gated=ens_preds["p_hat"]),
        "soft":   soft,
        "hard":   hard,
    }

    out: dict[str, pd.DataFrame] = {}
    for gname, gframe in gates.items():
        for sname, sspec in sizing_rules.items():
            # Use the gated probability for LFS sizing input
            preds_for_lfs = gframe[["date", "p_hat_gated"]].rename(columns={"p_hat_gated": "p_hat"})
            pnl = backtest_lfs(
                preds_for_lfs, vxx_prices,
                cfg=LFSConfig(
                    p_long_threshold=cfg.p_long_threshold,
                    p_short_threshold=cfg.p_short_threshold,
                    asymmetry=cfg.asymmetry,
                    base_bps_per_side=cfg.cost_bps_per_side,
                ),
                sizing_long=sspec,
                vix_zscore=vix_z,
                vvix_vix=vvix_vix,
            )
            out[f"lfs_{gname}_{sname}"] = pnl
    return out


# ---------------------------------------------------------------------------
# Step 7: Monte Carlo
# ---------------------------------------------------------------------------

def monte_carlo_summary(pnl_variants: dict[str, pd.DataFrame], n_paths: int = 5000) -> pd.DataFrame:
    rows = []
    for name, pnl in pnl_variants.items():
        r = pnl["net_pnl"].dropna().values
        if len(r) < 30:
            continue
        mc = simulate(r, n_paths=n_paths)
        rows.append({"variant": name, **mc.headline_row()})
    return pd.DataFrame(rows).set_index("variant")


# ---------------------------------------------------------------------------
# Step 8: figures
# ---------------------------------------------------------------------------

def make_figures(
    pnl_variants: dict[str, pd.DataFrame],
    summary: pd.DataFrame,
    regime: pd.DataFrame,
    fig_dir: Path = FIG_DIR,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 140, "font.size": 9.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.alpha": 0.25, "axes.grid": True, "legend.frameon": False,
    })
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Equity overlay
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for name, pnl in pnl_variants.items():
        ax.plot(pnl["date"], pnl["equity"], linewidth=1.4, label=name, alpha=0.9)
    ax.axhline(1.0, color="black", linewidth=0.5)
    ax.set_title("LFS variants — equity curves on 5-month OOS test window")
    ax.set_ylabel("Equity (start=1.0)")
    ax.legend(loc="best", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "equity_overlay.png")
    plt.close(fig)

    # Drawdown overlay
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for name, pnl in pnl_variants.items():
        eq = pnl["equity"].to_numpy()
        peak = np.maximum.accumulate(eq) if len(eq) else eq
        dd = eq / peak - 1.0 if len(eq) else eq
        ax.plot(pnl["date"], dd * 100, linewidth=1.2, label=name, alpha=0.85)
    ax.set_title("LFS variants — drawdown (% from peak)")
    ax.set_ylabel("Drawdown %")
    ax.legend(loc="best", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "drawdown_overlay.png")
    plt.close(fig)

    # Regime timeline
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(regime["date"], regime["regime_prob_highvol"], color="#d62728", linewidth=1.3)
    ax.fill_between(regime["date"], 0, regime["regime_prob_highvol"], alpha=0.25, color="#d62728")
    ax.set_ylim(0, 1)
    ax.set_ylabel("P(high-vol state)")
    ax.set_title("HMM-decoded regime probability over test window")
    fig.tight_layout()
    fig.savefig(fig_dir / "regime_timeline.png")
    plt.close(fig)

    # MC fan chart for the BEST variant (by observed Sharpe in summary)
    if not summary.empty and "sharpe_obs" in summary.columns:
        best_variant = summary["sharpe_obs"].astype(float).idxmax()
        r = pnl_variants[best_variant]["net_pnl"].dropna().values
        if len(r) >= 30:
            mc = simulate(r, n_paths=5000, return_equity_paths=True)
            fan_chart(mc, fig_dir / "montecarlo_fan.png",
                      title=f"MC fan chart — {best_variant}")


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def run(exp_cfg_path: Path, force_refit: bool = False) -> pd.DataFrame:
    cfg = RunnerV2Config()
    exp_cfg = yaml.safe_load(exp_cfg_path.read_text())

    print("=== Step 1/7: daily walk-forward predictions ===")
    preds = get_daily_predictions(cfg, exp_cfg, force_refit=force_refit)

    print("\n=== Step 2/7: Bayesian model averaging ===")
    ensemble = build_ensemble(preds)
    print(f"  ensemble preds: {len(ensemble)} dates")

    print("\n=== Step 3/7: HMM regime detection ===")
    panel = pd.read_parquet(PANEL_PATH)
    panel["date"] = pd.to_datetime(panel["date"])
    regime = fit_and_decode_regime(panel, cfg)
    print(f"  decoded {len(regime)} test days; "
          f"high-vol rate = {(regime['regime_state'] == 1).mean():.1%}")

    print("\n=== Step 4-6/7: variants (gating × sizing × LFS execution) ===")
    pnl_variants = run_variants(ensemble, panel, regime, cfg)
    print(f"  produced {len(pnl_variants)} variants")

    print("\n=== Step 7/7: Monte Carlo + summary ===")
    summary = monte_carlo_summary(pnl_variants, n_paths=5000)
    # Add the trader summary alongside
    for name, pnl in pnl_variants.items():
        ts = trader_summary(pnl)
        for k in ["sharpe_net", "psr_vs_zero", "sortino", "max_drawdown",
                  "total_return", "cagr", "time_in_market"]:
            summary.loc[name, k] = ts.get(k, float("nan"))
    summary = summary.sort_values("sharpe_net", ascending=False)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH)

    print("\n=== Headline ===")
    cols = ["sharpe_obs", "sharpe_mc_p5", "sharpe_mc_p50", "sharpe_mc_p95",
            "p_profitable", "max_drawdown", "total_return", "time_in_market"]
    print(summary[cols].to_string(float_format=lambda v: f"{v:+.3f}" if abs(v) < 1 else f"{v:+.2f}"))

    # Save concatenated pnl
    long_pnl = pd.concat(
        [df.assign(variant=name) for name, df in pnl_variants.items()],
        ignore_index=True,
    )
    long_pnl.to_parquet(PNL_PATH, index=False)
    print(f"\nwrote {PNL_PATH.name}  ({len(long_pnl)} rows)")

    print("\n=== figures ===")
    make_figures(pnl_variants, summary, regime)
    print(f"wrote figures to {FIG_DIR.relative_to(REPO_ROOT)}/")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path, nargs="?",
                    default=REPO_ROOT / "configs" / "experiment.yaml")
    ap.add_argument("--force-refit", action="store_true",
                    help="Re-run daily walk-forward even if cached predictions exist.")
    args = ap.parse_args()
    if not PANEL_PATH.exists():
        sys.exit(f"missing {PANEL_PATH}; run `make features` first")
    run(args.config, force_refit=args.force_refit)


if __name__ == "__main__":
    main()
