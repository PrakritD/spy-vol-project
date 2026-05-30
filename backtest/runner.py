"""End-to-end backtest runner.

Reads `configs/experiment.yaml` and `data/processed/features_panel.parquet`.
For each model spec:
    1. Resolves `feature_groups:` -> concrete column list.
    2. Runs the walk-forward harness.
    3. Runs `backtest/execution.backtest()` to convert p_hat -> P&L.
    4. Computes the full trader-grade metrics block (Sharpe + CI, PSR,
       Sortino, CAGR, max DD, VaR/CVaR, skew/kurt, etc.).

Also reports two benchmarks for sanity:
    - VXX buy-and-hold (the do-nothing-long baseline)
    - Cash (zero return) — the do-nothing baseline a vol trader must beat

Outputs:
    data/processed/walk_forward_preds.parquet   one row per (date, model)
    data/processed/walk_forward_pnl.parquet     one row per (date, model)
    data/processed/backtest_summary.csv         metrics table, one row per model
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

from backtest.execution import ExecConfig, backtest
from backtest.metrics import (
    classification_metrics,
    trader_summary,
)
from backtest.walk_forward import WalkForwardConfig, run as walk_forward_run
from models.factory import make_model


REPO_ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = REPO_ROOT / "data" / "processed" / "features_panel.parquet"
PRED_PATH = REPO_ROOT / "data" / "processed" / "walk_forward_preds.parquet"
PNL_PATH = REPO_ROOT / "data" / "processed" / "walk_forward_pnl.parquet"
SUMMARY_PATH = REPO_ROOT / "data" / "processed" / "backtest_summary.csv"


def _resolve_feature_cols(spec: dict, groups: dict) -> list[str]:
    cols: list[str] = []
    for g in spec.get("feature_groups", []):
        if g not in groups:
            raise KeyError(f"feature_group {g!r} not defined in config")
        cols.extend(groups[g])
    # Preserve order, drop duplicates.
    return list(dict.fromkeys(cols))


def _model_factory(spec: dict):
    """Build a zero-arg factory `() -> Model` from a config spec."""
    model_type = spec["type"]
    hyperparams = spec.get("hyperparams", {}) or {}

    def factory():
        return make_model(model_type, **hyperparams)
    return factory


def _benchmark_pnl(panel: pd.DataFrame, oos_dates: pd.Series,
                    kind: str) -> pd.DataFrame:
    """Construct a benchmark P&L frame on the OOS dates only.

    kind:
        "vxx_bah"  long-VXX, fully invested every day
        "cash"     zero return every day (the do-nothing baseline)
    """
    px = panel[["date", "vxx_close"]].copy()
    px["date"] = pd.to_datetime(px["date"]).dt.normalize()
    px = px.sort_values("date").reset_index(drop=True)
    px["vxx_ret_next"] = px["vxx_close"].pct_change().shift(-1)
    mask = px["date"].isin(pd.to_datetime(oos_dates).dt.normalize().unique())
    sub = px.loc[mask].copy()
    if kind == "vxx_bah":
        sub["size"] = 1.0
        sub["gross_pnl"] = sub["size"] * sub["vxx_ret_next"]
        sub["cost"] = 0.0
    elif kind == "cash":
        sub["size"] = 0.0
        sub["gross_pnl"] = 0.0
        sub["cost"] = 0.0
    else:
        raise ValueError(f"unknown benchmark kind: {kind}")
    sub["net_pnl"] = sub["gross_pnl"].fillna(0) - sub["cost"]
    sub["equity"] = (1.0 + sub["net_pnl"]).cumprod()
    return sub.reset_index(drop=True)


def run(config_path: Path, verbose: bool = True) -> pd.DataFrame:
    cfg = yaml.safe_load(config_path.read_text())
    if not PANEL_PATH.exists():
        sys.exit(f"missing {PANEL_PATH} — run `python -m features.assemble` first")
    panel = pd.read_parquet(PANEL_PATH)
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()

    groups = cfg.get("feature_groups", {})
    wf_cfg = WalkForwardConfig(
        initial_train_months=cfg["walk_forward"]["initial_train_months"],
        refit_freq_months=cfg["walk_forward"]["refit_freq_months"],
        expanding=cfg["walk_forward"].get("expanding", True),
    )
    rolling = None if wf_cfg.expanding else cfg["walk_forward"].get("rolling_train_months")
    exec_cfg = ExecConfig(
        base_bps_per_side=cfg["execution"].get("cost_bps_per_side", 5.0),
        extra_bps_high_vol=cfg["execution"].get("slippage_bps_on_flip", 1.0),
    )
    vix_z = panel.set_index("date")["vix_zscore_lag1"]

    all_preds: list[pd.DataFrame] = []
    all_pnl: list[pd.DataFrame] = []
    rows: list[dict] = []

    for spec in cfg["models"]:
        name = spec["name"]
        feature_cols = _resolve_feature_cols(spec, groups)
        missing = [c for c in feature_cols if c not in panel.columns]
        if missing:
            print(f"[skip] {name}: missing columns {missing}")
            continue

        sub = panel.dropna(subset=feature_cols + ["y_next"])
        if len(sub) < 50:
            print(f"[skip] {name}: only {len(sub)} clean rows after dropna")
            continue
        if verbose:
            print(f"[run]  {name:<24s} features={len(feature_cols):2d}  "
                  f"rows={len(sub):4d}")

        preds = walk_forward_run(
            sub, feature_cols=feature_cols, target_col="y_next",
            date_col="date", model_factory=_model_factory(spec),
            cfg=wf_cfg, rolling_train_months=rolling,
        )
        if preds.empty:
            print(f"[skip] {name}: walk-forward produced no predictions")
            continue
        preds["model_name"] = name
        all_preds.append(preds)

        pnl = backtest(preds, panel[["date", "vxx_close"]],
                       cfg=exec_cfg, vix_zscore=vix_z)
        pnl["model_name"] = name
        all_pnl.append(pnl)

        cls = classification_metrics(preds)
        bench_cash = _benchmark_pnl(panel, preds["date"], "cash")
        trd = trader_summary(pnl, benchmark_returns=bench_cash["net_pnl"])
        rows.append({"model": name, **cls, **trd})

    if not rows:
        sys.exit("no models produced predictions — check the panel + config")

    # Benchmark rows (computed over the union of all OOS dates).
    oos_dates = pd.concat(all_preds, ignore_index=True)["date"].drop_duplicates()
    for bench_name in ["vxx_bah", "cash"]:
        bp = _benchmark_pnl(panel, oos_dates, bench_name)
        if bp.empty:
            continue
        trd = trader_summary(bp)
        rows.append({
            "model": f"BENCH_{bench_name}",
            "auc": float("nan"), "brier": float("nan"),
            "logloss": float("nan"), "logloss_base": float("nan"),
            "base_rate": float("nan"), "n_obs": len(bp),
            **trd,
        })

    PRED_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(all_preds, ignore_index=True).to_parquet(PRED_PATH, index=False)
    pd.concat(all_pnl, ignore_index=True).to_parquet(PNL_PATH, index=False)

    summary = pd.DataFrame(rows).set_index("model")
    summary.to_csv(SUMMARY_PATH)

    if verbose:
        _print_summary(summary)
    return summary


def _print_summary(summary: pd.DataFrame) -> None:
    """Compact, recruiter-friendly metrics table."""
    cols = [
        ("auc", "AUC", "{:.3f}"),
        ("n_obs", "N", "{:.0f}"),
        ("total_return", "TotRet", "{:+.1%}"),
        ("cagr", "CAGR", "{:+.1%}"),
        ("ann_vol", "AnnVol", "{:.1%}"),
        ("sharpe_net", "Sharpe", "{:+.2f}"),
        ("sharpe_ci_lo", "SR-lo", "{:+.2f}"),
        ("sharpe_ci_hi", "SR-hi", "{:+.2f}"),
        ("psr_vs_zero", "PSR", "{:.2f}"),
        ("sortino", "Sortino", "{:+.2f}"),
        ("calmar", "Calmar", "{:+.2f}"),
        ("max_drawdown", "MaxDD", "{:.1%}"),
        ("cvar_95", "CVaR95", "{:.2%}"),
        ("turnover", "Turn", "{:.2%}"),
        ("time_in_market", "TIM", "{:.0%}"),
    ]
    headers = [label for _, label, _ in cols]
    print()
    print(f"{'model':<24s}  " + "  ".join(f"{h:>7s}" for h in headers))
    print("-" * (26 + 9 * len(headers)))
    for model, row in summary.iterrows():
        cells = []
        for c, _, fmt in cols:
            v = row.get(c, float("nan"))
            cells.append(fmt.format(v) if pd.notna(v) else "    nan")
        print(f"{model:<24s}  " + "  ".join(f"{c:>7s}" for c in cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path, nargs="?",
                    default=REPO_ROOT / "configs" / "experiment.yaml")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    run(args.config, verbose=not args.quiet)


if __name__ == "__main__":
    main()
