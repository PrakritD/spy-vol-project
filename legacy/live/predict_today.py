"""Live-readiness stub: today's sized position from the trained model.

This is what a production cron job would look like, stripped of broker /
risk-limit / monitoring concerns. Every Friday at 16:30 ET (after the
US options close), a scheduler would run:

    python -m live.predict_today --model xgb_full --threshold 0.55

and ship the printed result to the order management system.

What this function does:
    1. Loads `data/processed/features_panel.parquet` (latest features panel
       built by `make features`).
    2. Trains the chosen model on ALL available rows (no holdout — at live
       time we use every bit of history).
    3. Pulls the latest row's feature vector (most recent t in the panel).
    4. Calls `model.predict_proba(X_today)` to get P(y=1).
    5. Maps the probability to a sized long-flat VXX position via the
       linear-confidence sizing rule from `backtest/sizing.py`.
    6. Returns and prints `(date, p_hat, size, action)`.

What it does NOT do (the production layer this layer would feed into):
    - Refresh raw data (yfinance / Databento) — assumes panel is current.
    - Place orders through a broker API.
    - Enforce per-strategy or portfolio-level risk limits.
    - Log to a monitoring system / paper-trading audit trail.
    - Handle exceptions for stale data, broker timeouts, etc.
    - Run on a schedule (cron / Airflow / Prefect orchestration).

The boundary between this stub and a real production system is documented
above so an interviewer can see the engineering reasoning without having
to read between the lines.

Usage:
    from live.predict_today import predict_today
    out = predict_today(model_type="xgb_full", threshold=0.55)
    # out = {"date": "2026-04-29", "p_hat": 0.637, "size": 0.274, "action": "long", "model": "xgb_full"}
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from backtest.sizing import linear_sizing
from models.factory import make_model


REPO_ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = REPO_ROOT / "data" / "processed" / "features_panel.parquet"
EXPERIMENT_CFG = REPO_ROOT / "configs" / "experiment.yaml"


def _resolve_feature_cols(spec: dict, groups: dict) -> list[str]:
    cols: list[str] = []
    for g in spec.get("feature_groups", []):
        if g not in groups:
            raise KeyError(f"feature_group {g!r} not defined in config")
        cols.extend(groups[g])
    return list(dict.fromkeys(cols))


def predict_today(
    model_type: str = "xgb_full",
    threshold: float = 0.55,
    config_path: Path | None = None,
    panel_path: Path | None = None,
) -> dict:
    """Produce today's sized VXX position from the trained classifier.

    Args:
        model_type: name of the model spec in `configs/experiment.yaml`
                    (must be one of: logistic_vix_only, logistic_interactions,
                    har_x, xgb_full, mlp_small, bayesian_head).
        threshold:  P(y=1) cutoff — long below threshold means size=0.
        config_path: override path to experiment.yaml.
        panel_path:  override path to features_panel.parquet.

    Returns:
        dict with keys:
            date    -- the trading date the prediction is FOR (panel's latest row)
            p_hat   -- model's predicted P(y=1)
            size    -- sized position in [0, 1] (fraction of unit notional)
            action  -- "long" if size > 0 (above threshold), else "flat"
            model   -- the model name used
            note    -- short status / metadata string
    """
    config_path = config_path or EXPERIMENT_CFG
    panel_path = panel_path or PANEL_PATH

    if not panel_path.exists():
        raise FileNotFoundError(
            f"missing {panel_path}. Run `make features` first to build the panel."
        )

    cfg = yaml.safe_load(config_path.read_text())

    # Find the chosen model's spec in experiment.yaml
    spec = next((m for m in cfg["models"] if m["name"] == model_type), None)
    if spec is None:
        avail = [m["name"] for m in cfg["models"]]
        raise ValueError(
            f"model_type {model_type!r} not in experiment.yaml. Available: {avail}"
        )

    groups = cfg.get("feature_groups", {})
    feature_cols = _resolve_feature_cols(spec, groups)

    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values("date").reset_index(drop=True)

    # Training set: every row where features + target are observed.
    train = panel.dropna(subset=feature_cols + ["y_next"])
    if len(train) < 50:
        raise RuntimeError(
            f"insufficient training data ({len(train)} clean rows). "
            "Need at least 50 to train any model."
        )

    # Today's prediction row: the latest row where features are observed
    # (the target y_next may or may not be available yet — that's fine,
    # we don't need it for prediction).
    pred_rows = panel.dropna(subset=feature_cols).copy()
    if pred_rows.empty:
        raise RuntimeError("no row in panel has all required features non-null")
    latest = pred_rows.iloc[-1]
    pred_date = pd.Timestamp(latest["date"])

    # Train on history strictly BEFORE today.
    train_for_today = train.loc[train["date"] < pred_date]
    if train_for_today.empty:
        # Edge case: today is the earliest panel row. Fall back to full panel.
        train_for_today = train

    hyper = spec.get("hyperparams", {}) or {}
    # `model_type` is the spec NAME (e.g. "logistic_vix_only"); the factory
    # takes the spec TYPE (e.g. "logistic"). Resolve via the spec we already
    # looked up.
    model = make_model(spec["type"], **hyper)
    model.fit(train_for_today[feature_cols], train_for_today["y_next"].astype(int))

    # predict_proba returns 1-D array per the Model protocol contract.
    X_today = pred_rows.iloc[[-1]][feature_cols]
    p_hat = float(model.predict_proba(X_today)[0])

    size = float(np.clip(linear_sizing(np.array([p_hat]))[0], 0.0, 1.0))
    if threshold is not None:
        # Apply a probability cutoff in addition to confidence sizing.
        # Below threshold -> size=0. Above -> size as computed.
        if p_hat < threshold:
            size = 0.0

    action = "long" if size > 0.0 else "flat"

    result = {
        "date": pred_date.date().isoformat(),
        "p_hat": round(p_hat, 4),
        "size": round(size, 4),
        "action": action,
        "model": model_type,
        "threshold": threshold,
        "n_train_rows": int(len(train_for_today)),
        "note": f"Trained on {len(train_for_today)} rows up to {pred_date.date()}",
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="xgb_full",
                    help="Model spec name from configs/experiment.yaml.")
    ap.add_argument("--threshold", type=float, default=0.55,
                    help="P(y=1) cutoff below which size=0.")
    ap.add_argument("--config", type=Path, default=None,
                    help="Override experiment.yaml path.")
    ap.add_argument("--panel", type=Path, default=None,
                    help="Override features_panel.parquet path.")
    ap.add_argument("--json", action="store_true",
                    help="Print result as JSON instead of pretty text.")
    args = ap.parse_args()

    try:
        result = predict_today(
            model_type=args.model,
            threshold=args.threshold,
            config_path=args.config,
            panel_path=args.panel,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"=== predict_today: {result['model']} ===")
        print(f"  for trading date : {result['date']}")
        print(f"  P(y=1)           : {result['p_hat']:.4f}")
        print(f"  threshold        : {result['threshold']}")
        print(f"  sized position   : {result['size']:.4f}  ({result['action']})")
        print(f"  trained on       : {result['n_train_rows']} rows")
        print(f"  generated at     : {result['generated_at_utc']}")


if __name__ == "__main__":
    main()
