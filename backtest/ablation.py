"""Leave-one-feature-group-out ablation.

Given a feature-group mapping and a base model factory, fits the model under the
walk-forward harness with each group dropped in turn, and reports the resulting
classification metrics. The headline artefact for the report.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

from backtest.metrics import classification_metrics
from backtest.walk_forward import WalkForwardConfig, run as walk_forward_run


def run(
    panel: pd.DataFrame,
    feature_groups: dict[str, list[str]],
    target_col: str,
    date_col: str,
    model_factory: Callable[[], object],
    cfg: WalkForwardConfig,
) -> pd.DataFrame:
    all_features = [c for cols in feature_groups.values() for c in cols]
    rows = []

    # Full feature set baseline
    preds = walk_forward_run(panel, all_features, target_col, date_col, model_factory, cfg)
    m = classification_metrics(preds)
    rows.append({"variant": "full", "dropped_group": None, "n_features": len(all_features), **m})

    # Leave-one-group-out
    for group, cols in feature_groups.items():
        kept = [c for c in all_features if c not in cols]
        if not kept:
            continue
        preds = walk_forward_run(panel, kept, target_col, date_col, model_factory, cfg)
        m = classification_metrics(preds)
        rows.append({
            "variant": f"drop_{group}",
            "dropped_group": group,
            "n_features": len(kept),
            **m,
        })

    return pd.DataFrame(rows)
