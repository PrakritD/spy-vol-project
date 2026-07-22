"""No-lookahead gate for the walk-forward term-structure PCA sizing signal.

Mirrors test_strategy.py::test_ml_sizing_is_causal_and_lagged for the new PC2 (slope) walk-
forward construction in analysis/vix_futures_term_pca.py: perturb raw tenor levels only in the
far-future tail of a synthetic panel and assert every earlier PC2 score and size multiplier is
byte-identical.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import vix_futures_term_pca as T  # noqa: E402


def _synthetic_term_panel(n: int = 800, seed: int = 0) -> pd.DataFrame:
    """A daily panel with the lagged tenor-level columns and t_30_90 pc2_walkforward needs."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2012-01-02", periods=n)
    level = 20 + np.cumsum(rng.normal(0, 0.3, n))
    slope = rng.uniform(-3, 3, n)                          # crosses contango/backwardation
    d = {"date": dates, "t_30_90": np.clip(1.0 - slope / 20, 0.7, 1.3)}
    for i, c in enumerate(T.TENOR_COLS):
        d[c + "_l"] = level + slope * (i + 1) / len(T.TENOR_COLS) + rng.normal(0, 0.5, n)
    return pd.DataFrame(d)


def test_pc2_walkforward_no_lookahead():
    n = 800
    panel = _synthetic_term_panel(n=n)
    pc2_0 = T.pc2_walkforward(panel)
    mult_0 = T.size_multiplier(pc2_0)

    perturbed = panel.copy()
    fut = perturbed.index >= (n - 100)
    rng = np.random.default_rng(99)
    for c in T.LAGGED_COLS:
        perturbed.loc[fut, c] = perturbed.loc[fut, c] * rng.uniform(0.5, 1.5, fut.sum())
    pc2_1 = T.pc2_walkforward(perturbed)
    mult_1 = T.size_multiplier(pc2_1)

    c = min(len(pc2_0), len(pc2_1)) - 200          # margin beyond any rolling/expanding warmup
    assert c > 300
    np.testing.assert_array_equal(pc2_0[:c], pc2_1[:c])
    np.testing.assert_array_equal(mult_0[:c], mult_1[:c])


def test_size_multiplier_bounded():
    panel = _synthetic_term_panel()
    pc2 = T.pc2_walkforward(panel)
    mult = T.size_multiplier(pc2)
    assert (mult >= 0).all() and (mult <= T.CAP).all()
