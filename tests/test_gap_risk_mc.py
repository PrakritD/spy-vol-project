"""Data-free tests for the gap-risk regime-conditional bootstrap (analysis/gap_risk_mc.py).

Mirrors test_fast_iv.py's convention of validating a numerical method on synthetic inputs
before trusting it on real data, and test_vix_futures_term_pca.py's convention of testing
against the module directly rather than through main(). No parquet read is exercised here
(strategy_equity.parquet is gitignored, so CI must run without it, same discipline as
test_strategy.py's synthetic-panel tests) -- only the block-bootstrap mechanics and the
metrics/aggregation functions, which take plain arrays.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import gap_risk_mc as G  # noqa: E402


def test_stationary_bootstrap_indices_shape_and_range():
    n, n_draws = 200, 50
    rng = np.random.default_rng(0)
    idx = G.stationary_bootstrap_indices(n, n_draws, mean_block=30.0, rng=rng)
    assert idx.shape == (n_draws, n)
    assert idx.min() >= 0
    assert idx.max() < n


def test_stationary_bootstrap_is_contiguous_with_wraparound_for_long_blocks():
    """With mean_block >> n, restarts only fire at column 0 (forced), so every row is one
    contiguous run of consecutive original indices, wrapping around mod n."""
    n, n_draws = 100, 20
    rng = np.random.default_rng(1)
    idx = G.stationary_bootstrap_indices(n, n_draws, mean_block=1e9, rng=rng)
    for row in idx:
        diffs = np.diff(row) % n
        assert np.all(diffs == 1)


def test_stationary_bootstrap_short_blocks_look_close_to_iid():
    """With mean_block ~ 1, almost every position restarts, so consecutive resampled
    indices should rarely be exactly contiguous (unlike the long-block case above)."""
    n, n_draws = 500, 20
    rng = np.random.default_rng(2)
    idx = G.stationary_bootstrap_indices(n, n_draws, mean_block=1.0, rng=rng)
    diffs = np.diff(idx, axis=1) % n
    # contiguous continuation would show up as diffs == 1; with block length ~1 this
    # should be rare, not the near-universal pattern seen in the long-block test
    assert (diffs == 1).mean() < 0.1


def test_path_metrics_shape_and_known_drawdown():
    """A flat day (establishing the peak) followed by a 10% drop and a partial recovery
    has a known maxDD of -10%. The initial day must be flat (not itself the drop) because
    the running-peak accumulator starts from the first observation, so a drop on day 0
    would not register as a decline from an established peak -- same convention as
    drawdown_inference.py's identical path_metrics."""
    r = np.array([0.0, -0.10, 0.05, 0.05])
    m = G.path_metrics(r[None, :])
    assert m["maxdd"].shape == (1,)
    np.testing.assert_allclose(m["maxdd"][0], -0.10, atol=1e-9)


def test_path_metrics_flat_series_has_zero_drawdown():
    r = np.zeros((3, 50))
    m = G.path_metrics(r)
    np.testing.assert_allclose(m["maxdd"], 0.0, atol=1e-12)


def _synthetic_regime_series(n: int = 1500, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Flat days (~0 return) interspersed with in-market stretches that include one sharp
    crisis block, mimicking the shape of the real carry series closely enough to exercise
    the regime-conditioned bootstrap mechanics."""
    rng = np.random.default_rng(seed)
    inmkt = (rng.random(n) < 0.9).astype(float)
    r = np.where(inmkt > 0, rng.normal(0.0003, 0.01, n), 0.0)
    crisis = slice(400, 460)
    r[crisis] = rng.normal(-0.02, 0.01, 460 - 400)
    inmkt[crisis] = 1.0
    return r, inmkt


def test_run_bootstrap_reproducible_with_fixed_seed():
    r, inmkt = _synthetic_regime_series()
    res1 = G.run_bootstrap(r, inmkt, mean_block=90.0, n_draws=200, seed=42)
    res2 = G.run_bootstrap(r, inmkt, mean_block=90.0, n_draws=200, seed=42)
    assert res1 == res2


def test_run_bootstrap_output_sanity():
    r, inmkt = _synthetic_regime_series()
    res = G.run_bootstrap(r, inmkt, mean_block=90.0, n_draws=300, seed=1)
    lo, hi = res["maxdd_ci95"]
    assert lo <= res["maxdd_median"] <= hi
    assert lo <= 0.0
    for key in (
        "p_maxdd_worse_than_-20pct",
        "p_maxdd_worse_than_-25pct",
        "p_maxdd_worse_than_-30pct",
    ):
        assert 0.0 <= res[key] <= 1.0
    # stricter thresholds are (weakly) rarer than looser ones
    assert res["p_maxdd_worse_than_-20pct"] >= res["p_maxdd_worse_than_-25pct"]
    assert res["p_maxdd_worse_than_-25pct"] >= res["p_maxdd_worse_than_-30pct"]
    assert 0.0 <= res["mean_inmkt_fraction"] <= 1.0


def test_longer_blocks_preserve_more_crisis_clustering():
    """A single concentrated crisis block should produce a fatter left tail of maxDD under
    longer mean block lengths, since short blocks are more likely to slice through the
    crisis stretch and dilute it across draws rather than reproduce it whole."""
    r, inmkt = _synthetic_regime_series()
    short = G.run_bootstrap(r, inmkt, mean_block=5.0, n_draws=2000, seed=3)
    long_ = G.run_bootstrap(r, inmkt, mean_block=90.0, n_draws=2000, seed=3)
    assert long_["maxdd_ci95"][0] <= short["maxdd_ci95"][0]
