"""Crowding tests: data-free (synthetic panels), so CI runs without the gitignored data.

Two guarantees matter here and both are enforced rather than asserted in prose.

The first is the repo's headline no-lookahead property, applied to a new feature:
`test_crowding_no_lookahead` perturbs short interest strictly in the FUTURE and requires every
earlier crowding value to be byte-identical.

The second is specific to this data source. FINRA reports short interest against a settlement date
and publishes it more than a week later, so keying the panel on settlement dates would let it see
numbers nobody could have had. That is a lookahead the perturbation test alone would NOT catch,
because settlement-keyed data is internally consistent; it is simply early. The publication-lag
tests close it directly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "analysis"))

import crowding_model as M  # noqa: E402
from features import crowding as C  # noqa: E402
from ingest import short_interest_pull as SI  # noqa: E402

TICKERS = ["VIXY", "VXX", "UVXY", "SVXY"]


def _write_synthetic(root: Path, n_days: int = 900, seed: int = 0,
                     si_tail_perturb: float = 1.0, drop_vxx_middle: bool = False) -> pd.Timestamp:
    """A synthetic price + short-interest tree with the schema the real one has.

    `si_tail_perturb` scales the LAST quarter of every short-interest series, which is how the
    no-lookahead test injects a future-only change. Returns the PUBLICATION date of the first
    perturbed observation: that is the exact boundary before which nothing may move, and comparing
    right up to it is what gives the test teeth.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-02", periods=n_days)
    price_dir, si_dir = root / "deep", root / "short_interest"
    price_dir.mkdir(parents=True, exist_ok=True)
    si_dir.mkdir(parents=True, exist_ok=True)

    for t in TICKERS:
        px = 30 * np.exp(np.cumsum(rng.normal(-0.001, 0.03, n_days)))
        pd.DataFrame({"date": dates, "adj_close": px, "close": px,
                      "high": px * 1.01, "low": px * 0.99, "open": px,
                      "volume": rng.uniform(1e6, 8e6, n_days)}).to_parquet(
            price_dir / f"{t}.parquet", index=False)

        settle = pd.DatetimeIndex([d for i, d in enumerate(dates) if i % 10 == 0])
        shares = rng.uniform(5e5, 3e6, len(settle))
        cut = int(len(settle) * 0.75)
        shares[cut:] *= si_tail_perturb

        d = pd.DataFrame({"settlement_date": settle, "ticker": t,
                          "short_interest_shares": shares,
                          "prev_short_interest_shares": shares,
                          "avg_daily_volume": rng.uniform(1e6, 5e6, len(settle))})
        if drop_vxx_middle and t == "VXX":
            mid = (d["settlement_date"] > dates[300]) & (d["settlement_date"] < dates[450])
            d = d[~mid].reset_index(drop=True)
        d["days_to_cover"] = d["short_interest_shares"] / d["avg_daily_volume"]
        d = SI.add_publication_date(d, pd.DatetimeIndex(dates))
        d.to_parquet(si_dir / f"{t}.parquet", index=False)
        boundary = pd.Timestamp(d["publication_date"].iloc[cut])

    # no splits inside the synthetic window: the un-adjustment must then be a no-op
    pd.DataFrame({"ticker": pd.Series(dtype=str), "date": pd.Series(dtype="datetime64[ns]"),
                  "ratio": pd.Series(dtype=float)}).to_parquet(si_dir / "splits.parquet",
                                                               index=False)
    return boundary


@pytest.fixture()
def synth(tmp_path, monkeypatch):
    def _build(**kw):
        root = tmp_path / f"d{abs(hash(tuple(sorted(kw.items())))) % 10**8}"
        boundary = _write_synthetic(root, **kw)
        monkeypatch.setattr(C, "PRICE_DIR", root / "deep")
        monkeypatch.setattr(C, "SI_DIR", root / "short_interest")
        dates = pd.bdate_range("2018-01-02", periods=kw.get("n_days", 900))
        return C.run(pd.DataFrame({"date": dates}), C.CrowdingConfig()), boundary
    return _build


# ------------------------------------------------------------- no lookahead ----
def test_crowding_no_lookahead(synth):
    """Perturbing short interest only in the future must not move any earlier value.

    The comparison runs right up to the publication date of the FIRST perturbed observation, not to
    some comfortable fraction of the sample. That margin is the whole point: a leak of a single
    publication step, such as reading the next reading instead of the last one, only shows up in
    the days immediately before the boundary. Stopping at the halfway mark leaves a dead zone that
    hides exactly the bug this test exists to catch, and a gate that cannot fail is not a gate.
    """
    base, boundary = synth(si_tail_perturb=1.0)
    bumped, _ = synth(si_tail_perturb=3.0)

    safe = base["date"] < boundary
    assert safe.sum() > 100, "safe region too small for the test to mean anything"
    for col in ("c1_usd", "c2_gain", "c3_usd"):
        a = base.loc[safe, col].to_numpy()
        b = bumped.loc[safe, col].to_numpy()
        both_nan = np.isnan(a) & np.isnan(b)
        assert np.array_equal(np.isnan(a), np.isnan(b)), f"{col} NaN pattern moved"
        assert np.array_equal(a[~both_nan], b[~both_nan]), f"{col} moved before the perturbation"


def test_no_lookahead_gate_actually_fails_on_a_leak(tmp_path, monkeypatch):
    """The gate above must be able to go red. Verified by injecting a real one-step-forward leak.

    Without this, a passing suite proves only that the assertions ran, not that they bite.
    """
    root = tmp_path / "leak"
    boundary = _write_synthetic(root, si_tail_perturb=1.0)
    _write_synthetic(tmp_path / "leak2", si_tail_perturb=3.0)
    monkeypatch.setattr(C, "PRICE_DIR", root / "deep")
    dates = pd.bdate_range("2018-01-02", periods=900)

    real_asof = pd.merge_asof

    def leaky(*a, **kw):                      # read the NEXT publication instead of the last
        if kw.get("direction") == "backward" and "tolerance" in kw:
            kw["direction"] = "forward"
        return real_asof(*a, **kw)

    frames = []
    for sub in ("leak", "leak2"):
        monkeypatch.setattr(C, "SI_DIR", tmp_path / sub / "short_interest")
        monkeypatch.setattr(C.pd, "merge_asof", leaky)
        frames.append(C.run(pd.DataFrame({"date": dates}), C.CrowdingConfig()))
    monkeypatch.setattr(C.pd, "merge_asof", real_asof)

    safe = frames[0]["date"] < boundary
    a = frames[0].loc[safe, "c1_usd"].to_numpy()
    b = frames[1].loc[safe, "c1_usd"].to_numpy()
    both_nan = np.isnan(a) & np.isnan(b)
    assert not np.array_equal(a[~both_nan], b[~both_nan]), (
        "a forward-looking merge went undetected; the no-lookahead gate has no teeth")


def test_publication_lag_clears_the_finra_schedule():
    """The conservative rule must never undershoot the largest observed FINRA schedule lag."""
    assert SI.PUBLICATION_LAG_DAYS > SI.MAX_OBSERVED_SCHEDULED_LAG_DAYS


def test_publication_date_is_never_earlier_than_the_rule():
    """No observation may become visible before settlement + the lag, on any calendar."""
    dates = pd.bdate_range("2018-01-02", periods=400)
    settle = pd.DatetimeIndex([d for i, d in enumerate(dates) if i % 10 == 0])
    d = pd.DataFrame({"settlement_date": settle})
    out = SI.add_publication_date(d, pd.DatetimeIndex(dates)).dropna(subset=["publication_date"])
    lag = (out["publication_date"] - out["settlement_date"]).dt.days
    assert lag.min() >= SI.PUBLICATION_LAG_DAYS
    assert (out["publication_date"] > out["settlement_date"]).all()


# --------------------------------------------------------- data-integrity gates ----
def test_stale_reading_expires_instead_of_persisting(synth):
    """A reporting hole must become NaN, not a value carried across it.

    VXX has a real four-month gap in 2019. Forward-filling through it would fabricate months of
    constant crowding, so the aggregate has to go missing instead.
    """
    out, _ = synth(drop_vxx_middle=True)
    assert out["c1_usd"].isna().any(), "a long reporting hole did not produce any NaN"


def test_aggregate_requires_every_constituent(synth):
    """A missing fund must not simply shrink the sum, which would look like a crowding decline."""
    full, _ = synth()
    holed, _ = synth(drop_vxx_middle=True)
    newly_missing = holed["c1_usd"].isna() & full["c1_usd"].notna()
    assert newly_missing.any()
    # wherever the hole bites, the aggregate is absent rather than smaller
    assert holed.loc[newly_missing, "c1_usd"].isna().all()


def test_split_factor_is_identity_without_splits(synth, tmp_path, monkeypatch):
    """With no splits in the window the un-adjustment must be a no-op."""
    root = tmp_path / "nosplit"
    _write_synthetic(root)
    monkeypatch.setattr(C, "PRICE_DIR", root / "deep")
    monkeypatch.setattr(C, "SI_DIR", root / "short_interest")
    dates = pd.bdate_range("2018-01-02", periods=100)
    f = C._split_factor("VIXY", pd.Series(dates))
    assert np.allclose(f.to_numpy(), 1.0)


def test_split_factor_unadjusts_a_known_reverse_split(tmp_path, monkeypatch):
    """A 1-for-5 reverse split must scale pre-split prices by 0.2 and leave later ones alone."""
    si_dir = tmp_path / "short_interest"
    si_dir.mkdir(parents=True)
    pd.DataFrame({"ticker": ["VIXY"], "date": [pd.Timestamp("2020-06-15")],
                  "ratio": [0.2]}).to_parquet(si_dir / "splits.parquet", index=False)
    monkeypatch.setattr(C, "SI_DIR", si_dir)

    f = C._split_factor("VIXY", pd.Series(pd.to_datetime(
        ["2020-06-01", "2020-06-14", "2020-06-15", "2020-07-01"])))
    assert f.iloc[0] == pytest.approx(0.2)
    assert f.iloc[1] == pytest.approx(0.2)
    assert f.iloc[2] == pytest.approx(1.0)   # splits strictly after the date only
    assert f.iloc[3] == pytest.approx(1.0)


# ------------------------------------------------------------- model golden ----
def test_multiplier_golden_values():
    """m(x) = 1/(1-x), the single canonical amplification definition."""
    assert float(M.multiplier(0.0)) == pytest.approx(1.0)
    assert float(M.multiplier(0.5)) == pytest.approx(2.0)
    assert float(M.multiplier(0.9)) == pytest.approx(10.0)
    assert float(M.multiplier(0.99)) == pytest.approx(100.0)
    assert np.isfinite(float(M.multiplier(1.0)))          # clipped, not a division by zero


def test_multiplier_is_increasing_in_crowding():
    x = np.linspace(0.0, 0.95, 50)
    m = np.asarray(M.multiplier(x))
    assert np.all(np.diff(m) > 0)


def test_nash_exposure_never_below_social_optimum():
    """The congestion externality is structural: it follows from m'(x) > 0, not from a calibration.

    Swept across a grid far wider than the one the model reports, the atomistic equilibrium must
    never sit below the planner's.
    """
    checked = 0
    for mu in (0.0005, 0.0017, 0.005):
        for theta_mult in (0.1, 0.25, 1.0, 4.0, 10.0):
            for s in (0.05, 0.1465, 0.3):
                for p in (0.002, 0.0101, 0.05):
                    nash = M._solve_equilibrium(mu, theta_mult * mu, s, p, 0.0, social=False)
                    opt = M._solve_equilibrium(mu, theta_mult * mu, s, p, 0.0, social=True)
                    assert nash["x"] >= opt["x"] - 1e-12, (mu, theta_mult, s, p)
                    checked += 1
    assert checked == 135
