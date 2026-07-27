"""Crowding in the short-volatility trade, from FINRA short interest (CROWDING.md).

Short interest in a LONG-volatility ETP (VIXY, VXX, UVXY) is short-volatility exposure held by
someone. Short interest in the INVERSE fund SVXY is the opposite, long-volatility exposure, so it
enters with a negative sign. Weighting each fund by its leverage puts them on a common vega-
equivalent footing.

Exposes the repo's standard `run(df, cfg) -> daily_frame`. It is deliberately NOT joined in
`features/assemble.py`: that module builds the 21-month OPRA options panel, whereas this is a
deep-history daily feature belonging to the strategy panel. It is joined instead in
`analysis/crowding_test.py`, against `build_signals(load_panel())`, with an explicit `shift(1)` at
the point of use, matching how every other predictor in `build_signals` is lagged.

## Three decisions that determine whether the measure means anything

**Publication dates, not settlement dates.** `ingest/short_interest_pull.py` stamps every
observation with the date it became public (settlement + 15 calendar days, snapped to a trading
day). This module keys off that column only. A value is visible on date t only if it was published
on or before t.

**Notional is frozen at the settlement-date price.** Dollar short interest could be computed as
shares times the CURRENT price, which would update daily, or shares times the price on the
settlement date, which updates only when new short interest is published. The second is used. The
first would make the crowding measure move daily with the VIXY price, and the VIXY price moves with
VIX, so the measure would be contaminated by the very quantity the study has to prove it is not
merely echoing. Freezing the notional means the measure's daily variation comes from positioning
alone.

**A stale reading expires rather than persisting.** The series is biweekly, so it is forward-filled
to daily, but only for `max_staleness_days`. VXX has a real four-month reporting hole in 2019, and
carrying a January number through to May would fabricate four months of constant crowding. Past the
staleness limit the value goes NaN. Relatedly, the aggregate requires ALL its constituents: summing
across a missing one would make the total fall when a fund simply was not reported, manufacturing a
crowding decline that never happened.

## Measures

    c1_usd    leverage-weighted dollar short interest across VIXY, VXX, UVXY
    c2_gain   c1_usd / ADV$, the loop gain of the model in analysis/crowding_model.py up to a
              constant. PRIMARY, per CROWDING-PREREG.md, chosen by the theory rather than by
              inspection of results
    c3_usd    c1_usd net of the leverage-weighted SVXY term

ADV$ is the rolling median dollar volume of the same three long-volatility ETPs, on the window in
`CrowdingConfig.adv_window`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SI_DIR = REPO_ROOT / "data" / "raw" / "short_interest"
PRICE_DIR = REPO_ROOT / "data" / "raw" / "deep"


@dataclass(frozen=True)
class CrowdingConfig:
    long_vol: tuple[str, ...] = ("VIXY", "VXX", "UVXY")
    inverse: tuple[str, ...] = ("SVXY",)

    leverage: dict[str, float] = field(default_factory=lambda: {
        # Verified empirically in CROWDING-PREREG.md by regressing each ETP's daily return on
        # VIXY's across the 2018-02-28 ProShares deleverage. The inference window sits entirely
        # after the break, so post-break weights apply throughout.
        "VIXY": 1.0, "VXX": 1.0, "UVXY": 1.5, "SVXY": -0.5,
    })

    max_staleness_days: int = 25
    """A published reading stays valid this long. The cadence is ~15 days, so this tolerates a long
    month without bridging VXX's real four-month 2019 hole."""

    adv_window: int = 21


def _load_prices(ticker: str) -> pd.DataFrame:
    d = pd.read_parquet(PRICE_DIR / f"{ticker}.parquet")
    d["date"] = pd.to_datetime(d["date"])
    return d[["date", "adj_close", "close", "volume"]].sort_values("date").reset_index(drop=True)


def _split_factor(ticker: str, dates: pd.Series) -> pd.Series:
    """Cumulative split factor F(t), converting a back-adjusted price to the contemporaneous one.

    yfinance back-adjusts historical prices, so a price stored for date t reflects every split that
    happened AFTER t. With split ratios expressed as new-per-old (0.2 for a 1-for-5 reverse split),

        true_price(t) = adjusted_price(t) * prod(ratio for every split dated after t)

    These ETPs reverse-split relentlessly, so F is far from 1: on UVXY it reaches 1/2500 early in
    the crowding window. FINRA share counts are contemporaneous and unadjusted, so without this the
    two sides of `shares * price` are in different unit systems.
    """
    sp = pd.read_parquet(SI_DIR / "splits.parquet")
    sp = sp[sp["ticker"] == ticker].sort_values("date")
    d = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    if sp.empty:
        return pd.Series(1.0, index=d.values)

    sd = pd.to_datetime(sp["date"]).values
    ratios = sp["ratio"].values
    # product of ratios strictly after each date: walk from the end
    factors = []
    for t in d.values:
        factors.append(float(ratios[sd > t].prod()) if (sd > t).any() else 1.0)
    return pd.Series(factors, index=d.values)


def _load_short_interest(ticker: str) -> pd.DataFrame:
    d = pd.read_parquet(SI_DIR / f"{ticker}.parquet")
    for c in ("settlement_date", "publication_date"):
        d[c] = pd.to_datetime(d[c])
    return d.sort_values("publication_date").reset_index(drop=True)


def _dollar_series(ticker: str, dates: pd.Series, cfg: CrowdingConfig) -> pd.Series:
    """Leverage-weighted dollar short interest for one ETP, aligned to `dates`.

    Notional is fixed at the settlement-date price (see module docstring), then held from the
    publication date until the next one, expiring after `max_staleness_days`.
    """
    si = _load_short_interest(ticker)
    px = _load_prices(ticker)

    # price on the settlement date; backward asof covers a settlement date that is a holiday
    si = pd.merge_asof(si, px[["date", "close"]].rename(columns={"date": "settlement_date"}),
                       on="settlement_date", direction="backward")
    # un-adjust to the contemporaneous price, so the stored price and FINRA's unadjusted share
    # count are in the same unit system (see _split_factor)
    si["true_close"] = si["close"] * _split_factor(ticker, si["settlement_date"]).values
    si["dollar_si"] = (si["short_interest_shares"] * si["true_close"]
                       * abs(cfg.leverage[ticker]))
    si = si.dropna(subset=["dollar_si", "publication_date"])

    target = pd.DataFrame({"date": pd.to_datetime(dates)}).sort_values("date")
    joined = pd.merge_asof(
        target,
        si[["publication_date", "dollar_si"]].rename(columns={"publication_date": "date"}),
        on="date", direction="backward",
        tolerance=pd.Timedelta(days=cfg.max_staleness_days),
    )
    return joined.set_index("date")["dollar_si"]


def _adv_dollars(dates: pd.Series, cfg: CrowdingConfig) -> pd.Series:
    """Rolling median dollar volume across the long-volatility ETPs, on `dates`."""
    target = pd.DataFrame({"date": pd.to_datetime(dates)}).sort_values("date")
    total = pd.Series(0.0, index=target["date"].values)
    for t in cfg.long_vol:
        px = _load_prices(t)
        dv = (px["volume"] * px["close"]).values
        s = pd.Series(dv, index=px["date"].values).reindex(target["date"].values)
        total = total.add(s.fillna(0.0), fill_value=0.0)
    return total.rolling(cfg.adv_window, min_periods=cfg.adv_window).median()


def run(df: pd.DataFrame, cfg: CrowdingConfig | None = None) -> pd.DataFrame:
    """Daily crowding measures aligned to `df['date']`.

    Returns date, c1_usd, c2_gain, c3_usd. Values are point-in-time as of the close of each date
    (publication-date keyed); callers apply `shift(1)` at the join, per repo convention.
    """
    cfg = cfg or CrowdingConfig()
    dates = pd.to_datetime(df["date"]).sort_values().reset_index(drop=True)

    long_parts = {t: _dollar_series(t, dates, cfg) for t in cfg.long_vol}
    inv_parts = {t: _dollar_series(t, dates, cfg) for t in cfg.inverse}

    long_frame = pd.DataFrame(long_parts)
    # require every constituent: a missing fund must not shrink the aggregate
    c1 = long_frame.sum(axis=1).where(long_frame.notna().all(axis=1))

    inv_frame = pd.DataFrame(inv_parts)
    c3 = (c1 - inv_frame.sum(axis=1)).where(inv_frame.notna().all(axis=1) & c1.notna())

    adv = _adv_dollars(dates, cfg)
    c2 = c1 / adv.replace(0.0, pd.NA)

    out = pd.DataFrame({
        "date": dates.values,
        "c1_usd": c1.reindex(dates.values).values,
        "c2_gain": c2.reindex(dates.values).values,
        "c3_usd": c3.reindex(dates.values).values,
    })
    return out.reset_index(drop=True)
