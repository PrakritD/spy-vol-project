"""Fetch FINRA consolidated short interest for the volatility ETPs (CROWDING.md).

Short interest in a LONG-volatility ETP is short-volatility exposure held by someone. Aggregated
across the vehicles and weighted by each fund's leverage, it is the only free, direct measure of
crowding in the trade STRATEGY.md runs. `ingest/deep_pull.py` pulls OHLCV only, so none of this is
otherwise on disk.

Writes:

    data/raw/short_interest/<TICKER>.parquet   per-ETP biweekly series
    data/raw/short_interest_manifest.json      rows, dates, sha256, source, versions

## The publication-date rule, which is the whole point of this file

FINRA reports short interest against a SETTLEMENT date and disseminates it more than a week later.
Keying a daily panel on the settlement date would let the panel see a number nobody could have had,
which is a lookahead error of exactly the kind `tests/test_strategy.py` exists to catch.

FINRA's published reporting schedule was read at pre-registration time across 24 consecutive
periods. The settlement-to-publication lag ranged from 9 to 12 calendar days and never exceeded 12.
This module therefore stamps each observation with

    publication_date = settlement_date + 15 calendar days, snapped forward to the next trading day

Fifteen clears every observed scheduled lag with margin. On a biweekly series the extra staleness
costs nothing; the protection against a lookahead is not optional. `MAX_OBSERVED_SCHEDULED_LAG_DAYS`
records what was measured so the margin can be audited rather than taken on trust, and
`tests/test_crowding.py` asserts the rule never undershoots it.

The trading-day snap uses SPY's own date index from `data/raw/deep/SPY.parquet` as the calendar,
rather than a holiday library, so the calendar is by construction the same one the panel runs on.

## Coverage

Verified 2026-07-27: VIXY, UVXY and SVXY each 206 biweekly observations from 2017-12-29, VXX 199
over the same span with one 120-day gap. The window opens well after VIXY's 2011 inception, so the
crowding study runs on a shorter sample than the flagship. That is a property of the free data, not
a choice.

Run:  python -m ingest.short_interest_pull            # fetch whatever is missing
      python -m ingest.short_interest_pull --force    # refetch (new vintage)
      python -m ingest.short_interest_pull --check    # validate coverage + the lag rule
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW = REPO_ROOT / "data" / "raw"
OUT_DIR = RAW / "short_interest"
MANIFEST = RAW / "short_interest_manifest.json"
SPY_PATH = RAW / "deep" / "SPY.parquet"

FINRA_URL = "https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest"
TICKERS = ["VIXY", "VXX", "UVXY", "SVXY"]

# Pinned to the vintage behind the committed strategy_results.json, matching deep_pull.py.
END = "2026-05-31"

PUBLICATION_LAG_DAYS = 15
MAX_OBSERVED_SCHEDULED_LAG_DAYS = 12
"""Largest settlement-to-publication gap in FINRA's published schedule, read across 24 consecutive
periods on 2026-07-27 (range 9 to 12 calendar days). PUBLICATION_LAG_DAYS must stay above this."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _trading_days() -> pd.DatetimeIndex | None:
    """SPY's own date index, used as the trading calendar for the publication-date snap."""
    if not SPY_PATH.exists():
        return None
    d = pd.read_parquet(SPY_PATH)
    return pd.DatetimeIndex(pd.to_datetime(d["date"])).sort_values()


def add_publication_date(df: pd.DataFrame, calendar: pd.DatetimeIndex | None) -> pd.DataFrame:
    """Stamp each settlement date with the date the number became public.

    Falls back to a next-business-day snap when SPY is not on disk, so the module still works on a
    bare clone; the calendar path is the one used in practice.
    """
    d = df.copy()
    raw = d["settlement_date"] + pd.Timedelta(days=PUBLICATION_LAG_DAYS)
    if calendar is not None and len(calendar):
        idx = calendar.searchsorted(raw.values, side="left")
        inside = idx < len(calendar)
        pub = pd.Series(pd.NaT, index=d.index, dtype="datetime64[ns]")
        pub.loc[inside] = calendar[idx[inside]]
        # Observations whose publication date runs past the calendar's end are dropped rather than
        # guessed: they are not yet usable by a panel that stops where the calendar stops.
        d["publication_date"] = pub
    else:
        d["publication_date"] = raw + pd.offsets.BDay(0)
    return d


def fetch_one(ticker: str) -> pd.DataFrame:
    body = {"limit": 5000,
            "compareFilters": [{"fieldName": "symbolCode",
                                "fieldValue": ticker,
                                "compareType": "EQUAL"}]}
    r = requests.post(FINRA_URL, json=body, timeout=60)
    r.raise_for_status()
    raw = pd.read_csv(io.StringIO(r.text))
    if raw.empty:
        raise RuntimeError(f"FINRA returned no rows for {ticker}")

    d = pd.DataFrame({
        "settlement_date": pd.to_datetime(raw["settlementDate"]),
        "ticker": ticker,
        "short_interest_shares": pd.to_numeric(raw["currentShortPositionQuantity"],
                                               errors="coerce"),
        "prev_short_interest_shares": pd.to_numeric(raw["previousShortPositionQuantity"],
                                                    errors="coerce"),
        "avg_daily_volume": pd.to_numeric(raw["averageDailyVolumeQuantity"], errors="coerce"),
    })
    # daysToCoverQuantity is floored at 1.00 on every row for these tickers and carries no
    # information; cover days are recomputed from the two quantities instead.
    d["days_to_cover"] = d["short_interest_shares"] / d["avg_daily_volume"].replace(0, pd.NA)
    return d.dropna(subset=["short_interest_shares"]).sort_values("settlement_date").reset_index(
        drop=True)


def pull(end: str, force: bool) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    calendar = _trading_days()
    if calendar is None:
        print("  WARNING: data/raw/deep/SPY.parquet absent; falling back to a business-day snap "
              "for publication dates. Run `make deep` first for the exact panel calendar.")

    entries: dict = {}
    for t in TICKERS:
        path = OUT_DIR / f"{t}.parquet"
        if path.exists() and not force:
            print(f"  {path.relative_to(REPO_ROOT)} exists, skipping (--force to refetch)")
            d = pd.read_parquet(path)
        else:
            d = fetch_one(t)
            d = add_publication_date(d, calendar)
            d = d[d["publication_date"].notna() & (d["publication_date"] <= pd.Timestamp(end))]
            d = d.reset_index(drop=True)
            d.to_parquet(path, index=False)

        entries[str(path.relative_to(REPO_ROOT))] = {
            "source": FINRA_URL,
            "sha256": _sha256(path),
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "rows": len(d),
            "settlement_start": str(d["settlement_date"].min())[:10],
            "settlement_end": str(d["settlement_date"].max())[:10],
            "publication_start": str(d["publication_date"].min())[:10],
            "publication_end": str(d["publication_date"].max())[:10],
            "max_settlement_gap_days": int(d["settlement_date"].diff().dt.days.max()),
        }
        e = entries[str(path.relative_to(REPO_ROOT))]
        print(f"  {path.relative_to(REPO_ROOT)}  ({e['rows']} rows, settle "
              f"{e['settlement_start']} -> {e['settlement_end']}, published from "
              f"{e['publication_start']}, max gap {e['max_settlement_gap_days']}d)")
    return entries


def check() -> int:
    """Validate the lag rule and report coverage. Returns a process exit code."""
    ok = True
    if PUBLICATION_LAG_DAYS <= MAX_OBSERVED_SCHEDULED_LAG_DAYS:
        print(f"  FAIL: PUBLICATION_LAG_DAYS={PUBLICATION_LAG_DAYS} does not clear the largest "
              f"observed FINRA schedule lag ({MAX_OBSERVED_SCHEDULED_LAG_DAYS}d)")
        ok = False
    else:
        print(f"  lag rule: {PUBLICATION_LAG_DAYS}d > {MAX_OBSERVED_SCHEDULED_LAG_DAYS}d "
              f"largest observed FINRA schedule lag  PASS")

    for t in TICKERS:
        path = OUT_DIR / f"{t}.parquet"
        if not path.exists():
            print(f"  FAIL: {path.relative_to(REPO_ROOT)} missing; run without --check first")
            ok = False
            continue
        d = pd.read_parquet(path)
        lag = (d["publication_date"] - d["settlement_date"]).dt.days
        if lag.min() < PUBLICATION_LAG_DAYS:
            print(f"  FAIL: {t} has a publication lag of {lag.min()}d, under the {PUBLICATION_LAG_DAYS}d rule")
            ok = False
        gap = int(d["settlement_date"].diff().dt.days.max())
        flag = "  (investigate)" if gap > 25 else ""
        print(f"  {t}: n={len(d)}, settle {str(d['settlement_date'].min())[:10]} -> "
              f"{str(d['settlement_date'].max())[:10]}, lag {lag.min()}-{lag.max()}d, "
              f"max gap {gap}d{flag}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--end", default=END, help=f"window end (default {END}, the committed vintage)")
    ap.add_argument("--force", action="store_true", help="refetch even if files exist")
    ap.add_argument("--check", action="store_true", help="validate coverage and the lag rule only")
    a = ap.parse_args()

    if a.check:
        return check()

    print(f"FINRA consolidated short interest -> {OUT_DIR.relative_to(REPO_ROOT)} (end {a.end})")
    entries = pull(a.end, a.force)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "end_pin": a.end,
        "publication_lag_days": PUBLICATION_LAG_DAYS,
        "max_observed_scheduled_lag_days": MAX_OBSERVED_SCHEDULED_LAG_DAYS,
        "versions": {"pandas": pd.__version__, "requests": requests.__version__},
        "files": entries,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {MANIFEST.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
