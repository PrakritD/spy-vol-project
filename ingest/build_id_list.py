"""Build a per-day SPY ATM-band instrument-id list from OPRA definition data.

Stage 2 of the two-stage Databento pull. Reads downloaded `definition` DBN
files, joins against daily SPY spot price (free yfinance), filters to live
contracts with strike within ±band of spot AND DTE in [lo, hi] days, and
emits one parquet per quarter (so each statistics batch job stays under
Databento's per-request id-list size limits).

Output:
    data/interim/spy_atm_id_list.parquet   — flat: date, instrument_id
    data/interim/quarterly_id_lists/{Q}.json — per-quarter unique-id lists
                                                for batch submission.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEF_DIR = REPO_ROOT / "data" / "raw" / "opra_definition"
SPY_PARQUET = REPO_ROOT / "data" / "raw" / "yfinance" / "SPY.parquet"
INTERIM_DIR = REPO_ROOT / "data" / "interim"
ID_PANEL_PATH = INTERIM_DIR / "spy_atm_id_list.parquet"
CHUNK_DIR = INTERIM_DIR / "id_list_chunks"
MAX_IDS_PER_REQUEST = 1900   # Databento hard cap is 2000; leave a safety margin


@dataclass(frozen=True)
class IdListConfig:
    moneyness_band: float = 0.20      # ±20 % of spot
    dte_lo_days: int = 7
    dte_hi_days: int = 60
    start_date: str = "2024-08-01"    # earliest file_date kept in the panel —
                                      # lands the monthly-only pull at ~$95,
                                      # inside the free Databento credit
    monthly_only: bool = True         # keep only standard 3rd-Friday monthlies
                                      # — drops weeklies/0DTE to fit the pull
                                      # inside the free Databento credit


_DEF_FILE_DATE_RE = re.compile(r"(\d{8})\.definition")


def _load_definition_by_file_date() -> pd.DataFrame:
    """Read all DBN definition files keyed by file-date.

    OPRA's instrument_id mapping is date-scoped. An id is only valid in
    OPRA.PILLAR for the dates it actually appears in a definition snapshot.
    This loader returns one row per (file_date, instrument_id) so downstream
    chunking can request statistics over the id's *actual* valid window.

    Returned columns:
        file_date, instrument_id, strike, expiry, option_type
    """
    try:
        import databento as db
    except ImportError as e:
        raise ImportError("databento package required to read DBN files") from e

    files = sorted(DEF_DIR.rglob("*.dbn.zst"))
    if not files:
        raise FileNotFoundError(
            f"no DBN definition files in {DEF_DIR} — pull OPRA definition first"
        )
    frames = []
    for f in files:
        m = _DEF_FILE_DATE_RE.search(f.name)
        if not m:
            continue
        file_date = pd.Timestamp(m.group(1))    # 'YYYYMMDD'
        df = db.DBNStore.from_file(f).to_df()
        keep = pd.DataFrame({
            "file_date": file_date,
            "instrument_id": df["instrument_id"].astype("int64"),
            "strike": df["strike_price"].astype("float64"),
            "expiry": pd.to_datetime(df["expiration"], utc=True).dt.tz_convert(None).dt.normalize(),
            "option_type": df["instrument_class"].astype(str).str[0].str.upper(),
        })
        # Within one file the same id can appear multiple times (e.g. action
        # updates); keep the first occurrence — fields are static.
        keep = keep.drop_duplicates(["file_date", "instrument_id"])
        frames.append(keep)
    return pd.concat(frames, ignore_index=True)


def _load_spy_spot() -> pd.Series:
    df = pd.read_parquet(SPY_PARQUET)[["date", "close"]]
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df.set_index("date")["close"].astype(float)


def build(cfg: IdListConfig) -> pd.DataFrame:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    # Wipe stale chunks so re-runs don't mix old + new files.
    if CHUNK_DIR.exists():
        for p in CHUNK_DIR.glob("*.json"):
            p.unlink()
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)

    contracts_by_day = _load_definition_by_file_date()
    spy_spot = _load_spy_spot()

    # Each definition file is an EOD snapshot of valid SPY contracts on
    # that date. An id is only valid in OPRA's symbology for the dates it
    # appears in a snapshot — so we filter strictly per file_date.
    merged = contracts_by_day.merge(
        spy_spot.rename("spot").to_frame(),
        left_on="file_date",
        right_index=True,
        how="inner",
    )
    merged["dte_days"] = (merged["expiry"] - merged["file_date"]).dt.days
    start_ts = pd.Timestamp(cfg.start_date)
    in_band = (
        merged["strike"].between(merged["spot"] * (1 - cfg.moneyness_band),
                                 merged["spot"] * (1 + cfg.moneyness_band))
        & merged["dte_days"].between(cfg.dte_lo_days, cfg.dte_hi_days)
        & (merged["expiry"] >= merged["file_date"])
        & (merged["file_date"] >= start_ts)
    )
    if cfg.monthly_only:
        # Standard monthly options expire the 3rd Friday — weekday==Fri and
        # day-of-month in [15, 21]. This is where dealer OI concentrates;
        # dropping weeklies/0DTE cuts the universe ~80% to fit the budget.
        is_third_friday = (
            (merged["expiry"].dt.weekday == 4)
            & merged["expiry"].dt.day.between(15, 21)
        )
        in_band = in_band & is_third_friday
    panel = (merged.loc[in_band, ["file_date", "instrument_id"]]
                    .rename(columns={"file_date": "date"})
                    .drop_duplicates()
                    .sort_values(["date", "instrument_id"])
                    .reset_index(drop=True))
    panel["date"] = pd.to_datetime(panel["date"])
    panel.to_parquet(ID_PANEL_PATH, index=False)

    # Monthly id chunks for stage-2 batch submission. Databento caps each
    # request at 2000 symbols; chunk any month over MAX_IDS_PER_REQUEST.
    # Each chunk's date range is tightened to the *actual* span when its ids
    # appear in the panel — avoids `symbology_invalid_request` errors when a
    # chunk's ids only overlap a sliver of the calendar month.
    panel["month"] = panel["date"].dt.to_period("M").astype(str)
    n_chunks = 0
    for month, sub in panel.groupby("month"):
        ids = sorted(set(sub["instrument_id"].astype(int).tolist()))
        for chunk_idx, start_i in enumerate(range(0, len(ids), MAX_IDS_PER_REQUEST)):
            chunk = ids[start_i:start_i + MAX_IDS_PER_REQUEST]
            chunk_set = set(chunk)
            chunk_dates = sub.loc[sub["instrument_id"].isin(chunk_set), "date"]
            if chunk_dates.empty:
                continue
            start = chunk_dates.min().date().isoformat()
            # Databento batch API treats `end` as exclusive at day-precision —
            # `start == end` raises 422 data_time_range_start_on_or_after_end,
            # and `end = chunk_dates.max()` silently drops the last day's data.
            end = (chunk_dates.max() + pd.Timedelta(days=1)).date().isoformat()
            label = f"{month}_p{chunk_idx:02d}" if len(ids) > MAX_IDS_PER_REQUEST else month
            (CHUNK_DIR / f"{label}.json").write_text(json.dumps({
                "label": label,
                "month": month,
                "chunk_idx": chunk_idx,
                "start": start,
                "end": end,
                "n_ids": len(chunk),
                "instrument_ids": chunk,
            }, indent=2))
            n_chunks += 1
    print(f"wrote {len(panel)} (date, instrument_id) rows -> {ID_PANEL_PATH.relative_to(REPO_ROOT)}")
    print(f"wrote {n_chunks} chunk files -> {CHUNK_DIR.relative_to(REPO_ROOT)}")
    return panel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "features.yaml")
    args = ap.parse_args()
    cfg_yaml = yaml.safe_load(args.config.read_text())
    gex = cfg_yaml.get("gex", {})
    cfg = IdListConfig(
        moneyness_band=0.20,                    # not in features yaml; default sane
        dte_lo_days=gex.get("dte_filter", [7, 60])[0],
        dte_hi_days=gex.get("dte_filter", [7, 60])[1],
    )
    build(cfg)


if __name__ == "__main__":
    main()
