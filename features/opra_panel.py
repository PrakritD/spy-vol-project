"""Preprocess OPRA statistics + definition DBN into a daily contract panel.

Reads:
    data/raw/opra_statistics_*/full/*.dbn.zst   (stage 2 download)
    data/raw/opra_definition/**/*.dbn.zst       (stage 1 download)
    data/raw/yfinance/SPY.parquet               (daily SPY close as spot)
    data/raw/fred/dgs3mo.parquet                (risk-free rate, %)

Output:
    data/processed/options_panel.parquet
    columns: date, instrument_id, strike, expiry, option_type,
             open_interest, price, spot, r, q

OPRA stat_types used (empirically verified from a sample DBN):
    9  = OpenInterest  (value in `quantity`, not `price`)
    11 = ClosePrice    (value in `price`)

Date convention: use the filename date as the trading date the snapshot
describes — matches `build_id_list.py`'s file_date convention. Downstream
`features/assemble.py` applies shift(1) on the GEX series before joining
to the panel so the model never sees same-day GEX.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
DEF_DIR = RAW_DIR / "opra_definition"
SPY_PATH = RAW_DIR / "yfinance" / "SPY.parquet"
FRED_PATH = RAW_DIR / "fred" / "dgs3mo.parquet"
OUT_PATH = REPO_ROOT / "data" / "processed" / "options_panel.parquet"

STAT_OPEN_INTEREST = 9
STAT_CLOSE_PRICE = 11
SPY_DIV_YIELD = 0.015   # constant; SPY trailing yield typically 1.3-1.6%

_DEF_FILE_DATE = re.compile(r"(\d{8})\.definition")
_STATS_FILE_DATE = re.compile(r"(\d{8})\.statistics")


def _load_definitions() -> pd.DataFrame:
    import databento as db
    files = sorted(DEF_DIR.rglob("*.dbn.zst"))
    if not files:
        raise FileNotFoundError(f"no definition files in {DEF_DIR}")
    rows = []
    for f in files:
        m = _DEF_FILE_DATE.search(f.name)
        if not m:
            continue
        fdate = pd.Timestamp(m.group(1))
        df = db.DBNStore.from_file(f).to_df()
        rows.append(pd.DataFrame({
            "file_date": fdate,
            "instrument_id": df["instrument_id"].astype("int64"),
            "strike": df["strike_price"].astype("float64"),
            "expiry": pd.to_datetime(df["expiration"], utc=True)
                        .dt.tz_convert(None).dt.normalize(),
            "option_type": df["instrument_class"].astype(str).str[0].str.upper(),
        }))
    return (pd.concat(rows, ignore_index=True)
              .drop_duplicates(["file_date", "instrument_id"]))


def _load_statistics() -> pd.DataFrame:
    import databento as db
    stat_dirs = sorted(RAW_DIR.glob("opra_statistics_*/full"))
    if not stat_dirs:
        raise FileNotFoundError("no stage-2 statistics dirs in data/raw/")
    frames = []
    for d in stat_dirs:
        for f in sorted(d.rglob("*.dbn.zst")):
            m = _STATS_FILE_DATE.search(f.name)
            if not m:
                continue
            fdate = pd.Timestamp(m.group(1))
            df = db.DBNStore.from_file(f).to_df()
            # OPRA publishes multiple OI / close updates per (day, instrument)
            # (initial publish, corrections, late updates). Keep the latest
            # ts_event per instrument so a contract-day has exactly one row.
            df = df.sort_values("ts_event")
            oi = (df.loc[df["stat_type"] == STAT_OPEN_INTEREST,
                         ["instrument_id", "quantity"]]
                    .rename(columns={"quantity": "open_interest"})
                    .drop_duplicates("instrument_id", keep="last"))
            px = (df.loc[df["stat_type"] == STAT_CLOSE_PRICE,
                         ["instrument_id", "price"]]
                    .drop_duplicates("instrument_id", keep="last"))
            px = px.loc[px["price"].notna() & (px["price"] > 0)]
            merged = oi.merge(px, on="instrument_id", how="inner")
            merged["date"] = fdate
            frames.append(merged)
    return pd.concat(frames, ignore_index=True)


def _load_spot_and_rate() -> tuple[pd.Series, pd.Series]:
    spy = pd.read_parquet(SPY_PATH)[["date", "close"]]
    spy["date"] = pd.to_datetime(spy["date"]).dt.normalize()
    spot = spy.set_index("date")["close"].astype(float)

    fred = pd.read_parquet(FRED_PATH)
    fred["date"] = pd.to_datetime(fred["date"]).dt.normalize()
    r = (fred.set_index("date")["dgs3mo"].astype(float) / 100.0)
    # ffill across weekends/holidays so any trading day picks up the nearest rate
    r = r.asfreq("D").ffill().bfill()
    return spot, r


def build() -> pd.DataFrame:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("loading definitions…")
    defs = _load_definitions()
    print(f"  {len(defs):,} (file_date, instrument_id) rows")

    print("loading statistics…")
    stats = _load_statistics()
    print(f"  {len(stats):,} (date, instrument_id) rows")

    print("loading spot + rate…")
    spot, r = _load_spot_and_rate()

    print("joining…")
    panel = stats.merge(
        defs,
        left_on=["date", "instrument_id"],
        right_on=["file_date", "instrument_id"],
        how="inner",
    ).drop(columns=["file_date"])
    panel["spot"] = panel["date"].map(spot)
    panel["r"] = panel["date"].map(r)
    panel["q"] = SPY_DIV_YIELD
    # drop rows where SPY didn't trade (e.g. an OPRA file dated to a non-trading
    # day surfaces); also drop nan rates from FRED gaps the ffill missed
    panel = panel.dropna(subset=["spot", "r"]).rename(columns={"price": "price"})
    panel = panel[["date", "instrument_id", "strike", "expiry", "option_type",
                   "open_interest", "price", "spot", "r", "q"]]
    panel.to_parquet(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH.relative_to(REPO_ROOT)} "
          f"({len(panel):,} contract-day rows, "
          f"{panel['date'].nunique()} trading days, "
          f"{panel['instrument_id'].nunique()} unique contracts)")
    return panel


def main():
    build()


if __name__ == "__main__":
    main()
