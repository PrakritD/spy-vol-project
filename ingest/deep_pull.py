"""Fetch every free deep-history input behind STRATEGY.md and FINDINGS.md.

Writes exactly the files `analysis/strategy_two_sleeve.py::load_panel` and
`analysis/phase1_deep_history.py` read:

    data/raw/deep/SPY.parquet          yfinance daily OHLCV + adj_close
    data/raw/deep/VIXY.parquet         (plus VXX/SVXY/UVXY for cross-vehicle work)
    data/raw/deep/VIX3M.parquet        index closes; yfinance, CBOE CDN fallback
    data/raw/deep/VIX9D.parquet
    data/raw/deep/VVIX.parquet
    data/raw/fred/dgs3mo_deep.parquet  FRED DGS3MO (risk-free)
    data/raw/cboe_vix.csv              CBOE VIX_History.csv verbatim (1990->)
    data/raw/squeeze_dix.csv           SqueezeMetrics DIX/GEX (personal-use fetch;
                                       manual fallback printed on failure)

Every fetch lands in data/raw/deep_manifest.json (rows, dates, sha256, source, package
versions), so a result can always be reconciled to the data vintage that produced it.

The default window end is pinned to the vintage of the committed results
(strategy_results.json); pass --end to extend the sample, which will move the headline
numbers. Existing files are never overwritten unless --force, so the vintage behind
committed numbers cannot be silently replaced.

Run:  python -m ingest.deep_pull            # fetch whatever is missing
      python -m ingest.deep_pull --force    # refetch everything (new vintage)
      python -m ingest.deep_pull --check    # validate VIXY adjusted series vs VXX
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW = REPO_ROOT / "data" / "raw"
DEEP = RAW / "deep"
MANIFEST = RAW / "deep_manifest.json"

# Window start predates every series (sources clip to their own inception); the end is
# pinned to the vintage behind the committed strategy_results.json.
START = "1990-01-01"
END = "2026-05-31"

EQUITIES = ["SPY", "VIXY", "VXX", "SVXY", "UVXY"]
INDICES = {"^VIX3M": "VIX3M", "^VIX9D": "VIX9D", "^VVIX": "VVIX"}
# CBOE publishes full index histories on its CDN; used when yfinance comes back empty.
CBOE_CDN = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{name}_History.csv"
VIX_HISTORY_URL = CBOE_CDN.format(name="VIX")
DIX_URL = "https://squeezemetrics.com/monitor/static/DIX.csv"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}


@dataclass
class Args:
    end: str
    force: bool
    check: bool


# ------------------------------------------------------------------ helpers ----
def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(entries: dict, path: Path, source: str, df: pd.DataFrame | None) -> None:
    e = {"source": source, "sha256": _sha256(path),
         "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if df is not None and "date" in df.columns:
        e.update(rows=len(df), start=str(df["date"].min())[:10], end=str(df["date"].max())[:10])
    entries[str(path.relative_to(REPO_ROOT))] = e
    print(f"  {path.relative_to(REPO_ROOT)}  ({e.get('rows', '?')} rows, "
          f"{e.get('start', '?')} -> {e.get('end', '?')})")


def _flatten_yf(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    return df


def _clip(df: pd.DataFrame, end: str) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"])
    return df[df["date"] <= pd.Timestamp(end)].reset_index(drop=True)


# ------------------------------------------------------------------ fetchers ----
def pull_equities(end: str, force: bool, entries: dict) -> None:
    import yfinance as yf
    for t in EQUITIES:
        path = DEEP / f"{t}.parquet"
        if path.exists() and not force:
            print(f"  {path.relative_to(REPO_ROOT)} exists, skipping (--force to refetch)")
            continue
        df = yf.download(t, start=START, end=end, auto_adjust=False, progress=False)
        if df.empty:
            print(f"  [warn] {t} returned empty frame", file=sys.stderr)
            continue
        df = _clip(_flatten_yf(df), end)
        df.to_parquet(path, index=False)
        _record(entries, path, f"yfinance:{t}", df)


def pull_indices(end: str, force: bool, entries: dict) -> None:
    import yfinance as yf
    for ticker, name in INDICES.items():
        path = DEEP / f"{name}.parquet"
        if path.exists() and not force:
            print(f"  {path.relative_to(REPO_ROOT)} exists, skipping (--force to refetch)")
            continue
        df = yf.download(ticker, start=START, end=end, auto_adjust=False, progress=False)
        source = f"yfinance:{ticker}"
        if df.empty:
            url = CBOE_CDN.format(name=name)
            print(f"  [info] {ticker} empty on yfinance, falling back to {url}")
            r = requests.get(url, headers=UA, timeout=30)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text)).rename(columns=str.lower)
            df = df.rename(columns={"date": "date"})
            source = url
        else:
            df = _flatten_yf(df)
        df = _clip(df, end)
        df.to_parquet(path, index=False)
        _record(entries, path, source, df)


def pull_fred(end: str, force: bool, entries: dict) -> None:
    path = RAW / "fred" / "dgs3mo_deep.parquet"
    if path.exists() and not force:
        print(f"  {path.relative_to(REPO_ROOT)} exists, skipping (--force to refetch)")
        return
    import pandas_datareader.data as pdr
    df = pdr.DataReader("DGS3MO", "fred", START, end).reset_index()
    df.columns = ["date", "dgs3mo"]
    df = _clip(df, end)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    _record(entries, path, "fred:DGS3MO", df)


def pull_cboe_vix(end: str, force: bool, entries: dict) -> None:
    """CBOE VIX_History.csv saved verbatim (DATE/OPEN/HIGH/LOW/CLOSE, 1990->)."""
    path = RAW / "cboe_vix.csv"
    if path.exists() and not force:
        print(f"  {path.relative_to(REPO_ROOT)} exists, skipping (--force to refetch)")
        return
    r = requests.get(VIX_HISTORY_URL, headers=UA, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    if not {"DATE", "CLOSE"} <= set(df.columns):
        raise RuntimeError(f"unexpected VIX_History schema: {list(df.columns)}")
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df[df["DATE"] <= pd.Timestamp(end)]
    df.to_csv(path, index=False)
    _record(entries, path, VIX_HISTORY_URL,
            df.rename(columns={"DATE": "date"}))


def pull_squeeze(end: str, force: bool, entries: dict) -> None:
    """SqueezeMetrics DIX/GEX. Their terms bar *redistribution*, so this file is fetched
    for personal use and never committed (it is gitignored with the rest of data/)."""
    path = RAW / "squeeze_dix.csv"
    if path.exists() and not force:
        print(f"  {path.relative_to(REPO_ROOT)} exists, skipping (--force to refetch)")
        return
    try:
        r = requests.get(DIX_URL, headers=UA, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        if not {"date", "dix", "gex"} <= set(df.columns):
            raise RuntimeError(f"unexpected DIX.csv schema: {list(df.columns)}")
    except Exception as exc:  # noqa: BLE001 - any failure gets the manual instructions
        print(f"  [warn] SqueezeMetrics fetch failed ({exc}).\n"
              f"         Manual fallback: download the DIX csv from "
              f"https://squeezemetrics.com/monitor/dix (Download button) and save it as "
              f"data/raw/squeeze_dix.csv (columns: date,price,dix,gex).", file=sys.stderr)
        return
    df = _clip(df, end)
    df.to_csv(path, index=False)
    _record(entries, path, DIX_URL, df)


# ------------------------------------------------------------------ validation ----
def check_vixy() -> int:
    """Cross-validate VIXY's adjusted series against VXX. yfinance mishandling any of
    VIXY's ~8 reverse splits would show up as (a) broken daily-return correlation with
    VXX (both roll the same SPVXSTR-style front-two-month exposure) and (b) isolated
    +-60..80% one-day 'returns' at split dates."""
    vixy = pd.read_parquet(DEEP / "VIXY.parquet")[["date", "adj_close"]]
    vxx = pd.read_parquet(DEEP / "VXX.parquet")[["date", "adj_close"]]
    m = vixy.merge(vxx, on="date", suffixes=("_vixy", "_vxx")).set_index("date")
    r = m.pct_change().dropna()
    corr = r["adj_close_vixy"].corr(r["adj_close_vxx"])
    gap = (r["adj_close_vixy"] - r["adj_close_vxx"]).abs()
    big = r["adj_close_vixy"].abs()
    years = len(r) / 252
    decay = (m["adj_close_vixy"].iloc[-1] / m["adj_close_vixy"].iloc[0]) ** (1 / years) - 1

    print(f"VIXY-vs-VXX daily-return corr: {corr:.4f}  (expect > 0.98)")
    print(f"VIXY annualised decay: {decay:+.1%}  (expect roughly -40%..-55%)")
    print(f"days with |VIXY ret| > 25%: {int((big > 0.25).sum())}  "
          f"(expect ~2: Feb-2018, Mar-2020 area; a split artifact would add +-60..80% days)")
    print("largest |VIXY - VXX| daily-return gaps:")
    print(gap.nlargest(5).to_string())

    ok = corr > 0.98 and (big > 0.55).sum() == 0
    print("PASS" if ok else "FAIL: inspect the dates above against VIXY's split history")
    return 0 if ok else 1


# ------------------------------------------------------------------ main ----
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--end", default=END,
                    help=f"window end (default {END}, the committed-results vintage)")
    ap.add_argument("--force", action="store_true", help="refetch even if files exist")
    ap.add_argument("--check", action="store_true",
                    help="validate the VIXY adjusted series against VXX and exit")
    a = ap.parse_args()

    if a.check:
        return check_vixy()

    DEEP.mkdir(parents=True, exist_ok=True)
    entries = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    entries.setdefault("_meta", {})
    import yfinance
    entries["_meta"].update(window_start=START, window_end=a.end,
                            yfinance=yfinance.__version__, pandas=pd.__version__)

    pull_equities(a.end, a.force, entries)
    pull_indices(a.end, a.force, entries)
    pull_fred(a.end, a.force, entries)
    pull_cboe_vix(a.end, a.force, entries)
    pull_squeeze(a.end, a.force, entries)

    MANIFEST.write_text(json.dumps(entries, indent=2, sort_keys=True) + "\n")
    print(f"manifest -> {MANIFEST.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
