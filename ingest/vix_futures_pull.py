"""Fetch CBOE's free per-contract VIX futures (VX) daily settlement archive.

CBOE's site (cboe.com/us/futures/market_statistics/historical_data) advertises this CDN
archive as "2013 to Current." That is stale marketing copy. Verified live 2026-07-22 by
probing every contract code F04..Z20 and cross-checking settle-at-near-expiry against the
committed spot VIX series (`data/raw/cboe_vix.csv`):

- Contract-months 2006-01..2013-12 are gap-free (all 12 months every year).
- 2014-2018 only has the front 8 months (Jan-Aug, codes F-Q) of each year; Sep-Dec
  contracts were never populated under this URL scheme.
- Every contract still open on 2018-02-23 (H18 / Mar-2018 onward) is truncated to that
  date; nothing exists after it. The whole archive looks abandoned shortly after
  Volmageddon (2018-02-05).
- 2004-2006 settle prices run ~10x the correctly-scaled series (near-expiry settle should
  track spot VIX; the ratio is ~10 for 2004-2006, transitions mid-2007, and is ~1 from
  2008 on). Not corrected here -- flagged so nobody trusts pre-2008 levels at face value.

Net effect: the only stretch that is simultaneously gap-free, full-year, and correctly
scaled is **2008-01 through 2013-12**. `analysis/vix_futures_curve.py` restricts its
constant-maturity construction to that window. Continuing coverage past 2018 needs a paid
source (CBOE DataShop or Databento's CFE feed) -- out of scope while the project's budget
is $0. This module fetches every contract that does exist and records the true per-contract
coverage in the manifest so nobody re-discovers any of the above by trial and error.

Per-contract file schema (verbatim from CBOE): trade_date, futures (month label), open,
high, low, close, settle, change, total_volume, efp, open_interest. No adjustment is
applied; settle is the field used everywhere downstream.

Run: python -m ingest.vix_futures_pull            # fetch every available contract
     python -m ingest.vix_futures_pull --force     # refetch everything
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
CONTRACTS_DIR = RAW / "vix_futures" / "contracts"
PANEL_PATH = RAW / "vix_futures" / "vix_futures_panel.parquet"
MANIFEST = RAW / "vix_futures_manifest.json"

CDN_URL = "https://cdn.cboe.com/resources/futures/archive/volume-and-price/CFE_{code}_VX.csv"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}

MONTH_CODES = "FGHJKMNQUVXZ"  # Jan..Dec futures month letters
MONTH_NUM = {c: i + 1 for i, c in enumerate(MONTH_CODES)}
# Probe range: VX launched 2004-03-26; the archive's last real data is the Q18 (Aug-2018)
# contract. Probing a couple of years past the known cutoff costs nothing and guards
# against the boundary silently moving if CBOE ever backfills further.
YEARS = range(2004, 2021)


@dataclass
class Args:
    force: bool
    probe_only: bool


def _all_codes() -> list[str]:
    return [f"{m}{y % 100:02d}" for y in YEARS for m in MONTH_CODES]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fetch_contract(code: str) -> pd.DataFrame | None:
    url = CDN_URL.format(code=code)
    r = requests.get(url, headers=UA, timeout=30)
    if r.status_code != 200 or len(r.text) < 100:
        return None
    lines = r.text.splitlines()
    # Some files prepend a disclaimer paragraph before the real header row; find it.
    header_idx = next((i for i, ln in enumerate(lines) if ln.startswith("Trade Date")), None)
    if header_idx is None:
        return None
    # Some rows carry a stray trailing comma (extra empty field); strip it so every row
    # tokenizes to the same 11 columns.
    text = "\n".join(ln.rstrip(",") for ln in lines[header_idx:])
    df = pd.read_csv(io.StringIO(text))
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df = df.rename(columns={"trade_date": "trade_date", "futures": "label"})
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    month_letter, yy = code[0], int(code[1:])
    year = 2000 + yy
    df.insert(0, "contract_code", code)
    df.insert(1, "contract_month", MONTH_NUM[month_letter])
    df.insert(2, "contract_year", year)
    # The archive's own last trade-date row is each contract's final settlement day
    # (verified against CBOE VIX_History spot on that date; see vix_futures_curve.py).
    df["expiry_date"] = df["trade_date"].max()
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--force", action="store_true", help="refetch even if files exist")
    ap.add_argument("--probe-only", action="store_true",
                     help="print which contract codes exist, fetch nothing")
    a = ap.parse_args()

    codes = _all_codes()

    if a.probe_only:
        found = []
        for code in codes:
            if _fetch_contract(code) is not None:
                found.append(code)
        print(f"{len(found)}/{len(codes)} contracts available")
        print(found)
        return 0

    CONTRACTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    manifest.setdefault("_meta", {})
    manifest["_meta"].update(
        fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        source_url_template=CDN_URL,
        note=("Free CBOE per-contract archive. Gap-free 2006-01..2013-12; only Jan-Aug "
              "contracts present 2014-2018; every contract still open on 2018-02-23 is "
              "truncated there and nothing exists after; 2004-2006 settle prices run "
              "~10x the correctly-scaled series (not corrected). The only stretch that "
              "is gap-free, full-year, AND correctly scaled is 2008-01..2013-12 -- see "
              "module docstring and analysis/vix_futures_curve.py."),
        pandas=pd.__version__,
    )

    frames = []
    available_codes = []
    for code in codes:
        path = CONTRACTS_DIR / f"{code}.parquet"
        if path.exists() and not a.force:
            df = pd.read_parquet(path)
            available_codes.append(code)
            frames.append(df)
            continue
        df = _fetch_contract(code)
        if df is None:
            continue
        df.to_parquet(path, index=False)
        available_codes.append(code)
        frames.append(df)
        manifest[code] = {
            "sha256": _sha256(path),
            "rows": len(df),
            "start": str(df["trade_date"].min())[:10],
            "end": str(df["trade_date"].max())[:10],
        }
        print(f"  {code}: {len(df)} rows, {df['trade_date'].min():%Y-%m-%d} -> "
              f"{df['trade_date'].max():%Y-%m-%d}")

    if not frames:
        print("[warn] no contracts fetched or found on disk", file=sys.stderr)
        return 1

    panel = pd.concat(frames, ignore_index=True).sort_values(
        ["expiry_date", "trade_date"]).reset_index(drop=True)
    panel.to_parquet(PANEL_PATH, index=False)
    manifest["_meta"]["contracts_available"] = available_codes
    manifest["_meta"]["panel_rows"] = len(panel)
    manifest["_meta"]["panel_path"] = str(PANEL_PATH.relative_to(REPO_ROOT))
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    print(f"\n{len(available_codes)}/{len(codes)} contracts, panel rows={len(panel)}")
    print(f"manifest -> {MANIFEST.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
