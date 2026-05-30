"""Pull daily OHLC for VIX family, SPY, VXX via yfinance.

Writes one parquet per ticker into data/raw/yfinance/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml
import yfinance as yf


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "raw" / "yfinance"


def pull(tickers: list[str], start: str, end: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for t in tickers:
        df = yf.download(t, start=start, end=end, auto_adjust=False, progress=False)
        if df.empty:
            print(f"  [warn] {t} returned empty frame", file=sys.stderr)
            continue
        # Modern yfinance returns a MultiIndex on columns even for one ticker:
        #   [('Open','SPY'), ('High','SPY'), ...]. Flatten by taking level 0.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
        safe = t.lstrip("^").replace("/", "_")
        path = OUT_DIR / f"{safe}.parquet"
        df.to_parquet(path, index=False)
        print(f"  {t} -> {path.relative_to(REPO_ROOT)} ({len(df)} rows, "
              f"cols={list(df.columns)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    pull(cfg["yfinance"], cfg["start"], cfg["end"])


if __name__ == "__main__":
    main()
