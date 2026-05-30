"""Pull risk-free rate series from FRED for IV inversion."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas_datareader.data as pdr
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "raw" / "fred"


def pull(series: list[str], start: str, end: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for code in series:
        df = pdr.DataReader(code, "fred", start, end).reset_index()
        df.columns = ["date", code.lower()]
        path = OUT_DIR / f"{code.lower()}.parquet"
        df.to_parquet(path, index=False)
        print(f"  {code} -> {path.relative_to(REPO_ROOT)} ({len(df)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    pull(cfg["fred"], cfg["start"], cfg["end"])


if __name__ == "__main__":
    main()
