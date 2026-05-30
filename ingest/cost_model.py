"""Estimate stage-2 OPRA statistics cost under tighter universe filters.

WARNING — id-day scaling under-estimates band/dte tightening.
The SCAN modes below project cost as

    proj_$_M = (current_$_M / current_id_days_M) * candidate_id_days_M

i.e. they assume statistics-record density per (id, day) is constant under
filtering. It is NOT. Tightening the moneyness band or DTE window removes
the wing / long-dated contracts, which have the *lowest* record density;
what remains (ATM, short-dated) is ~2x denser than the global average.
Measured: band=±20% dte=[7,60] → $0.000164/id-day; band=±10% dte=[7,30] →
$0.000320/id-day. So SCAN projections for tighter filters read LOW — a
±10%/[7,30] config projected at $99 actually quoted at $170.

Trust the scans only for *relative* ranking of band/dte knobs. For an
absolute number, run `databento_pull --quote-stage2` against real chunks.
The --from-quote mode below parses a real quote output and is accurate.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ingest.build_id_list import (
    _load_definition_by_file_date,
    _load_spy_spot,
)


# Per-month $ from the most recent --quote-stage2 output, summed across
# all sub-chunks of that month. Two chunks errored in that quote run
# (2023-05_p06 was the start==end bug; 2024-09_p02 was a transient HTTPS
# blip) — their actual cost is small (<$1) and absorbed in noise.
QUOTE_PER_MONTH_USD: dict[str, float] = {
    "2023-01": 0.06,  "2023-02": 0.06,  "2023-03": 0.82,
    "2023-04": 7.12,  "2023-05": 8.75,  "2023-06": 7.57,
    "2023-07": 7.56,  "2023-08": 8.67,  "2023-09": 7.21,
    "2023-10": 8.84,  "2023-11": 7.33,  "2023-12": 6.76,
    "2024-01": 7.16,  "2024-02": 6.24,  "2024-03": 6.65,
    "2024-04": 7.36,  "2024-05": 7.25,  "2024-06": 6.85,
    "2024-07": 9.58,  "2024-08": 10.53, "2024-09": 8.03,
    "2024-10": 10.60, "2024-11": 9.42,  "2024-12": 9.68,
    "2025-01": 8.89,  "2025-02": 8.62,  "2025-03": 10.77,
    "2025-04": 11.42, "2025-05": 9.83,  "2025-06": 8.59,
    "2025-07": 9.49,  "2025-08": 9.08,  "2025-09": 10.17,
    "2025-10": 10.86, "2025-11": 9.47,  "2025-12": 10.74,
    "2026-01": 10.65, "2026-02": 11.07, "2026-03": 13.98,
    "2026-04": 12.22,
}


@dataclass(frozen=True)
class FilterSpec:
    band: float = 0.20
    dte_lo: int = 7
    dte_hi: int = 60
    start: str = "2023-01-01"
    end: str = "2026-05-01"

    def label(self) -> str:
        return (f"band=±{self.band*100:.1f}% "
                f"dte=[{self.dte_lo},{self.dte_hi}] "
                f"{self.start[:7]}→{self.end[:7]}")


def _build_panel(contracts: pd.DataFrame, spy_spot: pd.Series,
                 spec: FilterSpec) -> pd.DataFrame:
    """Reproduces build_id_list filter, returning (date, instrument_id) rows."""
    merged = contracts.merge(
        spy_spot.rename("spot").to_frame(),
        left_on="file_date", right_index=True, how="inner",
    )
    merged["dte_days"] = (merged["expiry"] - merged["file_date"]).dt.days
    in_band = (
        merged["strike"].between(
            merged["spot"] * (1 - spec.band),
            merged["spot"] * (1 + spec.band),
        )
        & merged["dte_days"].between(spec.dte_lo, spec.dte_hi)
        & (merged["expiry"] >= merged["file_date"])
    )
    panel = (merged.loc[in_band, ["file_date", "instrument_id"]]
                    .rename(columns={"file_date": "date"})
                    .drop_duplicates())
    start_ts = pd.Timestamp(spec.start)
    end_ts = pd.Timestamp(spec.end)
    panel = panel.loc[(panel["date"] >= start_ts) & (panel["date"] < end_ts)]
    return panel


def _by_month(panel: pd.DataFrame) -> pd.Series:
    return panel.assign(month=panel["date"].dt.to_period("M").astype(str)) \
                .groupby("month").size()


def project(contracts: pd.DataFrame, spy_spot: pd.Series,
            current: FilterSpec, candidate: FilterSpec) -> float:
    """Project candidate cost from current per-month $ and id-day deltas."""
    cur_panel = _build_panel(contracts, spy_spot, current)
    cand_panel = _build_panel(contracts, spy_spot, candidate)
    cur_by_m = _by_month(cur_panel)
    cand_by_m = _by_month(cand_panel)
    total = 0.0
    for month, cand_count in cand_by_m.items():
        cur_count = cur_by_m.get(month, 0)
        cost = QUOTE_PER_MONTH_USD.get(month, 0.0)
        if cur_count == 0:
            continue
        total += cost * (cand_count / cur_count)
    return total


def per_month_breakdown(contracts: pd.DataFrame, spy_spot: pd.Series,
                         current: FilterSpec, candidate: FilterSpec) -> pd.DataFrame:
    """Per-month projected $ for one candidate, with id-day deltas."""
    cur_panel = _build_panel(contracts, spy_spot, current)
    cand_panel = _build_panel(contracts, spy_spot, candidate)
    cur_by_m = _by_month(cur_panel)
    cand_by_m = _by_month(cand_panel)
    rows = []
    for month in sorted(set(cand_by_m.index) | set(cur_by_m.index)):
        cur_c = cur_by_m.get(month, 0)
        cand_c = cand_by_m.get(month, 0)
        cur_cost = QUOTE_PER_MONTH_USD.get(month, 0.0)
        if cur_c == 0 or cand_c == 0:
            proj = 0.0
        else:
            proj = cur_cost * (cand_c / cur_c)
        rows.append({"month": month, "cur_id_days": cur_c, "cand_id_days": cand_c,
                     "cur_usd": cur_cost, "proj_usd": proj})
    return pd.DataFrame(rows)


def from_quote(quote_path: Path, train_months: int = 12,
               data_end: str = "2026-04-30") -> None:
    """Accurate per-month + cumulative cost parsed from a real quote output.

    Pipe `databento_pull --quote-stage2 ... > quote.txt` then point this at it.
    Unlike the SCAN modes, this uses Databento's actual per-chunk dollars, so
    the cumulative-from-start-date column is trustworthy for date-cut decisions
    *within the band/dte universe the chunks were built for*.
    """
    import datetime as dt
    import re

    # Chunk label is "{month}" for single-chunk months, "{month}_pNN" otherwise.
    pat = re.compile(r"opra_statistics_(\d{4}-\d{2})(?:_p\d+)?\s+\S+\s+\S+\s+([0-9.]+)\s*$")
    by_month: dict[str, float] = {}
    n = 0
    for ln in quote_path.read_text().splitlines():
        m = pat.match(ln.strip())
        if m:
            by_month[m.group(1)] = by_month.get(m.group(1), 0.0) + float(m.group(2))
            n += 1
    months = sorted(by_month)
    total = sum(by_month.values())
    end = dt.date.fromisoformat(data_end)
    print(f"parsed {n} chunks from {quote_path}, total ${total:.2f}\n")
    print(f"{'start':<9}{'month $':>9}{'cum cost':>11}{'OOS test mo':>13}")
    print("-" * 42)
    for i, mth in enumerate(months):
        cum = sum(by_month[m] for m in months[i:])
        y, mo = map(int, mth.split("-"))
        fp = y * 12 + (mo - 1) + train_months
        test_mo = (end.year - fp // 12) * 12 + (end.month - (fp % 12 + 1)) + 1
        flag = ""
        if cum <= 100:
            flag += " ≤$100"
        if test_mo >= 6:
            flag += " test≥6mo"
        print(f"{mth:<9}{by_month[mth]:>9.2f}{cum:>11.2f}{test_mo:>13d}{flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--current-cost", type=float, default=336.0,
                    help="Sanity check: current quote total.")
    ap.add_argument("--breakdown", action="store_true",
                    help="Print per-month breakdown for two candidate combos.")
    ap.add_argument("--from-quote", type=Path,
                    help="Parse a real --quote-stage2 output file for accurate "
                         "per-month + cumulative-from-start cost. Skips the scans.")
    args = ap.parse_args()

    if args.from_quote:
        from_quote(args.from_quote)
        return

    print("loading definitions + spot... (~1-2 min)")
    contracts = _load_definition_by_file_date()
    spot = _load_spy_spot()
    current = FilterSpec()

    print(f"\ncurrent filter: {current.label()}")
    cur_panel = _build_panel(contracts, spot, current)
    print(f"  id-day rows: {len(cur_panel):,}   "
          f"projected $ (sanity): ${project(contracts, spot, current, current):.2f}   "
          f"actual quote: ${args.current_cost:.2f}\n")

    print("=" * 78)
    print("SCAN 1 — vary moneyness band (dte=[7,60], full timeframe)")
    print("=" * 78)
    print(f"{'band':>8}  {'id-days':>12}  {'projected $':>12}  Δ from current")
    print("-" * 78)
    for band in [0.20, 0.175, 0.15, 0.125, 0.10, 0.075, 0.05]:
        spec = FilterSpec(band=band)
        panel = _build_panel(contracts, spot, spec)
        cost = project(contracts, spot, current, spec)
        delta = cost - args.current_cost
        print(f"  ±{band*100:5.1f}%  {len(panel):>12,}  ${cost:>10.2f}    {delta:+8.2f}")

    print()
    print("=" * 78)
    print("SCAN 2 — vary dte_hi (band=±20%, full timeframe)")
    print("=" * 78)
    print(f"{'dte_hi':>8}  {'id-days':>12}  {'projected $':>12}  Δ from current")
    print("-" * 78)
    for dte_hi in [60, 50, 45, 35, 30, 21, 14]:
        spec = FilterSpec(dte_hi=dte_hi)
        panel = _build_panel(contracts, spot, spec)
        cost = project(contracts, spot, current, spec)
        delta = cost - args.current_cost
        print(f"  [7,{dte_hi:2d}]  {len(panel):>12,}  ${cost:>10.2f}    {delta:+8.2f}")

    print()
    print("=" * 78)
    print("SCAN 3 — vary start_date (band=±20%, dte=[7,60])")
    print("=" * 78)
    print(f"{'start':>12}  {'id-days':>12}  {'projected $':>12}  Δ from current")
    print("-" * 78)
    for start in ["2023-01-01", "2023-04-01", "2023-07-01", "2024-01-01",
                  "2024-07-01", "2025-01-01"]:
        spec = FilterSpec(start=start)
        panel = _build_panel(contracts, spot, spec)
        cost = project(contracts, spot, current, spec)
        delta = cost - args.current_cost
        print(f"  {start}  {len(panel):>12,}  ${cost:>10.2f}    {delta:+8.2f}")

    print()
    print("=" * 78)
    print("SCAN 4 — combined targets aimed at ~$100")
    print("=" * 78)
    targets = [
        FilterSpec(band=0.10, dte_hi=45, start="2023-01-01"),
        FilterSpec(band=0.10, dte_hi=45, start="2023-07-01"),
        FilterSpec(band=0.10, dte_hi=30, start="2023-01-01"),
        FilterSpec(band=0.10, dte_hi=30, start="2023-07-01"),
        FilterSpec(band=0.10, dte_hi=30, start="2024-01-01"),
        FilterSpec(band=0.075, dte_hi=45, start="2023-07-01"),
        FilterSpec(band=0.075, dte_hi=30, start="2023-01-01"),
        FilterSpec(band=0.05, dte_hi=45, start="2023-01-01"),
        FilterSpec(band=0.05, dte_hi=30, start="2023-01-01"),
    ]
    print(f"  {'filter':<55} {'id-days':>12}  {'$ proj':>9}")
    print("-" * 78)
    for spec in targets:
        panel = _build_panel(contracts, spot, spec)
        cost = project(contracts, spot, current, spec)
        flag = "  ←≤$100" if cost <= 100 else ""
        print(f"  {spec.label():<55} {len(panel):>12,}  ${cost:>7.2f}{flag}")

    if args.breakdown:
        for name, spec in [
            ("A: defensible (band=±10%, dte=[7,45], 2023-07)",
             FilterSpec(band=0.10, dte_hi=45, start="2023-07-01")),
            ("B: aggressive (band=±10%, dte=[7,30], 2024-01)",
             FilterSpec(band=0.10, dte_hi=30, start="2024-01-01")),
            ("C: dte=[7,45] floor (band=±10%, dte=[7,45], 2024-01)",
             FilterSpec(band=0.10, dte_hi=45, start="2024-01-01")),
        ]:
            print()
            print("=" * 78)
            print(f"PER-MONTH BREAKDOWN — {name}")
            print("=" * 78)
            df = per_month_breakdown(contracts, spot, current, spec)
            df_nonzero = df.loc[df["cand_id_days"] > 0]
            print(f"  {'month':<8}  {'cur_id_days':>12}  {'cand_id_days':>14}  "
                  f"{'cur_$':>7}  {'proj_$':>8}")
            print("  " + "-" * 60)
            for r in df_nonzero.itertuples(index=False):
                print(f"  {r.month:<8}  {r.cur_id_days:>12,}  {r.cand_id_days:>14,}  "
                      f"${r.cur_usd:>5.2f}  ${r.proj_usd:>6.2f}")
            print("  " + "-" * 60)
            print(f"  {'TOTAL':<8}  {df_nonzero['cur_id_days'].sum():>12,}  "
                  f"{df_nonzero['cand_id_days'].sum():>14,}  "
                  f"${df_nonzero['cur_usd'].sum():>5.2f}  "
                  f"${df_nonzero['proj_usd'].sum():>6.2f}")


if __name__ == "__main__":
    main()
