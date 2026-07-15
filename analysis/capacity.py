"""Capacity check for the VRP-carry strategy (STRATEGY.md §5).

The strategy trades VIXY at 0.2x book notional (`NOTIONAL` in strategy_two_sleeve.py); a flip
(contango <-> backwardation) moves the whole 0.2x book through the market. Compares that flip
size against VIXY's dollar ADV across representative book sizes. Flip frequency comes from the
already-committed execution_lag_results.json (~12.5/yr); this script only adds the ADV side.

Run: python analysis/capacity.py -> analysis/capacity_results.json
"""
from __future__ import annotations
import json
import sys

REPO = __file__.rsplit("/analysis/", 1)[0]
sys.path.insert(0, REPO + "/analysis")
from strategy_two_sleeve import build_signals, load_panel  # noqa: E402

FLIP_FRACTION = 0.20  # NOTIONAL in strategy_two_sleeve.py: a flip trades the whole 0.2x book
BOOK_SIZES = [1_000_000, 10_000_000, 50_000_000]


def main():
    d = build_signals(load_panel())
    dollar_vol = d["vixy_vol"] * d["vixy_adj"]
    adv_21d = dollar_vol.rolling(21).median()
    # full-sample median of the 21d rolling median: a stable "typical" ADV, not a single stale day
    typical_adv = float(adv_21d.median())
    recent_adv = float(adv_21d.dropna().iloc[-1])

    flips = json.load(open(f"{REPO}/analysis/execution_lag_results.json"))["flips"]

    table = []
    for book in BOOK_SIZES:
        flip_dollars = FLIP_FRACTION * book
        table.append({
            "book_usd": book,
            "flip_trade_usd": flip_dollars,
            "flip_pct_of_typical_adv": flip_dollars / typical_adv,
            "flip_pct_of_recent_adv": flip_dollars / recent_adv,
        })

    out = {
        "note": "adjusted-close x share volume is an approximation for dollar ADV (no VWAP data free)",
        "vixy_dollar_adv": {"typical_21d_median": typical_adv, "most_recent_21d_median": recent_adv},
        "flip_fraction_of_book": FLIP_FRACTION,
        "flips_per_year": flips["per_year"],
        "capacity_table": table,
    }
    with open(f"{REPO}/analysis/capacity_results.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"VIXY dollar ADV: typical (full-sample median of 21d rolling median) "
          f"${typical_adv:,.0f}, most recent ${recent_adv:,.0f}")
    print(f"{'book':>12s} {'flip trade':>14s} {'% typical ADV':>14s} {'% recent ADV':>14s}")
    for row in table:
        print(f"${row['book_usd']:>10,.0f} ${row['flip_trade_usd']:>12,.0f} "
              f"{row['flip_pct_of_typical_adv']*100:>13.2f}% {row['flip_pct_of_recent_adv']*100:>13.2f}%")
    print("saved analysis/capacity_results.json")


if __name__ == "__main__":
    main()
