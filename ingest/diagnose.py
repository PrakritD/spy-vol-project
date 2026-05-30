"""Print available schemas + date range for each dataset we care about.

Read-only — no charge. Run this when a quote fails with
`dataset_schema_not_supported` or `data_start_before_available_start`.
"""

from __future__ import annotations

import os
import sys

import databento as db


DATASETS = [
    "OPRA.PILLAR",
    "DBEQ.BASIC",
    "XNAS.ITCH",
    "XNYS.PILLAR",
    "EQUS.MINI",
    "EQUS.SUMMARY",
]


def main():
    key = os.environ.get("DATABENTO_API_KEY")
    if not key:
        sys.exit("DATABENTO_API_KEY not set.")
    client = db.Historical(key)
    for ds in DATASETS:
        print(f"\n=== {ds} ===")
        try:
            schemas = client.metadata.list_schemas(dataset=ds)
            print(f"  schemas: {schemas}")
        except Exception as e:
            print(f"  schemas ERROR: {e}")
        try:
            rng = client.metadata.get_dataset_range(dataset=ds)
            print(f"  range:   {rng}")
        except Exception as e:
            print(f"  range ERROR: {e}")


if __name__ == "__main__":
    main()
