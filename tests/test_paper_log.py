"""paper_log.py append logic: idempotent on repeated calls for the same date."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import paper_log as P  # noqa: E402


def test_append_row_is_idempotent(tmp_path):
    log_path = tmp_path / "paper_log.csv"
    row = {"date": "2026-07-14", "vix": 14.2, "vix3m": 15.8,
           "ratio": 14.2 / 15.8, "signal_for_next_session": "short", "vixy_close": 9.31}

    assert P.append_row(row, log_path) is True
    assert P.append_row(row, log_path) is False  # same date: no duplicate

    out = pd.read_csv(log_path, dtype={"date": str})
    assert len(out) == 1
    assert list(out.columns) == P.COLUMNS
    assert out.loc[0, "date"] == "2026-07-14"
    assert out.loc[0, "signal_for_next_session"] == "short"


def test_append_row_adds_new_dates(tmp_path):
    log_path = tmp_path / "paper_log.csv"
    row1 = {"date": "2026-07-13", "vix": 14.0, "vix3m": 15.5,
            "ratio": 14.0 / 15.5, "signal_for_next_session": "short", "vixy_close": 9.40}
    row2 = {"date": "2026-07-14", "vix": 16.2, "vix3m": 15.8,
            "ratio": 16.2 / 15.8, "signal_for_next_session": "flat", "vixy_close": 9.31}

    P.append_row(row1, log_path)
    P.append_row(row2, log_path)

    out = pd.read_csv(log_path, dtype={"date": str})
    assert list(out["date"]) == ["2026-07-13", "2026-07-14"]
