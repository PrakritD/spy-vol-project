"""Append today's VIX/VIX3M/VIXY closes and the pre-registered contango signal to
analysis/paper_log.csv, the strategy's live forward-test record.

Columns: date, vix, vix3m, ratio, signal_for_next_session, vixy_close. `ratio` is
VIX/VIX3M at that date's close; `signal_for_next_session` ("short"/"flat") is the same
zero-parameter contango rule as `strategy_two_sleeve.contango_flag` (ratio < 1 => short),
computed from THIS row's close and applying to the NEXT session, the identical t-1-close
convention the backtest uses, so this log is directly comparable to it. Fills are derived
from vixy_close after the fact, not logged daily.

Idempotent: running twice on the same date does not duplicate the row. Refuses to
silently backfill: if trading days are missing between the last logged row and the
latest available close, prints which and appends only the latest one.

Run: python analysis/paper_log.py
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = REPO_ROOT / "analysis" / "paper_log.csv"
COLUMNS = ["date", "vix", "vix3m", "ratio", "signal_for_next_session", "vixy_close"]

CBOE_CDN = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{name}_History.csv"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}


def _index_series(cboe_name: str, yf_ticker: str) -> pd.DataFrame:
    """Daily (date, close) for a cash index. CBOE CDN primary: it is the authoritative
    publisher and, empirically, more complete/current than yfinance's index tickers
    (^VIX3M lags several sessions there; ^VIX intraday rows can leave gaps on the prior
    close). yfinance is the fallback if the CBOE request itself fails."""
    try:
        url = CBOE_CDN.format(name=cboe_name)
        r = requests.get(url, headers=UA, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text)).rename(columns=str.lower)
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        return df[["date", "close"]].dropna().sort_values("date")
    except Exception as exc:  # noqa: BLE001 - any failure falls back to yfinance
        print(f"paper_log: CBOE fetch for {cboe_name} failed ({exc}), falling back to yfinance",
              file=sys.stderr)
        import yfinance as yf
        df = yf.download(yf_ticker, period="10d", auto_adjust=False, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index().rename(columns={"Date": "date", "Close": "close"})
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        return df[["date", "close"]].dropna().sort_values("date")


def _vixy_series() -> pd.DataFrame:
    """Daily (date, close) for VIXY. yfinance only (no CBOE index equivalent for an ETF).
    Prefers Adj Close; the most recent bar sometimes posts with a NaN Adj Close before
    settling, so falls back to raw Close for that row rather than dropping it."""
    import yfinance as yf
    df = yf.download("VIXY", period="10d", auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index().rename(columns={"Date": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["close"] = df["Adj Close"].fillna(df["Close"])
    return df[["date", "close"]].dropna().sort_values("date")


def fetch_latest_row() -> dict:
    vix_s = _index_series("VIX", "^VIX")
    vix3m_s = _index_series("VIX3M", "^VIX3M")
    vixy_s = _vixy_series()

    shared = set(vix_s["date"]) & set(vix3m_s["date"]) & set(vixy_s["date"])
    if not shared:
        raise RuntimeError("no trading date is present across VIX, VIX3M, and VIXY sources")
    date = max(shared)

    vix = float(vix_s.loc[vix_s["date"] == date, "close"].iloc[0])
    vix3m = float(vix3m_s.loc[vix3m_s["date"] == date, "close"].iloc[0])
    vixy_close = float(vixy_s.loc[vixy_s["date"] == date, "close"].iloc[0])
    ratio = vix / vix3m
    signal = "short" if ratio < 1.0 else "flat"
    return {"date": date.strftime("%Y-%m-%d"), "vix": vix, "vix3m": vix3m,
             "ratio": ratio, "signal_for_next_session": signal, "vixy_close": vixy_close}


def append_row(row: dict, log_path: Path = LOG_PATH) -> bool:
    """Append `row` to `log_path` unless its date is already logged. Returns True iff a
    row was written."""
    if log_path.exists():
        existing = pd.read_csv(log_path, dtype={"date": str})
    else:
        existing = pd.DataFrame(columns=COLUMNS)

    if row["date"] in set(existing["date"]):
        print(f"paper_log: {row['date']} already logged, skipping (idempotent no-op)")
        return False

    if not existing.empty:
        last_logged = pd.Timestamp(existing["date"].max())
        span = pd.bdate_range(last_logged + pd.Timedelta(days=1), pd.Timestamp(row["date"]))
        missing = [d.strftime("%Y-%m-%d") for d in span[:-1]]  # exclude the row being appended
        if missing:
            print(f"paper_log: NOT backfilling {len(missing)} missing session(s): {missing}. "
                  f"Appending only {row['date']}.")

    new_row = pd.DataFrame([row])[COLUMNS]
    out = new_row if existing.empty else pd.concat([existing, new_row], ignore_index=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(log_path, index=False)
    print(f"paper_log: appended {row['date']}  vix={row['vix']:.2f} vix3m={row['vix3m']:.2f} "
          f"ratio={row['ratio']:.4f} signal_for_next_session={row['signal_for_next_session']} "
          f"vixy_close={row['vixy_close']:.2f}")
    return True


def main() -> int:
    row = fetch_latest_row()
    append_row(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
