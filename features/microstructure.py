"""Intraday SPY microstructure features.

Inputs:
    trades  : columns date, ts (UTC), price, size, side ('B'|'S'|'?')
    bbo     : columns date, ts (UTC), bid, ask, bid_size, ask_size

All features are computed *only* from data within the session and cut at
session_cutoff_et (default 15:55 ET). The output is one row per trading date,
suitable for predicting next-day RV.

Lee-Ready sign inference is applied when trades arrive without a side label.

The `tbbo_to_session_inputs` helper reads ARCX.PILLAR tbbo DBN files
(trade event + BBO snapshot at trade time) and emits the (trades, bbo)
pair this module's `session_features` consumes. tbbo replaces mbp-1 in the
buy plan because mbp-1 captures every quote update and explodes cost (~$288
vs ~$30 for SPY 2023→2026).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ET = "America/New_York"


# ---------------------------------------------------------------------------
# tbbo parser (raw -> (trades, bbo_at_trade))
# ---------------------------------------------------------------------------

def tbbo_to_session_inputs(
    dbn_paths: list[Path],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse ARCX.PILLAR tbbo DBN files into (trades, bbo_at_trade_time).

    tbbo emits one record per trade event with the prevailing BBO embedded.
    Each record yields one trade row AND one bbo row (timestamped at the
    trade). For our features that's enough: Lee-Ready needs the BBO at
    trade time, microprice at session-end uses the last record of the day.

    Returned schemas match what `session_features` expects:
        trades: columns ts (UTC), price, size, side
        bbo:    columns ts (UTC), bid, ask, bid_size, ask_size
    """
    try:
        import databento as db
    except ImportError as e:
        raise ImportError("databento package required to parse tbbo DBN files") from e

    trade_frames = []
    bbo_frames = []
    for p in dbn_paths:
        df = db.DBNStore.from_file(p).to_df()
        df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True)

        trades = pd.DataFrame({
            "ts": df["ts_event"],
            "price": df["price"].astype(float) / 1e9,
            "size": df["size"].astype("int64"),
            # ARCX side codes: 'B' (buy aggressor), 'A' (ask = sell aggressor),
            # 'N' (no info) → fall back to Lee-Ready downstream.
            "side": df["side"].astype(str).map({"B": "B", "A": "S", "S": "S"}).fillna("?"),
        })
        trade_frames.append(trades)

        bbo = pd.DataFrame({
            "ts": df["ts_event"],
            "bid": df["bid_px_00"].astype(float) / 1e9,
            "ask": df["ask_px_00"].astype(float) / 1e9,
            "bid_size": df["bid_sz_00"].astype("int64"),
            "ask_size": df["ask_sz_00"].astype("int64"),
        }).dropna(subset=["bid", "ask"])
        bbo_frames.append(bbo)

    trades = (pd.concat(trade_frames, ignore_index=True).sort_values("ts").reset_index(drop=True)
              if trade_frames else pd.DataFrame(columns=["ts", "price", "size", "side"]))
    bbo = (pd.concat(bbo_frames, ignore_index=True).sort_values("ts").reset_index(drop=True)
           if bbo_frames else pd.DataFrame(columns=["ts", "bid", "ask", "bid_size", "ask_size"]))
    return trades, bbo


# Backwards-compat shim — keep the old name pointing at the new function so
# any caller that already imports `mbp_1_to_session_inputs` doesn't break.
mbp_1_to_session_inputs = tbbo_to_session_inputs


@dataclass(frozen=True)
class MicroConfig:
    session_cutoff_et: str = "15:55:00"
    obi_windows_minutes: tuple[int, ...] = (5, 60)


def _to_et(ts: pd.Series) -> pd.Series:
    ts = pd.to_datetime(ts, utc=True)
    return ts.dt.tz_convert(ET)


def lee_ready_sign(trade_price: pd.Series, bbo_mid: pd.Series) -> pd.Series:
    """Crude Lee-Ready: +1 if trade above mid, -1 if below, else previous tick rule."""
    diff = trade_price - bbo_mid
    sign = np.sign(diff).replace(0, np.nan)
    # previous-tick fallback
    return sign.ffill().fillna(0).astype(int)


def session_features(
    trades: pd.DataFrame,
    bbo: pd.DataFrame,
    cfg: MicroConfig | None = None,
) -> pd.DataFrame:
    cfg = cfg or MicroConfig()
    trades = trades.copy()
    bbo = bbo.copy()
    trades["ts_et"] = _to_et(trades["ts"])
    bbo["ts_et"] = _to_et(bbo["ts"])
    cutoff = pd.to_timedelta(cfg.session_cutoff_et)

    # Restrict to session (regular hours; cutoff applied per-day).
    trades["date"] = trades["ts_et"].dt.normalize().dt.tz_localize(None)
    bbo["date"] = bbo["ts_et"].dt.normalize().dt.tz_localize(None)
    trades = trades[trades["ts_et"].dt.time <= (pd.Timestamp("00:00") + cutoff).time()]
    bbo = bbo[bbo["ts_et"].dt.time <= (pd.Timestamp("00:00") + cutoff).time()]

    # Align mid quote to trades for Lee-Ready when 'side' missing/'?'
    bbo["mid"] = (bbo["bid"] + bbo["ask"]) / 2.0
    bbo = bbo.sort_values("ts_et")
    trades = trades.sort_values("ts_et")
    aligned = pd.merge_asof(
        trades[["ts_et", "date", "price", "size", "side"]],
        bbo[["ts_et", "mid"]],
        on="ts_et",
        direction="backward",
        allow_exact_matches=True,
    )
    inferred = lee_ready_sign(aligned["price"], aligned["mid"])
    side_num = aligned["side"].map({"B": 1, "S": -1}).fillna(inferred)
    aligned["signed_size"] = side_num * aligned["size"]

    rows = []
    for date, grp in aligned.groupby("date"):
        bbo_d = bbo[bbo["date"] == date]
        row = {"date": date}
        # microprice deviation at last bbo before cutoff
        if not bbo_d.empty:
            last = bbo_d.iloc[-1]
            denom = last["bid_size"] + last["ask_size"]
            if denom > 0:
                mp = (last["bid"] * last["ask_size"] + last["ask"] * last["bid_size"]) / denom
                row["microprice_dev_bps"] = (mp - last["mid"]) / last["mid"] * 1e4
            else:
                row["microprice_dev_bps"] = np.nan
        else:
            row["microprice_dev_bps"] = np.nan

        # rolling OBI over the last N minutes before cutoff
        cutoff_ts = grp["ts_et"].max()
        for w in cfg.obi_windows_minutes:
            window_start = cutoff_ts - pd.Timedelta(minutes=w)
            sub = grp[grp["ts_et"] >= window_start]
            tot = sub["size"].sum()
            row[f"obi_{w}m"] = (sub["signed_size"].sum() / tot) if tot > 0 else np.nan

        # signed-volume z-score requires history — left to assemble step
        row["signed_volume"] = grp["signed_size"].sum()
        row["total_volume"] = grp["size"].sum()
        rows.append(row)

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def add_history_features(df: pd.DataFrame, lookback_days: int = 20) -> pd.DataFrame:
    """Adds z-score of signed_volume vs trailing window. Uses only past data."""
    out = df.copy().sort_values("date")
    mu = out["signed_volume"].shift(1).rolling(lookback_days, min_periods=lookback_days).mean()
    sd = out["signed_volume"].shift(1).rolling(lookback_days, min_periods=lookback_days).std()
    out["signed_volume_z"] = (out["signed_volume"] - mu) / sd
    return out
