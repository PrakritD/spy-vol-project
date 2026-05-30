"""Assemble the daily feature panel.

Joins (point-in-time):
    Yang-Zhang RV target    features/rv_target.daily_yang_zhang_rv + build_target
    VIX family features     features/vix_termstructure.compute   (shift(1) at join)
    GEX                     features/gex.run                     (shift(1) at join)
    SPY/VXX close (current) for execution + sizing context

The microstructure feature group is excluded — it needs ARCX SPY tbbo which
was never pulled (would have meant spending real money beyond the $100
Databento credit).

Every feature in the output panel at date t reflects information observed at
the close of date t-1 (the shift(1) lag), so a model trained on the panel
cannot see same-day or future data. y_next[t] is the binary target =
RV[t+1] > rolling_mean(RV[t-20..t]).

Writes data/processed/features_panel.parquet.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from features import gex, rv_target, vix_termstructure


REPO_ROOT = Path(__file__).resolve().parents[1]
OPTIONS_PANEL_PATH = REPO_ROOT / "data" / "processed" / "options_panel.parquet"
YF_DIR = REPO_ROOT / "data" / "raw" / "yfinance"


def _load_yfinance_close(ticker: str) -> pd.Series:
    p = YF_DIR / f"{ticker}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"missing {p} — run free-data pulls first")
    df = pd.read_parquet(p)[["date", "close"]]
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df.set_index("date")["close"].astype(float).rename(ticker.lower())


def _load_spy_ohlc() -> pd.DataFrame:
    df = pd.read_parquet(YF_DIR / "SPY.parquet")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return (df[["date", "open", "high", "low", "close"]]
              .sort_values("date").reset_index(drop=True))


def _load_vix_panel() -> pd.DataFrame:
    """date-indexed wide frame: columns vix, vix9d, vix3m, vvix."""
    frames = []
    for ticker, code in [("VIX", "vix"), ("VIX9D", "vix9d"),
                         ("VIX3M", "vix3m"), ("VVIX", "vvix")]:
        s = _load_yfinance_close(ticker).rename(code)
        frames.append(s)
    return pd.concat(frames, axis=1).sort_index()


def assemble(cfg_path: Path) -> pd.DataFrame:
    cfg = yaml.safe_load(cfg_path.read_text())
    out_path = REPO_ROOT / cfg["output"]["path"]

    # ---- 1. Raw daily series ---------------------------------------------
    spy_ohlc = _load_spy_ohlc()
    spy_close = spy_ohlc.set_index("date")["close"]
    vxx_close = _load_yfinance_close("VXX")
    vix_wide = _load_vix_panel()

    # ---- 2. VIX termstructure features (date-indexed) --------------------
    vix_feats = vix_termstructure.compute(
        vix_wide,
        vix_termstructure.VixConfig(zscore_window_days=cfg["vix"]["zscore_window_days"]),
    )

    # ---- 3. GEX daily aggregates (date-indexed) --------------------------
    if not OPTIONS_PANEL_PATH.exists():
        raise FileNotFoundError(
            f"{OPTIONS_PANEL_PATH} missing — run `python -m features.opra_panel` first."
        )
    options_panel = pd.read_parquet(OPTIONS_PANEL_PATH)
    gex_cfg = gex.GexConfig(
        delta_lo=cfg["gex"]["delta_filter"][0],
        delta_hi=cfg["gex"]["delta_filter"][1],
        dte_lo_days=cfg["gex"]["dte_filter"][0],
        dte_hi_days=cfg["gex"]["dte_filter"][1],
        multiplier=cfg["gex"]["multiplier"],
    )
    print(f"computing GEX over {len(options_panel):,} contract-days "
          f"({options_panel['date'].nunique()} trading days)...")
    gex_daily = gex.run(options_panel, gex_cfg).set_index("date").sort_index()
    gex_daily = gex_daily.drop(columns=["gex_regime"], errors="ignore")  # str col

    # ---- 4. Yang-Zhang RV + binary regime target -------------------------
    rv_daily = rv_target.daily_yang_zhang_rv(spy_ohlc)
    target = rv_target.build_target(
        rv_daily,
        rv_target.TargetConfig(
            rolling_window_days=cfg["target"]["rolling_window_days"],
            trading_minutes_per_day=cfg["target"].get("trading_minutes_per_day", 390),
            bar_minutes=cfg["target"].get("rv_window_minutes", 5),
            forward_horizon_days=cfg["target"].get("forward_horizon_days", 1),
        ),
    ).set_index("date")
    # HAR triple: rv (daily), rv_5d_mean, rv_rolling_mean (21d). All ending at t.
    target["rv_5d_mean"] = target["rv"].rolling(5, min_periods=5).mean()

    # ---- 5. Join with lookahead-safe lags --------------------------------
    # shift(1) on contemporaneous-day features so date-t rows contain only
    # info available before session t opens.
    feats = vix_feats.shift(1).add_suffix("_lag1").join(
        gex_daily.shift(1).add_suffix("_lag1"), how="outer"
    )
    panel = feats.join(
        target[["rv", "rv_5d_mean", "rv_rolling_mean", "rv_next", "y_next"]],
        how="left",
    )
    panel["spy_close"] = spy_close   # unshifted — execution / sizing context
    panel["vxx_close"] = vxx_close

    # Drop warmup rows (no rolling mean or no target yet).
    panel = panel.dropna(subset=["y_next"])

    # ---- 6. Write --------------------------------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.reset_index().rename(columns={"index": "date"}).to_parquet(out_path, index=False)

    print(f"\nwrote {out_path.relative_to(REPO_ROOT)}")
    print(f"  rows         : {len(panel):,}")
    print(f"  columns      : {panel.shape[1]}")
    print(f"  date range   : {panel.index.min().date()} → {panel.index.max().date()}")
    print(f"  label balance: y=1 in {(panel['y_next'] == 1).mean():.1%} of rows")
    return panel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    args = ap.parse_args()
    assemble(args.config)


if __name__ == "__main__":
    main()
