"""Walk-forward backtest harness.

Schedule:
    - Sort the feature panel by date.
    - Initial training window: `initial_train_months` of data.
    - Refit every `refit_freq_months`.
    - Expanding window unless cfg says rolling.

For each refit segment:
    - Train on all data strictly before segment start.
    - Predict probability for each date in the segment.
    - Concatenate predictions into one out-of-sample series.

The harness is model-agnostic: pass a Model factory + a feature-column list.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd


@dataclass(frozen=True)
class WalkForwardConfig:
    initial_train_months: int = 12
    refit_freq_months: int = 1
    expanding: bool = True
    # Daily-resolution overrides. If either is set, the schedule loops by day
    # over actual trading dates rather than by calendar month.
    refit_freq_days: int | None = None
    test_start: str | None = None   # explicit OOS start date, overrides initial_train_months
    test_end: str | None = None     # explicit OOS end date


def _add_months(ts: pd.Timestamp, n: int) -> pd.Timestamp:
    return (ts + pd.DateOffset(months=n)).normalize()


def _daily_schedule(
    dates: pd.Series,
    cfg: WalkForwardConfig,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Daily refit: one (train_end, seg_start, seg_end) per trading day in test window.

    Each "segment" is a single trading day. Per day we re-fit on data strictly
    before that day, then predict the day. Compute is N_test_days × model_fit_cost.
    """
    dates = pd.to_datetime(dates).sort_values().reset_index(drop=True)
    if cfg.test_start is not None:
        test_start = pd.Timestamp(cfg.test_start).normalize()
    else:
        test_start = _add_months(dates.iloc[0], cfg.initial_train_months)
    test_end = (pd.Timestamp(cfg.test_end).normalize()
                if cfg.test_end is not None else dates.iloc[-1])

    test_dates = dates[(dates >= test_start) & (dates <= test_end)].reset_index(drop=True)
    step = max(1, cfg.refit_freq_days or 1)
    segments = []
    for i in range(0, len(test_dates), step):
        d = test_dates.iloc[i]
        # train on all dates strictly before d; segment is [d, d_step_end]
        seg_end_idx = min(i + step - 1, len(test_dates) - 1)
        seg_end = test_dates.iloc[seg_end_idx]
        segments.append((d, d, seg_end))
    return segments


def schedule(dates: pd.Series, cfg: WalkForwardConfig) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Return list of (train_end_exclusive, segment_start, segment_end_inclusive) tuples.

    If `cfg.refit_freq_days` is set, schedule by trading day. Otherwise schedule
    by calendar month as before.
    """
    if cfg.refit_freq_days is not None or cfg.test_start is not None:
        return _daily_schedule(dates, cfg)

    dates = pd.to_datetime(dates).sort_values().reset_index(drop=True)
    start = dates.iloc[0]
    end = dates.iloc[-1]
    first_oos = _add_months(start, cfg.initial_train_months)

    segments = []
    cursor = first_oos
    while cursor <= end:
        seg_end = min(_add_months(cursor, cfg.refit_freq_months) - pd.Timedelta(days=1), end)
        segments.append((cursor, cursor, seg_end))
        cursor = _add_months(cursor, cfg.refit_freq_months)
    return segments


def run(
    panel: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    date_col: str,
    model_factory: Callable[[], "object"],
    cfg: WalkForwardConfig,
    rolling_train_months: int | None = None,
    rolling_train_days: int | None = None,
) -> pd.DataFrame:
    """Return a frame with date, y_true, p_hat, model_name.

    rolling_train_days overrides rolling_train_months when set (used by the
    daily-refit mode together with `cfg.refit_freq_days`).
    """
    p = panel[[date_col, target_col, *feature_cols]].dropna(subset=[target_col]).copy()
    p[date_col] = pd.to_datetime(p[date_col]).dt.normalize()
    p = p.sort_values(date_col).reset_index(drop=True)

    segments = schedule(p[date_col], cfg)
    preds = []
    for train_end, seg_start, seg_end in segments:
        train_mask = p[date_col] < train_end
        if rolling_train_days is not None:
            train_start_cutoff = train_end - pd.Timedelta(days=rolling_train_days)
            train_mask &= p[date_col] >= train_start_cutoff
        elif rolling_train_months is not None:
            train_start_cutoff = _add_months(train_end, -rolling_train_months)
            train_mask &= p[date_col] >= train_start_cutoff
        train = p.loc[train_mask]
        seg_mask = (p[date_col] >= seg_start) & (p[date_col] <= seg_end)
        seg = p.loc[seg_mask]
        if train.empty or seg.empty:
            continue
        Xtr, ytr = train[feature_cols], train[target_col].astype(int)
        Xte = seg[feature_cols]
        model = model_factory()
        model.fit(Xtr, ytr)
        p_hat = model.predict_proba(Xte)
        preds.append(pd.DataFrame({
            date_col: seg[date_col].to_numpy(),
            "y_true": seg[target_col].astype(int).to_numpy(),
            "p_hat": p_hat,
            "model_name": getattr(model, "name", model.__class__.__name__),
        }))
    if not preds:
        return pd.DataFrame(columns=[date_col, "y_true", "p_hat", "model_name"])
    return pd.concat(preds, ignore_index=True)
