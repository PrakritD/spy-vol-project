"""Top-N / bottom-N P&L day extractor with feature context.

For the report's trade-case-studies section: pick the days where the
strategy made the most and lost the most, and dump the signal context
so the human can write a one-line market narrative.
"""

from __future__ import annotations

import pandas as pd


CONTEXT_COLS_DEFAULT = (
    "p_hat", "size", "vxx_close", "vxx_ret_next",
    "vix_z", "gex_net", "gex_regime",
    "obi_5m", "obi_60m", "microprice_dev_bps",
)


def top_bottom_days(
    pnl_df: pd.DataFrame,
    feature_panel: pd.DataFrame,
    n: int = 5,
    context_cols: tuple[str, ...] = CONTEXT_COLS_DEFAULT,
    date_col: str = "date",
) -> dict[str, pd.DataFrame]:
    """Return {'top': df, 'bottom': df} of the n best/worst net-P&L days.

    pnl_df:        output of backtest.execution.backtest()
    feature_panel: the daily features panel (data/processed/features_panel.parquet)
    """
    pnl = pnl_df.copy()
    pnl[date_col] = pd.to_datetime(pnl[date_col]).dt.normalize()
    panel = feature_panel.copy()
    panel[date_col] = pd.to_datetime(panel[date_col]).dt.normalize()
    keep = [c for c in context_cols if c in panel.columns or c in pnl.columns]

    pnl_keep = ["net_pnl"] + [c for c in keep if c in pnl.columns]
    panel_keep = [date_col] + [c for c in keep if c in panel.columns and c not in pnl.columns]
    merged = pnl[[date_col, *pnl_keep]].merge(panel[panel_keep], on=date_col, how="left")
    merged = merged.dropna(subset=["net_pnl"]).sort_values("net_pnl")

    bottom = merged.head(n).reset_index(drop=True)
    top = merged.tail(n).iloc[::-1].reset_index(drop=True)
    return {"top": top, "bottom": bottom}


def to_markdown(case_studies: dict[str, pd.DataFrame], precision: int = 4) -> str:
    """Format for inclusion in the report .qmd."""
    parts = []
    for label in ("top", "bottom"):
        df = case_studies[label]
        if df.empty:
            parts.append(f"### {label.title()} days\n\n_no data_\n")
            continue
        parts.append(f"### {label.title()} days\n")
        parts.append(df.to_markdown(index=False, floatfmt=f".{precision}f"))
        parts.append("")
    return "\n".join(parts)
