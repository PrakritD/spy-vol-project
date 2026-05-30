"""Generate report figures from the backtest runner outputs.

Reads:
    data/processed/walk_forward_preds.parquet
    data/processed/walk_forward_pnl.parquet
    data/processed/backtest_summary.csv

Writes:
    report/_build/equity_curves.png
    report/_build/drawdown.png
    report/_build/calibration.png
    report/_build/monthly_returns_<model>.png
    report/_build/auc_bars.png

Each figure is one function so callers can render them individually.
Matplotlib only — no seaborn dependency.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data" / "processed"
BUILD = REPO_ROOT / "report" / "_build"

PRED_PATH = DATA / "walk_forward_preds.parquet"
PNL_PATH = DATA / "walk_forward_pnl.parquet"
SUMMARY_PATH = DATA / "backtest_summary.csv"

# Distinct, color-blind-friendly palette + neutral grey for benchmarks
_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#17becf", "#bcbd22",
]
_BENCH_COLOR = "#888888"


def _setup_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 140,
        "font.size": 9.5,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _color(name: str, i: int) -> str:
    return _BENCH_COLOR if name.startswith("BENCH_") else _PALETTE[i % len(_PALETTE)]


def equity_curves(pnl: pd.DataFrame, summary: pd.DataFrame) -> Path:
    """All models + benchmarks on one chart."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, (name, grp) in enumerate(pnl.groupby("model_name", sort=False)):
        grp = grp.sort_values("date")
        ax.plot(grp["date"], grp["equity"], label=name,
                color=_color(name, i),
                linewidth=2 if not name.startswith("BENCH_") else 1.5,
                alpha=0.95 if not name.startswith("BENCH_") else 0.65,
                linestyle="-" if not name.startswith("BENCH_") else "--")
    ax.axhline(1.0, color="black", linewidth=0.5, alpha=0.5)
    ax.set_title("Out-of-sample equity curves (1.0 = starting capital)")
    ax.set_ylabel("Equity (multiplier)")
    ax.legend(loc="best", ncol=2, fontsize=8)
    out = BUILD / "equity_curves.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def drawdown(pnl: pd.DataFrame) -> Path:
    """Underwater plot — equity / peak − 1."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for i, (name, grp) in enumerate(pnl.groupby("model_name", sort=False)):
        grp = grp.sort_values("date").copy()
        eq = grp["equity"].values
        if len(eq) == 0:
            continue
        peak = np.maximum.accumulate(eq)
        dd = eq / peak - 1.0
        ax.plot(grp["date"], dd * 100, label=name,
                color=_color(name, i),
                linewidth=1.5,
                linestyle="-" if not name.startswith("BENCH_") else "--",
                alpha=0.9 if not name.startswith("BENCH_") else 0.5)
    ax.set_title("Drawdown over time (% from peak)")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(loc="best", ncol=2, fontsize=8)
    out = BUILD / "drawdown.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def calibration(preds: pd.DataFrame, n_bins: int = 8) -> Path:
    """Reliability plots — predicted probability vs realised hit rate, binned."""
    models = preds["model_name"].unique()
    n = len(models)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows),
                              squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    for i, name in enumerate(models):
        ax = axes[i // ncols, i % ncols]
        ax.set_visible(True)
        sub = preds.loc[preds["model_name"] == name].dropna(subset=["p_hat", "y_true"])
        if sub.empty:
            ax.set_title(f"{name}\n(no data)")
            continue
        bins = pd.cut(sub["p_hat"], bin_edges, include_lowest=True)
        agg = sub.groupby(bins, observed=True).agg(
            mean_p=("p_hat", "mean"),
            hit=("y_true", "mean"),
            n=("y_true", "size"),
        ).dropna()
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5, label="perfect")
        ax.scatter(agg["mean_p"], agg["hit"],
                   s=np.clip(agg["n"], 5, 200), color=_color(name, i),
                   alpha=0.85, edgecolors="white", linewidth=0.8)
        ax.plot(agg["mean_p"], agg["hit"], color=_color(name, i), linewidth=1.2, alpha=0.6)
        ax.set_title(name)
        ax.set_xlabel("Predicted P(y=1)")
        ax.set_ylabel("Empirical hit rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    fig.suptitle("Calibration (reliability plots)", y=1.02)
    out = BUILD / "calibration.png"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def monthly_heatmap(pnl: pd.DataFrame, model_name: str) -> Path:
    """One heatmap per model: year × month colored by compounded net return."""
    sub = pnl.loc[pnl["model_name"] == model_name].copy()
    sub["date"] = pd.to_datetime(sub["date"])
    sub["year"] = sub["date"].dt.year
    sub["month"] = sub["date"].dt.month
    monthly = (sub.groupby(["year", "month"])["net_pnl"]
                  .apply(lambda r: (1.0 + r).prod() - 1.0)
                  .unstack("month")
                  .reindex(columns=range(1, 13)))
    if monthly.empty:
        return BUILD / f"monthly_returns_{model_name}.png"
    fig, ax = plt.subplots(figsize=(7.5, max(2.0, 0.45 * len(monthly) + 1.0)))
    vmax = max(abs(monthly.values[np.isfinite(monthly.values)]).max(), 1e-4) if np.any(np.isfinite(monthly.values)) else 0.01
    im = ax.imshow(monthly.values * 100, cmap="RdYlGn", vmin=-vmax * 100, vmax=vmax * 100,
                    aspect="auto")
    ax.set_xticks(range(12))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_yticks(range(len(monthly)))
    ax.set_yticklabels(monthly.index)
    for (y, m), v in np.ndenumerate(monthly.values):
        if np.isfinite(v):
            ax.text(m, y, f"{v*100:+.1f}", ha="center", va="center",
                    fontsize=7.5, color="black")
    ax.set_title(f"Monthly returns (%) — {model_name}")
    fig.colorbar(im, ax=ax, label="net return (%)", shrink=0.7)
    out = BUILD / f"monthly_returns_{model_name}.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def auc_with_ci(preds: pd.DataFrame, summary: pd.DataFrame) -> Path:
    """Bar chart of AUC per model with autocorrelation-adjusted CI.

    Uses a bootstrap directly over (y_true, p_hat) pairs to get CIs that
    respect the temporal autocorrelation of the daily target.
    """
    from sklearn.metrics import roc_auc_score
    rows = []
    for name, sub in preds.groupby("model_name", sort=False):
        sub = sub.dropna(subset=["y_true", "p_hat"]).sort_values("date").reset_index(drop=True)
        if sub.empty:
            continue
        auc = roc_auc_score(sub["y_true"], sub["p_hat"])
        # Stationary block bootstrap on (y, p) — block size ~ N^(1/3)
        n = len(sub)
        b = max(8, int(round(n ** (1/3))))
        rng = np.random.default_rng(13)
        boots = []
        for _ in range(500):
            n_blocks = int(np.ceil(n / b))
            starts = rng.integers(0, n - b + 1, size=n_blocks)
            idx = np.concatenate([np.arange(s, s + b) for s in starts])[:n]
            samp = sub.iloc[idx]
            if samp["y_true"].nunique() < 2:
                continue
            try:
                boots.append(roc_auc_score(samp["y_true"], samp["p_hat"]))
            except ValueError:
                continue
        if not boots:
            continue
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append({"model": name, "auc": auc, "lo": lo, "hi": hi})

    if not rows:
        return BUILD / "auc_bars.png"
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(df) + 2), 4.5))
    colors = [_color(m, i) for i, m in enumerate(df["model"])]
    bars = ax.bar(df["model"], df["auc"], color=colors, alpha=0.85)
    yerr_lo = df["auc"] - df["lo"]
    yerr_hi = df["hi"] - df["auc"]
    ax.errorbar(df["model"], df["auc"], yerr=[yerr_lo, yerr_hi],
                 fmt="none", ecolor="black", capsize=4, linewidth=1)
    ax.axhline(0.5, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_ylabel("AUC")
    ax.set_title("Out-of-sample AUC per model (95% block-bootstrap CI)")
    ax.set_ylim(0.40, max(0.75, df["hi"].max() + 0.03))
    for bar, val in zip(bars, df["auc"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                 f"{val:.3f}", ha="center", fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    out = BUILD / "auc_bars.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def render_all() -> list[Path]:
    """Generate every figure. Returns list of paths produced."""
    _setup_style()
    BUILD.mkdir(parents=True, exist_ok=True)
    preds = pd.read_parquet(PRED_PATH)
    pnl = pd.read_parquet(PNL_PATH)
    summary = pd.read_csv(SUMMARY_PATH)
    preds["date"] = pd.to_datetime(preds["date"])
    pnl["date"] = pd.to_datetime(pnl["date"])

    produced: list[Path] = []
    produced.append(equity_curves(pnl, summary))
    produced.append(drawdown(pnl))
    produced.append(calibration(preds))
    for m in preds["model_name"].unique():
        produced.append(monthly_heatmap(pnl, m))
    produced.append(auc_with_ci(preds, summary))

    print(f"wrote {len(produced)} figures to {BUILD.relative_to(REPO_ROOT)}/")
    for p in produced:
        print(f"  {p.relative_to(REPO_ROOT)}")
    return produced


if __name__ == "__main__":
    render_all()
