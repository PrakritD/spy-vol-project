"""One-page risk tearsheet for the VRP-carry strategy: STRATEGY.md's risk-desk subsection.

Reads artifacts written by analysis/strategy_two_sleeve.py and analysis/risk_tearsheet.py:
  analysis/strategy_equity.parquet         (equity curves)
  analysis/strategy_results.json           (headline metrics, for panel titles)
  analysis/risk_tearsheet_results.json     (VaR/ES, stress table, rolling beta)
Produces:
  analysis/figures/risk_tearsheet.png

Four panels: equity + drawdown (adapted from make_figure_strategy.py's headline dashboard,
reusing the same data rather than re-deriving it), a VaR/ES bar comparison (historical vs
Cornish-Fisher), the stress-scenario table rendered as a matplotlib table, and the rolling
beta time series.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = __file__.rsplit("/analysis/", 1)[0]
FIG = f"{REPO}/analysis/figures"
os.makedirs(FIG, exist_ok=True)

GREEN, GREY, RED, BLUE = "#27ae60", "#7f8c8d", "#c0392b", "#2c6fbb"

eq = pd.read_parquet(f"{REPO}/analysis/strategy_equity.parquet")
eq["date"] = pd.to_datetime(eq["date"])
R = json.load(open(f"{REPO}/analysis/strategy_results.json"))
RT = json.load(open(f"{REPO}/analysis/risk_tearsheet_results.json"))
H = R["headline_metrics"]


def dd(curve: pd.Series) -> np.ndarray:
    a = curve.to_numpy()
    return a / np.maximum.accumulate(a) - 1.0


fig = plt.figure(figsize=(13.5, 11))
gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1.1, 1.3], hspace=0.55, wspace=0.28)

# ---- panel A: equity (log) --------------------------------------------------
ax = fig.add_subplot(gs[0, 0])
ax.plot(eq["date"], eq["carry"], color=GREEN, lw=1.7,
        label=f"VRP carry (Sharpe {H['sharpe']:.2f})")
ax.plot(eq["date"], eq["spy_total"], color=GREY, lw=1.1, alpha=0.9, label="Buy-hold SPY")
ax.set_yscale("log")
ax.set_ylabel("growth of $1 (log)")
ax.set_title("Equity", fontsize=11)
ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
ax.grid(alpha=0.25, which="both")

# ---- panel B: drawdown -------------------------------------------------------
ax = fig.add_subplot(gs[0, 1])
ax.fill_between(eq["date"], dd(eq["carry"]) * 100, 0, color=GREEN, alpha=0.55,
                 label=f"VRP carry (maxDD {H['maxdd']*100:.0f}%)")
ax.plot(eq["date"], dd(eq["spy_total"]) * 100, color=GREY, lw=1.1, label="Buy-hold SPY")
ax.set_ylabel("drawdown (%)")
ax.set_title("Drawdown", fontsize=11)
ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
ax.grid(alpha=0.25)

# ---- panel C: VaR/ES bar comparison ------------------------------------------
ax = fig.add_subplot(gs[1, 0])
var_es = RT["var_es"]
labels = ["VaR 99%\n(historical)", "ES 99%\n(historical)",
          "VaR 99%\n(Cornish-Fisher)", "ES 99%\n(Cornish-Fisher)"]
vals = [var_es["historical"]["var_99_1d"] * 100, var_es["historical"]["es_99_1d"] * 100,
        var_es["cornish_fisher"]["var_99_1d"] * 100, var_es["cornish_fisher"]["es_99_1d"] * 100]
colors = [BLUE, "#1c4e7a", RED, "#7a1f14"]
bars = ax.bar(range(4), vals, color=colors, alpha=0.9)
ax.set_xticks(range(4))
ax.set_xticklabels(labels, fontsize=8.5)
ax.set_ylabel("1-day loss (%)")
ax.set_title("99% 1-day VaR / Expected Shortfall,\ncarry sleeve daily returns", fontsize=11)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.2f}%", ha="center", fontsize=8.5)
ax.grid(alpha=0.25, axis="y")

# ---- panel D: rolling beta ----------------------------------------------------
ax = fig.add_subplot(gs[1, 1])
rb = RT["rolling_beta"]
r = eq["carry"].to_numpy(float)
r = r / np.concatenate(([1.0], r[:-1])) - 1.0
m = eq["spy_total"].to_numpy(float)
m = m / np.concatenate(([1.0], m[:-1])) - 1.0
win = rb["window_days"]
cov = pd.Series(r).rolling(win).cov(pd.Series(m))
var = pd.Series(m).rolling(win).var()
beta = (cov / var).to_numpy()
ax.plot(eq["date"], beta, color=BLUE, lw=1.3)
ax.axhline(rb["mean"], color="k", ls="--", lw=0.9, label=f"mean {rb['mean']:+.2f}")
ax.set_ylabel(f"rolling {win}d beta vs SPY")
ax.set_title(f"Rolling beta ({win}d, ~6mo window)", fontsize=11)
ax.legend(fontsize=8, loc="upper right")
ax.grid(alpha=0.25)

# ---- panel E: stress table (full width) --------------------------------------
ax = fig.add_subplot(gs[2, :])
ax.axis("off")
episodes = RT["stress_table"]["episodes"]
col_labels = ["Scenario", "SPY move", "Strategy move", "Days pk->trough", "% days in-mkt"]
rows = [[e["scenario"], f"{e['spy_move_pct']:+.1f}%", f"{e['strategy_move_pct']:+.1f}%",
         str(e["days_peak_to_trough"]), f"{e['pct_days_in_market']:.0f}%"] for e in episodes]
col_widths = [0.34, 0.16, 0.18, 0.18, 0.14]
tbl = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center",
               colWidths=col_widths)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.6)
for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")
    else:
        cell.set_facecolor("#f2f2f2" if row % 2 == 0 else "white")
    if col == 0:
        cell.set_text_props(ha="left")
        cell.PAD = 0.02
ax.set_title("Stress scenarios: SPY total-return drawdowns > 10%, reformatted from "
             "factor_regression_results.json's co-drawdown table", fontsize=11, pad=14)

fig.suptitle("Risk tearsheet: VRP-carry strategy (short VIXY in contango), 2011-2026",
             fontsize=13, y=0.995)
fig.savefig(f"{FIG}/risk_tearsheet.png", dpi=150, bbox_inches="tight")
print(f"wrote {FIG}/risk_tearsheet.png")
