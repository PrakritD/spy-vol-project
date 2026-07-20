"""Figure for FORECASTING.md.

Reads analysis/forecast_bench_results.json (written by analysis/forecast_bench.py).
Produces analysis/figures/forecast_bench.png: per-model mean CRPS with 95% CI, one panel
per regime block (overall / pre2020 / 2020-21 / 2022+). Lower is better.
"""
from __future__ import annotations
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = __file__.rsplit("/analysis/", 1)[0]
FIG = f"{REPO}/analysis/figures"
os.makedirs(FIG, exist_ok=True)

R = json.load(open(f"{REPO}/analysis/forecast_bench_results.json"))
MODELS = ["har", "har_vix", "mlp", "qgb"]
LABELS = {"har": "HAR", "har_vix": "HAR+VIX", "mlp": "MLP", "qgb": "Quantile GBM"}
COLORS = {"har": "#7f8c8d", "har_vix": "#2c6fbb", "mlp": "#c0392b", "qgb": "#27ae60"}
BLOCKS = ["overall", "pre2020", "2020-21", "2022+"]
BLOCK_LABELS = {"overall": "Overall", "pre2020": "Pre-2020", "2020-21": "2020-21", "2022+": "2022+"}

fig, axes = plt.subplots(1, 4, figsize=(14, 4.2), sharey=False)

for ax, blk in zip(axes, BLOCKS):
    stats = R["models_by_era"][blk]
    means = [stats[m]["mean_crps"] for m in MODELS]
    los = [stats[m]["mean_crps"] - stats[m]["crps_ci95"][0] for m in MODELS]
    his = [stats[m]["crps_ci95"][1] - stats[m]["mean_crps"] for m in MODELS]
    x = np.arange(len(MODELS))
    ax.bar(x, means, yerr=[los, his], capsize=4,
           color=[COLORS[m] for m in MODELS], alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels([LABELS[m] for m in MODELS], rotation=30, ha="right", fontsize=8.5)
    ax.set_title(f"{BLOCK_LABELS[blk]}  (n={stats['qgb']['n']})", fontsize=10)
    ax.grid(alpha=0.25, axis="y")
    if blk == "overall":
        ax.set_ylabel("mean CRPS (lower is better)")

fig.suptitle("Next-day RV forecast: quantile gradient boosting beats HAR+VIX in every regime\n"
             "(significant except the smaller 2020-21 block); a small MLP does not", fontsize=11.5)
fig.text(0.5, -0.02,
         "95% CI from Newey-West SE of the mean CRPS. Common evaluation window across all "
         "four models (n given per panel).", ha="center", fontsize=8, color="#555555")
fig.tight_layout()
out = f"{FIG}/forecast_bench.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}")
