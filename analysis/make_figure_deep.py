"""Headline figure for the deep-history result (FINDINGS.md §5b).

Left : gamma's incremental skill over VIX/HAR by regime block (DM stat on CRPS).
Right: confound decomposition -- it's gamma (not DIX) and not a stale-VIX proxy.
Stats are the verified outputs of analysis/phase1_deep_history.py and
analysis/phase1_robustness.py (cited inline); effect size noted in the caption.
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = __file__.rsplit("/analysis/", 1)[0]
OUT = f"{REPO}/analysis/figures/deep_history_result.png"

# --- from phase1_deep_history.py (gamma+DIX increment vs VIX/HAR, DM stat on CRPS) ---
regimes = [("pre-2020\n(n=1617)", 1.58), ("2020-21\n(n=498)", 1.20),
           ("2022+\n(n=1104)", 2.30), ("ALL\n(n=3219)", 2.77)]
# --- from phase1_robustness.py (full-OOS decomposition, DM stat on CRPS) ---
confound = [("gamma only", 3.22), ("DIX only", 0.03), ("gamma\nover VIX+ΔVIX", 3.21)]

fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.6))

for ax, data, title in [
    (axL, regimes, "Gamma's incremental skill over VIX/HAR, by regime\n(deep history 2011–2026; null on the calm 21-mo window)"),
    (axR, confound, "It's gamma — not DIX, and not stale VIX\n(full out-of-sample decomposition)"),
]:
    labels = [d[0] for d in data]
    vals = [d[1] for d in data]
    colors = ["#27ae60" if v > 1.96 else ("#7f8c8d" if v > 0 else "#c0392b") for v in vals]
    ax.axhspan(-1.96, 1.96, color="grey", alpha=0.13)
    ax.axhline(1.96, color="grey", ls="--", lw=0.8)
    ax.axhline(0, color="black", lw=0.8)
    bars = ax.bar(range(len(vals)), vals, color=colors, alpha=0.88)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Diebold-Mariano stat (σ; + ⇒ gamma helps)")
    ax.set_ylim(-0.6, 3.8)
    ax.set_title(title, fontsize=10.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.08, f"{v:+.2f}", ha="center", fontsize=8.5)
    ax.text(0.02, 0.96, "p<0.05 above dashed line", transform=ax.transAxes, fontsize=8, va="top", color="#555")

fig.suptitle("Dealer gamma carries a SMALL but robust increment beyond VIX (deep, multi-regime, confound-checked)",
             fontsize=12, y=1.02)
fig.text(0.5, -0.05,
         "Effect is economically small: ΔCRPS ≈ +0.002 on baseline ≈0.22 (<1% relative), ΔAUC ≈ +0.007. "
         "Gamma is ~95% a VIX echo; the surviving sliver is real. Gamma-only is cleaner than gamma+DIX (p=0.001 vs 0.006).",
         ha="center", fontsize=7.8, style="italic")
fig.tight_layout()
import os
os.makedirs(f"{REPO}/analysis/figures", exist_ok=True)
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("wrote", OUT)
