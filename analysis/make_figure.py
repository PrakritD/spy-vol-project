"""Evidence figure for FINDINGS.md: mechanism-is-real vs no-incremental-skill.

Left  : log realized vol conditioned on dealer-gamma sign (the mechanism is real).
Right : incremental-skill test statistic for the six pre-registered formulations
        (positive => gamma helps beyond VIX/HAR); shaded |z|<1.96 = not significant.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = __file__.rsplit("/analysis/", 1)[0]
OUT = f"{REPO}/analysis/figures/gamma_null_summary.png"

# Left panel: from the panel itself (faithful, not hardcoded)
df = pd.read_parquet(f"{REPO}/data/processed/features_panel.parquet")
d = df.dropna(subset=["gex_net_lag1", "rv"]).copy()
short = np.log(d.loc[d["gex_net_lag1"] < 0, "rv"])
long = np.log(d.loc[d["gex_net_lag1"] >= 0, "rv"])

# Right panel: standardized test statistics in the "gamma helps as predicted" direction.
# Sourced from analysis/phase0_gonogo.py (level), phase05_reframe.py (2-5), phase05b_profile.py (6).
# Binary tests converted to z = sign(effect)*Phi^{-1}(1 - p/2); DM tests use the DM stat directly.
from scipy.stats import norm  # noqa: E402
z_from_p = lambda eff, p: np.sign(eff) * norm.ppf(1 - p / 2)
formulations = [
    ("Level (RV)",            -0.63),                 # Phase0 M1 gamma-level DM
    ("Intraday range",        -1.42),                 # Phase05 T1 DM
    ("Mean-reversion",        -0.99),                 # Phase05 T2: predicted reversal (neg); observed +0.99 => helps-dir negative
    ("Downside tails",        z_from_p(-0.026, 0.43)),# Phase05 T3 dAUC<0
    ("Regime direction",      z_from_p(+0.013, 0.26)),# Phase05 T4 dAUC>0
    ("Profile shape*",        +0.88),                 # Phase05b P1 (best/most favorable; move-size sub-test overfit, DM -2.67)
]
names = [f[0] for f in formulations][::-1]
zs = [f[1] for f in formulations][::-1]

fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.6))

# --- left ---
parts = axL.violinplot([short.values, long.values], showmeans=True, showextrema=False)
for pc in parts["bodies"]:
    pc.set_alpha(0.45)
axL.set_xticks([1, 2]); axL.set_xticklabels([f"short γ (neg)\nn={len(short)}", f"long γ (pos)\nn={len(long)}"])
axL.set_ylabel("log realized vol  (Yang-Zhang)")
axL.set_title("The mechanism is real\nshort-gamma RV ~60% higher  (Welch t = +9.1, p≈0)")
axL.axhline(short.mean(), color="C0", ls="--", lw=0.8); axL.axhline(long.mean(), color="C1", ls="--", lw=0.8)

# --- right ---
colors = ["#c0392b" if z < 0 else "#27ae60" for z in zs]
axR.axvspan(-1.96, 1.96, color="grey", alpha=0.15, label="|z| < 1.96 (not significant)")
axR.axvline(0, color="black", lw=0.8)
axR.barh(range(len(zs)), zs, color=colors, alpha=0.85)
axR.set_yticks(range(len(names))); axR.set_yticklabels(names)
axR.set_xlim(-3, 3)
axR.set_xlabel("incremental-skill test statistic (σ; positive ⇒ gamma beats VIX/HAR)")
axR.set_title("No incremental skill over VIX/HAR\nall six pre-registered formulations")
axR.legend(loc="lower right", fontsize=8)

fig.suptitle("Dealer gamma vs VIX: a real mechanism, fully priced by VIX (21-mo OPRA window)", fontsize=12, y=1.02)
fig.text(0.5, -0.04, "*Profile shape: most-favorable of 3 sub-tests shown; its move-size sub-test significantly DEGRADED the forecast (overfit, DM=-2.7).",
         ha="center", fontsize=7.5, style="italic")
fig.tight_layout()
import os
os.makedirs(f"{REPO}/analysis/figures", exist_ok=True)
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("wrote", OUT)
