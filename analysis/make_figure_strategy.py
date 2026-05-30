"""Headline + research figures for the VRP-carry strategy (STRATEGY.md).

Reads artifacts written by analysis/strategy_two_sleeve.py:
  analysis/strategy_equity.parquet  (equity curves + in-market flag + VIX/VIX3M)
  analysis/strategy_results.json    (metrics, ladder, attribution, robustness)
Produces:
  analysis/figures/strategy_headline.png  (2x2: equity / drawdown / blowup-dodge / borrow-sensitivity)
  analysis/figures/strategy_research.png   (construction ladder + signal attribution)

Honest framing: the carry TRAILS SPY on Sharpe; its durable edge is DRAWDOWN CONTROL.
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

eq = pd.read_parquet(f"{REPO}/analysis/strategy_equity.parquet")
eq["date"] = pd.to_datetime(eq["date"])
R = json.load(open(f"{REPO}/analysis/strategy_results.json"))
H = R["headline_metrics"]
SPYx, SPYt = R["benchmarks"]["buy-hold SPY (excess)"], R["benchmarks"]["buy-hold SPY (total ret)"]

GREEN, GREY, RED, BLUE = "#27ae60", "#7f8c8d", "#c0392b", "#2c6fbb"


def dd(curve):
    a = curve.to_numpy(); return a / np.maximum.accumulate(a) - 1.0


# =================== FIGURE 1: 2x2 headline dashboard ===================
fig, ax = plt.subplots(2, 2, figsize=(13.5, 9))

# (0,0) equity, log
a = ax[0, 0]
a.plot(eq["date"], eq["carry"], color=GREEN, lw=1.9, label=f"VRP carry (Sharpe {H['sharpe']:.2f}, maxDD {H['maxdd']*100:.0f}%)")
a.plot(eq["date"], eq["spy_total"], color=GREY, lw=1.3, alpha=0.9, label=f"Buy-hold SPY (Sharpe {SPYt['sharpe']:.2f}, maxDD {SPYt['maxdd']*100:.0f}%)")
a.plot(eq["date"], eq["constant"], color=RED, lw=1.0, alpha=0.65, label="Unfiltered constant short")
a.set_yscale("log"); a.set_ylabel("growth of $1 (log)")
a.set_title("Equity — carry compounds steadily; SPY ends higher (it should:\nhigher Sharpe), the unfiltered short blows up", fontsize=10.5)
a.legend(loc="upper left", fontsize=8.5, framealpha=0.9); a.grid(alpha=0.25, which="both")

# (0,1) drawdown underwater — THE edge
a = ax[0, 1]
a.fill_between(eq["date"], dd(eq["carry"]) * 100, 0, color=GREEN, alpha=0.55, label=f"VRP carry (maxDD {H['maxdd']*100:.0f}%)")
a.plot(eq["date"], dd(eq["spy_total"]) * 100, color=GREY, lw=1.2, label=f"Buy-hold SPY (maxDD {SPYt['maxdd']*100:.0f}%)")
a.set_ylabel("drawdown (%)")
a.set_title(f"Drawdown — the durable edge: −15% vs SPY −34%\nCalmar {H['calmar']:.2f} vs {SPYx['calmar']:.2f} (a third of the depth)", fontsize=10.5)
a.legend(loc="lower left", fontsize=9, framealpha=0.9); a.grid(alpha=0.25)

# (1,0) blowup dodging
a = ax[1, 0]
events = [("Volmageddon\n2018", -3.9, 75), ("COVID\n2020", -5.6, 273), ("2022\nbear", -1.9, -25)]
x = np.arange(len(events)); w = 0.38
a.bar(x - w / 2, [e[1] for e in events], w, color=GREEN, label="strategy P&L")
a.bar(x + w / 2, [e[2] for e in events], w, color=RED, alpha=0.75, label="long-VIXY move")
a.axhline(0, color="k", lw=0.8)
a.set_xticks(x); a.set_xticklabels([e[0] for e in events], fontsize=9)
a.set_ylabel("% over event window")
a.set_title("Blowup dodging — the filter flattens INTO\nbackwardation, re-enters after", fontsize=10.5)
for xi, (_, s, l) in zip(x, events):
    a.text(xi - w / 2, s + (5 if s > 0 else -11), f"{s:+.0f}", ha="center", fontsize=8)
    a.text(xi + w / 2, l + 6, f"{l:+.0f}", ha="center", fontsize=8)
a.legend(fontsize=9); a.grid(alpha=0.25, axis="y")

# (1,1) borrow sensitivity — the binding cost axis
a = ax[1, 1]
cb = R["cost_borrow_sharpe"]
bx = [int(k.replace("pct", "")) for k in cb]; by = list(cb.values())
a.plot(bx, by, "o-", color=GREEN, lw=1.8, label="carry Sharpe")
a.axhline(SPYx["sharpe"], color=GREY, ls="--", lw=1.1, label=f"SPY Sharpe (excess {SPYx['sharpe']:.2f})")
a.axhline(SPYt["sharpe"], color="k", ls=":", lw=1.0, label=f"SPY Sharpe (total {SPYt['sharpe']:.2f})")
a.axvline(3, color=BLUE, ls=":", lw=1.0); a.text(3.2, 0.42, "headline\n3%/yr", fontsize=7.5, color=BLUE)
a.set_xlabel("VIXY borrow fee (%/yr) — the binding cost"); a.set_ylabel("Sharpe (@10bps spread)")
a.set_title("Borrow sensitivity — carry NEVER beats SPY's Sharpe\nnet of real borrow; it's a drawdown play, not a Sharpe play", fontsize=10.5)
a.legend(fontsize=8.5, loc="upper right"); a.grid(alpha=0.25)

fig.suptitle("Risk-managed short-vol VRP carry (short VIXY only in contango) — 2011–2026, net of costs+borrow, every blowup in-sample\n"
             "Honest verdict: LOWER Sharpe than SPY (0.74 vs 0.78–0.88), but a THIRD of the drawdown (−15% vs −34%; Calmar 0.56 vs 0.38)",
             fontsize=12, y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(f"{FIG}/strategy_headline.png", dpi=150, bbox_inches="tight")
print("wrote", f"{FIG}/strategy_headline.png")

# =================== FIGURE 2: ladder + signal attribution ===================
fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))

ladder = R["ladder"]
order = ["1. constant short (no controls)", "2. + vol-targeting (no filter)",
         "3. + CONTANGO FILTER  <<HEADLINE", "   alt: continuous roll-yield",
         "4. + extra signal gates (full)"]
short = ["constant\nshort", "+ vol-\ntargeting", "+ CONTANGO\nFILTER", "alt: roll-\nyield", "+ extra\ngates"]
sh = [ladder[k]["sharpe"] for k in order]
cols = [GREEN if "CONTANGO" in k else (BLUE if "roll" in k else GREY) for k in order]
b = axL.bar(range(len(sh)), sh, color=cols, alpha=0.9)
axL.set_xticks(range(len(short))); axL.set_xticklabels(short, fontsize=8.5)
axL.set_ylabel("Sharpe (full sample, net)")
axL.axhline(SPYx["sharpe"], color="k", ls="--", lw=0.9)
axL.text(0.02, SPYx["sharpe"] + 0.012, f"SPY Sharpe {SPYx['sharpe']:.2f} (the carry does not beat it)", fontsize=8, color="#333")
axL.set_title("Construction ladder: the contango filter is the lever\nvol-targeting ≈ neutral; extra gates only sacrifice carry", fontsize=11)
for bi, v in zip(b, sh):
    axL.text(bi.get_x() + bi.get_width() / 2, v + 0.012, f"{v:.2f}", ha="center", fontsize=8.5)
axL.set_ylim(0, max(max(sh), SPYx["sharpe"]) * 1.18)

attrib = R["addon_attrib"]
feats = ["gamma", "vvix", "vix_z", "liquidity"]
dcal = [attrib[f]["calmar"] - H["calmar"] for f in feats]
b2 = axR.barh(range(len(feats)), dcal, color=[GREEN if v > 0 else RED for v in dcal], alpha=0.9)
axR.axvline(0, color="k", lw=0.9)
axR.set_yticks(range(len(feats))); axR.set_yticklabels([f"+ {f}" for f in feats], fontsize=9.5)
axR.set_xlabel("Δ Calmar vs the filter-only headline")
axR.set_title("Signal attribution (add-one): every extra signal HURTS\ngamma ≈ null — consistent with FINDINGS (gamma is a VIX echo)", fontsize=11)
for bi, v in zip(b2, dcal):
    axR.text(0.004, bi.get_y() + bi.get_height() / 2, f"{v:+.2f}", va="center", ha="left", fontsize=8.5, color="#333")
axR.set_xlim(min(dcal) * 1.25, 0.03); axR.invert_yaxis()

fig.suptitle("Research depth: where the edge is (the term-structure filter) and where it is NOT (vol-targeting, gamma, extra gates)",
             fontsize=12, y=1.02)
fig.text(0.5, -0.04,
         f"Net of 10bps + 3%/yr borrow. Deflated Sharpe {R['dsr']['range'][0]:.2f}–{R['dsr']['range'][1]:.2f} over {R['dsr']['n_trials']} variants "
         f"(NOT the clone-inflated 0.98); bootstrap P(SR≤0)={R['bootstrap_ci_vs0'][2]:.3f} vs-zero. "
         "SPY-timing sleeve is a separate honest null (OOS AUC 0.51).",
         ha="center", fontsize=8, style="italic")
fig.tight_layout()
fig.savefig(f"{FIG}/strategy_research.png", dpi=150, bbox_inches="tight")
print("wrote", f"{FIG}/strategy_research.png")
