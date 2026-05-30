"""Phase 2 (growth) — is the gamma edge NONLINEAR/threshold, or just the small linear bit?

The deep-history test found a small LINEAR increment from gamma. The F2 thesis is that
the gamma->RV relationship is a THRESHOLD (the gamma flip): dealer dynamics switch sign
when gamma goes negative. If true, a regime-switching (threshold) model should extract MORE
than the linear gamma term -> justifies building the full learned-flip ML ("AI on the
mechanism"). If not, the edge is purely the linear sliver and a fancy model won't help.

Test: regime-switching HAR (HAR dynamics interacted with the gamma-sign flip) vs the
linear-gamma model, OOS, DM-on-CRPS. Plus a descriptive look at the shape of the
gamma->RV relationship (is it a smooth gradient or a sharp threshold near the flip?).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from phase1_robustness import build, load_panel
from phase1_deep_history import wf_ols_crps, dm_test, TRAIN0

d = build(load_panel())
for c in ["har_d", "har_w", "har_m"]:
    d["sw_" + c] = d["gex_neg"] * d[c]          # HAR dynamics that switch in the short-gamma regime
vixf = ["vix_l", "vix_z", "t_9_30", "t_30_90", "vvix_vix"]
base = ["har_d", "har_w", "har_m"] + vixf
lin = base + ["gex_pct", "gex_neg"]                          # linear gamma (the deep-history increment)
nonlin = lin + ["sw_har_d", "sw_har_w", "sw_har_m"]         # + threshold/regime-switching dynamics
d = d.dropna(subset=nonlin + ["y"]).reset_index(drop=True)
y = d["y"].to_numpy(); oos = np.arange(TRAIN0, len(d))
print(f"rows {len(d)}  {d['date'].min().date()}->{d['date'].max().date()}  OOS n={len(oos)}  %short-γ={d['gex_neg'].mean()*100:.1f}%\n")

cB, _ = wf_ols_crps(d[base].to_numpy(), y)
cL, _ = wf_ols_crps(d[lin].to_numpy(), y)
cN, _ = wf_ols_crps(d[nonlin].to_numpy(), y)

def show(tag, a, b):
    db, dm, p = dm_test(a[oos], b[oos]); print(f"{tag:34s} dCRPS={db:+.5f} DM={dm:+.2f} p={p:.3f}")

print("=== OOS CRPS comparisons (positive dCRPS => second model better) ===")
show("linear gamma  vs  VIX/HAR", cB, cL)
show("threshold     vs  VIX/HAR", cB, cN)
show("threshold     vs  linear gamma  (KEY)", cL, cN)

print("\n=== shape of gamma -> log-RV (residual after VIX/HAR), by gamma percentile decile ===")
# in-sample descriptive: does the relationship steepen / break at the low (short-gamma) end?
import numpy as np
X = np.column_stack([np.ones(len(d))] + [d[c].to_numpy() for c in base])
beta, *_ = np.linalg.lstsq(X, y, rcond=None)
resid = y - X @ beta            # log-RV not explained by VIX/HAR
d["_resid"] = resid
d["_dec"] = pd.qcut(d["gex_pct"], 10, labels=False, duplicates="drop")
g = d.groupby("_dec")["_resid"].mean()
for dec, v in g.items():
    bar = "#" * int(abs(v) * 200)
    print(f"  gex_pct decile {int(dec)+1:2d} (low=short-γ): resid={v:+.4f} {bar}")
print("  (monotone gradient => linear; a jump at the lowest deciles => threshold/flip structure)")

print("\n=== VERDICT ===")
db, dm, p = dm_test(cL[oos], cN[oos])
if db > 0 and p < 0.05:
    print(f"NONLINEAR adds value: regime-switching beats linear gamma (dCRPS={db:+.5f}, p={p:.3f}) "
          "-> the full learned-flip/gated-density ML is justified.")
elif db <= 0 and p < 0.05:
    print(f"Threshold model is WORSE than linear (overfit; dCRPS={db:+.5f}, p={p:.3f}) "
          "-> the edge is the small linear sliver; a fancier model would not help here.")
else:
    print(f"No significant difference (dCRPS={db:+.5f}, p={p:.3f}) -> on daily deep data the "
          "threshold form doesn't beat the linear gamma term; revisit at the intraday timescale.")
