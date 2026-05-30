# Dealer Gamma vs Realized Volatility — a powered, honest investigation

**Is dealer gamma exposure a VIX echo?** Mostly — but not entirely.

> Dealer gamma tracks realized vol enormously (short-gamma days carry far higher RV, t ≈ +28 over 15 years) yet is **~95% redundant with VIX**. On a calm 21-month window the residual is **undetectable** (a clean null across six pre-registered formulations). On **15 years across real stress regimes it is a small but statistically robust increment** beyond a full VIX/HAR baseline — **gamma-only Diebold-Mariano on CRPS p = 0.001, ΔAUC p = 0.001** — and that increment is genuinely *gamma* (not the Dark-Index flow signal) and survives a richer VIX baseline. The edge is **real and economically small**. Finding it required statistical power, multiple regimes, and a confound check.

**→ Full write-up: [`FINDINGS.md`](FINDINGS.md).**

![Deep-history result](analysis/figures/deep_history_result.png)

## What this is

A study of whether options-dealer **gamma exposure** carries realized-volatility-regime information *incremental to* VIX — asked as a sharp, falsifiable question and answered with the method that makes a negative-or-small result trustworthy:

- **Contamination-fixed target**, pre-registration, strict no-lookahead.
- Out-of-sample **expanding walk-forward**; the **nested Diebold-Mariano test on CRPS** (not raw correlation); block-bootstrap for classification.
- **Per-regime reporting** (never pooled across the 0DTE structural break); confound decomposition (gamma vs DIX vs stale-VIX); multiple-testing control.
- All on **free data**: SqueezeMetrics GEX/DIX (2011→), CBOE VIX (1990→), yfinance SPY/VIX-family, FRED.

## Honest provenance (v1 → v2)

This repo began as a "vol-regime classifier → VXX long-flat strategy" (v1). An audit found that v1 was **fooling itself**: its "+0.84 Sharpe" was a single lucky day, its GEX feature was a VIX echo, and its headline numbers didn't match the shipped config. That story — and why it's instructive — is in [`docs/v1-retrospective.md`](docs/v1-retrospective.md). The reframe to the falsifiable gamma question is in [`docs/superpowers/specs/2026-05-29-gamma-regime-vol-design.md`](docs/superpowers/specs/2026-05-29-gamma-regime-vol-design.md). The v1 pipeline code remains in `features/`, `models/`, `backtest/`, but **its conclusions are superseded by `FINDINGS.md`.**

## Reproduce

```bash
# env with pandas/numpy/scipy/scikit-learn + pyarrow
python analysis/phase1_deep_history.py   # deep-history test, per regime
python analysis/phase1_robustness.py     # gamma-vs-DIX + richer-VIX decomposition
python analysis/phase0_gonogo.py         # 21-month sub-study (level)
python analysis/phase05_reframe.py       # 21-month sub-study (path/dynamics/tails/regime)
python analysis/phase05b_profile.py      # 21-month sub-study (by-strike profile shape)
python analysis/make_figure_deep.py      # headline figure
```

Data is **fetched, not committed** — SqueezeMetrics' terms bar redistribution, and OPRA/price history is large. The fetchers download it into the (git-ignored) `data/` tree.

## What this demonstrates

Calibrated judgment in both directions: refusing a fake +0.84 edge, *and* refusing to dismiss a small real one — establishing the latter only with a powered, multi-regime, confound-checked test (and catching my own bug along the way). Quant rigor on a real question, not a manufactured headline.

## License

MIT.
