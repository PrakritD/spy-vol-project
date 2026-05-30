# Build Plan — Two-Sleeve Daily Strategy (VRP carry + SPY timing)

**Date:** 2026-05-30 · **Goal:** a real, honest trading **headline number** + **research depth**, on $0 daily free data. Pivot from intraday (verified cost-gated). Gamma is one adjacent signal, not the engine.

## Edge sources (so the headline can't be a mirage)
- **Sleeve 1 — VRP carry:** the variance risk premium (VIX persistently > realized) is a real, persistent edge. **Vehicle = short VIXY** (ProShares VIX short-term, 2011→2026, no splice, *includes* Feb-2018 +72% and Mar-2020 +177% blowups → honest tail). The overlay's job is to *manage the tail*, and we measure exactly that vs naive always-short carry.
- **Sleeve 2 — SPY timing:** tactical long/flat/short SPY from flow/positioning + trend. DIX is the untapped star (a short-volume/flow signal, tested null for *vol*, never for *direction* — its actual purpose).

## Data (all verified, on disk / free fetch)
VIXY (carry vehicle) · SPY OHLC+**volume** (deep, Yang-Zhang RV + timing) · VIX/VIX9D/VIX3M/VVIX (term structure) · SqueezeMetrics **gex + dix** (2011→) · FRED DGS3MO (rf, borrow proxy). Binding window ~2011-05 → 2026.

## Shared signal stack (pre-registered, all lagged / no-lookahead)
- **Gamma:** gex percentile, negative-gamma flag (regime/stability).
- **Term structure:** VIX9D/VIX, VIX/VIX3M (contango vs backwardation), VVIX/VIX, VIX z-score.
- **Flow:** DIX level + change.
- **Volume/liquidity:** relative volume (vol / trailing mean), Amihud illiquidity (|ret|/$vol), volume-trend confirmation.
- **Trend/momentum:** SPY MA cross, 1m/3m return momentum.
- **Realized vol:** HAR terms (for sizing / regime).

## Sleeve 1 — risk-managed carry
Base = short VIXY (harvest VRP). Overlay gates/sizes: reduce/flatten/flip when negative-gamma OR VIX backwardation OR DIX bearish OR liquidity stress. Vol-target sizing. Costs: spread + short-borrow on VIXY.

## Sleeve 2 — SPY tactical timing
SPY long/flat/short from {DIX, volume, gamma-stability, trend, VIX regime}. Costs: SPY spread (cheap).

## Combination
Risk-weighted (vol-parity) two-sleeve book; report sleeves individually AND combined (the "suite, no hidden core" principle).

## Backtest discipline
Purged + embargoed **expanding walk-forward**; pre-registered signal defs (count logged for DSR); realistic costs; strict no-lookahead (every signal at t uses ≤ t-1; gex lagged for OCC T-1 OI). Trades sized from prior-close signals, filled next open where possible.

## Headline metrics (full-sample, blowups included)
Sharpe · Sortino · **Calmar** · maxDD + duration · CAGR · ann vol · time-in-market · hit rate/profit factor · turnover/cost.
**Benchmarks (mandatory):** buy-hold SPY, **naive always-short-VIXY carry**, 60/40, cash.
**Robustness:** **Deflated Sharpe** over all configs tried · Sharpe minus top-1/5 days · block-bootstrap / MC CI · per-regime blocks (pre-2020 / 2020-21 / 2022+, never pooled).

## Research depth (the "why it's not a mirage")
- **Carry-vs-signal decomposition:** does the overlay beat naive carry on Calmar/tail (the real value), not just raw Sharpe?
- **Signal attribution:** per-signal marginal risk-adjusted value via ablation (drop-one) — which signals earn their place; honest about which don't (incl. gamma, which deep-history showed is a small VIX echo).
- **Per-regime** behavior; **fragility** (top-k-day, blowup contribution).

## Honesty guardrails (anti-v1)
The carry IS the edge; the signals are risk management — say so, and prove the risk-management value explicitly. No headline Sharpe without the naive-carry benchmark, DSR, maxDD, and the 2018/2020 blowups in-sample beside it. Report a clean null for any signal (incl. gamma) that doesn't add risk-adjusted value.

## Deliverables
`analysis/strategy_two_sleeve.py` (+ helpers reused from `analysis/phase1_*`), figures (equity curve incl. blowups, signal-attribution, per-regime), and a `STRATEGY.md` / FINDINGS section with the headline table + the research. Commit; push on request.

## Execution note
Large build → run as a Workflow (data+signals → sleeve backtests → attribution/ablation → synthesis) or stepwise. Resume from this file after `/compact`.
