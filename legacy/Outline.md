# SPY Volatility Project — Original Spec & Roadmap

> **Status note (2026-05):** This document is the *original* design spec, preserved as written. The shipped pipeline differs in two important ways — both forced by the $0-real-money / free-Databento-credit constraint:
> 1. The **intraday microstructure feature group** (OBI, signed flow, microprice) was designed and implemented (`features/microstructure.py`) but never wired into the final pipeline because the ARCX SPY `tbbo` pull (~$30) was deferred to avoid exceeding the free credit.
> 2. The **realised-volatility target** was originally specified as a 5-minute intraday estimator; the shipped version substitutes the **Yang-Zhang OHLC estimator** (`features/rv_target.daily_yang_zhang_rv`) computed from free daily yfinance data — standard practice in the realised-vol literature when tick data isn't available.
>
> The shipped models replace the original `logistic_vix_only / xgb_full / lstm_intraday` triple with six classes: `logistic`, `logistic_interactions`, `har_x` (Corsi 2009 HAR-X — strongest classical), `xgb_calibrated`, `mlp_small`, `bayesian_head` (GP classifier for calibrated uncertainty). The `lstm_intraday` would have run on the deferred microstructure features.
>
> See `README.md` for actual results, `STATISTICAL_RIGOR.md` for the math-olympiad analysis, and `ship.md` for the development history.

## Objective
	
Build a regime-conditional volatility signal framework for SPY, using options market microstructure (GEX, delta dynamics, flow imbalances) combined with lagged VIX to predict intraday/next-day volatility regime. Backtest a strategy that conditions position sizing or direction on the regime.

For position sizing, use kelly criteroin ::

$$
f^{*}= p - \frac{{1-p}}{b}
$$
Where P is prob win, b is proportion of bet gained with a win. Here RL estimate of P?


---

## Core Hypothesis

Options market maker hedging flows (captured via GEX) suppress or amplify SPY volatility in a regime-conditional way. Order flow imbalances (OBI) and delta dynamics carry directional information that, when combined with the GEX regime, produce exploitable signals that a VIX-only model misses.

---

## Signal Stack

### 1. GEX (Gamma Exposure)

- **What:** Net gamma exposure of SPY options market makers
- **Mechanism:** Positive GEX → MMs fight price moves (mean reversion, vol suppression). Negative GEX → MMs amplify moves (trending, vol expansion)
- **Computation:** `sum(gamma * OI * spot^2 * 0.01)` for calls (positive), puts (negative)
- **Data source:** yfinance (EOD), Databento OPRA (intraday if needed)
- **Regime thresholds:** >+$1 B = strong pin, <-$1 B = vol expansion, else neutral

### 2. Delta Dynamics

- **What:** Aggregate delta of open positions, how it shifts intraday
- **Mechanism:** Delta imbalance forces directional hedging, creates predictable flow
- **Features:** net_delta, delta_change_rate, delta_exhaustion (reversal signal)
- **Data source:** Databento OPRA options trades + greeks

### 3. Order Flow Imbalance (OBI)

- **What:** Buy-initiated vs sell-initiated volume imbalance in SPY equity + options
- **Mechanism:** Persistent OBI precedes directional moves; spike + reversal = fade signal
- **Features:** rolling_obi_5 m, rolling_obi_1 h, obi_divergence (price vs flow disagree)
- **Data source:** Databento MBO (SPY equity), OPRA (options)

### 4. Lagged VIX

- **What:** VIX from prior close (s) as a baseline vol regime anchor
- **Mechanism:** VIX mean-reverts but has momentum at extremes; lag adds info without lookahead
- **Features:** vix_lag 1, vix_lag 5, vix_zscore_20 d, vix_term_structure (VIX 9 D vs VIX vs VIX 3 M)
- **Data source:** yfinance (free)

---

## Target Variable Options (pick one to start)

1. **Binary vol regime:** realised_vol_next_day > threshold (classification, simpler backtest)
2. **Next-day realised vol:** continuous regression target
3. **SPY direction conditioned on vol regime:** if GEX negative + OBI buy → long, etc.

**Recommended starting point:** Option 1 (binary classification). Cleanest backtest, most interpretable.

---

## Architecture

```
Raw Data
├── Databento: SPY MBO trades (equity order flow)
├── Databento: OPRA options trades + greeks
└── yfinance: VIX, VVIX, SPY EOD

Feature Engineering
├── GEX profile (by strike, total, sign)
├── Delta dynamics (net, rate of change)
├── OBI (5m, 1h rolling windows)
├── Lagged VIX features (level, zscore, term structure)
└── Interaction features (GEX_sign * OBI, delta_exhaustion * vix_spike)

Model Layer
├── Baseline: Logistic regression / XGBoost (interpretable, fast to iterate)
└── Extension: LSTM or Transformer on intraday sequence (later)

Backtest
├── Walk-forward validation (no lookahead)
├── Transaction costs: SPY options spread + slippage model
├── Metrics: Sharpe, max drawdown, regime-conditional accuracy
└── Benchmark: VIX-only model, random baseline
```

---

## Data to Pull from Databento

See `pull_data.py` for extraction script.

|Dataset|Schema|Symbols|Date Range|Approx Cost|
|---|---|---|---|---|
|OPRA. PILLAR|trades|SPY options (all strikes near ATM)|2024-01-01 to present|~$20-40|
|XNAS. ITCH or XNYS. PILLAR|mbp-1 or trades|SPY|2024-01-01 to present|~$10-20|

**Use $125 Databento credit. Pull 2024 data first — enough for a meaningful backtest.**

---

## Project Phases

### Phase 1 — Data & Features (1-2 sessions)

- [ ] Pull SPY equity trades from Databento
- [ ] Pull OPRA options trades for SPY near-ATM strikes
- [ ] Pull VIX/VVIX from yfinance
- [ ] Compute GEX daily and intraday
- [ ] Compute OBI at 5 m and 1 h resolution
- [ ] Compute delta dynamics features
- [ ] Save clean feature matrix as parquet

### Phase 2 — Baseline Model (1 session)

- [ ] Define binary target: next-day vol high/low
- [ ] Train XGBoost on 2024 data, validate on 2025
- [ ] Feature importance analysis
- [ ] Compare to VIX-only baseline

### Phase 3 — Backtest (1-2 sessions)

- [ ] Walk-forward framework (monthly refit)
- [ ] Simple strategy: long straddle when vol expansion predicted, flat otherwise
- [ ] Transaction cost model
- [ ] Sharpe, drawdown, regime breakdown

### Phase 4 — Polish & Write-up (1 session)

- [ ] Visualisations: GEX profile, signal heatmap, equity curve
- [ ] 1-page project summary (for portfolio/interviews)
- [ ] GitHub README

---

## Resume/Interview Framing

"Built a regime-conditional vol prediction framework for SPY using options market microstructure — GEX, delta dynamics, and order flow imbalances — combined with lagged VIX features. Trained on Databento OPRA data, walk-forward backtested with explicit transaction cost modelling. The core finding was [X]."

---

## Open Decisions (resolve before Phase 2)

1. Target variable: binary regime vs continuous vol?
2. Prediction horizon: intraday (next 1 h) vs EOD (next day)?
3. Options strategy to backtest: straddles, or just directional SPY sizing?
4. Model: start with XGBoost or go straight to sequence model?

---

## Notes

- Keep it to one shipped thing. Don't extend to multi-asset or add crypto until this is done.
- RWE context (April 2026+) will sharpen intuition on signal construction — use that.
- WAM improvement is the parallel priority; don't let this eat assignment time.