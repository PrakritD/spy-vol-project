# Short-Vol VRP Strategy — Extension Research Roadmap (financial + stochastic depth, free data)

> **Provenance.** This is an AI-assisted, multi-source research synthesis (parallel web-research agents → consolidated report), commissioned to answer: *what are the highest-value extensions to the contango-filtered short-VIXY VRP carry ([`STRATEGY.md`](../STRATEGY.md)), grounded in financial-economics and stochastic-calculus depth, implementable on **free data**, and what is worth keeping vs dropping?* Claims are cited to the primary sources listed at the end; passages that are widely repeated but **lack out-of-sample support are explicitly flagged**. Treat this as a literature-grounded roadmap, not validated results — every "EDGE" below still has to survive the same out-of-sample / Deflated-Sharpe discipline the base strategy did.

---

## 1. Executive summary (ranked by expected value on free data)

1. **Replace the binary contango switch with a continuous, magnitude-scaled roll/slope signal.** Highest-EV, lowest-cost upgrade. `size ∝ clip((VIX3M−VIX)/VIX or daily roll-yield, 0, cap)`. The term-structure SLOPE (2nd PC) prices variance risk and predicts vol-asset returns across maturities [1]; walk-forward ML on roll features shows the value is in *sizing on predicted magnitude*, not direction (IR 0.623 vs 0.404 naive) [2]. Free: FRED `VXVCLS` + CBOE `VIX_History.csv`.
2. **Make the VRP explicit and condition size on it (model-free IV − range-based RV forecast).** The economic primitive is VRP = model-free implied variance − expected RV, not the term-structure sign [3][4]. Cut size as the *ex-ante* premium collapses — precisely the regime preceding short-vol blowups (the premium falls as risk rises [5]). Free: CBOE VIX² minus Yang–Zhang RV from SPY OHLC [6].
3. **Add a downside-jump / left-tail kill-switch sizing overlay.** The VRP you harvest *is* crash-risk compensation; its predictive power comes from the left-vs-right jump-tail premium, with downside semivariance dominant [7][8]. Use SKEW / VVIX / left-semivariance as a *de-gross* trigger, not a return timer. Directly reinforces the documented drawdown-control edge.
4. **Build the real VIX-futures term structure from CBOE CFE settlement CSVs (2004→).** Upgrades the index-ratio proxy into the *measured* front-2m carry VIXY actually earns, and is model-fittable (OU / Gruenbichler–Longstaff). Free, raw data, zero overfit risk [9].
5. **OU optimal-stopping entry/exit thresholds on the daily log-basis.** Replaces the ad-hoc "flat when backwardated" rule with cost-aware analytic entry/exit boundaries (variational inequalities / double-stopping) [10]. Free daily data; genuinely tradeable continuous-time depth.
6. **A small static convex left-tail floor (VIX-call ladder / SPX put-spread), sized as negative carry.** The contango gate does NOT defend the Feb-2018 fast spike (it was in contango pre-event) [11]. Accept a Sharpe cap to convert the −15% maxDD tail toward a defined-loss profile. Backtestable on free VIX/SPY; intraday monetization can't be modeled free, so size buy-and-hold-to-event.
7. **DROP, do not relitigate:** GEX/gamma timing (our null, VIX echo), DIX directional sleeve (null), naive vol-targeting (~neutral), fixed roll thresholds (Simon–Campasano OOS failure [12]), backwardation as a direction/re-entry timer [13], and any "beats SPY on Sharpe" claim (it is repackaged equity beta).
8. **Reframe the deliverable: this is a beta-like VRP harvest whose ONLY differentiation is documented drawdown control.** +0.61 SPY correlation and equity-like Sharpe are the *identity* of the premium [14][15], not a flaw. Binding constraints are capacity (Volmageddon rebalancing mechanics [11]) and the post-0DTE migration of VRP intraday [16].

## 2. KEEP / DROP / ADD

| Action | Component | Rationale |
|---|---|---|
| **KEEP** | Contango gate (VIX<VIX3M) as a *floor* condition | Roll yield positive only in contango (~92% of days in our sample); free, mechanical, economically motivated [12][14] |
| **KEEP** | Drawdown-control framing as the deliverable | Calmar 0.56 / maxDD −15% vs the unfiltered short; the only durable edge, consistent with post-Volmageddon literature [11] |
| **DROP** | GEX/gamma timing | Our null; ~95% VIX echo, adds nothing |
| **DROP** | DIX directional sleeve | Our null; no next-day SPY direction |
| **DROP** | Naive vol-targeting on VIXY | ~Neutral; vol-targeting aids long-equity convexity, not a roll-driven payoff [2] |
| **DROP** | Fixed roll-yield threshold rules | Simon–Campasano +19.7% CAGR in-sample → "slightly negative" OOS; textbook overfit [12] |
| **DROP** | Backwardation as direction/re-entry timer | CBOE: not a reliable down-market predictor; best periods follow biggest selloffs [13] |
| **DROP** | "Beats SPY on Sharpe" claim | Invites correct rebuttal: repackaged equity beta [14][15] |
| **ADD** | Continuous roll/slope-scaled sizing | OOS: SLOPE prices variance risk [1]; magnitude-sizing adds ~0.22 IR walk-forward [2] |
| **ADD** | Explicit VRP conditioning (IV − range-RV forecast) | OOS-foundational [3][4]; cuts size as ex-ante premium collapses [5] |
| **ADD** | Downside-jump / left-tail de-gross overlay | VRP is downside-tail compensation [7][8]; reinforces drawdown edge |
| **ADD** | CFE settlement-curve backbone | Measured carry, model-fittable, $0, no overfit [9] |
| **ADD** | OU optimal-stopping entry/exit on log-basis | Cost-aware analytic thresholds [10]; tradeable on free daily data |
| **ADD** | Static convex left-tail floor (negative carry) | Only thing that survives a fast Volmageddon spike [11] |

## 3. Financial-depth extensions

### 3a. Continuous roll/slope-scaled sizing (replaces binary switch)
- **Idea:** `size ∝ clip((VIX3M−VIX)/VIX or daily roll-yield, 0, cap)` instead of on/off.
- **OOS evidence:** Johnson (2017, JFQA) — SLOPE (2nd PC) summarizes the term structure and predicts next-day excess returns across 18 asset/maturity cells, 1996–2013, conveying the *price of variance risk* and rejecting the expectations hypothesis [1]. Wang et al. (2024, PLOS ONE) — strict expanding-window walk-forward (train 2005–10, OOS 2011–22) on roll features; constrained mean-variance sizing reaches IR 0.623 vs 0.404 naive long-short [2].
- **Free data:** FRED `VXVCLS` (VIX3M) + `VIXCLS`; CBOE CFE per-contract settlement for true roll yield.
- **Verdict: EDGE.** The single highest-value upgrade. Realistic gain is better Calmar/drawdown-adjusted, not higher gross Sharpe (borrow remains binding) — consistent with our existing finding. *(Note: our own `carry_rollyield` variant already tried a crude continuous version and under-performed the binary gate on the un-tuned scale; the literature value is in a properly walk-forward-sized magnitude signal, not a hand-set cap.)*

### 3b. Forward-looking VRP overlay (de-risk as ex-ante premium collapses)
- **Idea:** Compute VRP = scaled VIX² − expected RV; cut short size when implied is cheap relative to a realized forecast.
- **OOS evidence:** Bollerslev-Tauchen-Zhou (2009, RFS) — IV−RV VRP is the canonical free return predictor [3]; Carr-Wu — model-free VRP carries the premium [4]; Cheng (2018, RFS) — ex-ante VIX premium predicts VIX-futures returns with slope ≈ 1, and *falls* as risk rises [5]. The last is the formal "carry shrinks right before it bites."
- **Free data:** CBOE VIX (the model-free integral, already free) minus a HAR or Yang–Zhang range-RV forecast from SPY OHLC [6].
- **Verdict: EDGE.** Directly attacks the drawdown that is our stated edge.

### 3c. Downside-jump / left-tail de-gross overlay
- **Idea:** A left-tail gauge (SKEW, VVIX, left-semivariance) as a sizing-down kill-switch, not a return timer.
- **OOS evidence:** Bollerslev-Todorov — VRP predictability arises from the left-vs-right jump-tail premium [7]; up/down VRP studies — downside semivariance premium is the stable, dominant component [8].
- **Free data:** VVIX (CBOE CDN), SKEW (via FRED/DataShop), or left-semivariance from free OTM SPX chains.
- **Verdict: EDGE (as risk overlay).** Computable free; reinforces drawdown control. Not an entry-timing input.

### 3d. Convex tail hedge (static floor)
- **Idea:** Small VIX-call ladder / SPX put-spread funded out of carry; accept negative carry for a defined-loss tail.
- **OOS evidence:** Naive long puts are negative-expectancy (VIX 19.3% vs RV 15.1% avg, 1990–2018) [17]; convexity (volga), not intrinsic value, pays in a spike (30%-OTM SPX put +39.3% Mar-2020) [18]. Volmageddon was an endogenous *intraday* rebalancing loop in contango — the gate could not have defended it [11].
- **Free data:** VIX/SPY for backtesting put-spreads / VIX-call ladders (buy-and-hold-to-event; intraday monetization not modelable free).
- **Verdict: EDGE for survivorship, honestly NEGATIVE carry.** Do NOT claim it is positive-carry. **FLAG:** Universa "+4,144% Q1-2020" type figures are vendor-reported, survivorship-prone, NOT OOS — do not cite as expected return.

### 3e. Trend-following / TSMOM diversifier
- **Idea:** A free-data 12-1 TSMOM sleeve (SPY/bond/commodity ETFs) for *sustained* selloffs.
- **OOS evidence:** AQR century study — trend's negative equity correlation (≈−0.5) holds in *prolonged* drawdowns (1973–74, 2000–02) but is mixed-to-negative in the first 10% leg [19][20].
- **Free data:** ETF daily OHLC.
- **Verdict: PARTIAL — duration-mismatched.** Helps the Mar-2020 (sustained) mode; WHIPSAWS the Feb-2018 (one-day) mode. Backtest its correlation to short-VIXY P&L *conditional on fast-vs-slow drawdowns separately*; unconditional correlation overstates protection.

## 4. Stochastic-calculus extensions (honest tradeable-vs-ornament triage)

### 4a. Model-free implied variance vs realized variance — **HIGH VALUE, FREE (EDGE)**
Use the model-free VRP as the continuous, sized primitive; CBOE publishes the model-free VIX free, and RV is proxied by Yang–Zhang from daily SPY OHLC — *no intraday data needed* [3][4][6]. Foundational, multi-decade OOS [3][4]; Yang–Zhang is a closed-form drift-independent estimator (~14× more efficient than close-to-close), consistent for quadratic variation in the diffusion limit [6]. **Verdict: the single best continuous-time upgrade.** Keep contango gate as floor; ADD VRP-scaled position.

### 4b. Heston / SV calibration to the term structure — **LOW VALUE (ORNAMENT)**
Adds parameters and estimation noise without a free-daily edge over the model-free VIX, which is already the integral SV models approximate. **DROP for signal;** optional only to simulate basis paths for stress-testing.

### 4c. Rough volatility (Gatheral-Jaisson-Rosenbaum) — **MOSTLY ORNAMENT (daily/free)**
H≈0.1 is robust in-sample and improves RV *forecasting* [21], but OOS gains are marginal/contested ("hard to unambiguously beat HAR") and estimation needs intraday 5-min data (cost-gated) [22]. A 2026 HAR-RV-RHeston paper reports a small OOS edge but leans on paid OptionMetrics [23]. **DROP as a live daily signal** for a $0 overlay. **FLAG:** rough-vol "tradeability" on daily data is widely repeated but lacks clean fee-free OOS evidence beating HAR.

### 4d. OU mean-reversion of the basis + optimal stopping — **HIGH VALUE, FREE (EDGE)**
Fit OU (or regime-switching OU) to the daily log-basis; derive analytic entry/exit thresholds with transaction costs and stop-loss via variational inequalities / double-stopping; half-life sets holding period [10]. Li (2016) solves the optimal double-stopping problem for VIX futures under mean reversion with regime switching and costs [10]. **ADD as the execution layer** — replaces the binary gate's on/off with principled thresholds. Tradeable on free daily data.

### 4e. Jump-diffusion / tail modeling — **VALUE as risk overlay, not entry signal (EDGE, scoped)**
The premium harvested *is* crash-risk compensation; downside semivariance dominates [7][8]. Computable from free OTM SPX chains or proxied via VIX/SKEW. **ADD as a de-risking kill-switch** (the documented drawdown edge), NOT a return-timer.

## 5. Free data source map

| Source | What it provides | Cost | Frequency | How it plugs in |
|---|---|---|---|---|
| **CBOE CFE settlement CSVs** [9] | Daily VX settlement/volume/OI per contract, 2004→ | Free (no redistribution warranty) | Daily, same evening | **Backbone.** True constant-maturity front-2m series → measured roll yield; fit OU / Gruenbichler–Longstaff |
| **FRED `VXVCLS`** [9] | CBOE 3-Month VIX (VIX3M), 2007→ | Free, public domain | Daily, ~1-day lag | Canonical free VIX3M leg of the contango gate |
| **FRED `VIXCLS`, `DGS3MO`** [9] | VIX 1990→; 3M risk-free | Free, public domain | Daily | Gate level; risk-free leg (already used) |
| **CBOE CDN `VVIX_History.csv`, `VIX9D_History.csv`** [9] | Vol-of-vol; 9-day VIX | Free (disclaimer) | Daily | VVIX = orthogonal convexity/vega de-gross trigger; VIX9D vs VIX = near-dated steepness sub-gate |
| **Yang–Zhang RV from SPY OHLC** (yfinance/Stooq) [6] | Drift-independent realized vol | Free | Daily | Explicit VRP (VIX²−RV²) and realized-vs-implied filter |
| **CFTC TFF VIX positioning** [24] | Dealer/asset-mgr/leveraged-fund net positioning | Free, public domain | Weekly (Tue snapshot, Fri release; 3-day lag) | **Slow de-grossing/crowding TILT only**, never a timer (see flag) |
| **CBOE OTM SPX option chains / SKEW** [9] | Left-tail / jump-tail gauge | Free (chains) / inconsistent (SKEW CSV) | Daily | Downside-jump de-gross overlay |

**Avoid:** the Oxford-Man Realized Library — **discontinued Feb 2023, no replacement** [6]; any "free 5-min RV" route silently depends on it. Use Yang–Zhang instead.

## 6. Structural critique & the single highest-EV next step

**Is daily short-vol alpha? No — it is a compensated, beta-like risk premium.** The literature is near-unanimous: the equity VRP is priced compensation for jump/crash risk under the SDF, loading on the same bad states as equity beta [14][15][25]. Our +0.61 SPY correlation and equity-like Sharpe (0.74 vs SPY 0.78–0.88) are *exactly what theory predicts* — the identity of the premium, not a flaw. Honest framing: a beta-like premium with a differentiated *drawdown profile*, not alpha.

**The two surviving structural critiques:**
- **Negative skew / "pennies in front of a steamroller."** High hit-rate, small wins, rare catastrophic losses — a Sharpe artifact of under-sampled tails. Feb-2018 (XIV/SVXY −90% in one session) is the proof [26].
- **Capacity + post-0DTE break.** On 2 Feb 2018, XIV+SVXY were short ~280k VIX futures, a meaningful fraction of OI; the rebalancing feedback loop is a hard capacity ceiling front-loaded into the most illiquid part of the curve [11]. Separately, 0DTE is now a large share of SPX volume and the VRP has migrated intraday/overnight — the front-month daily VRP VIXY harvests is the most competed, lowest-residual slice [16].

**Single highest-EV next free-data direction: downside/asymmetric VRP conditioning + convex left-tail truncation, both from free VIX-family + SPY data.** Concretely: (1) condition position size on a *forecastable, sign-asymmetric* signal — the downside VRP [27] and a HAR-RV next-period forecast vs IV, both from daily data — replacing the symmetric contango gate; and (2) replace binary long/flat with a **VIX-call ladder / VIXY-call hard-cap overlay** that explicitly buys back the catastrophic tail the steamroller critique targets. Then measure whether *risk-adjusted-after-tail* metrics (Calmar, CVaR, PSR) improve. This attacks both surviving critiques, is fully free-data implementable, and is *interesting whether positive or null* — matching the project's statistical-honesty posture.

**Flagged as repeated-but-thin / lacking OOS support:**
- CFTC CoT extreme spec positioning "predicts VIX spikes" — rests on a handful of in-sample events; 3-day lag kills any short-horizon edge. Slow de-gross tilt only; pre-register it [24].
- ML term-structure backtests reporting Sharpe 2.2–2.6 — canonical overfit; Simon–Campasano went "slightly negative" OOS [12]. Treat any Sharpe >1.5 for a daily VIX-futures timing rule as unvalidated.
- Rough-vol daily tradeability beating HAR — no clean fee-free OOS evidence [22].
- Vendor-reported tail-hedge returns (Universa, "carry-neutral" marketing) — survivorship-prone, not OOS [18].
- BTZ VRP-as-return-predictor regressions — in-sample-strong, OOS-fragile and horizon-sensitive [3].

## Sources

1. Johnson (2017), *Risk Premia and the VIX Term Structure*, JFQA — https://www.travislakejohnson.com/pdfs/Johnson%20VIXTS%202017%20(JFQA).pdf
2. Wang et al. (2024), *VIX constant maturity futures trading strategy: a walk-forward ML study*, PLOS ONE — https://journals.plos.org/plosone/article/file?id=10.1371%2Fjournal.pone.0302289&type=printable
3. Bollerslev, Tauchen, Zhou (2009), *Expected Stock Returns and Variance Risk Premia*, RFS — https://public.econ.duke.edu/~boller/Published_Papers/rfs_09.pdf
4. Carr, Wu (2009), *Variance Risk Premiums*, RFS — https://engineering.nyu.edu/sites/default/files/2019-01/CarrReviewofFinStudiesMarch2009-a.pdf
5. Cheng (2019), *The VIX Premium*, RFS — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2495414
6. Yang–Zhang RV / Oxford-Man discontinuation — https://www.sciencedirect.com/science/article/pii/S2665963824000010 ; https://realized.oxford-man.ox.ac.uk/
7. Bollerslev, Todorov, *Tail Risk Premia and Return Predictability* — https://www.kellogg.northwestern.edu/faculty/todorov/htm/papers/tvt_pred.pdf
8. *Up- and downside variance risk premia in global equity markets* — https://www.sciencedirect.com/science/article/abs/pii/S0378426620301412
9. CBOE CFE historical data & VIX term structure; FRED mirrors — https://www.cboe.com/us/futures/market_statistics/historical_data/ ; https://fred.stlouisfed.org/series/VXVCLS ; https://fred.stlouisfed.org/series/VIXCLS
10. Li (2016), *Trading VIX Futures under Mean Reversion with Regime Switching* — https://arxiv.org/abs/1605.07945
11. Augustin, Cheng, Van den Bergen (2021), *Volmageddon and the Failure of Short-Volatility Products*, FAJ — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3819342
12. Simon & Campasano, *The VIX Futures Basis* — https://www.efmaefm.org/0efmameetings/efma%20annual%20meetings/2013-Reading/papers/VIX%20paper_EFMA.pdf ; https://quantpedia.com/strategies/exploiting-term-structure-of-vix-futures
13. CBOE, *Is VIX Backwardation Necessarily a Sign of a Future Down Market?* — https://www.cboe.com/insights/posts/inside-volatility-trading-is-vix-backwardation-necessarily-a-sign-of-a-future-down-market/
14. AQR (Israelov et al.), *Understanding the Volatility Risk Premium* — https://www.aqr.com/Insights/Research/White-Papers/Understanding-the-Volatility-Risk-Premium
15. AQR, *Embracing Downside Risk* — https://www.aqr.com/-/media/AQR/Documents/Journal-Articles/JoAI-Embracing-Downside-Risk.pdf
16. 0DTE / VRP intraday migration — https://research-api.cbs.dk/ws/portalfiles/portal/105671291/1775874_O._Khalil_Zero_Day_to_Expiry_Options_Trading_and_Variance_Risk_Premium.pdf
17. Long-volatility premium synthesis (Bondarenko/Barclays) — https://philippdubach.com/posts/long-volatility-premium/
18. Newfound/ThinkNewfound, *Tail Hedging* — https://blog.thinknewfound.com/2020/06/tail-hedging/
19. AQR, *A Century of Evidence on Trend-Following Investing* — https://www.aqr.com/Insights/Research/Journal-Article/A-Century-of-Evidence-on-Trend-Following-Investing
20. Return Stacked, *Trend Following Through Turmoil* — https://www.returnstacked.com/trend-following-through-turmoil-why-the-best-protection-comes-after-the-first-punch/
21. Gatheral, Jaisson, Rosenbaum, *Volatility is Rough* — https://arxiv.org/abs/1410.3394
22. *Predicting Realized Variance Out of Sample: Can Anything Beat The Benchmark?* — https://arxiv.org/html/2506.07928v1
23. Options-driven RV forecasting via rough volatility (HAR-RV-RHeston) — https://arxiv.org/html/2604.02743v3
24. CFTC CoT / TFF VIX positioning — https://www.cftc.gov/dea/futures/deacboelf.htm ; https://publicreporting.cftc.gov/stories/s/Commitments-of-Traders/r4w3-av2u/
25. BIS, *The variance risk premium* — https://www.bis.org/publ/qtrpdf/r_qt1409v.htm
26. "Picking up pennies" steamroller critique (illustrative, not OOS) — https://sharpetwo.com/blog/picking-up-pennies-in-front-of-a/ ; CBOE *After the Volpocalypse* — https://cdn.cboe.com/resources/education/research_publications/after-the-volpocalypse-market-observation.pdf
27. Feunou, Fontaine, Kitsul (Fed 2015), *downside variance risk premium* — https://www.federalreserve.gov/econresdata/feds/2015/files/2015020pap.pdf
