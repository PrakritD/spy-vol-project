"""Closed-form crowding model for the short-volatility trade (CROWDING.md).

STRATEGY.md attributes the strategy's drawdown edge to the contango gate flattening the book
before a spike arrives. That describes what happens without accounting for why the spike is
violent. This script sets out a candidate account in a form that generates testable predictions.

Setup. Two classes of participant hold short-volatility exposure. Leveraged and inverse volatility
ETPs rebalance mechanically to maintain constant leverage, so they must BUY vega when volatility
rises, at a size known in advance and insensitive to price. Discretionary carry traders operate
under margin or risk-budget constraints and are forced to cover into the same move. Both demands
are same-signed and arrive while the price is already moving against the position.

With linear price impact and aggregate constrained exposure K, an exogenous shock r0 resolves to a
fixed point:

    r = r0 + lam * K * r    =>    r = r0 / (1 - lam*K)    =>    m(K) = 1 / (1 - lam*K)

m is the amplification multiplier and K* = 1/lam is the critical crowding level at which any shock
is self-amplifying. Below K* shocks are amplified by a finite factor; the loop is a geometric
series that converges. At K* it does not.

The strategic layer is what distinguishes this from a plain feedback loop. Each trader picks
exposure w_i, and K = sum(w_i) + k_M. Carry is a strategic COMPLEMENT, because the crowd's own
hedging flows suppress realized volatility and make the trade look better as more capital enters.
Tail risk is a strategic SUBSTITUTE, because a higher K raises m for everyone holding the trade.
Each trader internalises their own tail loss but not their marginal contribution to lam*K. That is
a congestion externality on liquidity, and it implies the Nash equilibrium sits above the socially
optimal exposure, with nothing in the private optimisation stopping it from crossing K*.

Two deliberate constraints on the calibration, both pre-registered in CROWDING-PREREG.md:

1. lam is calibrated from ORDINARY-DAY depth only. Days whose absolute VIXY return sits above the
   95th percentile are excluded, so no crisis observation informs the impact coefficient. February
   2018 is the one episode usually read as this loop firing, and a model whose parameters were
   fitted to it could not then be said to predict it.
2. This script never loads the crowding series at all. It emits K* and the m(K) curve from depth
   data alone. Comparing those to realised crowding, including the February 2018 case study, is
   done in crowding_test.py, after this file's numbers are already fixed. Keeping the model
   physically separate from the data it is judged against.

kappa (the vega-rebalance coefficient, how much constrained holders must trade per unit of exposure
per unit of move) is the one free O(1) parameter. It is swept, not fitted, and every headline is
reported across the sweep.

Run: /opt/anaconda3/envs/trading/bin/python analysis/crowding_model.py
Output: analysis/crowding_model_results.json
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "analysis" / "crowding_model_results.json"
sys.path.insert(0, str(REPO_ROOT / "analysis"))

from strategy_two_sleeve import NOTIONAL, build_signals, load_panel  # noqa: E402


@dataclass
class ModelCfg:
    """Every knob. Defaults here are the pre-registered specification."""

    # --- impact calibration ---
    ordinary_day_quantile: float = 0.95
    """Absolute-return quantile above which a day is excluded from the impact fit. Keeps crisis
    observations, including Feb 2018, out of lam entirely."""

    kappa_grid: tuple[float, ...] = (0.5, 1.0, 2.0)
    """Vega-rebalance coefficient sweep. kappa=1.0 is the baseline: a constrained holder trades a
    fraction r of exposure after a move r."""

    # --- strategic layer ---
    theta_over_mu_grid: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)
    """Aggregate capital cost Theta = theta/N, expressed as a MULTIPLE OF mu so the sweep is
    self-scaling and dimensionally consistent with the carry it trades off against. Free parameter,
    swept. The reported result is the Nash-to-optimal RATIO, which is far more robust to Theta than
    either level is."""

    spike_quantile: float = 0.99
    """Quantile of the adverse daily VIXY move used as the baseline spike size s."""

    k_mechanical_frac: float = 0.0
    """Mechanical ETP exposure k_M as a fraction of K*, held at zero in the baseline so the
    discretionary equilibrium is not silently pre-loaded. Swept separately."""


def calibrate_lambda(cfg: ModelCfg) -> dict:
    """Amihud-style linear impact coefficient from ordinary-day VIXY depth.

    lam has units of return per dollar traded: a demand of Q dollars moves the price by lam*Q.
    Uses the same |return| / dollar-volume construction already present in build_signals' amihud_z,
    but on VIXY rather than SPY, and restricted to ordinary days.
    """
    d = build_signals(load_panel())
    dollar_vol = (d["vixy_vol"] * d["vixy_adj"]).replace(0.0, np.nan)
    abs_ret = d["vixy_ret"].abs()

    ok = dollar_vol.notna() & abs_ret.notna() & (abs_ret > 0)
    cutoff = float(abs_ret[ok].quantile(cfg.ordinary_day_quantile))
    ordinary = ok & (abs_ret <= cutoff)

    impact = (abs_ret[ordinary] / dollar_vol[ordinary])
    lam = float(impact.median())

    return {
        "lambda_per_dollar": lam,
        "lambda_mean_per_dollar": float(impact.mean()),
        "abs_return_cutoff": cutoff,
        "n_days_total": int(ok.sum()),
        "n_days_ordinary": int(ordinary.sum()),
        "n_days_excluded_as_crisis": int((ok & (abs_ret > cutoff)).sum()),
        "typical_dollar_adv": float(dollar_vol[ordinary].median()),
        "note": ("median of |VIXY return| / VIXY dollar volume over ordinary days only; days above "
                 "the {:.0%} absolute-return quantile are excluded so no crisis day, Feb 2018 "
                 "included, informs the impact coefficient").format(cfg.ordinary_day_quantile),
    }


def multiplier(x: np.ndarray | float) -> np.ndarray | float:
    """Amplification m(x) = 1/(1 - x) in normalized crowding x = K/K*, clipped at the singularity.

    The single canonical definition of the amplification. Everything downstream calls this rather
    than re-deriving it inline, so the closed form has exactly one implementation to test.
    """
    gain = np.clip(np.asarray(x, dtype=float), -np.inf, 1.0 - 1e-12)
    return 1.0 / (1.0 - gain)


def _solve_equilibrium(mu: float, theta_agg: float, s: float, p: float,
                       x_m: float, social: bool) -> dict:
    """Fixed point for aggregate discretionary exposure, in NORMALIZED units x = K/K*.

    Working in x rather than dollars is what keeps the payoff dimensionally coherent: carry, the
    spike loss and the capital cost are all per-unit-exposure returns, so the exposure they
    multiply has to be dimensionless too. Since lam_eff*K = K/K* = x by construction, the
    amplification is simply m(x) = 1/(1-x), and the singularity sits at x = 1.

    Each trader's payoff per unit time is
        pi_i = mu*x_i - (Theta/2)*x_i^2 - x_i * m(x) * s * p
    carry, minus a quadratic capital cost, minus the expected spike loss, which is itself scaled by
    the crowd-determined amplification.

    Nash (atomistic): the trader takes m(x) as given.
        FOC:  mu - Theta*x - m(x)*s*p = 0

    Social planner: internalises that raising x raises m for every unit already held.
        FOC:  mu - Theta*x - m(x)*s*p - x*m'(x)*s*p = 0
    with m'(x) = 1/(1-x)^2 > 0, so the planner carries a strictly negative extra term and
    x_opt < x_Nash whenever any exposure is held at all. The wedge follows from m' > 0 alone, so it
    is structural rather than a parameter artifact.

    Solved by bisection on the discretionary part in [0, 1 - x_m).
    """
    x_max = max(1.0 - x_m, 0.0) * (1.0 - 1e-9)
    if x_max <= 0:
        return {"x": x_m, "converged": False}

    def foc(w: float) -> float:
        x = w + x_m
        m = float(multiplier(x))
        val = mu - theta_agg * w - m * s * p
        if social:
            val -= w * m ** 2 * s * p          # m'(x) = 1/(1-x)^2 = m(x)^2
        return val

    if foc(0.0) <= 0:                      # not worth holding any exposure
        return {"x": x_m, "converged": True}
    if foc(x_max) > 0:                     # optimum pinned at the singularity
        return {"x": x_max + x_m, "converged": False}

    lo, hi = 0.0, x_max
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if foc(mid) > 0:
            lo = mid
        else:
            hi = mid
    return {"x": 0.5 * (lo + hi) + x_m, "converged": True}


def calibrate_payoffs(cfg: ModelCfg) -> dict:
    """mu (carry per unit exposure per day) and s (spike loss per unit exposure) from the panel.

    Both are measured on the SHORT side, which is the side the strategy takes: mu is the mean daily
    gain to a unit short in VIXY, s is the adverse move at the spike quantile. p is the empirical
    frequency of a move that large. All three come from realised VIXY history, not from the
    strategy's own P&L, so the model is not calibrated to the thing it is meant to explain.
    """
    d = build_signals(load_panel())
    short_ret = (-d["vixy_ret"]).dropna()           # unit short in VIXY

    mu = float(short_ret.mean())
    s = float(-short_ret.quantile(1.0 - cfg.spike_quantile))   # adverse tail, positive magnitude
    p = float((short_ret <= -s).mean())

    return {
        "mu_daily_per_unit_short": mu,
        "spike_loss_s_per_unit_short": s,
        "spike_prob_p_daily": p,
        "spike_quantile": cfg.spike_quantile,
        "n_days": int(short_ret.size),
        "note": ("mu, s and p are measured on realised VIXY returns, not on the strategy's own P&L, "
                 "so the equilibrium is not calibrated to the result it is used to interpret"),
    }


def main() -> int:
    cfg = ModelCfg()

    impact = calibrate_lambda(cfg)
    payoffs = calibrate_payoffs(cfg)
    lam = impact["lambda_per_dollar"]
    mu, s, p = (payoffs["mu_daily_per_unit_short"], payoffs["spike_loss_s_per_unit_short"],
                payoffs["spike_prob_p_daily"])

    # --- critical crowding in dollars, per kappa ---
    # kappa rescales K* only. The amplification curve indexed on fractions of K* is invariant to
    # it by construction (lam_eff*K = K/K*), so it is reported once rather than per kappa.
    per_kappa = {f"kappa={kappa}": {"lambda_effective_per_dollar": lam * kappa,
                                    "K_star_usd": 1.0 / (lam * kappa)}
                 for kappa in cfg.kappa_grid}
    amplification_curve = {f"K={f:g}*K_star": float(multiplier(f))
                           for f in (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)}

    # --- Nash vs social planner, in normalized exposure x = K/K* ---
    # The normalized equilibrium is independent of kappa: kappa sets the dollar scale of K* but
    # cancels out of the FOCs, which are written entirely in x. So x_nash/x_social are solved once
    # per Theta, and dollar levels follow as x * K*(kappa).
    equilibria = []
    for theta_mult in cfg.theta_over_mu_grid:
        theta_agg = theta_mult * mu
        nash = _solve_equilibrium(mu, theta_agg, s, p, cfg.k_mechanical_frac, social=False)
        opt = _solve_equilibrium(mu, theta_agg, s, p, cfg.k_mechanical_frac, social=True)
        ratio = (nash["x"] / opt["x"]) if opt["x"] > 0 else float("nan")
        equilibria.append({
            "theta_over_mu": theta_mult,
            "theta_aggregate": theta_agg,
            "x_nash": nash["x"],
            "x_social": opt["x"],
            "nash_over_social": ratio,
            "nash_exceeds_critical": bool(nash["x"] >= 1.0),
            "amplification_at_nash": float(multiplier(nash["x"])),
            "amplification_at_social": float(multiplier(opt["x"])),
            "both_converged": bool(nash["converged"] and opt["converged"]),
            "K_nash_usd_by_kappa": {f"kappa={k}": nash["x"] / (lam * k) for k in cfg.kappa_grid},
            "K_social_usd_by_kappa": {f"kappa={k}": opt["x"] / (lam * k) for k in cfg.kappa_grid},
        })

    ratios = [e["nash_over_social"] for e in equilibria
              if np.isfinite(e["nash_over_social"]) and e["both_converged"]]

    out = {
        "config": {**asdict(cfg),
                   "kappa_grid": list(cfg.kappa_grid),
                   "theta_over_mu_grid": list(cfg.theta_over_mu_grid)},
        "impact_calibration": impact,
        "payoff_calibration": payoffs,
        "critical_crowding": per_kappa,
        "amplification_curve": amplification_curve,
        "equilibria": equilibria,
        "wedge_summary": {
            "n_converged_cells": len(ratios),
            "min_nash_over_social": float(np.min(ratios)) if ratios else None,
            "median_nash_over_social": float(np.median(ratios)) if ratios else None,
            "max_nash_over_social": float(np.max(ratios)) if ratios else None,
            "all_cells_nash_above_social": bool(all(r > 1.0 for r in ratios)) if ratios else None,
        },
        "strategy_notional_reference": NOTIONAL,
        "caveats": [
            "lambda is a linear (Kyle-form) impact coefficient. Real impact is closer to a square "
            "root in trade size; linear is what closes the fixed point in a form that can be "
            "solved, and it overstates amplification at large K.",
            "kappa and Theta are free O(1) parameters, swept rather than fitted. Levels of K_nash "
            "and K_star inherit that freedom; the Nash-above-social wedge does not, since it "
            "follows from m'(K) > 0 for any admissible parameters.",
            "No crowding observation of any kind enters this file. K_star here is a prediction to "
            "be compared against realised crowding in crowding_test.py, not a fit to it.",
        ],
    }

    OUT_PATH.write_text(json.dumps(out, indent=2))

    print(f"impact: lambda = {lam:.3e} per dollar "
          f"({impact['n_days_ordinary']} ordinary days, "
          f"{impact['n_days_excluded_as_crisis']} crisis days excluded)")
    print(f"payoffs: mu = {mu:.5f}/day, s = {s:.4f}, p = {p:.4f}")
    print()
    print("critical crowding K* (kappa rescales the dollar level only):")
    for kappa in cfg.kappa_grid:
        print(f"  kappa={kappa:<5.2f} K* = ${per_kappa[f'kappa={kappa}']['K_star_usd']/1e9:6.2f}bn")
    print(f"  amplification: m=2.0 at 0.5*K*, m={amplification_curve['K=0.9*K_star']:.1f} at 0.9*K*")
    print()
    print(f"{'Theta/mu':>9s} {'x_nash':>8s} {'x_social':>9s} {'ratio':>7s} {'m_nash':>7s}")
    for e in equilibria:
        print(f"{e['theta_over_mu']:>9.2f} {e['x_nash']:>8.4f} {e['x_social']:>9.4f} "
              f"{e['nash_over_social']:>7.2f} {e['amplification_at_nash']:>7.2f}")
    print()
    if ratios:
        print(f"Nash/social exposure ratio across {len(ratios)} converged cells: "
              f"min {min(ratios):.3f}, median {float(np.median(ratios)):.3f}, max {max(ratios):.3f}")
        print(f"all cells show Nash above social optimum: {all(r > 1.0 for r in ratios)}")
    print(f"\nWrote {OUT_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
