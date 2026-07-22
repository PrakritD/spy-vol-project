"""Black-76: European option pricing on a futures/forward underlying.

Groundwork for STRATEGY.md Sec.7's proposed tail floor (a VIX-call ladder sized as negative
carry), not a fitted strategy: this module is the pricing primitive, validated against
put-call parity and the standard boundary limits. `demo_tail_floor_cost.py` uses it to show,
illustratively, what a 30-day out-of-the-money VIX call would have cost across the
constant-maturity futures curve built in `vix_futures_curve.py` -- using a realized-vol proxy
for sigma, since no real VIX-options quotes are ingested in this project (SPY OPRA options are
ingested for FINDINGS.md's IV-surface work; VIX options are a different, unpaid-for feed).

F = forward/futures price, K = strike, T = years to expiry, r = continuously-compounded
risk-free rate, sigma = annualized lognormal vol of the forward.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _d1_d2(f: np.ndarray, k: np.ndarray, t: np.ndarray, sigma: np.ndarray) -> tuple:
    f, k, t, sigma = (np.asarray(x, float) for x in (f, k, t, sigma))
    vsqrt = sigma * np.sqrt(t)
    d1 = (np.log(f / k) + 0.5 * sigma ** 2 * t) / vsqrt
    d2 = d1 - vsqrt
    return d1, d2


def call_price(f, k, t, r, sigma):
    d1, d2 = _d1_d2(f, k, t, sigma)
    disc = np.exp(-np.asarray(r, float) * np.asarray(t, float))
    return disc * (np.asarray(f, float) * norm.cdf(d1) - np.asarray(k, float) * norm.cdf(d2))


def put_price(f, k, t, r, sigma):
    d1, d2 = _d1_d2(f, k, t, sigma)
    disc = np.exp(-np.asarray(r, float) * np.asarray(t, float))
    return disc * (np.asarray(k, float) * norm.cdf(-d2) - np.asarray(f, float) * norm.cdf(-d1))


def call_delta(f, k, t, r, sigma):
    d1, _ = _d1_d2(f, k, t, sigma)
    return np.exp(-np.asarray(r, float) * np.asarray(t, float)) * norm.cdf(d1)


def put_delta(f, k, t, r, sigma):
    d1, _ = _d1_d2(f, k, t, sigma)
    return -np.exp(-np.asarray(r, float) * np.asarray(t, float)) * norm.cdf(-d1)


def vega(f, k, t, r, sigma):
    """Price change per 1.00 (100 vol points) change in sigma; divide by 100 for per-vol-point."""
    d1, _ = _d1_d2(f, k, t, sigma)
    f, t, r = np.asarray(f, float), np.asarray(t, float), np.asarray(r, float)
    return np.exp(-r * t) * f * norm.pdf(d1) * np.sqrt(t)


def gamma(f, k, t, r, sigma):
    d1, _ = _d1_d2(f, k, t, sigma)
    f, t, r, sigma = np.asarray(f, float), np.asarray(t, float), np.asarray(r, float), np.asarray(sigma, float)
    return np.exp(-r * t) * norm.pdf(d1) / (f * sigma * np.sqrt(t))
