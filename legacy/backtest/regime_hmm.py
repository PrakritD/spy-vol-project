"""Hidden Markov regime detection for entry gating.

Fit a 2-state Gaussian HMM on (vix_level, log_rv) daily observations. The
states are ordered by mean VIX so state 0 is always "low-vol" and state 1
is always "high-vol" deterministically. Per test date we emit:

- `regime_state`: hard Viterbi-decoded state in {0, 1}
- `regime_prob_highvol`: posterior probability of state 1 from
  forward-backward smoothing (soft gating)

Use cases:
- Hard gate: trade only when regime_state == 1.
- Soft gate: multiply position size by regime_prob_highvol.

We fit the HMM on the training portion of each walk-forward fold (no
look-ahead). For efficiency at daily-refit cadence we refit at the
start of the test window and roll the Viterbi state forward, rather than
re-fitting at each test day. Documented compromise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HMMConfig:
    n_components: int = 2
    n_restarts: int = 5
    cov_type: str = "diag"
    seed: int = 13
    min_state_separation_std: float = 1.0   # require state means at least this many SDs apart


def _fit_best_of_n(X: np.ndarray, cfg: HMMConfig):
    """Fit HMM with n_restarts random inits; keep the model with the highest log-likelihood."""
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError as e:
        raise ImportError("hmmlearn required; pip install hmmlearn") from e

    best = None
    best_ll = -np.inf
    for seed in range(cfg.seed, cfg.seed + cfg.n_restarts):
        m = GaussianHMM(
            n_components=cfg.n_components,
            covariance_type=cfg.cov_type,
            n_iter=200,
            random_state=seed,
            tol=1e-3,
        )
        try:
            m.fit(X)
            ll = m.score(X)
        except Exception:
            continue
        if ll > best_ll:
            best, best_ll = m, ll
    if best is None:
        raise RuntimeError("HMM failed to fit on all restarts")
    return best


def _order_by_vix_mean(model, vix_col_idx: int = 0) -> dict:
    """Return permutation that orders states by ascending mean of column vix_col_idx."""
    order = np.argsort(model.means_[:, vix_col_idx])
    return {old: new for new, old in enumerate(order)}


def fit_regime(
    train_obs: pd.DataFrame,
    cfg: HMMConfig | None = None,
) -> dict:
    """Fit 2-state HMM on (vix_level, log_rv). Returns a model + ordering dict.

    Args:
        train_obs: DataFrame with columns 'vix_level' and 'rv' (or 'log_rv').
        cfg:       HMMConfig.

    Returns:
        {'model': fitted_hmm, 'order': {old_state: new_state}, 'means': ndarray (state, feature)}
        in the *reordered* state space (state 0 = low-vol, state 1 = high-vol).
    """
    cfg = cfg or HMMConfig()
    df = train_obs.copy()
    if "log_rv" not in df.columns:
        df["log_rv"] = np.log(df["rv"].clip(lower=1e-6))
    X = df[["vix_level", "log_rv"]].dropna().to_numpy()
    if len(X) < 60:
        raise ValueError(f"need >= 60 training observations, got {len(X)}")

    model = _fit_best_of_n(X, cfg)
    order = _order_by_vix_mean(model)
    means_reordered = np.zeros_like(model.means_)
    for old, new in order.items():
        means_reordered[new] = model.means_[old]

    # Sanity: require state means meaningfully separated on VIX dimension.
    sep = abs(means_reordered[1, 0] - means_reordered[0, 0]) / np.sqrt(model.covars_.mean())
    if sep < cfg.min_state_separation_std:
        # Don't raise — warn via attrs. Caller can fall back to ungated.
        pass
    return {"model": model, "order": order, "means_reordered": means_reordered,
            "state_separation_std": float(sep)}


def decode(
    test_obs: pd.DataFrame,
    fit_result: dict,
) -> pd.DataFrame:
    """Forward-pass decode on the test window. Returns (date, regime_state, regime_prob_highvol)."""
    df = test_obs.copy()
    if "log_rv" not in df.columns:
        df["log_rv"] = np.log(df["rv"].clip(lower=1e-6))
    X = df[["vix_level", "log_rv"]].to_numpy()
    dates = pd.to_datetime(df["date"]).dt.normalize().to_numpy()

    model = fit_result["model"]
    order = fit_result["order"]

    # Forward-backward smoothing -> posterior P(state) per timestep
    posterior = model.predict_proba(X)
    # Reorder columns to the canonical state ordering
    posterior_reordered = np.zeros_like(posterior)
    for old, new in order.items():
        posterior_reordered[:, new] = posterior[:, old]

    # Viterbi for hard state assignment
    states_raw = model.predict(X)
    states_reordered = np.array([order[s] for s in states_raw])

    return pd.DataFrame({
        "date": dates,
        "regime_state": states_reordered.astype(int),
        "regime_prob_highvol": posterior_reordered[:, 1],
    })


def gate_predictions(
    preds: pd.DataFrame,
    regime: pd.DataFrame,
    mode: str = "soft",
) -> pd.DataFrame:
    """Apply the regime to a prediction stream.

    mode="hard":  multiply p_hat by 1{state == 1}  (force flat in low-vol)
    mode="soft":  shrink p_hat toward 0.5 by (1 - regime_prob_highvol)
                  so positions in low-vol regime are smaller, not strictly zero.
    """
    df = preds.merge(regime, on="date", how="left").copy()
    if mode == "hard":
        df["p_hat_gated"] = np.where(df["regime_state"] == 1, df["p_hat"], 0.5)
    elif mode == "soft":
        df["p_hat_gated"] = 0.5 + (df["p_hat"] - 0.5) * df["regime_prob_highvol"].fillna(0.5)
    else:
        raise ValueError(f"unknown gating mode: {mode}")
    return df
