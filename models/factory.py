"""Map `configs/experiment.yaml` `type:` strings to model class instances.

Used by `backtest/walk_forward.run` via a zero-arg lambda:
    model_factory = lambda: make_model(spec["type"], **spec.get("hyperparams", {}))
"""

from __future__ import annotations

from models.bayesian_head import BayesianHeadModel
from models.har_x import HARXClassifier
from models.logistic import LogisticModel
from models.logistic_interactions import LogisticInteractionsModel
from models.mlp_small import SmallMLPModel
from models.xgb_calibrated import XGBCalibratedModel


_REGISTRY = {
    "logistic": LogisticModel,
    "logistic_interactions": LogisticInteractionsModel,
    "har_x": HARXClassifier,
    "xgb_calibrated": XGBCalibratedModel,
    "mlp_small": SmallMLPModel,
    "bayesian_head": BayesianHeadModel,
}


def make_model(model_type: str, **hyperparams):
    """Instantiate a model from its config type string."""
    if model_type not in _REGISTRY:
        raise ValueError(
            f"unknown model_type {model_type!r}; "
            f"available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[model_type](**hyperparams)


def available_models() -> list[str]:
    return sorted(_REGISTRY)
