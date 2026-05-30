"""Model classes for the SPY vol-regime walk-forward harness.

All conforming classes implement the `Model` protocol in `models/base.py`:
    name: str
    fit(X, y) -> Model
    predict_proba(X) -> np.ndarray  # 1D, P(y=1) per row

`factory.make_model(type, **kwargs)` wires config strings → class instances.
"""

from models.base import Model
from models.bayesian_head import BayesianHeadModel
from models.factory import available_models, make_model
from models.har_x import HARXClassifier
from models.logistic import LogisticModel
from models.logistic_interactions import LogisticInteractionsModel
from models.mlp_small import SmallMLPModel
from models.xgb_calibrated import XGBCalibratedModel

__all__ = [
    "Model",
    "LogisticModel",
    "LogisticInteractionsModel",
    "HARXClassifier",
    "XGBCalibratedModel",
    "SmallMLPModel",
    "BayesianHeadModel",
    "make_model",
    "available_models",
]
