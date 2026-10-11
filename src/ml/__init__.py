"""LightGBM property-modeling package.

Modules
-------
composition_features
    Formula → tabular features (fractions, stats, descriptors).
targets
    Ready / Niche registry from ``configs/models.yaml``.
data
    Load ``training_exports`` into train/val/test arrays.
transforms
    Optional label maps (e.g. log1p) + inverses.
metrics
    MAE / RMSE / R² helpers.
train
    Fit runs, write artifacts, update registry.
predict
    Load runs and predict from formulas or feature matrices.
registry
    ``registry.json`` + ``LEADERBOARD.md`` helpers.
mace
    Optional MACE helpers (``pip install -e ".[mace]"``): smoke, structure I/O,
    audits — see ``docs/mace_training.md``.
"""

from ml.composition_features import (
    COMP_STAT_COLS,
    ELEMENT_DESC_COLS,
    FRAC_COLS,
    add_composition_features,
)
from ml.data import SplitData, load_xy_splits, models_dir, resolve_feature_columns
from ml.metrics import metrics_block, regression_metrics
from ml.predict import (
    LoadedModel,
    load_model,
    predict_formulas,
    predict_matrix,
    predict_with_model,
)
from ml.targets import (
    TargetSpec,
    get_target,
    known_targets,
    list_targets,
    niche_targets,
    ready_targets,
)
from ml.train import TrainResult, train_many, train_target

__all__ = [
    "COMP_STAT_COLS",
    "ELEMENT_DESC_COLS",
    "FRAC_COLS",
    "LoadedModel",
    "SplitData",
    "TargetSpec",
    "TrainResult",
    "add_composition_features",
    "get_target",
    "known_targets",
    "list_targets",
    "load_model",
    "load_xy_splits",
    "metrics_block",
    "models_dir",
    "niche_targets",
    "predict_formulas",
    "predict_matrix",
    "predict_with_model",
    "ready_targets",
    "regression_metrics",
    "resolve_feature_columns",
    "train_many",
    "train_target",
]
