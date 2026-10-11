"""Regression metrics for property model evaluation.

All metrics are computed in the caller-provided units (typically original
physical units after any inverse label transform).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return MAE, RMSE, and R² for a regression head."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )
    if y_true.size == 0:
        raise ValueError("Cannot score empty arrays")

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"mae": mae, "rmse": rmse, "r2": r2}


def metrics_block(
    *,
    y_train: np.ndarray | None = None,
    pred_train: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    pred_val: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    pred_test: np.ndarray | None = None,
) -> dict[str, Any]:
    """Bundle per-split metrics under ``train`` / ``val`` / ``test`` keys."""
    out: dict[str, Any] = {}
    if y_train is not None and pred_train is not None:
        out["train"] = regression_metrics(y_train, pred_train)
    if y_val is not None and pred_val is not None:
        out["val"] = regression_metrics(y_val, pred_val)
    if y_test is not None and pred_test is not None:
        out["test"] = regression_metrics(y_test, pred_test)
    return out
