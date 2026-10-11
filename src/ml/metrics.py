"""Regression metrics for property model evaluation.

All metrics are computed in the caller-provided units (typically original
physical units after any inverse label transform).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr
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


def correlation_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return Pearson and Spearman correlations (and p-values)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )
    if y_true.size < 2:
        raise ValueError("Need at least 2 points for correlation")
    # Constant arrays → undefined correlation; return NaNs rather than raise.
    if np.allclose(y_true, y_true[0]) or np.allclose(y_pred, y_pred[0]):
        return {
            "pearson_r": float("nan"),
            "pearson_p": float("nan"),
            "spearman_r": float("nan"),
            "spearman_p": float("nan"),
        }
    pr, pp = pearsonr(y_true, y_pred)
    sr, sp = spearmanr(y_true, y_pred)
    return {
        "pearson_r": float(pr),
        "pearson_p": float(pp),
        "spearman_r": float(sr),
        "spearman_p": float(sp),
    }


def _split_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Regression metrics plus correlations when n ≥ 2."""
    block = regression_metrics(y_true, y_pred)
    if np.asarray(y_true).size >= 2:
        block.update(correlation_metrics(y_true, y_pred))
    return block


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
        out["train"] = _split_metrics(y_train, pred_train)
    if y_val is not None and pred_val is not None:
        out["val"] = _split_metrics(y_val, pred_val)
    if y_test is not None and pred_test is not None:
        out["test"] = _split_metrics(y_test, pred_test)
    return out
