"""Tests for regression metrics helpers."""

from __future__ import annotations

import numpy as np
import pytest

from ml.metrics import metrics_block, regression_metrics


def test_regression_metrics_perfect() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    m = regression_metrics(y, y)
    assert m["mae"] == pytest.approx(0.0)
    assert m["rmse"] == pytest.approx(0.0)
    assert m["r2"] == pytest.approx(1.0)


def test_regression_metrics_known_error() -> None:
    y_true = np.array([0.0, 2.0, 4.0])
    y_pred = np.array([1.0, 1.0, 5.0])
    m = regression_metrics(y_true, y_pred)
    assert m["mae"] == pytest.approx(1.0)
    assert m["rmse"] == pytest.approx(np.sqrt((1 + 1 + 1) / 3))
    assert m["r2"] < 1.0


def test_metrics_block_splits() -> None:
    y = np.array([1.0, 2.0, 3.0])
    pred = y + 0.5
    block = metrics_block(y_train=y, pred_train=pred, y_val=y, pred_val=pred)
    assert "train" in block and "val" in block
    assert "test" not in block
    assert block["train"]["mae"] == pytest.approx(0.5)


def test_regression_metrics_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        regression_metrics(np.array([]), np.array([]))
