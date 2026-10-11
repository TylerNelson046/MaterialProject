"""Correlation metric helpers."""

from __future__ import annotations

import numpy as np
import pytest

from ml.metrics import correlation_metrics


def test_correlation_perfect_linear() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    metrics = correlation_metrics(y, 2.0 * y + 1.0)
    assert metrics["pearson_r"] == pytest.approx(1.0)
    assert metrics["spearman_r"] == pytest.approx(1.0)
