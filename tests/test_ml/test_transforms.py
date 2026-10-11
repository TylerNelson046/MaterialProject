"""Tests for label transforms."""

from __future__ import annotations

import numpy as np
import pytest

from ml.transforms import inverse_transform_y, normalize_transform, transform_y


def test_log1p_roundtrip() -> None:
    y = np.array([0.0, 0.5, 2.0, 10.0])
    z = transform_y(y, "log1p")
    back = inverse_transform_y(z, "log1p")
    assert back == pytest.approx(y)


def test_log1p_clips_negatives() -> None:
    y = np.array([-0.1, 1.0])
    z = transform_y(y, "log1p")
    assert z[0] == pytest.approx(0.0)


def test_identity() -> None:
    y = np.array([1.0, -2.0])
    assert transform_y(y, None) == pytest.approx(y)
    assert normalize_transform("identity") is None


def test_unknown_transform() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        normalize_transform("boxcox")
