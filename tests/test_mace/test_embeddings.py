"""MACE embedding pooling tests (Phase 4.8)."""

from __future__ import annotations

import numpy as np
import pytest

from ml.mace.embeddings import (
    PROTOCOL_DESC_MEANSTD_V1,
    feature_column_names,
    pool_descriptors,
)


def test_pool_descriptors_meanstd_shape() -> None:
    per_atom = np.arange(12, dtype=np.float64).reshape(3, 4)
    vec = pool_descriptors(per_atom, protocol=PROTOCOL_DESC_MEANSTD_V1)
    assert vec.shape == (8,)
    np.testing.assert_allclose(vec[:4], per_atom.mean(axis=0))
    np.testing.assert_allclose(vec[4:], per_atom.std(axis=0))


def test_feature_column_names() -> None:
    cols = feature_column_names(6)
    assert cols == [
        "mace_mean_0",
        "mace_mean_1",
        "mace_mean_2",
        "mace_std_0",
        "mace_std_1",
        "mace_std_2",
    ]


def test_pool_rejects_bad_protocol() -> None:
    with pytest.raises(ValueError, match="Unsupported embedding protocol"):
        pool_descriptors(np.ones((2, 3)), protocol="nope")
