"""Runtime stability helpers (no torch forward required)."""

from __future__ import annotations

import os
import sys

import pytest

from ml.mace.runtime import coerce_stable_dtype, configure_mace_runtime


def test_configure_mace_runtime_sets_thread_caps() -> None:
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TORCH_NUM_THREADS",
    ):
        os.environ.pop(key, None)
    configure_mace_runtime()
    assert os.environ["OMP_NUM_THREADS"] == "1"
    assert os.environ["TORCH_NUM_THREADS"] == "1"


def test_coerce_stable_dtype_float32() -> None:
    assert coerce_stable_dtype("float32") == "float32"


@pytest.mark.skipif(sys.version_info < (3, 14), reason="coercion only on 3.14+")
def test_coerce_float64_on_py314() -> None:
    with pytest.warns(UserWarning, match="float64 → float32"):
        assert coerce_stable_dtype("float64") == "float32"
