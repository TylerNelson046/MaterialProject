"""Tests for training-export loaders."""

from __future__ import annotations

import numpy as np
import pytest

from ml.composition_features import COMP_STAT_COLS, ELEMENT_DESC_COLS, FRAC_COLS
from ml.data import (
    export_dir,
    load_target_frame,
    load_xy_splits,
    resolve_feature_columns,
    target_parquet_path,
)
from ml.targets import get_target, load_models_config


def test_resolve_feature_columns_default() -> None:
    cols = resolve_feature_columns()
    assert cols[:3] == list(FRAC_COLS[:3])
    assert COMP_STAT_COLS[0] in cols
    assert ELEMENT_DESC_COLS[0] in cols
    assert len(cols) == len(FRAC_COLS) + len(COMP_STAT_COLS) + len(ELEMENT_DESC_COLS)


def test_export_paths_exist() -> None:
    cfg = load_models_config()
    root = export_dir(cfg)
    assert root.is_dir()
    spec = get_target("band_gap", cfg)
    path = target_parquet_path(spec, cfg)
    assert path.is_file()


def test_load_xy_splits_band_gap_shapes() -> None:
    data = load_xy_splits("band_gap")
    assert data.target == "band_gap"
    assert data.n_features == (
        len(FRAC_COLS) + len(COMP_STAT_COLS) + len(ELEMENT_DESC_COLS)
    )
    counts = data.counts()
    assert counts["train"] > counts["val"] > 0
    assert counts["test"] > 0
    assert data.X_train.shape == (counts["train"], data.n_features)
    assert data.y_train.shape == (counts["train"],)
    assert data.ids_train.shape == (counts["train"],)
    assert data.X_val.shape[0] == counts["val"]
    assert data.X_test.shape[0] == counts["test"]
    # No NaNs in packed matrices
    assert np.isfinite(data.X_train).all()
    assert np.isfinite(data.y_train).all()


def test_load_xy_splits_unknown_target() -> None:
    with pytest.raises(KeyError, match="Unknown target"):
        load_xy_splits("not_a_real_target")


def test_load_target_frame_column_subset() -> None:
    frame = load_target_frame("density", columns=["material_id", "label", "split"])
    assert list(frame.columns) == ["material_id", "label", "split"]
    assert len(frame) > 0
