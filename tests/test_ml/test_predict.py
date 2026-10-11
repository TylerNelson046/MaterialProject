"""Tests for model loading and formula prediction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from ml.composition_features import COMP_STAT_COLS, FRAC_COLS
from ml.predict import features_from_formula, load_model, predict_formulas, predict_matrix
from ml.train import train_target


def _mini_project(root: Path) -> dict:
    export = root / "data" / "processed" / "training_exports" / "targets"
    export.mkdir(parents=True)
    n = 240
    rng = np.random.default_rng(1)
    frac = rng.random((n, len(FRAC_COLS)))
    frac = frac / frac.sum(axis=1, keepdims=True)
    y = 3.0 * frac[:, 0] + rng.normal(0, 0.005, size=n)
    splits = np.array(["train"] * 160 + ["val"] * 40 + ["test"] * 40)
    frame = pd.DataFrame(frac, columns=list(FRAC_COLS))
    for col in COMP_STAT_COLS:
        frame[col] = rng.random(n)
    frame["material_id"] = [f"MAT-{i}" for i in range(n)]
    frame["formula_pretty"] = ["SiO2" if i % 2 == 0 else "Al2O3" for i in range(n)]
    frame["label"] = y
    frame["split"] = splits
    frame.to_parquet(export / "density.parquet", index=False)

    cfg = {
        "export": {
            "dir": "data/processed/training_exports",
            "targets_subdir": "targets",
            "splits_file": "splits.parquet",
        },
        "output": {"models_dir": "data/models"},
        "features": {
            "use_frac_elements": True,
            "use_composition_stats": True,
            "use_element_descriptors": True,
            "extra_cols": [],
        },
        "splits": {
            "column": "split",
            "train_value": "train",
            "val_value": "val",
            "test_value": "test",
        },
        "training": {
            "seed": 1,
            "default_tier": "Ready",
            "early_stopping_rounds": 15,
            "lightgbm": {
                "n_estimators": 120,
                "learning_rate": 0.1,
                "num_leaves": 31,
                "min_child_samples": 5,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "reg_lambda": 0.0,
                "random_state": 1,
                "n_jobs": 1,
                "verbose": -1,
            },
        },
        "targets": {
            "density": {
                "tier": "Ready",
                "parquet": "density.parquet",
                "label_col": "label",
            }
        },
    }
    cfg_path = root / "configs" / "models.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg


def test_predict_formulas_after_train(tmp_path: Path) -> None:
    config = _mini_project(tmp_path)
    train_target("density", config, project_root=tmp_path, write_predictions=False)

    loaded = load_model("density", config, project_root=tmp_path)
    assert loaded.target == "density"
    assert loaded.run_dir.is_dir()

    frame = predict_formulas(
        "density",
        ["SiO2", "Fe2O3"],
        config,
        project_root=tmp_path,
    )
    assert list(frame["formula"]) == ["SiO2", "Fe2O3"]
    assert frame["prediction"].notna().all()
    assert (frame["target"] == "density").all()

    x = features_from_formula("SiO2", loaded.feature_cols)
    y_mat = predict_matrix("density", x, config, project_root=tmp_path)
    assert y_mat.shape == (1,)
    assert float(y_mat[0]) == pytest.approx(float(frame.loc[0, "prediction"]))


def test_predict_matrix_shape_guard(tmp_path: Path) -> None:
    config = _mini_project(tmp_path)
    train_target("density", config, project_root=tmp_path, write_predictions=False)
    loaded = load_model("density", config, project_root=tmp_path)
    bad = np.zeros((2, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="Expected X shape"):
        predict_matrix(
            "density",
            bad,
            config,
            project_root=tmp_path,
            run_dir=loaded.run_dir,
        )
