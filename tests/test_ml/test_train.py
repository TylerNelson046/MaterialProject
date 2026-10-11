"""Tests for LightGBM training + registry."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ml.composition_features import COMP_STAT_COLS, FRAC_COLS
from ml.registry import latest_summary_rows, load_registry
from ml.train import train_target


def _write_mini_export(root: Path) -> Path:
    """Create a tiny training_exports layout + models.yaml under root."""
    export = root / "data" / "processed" / "training_exports" / "targets"
    export.mkdir(parents=True)

    n = 240
    rng = np.random.default_rng(0)
    # Label depends on first two frac features so the model can fit something.
    frac = rng.random((n, len(FRAC_COLS)))
    frac = frac / frac.sum(axis=1, keepdims=True)
    y = 2.0 * frac[:, 0] + 0.5 * frac[:, 1] + rng.normal(0, 0.005, size=n)
    splits = np.array(["train"] * 160 + ["val"] * 40 + ["test"] * 40)

    frame = pd.DataFrame(frac, columns=list(FRAC_COLS))
    for col in COMP_STAT_COLS:
        frame[col] = rng.random(n)
    frame["material_id"] = [f"MAT-{i}" for i in range(n)]
    frame["formula_pretty"] = ["SiO2" if i % 2 == 0 else "Fe2O3" for i in range(n)]
    frame["label"] = y
    frame["split"] = splits
    frame.to_parquet(export / "band_gap.parquet", index=False)

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
            "seed": 0,
            "default_tier": "Ready",
            "early_stopping_rounds": 20,
            "lightgbm": {
                "n_estimators": 200,
                "learning_rate": 0.1,
                "num_leaves": 31,
                "min_child_samples": 5,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "reg_lambda": 0.0,
                "random_state": 0,
                "n_jobs": 1,
                "verbose": -1,
            },
        },
        "targets": {
            "band_gap": {
                "tier": "Ready",
                "parquet": "band_gap.parquet",
                "label_col": "label",
                "transform": "log1p",
            }
        },
    }
    cfg_path = root / "configs" / "models.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg_path


def test_train_target_writes_artifacts(tmp_path: Path) -> None:
    cfg_path = _write_mini_export(tmp_path)
    config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    result = train_target(
        "band_gap",
        config,
        project_root=tmp_path,
        write_predictions=True,
    )

    assert result.run_dir.is_dir()
    assert result.model_path.is_file()
    assert (result.run_dir / "metrics.json").is_file()
    assert (result.run_dir / "feature_cols.json").is_file()
    assert (result.run_dir / "run_record.json").is_file()
    assert (result.run_dir / "predictions_test.parquet").is_file()
    assert "test" in result.metrics
    assert result.metrics["test"]["r2"] > 0.5

    models = tmp_path / "data" / "models"
    registry = load_registry(models)
    assert "band_gap" in registry["runs"]
    rows = latest_summary_rows(models)
    assert rows[0]["target"] == "band_gap"
    assert (models / "LEADERBOARD.md").is_file()
