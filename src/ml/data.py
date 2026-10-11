"""Load train/val/test matrices from the training export package.

Pipeline:
1. Read a target parquet from ``training_exports/targets/``.
2. Optionally enrich with elemental descriptors from ``formula_pretty``.
3. Resolve feature columns from ``configs/models.yaml``.
4. Impute feature NaNs using **train-split medians only** (no val/test leakage).
5. Pack float64 arrays for LightGBM.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.settings import PROJECT_ROOT
from ml.composition_features import (
    COMP_STAT_COLS,
    ELEMENT_DESC_COLS,
    FRAC_COLS,
    add_element_descriptors,
)
from ml.targets import TargetSpec, get_target, load_models_config


@dataclass(frozen=True)
class SplitData:
    """Feature/label/id arrays for one target, already split."""

    target: str
    feature_cols: tuple[str, ...]
    X_train: np.ndarray
    y_train: np.ndarray
    ids_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    ids_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    ids_test: np.ndarray

    @property
    def n_features(self) -> int:
        return len(self.feature_cols)

    def counts(self) -> dict[str, int]:
        return {
            "train": int(len(self.y_train)),
            "val": int(len(self.y_val)),
            "test": int(len(self.y_test)),
        }


def export_dir(
    config: dict[str, Any] | None = None, *, project_root: Path | None = None
) -> Path:
    """Root directory of the frozen training export package."""
    cfg = config if config is not None else load_models_config()
    root = project_root or PROJECT_ROOT
    rel = Path(cfg.get("export", {}).get("dir", "data/processed/training_exports"))
    return rel if rel.is_absolute() else root / rel


def models_dir(
    config: dict[str, Any] | None = None, *, project_root: Path | None = None
) -> Path:
    """Directory where trained runs / registry are written (gitignored)."""
    cfg = config if config is not None else load_models_config()
    root = project_root or PROJECT_ROOT
    rel = Path(cfg.get("output", {}).get("models_dir", "data/models"))
    return rel if rel.is_absolute() else root / rel


def target_parquet_path(
    spec: TargetSpec,
    config: dict[str, Any] | None = None,
    *,
    project_root: Path | None = None,
) -> Path:
    """Absolute path to one target parquet inside the export package."""
    cfg = config if config is not None else load_models_config()
    sub = cfg.get("export", {}).get("targets_subdir", "targets")
    return export_dir(cfg, project_root=project_root) / sub / spec.parquet


def resolve_feature_columns(config: dict[str, Any] | None = None) -> list[str]:
    """Ordered feature column list from ``models.yaml`` feature flags."""
    cfg = config if config is not None else load_models_config()
    feat_cfg = cfg.get("features") or {}
    cols: list[str] = []
    if feat_cfg.get("use_frac_elements", True):
        cols.extend(FRAC_COLS)
    if feat_cfg.get("use_composition_stats", True):
        cols.extend(COMP_STAT_COLS)
    if feat_cfg.get("use_element_descriptors", True):
        cols.extend(ELEMENT_DESC_COLS)
    for col in feat_cfg.get("extra_cols") or []:
        if col not in cols:
            cols.append(str(col))
    return cols


def load_target_frame(
    target: str | TargetSpec,
    config: dict[str, Any] | None = None,
    *,
    project_root: Path | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load one target parquet (optionally a column subset)."""
    cfg = config if config is not None else load_models_config()
    spec = target if isinstance(target, TargetSpec) else get_target(target, cfg)
    path = target_parquet_path(spec, cfg, project_root=project_root)
    if not path.exists():
        raise FileNotFoundError(f"Missing training export for {spec.name}: {path}")
    return pd.read_parquet(path, columns=columns)


def _enrich_features(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Add descriptor columns when enabled and not already present."""
    feat_cfg = config.get("features") or {}
    if not feat_cfg.get("use_element_descriptors", True):
        return frame
    missing = [c for c in ELEMENT_DESC_COLS if c not in frame.columns]
    if not missing:
        return frame
    return add_element_descriptors(frame, formula_col="formula_pretty")


def _impute_features_train_median(
    work: pd.DataFrame,
    feature_cols: list[str],
    *,
    split_col: str,
    train_value: str,
) -> pd.DataFrame:
    """Fill feature NaNs using medians estimated on the train split only."""
    out = work
    train_mask = out[split_col] == train_value
    for col in feature_cols:
        if not out[col].isna().any():
            continue
        median = float(out.loc[train_mask, col].median(skipna=True))
        if not np.isfinite(median):
            median = 0.0
        out[col] = out[col].fillna(median)
    return out


def load_xy_splits(
    target: str,
    config: dict[str, Any] | None = None,
    *,
    project_root: Path | None = None,
    id_col: str = "material_id",
) -> SplitData:
    """Load features/labels for train, val, and test splits of one target."""
    cfg = config if config is not None else load_models_config()
    spec = get_target(target, cfg)
    split_cfg = cfg.get("splits") or {}
    split_col = str(split_cfg.get("column", "split"))
    train_value = str(split_cfg.get("train_value", "train"))
    val_value = str(split_cfg.get("val_value", "val"))
    test_value = str(split_cfg.get("test_value", "test"))

    frame = load_target_frame(spec, cfg, project_root=project_root)
    frame = _enrich_features(frame, cfg)
    feature_cols = resolve_feature_columns(cfg)
    required = [spec.label_col, split_col, id_col, *feature_cols]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise KeyError(f"{spec.name}: missing columns {missing}")

    work = frame.loc[:, required].copy()
    work = _impute_features_train_median(
        work,
        feature_cols,
        split_col=split_col,
        train_value=train_value,
    )
    work = work.dropna(subset=[spec.label_col, *feature_cols])

    def _pack(split_value: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        part = work.loc[work[split_col] == split_value]
        if part.empty:
            raise ValueError(f"{spec.name}: no rows for split={split_value!r}")
        x = part.loc[:, feature_cols].to_numpy(dtype=np.float64, copy=True)
        y = part.loc[:, spec.label_col].to_numpy(dtype=np.float64, copy=True)
        ids = part.loc[:, id_col].to_numpy(copy=True)
        return x, y, ids

    x_train, y_train, ids_train = _pack(train_value)
    x_val, y_val, ids_val = _pack(val_value)
    x_test, y_test, ids_test = _pack(test_value)

    return SplitData(
        target=spec.name,
        feature_cols=tuple(feature_cols),
        X_train=x_train,
        y_train=y_train,
        ids_train=ids_train,
        X_val=x_val,
        y_val=y_val,
        ids_val=ids_val,
        X_test=x_test,
        y_test=y_test,
        ids_test=ids_test,
    )
