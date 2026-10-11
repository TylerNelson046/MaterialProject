"""Load trained LightGBM runs and predict property values.

By default the latest registered run for a target is used
(``data/models/registry.json``). Formula inference rebuilds the same
composition feature vector used at training time, then applies any inverse
label transform stored in ``run_record.json``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from config.settings import PROJECT_ROOT
from ml.composition_features import composition_feature_row
from ml.data import models_dir
from ml.io_util import read_json
from ml.registry import load_registry
from ml.targets import get_target, load_models_config
from ml.transforms import inverse_transform_y, normalize_transform


@dataclass(frozen=True)
class LoadedModel:
    """A trained run ready for inference."""

    target: str
    tier: str
    run_id: str
    run_dir: Path
    model: Any
    feature_cols: tuple[str, ...]
    metrics: dict[str, Any]
    transform: str | None


def latest_run_dir(
    target: str,
    config: dict[str, Any] | None = None,
    *,
    project_root: Path | None = None,
) -> Path:
    """Resolve the most recently registered run directory for a target."""
    cfg = config if config is not None else load_models_config()
    root = project_root or PROJECT_ROOT
    get_target(target, cfg)  # validate name early
    registry = load_registry(models_dir(cfg, project_root=root))
    runs = (registry.get("runs") or {}).get(target) or []
    if not runs:
        raise FileNotFoundError(
            f"No registered runs for target {target!r} under "
            f"{models_dir(cfg, project_root=root)}"
        )
    record = runs[-1]
    run_dir = Path(record["run_dir"])
    if not run_dir.is_absolute():
        run_dir = root / run_dir
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory missing for {target}: {run_dir}")
    return run_dir


def load_model(
    target: str,
    config: dict[str, Any] | None = None,
    *,
    project_root: Path | None = None,
    run_dir: Path | None = None,
) -> LoadedModel:
    """Load ``model.joblib`` + metadata for a target (latest run by default)."""
    cfg = config if config is not None else load_models_config()
    root = project_root or PROJECT_ROOT
    spec = get_target(target, cfg)
    directory = Path(run_dir) if run_dir is not None else latest_run_dir(
        target, cfg, project_root=root
    )
    if not directory.is_absolute():
        directory = root / directory

    model_path = directory / "model.joblib"
    feature_path = directory / "feature_cols.json"
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing model artifact: {model_path}")
    if not feature_path.is_file():
        raise FileNotFoundError(f"Missing feature_cols.json: {feature_path}")

    feature_cols = tuple(read_json(feature_path))
    metrics = read_json(directory / "metrics.json") or {}
    record = read_json(directory / "run_record.json") or {}
    model = joblib.load(model_path)
    transform = normalize_transform(record.get("transform"))

    return LoadedModel(
        target=spec.name,
        tier=str(record.get("tier") or spec.tier),
        run_id=str(record.get("run_id") or directory.name),
        run_dir=directory,
        model=model,
        feature_cols=feature_cols,
        metrics=metrics if isinstance(metrics, dict) else {},
        transform=transform,
    )


def features_from_formula(
    formula: str, feature_cols: tuple[str, ...] | list[str]
) -> np.ndarray:
    """Build a single feature row from a chemical formula."""
    row = composition_feature_row(formula)
    missing = [c for c in feature_cols if c not in row]
    if missing:
        raise KeyError(f"Formula features missing columns: {missing[:8]}")
    values: list[float] = []
    for col in feature_cols:
        val = float(row[col])
        # Match training imputation fallback for rare missing elemental props.
        if not np.isfinite(val):
            val = 0.0
        values.append(val)
    return np.asarray(values, dtype=np.float64).reshape(1, -1)


def _predict_loaded(loaded: LoadedModel, X: np.ndarray) -> np.ndarray:
    raw = np.asarray(loaded.model.predict(X), dtype=np.float64)
    return inverse_transform_y(raw, loaded.transform)


def predict_with_model(
    loaded: LoadedModel,
    formulas: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    """Predict formulas with an already-loaded run (avoids re-reading artifacts)."""
    rows: list[dict[str, Any]] = []
    for formula in formulas:
        x = features_from_formula(formula, loaded.feature_cols)
        pred = float(_predict_loaded(loaded, x)[0])
        rows.append(
            {
                "formula": formula,
                "target": loaded.target,
                "prediction": pred,
                "run_id": loaded.run_id,
            }
        )
    return pd.DataFrame(rows)


def predict_formulas(
    target: str,
    formulas: list[str] | tuple[str, ...],
    config: dict[str, Any] | None = None,
    *,
    project_root: Path | None = None,
    run_dir: Path | None = None,
) -> pd.DataFrame:
    """Predict a property for one or more formulas using a trained run."""
    loaded = load_model(target, config, project_root=project_root, run_dir=run_dir)
    return predict_with_model(loaded, formulas)


def predict_matrix(
    target: str,
    X: np.ndarray,
    config: dict[str, Any] | None = None,
    *,
    project_root: Path | None = None,
    run_dir: Path | None = None,
) -> np.ndarray:
    """Predict from a feature matrix aligned with the run's ``feature_cols``."""
    loaded = load_model(target, config, project_root=project_root, run_dir=run_dir)
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != len(loaded.feature_cols):
        raise ValueError(
            f"Expected X shape (n, {len(loaded.feature_cols)}), got {X.shape}"
        )
    return _predict_loaded(loaded, X)
