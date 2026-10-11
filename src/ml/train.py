"""Train LightGBM regressors on ``training_exports`` targets.

Each run writes artifacts under ``data/models/<target>/<run_id>/`` and appends
a record to ``registry.json``. Labels may be transformed for fitting (e.g.
``log1p``); reported metrics are always in original physical units.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from config.settings import PROJECT_ROOT
from ml.data import SplitData, load_xy_splits, models_dir
from ml.io_util import relpath_under, write_json
from ml.metrics import metrics_block
from ml.registry import register_run
from ml.targets import get_target, list_targets, load_models_config, ready_targets
from ml.transforms import inverse_transform_y, normalize_transform, transform_y


@dataclass(frozen=True)
class TrainResult:
    """Artifacts and scores from one training run."""

    target: str
    tier: str
    run_id: str
    run_dir: Path
    model_path: Path
    metrics: dict[str, Any]
    feature_cols: tuple[str, ...]
    counts: dict[str, int]
    transform: str | None
    best_iteration: int | None


def lightgbm_params(
    config: dict[str, Any] | None = None,
    *,
    target: str | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Merge global LightGBM params with optional per-target overrides."""
    cfg = config if config is not None else load_models_config()
    params = dict((cfg.get("training") or {}).get("lightgbm") or {})
    if target is not None:
        target_meta = (cfg.get("targets") or {}).get(target) or {}
        target_params = target_meta.get("lightgbm") or {}
        if isinstance(target_params, dict):
            params.update(target_params)
    params.update({k: v for k, v in overrides.items() if v is not None})
    return params


def _run_id(target: str, when: datetime | None = None) -> str:
    ts = when or datetime.now(UTC)
    return f"{target}_{ts.strftime('%Y%m%dT%H%M%SZ')}"


def _maybe_subsample(data: SplitData, max_train_rows: int | None, seed: int) -> SplitData:
    """Optionally downsample the train split for smoke / debug runs."""
    if max_train_rows is None or len(data.y_train) <= max_train_rows:
        return data
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(data.y_train), size=max_train_rows, replace=False))
    return SplitData(
        target=data.target,
        feature_cols=data.feature_cols,
        X_train=data.X_train[idx],
        y_train=data.y_train[idx],
        ids_train=data.ids_train[idx],
        X_val=data.X_val,
        y_val=data.y_val,
        ids_val=data.ids_val,
        X_test=data.X_test,
        y_test=data.y_test,
        ids_test=data.ids_test,
    )


def _predict_original(
    model: lgb.LGBMRegressor,
    X: np.ndarray,
    transform: str | None,
) -> np.ndarray:
    """Model predict + inverse label transform → original units."""
    raw = np.asarray(model.predict(X), dtype=np.float64)
    return inverse_transform_y(raw, transform)


def train_target(
    target: str,
    config: dict[str, Any] | None = None,
    *,
    project_root: Path | None = None,
    write_predictions: bool = True,
    max_train_rows: int | None = None,
    lgb_overrides: dict[str, Any] | None = None,
) -> TrainResult:
    """Train one target head and write artifacts under ``data/models/``."""
    cfg = config if config is not None else load_models_config()
    root = project_root or PROJECT_ROOT
    spec = get_target(target, cfg)
    seed = int((cfg.get("training") or {}).get("seed", 42))
    transform = normalize_transform(spec.transform)
    early_stopping_rounds = int(
        (cfg.get("training") or {}).get("early_stopping_rounds", 50)
    )

    data = load_xy_splits(spec.name, cfg, project_root=root)
    data = _maybe_subsample(data, max_train_rows, seed)

    params = lightgbm_params(cfg, target=spec.name, **(lgb_overrides or {}))
    params.setdefault("random_state", seed)

    # Fit in transformed space; score in original units.
    y_train_m = transform_y(data.y_train, transform)
    y_val_m = transform_y(data.y_val, transform)

    model = lgb.LGBMRegressor(**params)
    callbacks = []
    if early_stopping_rounds > 0:
        callbacks.append(
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)
        )
    model.fit(
        data.X_train,
        y_train_m,
        eval_X=data.X_val,
        eval_y=y_val_m,
        callbacks=callbacks or None,
    )

    pred_train = _predict_original(model, data.X_train, transform)
    pred_val = _predict_original(model, data.X_val, transform)
    pred_test = _predict_original(model, data.X_test, transform)
    metrics = metrics_block(
        y_train=data.y_train,
        pred_train=pred_train,
        y_val=data.y_val,
        pred_val=pred_val,
        y_test=data.y_test,
        pred_test=pred_test,
    )

    best_iteration = getattr(model, "best_iteration_", None)
    if best_iteration is None:
        best_iteration = getattr(model, "best_iteration", None)

    run_id = _run_id(spec.name)
    out_root = models_dir(cfg, project_root=root)
    run_dir = out_root / spec.name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    model_path = run_dir / "model.joblib"
    joblib.dump(model, model_path)
    write_json(run_dir / "feature_cols.json", list(data.feature_cols))
    write_json(run_dir / "metrics.json", metrics)

    if write_predictions:
        for split_name, ids, y_true, y_pred in (
            ("train", data.ids_train, data.y_train, pred_train),
            ("val", data.ids_val, data.y_val, pred_val),
            ("test", data.ids_test, data.y_test, pred_test),
        ):
            frame = pd.DataFrame(
                {
                    "material_id": ids,
                    "y_true": y_true,
                    "y_pred": y_pred,
                }
            )
            frame.to_parquet(run_dir / f"predictions_{split_name}.parquet", index=False)

    record = {
        "target": spec.name,
        "tier": spec.tier,
        "run_id": run_id,
        "run_dir": relpath_under(run_dir, root),
        "model_path": relpath_under(model_path, root),
        "created_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": seed,
        "transform": transform,
        "early_stopping_rounds": early_stopping_rounds,
        "best_iteration": best_iteration,
        "lightgbm": params,
        "feature_cols": list(data.feature_cols),
        "n_features": len(data.feature_cols),
        "counts": data.counts(),
        "metrics": metrics,
    }
    write_json(run_dir / "run_record.json", record)
    register_run(out_root, record)

    return TrainResult(
        target=spec.name,
        tier=spec.tier,
        run_id=run_id,
        run_dir=run_dir,
        model_path=model_path,
        metrics=metrics,
        feature_cols=data.feature_cols,
        counts=data.counts(),
        transform=transform,
        best_iteration=int(best_iteration) if best_iteration is not None else None,
    )


def train_many(
    targets: list[str] | tuple[str, ...],
    config: dict[str, Any] | None = None,
    *,
    project_root: Path | None = None,
    write_predictions: bool = True,
    max_train_rows: int | None = None,
    lgb_overrides: dict[str, Any] | None = None,
) -> list[TrainResult]:
    """Train multiple targets sequentially."""
    return [
        train_target(
            name,
            config,
            project_root=project_root,
            write_predictions=write_predictions,
            max_train_rows=max_train_rows,
            lgb_overrides=lgb_overrides,
        )
        for name in targets
    ]


def default_train_targets(config: dict[str, Any] | None = None) -> tuple[str, ...]:
    """Targets implied by ``training.default_tier`` (usually Ready)."""
    cfg = config if config is not None else load_models_config()
    tier = str((cfg.get("training") or {}).get("default_tier", "Ready"))
    if tier.lower() == "ready":
        return ready_targets(cfg)
    return tuple(spec.name for spec in list_targets(cfg, tier=tier))
