"""Train a LightGBM readout on cached MACE embeddings (Phase 4.8)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

from config.settings import PROJECT_ROOT
from ml.io_util import relpath_under, write_json
from ml.mace.compare import compare_lightgbm_mace
from ml.mace.config import load_mace_config
from ml.mace.embeddings import (
    PROTOCOL_DESC_MEANSTD_V1,
    default_embeddings_cache_path,
    extract_embeddings,
    load_embeddings_frame,
)
from ml.mace.pack import mace_pack_dir
from ml.metrics import correlation_metrics, regression_metrics
from ml.registry import load_registry, register_run, write_leaderboard
from ml.train import lightgbm_params
from ml.transforms import inverse_transform_y, normalize_transform, transform_y


def _patch_latest_registry_record(
    models_dir: Path,
    *,
    target: str,
    run_id: str,
    patch: dict[str, Any],
) -> None:
    """Merge ``patch`` into the matching registry run (by run_id)."""
    registry = load_registry(models_dir)
    runs = list((registry.get("runs") or {}).get(target) or [])
    for i in range(len(runs) - 1, -1, -1):
        if str(runs[i].get("run_id")) == run_id:
            updated = dict(runs[i])
            updated.update(patch)
            runs[i] = updated
            break
    registry.setdefault("runs", {})[target] = runs
    write_json(models_dir / "registry.json", registry)
    write_leaderboard(models_dir)


@dataclass(frozen=True)
class EmbeddingHeadResult:
    """Artifacts from one embedding-head training run."""

    target: str
    run_id: str
    run_dir: Path
    protocol: str
    metrics: dict[str, Any]
    counts: dict[str, int]
    promoted: bool | None
    compare_report: Path | None


def _run_id(target: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{target}_mace_embedding_head_{stamp}"


def _models_root(config: dict[str, Any], project_root: Path) -> Path:
    output = config.get("output") or {}
    return project_root / str(output.get("models_dir") or "data/models")


def _mace_run_dir(
    config: dict[str, Any], target: str, run_id: str, project_root: Path
) -> Path:
    output = config.get("output") or {}
    models_dir = _models_root(config, project_root)
    sub = str(output.get("mace_subdir") or "mace")
    return models_dir / sub / target / run_id


def _label_transform(target: str, config: dict[str, Any]) -> str | None:
    spec = (config.get("targets") or {}).get(target) or {}
    if "transform" in spec:
        return normalize_transform(spec.get("transform"))
    # Fall back to LightGBM models.yaml transform via ml.targets if available.
    try:
        from ml.targets import get_target

        meta = get_target(target)
        return normalize_transform(meta.get("transform"))
    except Exception:  # noqa: BLE001
        return None


def _tier(target: str, config: dict[str, Any]) -> str:
    spec = (config.get("targets") or {}).get(target) or {}
    if spec.get("tier"):
        return str(spec["tier"])
    try:
        from ml.targets import get_target

        return str(get_target(target).get("tier") or "Ready")
    except Exception:  # noqa: BLE001
        return "Ready"


def train_embedding_head(
    target: str = "band_gap",
    *,
    protocol: str = PROTOCOL_DESC_MEANSTD_V1,
    cache_path: Path | None = None,
    extract_if_missing: bool = True,
    limit_per_split: int | None = None,
    max_train_rows: int | None = None,
    device: str | None = "cpu",
    model_id: str | None = None,
    run_id: str | None = None,
    run_dir: Path | None = None,
    config: dict[str, Any] | None = None,
    project_root: Path | None = None,
    register: bool = True,
    compare: bool = True,
    progress_every: int = 200,
) -> EmbeddingHeadResult:
    """Fit LightGBM on MACE embeddings; register as ``mace_embedding_head``."""
    cfg = config if config is not None else load_mace_config()
    root = project_root or PROJECT_ROOT
    transform = _label_transform(target, cfg)
    tier = _tier(target, cfg)

    resolved_cache = cache_path or default_embeddings_cache_path(
        target, protocol=protocol, model_id=model_id, config=cfg
    )
    if not resolved_cache.is_absolute():
        resolved_cache = root / resolved_cache
    if not resolved_cache.is_file():
        if not extract_if_missing:
            raise FileNotFoundError(f"Embedding cache missing: {resolved_cache}")
        extract_embeddings(
            target,
            protocol=protocol,
            limit_per_split=limit_per_split,
            device=device,
            model_id=model_id,
            cache_path=resolved_cache,
            config=cfg,
            project_root=root,
            progress_every=progress_every,
            overwrite=True,
        )

    frame = load_embeddings_frame(
        target, protocol=protocol, cache_path=resolved_cache, config=cfg
    )
    feature_cols = [c for c in frame.columns if c.startswith("mace_")]
    if not feature_cols:
        raise ValueError(f"No mace_* feature columns in {resolved_cache}")

    def _split(name: str) -> pd.DataFrame:
        return frame.loc[frame["split"].astype(str) == name].copy()

    train = _split("train")
    val = _split("val")
    test = _split("test")
    if max_train_rows is not None and len(train) > int(max_train_rows):
        train = train.sample(n=int(max_train_rows), random_state=42)

    if len(train) < 10 or len(val) < 2 or len(test) < 2:
        raise RuntimeError(
            f"Need enough embedding rows (train≥10, val/test≥2); "
            f"got train={len(train)} val={len(val)} test={len(test)}. "
            "Increase --limit-per-split or extract a fuller cache."
        )

    X_train = train.loc[:, feature_cols].to_numpy(dtype=np.float64)
    y_train = transform_y(train["label"].to_numpy(dtype=np.float64), transform)
    X_val = val.loc[:, feature_cols].to_numpy(dtype=np.float64)
    y_val = transform_y(val["label"].to_numpy(dtype=np.float64), transform)
    X_test = test.loc[:, feature_cols].to_numpy(dtype=np.float64)
    y_test_orig = test["label"].to_numpy(dtype=np.float64)

    params = lightgbm_params(target=target)
    # Embedding dim is modest; keep trees reasonable for smoke/full alike.
    training_cfg = cfg.get("training") or {}
    early_stopping_rounds = int(training_cfg.get("early_stopping_rounds", 40))
    model = lgb.LGBMRegressor(**params)
    fit_kwargs: dict[str, Any] = {
        "eval_metric": "l1",
    }
    # Prefer modern eval_X/eval_y; fall back to eval_set for older LightGBM.
    try:
        model.fit(
            X_train,
            y_train,
            eval_X=[X_val],
            eval_y=[y_val],
            callbacks=(
                [
                    lgb.early_stopping(
                        stopping_rounds=early_stopping_rounds, verbose=False
                    )
                ]
                if early_stopping_rounds > 0
                else None
            ),
            **fit_kwargs,
        )
    except TypeError:
        if early_stopping_rounds > 0:
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(
                    stopping_rounds=early_stopping_rounds, verbose=False
                )
            ]
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            **fit_kwargs,
        )

    def _pred(X: np.ndarray) -> np.ndarray:
        return inverse_transform_y(
            np.asarray(model.predict(X), dtype=np.float64), transform
        )

    y_val_pred = _pred(X_val)
    y_test_pred = _pred(X_test)
    y_val_orig = val["label"].to_numpy(dtype=np.float64)

    split_metrics = {
        "val": {
            **regression_metrics(y_val_orig, y_val_pred),
            **correlation_metrics(y_val_orig, y_val_pred),
        },
        "test": {
            **regression_metrics(y_test_orig, y_test_pred),
            **correlation_metrics(y_test_orig, y_test_pred),
        },
    }
    _corr_keys = ("pearson_r", "pearson_p", "spearman_r", "spearman_p")
    registry_metrics = {
        split: {
            "mae": split_metrics[split]["mae"],
            "rmse": split_metrics[split]["rmse"],
            "r2": split_metrics[split]["r2"],
            **{k: split_metrics[split][k] for k in _corr_keys if k in split_metrics[split]},
        }
        for split in ("val", "test")
    }
    counts = {
        "train": int(len(train)),
        "val": int(len(val)),
        "test": int(len(test)),
    }

    resolved_run_id = run_id or _run_id(target)
    resolved_run_dir = run_dir or _mace_run_dir(cfg, target, resolved_run_id, root)
    resolved_run_dir.mkdir(parents=True, exist_ok=True)

    model_path = resolved_run_dir / "head.joblib"
    joblib.dump(model, model_path)
    write_json(resolved_run_dir / "feature_names.json", {"feature_cols": feature_cols})
    write_json(
        resolved_run_dir / "descriptor_protocol.json",
        {
            "protocol": protocol,
            "cache_path": relpath_under(resolved_cache, root),
            "n_features": len(feature_cols),
            "label_transform": transform,
        },
    )
    (resolved_run_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=True),
        encoding="utf-8",
    )

    for split_name, part, y_true, y_pred in (
        ("val", val, y_val_orig, y_val_pred),
        ("test", test, y_test_orig, y_test_pred),
    ):
        pred = pd.DataFrame(
            {
                "material_id": part["material_id"].astype(str).to_numpy(),
                "y_true": y_true,
                "y_pred": y_pred,
                "split": split_name,
            }
        )
        pred.to_parquet(resolved_run_dir / f"predictions_{split_name}.parquet", index=False)

    metrics_doc = {
        "protocol": protocol,
        "model_family": "mace_embedding_head",
        "label_transform": transform,
        "feature_cols": feature_cols,
        "embedding_cache": relpath_under(resolved_cache, root),
        "splits": split_metrics,
        "primary": target,
        "registry_metrics": registry_metrics,
        "lightgbm_params": params,
        "early_stopping_rounds": early_stopping_rounds,
        "best_iteration": getattr(model, "best_iteration_", None),
    }
    write_json(resolved_run_dir / "metrics.json", metrics_doc)

    record = {
        "target": target,
        "tier": tier,
        "model_family": "mace_embedding_head",
        "run_id": resolved_run_id,
        "run_dir": relpath_under(resolved_run_dir, root),
        "created_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "foundation_model": model_id
        or ((cfg.get("foundation") or {}).get("model_id")),
        "embedding_protocol": protocol,
        "label_transform": transform,
        "head_path": relpath_under(model_path, root),
        "mace_pack_dir": relpath_under(mace_pack_dir(cfg), root),
        "embedding_cache": relpath_under(resolved_cache, root),
        "limit_per_split": limit_per_split,
        "max_train_rows": max_train_rows,
        "counts": counts,
        "metrics": registry_metrics,
        "metrics_detailed": metrics_doc,
        # Placeholder for compare — filled below when enabled.
        "promoted_vs_lightgbm": None,
    }
    promoted: bool | None = None
    compare_report: Path | None = None
    # Register once so compare can find this run's predictions_test.parquet.
    write_json(resolved_run_dir / "run_record.json", record)
    if register:
        register_run(_models_root(cfg, root), record)

    if compare:
        try:
            cmp = compare_lightgbm_mace(
                target,
                mace_family="mace_embedding_head",
                out_dir=resolved_run_dir / "comparison",
            )
            compare_report = cmp.report_md
            promoted = cmp.promote_winner == "mace"
            record["promoted_vs_lightgbm"] = promoted
            record["promote_metric"] = cmp.promote_metric
            record["compare_n_overlap"] = cmp.n_overlap
            record["compare_lightgbm_mae"] = cmp.lightgbm_mae
            record["compare_mace_mae"] = cmp.mace_mae
            record["compare_lightgbm_spearman"] = cmp.lightgbm_spearman
            record["compare_mace_spearman"] = cmp.mace_spearman
            write_json(resolved_run_dir / "run_record.json", record)
            # Update leaderboard metrics in-place via a lightweight rewrite of the
            # last matching registry entry (avoid a duplicate append).
            if register:
                _patch_latest_registry_record(
                    _models_root(cfg, root),
                    target=target,
                    run_id=resolved_run_id,
                    patch={
                        "promoted_vs_lightgbm": promoted,
                        "promote_metric": cmp.promote_metric,
                        "compare_n_overlap": cmp.n_overlap,
                        "compare_lightgbm_mae": cmp.lightgbm_mae,
                        "compare_mace_mae": cmp.mace_mae,
                        "compare_lightgbm_spearman": cmp.lightgbm_spearman,
                        "compare_mace_spearman": cmp.mace_spearman,
                    },
                )
        except Exception as exc:  # noqa: BLE001 — compare is best-effort
            write_json(
                resolved_run_dir / "comparison_error.json",
                {"error": f"{type(exc).__name__}: {exc}"},
            )

    return EmbeddingHeadResult(
        target=target,
        run_id=resolved_run_id,
        run_dir=resolved_run_dir,
        protocol=protocol,
        metrics=metrics_doc,
        counts=counts,
        promoted=promoted,
        compare_report=compare_report,
    )
