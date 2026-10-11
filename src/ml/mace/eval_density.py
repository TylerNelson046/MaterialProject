"""Evaluate structure density protocols vs Ready labels (Phase 4.7)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from config.settings import PROJECT_ROOT
from ml.io_util import relpath_under, write_json
from ml.mace.config import load_mace_config
from ml.mace.density import (
    PROTOCOL_GEOMETRY_RELAX_V1,
    PROTOCOL_GEOMETRY_V1,
    predict_density,
)
from ml.mace.pack import load_mace_pack_frame, mace_pack_dir
from ml.mace.structures import load_structure
from ml.metrics import correlation_metrics, regression_metrics
from ml.registry import register_run


@dataclass(frozen=True)
class DensityEvalResult:
    """Artifacts from a density protocol evaluation."""

    target: str
    run_id: str
    run_dir: Path
    protocol: str
    device: str | None
    metrics: dict[str, Any]
    counts: dict[str, int]
    n_failed_loads: int
    n_failed_infer: int


def _run_id(protocol: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"density_{protocol}_{stamp}"


def _models_root(config: dict[str, Any], project_root: Path) -> Path:
    output = config.get("output") or {}
    return project_root / str(output.get("models_dir") or "data/models")


def _mace_run_dir(config: dict[str, Any], run_id: str, project_root: Path) -> Path:
    output = config.get("output") or {}
    models_dir = _models_root(config, project_root)
    sub = str(output.get("mace_subdir") or "mace")
    return models_dir / sub / "density" / run_id


def _score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    return {
        **regression_metrics(y_true, y_pred),
        **correlation_metrics(y_true, y_pred),
    }


def _load_cod_holdout_frame(
    *,
    config: dict[str, Any],
    project_root: Path,
    limit: int | None,
) -> pd.DataFrame:
    export = config.get("export") or {}
    root = project_root / str(export.get("dir") or "data/processed/training_exports")
    sub = str(export.get("targets_subdir") or "targets")
    path = root / sub / "density.parquet"
    cols = [
        "material_id",
        "structure_path",
        "label",
        "split",
        "source",
        "holdout_cod_geometry",
        "formula_pretty",
    ]
    frame = pd.read_parquet(path)
    keep = [c for c in cols if c in frame.columns]
    frame = frame.loc[:, keep]
    if "holdout_cod_geometry" not in frame.columns:
        raise ValueError("density export missing holdout_cod_geometry")
    frame = frame.loc[frame["holdout_cod_geometry"].astype(bool)].copy()
    frame = frame.loc[frame["structure_path"].notna()].copy()
    if limit is not None:
        frame = frame.head(int(limit))
    return frame


def eval_density(
    *,
    protocol: str | None = None,
    splits: tuple[str, ...] = ("val", "test"),
    limit_per_split: int | None = None,
    also_cod_holdout: bool = True,
    cod_limit: int | None = None,
    device: str | None = "cpu",
    model_id: str | None = None,
    relax_fmax: float | None = None,
    relax_steps: int | None = None,
    run_id: str | None = None,
    run_dir: Path | None = None,
    config: dict[str, Any] | None = None,
    project_root: Path | None = None,
    write_predictions: bool = True,
    register: bool = True,
    progress_every: int = 500,
) -> DensityEvalResult:
    """Evaluate density protocol on the structure pack (+ optional COD holdout)."""
    cfg = config if config is not None else load_mace_config()
    root = project_root or PROJECT_ROOT
    dens_cfg = (cfg.get("targets") or {}).get("density") or {}
    protocol = str(protocol or dens_cfg.get("protocol") or PROTOCOL_GEOMETRY_V1)
    relax_cfg = dens_cfg.get("relax") or {}
    fmax = float(relax_fmax if relax_fmax is not None else relax_cfg.get("fmax", 0.5))
    steps = int(relax_steps if relax_steps is not None else relax_cfg.get("steps", 25))

    calc = None
    resolved_model = None
    ckpt_path = None
    resolved_device: str | None = None
    if protocol == PROTOCOL_GEOMETRY_RELAX_V1:
        from ml.mace.calculator import load_foundation_calculator

        calc, resolved_model, ckpt_path, resolved_device = load_foundation_calculator(
            model_id=model_id,
            device=device,
            config=cfg,
        )

    pack = load_mace_pack_frame("density", train_ok_only=True, config=cfg)
    resolved_run_id = run_id or _run_id(protocol)
    resolved_run_dir = run_dir or _mace_run_dir(cfg, resolved_run_id, root)
    resolved_run_dir.mkdir(parents=True, exist_ok=True)
    (resolved_run_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=True),
        encoding="utf-8",
    )

    split_metrics: dict[str, Any] = {}
    counts: dict[str, int] = {}
    n_failed_loads = 0
    n_failed_infer = 0

    def _infer_rows(frame: pd.DataFrame, split_name: str) -> pd.DataFrame:
        nonlocal n_failed_loads, n_failed_infer
        rows: list[dict[str, Any]] = []
        for i, rec in enumerate(frame.itertuples(index=False), start=1):
            structure_path = getattr(rec, "structure_path", None)
            loaded = load_structure(structure_path, attach_atoms=True)
            if not loaded.ok or loaded.atoms is None:
                n_failed_loads += 1
                continue
            try:
                y_pred, _ = predict_density(
                    loaded.atoms,
                    protocol=protocol,
                    calc=calc,
                    relax_fmax=fmax,
                    relax_steps=steps,
                )
            except Exception:  # noqa: BLE001
                n_failed_infer += 1
                continue
            rows.append(
                {
                    "material_id": str(getattr(rec, "material_id", "")),
                    "split": split_name,
                    "structure_path": loaded.path,
                    "y_true": float(rec.label),
                    "y_pred": float(y_pred),
                    "n_atoms": loaded.n_atoms,
                    "formula": loaded.formula,
                    "holdout_cod_geometry": bool(
                        getattr(rec, "holdout_cod_geometry", False)
                    ),
                }
            )
            if progress_every and i % int(progress_every) == 0:
                print(f"[{split_name}] density {i}/{len(frame)} …", flush=True)
        return pd.DataFrame(rows)

    for split in splits:
        part = pack.loc[pack["split"].astype(str) == split].copy()
        if limit_per_split is not None:
            part = part.head(int(limit_per_split))
        pred = _infer_rows(part, split)
        counts[split] = int(len(pred))
        if write_predictions:
            pred.to_parquet(
                resolved_run_dir / f"predictions_{split}.parquet", index=False
            )
        if len(pred) < 2:
            split_metrics[split] = {"error": "insufficient_predictions", "n": int(len(pred))}
            continue
        split_metrics[split] = _score(
            pred["y_true"].to_numpy(dtype=np.float64),
            pred["y_pred"].to_numpy(dtype=np.float64),
        )

    # Pack-internal COD holdout slice (DFT∩COD overlap is usually empty).
    if "holdout_cod_geometry" in pack.columns:
        cod_pack = pack.loc[pack["holdout_cod_geometry"].astype(bool)].copy()
        if limit_per_split is not None:
            cod_pack = cod_pack.head(int(limit_per_split))
        if len(cod_pack):
            pred = _infer_rows(cod_pack, "holdout_cod_geometry_pack")
            counts["holdout_cod_geometry_pack"] = int(len(pred))
            if write_predictions:
                pred.to_parquet(
                    resolved_run_dir / "predictions_holdout_cod_geometry_pack.parquet",
                    index=False,
                )
            if len(pred) >= 2:
                split_metrics["holdout_cod_geometry_pack"] = _score(
                    pred["y_true"].to_numpy(dtype=np.float64),
                    pred["y_pred"].to_numpy(dtype=np.float64),
                )

    if also_cod_holdout:
        try:
            cod = _load_cod_holdout_frame(
                config=cfg, project_root=root, limit=cod_limit
            )
            pred = _infer_rows(cod, "holdout_cod_geometry")
            counts["holdout_cod_geometry"] = int(len(pred))
            if write_predictions:
                pred.to_parquet(
                    resolved_run_dir / "predictions_holdout_cod_geometry.parquet",
                    index=False,
                )
            if len(pred) >= 2:
                split_metrics["holdout_cod_geometry"] = _score(
                    pred["y_true"].to_numpy(dtype=np.float64),
                    pred["y_pred"].to_numpy(dtype=np.float64),
                )
            elif len(pred) == 0:
                split_metrics["holdout_cod_geometry"] = {
                    "error": "no_successful_predictions"
                }
        except Exception as exc:  # noqa: BLE001
            split_metrics["holdout_cod_geometry"] = {
                "error": f"{type(exc).__name__}: {exc}"
            }

    registry_metrics: dict[str, Any] = {}
    for key in ("test", "val", "holdout_cod_geometry"):
        block = split_metrics.get(key) or {}
        if "mae" in block:
            dest = key if key != "holdout_cod_geometry" else "holdout_cod"
            registry_metrics[dest] = {
                "mae": block["mae"],
                "rmse": block["rmse"],
                "r2": block["r2"],
                **{
                    k: block[k]
                    for k in (
                        "pearson_r",
                        "pearson_p",
                        "spearman_r",
                        "spearman_p",
                    )
                    if k in block
                },
            }

    metrics_doc = {
        "protocol": protocol,
        "model_id": resolved_model,
        "model_family": "mace_pretrained",
        "checkpoint": relpath_under(ckpt_path, root) if ckpt_path else None,
        "device": resolved_device,
        "relax": {"fmax": fmax, "steps": steps}
        if protocol == PROTOCOL_GEOMETRY_RELAX_V1
        else None,
        "splits": split_metrics,
        "primary": "density",
        "registry_metrics": registry_metrics,
        "label_unit": "g/cm^3",
    }
    write_json(resolved_run_dir / "metrics.json", metrics_doc)

    record = {
        "target": "density",
        "tier": str(dens_cfg.get("tier") or "Ready"),
        "model_family": "mace_pretrained",
        "run_id": resolved_run_id,
        "run_dir": relpath_under(resolved_run_dir, root),
        "created_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "foundation_model": resolved_model,
        "density_protocol": protocol,
        "device": resolved_device,
        "mace_pack_dir": relpath_under(mace_pack_dir(cfg), root),
        "limit_per_split": limit_per_split,
        "cod_limit": cod_limit,
        "splits_evaluated": list(splits),
        "counts": counts,
        "n_failed_loads": n_failed_loads,
        "n_failed_infer": n_failed_infer,
        "metrics": registry_metrics,
        "metrics_detailed": metrics_doc,
    }
    write_json(resolved_run_dir / "run_record.json", record)
    if register:
        register_run(_models_root(cfg, root), record)

    return DensityEvalResult(
        target="density",
        run_id=resolved_run_id,
        run_dir=resolved_run_dir,
        protocol=protocol,
        device=resolved_device,
        metrics=metrics_doc,
        counts=counts,
        n_failed_loads=n_failed_loads,
        n_failed_infer=n_failed_infer,
    )
