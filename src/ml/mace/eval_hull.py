"""Evaluate named MACE phase-diagram E_hull protocol (Phase 4.7)."""

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
from ml.mace.calculator import load_foundation_calculator
from ml.mace.config import load_mace_config
from ml.mace.eform_cache import default_eform_cache_path, load_eform_cache
from ml.mace.hull import (
    PROTOCOL_MACE_PD_EFORM_V1,
    predict_e_above_hull,
    reference_entries_from_frame,
)
from ml.mace.pack import load_mace_pack_frame, mace_pack_dir
from ml.mace.reference import (
    PROTOCOL_ELEMENTAL_BULK_V1,
    formation_energy_per_atom,
    load_formation_references,
    reference_table_payload,
    write_reference_energies,
)
from ml.mace.structures import load_structure
from ml.metrics import correlation_metrics, regression_metrics
from ml.registry import register_run


@dataclass(frozen=True)
class HullEvalResult:
    """Artifacts from a hull-protocol evaluation."""

    target: str
    run_id: str
    run_dir: Path
    protocol: str
    device: str
    metrics: dict[str, Any]
    counts: dict[str, int]
    n_failed_loads: int
    n_failed_infer: int
    n_reference: int


def _run_id(protocol: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"energy_above_hull_{protocol}_{stamp}"


def _models_root(config: dict[str, Any], project_root: Path) -> Path:
    output = config.get("output") or {}
    return project_root / str(output.get("models_dir") or "data/models")


def _mace_run_dir(config: dict[str, Any], run_id: str, project_root: Path) -> Path:
    output = config.get("output") or {}
    models_dir = _models_root(config, project_root)
    sub = str(output.get("mace_subdir") or "mace")
    return models_dir / sub / "energy_above_hull" / run_id


def _infer_formation(
    frame: pd.DataFrame,
    *,
    calc: Any,
    e0s: dict[int, float],
    progress_every: int,
    split_name: str,
) -> tuple[pd.DataFrame, int, int]:
    rows: list[dict[str, Any]] = []
    n_fail_load = 0
    n_fail_infer = 0
    for i, rec in enumerate(frame.itertuples(index=False), start=1):
        structure_path = getattr(rec, "structure_path", None)
        loaded = load_structure(structure_path, attach_atoms=True)
        if not loaded.ok or loaded.atoms is None:
            n_fail_load += 1
            continue
        atoms = loaded.atoms
        try:
            atoms.calc = calc
            energy_tot = float(atoms.get_potential_energy())
            n_atoms = int(len(atoms))
            zs = [int(z) for z in atoms.get_atomic_numbers()]
            e_form = formation_energy_per_atom(energy_tot, zs, e0s)
        except Exception:  # noqa: BLE001
            n_fail_infer += 1
            continue
        formula = getattr(rec, "formula_pretty", None) or loaded.formula
        rows.append(
            {
                "material_id": str(getattr(rec, "material_id", "")),
                "split": str(getattr(rec, "split", split_name)),
                "structure_path": loaded.path,
                "formula": formula,
                "formula_pretty": formula,
                "chemsys": getattr(rec, "chemsys", None),
                "y_true": float(rec.label) if hasattr(rec, "label") else None,
                "energy_total_eV": energy_tot,
                "y_pred_formation": e_form,
                "e_form_per_atom": e_form,
                "n_atoms": n_atoms,
            }
        )
        if progress_every and i % int(progress_every) == 0:
            print(f"[{split_name}] MACE E_form {i}/{len(frame)} …", flush=True)
    return pd.DataFrame(rows), n_fail_load, n_fail_infer


def _reference_from_eform_cache(
    *,
    reference_split: str,
    limit_reference: int | None,
    energy_protocol: str,
    model_id: str,
    config: dict[str, Any],
    eform_cache_path: Path | None,
) -> pd.DataFrame | None:
    """Return cached E_form rows for the hull reference pool, or None."""
    path = eform_cache_path or default_eform_cache_path(
        "formation_energy_per_atom",
        protocol=energy_protocol,
        model_id=model_id,
        config=config,
    )
    if not path.is_file():
        return None
    try:
        cache = load_eform_cache(
            "formation_energy_per_atom",
            protocol=energy_protocol,
            cache_path=path,
            config=config,
        )
    except FileNotFoundError:
        return None
    if "split" in cache.columns:
        cache = cache.loc[cache["split"].astype(str) == reference_split].copy()
    if limit_reference is not None:
        cache = cache.head(int(limit_reference))
    if cache.empty:
        return None
    # Align columns with _infer_formation output.
    if "y_pred_formation" not in cache.columns and "e_form_per_atom" in cache.columns:
        cache = cache.copy()
        cache["y_pred_formation"] = cache["e_form_per_atom"]
    if "formula" not in cache.columns and "formula_pretty" in cache.columns:
        cache = cache.copy()
        cache["formula"] = cache["formula_pretty"]
    return cache


def eval_hull(
    *,
    protocol: str | None = None,
    reference_split: str = "train",
    eval_splits: tuple[str, ...] = ("val", "test"),
    limit_reference: int | None = None,
    limit_per_split: int | None = None,
    device: str | None = "cpu",
    model_id: str | None = None,
    checkpoint: Path | str | None = None,
    run_id: str | None = None,
    run_dir: Path | None = None,
    config: dict[str, Any] | None = None,
    project_root: Path | None = None,
    write_predictions: bool = True,
    register: bool = True,
    progress_every: int = 200,
    use_eform_cache: bool = True,
    eform_cache_path: Path | None = None,
) -> HullEvalResult:
    """Build a MACE phase diagram from the reference pool; score eval splits."""
    cfg = config if config is not None else load_mace_config()
    root = project_root or PROJECT_ROOT
    hull_cfg = (cfg.get("targets") or {}).get("energy_above_hull") or {}
    protocol = str(protocol or hull_cfg.get("hull_protocol") or PROTOCOL_MACE_PD_EFORM_V1)
    if protocol != PROTOCOL_MACE_PD_EFORM_V1:
        raise ValueError(f"Unsupported hull_protocol {protocol!r}")

    pack = load_mace_pack_frame("energy_above_hull", train_ok_only=True, config=cfg)
    calc, resolved_model, ckpt_path, resolved_device = load_foundation_calculator(
        model_id=model_id,
        device=device,
        config=cfg,
        checkpoint=checkpoint,
    )
    e0s, energy_protocol = load_formation_references(
        resolved_model, calc=calc, config=cfg
    )

    resolved_run_id = run_id or _run_id(protocol)
    resolved_run_dir = run_dir or _mace_run_dir(cfg, resolved_run_id, root)
    resolved_run_dir.mkdir(parents=True, exist_ok=True)

    ref_payload = reference_table_payload(
        e0s, protocol=energy_protocol, model_id=resolved_model
    )
    write_reference_energies(resolved_run_dir / "reference_energies.json", ref_payload)
    (resolved_run_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=True),
        encoding="utf-8",
    )

    n_fail_load = 0
    n_fail_infer = 0
    ref_source = "live_infer"
    ref_pred = None
    if use_eform_cache:
        ref_pred = _reference_from_eform_cache(
            reference_split=reference_split,
            limit_reference=limit_reference,
            energy_protocol=energy_protocol or PROTOCOL_ELEMENTAL_BULK_V1,
            model_id=resolved_model,
            config=cfg,
            eform_cache_path=eform_cache_path,
        )
        if ref_pred is not None and len(ref_pred) >= 2:
            ref_source = "eform_cache"
            print(
                f"Using E_form cache reference pool ({reference_split}): "
                f"{len(ref_pred)} rows …",
                flush=True,
            )

    if ref_pred is None or len(ref_pred) < 2:
        ref_part = pack.loc[pack["split"].astype(str) == reference_split].copy()
        if limit_reference is not None:
            ref_part = ref_part.head(int(limit_reference))
        print(
            f"Inferring reference pool ({reference_split}): {len(ref_part)} structures …",
            flush=True,
        )
        ref_pred, n_fail_load, n_fail_infer = _infer_formation(
            ref_part,
            calc=calc,
            e0s=e0s,
            progress_every=progress_every,
            split_name=reference_split,
        )
        ref_source = "live_infer"

    if write_predictions:
        ref_pred.to_parquet(
            resolved_run_dir / f"predictions_reference_{reference_split}.parquet",
            index=False,
        )
    reference_entries = reference_entries_from_frame(ref_pred)
    if len(reference_entries) < 2:
        raise RuntimeError(
            f"Need ≥2 reference formation energies for hull protocol "
            f"(got {len(reference_entries)}). Increase --limit-reference "
            "or run scripts/cache_mace_eform.py."
        )
    write_json(
        resolved_run_dir / "hull_reference.json",
        {
            "protocol": protocol,
            "energy_protocol": energy_protocol,
            "reference_split": reference_split,
            "n_reference": len(reference_entries),
            "reference_source": ref_source,
            "note": (
                "PhaseDiagram built per query from reference entries whose "
                "elements ⊆ query elements, plus elemental E=0 refs. "
                "Prefer resumable E_form cache when present."
            ),
        },
    )

    split_metrics: dict[str, Any] = {}
    counts: dict[str, int] = {f"reference_{reference_split}": int(len(ref_pred))}
    n_hull_fail = 0

    for split in eval_splits:
        part = pack.loc[pack["split"].astype(str) == split].copy()
        if limit_per_split is not None:
            part = part.head(int(limit_per_split))
        print(f"Inferring eval split {split}: {len(part)} structures …", flush=True)
        pred, fl, fi = _infer_formation(
            part,
            calc=calc,
            e0s=e0s,
            progress_every=progress_every,
            split_name=split,
        )
        n_fail_load += fl
        n_fail_infer += fi

        hull_rows: list[dict[str, Any]] = []
        for rec in pred.itertuples(index=False):
            try:
                e_hull = predict_e_above_hull(
                    str(rec.formula),
                    float(rec.e_form_per_atom),
                    reference_entries,
                    protocol=protocol,
                )
            except Exception:  # noqa: BLE001
                n_hull_fail += 1
                continue
            hull_rows.append(
                {
                    "material_id": rec.material_id,
                    "split": split,
                    "structure_path": rec.structure_path,
                    "formula": rec.formula,
                    "chemsys": rec.chemsys,
                    "y_true": rec.y_true,
                    "y_pred_formation": rec.y_pred_formation,
                    "y_pred": e_hull,
                    "n_atoms": rec.n_atoms,
                }
            )
        out = pd.DataFrame(hull_rows)
        counts[split] = int(len(out))
        if write_predictions:
            out.to_parquet(
                resolved_run_dir / f"predictions_{split}.parquet", index=False
            )
        if len(out) < 2:
            split_metrics[split] = {
                "error": "insufficient_predictions",
                "n": int(len(out)),
            }
            continue
        y_true = out["y_true"].to_numpy(dtype=np.float64)
        y_pred = out["y_pred"].to_numpy(dtype=np.float64)
        split_metrics[split] = {
            **regression_metrics(y_true, y_pred),
            **correlation_metrics(y_true, y_pred),
        }

    registry_metrics: dict[str, Any] = {}
    for split, block in split_metrics.items():
        if "mae" in (block or {}):
            registry_metrics[split] = {
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
        "energy_protocol": energy_protocol,
        "model_id": resolved_model,
        "model_family": "mace_pretrained",
        "checkpoint": relpath_under(ckpt_path, root),
        "device": resolved_device,
        "reference_split": reference_split,
        "n_reference": len(reference_entries),
        "n_hull_failures": n_hull_fail,
        "splits": split_metrics,
        "primary": "energy_above_hull",
        "registry_metrics": registry_metrics,
        "label_unit": "eV/atom",
        "note": (
            "Not a naive structure→label regressor; E_hull from MACE E_form "
            f"via {protocol}."
        ),
    }
    write_json(resolved_run_dir / "metrics.json", metrics_doc)

    record = {
        "target": "energy_above_hull",
        "tier": str(hull_cfg.get("tier") or "Ready"),
        "model_family": "mace_pretrained",
        "run_id": resolved_run_id,
        "run_dir": relpath_under(resolved_run_dir, root),
        "created_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "foundation_model": resolved_model,
        "checkpoint": relpath_under(ckpt_path, root),
        "hull_protocol": protocol,
        "reference_protocol": energy_protocol,
        "device": resolved_device,
        "mace_pack_dir": relpath_under(mace_pack_dir(cfg), root),
        "limit_reference": limit_reference,
        "limit_per_split": limit_per_split,
        "reference_split": reference_split,
        "splits_evaluated": list(eval_splits),
        "counts": counts,
        "n_failed_loads": n_fail_load,
        "n_failed_infer": n_fail_infer,
        "n_hull_failures": n_hull_fail,
        "metrics": registry_metrics,
        "metrics_detailed": metrics_doc,
    }
    write_json(resolved_run_dir / "run_record.json", record)
    if register:
        register_run(_models_root(cfg, root), record)

    return HullEvalResult(
        target="energy_above_hull",
        run_id=resolved_run_id,
        run_dir=resolved_run_dir,
        protocol=protocol,
        device=resolved_device,
        metrics=metrics_doc,
        counts=counts,
        n_failed_loads=n_fail_load,
        n_failed_infer=n_fail_infer,
        n_reference=len(reference_entries),
    )
