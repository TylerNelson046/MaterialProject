"""Zero-shot pretrained MACE evaluation on the structure pack (Phase 4.4)."""

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
from ml.mace.pack import load_mace_pack_frame, mace_pack_dir
from ml.mace.reference import (
    formation_energy_per_atom,
    load_formation_references,
    reference_table_payload,
    write_reference_energies,
)
from ml.mace.structures import load_structure
from ml.metrics import correlation_metrics, regression_metrics
from ml.registry import register_run


@dataclass(frozen=True)
class PretrainedEvalResult:
    """Artifacts from a zero-shot MACE evaluation run."""

    target: str
    run_id: str
    run_dir: Path
    model_id: str
    device: str
    metrics: dict[str, Any]
    counts: dict[str, int]
    n_failed_loads: int
    n_failed_infer: int


def _run_id(target: str, model_family: str = "mace_pretrained") -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{target}_{model_family}_{stamp}"


def _models_root(config: dict[str, Any], project_root: Path) -> Path:
    output = config.get("output") or {}
    models_dir = project_root / str(output.get("models_dir") or "data/models")
    return models_dir


def _mace_run_dir(config: dict[str, Any], target: str, run_id: str, project_root: Path) -> Path:
    output = config.get("output") or {}
    models_dir = _models_root(config, project_root)
    sub = str(output.get("mace_subdir") or "mace")
    return models_dir / sub / target / run_id


def _score_split(
    y_true: np.ndarray,
    energy_per_atom: np.ndarray,
    e_form: np.ndarray,
) -> dict[str, Any]:
    """Metrics for raw E/atom correlation and E_form vs label."""
    out: dict[str, Any] = {
        "energy_per_atom_vs_label": correlation_metrics(y_true, energy_per_atom),
        "formation_energy": regression_metrics(y_true, e_form),
        "formation_energy_corr": correlation_metrics(y_true, e_form),
    }
    return out


def eval_pretrained(
    target: str = "formation_energy_per_atom",
    *,
    splits: tuple[str, ...] = ("val", "test"),
    limit_per_split: int | None = None,
    device: str | None = "cpu",
    model_id: str | None = None,
    checkpoint: Path | str | None = None,
    model_family: str = "mace_pretrained",
    run_id: str | None = None,
    run_dir: Path | None = None,
    config: dict[str, Any] | None = None,
    project_root: Path | None = None,
    write_predictions: bool = True,
    register: bool = True,
    progress_every: int = 200,
) -> PretrainedEvalResult:
    """Evaluate a MACE checkpoint on the structure pack and register metrics.

    Defaults to the pinned foundation model (Phase 4.4). Pass ``checkpoint``
    for a fine-tuned ``.model`` (Phase 4.5).

    Reports:
    * Spearman/Pearson of ``E_tot/n`` vs export ``label``
    * MAE/RMSE/R² of formation energy vs ``label`` under the configured
      reference protocol (default ``elemental_bulk_v1``)
    """
    cfg = config if config is not None else load_mace_config()
    root = project_root or PROJECT_ROOT

    pack = load_mace_pack_frame(target, train_ok_only=True, config=cfg)
    calc, resolved_model, ckpt_path, resolved_device = load_foundation_calculator(
        model_id=model_id,
        device=device,
        config=cfg,
        checkpoint=checkpoint,
    )
    e0s, protocol = load_formation_references(
        resolved_model, calc=calc, config=cfg
    )

    resolved_run_id = run_id or _run_id(target, model_family)
    resolved_run_dir = run_dir or _mace_run_dir(cfg, target, resolved_run_id, root)
    resolved_run_dir.mkdir(parents=True, exist_ok=True)

    ref_payload = reference_table_payload(
        e0s, protocol=protocol, model_id=resolved_model
    )
    write_reference_energies(resolved_run_dir / "reference_energies.json", ref_payload)
    (resolved_run_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=True),
        encoding="utf-8",
    )

    split_metrics: dict[str, Any] = {}
    counts: dict[str, int] = {}
    n_failed_loads = 0
    n_failed_infer = 0

    for split in splits:
        part = pack.loc[pack["split"].astype(str) == split].copy()
        if limit_per_split is not None:
            part = part.head(int(limit_per_split))
        rows: list[dict[str, Any]] = []
        for i, rec in enumerate(part.itertuples(index=False), start=1):
            structure_path = getattr(rec, "structure_path", None)
            loaded = load_structure(structure_path, attach_atoms=True)
            if not loaded.ok or loaded.atoms is None:
                n_failed_loads += 1
                continue
            atoms = loaded.atoms
            try:
                atoms.calc = calc
                energy_tot = float(atoms.get_potential_energy())
                n_atoms = int(len(atoms))
                zs = [int(z) for z in atoms.get_atomic_numbers()]
                e_pa = energy_tot / float(n_atoms)
                e_form = formation_energy_per_atom(energy_tot, zs, e0s)
            except Exception:  # noqa: BLE001 — quarantine inference failures
                n_failed_infer += 1
                continue
            rows.append(
                {
                    "material_id": str(getattr(rec, "material_id", "")),
                    "split": split,
                    "structure_path": loaded.path,
                    "y_true": float(rec.label),
                    "energy_total_eV": energy_tot,
                    "energy_per_atom_eV": e_pa,
                    "y_pred_formation": e_form,
                    "n_atoms": n_atoms,
                }
            )
            if progress_every and i % int(progress_every) == 0:
                print(f"[{split}] inferred {i}/{len(part)} …", flush=True)

        pred = pd.DataFrame(rows)
        counts[split] = int(len(pred))
        if write_predictions:
            pred.to_parquet(
                resolved_run_dir / f"predictions_{split}.parquet", index=False
            )
        if len(pred) == 0:
            split_metrics[split] = {"error": "no_successful_predictions"}
            continue
        y_true = pred["y_true"].to_numpy(dtype=np.float64)
        e_pa = pred["energy_per_atom_eV"].to_numpy(dtype=np.float64)
        e_form = pred["y_pred_formation"].to_numpy(dtype=np.float64)
        split_metrics[split] = _score_split(y_true, e_pa, e_form)

    # Flatten primary metrics for registry (E_form MAE + ranking correlations).
    registry_metrics: dict[str, Any] = {}
    for split, block in split_metrics.items():
        form = (block or {}).get("formation_energy") or {}
        corr = (block or {}).get("formation_energy_corr") or {}
        if form:
            registry_metrics[split] = {
                **form,
                **{
                    k: corr[k]
                    for k in (
                        "pearson_r",
                        "pearson_p",
                        "spearman_r",
                        "spearman_p",
                    )
                    if k in corr
                },
            }

    metrics_doc = {
        "protocol": protocol,
        "model_id": resolved_model,
        "model_family": model_family,
        "checkpoint": relpath_under(ckpt_path, root),
        "device": resolved_device,
        "splits": split_metrics,
        "primary": "formation_energy",  # vs export label
        "registry_metrics": registry_metrics,
    }
    write_json(resolved_run_dir / "metrics.json", metrics_doc)

    targets_cfg = (cfg.get("targets") or {}).get(target) or {}
    tier = str(targets_cfg.get("tier") or "Ready")
    record = {
        "target": target,
        "tier": tier,
        "model_family": model_family,
        "run_id": resolved_run_id,
        "run_dir": relpath_under(resolved_run_dir, root),
        "created_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "foundation_model": resolved_model,
        "checkpoint": relpath_under(ckpt_path, root),
        "device": resolved_device,
        "reference_protocol": protocol,
        "mace_pack_dir": relpath_under(mace_pack_dir(cfg), root),
        "limit_per_split": limit_per_split,
        "splits_evaluated": list(splits),
        "counts": counts,
        "n_failed_loads": n_failed_loads,
        "n_failed_infer": n_failed_infer,
        "metrics": registry_metrics,
        "metrics_detailed": metrics_doc,
    }
    write_json(resolved_run_dir / "run_record.json", record)

    if register:
        models_dir = _models_root(cfg, root)
        register_run(models_dir, record)

    return PretrainedEvalResult(
        target=target,
        run_id=resolved_run_id,
        run_dir=resolved_run_dir,
        model_id=resolved_model,
        device=resolved_device,
        metrics=metrics_doc,
        counts=counts,
        n_failed_loads=n_failed_loads,
        n_failed_infer=n_failed_infer,
    )
