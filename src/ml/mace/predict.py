"""Predict formation energies with a registered MACE run (Phase 4.6)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import PROJECT_ROOT
from ml.io_util import read_json
from ml.mace.calculator import load_foundation_calculator
from ml.mace.config import foundation_model_id, load_mace_config
from ml.mace.pack import load_mace_pack_frame, mace_pack_dir
from ml.mace.reference import (
    formation_energy_per_atom,
    load_formation_references,
)
from ml.mace.structures import load_structure
from ml.registry import load_registry


@dataclass(frozen=True)
class LoadedMaceRun:
    """A registered MACE checkpoint ready for inference."""

    target: str
    model_family: str
    run_id: str
    run_dir: Path
    checkpoint: Path
    foundation_model: str
    metrics: dict[str, Any]
    record: dict[str, Any]


def _models_root(config: dict[str, Any], project_root: Path) -> Path:
    output = config.get("output") or {}
    return project_root / str(output.get("models_dir") or "data/models")


def latest_mace_run(
    target: str = "formation_energy_per_atom",
    *,
    model_family: str | None = "mace_pretrained",
    config: dict[str, Any] | None = None,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Return the latest registry record for a MACE family.

    If ``model_family`` is None, prefer frozen ``mace_pretrained`` then fine-tune.
    """
    cfg = config if config is not None else load_mace_config()
    root = project_root or PROJECT_ROOT
    registry = load_registry(_models_root(cfg, root))
    runs = list((registry.get("runs") or {}).get(target) or [])
    if not runs:
        raise FileNotFoundError(f"No registered runs for target {target!r}")

    preferred = (
        [model_family]
        if model_family is not None
        else ["mace_pretrained", "mace_finetuned"]
    )
    for family in preferred:
        for record in reversed(runs):
            if str(record.get("model_family") or "") == family:
                return record
    raise FileNotFoundError(
        f"No MACE runs for target={target!r} families={preferred} under "
        f"{_models_root(cfg, root)}"
    )


def load_mace_run(
    target: str = "formation_energy_per_atom",
    *,
    model_family: str | None = "mace_pretrained",
    run_dir: Path | None = None,
    config: dict[str, Any] | None = None,
    project_root: Path | None = None,
) -> LoadedMaceRun:
    """Load checkpoint path + metadata for a MACE run."""
    cfg = config if config is not None else load_mace_config()
    root = project_root or PROJECT_ROOT

    if run_dir is not None:
        directory = Path(run_dir)
        if not directory.is_absolute():
            directory = root / directory
        record = read_json(directory / "run_record.json") or {}
        family = str(record.get("model_family") or model_family or "mace_pretrained")
        run_id = str(record.get("run_id") or directory.name)
    else:
        record = latest_mace_run(
            target, model_family=model_family, config=cfg, project_root=root
        )
        family = str(record.get("model_family") or "mace_pretrained")
        run_id = str(record["run_id"])
        directory = Path(record["run_dir"])
        if not directory.is_absolute():
            directory = root / directory

    ckpt = directory / "checkpoint" / "model.model"
    if not ckpt.is_file():
        # Pretrained runs may only reference the foundation cache path.
        alt = record.get("checkpoint")
        if alt:
            ckpt = Path(alt)
            if not ckpt.is_absolute():
                ckpt = root / ckpt
    if not ckpt.is_file():
        # Resolve foundation cache for pretrained / embedding heads.
        if family in {"mace_pretrained", "mace_embedding_head"}:
            from ml.mace.checkpoints import ensure_foundation_checkpoint

            try:
                ckpt = ensure_foundation_checkpoint(
                    str(record.get("foundation_model") or foundation_model_id(cfg)),
                    config=cfg,
                )
            except Exception as exc:
                # Embedding heads only need head.joblib for the readout path;
                # pretrained energy runs must surface checkpoint failures.
                head = directory / "head.joblib"
                if family == "mace_embedding_head" and head.is_file():
                    ckpt = head
                else:
                    raise FileNotFoundError(
                        f"Missing MACE checkpoint under {directory}"
                    ) from exc
        else:
            raise FileNotFoundError(f"Missing MACE checkpoint under {directory}")

    metrics = read_json(directory / "metrics.json") or {}
    return LoadedMaceRun(
        target=target,
        model_family=family,
        run_id=run_id,
        run_dir=directory,
        checkpoint=ckpt,
        foundation_model=str(
            record.get("foundation_model") or foundation_model_id(cfg)
        ),
        metrics=metrics if isinstance(metrics, dict) else {},
        record=record,
    )


def _resolve_structure_inputs(
    *,
    structure_paths: list[str] | None,
    material_ids: list[str] | None,
    target: str,
    config: dict[str, Any],
) -> list[dict[str, str | None]]:
    """Normalize CLI inputs to ``{material_id, structure_path}`` rows."""
    rows: list[dict[str, str | None]] = []
    for path in structure_paths or []:
        rows.append({"material_id": None, "structure_path": path})

    if material_ids:
        pack = load_mace_pack_frame(target, train_ok_only=False, config=config)
        by_id = pack.set_index("material_id", drop=False)
        for mid in material_ids:
            if mid not in by_id.index:
                raise KeyError(
                    f"material_id {mid!r} not in MACE pack "
                    f"({mace_pack_dir(config) / 'targets' / (target + '.parquet')})"
                )
            rec = by_id.loc[mid]
            if isinstance(rec, pd.DataFrame):
                rec = rec.iloc[0]
            rows.append(
                {
                    "material_id": str(mid),
                    "structure_path": str(rec["structure_path"]),
                }
            )
    if not rows:
        raise ValueError("Provide at least one --structure or --material-id")
    return rows


def predict_structures(
    *,
    structure_paths: list[str] | None = None,
    material_ids: list[str] | None = None,
    target: str = "formation_energy_per_atom",
    model_family: str | None = "mace_pretrained",
    run_dir: Path | None = None,
    device: str | None = "cpu",
    config: dict[str, Any] | None = None,
    project_root: Path | None = None,
) -> pd.DataFrame:
    """Predict total / per-atom / formation energies for structures."""
    cfg = config if config is not None else load_mace_config()
    root = project_root or PROJECT_ROOT
    loaded = load_mace_run(
        target,
        model_family=model_family,
        run_dir=run_dir,
        config=cfg,
        project_root=root,
    )
    inputs = _resolve_structure_inputs(
        structure_paths=structure_paths,
        material_ids=material_ids,
        target=target,
        config=cfg,
    )

    calc, _, _, resolved_device = load_foundation_calculator(
        model_id=loaded.foundation_model,
        device=device,
        config=cfg,
        checkpoint=loaded.checkpoint,
    )
    e0s, protocol = load_formation_references(
        loaded.foundation_model, calc=calc, config=cfg
    )

    rows: list[dict[str, Any]] = []
    for item in inputs:
        structure_path = item["structure_path"]
        result = load_structure(structure_path, attach_atoms=True)
        if not result.ok or result.atoms is None:
            rows.append(
                {
                    "material_id": item["material_id"],
                    "structure_path": structure_path,
                    "ok": False,
                    "reason": result.reason,
                    "detail": result.detail,
                    "energy_total_eV": None,
                    "energy_per_atom_eV": None,
                    "formation_energy_per_atom": None,
                    "n_atoms": result.n_atoms,
                    "formula": result.formula,
                }
            )
            continue
        atoms = result.atoms
        atoms.calc = calc
        energy_tot = float(atoms.get_potential_energy())
        n_atoms = int(len(atoms))
        zs = [int(z) for z in atoms.get_atomic_numbers()]
        e_pa = energy_tot / float(n_atoms)
        e_form = formation_energy_per_atom(energy_tot, zs, e0s)
        rows.append(
            {
                "material_id": item["material_id"],
                "structure_path": result.path,
                "ok": True,
                "reason": "OK",
                "detail": None,
                "energy_total_eV": energy_tot,
                "energy_per_atom_eV": e_pa,
                "formation_energy_per_atom": e_form,
                "n_atoms": n_atoms,
                "formula": result.formula,
                "reference_protocol": protocol,
                "model_family": loaded.model_family,
                "run_id": loaded.run_id,
                "device": resolved_device,
            }
        )
    return pd.DataFrame(rows)
