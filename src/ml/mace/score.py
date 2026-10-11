"""Candidate scoring: relax → E_form → optional E_hull (frozen MACE).

Primary Ready-loop use case: rank generated / screened structures by
formation energy and energy above hull without fine-tuning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ml.mace.calculator import load_foundation_calculator
from ml.mace.config import foundation_model_id, load_mace_config
from ml.mace.eform_cache import load_eform_cache
from ml.mace.hull import (
    PROTOCOL_MACE_PD_EFORM_V1,
    HullEntry,
    predict_e_above_hull,
    reference_entries_from_frame,
)
from ml.mace.predict import _resolve_structure_inputs
from ml.mace.reference import (
    PROTOCOL_ELEMENTAL_BULK_V1,
    formation_energy_per_atom,
    load_formation_references,
)
from ml.mace.relax import relax_atoms_mace
from ml.mace.structures import load_structure


def load_hull_reference_entries(
    *,
    cache_path: Path | None = None,
    reference_parquet: Path | None = None,
    config: dict[str, Any] | None = None,
) -> list[HullEntry]:
    """Load hull reference pool from E_form cache or an explicit parquet."""
    cfg = config if config is not None else load_mace_config()
    if reference_parquet is not None:
        frame = pd.read_parquet(reference_parquet)
    else:
        frame = load_eform_cache(
            "formation_energy_per_atom",
            protocol=str(
                (cfg.get("energy") or {}).get("reference_protocol")
                or PROTOCOL_ELEMENTAL_BULK_V1
            ),
            cache_path=cache_path,
            config=cfg,
        )
    return reference_entries_from_frame(frame)


def score_structures(
    *,
    structure_paths: list[str] | None = None,
    material_ids: list[str] | None = None,
    target: str = "formation_energy_per_atom",
    relax: bool | None = None,
    relax_fmax: float | None = None,
    relax_steps: int | None = None,
    compute_hull: bool = True,
    hull_reference: list[HullEntry] | None = None,
    hull_cache_path: Path | None = None,
    hull_reference_parquet: Path | None = None,
    device: str | None = "cpu",
    model_id: str | None = None,
    config: dict[str, Any] | None = None,
    project_root: Path | None = None,  # noqa: ARG001 — API parity; unused today
) -> pd.DataFrame:
    """Score candidates with frozen MACE: optional relax → E_form → E_hull.

    Relax knobs default from ``configs/mace.yaml`` → ``energy.score``.
    When ``compute_hull`` is True, an E_form cache or ``hull_reference`` is required.
    """
    cfg = config if config is not None else load_mace_config()
    score_cfg = (cfg.get("energy") or {}).get("score") or {}
    if relax is None:
        relax = bool(score_cfg.get("relax", True))
    if relax_fmax is None:
        relax_fmax = float(score_cfg.get("relax_fmax", 0.05))
    if relax_steps is None:
        relax_steps = int(score_cfg.get("relax_steps", 50))

    inputs = _resolve_structure_inputs(
        structure_paths=structure_paths,
        material_ids=material_ids,
        target=target,
        config=cfg,
    )

    calc, resolved_model, ckpt_path, resolved_device = load_foundation_calculator(
        model_id=model_id or foundation_model_id(cfg),
        device=device,
        config=cfg,
    )
    mus, energy_protocol = load_formation_references(
        resolved_model, calc=calc, config=cfg
    )

    refs = hull_reference
    hull_note: str | None = None
    if compute_hull and refs is None:
        refs = load_hull_reference_entries(
            cache_path=hull_cache_path,
            reference_parquet=hull_reference_parquet,
            config=cfg,
        )

    rows: list[dict[str, Any]] = []
    for item in inputs:
        structure_path = item["structure_path"]
        loaded = load_structure(structure_path, attach_atoms=True)
        base = {
            "material_id": item["material_id"],
            "structure_path": structure_path,
            "ok": False,
            "reason": loaded.reason,
            "detail": loaded.detail,
            "formula": loaded.formula,
            "n_atoms": loaded.n_atoms,
            "relaxed": bool(relax),
            "relax_converged": None,
            "relax_n_steps": None,
            "energy_total_eV": None,
            "energy_per_atom_eV": None,
            "formation_energy_per_atom": None,
            "e_above_hull": None,
            "reference_protocol": energy_protocol,
            "model_id": resolved_model,
            "device": resolved_device,
            "checkpoint": str(ckpt_path),
            "hull_note": hull_note,
        }
        if not loaded.ok or loaded.atoms is None:
            rows.append(base)
            continue

        atoms = loaded.atoms
        relax_info: dict[str, Any] = {}
        try:
            if relax:
                atoms, relax_info = relax_atoms_mace(
                    atoms,
                    calc,
                    fmax=relax_fmax,
                    steps=relax_steps,
                )
            else:
                atoms = atoms.copy()
                atoms.calc = calc
            e_tot = float(atoms.get_potential_energy())
            zs = [int(z) for z in atoms.get_atomic_numbers()]
            if any(z not in mus for z in zs):
                base["reason"] = "MISSING_ELEMENT_MU"
                base["detail"] = "atomic number missing from formation references"
                rows.append(base)
                continue
            e_form = formation_energy_per_atom(e_tot, zs, mus)
            n_atoms = int(len(atoms))
            formula = loaded.formula
            e_hull = None
            row_hull_note = hull_note
            if compute_hull and refs is not None and formula:
                try:
                    e_hull = predict_e_above_hull(
                        str(formula),
                        float(e_form),
                        refs,
                        protocol=PROTOCOL_MACE_PD_EFORM_V1,
                    )
                except Exception as exc:  # noqa: BLE001
                    row_hull_note = f"hull_failed: {exc}"
            rows.append(
                {
                    **base,
                    "ok": True,
                    "reason": "OK",
                    "detail": None,
                    "formula": formula,
                    "n_atoms": n_atoms,
                    "relaxed": bool(relax),
                    "relax_converged": relax_info.get("relax_converged"),
                    "relax_n_steps": relax_info.get("n_steps"),
                    "energy_total_eV": e_tot,
                    "energy_per_atom_eV": e_tot / float(n_atoms),
                    "formation_energy_per_atom": e_form,
                    "e_above_hull": e_hull,
                    "hull_note": row_hull_note,
                }
            )
        except Exception as exc:  # noqa: BLE001
            base["reason"] = "INFER_ERROR"
            base["detail"] = str(exc)
            rows.append(base)

    return pd.DataFrame(rows)
