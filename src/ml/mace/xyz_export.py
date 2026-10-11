"""Export MACE pack rows to ASE extxyz with total-energy training targets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

from ml.mace.reference import formation_energy_per_atom, load_formation_references
from ml.mace.structures import load_structure


@dataclass(frozen=True)
class XyzExportResult:
    """Paths and counts for one split's extxyz export."""

    split: str
    path: Path
    n_written: int
    n_failed: int


def total_energy_target_eV(
    label_formation_per_atom: float,
    atomic_numbers: list[int] | tuple[int, ...],
    e0s: dict[int, float],
) -> float:
    """Invert formation reference: ``E_tot = E_form/atom * n + Σ μ[Z]``."""
    zs = [int(z) for z in atomic_numbers]
    n = len(zs)
    if n < 1:
        raise ValueError("empty atomic_numbers")
    # Consistency check against the forward map.
    e_tot = float(label_formation_per_atom) * float(n) + float(sum(e0s[z] for z in zs))
    roundtrip = formation_energy_per_atom(e_tot, zs, e0s)
    if abs(roundtrip - float(label_formation_per_atom)) > 1e-6:
        raise RuntimeError(
            f"E0 round-trip failed: label={label_formation_per_atom} got={roundtrip}"
        )
    return e_tot


def export_split_xyz(
    frame: pd.DataFrame,
    *,
    split: str,
    out_path: Path,
    e0s: dict[int, float],
    limit: int | None = None,
    progress_every: int = 2000,
) -> XyzExportResult:
    """Write extxyz for one split. ``frame`` must include structure_path + label."""
    part = frame.loc[frame["split"].astype(str) == split].copy()
    if limit is not None:
        part = part.head(int(limit))

    atoms_list = []
    n_failed = 0
    for i, rec in enumerate(part.itertuples(index=False), start=1):
        loaded = load_structure(rec.structure_path, attach_atoms=True)
        if not loaded.ok or loaded.atoms is None:
            n_failed += 1
            continue
        atoms = loaded.atoms.copy()
        zs = [int(z) for z in atoms.get_atomic_numbers()]
        try:
            e_tot = total_energy_target_eV(float(rec.label), zs, e0s)
        except Exception:  # noqa: BLE001
            n_failed += 1
            continue
        # ASE>=3.23 + mace-torch: use REF_energy (see mace.data.utils).
        # Avoid atoms.info["energy"] + SinglePointCalculator clash on write.
        atoms.info.pop("energy", None)
        atoms.info["REF_energy"] = float(e_tot)
        atoms.info["material_id"] = str(rec.material_id)
        atoms.info["config_type"] = "Default"
        # Zero forces placeholders: WeightedEnergyForcesLoss still indexes forces
        # even when forces_weight=0.
        n = len(atoms)
        zero_f = np.zeros((n, 3), dtype=np.float64)
        atoms.arrays["REF_forces"] = zero_f
        atoms.calc = SinglePointCalculator(atoms, energy=float(e_tot), forces=zero_f)
        if "forces" in atoms.arrays:
            del atoms.arrays["forces"]
        atoms_list.append(atoms)
        if progress_every and i % int(progress_every) == 0:
            print(f"[xyz:{split}] prepared {i}/{len(part)} …", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if atoms_list:
        write(str(out_path), atoms_list, format="extxyz")
    else:
        out_path.write_text("", encoding="utf-8")

    return XyzExportResult(
        split=split,
        path=out_path,
        n_written=len(atoms_list),
        n_failed=n_failed,
    )


def export_pack_xyz(
    frame: pd.DataFrame,
    *,
    out_dir: Path,
    e0s: dict[int, float] | None = None,
    model_id: str | None = None,
    config: dict[str, Any] | None = None,
    max_train_rows: int | None = None,
    max_val_rows: int | None = None,
    max_test_rows: int | None = None,
) -> dict[str, XyzExportResult]:
    """Export train/val/test extxyz files under ``out_dir``."""
    if e0s is None:
        e0s, _ = load_formation_references(model_id, config=config)
    limits = {
        "train": max_train_rows,
        "val": max_val_rows if max_val_rows is not None else max_train_rows,
        "test": max_test_rows if max_test_rows is not None else max_train_rows,
    }
    results: dict[str, XyzExportResult] = {}
    for split in ("train", "val", "test"):
        results[split] = export_split_xyz(
            frame,
            split=split,
            out_path=out_dir / f"{split}.xyz",
            e0s=e0s,
            limit=limits[split],
        )
    return results


def e0s_cli_string(e0s: dict[int, float]) -> str:
    """Format E0 table for ``mace_run_train --E0s``."""
    inner = ", ".join(f"{z}: {e0s[z]!r}" for z in sorted(e0s))
    return "{" + inner + "}"
