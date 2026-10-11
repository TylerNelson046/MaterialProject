"""Candidate scoring helpers (relax → E_form → hull)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("ase")
pytest.importorskip("pymatgen")

from ase.build import bulk  # noqa: E402
from ase.calculators.calculator import Calculator, all_changes  # noqa: E402

from ml.mace.hull import HullEntry  # noqa: E402
from ml.mace.score import score_structures  # noqa: E402


class _ConstEnergyCalc(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        assert atoms is not None
        self.results = {
            "energy": -10.0,
            "forces": np.zeros((len(atoms), 3)),
        }


def _patch_score(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, atoms) -> Path:
    from ml.mace import score as score_mod
    from ml.mace.structures import StructureLoadResult

    struct_path = tmp_path / "si.json"
    struct_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        score_mod,
        "load_foundation_calculator",
        lambda **k: (_ConstEnergyCalc(), "medium-mpa-0", tmp_path / "m.model", "cpu"),
    )
    monkeypatch.setattr(
        score_mod,
        "load_formation_references",
        lambda *a, **k: ({14: -5.0}, "elemental_bulk_v1"),
    )
    monkeypatch.setattr(
        score_mod,
        "load_structure",
        lambda path, attach_atoms=True: StructureLoadResult(
            ok=True,
            reason="OK",
            path=str(path),
            n_atoms=len(atoms),
            formula="Si2",
            atomic_numbers=tuple(int(z) for z in atoms.get_atomic_numbers()),
            atoms=atoms.copy(),
        ),
    )
    monkeypatch.setattr(
        score_mod,
        "_resolve_structure_inputs",
        lambda **k: [{"material_id": "MAT-1", "structure_path": str(struct_path)}],
    )
    return struct_path


def test_score_structures_no_relax_no_hull(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    atoms = bulk("Si", "diamond", a=5.43)
    struct_path = _patch_score(monkeypatch, tmp_path, atoms)

    frame = score_structures(
        structure_paths=[str(struct_path)],
        relax=False,
        compute_hull=False,
        model_id="medium-mpa-0",
        config={
            "foundation": {"model_id": "medium-mpa-0"},
            "energy": {"reference_protocol": "elemental_bulk_v1"},
        },
        project_root=tmp_path,
    )
    assert len(frame) == 1
    assert bool(frame.iloc[0]["ok"])
    # E_form = (-10 - 2*(-5)) / 2 = 0
    assert frame.iloc[0]["formation_energy_per_atom"] == pytest.approx(0.0)
    assert frame.iloc[0]["e_above_hull"] is None or pd.isna(frame.iloc[0]["e_above_hull"])


def test_score_structures_with_hull_refs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    atoms = bulk("Si", "diamond", a=5.43)
    struct_path = _patch_score(monkeypatch, tmp_path, atoms)

    refs = [
        HullEntry("r1", "Si2", -0.1, 2),
        HullEntry("r2", "Si", 0.0, 1),
    ]
    frame = score_structures(
        structure_paths=[str(struct_path)],
        relax=False,
        compute_hull=True,
        hull_reference=refs,
        model_id="medium-mpa-0",
        config={
            "foundation": {"model_id": "medium-mpa-0"},
            "energy": {"reference_protocol": "elemental_bulk_v1"},
        },
        project_root=tmp_path,
    )
    assert bool(frame.iloc[0]["ok"])
    assert frame.iloc[0]["e_above_hull"] is not None
    assert float(frame.iloc[0]["e_above_hull"]) >= -1e-8
