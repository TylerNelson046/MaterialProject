"""MACE structure pack export tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("ase")

from ml.mace.pack import export_mace_pack, load_mace_pack_frame  # noqa: E402


def _jarvis(path: Path, element: str = "C") -> None:
    path.write_text(
        json.dumps(
            {
                "lattice_mat": [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
                "coords": [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
                "elements": [element, element],
                "cartesian": True,
            }
        ),
        encoding="utf-8",
    )


def test_export_mace_pack_from_audit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ml.mace import audit as audit_mod
    from ml.mace import config as config_mod
    from ml.mace import pack as pack_mod
    from ml.mace import structures as structures_mod

    export_dir = tmp_path / "training_exports"
    targets = export_dir / "targets"
    targets.mkdir(parents=True)
    s1 = tmp_path / "data/structures/computational/a.json"
    s2 = tmp_path / "data/structures/computational/b.json"
    s1.parent.mkdir(parents=True)
    _jarvis(s1, "C")
    _jarvis(s2, "Si")

    src = pd.DataFrame(
        [
            {
                "material_id": "MAT-1",
                "source": "jarvis",
                "source_id": "J1",
                "formula_pretty": "C2",
                "chemsys": "C",
                "nelements": 1,
                "nsites": 2,
                "volume": 125.0,
                "density": 1.0,
                "space_group_number": 225,
                "space_group_symbol": "Fm-3m",
                "crystal_system": "cubic",
                "structure_path": str(s1.relative_to(tmp_path)),
                "n_sources": 1,
                "group_id": "g1",
                "split": "train",
                "label": -1.2,
            },
            {
                "material_id": "MAT-2",
                "source": "oqmd",
                "source_id": "o2",
                "formula_pretty": "Si2",
                "chemsys": "Si",
                "nelements": 1,
                "nsites": 2,
                "volume": 125.0,
                "density": 2.0,
                "space_group_number": 227,
                "space_group_symbol": "Fd-3m",
                "crystal_system": "cubic",
                "structure_path": None,
                "n_sources": 1,
                "group_id": "g2",
                "split": "test",
                "label": -0.5,
            },
            {
                "material_id": "MAT-3",
                "source": "jarvis",
                "source_id": "J3",
                "formula_pretty": "Si2",
                "chemsys": "Si",
                "nelements": 1,
                "nsites": 2,
                "volume": 125.0,
                "density": 2.0,
                "space_group_number": 227,
                "space_group_symbol": "Fd-3m",
                "crystal_system": "cubic",
                "structure_path": str(s2.relative_to(tmp_path)),
                "n_sources": 1,
                "group_id": "g3",
                "split": "val",
                "label": -0.8,
            },
        ]
    )
    src.to_parquet(targets / "formation_energy_per_atom.parquet", index=False)
    (export_dir / "db_freeze.json").write_text(
        json.dumps({"sha256": "abc", "snapshot_path": "x.duckdb"}),
        encoding="utf-8",
    )
    (export_dir / "export_manifest.json").write_text(
        json.dumps({"export_version": "4.1.0"}),
        encoding="utf-8",
    )

    # Audit: MAT-1 and MAT-3 OK; MAT-2 missing.
    audit_dir = tmp_path / "mace_exports" / "structure_audit" / "formation_energy_per_atom"
    audit_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "material_id": "MAT-1",
                "split": "train",
                "source": "jarvis",
                "structure_path": str(s1.relative_to(tmp_path)),
                "load_ok": True,
                "reason": "OK",
                "detail": None,
                "n_atoms": 2,
                "formula": "C2",
                "n_elements": 1,
                "max_atomic_number": 6,
            },
            {
                "material_id": "MAT-2",
                "split": "test",
                "source": "oqmd",
                "structure_path": None,
                "load_ok": False,
                "reason": "MISSING_PATH",
                "detail": None,
                "n_atoms": None,
                "formula": None,
                "n_elements": None,
                "max_atomic_number": None,
            },
            {
                "material_id": "MAT-3",
                "split": "val",
                "source": "jarvis",
                "structure_path": str(s2.relative_to(tmp_path)),
                "load_ok": True,
                "reason": "OK",
                "detail": None,
                "n_atoms": 2,
                "formula": "Si2",
                "n_elements": 1,
                "max_atomic_number": 14,
            },
        ]
    ).to_parquet(audit_dir / "audit.parquet", index=False)

    cfg = {
        "export": {
            "dir": str(export_dir),
            "targets_subdir": "targets",
            "mace_pack_dir": str(tmp_path / "mace_exports"),
        },
        "structure": {"path_column": "structure_path", "max_atoms_train": None},
        "energy": {"label_unit": "eV/atom"},
        "targets": {
            "formation_energy_per_atom": {"parquet": "formation_energy_per_atom.parquet"}
        },
    }
    for mod in (pack_mod, audit_mod, config_mod, structures_mod):
        monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)

    result = export_mace_pack(
        "formation_energy_per_atom",
        config=cfg,
        require_audit=True,
        max_atoms=1,  # both structures have 2 atoms → oversized
    )
    assert result.n_source_rows == 3
    assert result.n_pack_rows == 2
    assert result.n_oversized == 2
    assert result.n_train == 1
    assert result.n_val == 1
    assert result.n_test == 0
    assert result.manifest_path.is_file()
    assert result.dataset_card_path.is_file()
    assert (tmp_path / "mace_exports" / "db_freeze.json").is_file()

    packed = pd.read_parquet(result.target_parquet)
    assert set(packed["material_id"]) == {"MAT-1", "MAT-3"}
    assert "label" in packed.columns
    assert packed["mace_train_ok"].tolist() == [False, False]

    # load helper with train_ok_only drops oversized
    monkeypatch.setattr(pack_mod, "load_mace_config", lambda path=None: cfg)
    empty = load_mace_pack_frame("formation_energy_per_atom", train_ok_only=True, config=cfg)
    assert len(empty) == 0
    all_rows = load_mace_pack_frame(
        "formation_energy_per_atom", train_ok_only=False, config=cfg
    )
    assert len(all_rows) == 2
