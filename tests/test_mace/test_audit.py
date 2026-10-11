"""Structure audit writer tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("ase")

from ml.mace.audit import audit_structures  # noqa: E402


def _tiny_jarvis(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "lattice_mat": [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
                "coords": [[0.0, 0.0, 0.0]],
                "elements": ["C"],
                "cartesian": True,
            }
        ),
        encoding="utf-8",
    )


def test_audit_structures_writes_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ml.mace import audit as audit_mod
    from ml.mace import config as config_mod

    export_dir = tmp_path / "export"
    targets_dir = export_dir / "targets"
    targets_dir.mkdir(parents=True)
    struct = tmp_path / "data" / "structures" / "computational" / "x.json"
    struct.parent.mkdir(parents=True)
    _tiny_jarvis(struct)

    frame = pd.DataFrame(
        [
            {
                "material_id": "MAT-1",
                "split": "train",
                "source": "jarvis",
                "structure_path": str(struct.relative_to(tmp_path)),
            },
            {
                "material_id": "MAT-2",
                "split": "test",
                "source": "jarvis",
                "structure_path": None,
            },
        ]
    )
    frame.to_parquet(targets_dir / "formation_energy_per_atom.parquet", index=False)

    cfg = {
        "export": {
            "dir": str(export_dir),
            "targets_subdir": "targets",
            "mace_pack_dir": str(tmp_path / "mace_exports"),
        },
        "structure": {"path_column": "structure_path", "max_atoms_eval": None},
        "targets": {
            "formation_energy_per_atom": {
                "parquet": "formation_energy_per_atom.parquet",
            }
        },
    }
    monkeypatch.setattr(config_mod, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(audit_mod, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(audit_mod, "load_mace_config", lambda path=None: cfg)

    from ml.mace import structures as structures_mod

    monkeypatch.setattr(structures_mod, "PROJECT_ROOT", tmp_path)

    result = audit_structures(
        "formation_energy_per_atom",
        config=cfg,
        workers=1,
        progress_every=0,
    )
    assert result.n_rows == 2
    assert result.n_ok == 1
    assert result.n_fail == 1
    assert result.audit_parquet.is_file()
    assert result.summary_md.is_file()
    audit = pd.read_parquet(result.audit_parquet)
    assert set(audit["reason"]) == {"OK", "MISSING_PATH"}
