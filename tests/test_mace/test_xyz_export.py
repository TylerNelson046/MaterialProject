"""XYZ export / energy-target helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("ase")

from ml.mace.xyz_export import (  # noqa: E402
    e0s_cli_string,
    export_split_xyz,
    total_energy_target_eV,
)


def test_total_energy_target_roundtrip() -> None:
    e0s = {1: -1.0, 8: -2.0}
    label = -0.5
    e_tot = total_energy_target_eV(label, [1, 1, 8], e0s)
    assert e_tot == pytest.approx(label * 3 + (-1 - 1 - 2))


def test_e0s_cli_string_is_mapping() -> None:
    text = e0s_cli_string({1: -1.1, 8: -2.2})
    assert text.startswith("{") and text.endswith("}")
    assert "1:" in text and "8:" in text


def test_export_split_xyz(tmp_path: Path) -> None:
    from ase.io import read

    struct = tmp_path / "c.json"
    struct.write_text(
        json.dumps(
            {
                "lattice_mat": [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
                "coords": [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
                "elements": ["C", "C"],
                "cartesian": True,
            }
        ),
        encoding="utf-8",
    )
    frame = pd.DataFrame(
        [
            {
                "material_id": "MAT-1",
                "split": "train",
                "structure_path": str(struct),
                "label": -0.25,
            }
        ]
    )
    e0s = {6: -1.0}
    out = tmp_path / "train.xyz"
    result = export_split_xyz(frame, split="train", out_path=out, e0s=e0s)
    assert result.n_written == 1
    atoms = read(str(out))
    expected = -0.25 * 2 + (-1.0 * 2)
    energy = atoms.info.get("REF_energy", atoms.info.get("energy"))
    if energy is None:
        energy = atoms.get_potential_energy()
    assert energy == pytest.approx(expected)
    assert atoms.info["material_id"] == "MAT-1"
