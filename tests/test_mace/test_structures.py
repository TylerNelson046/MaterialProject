"""Structure I/O tests (pymatgen + ase; no foundation weights)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ase = pytest.importorskip("ase")

from ml.mace.structures import (  # noqa: E402
    REASON_ELEMENT_OUT_OF_COVERAGE,
    REASON_FILE_NOT_FOUND,
    REASON_MISSING_PATH,
    REASON_OK,
    REASON_PARSE_ERROR,
    load_structure,
    resolve_structure_path,
)


def _write_jarvis_si(path: Path) -> None:
    payload = {
        "lattice_mat": [[5.43, 0.0, 0.0], [0.0, 5.43, 0.0], [0.0, 0.0, 5.43]],
        "coords": [[0.0, 0.0, 0.0], [1.3575, 1.3575, 1.3575]],
        "elements": ["Si", "Si"],
        "cartesian": True,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_pymatgen_si(path: Path) -> None:
    from pymatgen.core import Lattice, Structure

    structure = Structure(Lattice.cubic(5.43), ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])
    path.write_text(structure.to_json(), encoding="utf-8")


def test_resolve_structure_path_relative(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ml.mace import structures as structures_mod

    monkeypatch.setattr(structures_mod, "PROJECT_ROOT", tmp_path)
    rel = Path("data/structures/computational/x.json")
    (tmp_path / rel).parent.mkdir(parents=True)
    (tmp_path / rel).write_text("{}", encoding="utf-8")
    resolved = resolve_structure_path(str(rel), project_root=tmp_path)
    assert resolved == tmp_path / rel


def test_load_jarvis_json(tmp_path: Path) -> None:
    path = tmp_path / "si.json"
    _write_jarvis_si(path)
    result = load_structure(path, project_root=tmp_path)
    assert result.ok
    assert result.reason == REASON_OK
    assert result.n_atoms == 2
    assert result.atoms is not None
    assert len(result.atoms) == 2


def test_load_pymatgen_json(tmp_path: Path) -> None:
    path = tmp_path / "si_pmg.json"
    _write_pymatgen_si(path)
    result = load_structure(path, project_root=tmp_path)
    assert result.ok
    assert result.n_atoms == 2


def test_missing_path() -> None:
    result = load_structure(None)
    assert not result.ok
    assert result.reason == REASON_MISSING_PATH


def test_file_not_found(tmp_path: Path) -> None:
    result = load_structure("missing.json", project_root=tmp_path)
    assert not result.ok
    assert result.reason == REASON_FILE_NOT_FOUND


def test_parse_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not-json", encoding="utf-8")
    result = load_structure(path, project_root=tmp_path)
    assert not result.ok
    assert result.reason == REASON_PARSE_ERROR


def test_element_out_of_coverage(tmp_path: Path) -> None:
    # Po (Z=84) is outside MACE-MPA-0 coverage.
    payload = {
        "lattice_mat": [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        "coords": [[0.0, 0.0, 0.0]],
        "elements": ["Po"],
        "cartesian": True,
    }
    path = tmp_path / "po.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = load_structure(path, project_root=tmp_path)
    assert not result.ok
    assert result.reason == REASON_ELEMENT_OUT_OF_COVERAGE


def test_load_real_jarvis_fixture_if_present() -> None:
    path = Path("data/structures/computational/JVASP-10.json")
    if not path.is_file():
        pytest.skip("local JARVIS structure not present")
    result = load_structure(path)
    assert result.ok, (result.reason, result.detail)
    assert result.n_atoms and result.n_atoms >= 1


def test_load_mp_json_with_null_site_properties() -> None:
    path = Path("data/structures/computational/mp-aaacpyaz.json")
    if not path.is_file():
        pytest.skip("local MP structure not present")
    result = load_structure(path)
    assert result.ok, (result.reason, result.detail)
    assert result.n_atoms and result.n_atoms >= 1


def test_pymatgen_json_null_selective_dynamics(tmp_path: Path) -> None:
    from pymatgen.core import Lattice, Structure

    structure = Structure(
        Lattice.cubic(4.0),
        ["Na", "Cl"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    payload = json.loads(structure.to_json())
    for site in payload["sites"]:
        site.setdefault("properties", {})
        site["properties"]["selective_dynamics"] = None
    path = tmp_path / "mp_like.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = load_structure(path, project_root=tmp_path)
    assert result.ok, (result.reason, result.detail)
    assert result.n_atoms == 2
