"""Density protocol unit tests (Phase 4.7)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("ase")
pytest.importorskip("pymatgen")

from ase import Atoms  # noqa: E402
from ase.build import bulk  # noqa: E402

from ml.mace.density import (  # noqa: E402
    PROTOCOL_GEOMETRY_V1,
    density_g_cm3_from_atoms,
    predict_density,
)


def test_density_diamond_si_reasonable() -> None:
    atoms = bulk("Si", "diamond", a=5.43)
    rho = density_g_cm3_from_atoms(atoms)
    # Crystallographic Si ~2.3 g/cm³
    assert 2.0 < rho < 2.6


def test_predict_density_geometry_protocol() -> None:
    atoms = Atoms(
        "Cu2",
        positions=[[0, 0, 0], [1.8, 1.8, 1.8]],
        cell=[3.6, 3.6, 3.6],
        pbc=True,
    )
    rho, used = predict_density(atoms, protocol=PROTOCOL_GEOMETRY_V1)
    assert used is atoms
    assert rho > 0
    assert np.isfinite(rho)


def test_predict_density_rejects_unknown_protocol() -> None:
    atoms = bulk("Al", "fcc", a=4.05)
    with pytest.raises(ValueError, match="Unsupported density protocol"):
        predict_density(atoms, protocol="not_a_protocol")
