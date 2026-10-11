"""Reference-energy helpers."""

from __future__ import annotations

import pytest

from ml.mace.reference import formation_energy_per_atom


def test_formation_energy_per_atom_formula() -> None:
    e0s = {1: -1.0, 8: -2.0}
    # H2O: E_tot = -10, refs = -1-1-2 = -4 → form/atom = (-10 - -4)/3 = -2
    value = formation_energy_per_atom(-10.0, [1, 1, 8], e0s)
    assert value == pytest.approx(-2.0)


def test_formation_energy_missing_e0() -> None:
    with pytest.raises(KeyError, match="Missing E0"):
        formation_energy_per_atom(-1.0, [1, 99], {1: 0.0})
