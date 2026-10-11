"""Tests for element nuclear reference weighting."""

from __future__ import annotations

from data_packs.element_nuclear import (
    THERMAL_ABSORPTION_BARNS,
    atom_fraction_weighted,
)


def test_weighted_sigma_zirconium_rich() -> None:
    # Pure Zr should match table.
    assert atom_fraction_weighted({"Zr": 1.0}, THERMAL_ABSORPTION_BARNS) == THERMAL_ABSORPTION_BARNS["Zr"]


def test_weighted_gadolinium_drives_high_absorption() -> None:
    sigma = atom_fraction_weighted({"Fe": 99.0, "Gd": 1.0}, THERMAL_ABSORPTION_BARNS)
    assert sigma is not None
    assert sigma > 400  # Gd dominates even at 1%


def test_missing_element_returns_none() -> None:
    assert atom_fraction_weighted({"Uu": 1.0}, THERMAL_ABSORPTION_BARNS) is None
