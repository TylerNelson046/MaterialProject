"""Tests for physical plausibility gates."""

from __future__ import annotations

from normalization.physical_ranges import (
    sanitize_band_gap,
    sanitize_density,
    sanitize_ehull,
    sanitize_elastic,
    sanitize_formation_energy,
    sanitize_poisson,
)


def test_sanitize_density_drops_oor() -> None:
    assert sanitize_density(2.5) == 2.5
    assert sanitize_density(0.0) is None
    assert sanitize_density(50.0) is None


def test_sanitize_formation_energy_bounds() -> None:
    assert sanitize_formation_energy(-3.0) == -3.0
    assert sanitize_formation_energy(-20.0) is None
    assert sanitize_formation_energy(10.0) is None


def test_sanitize_ehull_allows_tiny_negative() -> None:
    assert sanitize_ehull(-5e-5) == -5e-5
    assert sanitize_ehull(-1.0) is None
    assert sanitize_ehull(0.2) == 0.2


def test_sanitize_band_gap_and_elastic() -> None:
    assert sanitize_band_gap(-0.1) is None
    assert sanitize_band_gap(1.2) == 1.2
    assert sanitize_elastic(-1.0) is None
    assert sanitize_elastic(150.0) == 150.0
    assert sanitize_poisson(0.3) == 0.3
    assert sanitize_poisson(0.9) is None
