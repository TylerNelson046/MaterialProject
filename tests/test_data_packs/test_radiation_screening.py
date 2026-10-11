"""Tests for radiation screening helpers and Young’s modulus."""

from __future__ import annotations

from data_packs.radiation_screening import (
    has_strong_neutron_absorber,
    macroscopic_absorption_cm,
    radiation_hardness_score,
    youngs_modulus_vrh,
)


def test_absorber_flag_gadolinium() -> None:
    assert has_strong_neutron_absorber({"Fe": 99.0, "Gd": 1.0}) is True
    assert has_strong_neutron_absorber({"Zr": 1.0}) is False


def test_macroscopic_absorption_positive() -> None:
    sigma_m = macroscopic_absorption_cm(
        sigma_barns=0.185,
        density_g_cm3=6.5,
        composition={"Zr": 1.0},
    )
    assert sigma_m is not None
    assert 0.001 < sigma_m < 1.0


def test_hardness_penalizes_absorber() -> None:
    clean = radiation_hardness_score(
        sigma_barns=0.2,
        melting_point_k=2100,
        bulk_modulus_gpa=90,
        density_g_cm3=6.5,
        has_absorber=False,
    )
    dirty = radiation_hardness_score(
        sigma_barns=0.2,
        melting_point_k=2100,
        bulk_modulus_gpa=90,
        density_g_cm3=6.5,
        has_absorber=True,
    )
    assert clean is not None and dirty is not None
    assert clean > dirty * 2


def test_youngs_from_bulk_shear() -> None:
    e = youngs_modulus_vrh(160.0, 80.0)
    assert e is not None
    # E = 9BG/(3B+G) = 9*160*80/(480+80) = 115200/560 ≈ 205.7
    assert abs(e - 205.714) < 0.1
