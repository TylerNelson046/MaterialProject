"""Radiation / nuclear screening helpers (derived, not measured irradiation data).

- Strong neutron absorbers for control-rod vs cladding partitioning
- Macroscopic absorption Σ_a from microscopic σ_a and density
- Heuristic radiation_hardness_score for relative screening only
"""

from __future__ import annotations

from typing import Any

from data_packs.element_nuclear import THERMAL_ABSORPTION_BARNS

# Elements used as thermal-neutron poisons / strong absorbers in nuclear eng.
STRONG_NEUTRON_ABSORBERS: frozenset[str] = frozenset(
    {
        "B",
        "Cd",
        "Gd",
        "Sm",
        "Eu",
        "Dy",
        "Hf",
        "In",
        "Rh",
        "Ir",
        "Hg",
        "Li",  # ⁶Li absorption (natural Li still elevated)
        "Cl",
    }
)

# Approximate elemental atomic masses (g/mol) for mean molar mass.
ATOMIC_MASS: dict[str, float] = {
    "H": 1.008,
    "He": 4.003,
    "Li": 6.94,
    "Be": 9.012,
    "B": 10.81,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998,
    "Ne": 20.180,
    "Na": 22.990,
    "Mg": 24.305,
    "Al": 26.982,
    "Si": 28.085,
    "P": 30.974,
    "S": 32.06,
    "Cl": 35.45,
    "Ar": 39.948,
    "K": 39.098,
    "Ca": 40.078,
    "Sc": 44.956,
    "Ti": 47.867,
    "V": 50.942,
    "Cr": 51.996,
    "Mn": 54.938,
    "Fe": 55.845,
    "Co": 58.933,
    "Ni": 58.693,
    "Cu": 63.546,
    "Zn": 65.38,
    "Ga": 69.723,
    "Ge": 72.630,
    "As": 74.922,
    "Se": 78.971,
    "Br": 79.904,
    "Kr": 83.798,
    "Rb": 85.468,
    "Sr": 87.62,
    "Y": 88.906,
    "Zr": 91.224,
    "Nb": 92.906,
    "Mo": 95.95,
    "Tc": 98.0,
    "Ru": 101.07,
    "Rh": 102.91,
    "Pd": 106.42,
    "Ag": 107.87,
    "Cd": 112.41,
    "In": 114.82,
    "Sn": 118.71,
    "Sb": 121.76,
    "Te": 127.60,
    "I": 126.90,
    "Xe": 131.29,
    "Cs": 132.91,
    "Ba": 137.33,
    "La": 138.91,
    "Ce": 140.12,
    "Pr": 140.91,
    "Nd": 144.24,
    "Pm": 145.0,
    "Sm": 150.36,
    "Eu": 151.96,
    "Gd": 157.25,
    "Tb": 158.93,
    "Dy": 162.50,
    "Ho": 164.93,
    "Er": 167.26,
    "Tm": 168.93,
    "Yb": 173.05,
    "Lu": 174.97,
    "Hf": 178.49,
    "Ta": 180.95,
    "W": 183.84,
    "Re": 186.21,
    "Os": 190.23,
    "Ir": 192.22,
    "Pt": 195.08,
    "Au": 196.97,
    "Hg": 200.59,
    "Tl": 204.38,
    "Pb": 207.2,
    "Bi": 208.98,
    "Th": 232.04,
    "Pa": 231.04,
    "U": 238.03,
    "Np": 237.0,
    "Pu": 244.0,
    "Am": 243.0,
    "Cm": 247.0,
    "Bk": 247.0,
    "Cf": 251.0,
    "Ac": 227.0,
    "Ra": 226.0,
    "Po": 209.0,
}

N_AVOGADRO = 6.02214076e23
BARN_CM2 = 1.0e-24


def has_strong_neutron_absorber(
    composition: dict[str, float],
    *,
    min_atom_fraction: float = 1e-6,
) -> bool | None:
    """True if any strong-absorber element is present above ``min_atom_fraction``."""
    if not composition:
        return None
    total = sum(float(v) for v in composition.values() if float(v) > 0)
    if total <= 0:
        return None
    for el, amt in composition.items():
        if el in STRONG_NEUTRON_ABSORBERS and float(amt) / total >= min_atom_fraction:
            return True
    return False


def mean_atomic_mass(composition: dict[str, float]) -> float | None:
    total = 0.0
    mass = 0.0
    for el, amt in composition.items():
        if el not in ATOMIC_MASS:
            return None
        a = float(amt)
        if a <= 0:
            continue
        total += a
        mass += a * ATOMIC_MASS[el]
    if total <= 0:
        return None
    return mass / total


def macroscopic_absorption_cm(
    *,
    sigma_barns: float | None,
    density_g_cm3: float | None,
    composition: dict[str, float],
) -> float | None:
    """Σ_a (1/cm) ≈ N · σ with N = ρ N_A / M̄.

    ``sigma_barns`` should be the atom-fraction weighted microscopic σ_a.
    """
    if sigma_barns is None or density_g_cm3 is None:
        return None
    if sigma_barns < 0 or density_g_cm3 <= 0:
        return None
    mbar = mean_atomic_mass(composition)
    if mbar is None or mbar <= 0:
        return None
    n_atoms = density_g_cm3 * N_AVOGADRO / mbar
    return float(n_atoms * sigma_barns * BARN_CM2)


def radiation_hardness_score(
    *,
    sigma_barns: float | None,
    melting_point_k: float | None,
    bulk_modulus_gpa: float | None,
    density_g_cm3: float | None,
    has_absorber: bool | None,
) -> float | None:
    """Relative screening score in ~[0, 100]; higher ≈ preferred for cladding-like use.

    Combines low neutron absorption, high melting point, and stiffness.
    Strong absorbers are heavily penalized. Not a measured dpa lifetime.
    """
    if sigma_barns is None and melting_point_k is None and bulk_modulus_gpa is None:
        return None

    # Absorption term: Zr (~0.185 b) → high; Gd (~50 kb) → near zero.
    if sigma_barns is None:
        absorb_term = 40.0
    else:
        # soft saturation: score falls as log10(1+σ)
        absorb_term = 50.0 / (1.0 + float(sigma_barns) ** 0.35)

    if melting_point_k is None:
        melt_term = 15.0
    else:
        melt_term = 30.0 * min(max(float(melting_point_k) / 3000.0, 0.0), 1.2)

    if bulk_modulus_gpa is None:
        stiff_term = 8.0
    else:
        stiff_term = 15.0 * min(max(float(bulk_modulus_gpa) / 200.0, 0.0), 1.2)

    if density_g_cm3 is None:
        dens_term = 4.0
    else:
        # Prefer moderate density (structural metals ~4–12 g/cm3)
        dens = float(density_g_cm3)
        dens_term = 5.0 * max(0.0, 1.0 - abs(dens - 8.0) / 12.0)

    score = absorb_term + melt_term + stiff_term + dens_term
    if has_absorber:
        score *= 0.25
    return round(min(max(score, 0.0), 100.0), 3)


def youngs_modulus_vrh(
    bulk_gpa: float | None,
    shear_gpa: float | None,
    poisson: float | None = None,
) -> float | None:
    """Isotropic Young’s modulus from VRH bulk/shear (preferred) or G+ν."""
    if bulk_gpa is not None and shear_gpa is not None:
        b = float(bulk_gpa)
        g = float(shear_gpa)
        denom = 3.0 * b + g
        if b > 0 and g > 0 and denom > 0:
            return 9.0 * b * g / denom
    if shear_gpa is not None and poisson is not None:
        g = float(shear_gpa)
        nu = float(poisson)
        if g > 0 and -1.0 < nu < 0.5:
            return 2.0 * g * (1.0 + nu)
    return None


def parse_composition(raw: Any) -> dict[str, float] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        import json

        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None
    if not isinstance(raw, dict) or not raw:
        return None
    out: dict[str, float] = {}
    for key, val in raw.items():
        try:
            amount = float(val)
        except (TypeError, ValueError):
            return None
        if amount > 0:
            out[str(key)] = amount
    return out or None


def absorber_fraction(composition: dict[str, float]) -> float:
    total = sum(composition.values())
    if total <= 0:
        return 0.0
    return sum(composition.get(el, 0.0) for el in STRONG_NEUTRON_ABSORBERS) / total


# Re-export for convenience
__all__ = [
    "ATOMIC_MASS",
    "STRONG_NEUTRON_ABSORBERS",
    "THERMAL_ABSORPTION_BARNS",
    "absorber_fraction",
    "has_strong_neutron_absorber",
    "macroscopic_absorption_cm",
    "mean_atomic_mass",
    "parse_composition",
    "radiation_hardness_score",
    "youngs_modulus_vrh",
]
