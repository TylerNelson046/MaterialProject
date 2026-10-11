"""Physical plausibility gates for normalized material properties.

Out-of-range values are treated as missing (dropped) rather than clipped, so
training/search never silently trusts absurd numbers. Ranges are intentionally
conservative scientific defaults for DFT/experimental inorganic solids.
"""

from __future__ import annotations

from typing import Any

# Density (g/cm³)
DENSITY_LO = 0.01
DENSITY_HI = 30.0

# Formation energy (eV/atom)
FORMATION_ENERGY_LO = -10.0
FORMATION_ENERGY_HI = 5.0

# Energy above hull (eV/atom) — allow tiny negatives from numeric noise
EHULL_LO = -1.0e-4
EHULL_HI = 5.0

# Band gap (eV)
BAND_GAP_LO = 0.0
BAND_GAP_HI = 20.0

# Elastic moduli (GPa)
ELASTIC_LO = 0.0
ELASTIC_HI = 1000.0

# Poisson ratio
POISSON_LO = -1.0
POISSON_HI = 0.5

# Universal anisotropy (dimensionless)
ANISOTROPY_ABS_MAX = 1.0e4

# Volume (Å³)
VOLUME_LO = 1.0e-3
VOLUME_HI = 1.0e5

# AFLOW AGL thermal (aligned with download_aflow_thermal ingest gates)
THERMAL_KAPPA_LO = 0.001
THERMAL_KAPPA_HI = 5000.0
DEBYE_LO = 1.0
DEBYE_HI = 5000.0
HEAT_CAPACITY_CV_LO = 0.0
HEAT_CAPACITY_CV_HI = 1.0e6


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return number


def in_range(value: Any, lo: float, hi: float) -> float | None:
    number = finite_float(value)
    if number is None or number < lo or number > hi:
        return None
    return number


def sanitize_density(value: Any) -> float | None:
    return in_range(value, DENSITY_LO, DENSITY_HI)


def sanitize_formation_energy(value: Any) -> float | None:
    return in_range(value, FORMATION_ENERGY_LO, FORMATION_ENERGY_HI)


def sanitize_ehull(value: Any) -> float | None:
    return in_range(value, EHULL_LO, EHULL_HI)


def sanitize_band_gap(value: Any) -> float | None:
    return in_range(value, BAND_GAP_LO, BAND_GAP_HI)


def sanitize_elastic(value: Any) -> float | None:
    return in_range(value, ELASTIC_LO, ELASTIC_HI)


def sanitize_poisson(value: Any) -> float | None:
    return in_range(value, POISSON_LO, POISSON_HI)


def sanitize_anisotropy(value: Any) -> float | None:
    number = finite_float(value)
    if number is None or abs(number) > ANISOTROPY_ABS_MAX:
        return None
    return number


def sanitize_volume(value: Any) -> float | None:
    return in_range(value, VOLUME_LO, VOLUME_HI)


def sanitize_thermal_kappa(value: Any) -> float | None:
    return in_range(value, THERMAL_KAPPA_LO, THERMAL_KAPPA_HI)


def sanitize_debye(value: Any) -> float | None:
    return in_range(value, DEBYE_LO, DEBYE_HI)


# Properties → (lo, hi) for bulk DB repair DELETE of OOR rows.
PROPERTY_RANGE_GATES: dict[str, tuple[float, float]] = {
    "density": (DENSITY_LO, DENSITY_HI),
    "formation_energy_per_atom": (FORMATION_ENERGY_LO, FORMATION_ENERGY_HI),
    "energy_above_hull": (EHULL_LO, EHULL_HI),
    "band_gap": (BAND_GAP_LO, BAND_GAP_HI),
    "mbj_bandgap": (BAND_GAP_LO, BAND_GAP_HI),
    "bulk_modulus_vrh": (ELASTIC_LO, ELASTIC_HI),
    "shear_modulus_vrh": (ELASTIC_LO, ELASTIC_HI),
    "youngs_modulus_vrh": (ELASTIC_LO, ELASTIC_HI),
    "homogeneous_poisson": (POISSON_LO, POISSON_HI),
    "volume": (VOLUME_LO, VOLUME_HI),
    "thermal_conductivity": (THERMAL_KAPPA_LO, THERMAL_KAPPA_HI),
    "thermal_conductivity_proxy": (THERMAL_KAPPA_LO, THERMAL_KAPPA_HI),
    "debye_temperature": (DEBYE_LO, DEBYE_HI),
    "heat_capacity_cv": (HEAT_CAPACITY_CV_LO, HEAT_CAPACITY_CV_HI),
}

# Absolute-value gates (null when abs(value) > max).
PROPERTY_ABS_GATES: dict[str, float] = {
    "universal_anisotropy": ANISOTROPY_ABS_MAX,
}
