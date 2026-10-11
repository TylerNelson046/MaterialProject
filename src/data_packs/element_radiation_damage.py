"""Displacement-threshold energies (eV) for radiation-damage screening.

Approximate elemental E_d values used in NRT-like heuristics. Compound score is
an atom-fraction mix — screening only, not measured dpa.
"""

from __future__ import annotations

# Typical literature order-of-magnitude E_d (eV)
DISPLACEMENT_ENERGY_EV: dict[str, float] = {
    "H": 10, "He": 10, "Li": 10, "Be": 25, "B": 25, "C": 28, "N": 30, "O": 30,
    "F": 25, "Ne": 20, "Na": 10, "Mg": 20, "Al": 25, "Si": 25, "P": 20, "S": 20,
    "Cl": 20, "Ar": 20, "K": 10, "Ca": 15, "Sc": 25, "Ti": 30, "V": 40, "Cr": 40,
    "Mn": 40, "Fe": 40, "Co": 40, "Ni": 40, "Cu": 30, "Zn": 20, "Ga": 20, "Ge": 25,
    "As": 25, "Se": 20, "Br": 20, "Kr": 20, "Rb": 10, "Sr": 15, "Y": 40, "Zr": 40,
    "Nb": 60, "Mo": 60, "Tc": 60, "Ru": 50, "Rh": 50, "Pd": 40, "Ag": 30, "Cd": 20,
    "In": 20, "Sn": 20, "Sb": 20, "Te": 20, "I": 20, "Xe": 20, "Cs": 10, "Ba": 15,
    "La": 35, "Ce": 35, "Pr": 35, "Nd": 35, "Pm": 35, "Sm": 35, "Eu": 35, "Gd": 35,
    "Tb": 35, "Dy": 35, "Ho": 35, "Er": 35, "Tm": 35, "Yb": 30, "Lu": 35,
    "Hf": 50, "Ta": 90, "W": 90, "Re": 90, "Os": 60, "Ir": 50, "Pt": 40, "Au": 30,
    "Hg": 20, "Tl": 20, "Pb": 20, "Bi": 20, "Th": 40, "Pa": 40, "U": 40, "Np": 35,
    "Pu": 35, "Am": 35, "Cm": 35, "Ac": 30, "Ra": 15,
}


def displacement_energy_mix(composition: dict[str, float]) -> float | None:
    total = 0.0
    weighted = 0.0
    for el, amt in composition.items():
        if el not in DISPLACEMENT_ENERGY_EV:
            return None
        a = float(amt)
        if a <= 0:
            continue
        total += a
        weighted += a * DISPLACEMENT_ENERGY_EV[el]
    if total <= 0:
        return None
    return weighted / total


def radiation_damage_susceptibility(
    *,
    ed_mix: float | None,
    density: float | None,
    bulk_modulus: float | None,
) -> float | None:
    """Higher = more susceptible to displacement damage (0–100 screening).

    Lower E_d and lower stiffness → higher susceptibility.
    """
    if ed_mix is None:
        return None
    # Map E_d ~10–90 eV → susceptibility contribution
    ed_term = 60.0 * max(0.0, min(1.0, (50.0 - float(ed_mix)) / 40.0 + 0.5))
    if bulk_modulus is None:
        stiff_term = 20.0
    else:
        stiff_term = 25.0 * max(0.0, 1.0 - min(float(bulk_modulus) / 200.0, 1.0))
    if density is None:
        dens_term = 10.0
    else:
        dens_term = 15.0 * max(0.0, 1.0 - abs(float(density) - 8.0) / 12.0)
    return round(min(max(ed_term + stiff_term + dens_term, 0.0), 100.0), 3)
