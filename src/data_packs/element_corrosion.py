"""Element corrosion / oxidation screening proxies (not Pourbaix solvers)."""

from __future__ import annotations

# Higher = more corrosion-resistant in aqueous/oxidizing environments (heuristic).
CORROSION_RESISTANCE: dict[str, float] = {
    "H": 20, "He": 90, "Li": 8, "Be": 58, "B": 50, "C": 55, "N": 40, "O": 20,
    "F": 5, "Ne": 90, "Na": 5, "Mg": 15, "Al": 65, "Si": 60, "P": 30, "S": 25,
    "Cl": 10, "Ar": 90, "K": 5, "Ca": 10, "Sc": 45, "Ti": 82, "V": 50, "Cr": 76,
    "Mn": 30, "Fe": 40, "Co": 55, "Ni": 70, "Cu": 58, "Zn": 35, "Ga": 45, "Ge": 48,
    "As": 35, "Se": 30, "Br": 10, "Kr": 90, "Rb": 5, "Sr": 12, "Y": 48, "Zr": 80,
    "Nb": 78, "Mo": 74, "Tc": 60, "Ru": 88, "Rh": 94, "Pd": 90, "Ag": 75, "Cd": 40,
    "In": 42, "Sn": 50, "Sb": 40, "Te": 35, "I": 10, "Xe": 90, "Cs": 5, "Ba": 12,
    "La": 35, "Ce": 35, "Pr": 34, "Nd": 34, "Pm": 33, "Sm": 33, "Eu": 32, "Gd": 35,
    "Tb": 34, "Dy": 34, "Ho": 34, "Er": 34, "Tm": 34, "Yb": 32, "Lu": 36,
    "Hf": 78, "Ta": 85, "W": 72, "Re": 70, "Os": 86, "Ir": 96, "Pt": 98, "Au": 100,
    "Hg": 70, "Tl": 30, "Pb": 45, "Bi": 40, "Th": 40, "Pa": 30, "U": 25, "Np": 22,
    "Pu": 20, "Am": 20, "Cm": 20, "Ac": 25, "Ra": 10,
}

# Higher = more readily oxidized / reactive (rough inverse nobility).
OXIDATION_TENDENCY: dict[str, float] = {
    "H": 40, "He": 1, "Li": 100, "Be": 70, "B": 30, "C": 25, "N": 20, "O": 5,
    "F": 5, "Ne": 1, "Na": 96, "Mg": 85, "Al": 75, "Si": 35, "P": 35, "S": 40,
    "Cl": 15, "Ar": 1, "K": 98, "Ca": 94, "Sc": 72, "Ti": 70, "V": 60, "Cr": 55,
    "Mn": 58, "Fe": 50, "Co": 40, "Ni": 35, "Cu": 30, "Zn": 55, "Ga": 50, "Ge": 40,
    "As": 35, "Se": 30, "Br": 15, "Kr": 1, "Rb": 97, "Sr": 92, "Y": 70, "Zr": 68,
    "Nb": 55, "Mo": 45, "Tc": 40, "Ru": 20, "Rh": 8, "Pd": 5, "Ag": 15, "Cd": 45,
    "In": 48, "Sn": 40, "Sb": 35, "Te": 30, "I": 15, "Xe": 1, "Cs": 99, "Ba": 90,
    "La": 75, "Ce": 78, "Pr": 76, "Nd": 76, "Pm": 75, "Sm": 74, "Eu": 80, "Gd": 74,
    "Tb": 74, "Dy": 73, "Ho": 73, "Er": 73, "Tm": 73, "Yb": 78, "Lu": 72,
    "Hf": 66, "Ta": 50, "W": 40, "Re": 35, "Os": 15, "Ir": 2, "Pt": 1, "Au": 2,
    "Hg": 25, "Tl": 45, "Pb": 38, "Bi": 35, "Th": 65, "Pa": 70, "U": 70, "Np": 72,
    "Pu": 75, "Am": 76, "Cm": 75, "Ac": 80, "Ra": 88,
}


def atom_weighted(composition: dict[str, float], table: dict[str, float]) -> float | None:
    """Atom-fraction weighted average; None if any element is missing from table."""
    total = 0.0
    weighted = 0.0
    for el, amt in composition.items():
        if el not in table:
            return None
        a = float(amt)
        if a <= 0:
            continue
        total += a
        weighted += a * table[el]
    if total <= 0:
        return None
    return weighted / total
