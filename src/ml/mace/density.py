"""Structure density protocols (Phase 4.7).

Pinned protocols (see ``docs/mace_plan.md`` / ``configs/mace.yaml``):

* ``geometry_v1`` — ρ (g/cm³) from the as-loaded cell via pymatgen
  ``Structure.density`` (no energy model).
* ``geometry_relax_v1`` — short ASE relaxation with MACE, then ``geometry_v1``.
"""

from __future__ import annotations

from typing import Any

PROTOCOL_GEOMETRY_V1 = "geometry_v1"
PROTOCOL_GEOMETRY_RELAX_V1 = "geometry_relax_v1"

DENSITY_PROTOCOLS = frozenset({PROTOCOL_GEOMETRY_V1, PROTOCOL_GEOMETRY_RELAX_V1})


def density_g_cm3_from_atoms(atoms: Any) -> float:
    """Return crystallographic density in g/cm³ for an ASE ``Atoms`` cell."""
    from pymatgen.io.ase import AseAtomsAdaptor

    structure = AseAtomsAdaptor.get_structure(atoms)
    value = float(structure.density)
    if value <= 0 or value != value:  # NaN check
        raise ValueError(f"Non-physical density: {value}")
    return value


def relax_atoms_mace(
    atoms: Any,
    calc: Any,
    *,
    fmax: float = 0.5,
    steps: int = 25,
) -> Any:
    """Short FIRE relax; density path keeps looser fmax defaults for speed."""
    from ml.mace.relax import relax_atoms_mace as _relax

    relaxed, _ = _relax(atoms, calc, fmax=fmax, steps=steps)
    return relaxed


def predict_density(
    atoms: Any,
    *,
    protocol: str = PROTOCOL_GEOMETRY_V1,
    calc: Any | None = None,
    relax_fmax: float = 0.5,
    relax_steps: int = 25,
) -> tuple[float, Any]:
    """Return ``(density_g_cm3, atoms_used)`` under the named protocol."""
    if protocol == PROTOCOL_GEOMETRY_V1:
        return density_g_cm3_from_atoms(atoms), atoms
    if protocol == PROTOCOL_GEOMETRY_RELAX_V1:
        if calc is None:
            raise ValueError(f"{PROTOCOL_GEOMETRY_RELAX_V1} requires a MACE calculator")
        relaxed = relax_atoms_mace(atoms, calc, fmax=relax_fmax, steps=relax_steps)
        return density_g_cm3_from_atoms(relaxed), relaxed
    raise ValueError(
        f"Unsupported density protocol {protocol!r}; "
        f"expected one of {sorted(DENSITY_PROTOCOLS)}"
    )
