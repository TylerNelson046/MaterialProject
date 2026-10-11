"""Shared short MACE relaxation helper (candidate scoring / density)."""

from __future__ import annotations

from typing import Any


def relax_atoms_mace(
    atoms: Any,
    calc: Any,
    *,
    fmax: float = 0.05,
    steps: int = 50,
) -> tuple[Any, dict[str, Any]]:
    """FIRE-relax a copy of ``atoms``; return ``(relaxed, info)``.

    Defaults are tighter than the density smoke path (fmax=0.05 eV/Å) because
    candidate scoring / hull distance cares about near-equilibrium geometries.
    """
    from ase.optimize import FIRE

    work = atoms.copy()
    work.calc = calc
    opt = FIRE(work, logfile=None)
    converged = bool(opt.run(fmax=float(fmax), steps=int(steps)))
    info = {
        "relax_fmax": float(fmax),
        "relax_steps": int(steps),
        "relax_converged": converged,
        "n_steps": int(getattr(opt, "nsteps", steps)),
    }
    return work, info
