"""Smoke-test pretrained MACE: one structure → total energy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ml.mace import require_mace_deps
from ml.mace.checkpoints import ensure_foundation_checkpoint
from ml.mace.config import foundation_model_id, load_mace_config
from ml.mace.device import resolve_device


@dataclass(frozen=True)
class SmokeResult:
    """Outcome of a single-structure MACE energy smoke test."""

    model_id: str
    device: str
    formula: str
    n_atoms: int
    energy_eV: float
    energy_per_atom_eV: float


def _build_smoke_atoms():
    """Return a tiny periodic crystal (diamond Si) for CPU-friendly smokes."""
    from ase.build import bulk

    return bulk("Si", "diamond", a=5.43)


def run_smoke(
    *,
    device: str | None = None,
    model_id: str | None = None,
    config: dict[str, Any] | None = None,
) -> SmokeResult:
    """Load the pinned foundation calculator and evaluate energy once.

    Downloads the checkpoint on first use (network required). Prefer
    ``device=\"cpu\"`` for CI / laptop smokes.

    Always configures thread limits first — without them, Python 3.14 + torch
    can segfault (macOS: “Python quit unexpectedly”).
    """
    from ml.mace.runtime import (
        coerce_stable_dtype,
        configure_mace_runtime,
        warn_if_unstable_python,
    )

    configure_mace_runtime()
    warn_if_unstable_python()
    require_mace_deps()
    from mace.calculators import mace_mp

    cfg = config if config is not None else load_mace_config()
    resolved_model = model_id or foundation_model_id(cfg)
    resolved_device = resolve_device(device)
    foundation = cfg.get("foundation") or {}
    default_dtype = coerce_stable_dtype(str(foundation.get("default_dtype") or "float32"))
    dispersion = bool(foundation.get("dispersion", False))
    # Prefer a local path so smoke does not depend on mace_mp's urllib SSL path.
    checkpoint = ensure_foundation_checkpoint(resolved_model, config=cfg)

    atoms = _build_smoke_atoms()
    calc = mace_mp(
        model=str(checkpoint),
        device=resolved_device,
        default_dtype=default_dtype,
        dispersion=dispersion,
    )
    atoms.calc = calc
    energy = float(atoms.get_potential_energy())
    n_atoms = len(atoms)
    return SmokeResult(
        model_id=resolved_model,
        device=resolved_device,
        formula=atoms.get_chemical_formula(),
        n_atoms=n_atoms,
        energy_eV=energy,
        energy_per_atom_eV=energy / float(n_atoms),
    )
