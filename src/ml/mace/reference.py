"""Elemental reference energies for formation-energy conversion (Phase 4.4)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ml.mace.checkpoints import ensure_foundation_checkpoint
from ml.mace.config import foundation_model_id, load_mace_config

PROTOCOL_ELEMENTAL_E0_V1 = "elemental_e0_v1"  # isolated-atom checkpoint E0s (legacy)
PROTOCOL_ELEMENTAL_BULK_V1 = "elemental_bulk_v1"  # standard-state μ (default)


def load_elemental_e0s(
    model_id: str | None = None,
    *,
    config: dict[str, Any] | None = None,
) -> dict[int, float]:
    """Return ``{atomic_number: E0_eV}`` from the foundation checkpoint.

    These are the MACE *isolated-atom* energies stored in the model.
    Prefer ``load_formation_references`` / ``elemental_bulk_v1`` when comparing
    to DFT formation-energy labels.
    """
    from ml.mace import require_mace_deps
    from ml.mace.runtime import configure_mace_runtime

    configure_mace_runtime()
    require_mace_deps()
    import torch

    cfg = config if config is not None else load_mace_config()
    resolved = model_id or foundation_model_id(cfg)
    checkpoint = ensure_foundation_checkpoint(resolved, config=cfg)
    model = torch.load(checkpoint, map_location="cpu", weights_only=False)
    zs = [int(z) for z in model.atomic_numbers.detach().cpu().tolist()]
    energies = (
        model.atomic_energies_fn.atomic_energies.detach().cpu().to(dtype=torch.float64)
    )
    flat = energies.reshape(-1).tolist()
    if len(zs) != len(flat):
        raise ValueError(
            f"atomic_numbers ({len(zs)}) != atomic_energies ({len(flat)}) in {checkpoint}"
        )
    return {z: float(e) for z, e in zip(zs, flat, strict=True)}


def load_formation_references(
    model_id: str | None = None,
    *,
    protocol: str | None = None,
    calc: Any | None = None,
    config: dict[str, Any] | None = None,
    force_recompute: bool = False,
) -> tuple[dict[int, float], str]:
    """Return ``({Z: μ_eV}, protocol)`` for formation-energy conversion.

    Default protocol is ``elemental_bulk_v1`` (standard-state elemental μ).
    """
    cfg = config if config is not None else load_mace_config()
    resolved_protocol = str(
        protocol
        or (cfg.get("energy") or {}).get("reference_protocol")
        or PROTOCOL_ELEMENTAL_BULK_V1
    )
    mid = model_id or foundation_model_id(cfg)
    if resolved_protocol == PROTOCOL_ELEMENTAL_BULK_V1:
        from ml.mace.elemental_refs import ensure_elemental_bulk_refs

        mus = ensure_elemental_bulk_refs(
            mid, calc=calc, config=cfg, force=force_recompute
        )
        return mus, resolved_protocol
    if resolved_protocol == PROTOCOL_ELEMENTAL_E0_V1:
        return load_elemental_e0s(mid, config=cfg), resolved_protocol
    raise ValueError(
        f"Unsupported reference_protocol {resolved_protocol!r}; "
        f"expected {PROTOCOL_ELEMENTAL_BULK_V1!r} or {PROTOCOL_ELEMENTAL_E0_V1!r}"
    )


def reference_table_payload(
    e0s: dict[int, float],
    *,
    protocol: str = PROTOCOL_ELEMENTAL_BULK_V1,
    model_id: str,
) -> dict[str, Any]:
    """JSON-serializable reference table for ``reference_energies.json``."""
    return {
        "protocol": protocol,
        "model_id": model_id,
        "unit": "eV/atom_reference",
        "formula": "E_form/atom = (E_tot - sum_i μ[Z_i]) / n_atoms",
        "e0_by_atomic_number": {str(k): v for k, v in sorted(e0s.items())},
        "mu_by_atomic_number": {str(k): v for k, v in sorted(e0s.items())},
    }


def write_reference_energies(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def formation_energy_per_atom(
    energy_total_eV: float,
    atomic_numbers: list[int] | tuple[int, ...] | np.ndarray,
    e0s: dict[int, float],
) -> float:
    """Convert total cell energy to formation energy per atom using E0 table."""
    zs = [int(z) for z in atomic_numbers]
    if not zs:
        raise ValueError("atomic_numbers is empty")
    missing = sorted({z for z in zs if z not in e0s})
    if missing:
        raise KeyError(f"Missing E0 for atomic numbers: {missing}")
    ref = float(sum(e0s[z] for z in zs))
    return (float(energy_total_eV) - ref) / float(len(zs))
