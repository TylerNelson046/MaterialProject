"""Standard-state elemental μ for frozen MACE formation energies.

Export labels are DFT E_form vs elemental standard states. Using checkpoint
*isolated-atom* E0s (``elemental_e0_v1``) caused a ~3 eV/atom systematic offset.

Protocol ``elemental_bulk_v1`` (default):
1. Prefer unary structures from the structure pack (relaxed, same ecosystem).
2. Fall back to ASE bulk / dimer recipes for missing elements.
3. Cache ``{Z: μ_eV/atom}`` under the foundation checkpoint directory.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from config.settings import PROJECT_ROOT
from ml.mace.checkpoints import checkpoint_cache_dir
from ml.mace.config import foundation_model_id, load_mace_config
from ml.mace.structures import load_structure

PROTOCOL_ELEMENTAL_BULK_V1 = "elemental_bulk_v1"

# Safe ASE fallbacks only (exotic cells omitted — pack unaries preferred).
_ASE_FALLBACK: dict[str, tuple[str, dict[str, Any]]] = {
    "H": ("molecule", {"name": "H2"}),
    "Li": ("bulk", {"name": "Li", "crystalstructure": "bcc", "a": 3.51}),
    "Be": ("bulk", {"name": "Be", "crystalstructure": "hcp", "a": 2.29, "c": 3.58}),
    "C": ("bulk", {"name": "C", "crystalstructure": "diamond", "a": 3.57}),
    "N": ("molecule", {"name": "N2"}),
    "O": ("molecule", {"name": "O2"}),
    "F": ("molecule", {"name": "F2"}),
    "Na": ("bulk", {"name": "Na", "crystalstructure": "bcc", "a": 4.29}),
    "Mg": ("bulk", {"name": "Mg", "crystalstructure": "hcp", "a": 3.21, "c": 5.21}),
    "Al": ("bulk", {"name": "Al", "crystalstructure": "fcc", "a": 4.05}),
    "Si": ("bulk", {"name": "Si", "crystalstructure": "diamond", "a": 5.43}),
    "Cl": ("molecule", {"name": "Cl2"}),
    "K": ("bulk", {"name": "K", "crystalstructure": "bcc", "a": 5.33}),
    "Ca": ("bulk", {"name": "Ca", "crystalstructure": "fcc", "a": 5.59}),
    "Sc": ("bulk", {"name": "Sc", "crystalstructure": "hcp", "a": 3.31, "c": 5.27}),
    "Ti": ("bulk", {"name": "Ti", "crystalstructure": "hcp", "a": 2.95, "c": 4.68}),
    "V": ("bulk", {"name": "V", "crystalstructure": "bcc", "a": 3.03}),
    "Cr": ("bulk", {"name": "Cr", "crystalstructure": "bcc", "a": 2.88}),
    "Mn": ("bulk", {"name": "Mn", "crystalstructure": "bcc", "a": 2.87}),
    "Fe": ("bulk", {"name": "Fe", "crystalstructure": "bcc", "a": 2.87}),
    "Co": ("bulk", {"name": "Co", "crystalstructure": "hcp", "a": 2.51, "c": 4.07}),
    "Ni": ("bulk", {"name": "Ni", "crystalstructure": "fcc", "a": 3.52}),
    "Cu": ("bulk", {"name": "Cu", "crystalstructure": "fcc", "a": 3.61}),
    "Zn": ("bulk", {"name": "Zn", "crystalstructure": "hcp", "a": 2.66, "c": 4.95}),
    "Ga": (
        "bulk",
        {
            "name": "Ga",
            "crystalstructure": "orthorhombic",
            "a": 4.52,
            "b": 4.52,
            "c": 7.66,
        },
    ),
    "Ge": ("bulk", {"name": "Ge", "crystalstructure": "diamond", "a": 5.66}),
    "As": ("bulk", {"name": "As", "crystalstructure": "rhombohedral", "a": 4.13, "alpha": 54.1}),
    "Br": ("molecule", {"name": "Br2"}),
    "Rb": ("bulk", {"name": "Rb", "crystalstructure": "bcc", "a": 5.59}),
    "Sr": ("bulk", {"name": "Sr", "crystalstructure": "fcc", "a": 6.08}),
    "Y": ("bulk", {"name": "Y", "crystalstructure": "hcp", "a": 3.65, "c": 5.73}),
    "Zr": ("bulk", {"name": "Zr", "crystalstructure": "hcp", "a": 3.23, "c": 5.15}),
    "Nb": ("bulk", {"name": "Nb", "crystalstructure": "bcc", "a": 3.30}),
    "Mo": ("bulk", {"name": "Mo", "crystalstructure": "bcc", "a": 3.15}),
    "Ru": ("bulk", {"name": "Ru", "crystalstructure": "hcp", "a": 2.71, "c": 4.28}),
    "Rh": ("bulk", {"name": "Rh", "crystalstructure": "fcc", "a": 3.80}),
    "Pd": ("bulk", {"name": "Pd", "crystalstructure": "fcc", "a": 3.89}),
    "Ag": ("bulk", {"name": "Ag", "crystalstructure": "fcc", "a": 4.09}),
    "Cd": ("bulk", {"name": "Cd", "crystalstructure": "hcp", "a": 2.98, "c": 5.62}),
    "In": ("bulk", {"name": "In", "crystalstructure": "tetragonal", "a": 3.25, "c": 4.95}),
    "Sn": ("bulk", {"name": "Sn", "crystalstructure": "diamond", "a": 6.49}),
    "Sb": ("bulk", {"name": "Sb", "crystalstructure": "rhombohedral", "a": 4.51, "alpha": 57.1}),
    "I": ("molecule", {"name": "I2"}),
    "Cs": ("bulk", {"name": "Cs", "crystalstructure": "bcc", "a": 6.05}),
    "Ba": ("bulk", {"name": "Ba", "crystalstructure": "bcc", "a": 5.02}),
    "Hf": ("bulk", {"name": "Hf", "crystalstructure": "hcp", "a": 3.20, "c": 5.06}),
    "Ta": ("bulk", {"name": "Ta", "crystalstructure": "bcc", "a": 3.30}),
    "W": ("bulk", {"name": "W", "crystalstructure": "bcc", "a": 3.16}),
    "Re": ("bulk", {"name": "Re", "crystalstructure": "hcp", "a": 2.76, "c": 4.46}),
    "Os": ("bulk", {"name": "Os", "crystalstructure": "hcp", "a": 2.74, "c": 4.32}),
    "Ir": ("bulk", {"name": "Ir", "crystalstructure": "fcc", "a": 3.84}),
    "Pt": ("bulk", {"name": "Pt", "crystalstructure": "fcc", "a": 3.92}),
    "Au": ("bulk", {"name": "Au", "crystalstructure": "fcc", "a": 4.08}),
    "Pb": ("bulk", {"name": "Pb", "crystalstructure": "fcc", "a": 4.95}),
    "Bi": ("bulk", {"name": "Bi", "crystalstructure": "rhombohedral", "a": 4.75, "alpha": 57.2}),
}


def elemental_bulk_cache_path(
    model_id: str | None = None,
    *,
    config: dict[str, Any] | None = None,
) -> Path:
    cfg = config if config is not None else load_mace_config()
    mid = model_id or foundation_model_id(cfg)
    return checkpoint_cache_dir(cfg) / f"elemental_bulk_refs_{mid}.json"


def _ase_element_atoms(symbol: str):
    from ase.build import bulk, molecule

    kind, kwargs = _ASE_FALLBACK[symbol]
    if kind == "molecule":
        atoms = molecule(kwargs["name"])
        atoms.center(vacuum=6.0)
        return atoms
    name = kwargs["name"]
    crystal = kwargs["crystalstructure"]
    params = {k: v for k, v in kwargs.items() if k not in {"name", "crystalstructure"}}
    return bulk(name, crystal, **params)


def _mus_from_pack_unaries(
    calc: Any,
    *,
    config: dict[str, Any],
    max_per_element: int = 3,
    progress: Callable[[str], None] | None = None,
) -> dict[int, float]:
    """Lowest MACE E/atom among unary pack structures, per atomic number."""
    from ase.data import atomic_numbers, chemical_symbols

    try:
        from ml.mace.pack import load_mace_pack_frame

        pack = load_mace_pack_frame(
            "formation_energy_per_atom", train_ok_only=True, config=config
        )
    except FileNotFoundError:
        return {}

    if "nelements" not in pack.columns:
        return {}
    unaries = pack.loc[pack["nelements"].fillna(0).astype(int) == 1].copy()
    if unaries.empty:
        return {}

    # Map formula → Z (single-element formulas).
    best: dict[int, float] = {}
    counts: dict[int, int] = {}
    for rec in unaries.itertuples(index=False):
        formula = str(getattr(rec, "formula_pretty", "") or "")
        # Normalize: Si2 → Si, O2 → O
        sym = None
        for s in chemical_symbols[1:]:
            if formula == s or (
                formula.startswith(s) and formula[len(s) :].isdigit()
            ):
                sym = s
                break
        if sym is None:
            continue
        z = int(atomic_numbers[sym])
        if counts.get(z, 0) >= max_per_element and z in best:
            continue
        loaded = load_structure(getattr(rec, "structure_path", None), attach_atoms=True)
        if not loaded.ok or loaded.atoms is None:
            continue
        atoms = loaded.atoms
        # Confirm unary.
        zs = set(int(x) for x in atoms.get_atomic_numbers())
        if zs != {z}:
            continue
        try:
            atoms.calc = calc
            e_pa = float(atoms.get_potential_energy()) / float(len(atoms))
        except Exception:  # noqa: BLE001
            continue
        counts[z] = counts.get(z, 0) + 1
        if z not in best or e_pa < best[z]:
            best[z] = e_pa
            if progress:
                progress(f"pack unary {sym}: μ={e_pa:.6f} ({rec.material_id})")
    return best


def _mus_from_ase_fallback(
    calc: Any,
    *,
    missing_symbols: list[str],
    progress: Callable[[str], None] | None = None,
) -> dict[int, float]:
    from ase.data import atomic_numbers

    out: dict[int, float] = {}
    for symbol in missing_symbols:
        if symbol not in _ASE_FALLBACK:
            continue
        z = int(atomic_numbers[symbol])
        try:
            atoms = _ase_element_atoms(symbol)
            atoms.calc = calc
            e_pa = float(atoms.get_potential_energy()) / float(len(atoms))
            out[z] = e_pa
            if progress:
                progress(f"ASE fallback {symbol}: μ={e_pa:.6f}")
        except Exception as exc:  # noqa: BLE001
            if progress:
                progress(f"ASE fallback {symbol} FAILED: {type(exc).__name__}: {exc}")
    return out


def compute_elemental_bulk_refs(
    *,
    calc: Any,
    config: dict[str, Any] | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[int, float]:
    """Build ``{Z: μ}`` from pack unaries + ASE fallbacks."""
    from ase.data import chemical_symbols

    cfg = config if config is not None else load_mace_config()
    mus = _mus_from_pack_unaries(calc, config=cfg, progress=progress)
    have = {chemical_symbols[z] for z in mus}
    missing = [s for s in _ASE_FALLBACK if s not in have]
    mus.update(_mus_from_ase_fallback(calc, missing_symbols=missing, progress=progress))
    return mus


def ensure_elemental_bulk_refs(
    model_id: str | None = None,
    *,
    calc: Any | None = None,
    config: dict[str, Any] | None = None,
    force: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[int, float]:
    """Load or compute cached bulk elemental μ table."""
    cfg = config if config is not None else load_mace_config()
    mid = model_id or foundation_model_id(cfg)
    path = elemental_bulk_cache_path(mid, config=cfg)
    if path.is_file() and not force:
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw = payload.get("mu_by_atomic_number") or {}
        return {int(k): float(v) for k, v in raw.items()}

    if calc is None:
        from ml.mace.calculator import load_foundation_calculator

        calc, mid, _, _ = load_foundation_calculator(
            model_id=mid, device="cpu", config=cfg
        )

    mus = compute_elemental_bulk_refs(calc=calc, config=cfg, progress=progress)
    if len(mus) < 20:
        raise RuntimeError(
            f"elemental_bulk_v1 produced only {len(mus)} elemental refs "
            "(need pack unaries and/or ASE fallbacks)."
        )
    try:
        rel = str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        rel = str(path)
    payload = {
        "protocol": PROTOCOL_ELEMENTAL_BULK_V1,
        "model_id": mid,
        "unit": "eV/atom",
        "formula": "E_form/atom = (E_tot - sum_i μ[Z_i]) / n_atoms",
        "note": (
            "μ_Z from frozen MACE on unary pack structures when available, "
            "else ASE bulk/dimer fallbacks. Replaces isolated-atom checkpoint E0s."
        ),
        "n_elements": len(mus),
        "mu_by_atomic_number": {str(k): v for k, v in sorted(mus.items())},
        "cache_path": rel,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return mus


def reference_payload_bulk(
    mus: dict[int, float],
    *,
    model_id: str,
) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL_ELEMENTAL_BULK_V1,
        "model_id": model_id,
        "unit": "eV/atom",
        "formula": "E_form/atom = (E_tot - sum_i μ[Z_i]) / n_atoms",
        "e0_by_atomic_number": {str(k): v for k, v in sorted(mus.items())},
        "mu_by_atomic_number": {str(k): v for k, v in sorted(mus.items())},
    }
