"""Load project ``structure_path`` files into ASE ``Atoms`` (Phase 4.2).

Supports:
* JARVIS-style JSON (``lattice_mat`` / ``coords`` / ``elements``)
* pymatgen ``Structure`` JSON (``@class`` / ``lattice`` / ``sites``)
* CIF (experimental COD, etc.)
* VASP / POSCAR (``.vasp``, ``POSCAR*``)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from config.settings import PROJECT_ROOT
from ml.mace.elements import MACE_MPA0_ATOMIC_NUMBERS

# Reason codes written into audit tables (docs/mace_plan.md C5).
REASON_OK = "OK"
REASON_MISSING_PATH = "MISSING_PATH"
REASON_FILE_NOT_FOUND = "FILE_NOT_FOUND"
REASON_UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
REASON_PARSE_ERROR = "PARSE_ERROR"
REASON_EMPTY_STRUCTURE = "EMPTY_STRUCTURE"
REASON_NONFINITE_CELL = "NONFINITE_CELL"
REASON_NONFINITE_POSITIONS = "NONFINITE_POSITIONS"
REASON_UNKNOWN_SPECIES = "UNKNOWN_SPECIES"
REASON_ELEMENT_OUT_OF_COVERAGE = "ELEMENT_OUT_OF_COVERAGE"
REASON_TOO_MANY_ATOMS = "TOO_MANY_ATOMS"

SUPPORTED_SUFFIXES = frozenset({".json", ".cif", ".vasp", ".xyz"})


@dataclass(frozen=True)
class StructureLoadResult:
    """Outcome of resolving and validating one structure file."""

    ok: bool
    reason: str
    path: str | None
    n_atoms: int | None = None
    formula: str | None = None
    atomic_numbers: tuple[int, ...] | None = None
    detail: str | None = None
    atoms: Any | None = None  # ase.Atoms when ok and attach_atoms=True


def resolve_structure_path(
    structure_path: str | Path | None,
    *,
    project_root: Path | None = None,
) -> Path | None:
    """Resolve a repo-relative or absolute structure path. ``None`` if missing."""
    if structure_path is None:
        return None
    text = str(structure_path).strip()
    if not text or text.lower() in {"none", "null", "nan"}:
        return None
    root = project_root or PROJECT_ROOT
    path = Path(text)
    if not path.is_absolute():
        path = root / path
    return path


def _is_jarvis_atoms_dict(payload: dict[str, Any]) -> bool:
    return "lattice_mat" in payload and "coords" in payload and "elements" in payload


def _is_pymatgen_structure_dict(payload: dict[str, Any]) -> bool:
    return "lattice" in payload and "sites" in payload


def _structure_from_jarvis_dict(payload: dict[str, Any]):
    from pymatgen.core import Lattice, Structure

    lattice = Lattice(payload["lattice_mat"])
    elements = list(payload["elements"])
    coords = payload["coords"]
    cartesian = bool(payload.get("cartesian", False))
    return Structure(
        lattice,
        elements,
        coords,
        coords_are_cartesian=cartesian,
    )


def _structure_from_pymatgen_dict(payload: dict[str, Any]):
    from pymatgen.core import Structure

    return Structure.from_dict(payload)


def _atoms_from_pymatgen(structure) -> Any:
    """Convert pymatgen Structure → ASE Atoms.

    Rebuilds a property-free structure first. Stored MP JSONs often have
    ``selective_dynamics: null`` (and similar) which makes
    ``AseAtomsAdaptor.get_atoms`` raise ``TypeError`` on ``~None``.
    """
    from ase import Atoms
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor

    clean = Structure(
        structure.lattice,
        [site.species for site in structure],
        structure.frac_coords,
        coords_are_cartesian=False,
    )
    try:
        return AseAtomsAdaptor.get_atoms(clean)
    except Exception:
        # Last-resort manual conversion (ordered sites only).
        symbols: list[str] = []
        for site in clean:
            specie = site.species
            if hasattr(specie, "symbol"):
                symbols.append(str(specie.symbol))
            else:
                # Disordered: pick majority element.
                el = max(specie.items(), key=lambda kv: kv[1])[0]
                symbols.append(str(el.symbol))
        return Atoms(
            symbols=symbols,
            positions=clean.cart_coords,
            cell=clean.lattice.matrix,
            pbc=True,
        )


def _read_with_ase(path: Path) -> Any:
    from ase.io import read

    atoms = read(str(path))
    if isinstance(atoms, list):
        if not atoms:
            raise ValueError("ASE reader returned an empty list")
        atoms = atoms[0]
    return atoms


def _load_raw_atoms(path: Path) -> Any:
    """Load ASE Atoms from ``path`` (format sniffed by suffix / content)."""
    suffix = path.suffix.lower()
    name = path.name.upper()

    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("JSON structure root must be an object")
        if isinstance(payload.get("atoms"), dict) and _is_jarvis_atoms_dict(payload["atoms"]):
            structure = _structure_from_jarvis_dict(payload["atoms"])
            return _atoms_from_pymatgen(structure)
        if _is_jarvis_atoms_dict(payload):
            return _atoms_from_pymatgen(_structure_from_jarvis_dict(payload))
        if _is_pymatgen_structure_dict(payload):
            return _atoms_from_pymatgen(_structure_from_pymatgen_dict(payload))
        raise ValueError("Unrecognized JSON structure schema")

    if suffix == ".cif" or suffix in {".vasp", ".xyz"} or name.startswith("POSCAR"):
        try:
            from pymatgen.core import Structure

            structure = Structure.from_file(str(path))
            return _atoms_from_pymatgen(structure)
        except Exception:
            return _read_with_ase(path)

    raise ValueError(f"Unsupported structure suffix: {suffix or path.name}")


def _finite_matrix(cell: np.ndarray) -> bool:
    arr = np.asarray(cell, dtype=float)
    if arr.shape != (3, 3):
        return False
    if not np.isfinite(arr).all():
        return False
    # Reject singular / near-zero volume cells.
    try:
        vol = abs(float(np.linalg.det(arr)))
    except Exception:  # noqa: BLE001
        return False
    return vol > 1e-8


def validate_atoms(
    atoms: Any,
    *,
    allowed_atomic_numbers: frozenset[int] | None = MACE_MPA0_ATOMIC_NUMBERS,
    max_atoms: int | None = None,
) -> tuple[bool, str, str | None]:
    """Return ``(ok, reason, detail)`` for an ASE ``Atoms`` object."""
    n_atoms = len(atoms)
    if n_atoms < 1:
        return False, REASON_EMPTY_STRUCTURE, "n_atoms=0"

    if max_atoms is not None and n_atoms > int(max_atoms):
        return False, REASON_TOO_MANY_ATOMS, f"n_atoms={n_atoms} > max_atoms={max_atoms}"

    cell = np.asarray(atoms.get_cell(), dtype=float)
    if not _finite_matrix(cell):
        return False, REASON_NONFINITE_CELL, "cell missing, non-finite, or near-zero volume"

    positions = np.asarray(atoms.get_positions(), dtype=float)
    if positions.shape != (n_atoms, 3) or not np.isfinite(positions).all():
        return False, REASON_NONFINITE_POSITIONS, "positions non-finite or wrong shape"

    numbers = [int(z) for z in atoms.get_atomic_numbers()]
    if any(z < 1 or z > 118 for z in numbers):
        bad = sorted({z for z in numbers if z < 1 or z > 118})
        return False, REASON_UNKNOWN_SPECIES, f"atomic_numbers={bad}"

    if allowed_atomic_numbers is not None:
        oov = sorted({z for z in numbers if z not in allowed_atomic_numbers})
        if oov:
            return False, REASON_ELEMENT_OUT_OF_COVERAGE, f"atomic_numbers={oov}"

    # pbc: prefer periodic; warn via detail only if all False (still OK for load).
    _ = atoms  # kept for future pbc policy
    return True, REASON_OK, None


def load_structure(
    structure_path: str | Path | None,
    *,
    project_root: Path | None = None,
    attach_atoms: bool = True,
    allowed_atomic_numbers: frozenset[int] | None = MACE_MPA0_ATOMIC_NUMBERS,
    max_atoms: int | None = None,
) -> StructureLoadResult:
    """Resolve, parse, and validate one ``structure_path``.

    Failed loads never raise for ordinary bad inputs — they return
    ``ok=False`` with a reason code (audit / training gates).
    """
    resolved = resolve_structure_path(structure_path, project_root=project_root)
    if resolved is None:
        return StructureLoadResult(ok=False, reason=REASON_MISSING_PATH, path=None)

    rel = None
    root = project_root or PROJECT_ROOT
    try:
        rel = str(resolved.relative_to(root))
    except ValueError:
        rel = str(resolved)

    if not resolved.is_file():
        return StructureLoadResult(ok=False, reason=REASON_FILE_NOT_FOUND, path=rel)

    suffix = resolved.suffix.lower()
    name = resolved.name.upper()
    if suffix not in SUPPORTED_SUFFIXES and not name.startswith("POSCAR"):
        return StructureLoadResult(
            ok=False,
            reason=REASON_UNSUPPORTED_FORMAT,
            path=rel,
            detail=f"suffix={suffix or '<none>'}",
        )

    try:
        atoms = _load_raw_atoms(resolved)
    except Exception as exc:  # noqa: BLE001 — quarantine parse failures
        return StructureLoadResult(
            ok=False,
            reason=REASON_PARSE_ERROR,
            path=rel,
            detail=f"{type(exc).__name__}: {exc}",
        )

    ok, reason, detail = validate_atoms(
        atoms,
        allowed_atomic_numbers=allowed_atomic_numbers,
        max_atoms=max_atoms,
    )
    numbers = tuple(int(z) for z in atoms.get_atomic_numbers())
    formula = None
    try:
        formula = atoms.get_chemical_formula()
    except Exception:  # noqa: BLE001
        formula = None

    if not ok:
        return StructureLoadResult(
            ok=False,
            reason=reason,
            path=rel,
            n_atoms=len(atoms),
            formula=formula,
            atomic_numbers=numbers,
            detail=detail,
            atoms=None,
        )

    return StructureLoadResult(
        ok=True,
        reason=REASON_OK,
        path=rel,
        n_atoms=len(atoms),
        formula=formula,
        atomic_numbers=numbers,
        detail=detail,
        atoms=atoms if attach_atoms else None,
    )


def structure_row_dict(
    result: StructureLoadResult,
    *,
    material_id: str | None = None,
    split: str | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    """Flatten a load result for parquet / CSV audit rows."""
    return {
        "material_id": material_id,
        "split": split,
        "source": source,
        "structure_path": result.path,
        "load_ok": bool(result.ok),
        "reason": result.reason,
        "detail": result.detail,
        "n_atoms": result.n_atoms,
        "formula": result.formula,
        "n_elements": len(set(result.atomic_numbers)) if result.atomic_numbers else None,
        "max_atomic_number": max(result.atomic_numbers) if result.atomic_numbers else None,
    }


def is_finite_number(value: float) -> bool:
    """True when ``value`` is a real finite float."""
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False
