"""Normalize JARVIS-DFT raw documents into the unified schema."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterator

from config.constants import DataSource, PropertyName
from config.settings import Settings, get_settings
from normalization.physical_ranges import (
    sanitize_band_gap,
    sanitize_density,
    sanitize_ehull,
    sanitize_elastic,
    sanitize_formation_energy,
    sanitize_poisson,
    sanitize_volume,
)
from normalization.schema import (
    CompositionInfo,
    MaterialRecord,
    SourceProperty,
    SourceRecord,
    StructureRef,
    SymmetryInfo,
    make_material_id,
    make_source_property,
)

SOURCE = DataSource.JARVIS


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip().lower() in {"", "na", "nan", "none", "null"}:
        return True
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return True
    return False


def _num(value: Any) -> float | None:
    if _is_missing(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    number = _num(value)
    if number is None:
        return None
    return int(number)


def _bool_from_gap(gap: float | None) -> bool | None:
    if gap is None:
        return None
    return gap <= 0.0


def _composition_from_jarvis(doc: dict[str, Any]) -> CompositionInfo:
    """Build composition from formula (preferred) or atoms/elements lists.

    The bulk JARVIS JSONL omits ``atoms``/``elements`` for size; ``formula`` is
    always present and is the reliable source.
    """
    formula = str(doc.get("formula") or doc.get("jid") or "unknown")
    composition: dict[str, float] = {}
    uniq: list[str] = []

    # 1) Parse formula via pymatgen when available.
    if formula and formula != "unknown":
        try:
            from pymatgen.core import Composition

            parsed = Composition(formula)
            composition = {str(el): float(amt) for el, amt in parsed.as_dict().items()}
            uniq = sorted(composition.keys())
            formula = parsed.reduced_formula
        except Exception:  # noqa: BLE001 - fall through to element lists / regex
            composition = {}
            uniq = []

    # 2) Fall back to explicit element lists (tests / structure-rich docs).
    if not composition:
        elements_raw = doc.get("elements")
        elements: list[str] = []
        if isinstance(elements_raw, list):
            elements = [str(e) for e in elements_raw]
        elif isinstance(doc.get("atoms"), dict):
            atoms_elements = doc["atoms"].get("elements") or []
            elements = [str(e) for e in atoms_elements]
        if elements:
            for el in elements:
                composition[el] = composition.get(el, 0.0) + 1.0
            uniq = list(dict.fromkeys(elements))

    # 3) Last-resort regex on the formula string.
    if not composition and formula and formula != "unknown":
        import re

        for el, count in re.findall(r"([A-Z][a-z]*)(\d*)", formula):
            composition[el] = composition.get(el, 0.0) + float(count or 1)
        uniq = sorted(composition.keys())

    search = str(doc.get("search") or "")
    chemsys = "-".join(sorted(uniq)) if uniq else (search.strip("-") or None)

    return CompositionInfo(
        formula_pretty=str(doc.get("formula") or formula),
        reduced_formula=formula if composition else str(doc.get("formula") or formula),
        chemsys=chemsys,
        elements=tuple(sorted(uniq)) if uniq else tuple(uniq),
        nelements=len(uniq) or None,
        composition=composition,
    )


def _symmetry_from_jarvis(doc: dict[str, Any]) -> SymmetryInfo:
    return SymmetryInfo(
        crystal_system=str(doc["crys"]).title() if not _is_missing(doc.get("crys")) else None,
        space_group_symbol=None if _is_missing(doc.get("spg_symbol")) else str(doc.get("spg_symbol")),
        space_group_number=_int(doc.get("spg_number") or doc.get("spg")),
        point_group=None,
        number_of_species=None,
    )


def _volume_from_jarvis_atoms(atoms: dict[str, Any] | None) -> float | None:
    """Cell volume (Å³) from JARVIS atoms blob (prefer lattice_mat determinant)."""
    if not isinstance(atoms, dict):
        return None
    lattice_mat = atoms.get("lattice_mat")
    if isinstance(lattice_mat, list) and len(lattice_mat) == 3:
        try:
            import numpy as np

            matrix = np.asarray(lattice_mat, dtype=float)
            if matrix.shape == (3, 3):
                volume = abs(float(np.linalg.det(matrix)))
                return volume if volume > 0 else None
        except Exception:  # noqa: BLE001
            pass
    abc = atoms.get("abc")
    angles = atoms.get("angles")
    if isinstance(abc, list) and len(abc) >= 3 and isinstance(angles, list) and len(angles) >= 3:
        try:
            from pymatgen.core import Lattice

            return float(Lattice.from_parameters(
                float(abc[0]), float(abc[1]), float(abc[2]),
                float(angles[0]), float(angles[1]), float(angles[2]),
            ).volume)
        except Exception:  # noqa: BLE001
            return None
    return None


def _load_jarvis_atoms_from_path(rel_or_abs: str | None, *, settings: Settings) -> dict[str, Any] | None:
    if not rel_or_abs:
        return None
    path = Path(rel_or_abs)
    if not path.is_absolute():
        path = settings.project_root / path
    if not path.exists() or path.stat().st_size < 40:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(payload, dict) and isinstance(payload.get("atoms"), dict):
        return payload["atoms"]
    if isinstance(payload, dict) and ("lattice_mat" in payload or "abc" in payload):
        return payload
    return None


def _structure_ref_from_jarvis(doc: dict[str, Any], *, settings: Settings) -> StructureRef:
    source_id = str(doc.get("jid") or "")
    rel = doc.get("structure_path")
    if not rel and source_id:
        path = settings.computational_structures_dir() / f"{source_id.replace('/', '_')}.json"
        if path.exists():
            rel = str(path.relative_to(settings.project_root))

    atoms = doc.get("atoms") if isinstance(doc.get("atoms"), dict) else None
    if atoms is None:
        atoms = _load_jarvis_atoms_from_path(str(rel) if rel else None, settings=settings)
    abc = atoms.get("abc") if isinstance(atoms, dict) else None
    angles = atoms.get("angles") if isinstance(atoms, dict) else None
    volume = _volume_from_jarvis_atoms(atoms)

    return StructureRef(
        path=str(rel) if rel else None,
        format="json" if rel else None,
        nsites=_int(doc.get("nat")),
        volume=volume,
        density=_num(doc.get("density")),
        lattice_a=float(abc[0]) if isinstance(abc, list) and len(abc) >= 1 else None,
        lattice_b=float(abc[1]) if isinstance(abc, list) and len(abc) >= 2 else None,
        lattice_c=float(abc[2]) if isinstance(abc, list) and len(abc) >= 3 else None,
        lattice_alpha=float(angles[0]) if isinstance(angles, list) and len(angles) >= 1 else None,
        lattice_beta=float(angles[1]) if isinstance(angles, list) and len(angles) >= 2 else None,
        lattice_gamma=float(angles[2]) if isinstance(angles, list) and len(angles) >= 3 else None,
    )


def _properties_from_jarvis(
    doc: dict[str, Any],
    source_id: str,
    *,
    volume: float | None = None,
) -> tuple[SourceProperty, ...]:
    band_gap = _num(doc.get("optb88vdw_bandgap"))
    ehull = _num(doc.get("ehull"))
    mag = _num(doc.get("magmom_outcar"))
    if mag is None:
        mag = _num(doc.get("magmom_oszicar"))

    band_gap = sanitize_band_gap(band_gap)
    ehull = sanitize_ehull(ehull)
    mapping: list[tuple[PropertyName, Any, str]] = [
        (PropertyName.FORMATION_ENERGY_PER_ATOM, sanitize_formation_energy(doc.get("formation_energy_peratom")), "jarvis-optb88vdw"),
        (PropertyName.ENERGY_ABOVE_HULL, ehull, "jarvis-optb88vdw"),
        (PropertyName.ENERGY_PER_ATOM, _num(doc.get("optb88vdw_total_energy")), "jarvis-optb88vdw"),
        (PropertyName.BAND_GAP, band_gap, "jarvis-optb88vdw"),
        (PropertyName.IS_METAL, _bool_from_gap(band_gap), "jarvis-optb88vdw"),
        (PropertyName.IS_STABLE, (ehull is not None and ehull <= 1e-6), "jarvis-optb88vdw"),
        (PropertyName.DENSITY, sanitize_density(doc.get("density")), "jarvis-optb88vdw"),
        (PropertyName.VOLUME, sanitize_volume(volume), "jarvis-structure"),
        (PropertyName.NSITES, _int(doc.get("nat")), "jarvis-optb88vdw"),
        (PropertyName.TOTAL_MAGNETIZATION, mag, "jarvis-optb88vdw"),
        (PropertyName.IS_MAGNETIC, (abs(mag) > 0.1) if mag is not None else None, "jarvis-optb88vdw"),
        (PropertyName.BULK_MODULUS_VRH, sanitize_elastic(doc.get("bulk_modulus_kv")), "jarvis-optb88vdw"),
        (PropertyName.SHEAR_MODULUS_VRH, sanitize_elastic(doc.get("shear_modulus_gv")), "jarvis-optb88vdw"),
        (PropertyName.HOMOGENEOUS_POISSON, sanitize_poisson(doc.get("poisson")), "jarvis-optb88vdw"),
        (PropertyName.DIELECTRIC_TOTAL, _num(doc.get("dfpt_piezo_max_dielectric")), "jarvis-dfpt"),
        (PropertyName.DIELECTRIC_ELECTRONIC, _num(doc.get("dfpt_piezo_max_dielectric_electronic")), "jarvis-dfpt"),
        (PropertyName.DIELECTRIC_IONIC, _num(doc.get("dfpt_piezo_max_dielectric_ionic")), "jarvis-dfpt"),
        (PropertyName.MBJ_BANDGAP, sanitize_band_gap(doc.get("mbj_bandgap")), "jarvis-mbj"),
        (PropertyName.AVG_ELEC_MASS, _num(doc.get("avg_elec_mass")), "jarvis-optb88vdw"),
        (PropertyName.AVG_HOLE_MASS, _num(doc.get("avg_hole_mass")), "jarvis-optb88vdw"),
        (PropertyName.SLME, _num(doc.get("slme")), "jarvis-optb88vdw"),
        (PropertyName.TC_SUPERCON, _num(doc.get("Tc_supercon")), "jarvis-supercon"),
        (PropertyName.EXFOLIATION_ENERGY, _num(doc.get("exfoliation_energy")), "jarvis-optb88vdw"),
        (PropertyName.THEORETICAL, True, "jarvis-optb88vdw"),
    ]

    props: list[SourceProperty] = []
    for name, value, method in mapping:
        prop = make_source_property(name, value, SOURCE, source_id, method=method)
        if prop is not None:
            props.append(prop)
    return tuple(props)


def normalize_jarvis_document(
    doc: dict[str, Any],
    *,
    index: int,
    settings: Settings | None = None,
) -> MaterialRecord:
    settings = settings or get_settings()
    source_id = str(doc.get("jid") or "").strip()
    if not source_id:
        raise ValueError("JARVIS document missing jid")

    structure = _structure_ref_from_jarvis(doc, settings=settings)
    material_id = make_material_id(index)
    source_record = SourceRecord(
        source=SOURCE,
        source_id=source_id,
        composition=_composition_from_jarvis(doc),
        properties=_properties_from_jarvis(doc, source_id, volume=structure.volume),
        symmetry=_symmetry_from_jarvis(doc),
        structure=structure,
        original_metadata={
            "jid": source_id,
            "reference": doc.get("reference"),
            "func": doc.get("func"),
            "hse_gap": doc.get("hse_gap"),
            "spillage": doc.get("spillage"),
            "dimensionality": doc.get("dimensionality"),
            "typ": doc.get("typ"),
            "icsd": doc.get("icsd"),
            "epsx": doc.get("epsx"),
            "epsy": doc.get("epsy"),
            "epsz": doc.get("epsz"),
        },
        material_id=material_id,
    )
    return MaterialRecord(material_id=material_id, sources=(source_record,))


def iter_raw_jarvis_documents(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}") from exc


def iter_normalized_jarvis(
    path: Path,
    *,
    start_index: int = 1,
    settings: Settings | None = None,
) -> Iterator[MaterialRecord]:
    settings = settings or get_settings()
    for offset, doc in enumerate(iter_raw_jarvis_documents(path)):
        yield normalize_jarvis_document(doc, index=start_index + offset, settings=settings)


def iter_load_normalized_jarvis(
    settings: Settings | None = None,
    *,
    start_index: int = 1,
) -> Iterator[MaterialRecord]:
    settings = settings or get_settings()
    path = settings.raw_source_dir(DataSource.JARVIS) / "materials.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Raw JARVIS data not found at {path}. Run scripts/download_data.py --source jarvis."
        )
    yield from iter_normalized_jarvis(path, start_index=start_index, settings=settings)
