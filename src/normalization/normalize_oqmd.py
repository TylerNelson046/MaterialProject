"""Normalize OQMD raw documents into the unified schema."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterator

from config.constants import DataSource, PropertyName
from config.settings import Settings, get_settings
from database.fingerprints import space_group_number_from_symbol
from normalization.physical_ranges import (
    sanitize_band_gap,
    sanitize_density,
    sanitize_ehull,
    sanitize_formation_energy,
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

# amu/Å³ → g/cm³
_AMU_PER_A3_TO_GCM3 = 1.66053906660

# International Tables ranges → crystal system (avoids pymatgen per row).
_CRYSTAL_SYSTEM_BY_SG: tuple[tuple[int, int, str], ...] = (
    (1, 2, "Triclinic"),
    (3, 15, "Monoclinic"),
    (16, 74, "Orthorhombic"),
    (75, 142, "Tetragonal"),
    (143, 167, "Trigonal"),
    (168, 194, "Hexagonal"),
    (195, 230, "Cubic"),
)

SOURCE = DataSource.OQMD
_ELEMENT_TOKEN = re.compile(r"([A-Z][a-z]*)(\d*)")


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


def _parse_composition_string(text: str) -> dict[str, float]:
    """Parse strings like 'Lu1' / 'Ti2O3' into element amounts."""
    out: dict[str, float] = {}
    for element, count in _ELEMENT_TOKEN.findall(text or ""):
        out[element] = out.get(element, 0.0) + float(count or 1)
    return out


def _composition_from_oqmd(doc: dict[str, Any]) -> CompositionInfo:
    formula = str(doc.get("name") or doc.get("composition") or "unknown")
    amounts = _parse_composition_string(str(doc.get("composition") or formula))
    if not amounts:
        # Fallback via pymatgen when available.
        try:
            from pymatgen.core import Composition

            amounts = {
                str(el): float(amt)
                for el, amt in Composition(formula).as_dict().items()
            }
        except Exception:  # noqa: BLE001
            amounts = {}

    elements = tuple(sorted(amounts.keys()))
    chemsys = "-".join(elements) if elements else None
    return CompositionInfo(
        formula_pretty=formula,
        reduced_formula=formula,
        chemsys=chemsys,
        elements=elements,
        nelements=_int(doc.get("ntypes")) or (len(elements) or None),
        composition=amounts,
    )


def _density_from_composition_volume(
    amounts: dict[str, float],
    volume: float | None,
) -> float | None:
    """Compute density (g/cm³) from composition mass and cell volume (Å³)."""
    if not amounts or volume is None or volume <= 0:
        return None
    mass_amu = 0.0
    try:
        from pymatgen.core import Element

        for element, amount in amounts.items():
            mass_amu += float(Element(element).atomic_mass) * float(amount)
    except Exception:  # noqa: BLE001
        try:
            from pymatgen.core import Composition

            mass_amu = float(Composition(amounts).weight)
        except Exception:  # noqa: BLE001
            return None
    if mass_amu <= 0:
        return None
    return mass_amu / float(volume) * _AMU_PER_A3_TO_GCM3


def _crystal_system_from_number(number: int | None) -> str | None:
    if number is None:
        return None
    try:
        n = int(number)
    except (TypeError, ValueError):
        return None
    for lo, hi, name in _CRYSTAL_SYSTEM_BY_SG:
        if lo <= n <= hi:
            return name
    return None


def _symmetry_from_oqmd(doc: dict[str, Any]) -> SymmetryInfo:
    symbol = None if _is_missing(doc.get("spacegroup")) else str(doc.get("spacegroup"))
    number = space_group_number_from_symbol(symbol) if symbol else None
    return SymmetryInfo(
        crystal_system=_crystal_system_from_number(number),
        space_group_symbol=symbol,
        space_group_number=number,
        point_group=None,
        number_of_species=_int(doc.get("ntypes")),
    )


def _structure_ref_from_oqmd(
    doc: dict[str, Any],
    *,
    settings: Settings | None = None,
    source_id: str | None = None,
    density: float | None = None,
) -> StructureRef:
    # Bulk summary pull omits sites; structure files may exist from a later download.
    structure_path: str | None = None
    fmt: str | None = None
    if settings is not None and source_id:
        candidate = (
            settings.computational_structures_dir()
            / f"{source_id.replace('/', '_')}.json"
        )
        if candidate.exists() and candidate.stat().st_size > 80:
            structure_path = str(candidate.relative_to(settings.project_root))
            fmt = "json"
    return StructureRef(
        path=structure_path,
        format=fmt,
        nsites=_int(doc.get("natoms")),
        volume=_num(doc.get("volume")),
        density=density,
    )


def _properties_from_oqmd(
    doc: dict[str, Any],
    source_id: str,
    *,
    density: float | None = None,
) -> tuple[SourceProperty, ...]:
    band_gap = sanitize_band_gap(doc.get("band_gap"))
    stability = sanitize_ehull(doc.get("stability"))
    mapping: list[tuple[PropertyName, Any]] = [
        (
            PropertyName.FORMATION_ENERGY_PER_ATOM,
            sanitize_formation_energy(doc.get("delta_e")),
        ),
        (PropertyName.ENERGY_ABOVE_HULL, stability),
        (PropertyName.BAND_GAP, band_gap),
        (PropertyName.VOLUME, sanitize_volume(doc.get("volume"))),
        (PropertyName.DENSITY, density),
        (PropertyName.NSITES, _int(doc.get("natoms"))),
        (PropertyName.IS_STABLE, (stability is not None and stability <= 1e-6)),
        (PropertyName.IS_METAL, (band_gap is not None and band_gap <= 0.0)),
        (PropertyName.THEORETICAL, True),
    ]
    props: list[SourceProperty] = []
    for name, value in mapping:
        prop = make_source_property(name, value, SOURCE, source_id, method="oqmd-dft")
        if prop is not None:
            props.append(prop)
    return tuple(props)


def normalize_oqmd_document(
    doc: dict[str, Any],
    *,
    index: int,
    settings: Settings | None = None,
) -> MaterialRecord:
    settings = settings or get_settings()
    source_id = str(doc.get("source_id") or "").strip()
    if not source_id:
        entry_id = doc.get("entry_id")
        if entry_id is None:
            raise ValueError("OQMD document missing entry_id/source_id")
        source_id = f"oqmd-{entry_id}"

    composition = _composition_from_oqmd(doc)
    density = sanitize_density(
        _density_from_composition_volume(
            composition.composition,
            _num(doc.get("volume")),
        )
    )
    material_id = make_material_id(index)
    source_record = SourceRecord(
        source=SOURCE,
        source_id=source_id,
        composition=composition,
        properties=_properties_from_oqmd(doc, source_id, density=density),
        symmetry=_symmetry_from_oqmd(doc),
        structure=_structure_ref_from_oqmd(
            doc, settings=settings, source_id=source_id, density=density
        ),
        original_metadata={
            "entry_id": doc.get("entry_id"),
            "formationenergy_id": doc.get("formationenergy_id"),
            "calculation_id": doc.get("calculation_id"),
            "icsd_id": doc.get("icsd_id"),
            "prototype": doc.get("prototype"),
            "composition_generic": doc.get("composition_generic"),
            "fit": doc.get("fit"),
            "calculation_label": doc.get("calculation_label"),
            "duplicate_entry_id": doc.get("duplicate_entry_id"),
            "unit_cell": doc.get("unit_cell"),
        },
        material_id=material_id,
    )
    return MaterialRecord(material_id=material_id, sources=(source_record,))


def iter_raw_oqmd_documents(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}") from exc


def iter_normalized_oqmd(
    path: Path,
    *,
    start_index: int = 1,
    settings: Settings | None = None,
) -> Iterator[MaterialRecord]:
    settings = settings or get_settings()
    for offset, doc in enumerate(iter_raw_oqmd_documents(path)):
        yield normalize_oqmd_document(doc, index=start_index + offset, settings=settings)


def iter_load_normalized_oqmd(
    settings: Settings | None = None,
    *,
    start_index: int = 1,
) -> Iterator[MaterialRecord]:
    settings = settings or get_settings()
    path = settings.raw_source_dir(DataSource.OQMD) / "materials.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Raw OQMD data not found at {path}. Run scripts/download_data.py --source oqmd."
        )
    yield from iter_normalized_oqmd(path, start_index=start_index, settings=settings)
