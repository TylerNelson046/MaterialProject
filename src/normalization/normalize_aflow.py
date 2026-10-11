"""Normalize AFLOW raw documents into the unified schema."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterator

from config.constants import DataSource, PropertyName
from config.settings import Settings, get_settings
from normalization.physical_ranges import (
    sanitize_band_gap,
    sanitize_density,
    sanitize_elastic,
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

SOURCE = DataSource.AFLOW
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


def _canonical_source_id(auid: str) -> str:
    text = str(auid).strip()
    if text.startswith("aflow:"):
        return "aflow-" + text.removeprefix("aflow:")
    if text.startswith("aflow-"):
        return text
    return f"aflow-{text}"


def _parse_composition_string(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for element, count in _ELEMENT_TOKEN.findall(text or ""):
        out[element] = out.get(element, 0.0) + float(count or 1)
    return out


def _composition_from_aflow(doc: dict[str, Any]) -> CompositionInfo:
    formula = str(doc.get("compound") or "unknown")
    amounts = _parse_composition_string(formula)
    if not amounts:
        species = doc.get("species") or []
        if isinstance(species, list):
            amounts = {str(el): 1.0 for el in species if el}

    if not amounts:
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
        nelements=_int(doc.get("nspecies")) or (len(elements) or None),
        composition=amounts,
    )


def _symmetry_from_aflow(doc: dict[str, Any]) -> SymmetryInfo:
    number = _int(doc.get("spacegroup_relax"))
    symbol = None
    if number is not None:
        try:
            from pymatgen.symmetry.groups import SpaceGroup

            symbol = SpaceGroup.from_int_number(number).symbol
        except Exception:  # noqa: BLE001
            symbol = None
    return SymmetryInfo(
        crystal_system=None,
        space_group_symbol=symbol,
        space_group_number=number,
        point_group=None,
        number_of_species=_int(doc.get("nspecies")),
    )


def _structure_ref_from_aflow(
    doc: dict[str, Any],
    *,
    settings: Settings | None = None,
) -> StructureRef:
    source_id = str(doc.get("source_id") or "").strip()
    path = None
    fmt = None
    if settings is not None and source_id:
        candidate = (
            settings.computational_structures_dir()
            / f"{source_id.replace('/', '_')}.vasp"
        )
        if candidate.exists() and candidate.stat().st_size > 80:
            path = str(candidate.relative_to(settings.project_root))
            fmt = "vasp"
    return StructureRef(
        path=path,
        format=fmt,
        nsites=_int(doc.get("natoms")),
        volume=_num(doc.get("volume_cell")),
        density=_num(doc.get("density")),
    )


def _properties_from_aflow(doc: dict[str, Any], source_id: str) -> tuple[SourceProperty, ...]:
    band_gap = sanitize_band_gap(doc.get("Egap"))
    catalog = str(doc.get("catalog") or "").upper()
    # ICSD entries are experimentally reported structures (DFT properties still computed).
    theoretical = catalog != "ICSD"
    # AFLOW ICSD catalog export does not include energy-above-hull; do not invent it.
    mapping: list[tuple[PropertyName, Any]] = [
        (
            PropertyName.FORMATION_ENERGY_PER_ATOM,
            sanitize_formation_energy(doc.get("enthalpy_formation_atom")),
        ),
        (PropertyName.ENERGY_PER_ATOM, _num(doc.get("energy_atom"))),
        (PropertyName.BAND_GAP, band_gap),
        (PropertyName.DENSITY, sanitize_density(doc.get("density"))),
        (PropertyName.VOLUME, sanitize_volume(doc.get("volume_cell"))),
        (PropertyName.NSITES, _int(doc.get("natoms"))),
        (PropertyName.BULK_MODULUS_VRH, sanitize_elastic(doc.get("ael_bulk_modulus_vrh"))),
        (PropertyName.SHEAR_MODULUS_VRH, sanitize_elastic(doc.get("ael_shear_modulus_vrh"))),
        (PropertyName.IS_METAL, (band_gap is not None and band_gap <= 0.0)),
        (PropertyName.THEORETICAL, theoretical),
    ]
    props: list[SourceProperty] = []
    for name, value in mapping:
        prop = make_source_property(name, value, SOURCE, source_id, method="aflow-dft")
        if prop is not None:
            props.append(prop)
    return tuple(props)


def normalize_aflow_document(
    doc: dict[str, Any],
    *,
    index: int,
    settings: Settings | None = None,
) -> MaterialRecord:
    settings = settings or get_settings()
    source_id = str(doc.get("source_id") or "").strip()
    if not source_id:
        auid = doc.get("auid")
        if not auid:
            raise ValueError("AFLOW document missing auid/source_id")
        source_id = _canonical_source_id(str(auid))

    material_id = make_material_id(index)
    source_record = SourceRecord(
        source=SOURCE,
        source_id=source_id,
        composition=_composition_from_aflow(doc),
        properties=_properties_from_aflow(doc, source_id),
        symmetry=_symmetry_from_aflow(doc),
        structure=_structure_ref_from_aflow(doc, settings=settings),
        original_metadata={
            "auid": doc.get("auid"),
            "aurl": doc.get("aurl"),
            "catalog": doc.get("catalog"),
            "Pearson_symbol_relax": doc.get("Pearson_symbol_relax"),
            "species": doc.get("species"),
            # Explicit: this AFLOW slice has no hull energies — not comparable
            # to MP/JARVIS/OQMD energy_above_hull.
            "energy_above_hull_available": False,
            "hull_comparable": False,
        },
        material_id=material_id,
    )
    return MaterialRecord(material_id=material_id, sources=(source_record,))


def iter_raw_aflow_documents(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}") from exc


def iter_normalized_aflow(
    path: Path,
    *,
    start_index: int = 1,
    settings: Settings | None = None,
) -> Iterator[MaterialRecord]:
    settings = settings or get_settings()
    for offset, doc in enumerate(iter_raw_aflow_documents(path)):
        yield normalize_aflow_document(doc, index=start_index + offset, settings=settings)


def iter_load_normalized_aflow(
    settings: Settings | None = None,
    *,
    start_index: int = 1,
) -> Iterator[MaterialRecord]:
    settings = settings or get_settings()
    path = settings.raw_source_dir(DataSource.AFLOW) / "materials.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Raw AFLOW data not found at {path}. Run scripts/download_data.py --source aflow."
        )
    yield from iter_normalized_aflow(path, start_index=start_index, settings=settings)
