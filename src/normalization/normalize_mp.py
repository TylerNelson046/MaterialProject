"""Normalize Materials Project raw documents into the unified schema."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from config.constants import DataSource, PropertyName
from config.settings import Settings, get_settings
from normalization.physical_ranges import (
    ELASTIC_HI,
    ELASTIC_LO,
    sanitize_anisotropy,
    sanitize_band_gap,
    sanitize_density,
    sanitize_ehull,
    sanitize_elastic,
    sanitize_formation_energy,
    sanitize_poisson,
    sanitize_volume,
    finite_float,
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

SOURCE = DataSource.MATERIALS_PROJECT
MP_METHOD = "mp-gga-pbe"


def _as_mapping(value: Any) -> dict[str, Any]:
    """Normalize Arrow map values that may arrive as dicts or list[tuple]."""
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        out: dict[str, Any] = {}
        for item in value:
            if isinstance(item, (tuple, list)) and len(item) == 2:
                out[str(item[0])] = item[1]
        return out
    return {}


def _as_float_dict(composition: Any) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in _as_mapping(composition).items():
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def _reduced_formula_string(
    *,
    composition_reduced: Any = None,
    composition: Any = None,
    formula_pretty: str | None = None,
) -> str | None:
    """Canonical reduced formula string (e.g. ``LiEr2Ir``), never a repr of a list."""
    amounts = _as_float_dict(composition_reduced)
    if not amounts:
        amounts = _as_float_dict(composition)

    if amounts:
        try:
            from pymatgen.core import Composition

            return Composition(amounts).reduced_formula
        except Exception:  # noqa: BLE001
            # Deterministic fallback without pymatgen.
            items = sorted((el, amt) for el, amt in amounts.items())
            return "".join(
                f"{el}{int(amt) if float(amt).is_integer() else amt}" for el, amt in items
            )

    if formula_pretty:
        try:
            from pymatgen.core import Composition

            return Composition(str(formula_pretty)).reduced_formula
        except Exception:  # noqa: BLE001
            return str(formula_pretty)
    return None


def _elastic_modulus_gpa(
    value: Any,
    *,
    lo: float = ELASTIC_LO,
    hi: float = ELASTIC_HI,
) -> float | None:
    """Extract a physical elastic modulus (GPa) from MP voigt/reuss/vrh maps."""
    mapping = _as_mapping(value)
    if mapping:
        for key in ("vrh", "reuss", "voigt"):
            if key not in mapping:
                continue
            picked = sanitize_elastic(_as_mapping({key: mapping.get(key)}).get(key))
            # sanitize_elastic uses global bounds; apply lo/hi explicitly:
            number = finite_float(mapping.get(key))
            if number is not None and lo <= number <= hi:
                return number
        return None
    number = finite_float(value)
    if number is not None and lo <= number <= hi:
        return number
    return None


def _vrh(value: Any) -> float | None:
    return _elastic_modulus_gpa(value)


def _poisson_ratio(value: Any) -> float | None:
    return sanitize_poisson(value)


def _universal_anisotropy(value: Any) -> float | None:
    return sanitize_anisotropy(value)


def _symmetry_from_mp(raw: dict[str, Any] | None) -> SymmetryInfo | None:
    if not raw:
        return None
    return SymmetryInfo(
        crystal_system=raw.get("crystal_system"),
        space_group_symbol=raw.get("symbol") or raw.get("space_group_symbol"),
        space_group_number=raw.get("number") or raw.get("space_group_number"),
        point_group=raw.get("point_group"),
        number_of_species=raw.get("number_of_species"),
    )


def _structure_ref_from_mp(
    doc: dict[str, Any],
    *,
    settings: Settings,
) -> StructureRef:
    source_id = str(doc.get("material_id") or "")
    safe = source_id.replace("/", "_")
    path = settings.computational_structures_dir() / f"{safe}.json"
    rel = None
    fmt = None
    if path.exists():
        rel = str(path.relative_to(settings.project_root))
        fmt = "json"

    lattice = None
    structure = doc.get("structure")
    if isinstance(structure, dict):
        lattice = (structure.get("lattice") or {}) if isinstance(structure.get("lattice"), dict) else {}

    return StructureRef(
        path=rel,
        format=fmt,
        nsites=doc.get("nsites"),
        volume=doc.get("volume"),
        density=doc.get("density"),
        lattice_a=(lattice or {}).get("a"),
        lattice_b=(lattice or {}).get("b"),
        lattice_c=(lattice or {}).get("c"),
        lattice_alpha=(lattice or {}).get("alpha"),
        lattice_beta=(lattice or {}).get("beta"),
        lattice_gamma=(lattice or {}).get("gamma"),
    )


def _properties_from_mp(doc: dict[str, Any], source_id: str) -> tuple[SourceProperty, ...]:
    mapping: list[tuple[PropertyName, Any]] = [
        (
            PropertyName.FORMATION_ENERGY_PER_ATOM,
            sanitize_formation_energy(doc.get("formation_energy_per_atom")),
        ),
        (PropertyName.ENERGY_ABOVE_HULL, sanitize_ehull(doc.get("energy_above_hull"))),
        (PropertyName.ENERGY_PER_ATOM, finite_float(doc.get("energy_per_atom"))),
        (
            PropertyName.UNCORRECTED_ENERGY_PER_ATOM,
            finite_float(doc.get("uncorrected_energy_per_atom")),
        ),
        (
            PropertyName.EQUILIBRIUM_REACTION_ENERGY_PER_ATOM,
            finite_float(doc.get("equilibrium_reaction_energy_per_atom")),
        ),
        (PropertyName.BAND_GAP, sanitize_band_gap(doc.get("band_gap"))),
        (PropertyName.IS_GAP_DIRECT, doc.get("is_gap_direct")),
        (PropertyName.IS_METAL, doc.get("is_metal")),
        (PropertyName.EFERMI, finite_float(doc.get("efermi"))),
        (PropertyName.CBM, finite_float(doc.get("cbm"))),
        (PropertyName.VBM, finite_float(doc.get("vbm"))),
        (PropertyName.DENSITY, sanitize_density(doc.get("density"))),
        (PropertyName.VOLUME, sanitize_volume(doc.get("volume"))),
        (PropertyName.NSITES, doc.get("nsites")),
        (PropertyName.IS_STABLE, doc.get("is_stable")),
        (PropertyName.THEORETICAL, doc.get("theoretical")),
        (PropertyName.IS_MAGNETIC, doc.get("is_magnetic")),
        (PropertyName.TOTAL_MAGNETIZATION, finite_float(doc.get("total_magnetization"))),
        (
            PropertyName.TOTAL_MAGNETIZATION_NORMALIZED_VOL,
            finite_float(doc.get("total_magnetization_normalized_vol")),
        ),
        (PropertyName.NUM_MAGNETIC_SITES, doc.get("num_magnetic_sites")),
        (PropertyName.ORDERING, doc.get("ordering")),
        (PropertyName.BULK_MODULUS_VRH, _elastic_modulus_gpa(doc.get("bulk_modulus"))),
        (PropertyName.SHEAR_MODULUS_VRH, _elastic_modulus_gpa(doc.get("shear_modulus"))),
        (PropertyName.HOMOGENEOUS_POISSON, _poisson_ratio(doc.get("homogeneous_poisson"))),
        (PropertyName.UNIVERSAL_ANISOTROPY, _universal_anisotropy(doc.get("universal_anisotropy"))),
        (PropertyName.DIELECTRIC_TOTAL, finite_float(doc.get("e_total"))),
        (PropertyName.DIELECTRIC_IONIC, finite_float(doc.get("e_ionic"))),
        (PropertyName.DIELECTRIC_ELECTRONIC, finite_float(doc.get("e_electronic"))),
        (PropertyName.REFRACTIVE_INDEX, finite_float(doc.get("n"))),
        (PropertyName.WEIGHTED_SURFACE_ENERGY, finite_float(doc.get("weighted_surface_energy"))),
        (PropertyName.WEIGHTED_WORK_FUNCTION, finite_float(doc.get("weighted_work_function"))),
        (PropertyName.DEPRECATED, doc.get("deprecated")),
    ]
    props: list[SourceProperty] = []
    for name, value in mapping:
        prop = make_source_property(name, value, SOURCE, source_id, method=MP_METHOD)
        if prop is not None:
            props.append(prop)
    return tuple(props)


def normalize_mp_document(
    doc: dict[str, Any],
    *,
    index: int,
    settings: Settings | None = None,
) -> MaterialRecord:
    """Convert one raw MP summary dict into a MaterialRecord."""
    settings = settings or get_settings()
    source_id = str(doc.get("material_id") or "").strip()
    if not source_id:
        raise ValueError("MP document missing material_id")

    elements = tuple(str(e) for e in (doc.get("elements") or []))
    formula_pretty = str(doc.get("formula_pretty") or source_id)
    reduced_formula = _reduced_formula_string(
        composition_reduced=doc.get("composition_reduced") or doc.get("reduced_formula"),
        composition=doc.get("composition"),
        formula_pretty=formula_pretty,
    )

    composition = CompositionInfo(
        formula_pretty=formula_pretty,
        reduced_formula=reduced_formula,
        chemsys=doc.get("chemsys"),
        elements=elements,
        nelements=doc.get("nelements") if doc.get("nelements") is not None else len(elements),
        composition=_as_float_dict(doc.get("composition")),
    )

    material_id = make_material_id(index)
    source_record = SourceRecord(
        source=SOURCE,
        source_id=source_id,
        composition=composition,
        properties=_properties_from_mp(doc, source_id),
        symmetry=_symmetry_from_mp(doc.get("symmetry")),
        structure=_structure_ref_from_mp(doc, settings=settings),
        original_metadata={
            "material_id": source_id,
            "formula_anonymous": doc.get("formula_anonymous"),
            "density_atomic": doc.get("density_atomic"),
            "theoretical": doc.get("theoretical"),
            "deprecated": doc.get("deprecated"),
            "decomposes_to": doc.get("decomposes_to"),
            "has_props": doc.get("has_props"),
            "database_IDs": doc.get("database_IDs"),
            "possible_species": doc.get("possible_species"),
            "types_of_magnetic_species": doc.get("types_of_magnetic_species"),
            "num_unique_magnetic_sites": doc.get("num_unique_magnetic_sites"),
            "total_magnetization_normalized_formula_units": doc.get(
                "total_magnetization_normalized_formula_units"
            ),
            "weighted_surface_energy_EV_PER_ANG2": doc.get(
                "weighted_surface_energy_EV_PER_ANG2"
            ),
            "composition_reduced": doc.get("composition_reduced"),
        },
        material_id=material_id,
    )
    return MaterialRecord(material_id=material_id, sources=(source_record,))


def iter_raw_mp_documents(path: Path) -> Iterator[dict[str, Any]]:
    """Yield MP documents from a JSONL or Parquet file."""
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        table = pq.read_table(path)
        for batch in table.to_batches(max_chunksize=5000):
            yield from batch.to_pylist()
        return

    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}") from exc


def _raw_mp_path(settings: Settings) -> Path:
    raw_dir = settings.raw_source_dir(DataSource.MATERIALS_PROJECT)
    parquet = raw_dir / "materials.parquet"
    jsonl = raw_dir / "materials.jsonl"
    if parquet.exists():
        return parquet
    if jsonl.exists():
        return jsonl
    raise FileNotFoundError(
        f"Raw MP data not found under {raw_dir}. Run scripts/download_data.py first."
    )


def iter_normalized_mp(
    path: Path,
    *,
    start_index: int = 1,
    settings: Settings | None = None,
) -> Iterator[MaterialRecord]:
    settings = settings or get_settings()
    for offset, doc in enumerate(iter_raw_mp_documents(path)):
        yield normalize_mp_document(doc, index=start_index + offset, settings=settings)


def normalize_mp_jsonl(
    path: Path,
    *,
    start_index: int = 1,
    settings: Settings | None = None,
) -> list[MaterialRecord]:
    return list(iter_normalized_mp(path, start_index=start_index, settings=settings))


def load_normalized_mp(settings: Settings | None = None) -> list[MaterialRecord]:
    settings = settings or get_settings()
    return normalize_mp_jsonl(_raw_mp_path(settings), settings=settings)


def iter_load_normalized_mp(settings: Settings | None = None) -> Iterator[MaterialRecord]:
    settings = settings or get_settings()
    yield from iter_normalized_mp(_raw_mp_path(settings), settings=settings)
