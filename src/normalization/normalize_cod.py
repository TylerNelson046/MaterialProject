"""Normalize COD raw documents into the unified schema."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterator

from config.constants import DataSource, PropertyName
from config.settings import Settings, get_settings
from database.fingerprints import space_group_number_from_symbol
from normalization.physical_ranges import sanitize_density, sanitize_volume
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

SOURCE = DataSource.COD
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


def _canonical_source_id(doc: dict[str, Any]) -> str:
    if doc.get("source_id"):
        return str(doc["source_id"])
    file_id = doc.get("cod_id") or doc.get("file")
    if file_id is None:
        raise ValueError("COD document missing file/cod_id/source_id")
    return f"cod-{int(file_id)}"


def parse_cod_formula(raw: str | None) -> dict[str, float]:
    """Parse COD Hill formulas like ``- Ca Mg O6 Si2 -`` into amounts."""
    if not raw:
        return {}
    text = str(raw).strip().strip("-").strip().replace(" ", "")
    if not text:
        return {}
    try:
        from pymatgen.core import Composition

        return {str(el): float(amt) for el, amt in Composition(text).items()}
    except Exception:  # noqa: BLE001
        out: dict[str, float] = {}
        for el, count in _ELEMENT_TOKEN.findall(text):
            out[el] = out.get(el, 0.0) + float(count or 1)
        return out


def _composition_from_cod(doc: dict[str, Any]) -> CompositionInfo:
    amounts = parse_cod_formula(doc.get("calcformula") or doc.get("formula"))
    if not amounts:
        amounts = parse_cod_formula(doc.get("cellformula"))

    formula_pretty = None
    reduced = None
    if amounts:
        try:
            from pymatgen.core import Composition

            comp = Composition(amounts)
            formula_pretty = comp.reduced_formula
            reduced = comp.reduced_formula
            amounts = {str(el): float(amt) for el, amt in comp.items()}
        except Exception:  # noqa: BLE001
            formula_pretty = "".join(
                f"{el}{int(a) if float(a).is_integer() else a}"
                for el, a in sorted(amounts.items())
            )
            reduced = formula_pretty
    else:
        formula_pretty = str(doc.get("formula") or doc.get("file") or "unknown")

    elements = tuple(sorted(amounts.keys()))
    return CompositionInfo(
        formula_pretty=formula_pretty or "unknown",
        reduced_formula=reduced,
        chemsys="-".join(elements) if elements else None,
        elements=elements,
        nelements=_int(doc.get("nel")) or (len(elements) or None),
        composition=amounts,
    )


def _nsites_from_cod(doc: dict[str, Any]) -> int | None:
    """Estimate site count from COD cellformula (conventional cell contents)."""
    amounts = parse_cod_formula(doc.get("cellformula"))
    if not amounts:
        amounts = parse_cod_formula(doc.get("calcformula") or doc.get("formula"))
    if not amounts:
        return None
    return int(round(sum(amounts.values())))


def _symmetry_from_cod(
    doc: dict[str, Any],
    *,
    settings: Settings | None = None,
    source_id: str | None = None,
) -> SymmetryInfo:
    raw_sg = None if _is_missing(doc.get("sg")) else str(doc.get("sg"))
    symbol = raw_sg.replace(" ", "") if raw_sg else None
    # Strip superspace / modulated suffixes: Imam(00g)s00 → Imam
    if symbol:
        symbol = re.sub(r"\(.*$", "", symbol)
    number = _int(doc.get("sgNumber"))
    if number is None and symbol:
        # COD often omits hyphens: Fd3m → Fd-3m
        candidates = [symbol]
        hyphenated = re.sub(r"([FIPabcdmn])(\d)", r"\1-\2", symbol, count=1)
        if hyphenated != symbol:
            candidates.append(hyphenated)
        for cand in candidates:
            number = space_group_number_from_symbol(cand)
            if number is not None:
                symbol = cand
                break
    if number is None and settings is not None:
        number, symbol = _spacegroup_from_cif(doc, settings=settings, symbol=symbol)

    crystal_system = None
    if number is not None:
        for lo, hi, name in (
            (1, 2, "Triclinic"),
            (3, 15, "Monoclinic"),
            (16, 74, "Orthorhombic"),
            (75, 142, "Tetragonal"),
            (143, 167, "Trigonal"),
            (168, 194, "Hexagonal"),
            (195, 230, "Cubic"),
        ):
            if lo <= int(number) <= hi:
                crystal_system = name
                break

    return SymmetryInfo(
        crystal_system=crystal_system,
        space_group_symbol=symbol,
        space_group_number=number,
        point_group=None,
        number_of_species=_int(doc.get("nel")),
    )


def _spacegroup_from_cif(
    doc: dict[str, Any],
    *,
    settings: Settings,
    symbol: str | None,
) -> tuple[int | None, str | None]:
    rel = doc.get("structure_path")
    if not rel:
        file_id = doc.get("cod_id") or doc.get("file")
        if file_id is not None:
            path = settings.experimental_structures_dir() / f"cod-{int(file_id)}.cif"
            if path.exists():
                rel = str(path.relative_to(settings.project_root))
    if not rel:
        return None, symbol
    path = Path(rel)
    if not path.is_absolute():
        path = settings.project_root / path
    if not path.exists():
        return None, symbol
    try:
        from pymatgen.core import Structure
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        structure = Structure.from_file(str(path))
        analyzer = SpacegroupAnalyzer(structure, symprec=0.1)
        number = int(analyzer.get_space_group_number())
        hm = str(analyzer.get_space_group_symbol())
        return number, hm or symbol
    except Exception:  # noqa: BLE001
        return None, symbol


def _structure_ref_from_cod(doc: dict[str, Any], *, settings: Settings) -> StructureRef:
    rel = doc.get("structure_path")
    if not rel:
        file_id = doc.get("cod_id") or doc.get("file")
        if file_id is not None:
            path = settings.experimental_structures_dir() / f"cod-{int(file_id)}.cif"
            if path.exists():
                rel = str(path.relative_to(settings.project_root))

    return StructureRef(
        path=str(rel) if rel else None,
        format="cif" if rel else None,
        nsites=_nsites_from_cod(doc),
        volume=sanitize_volume(doc.get("vol")),
        density=None,
        lattice_a=_num(doc.get("a")),
        lattice_b=_num(doc.get("b")),
        lattice_c=_num(doc.get("c")),
        lattice_alpha=_num(doc.get("alpha")),
        lattice_beta=_num(doc.get("beta")),
        lattice_gamma=_num(doc.get("gamma")),
    )


def _properties_from_cod(doc: dict[str, Any], source_id: str) -> tuple[SourceProperty, ...]:
    volume = sanitize_volume(doc.get("vol"))
    density = None
    amounts = parse_cod_formula(doc.get("cellformula") or doc.get("calcformula") or doc.get("formula"))
    if volume and volume > 0 and amounts:
        try:
            from pymatgen.core import Composition

            # amu / Å³ → g/cm³
            mass_amu = float(Composition(amounts).weight)
            density = sanitize_density(mass_amu / float(volume) * 1.66053906660)
        except Exception:  # noqa: BLE001
            density = None

    mapping: list[tuple[PropertyName, Any]] = [
        (PropertyName.VOLUME, volume),
        (PropertyName.DENSITY, density),
        (PropertyName.NSITES, _nsites_from_cod(doc)),
        (PropertyName.THEORETICAL, False),
    ]
    props: list[SourceProperty] = []
    for name, value in mapping:
        prop = make_source_property(
            name, value, SOURCE, source_id, method="cod-experimental"
        )
        if prop is not None:
            props.append(prop)
    return tuple(props)


def normalize_cod_document(
    doc: dict[str, Any],
    *,
    index: int,
    settings: Settings | None = None,
) -> MaterialRecord:
    settings = settings or get_settings()
    source_id = _canonical_source_id(doc)
    material_id = make_material_id(index)
    source_record = SourceRecord(
        source=SOURCE,
        source_id=source_id,
        composition=_composition_from_cod(doc),
        properties=_properties_from_cod(doc, source_id),
        symmetry=_symmetry_from_cod(doc, settings=settings, source_id=source_id),
        structure=_structure_ref_from_cod(doc, settings=settings),
        original_metadata={
            "cod_id": doc.get("cod_id") or doc.get("file"),
            "formula_raw": doc.get("formula"),
            "calcformula": doc.get("calcformula"),
            "cellformula": doc.get("cellformula"),
            "mineral": doc.get("mineral"),
            "chemname": doc.get("chemname"),
            "commonname": doc.get("commonname"),
            "sg": doc.get("sg"),
            "sgHall": doc.get("sgHall"),
            "Z": doc.get("Z"),
            "year": doc.get("year"),
            "doi": doc.get("doi"),
            "journal": doc.get("journal"),
            "title": doc.get("title"),
            "cif_url": doc.get("cif_url"),
            "hull_comparable": False,
        },
        material_id=material_id,
    )
    return MaterialRecord(material_id=material_id, sources=(source_record,))


def iter_raw_cod_documents(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}") from exc


def iter_normalized_cod(
    path: Path,
    *,
    start_index: int = 1,
    settings: Settings | None = None,
) -> Iterator[MaterialRecord]:
    settings = settings or get_settings()
    for offset, doc in enumerate(iter_raw_cod_documents(path)):
        yield normalize_cod_document(doc, index=start_index + offset, settings=settings)


def iter_load_normalized_cod(
    settings: Settings | None = None,
    *,
    start_index: int = 1,
) -> Iterator[MaterialRecord]:
    settings = settings or get_settings()
    path = settings.raw_source_dir(DataSource.COD) / "materials.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Raw COD data not found at {path}. Run scripts/download_data.py --source cod."
        )
    yield from iter_normalized_cod(path, start_index=start_index, settings=settings)
