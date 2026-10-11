"""Known-material search workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config.settings import Settings, get_settings
from database.queries import (
    MaterialSearchFilters,
    database_stats,
    get_material_by_id,
    search_materials,
)


@dataclass(frozen=True)
class SearchRequest:
    elements_include: tuple[str, ...] = ()
    elements_exclude: tuple[str, ...] = ()
    formula_contains: str | None = None
    chemsys: str | None = None
    source: str | None = None
    band_gap_min: float | None = None
    band_gap_max: float | None = None
    energy_above_hull_max: float | None = None
    density_min: float | None = None
    density_max: float | None = None
    bulk_modulus_min: float | None = None
    mbj_bandgap_min: float | None = None
    mbj_bandgap_max: float | None = None
    is_stable: bool | None = None
    is_metal: bool | None = None
    is_magnetic: bool | None = None
    is_gap_direct: bool | None = None
    nelements_min: int | None = None
    nelements_max: int | None = None
    crystal_system: str | None = None
    ordering: str | None = None
    has_structure: bool | None = None
    exclude_deprecated: bool = True
    limit: int = 50
    offset: int = 0


@dataclass(frozen=True)
class SearchResult:
    materials: list[dict[str, Any]]
    count: int
    request: SearchRequest
    database_path: str


def _clean_elements(values: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    if not values:
        return ()
    cleaned: list[str] = []
    for value in values:
        symbol = str(value).strip()
        if not symbol:
            continue
        cleaned.append(symbol[:1].upper() + symbol[1:])
    return tuple(dict.fromkeys(cleaned))


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    return text or None


def run_search(
    request: SearchRequest,
    *,
    settings: Settings | None = None,
) -> SearchResult:
    settings = settings or get_settings()
    filters = MaterialSearchFilters(
        elements_include=_clean_elements(request.elements_include),
        elements_exclude=_clean_elements(request.elements_exclude),
        formula_contains=_optional_text(request.formula_contains),
        chemsys=_optional_text(request.chemsys),
        source=_optional_text(request.source),
        band_gap_min=request.band_gap_min,
        band_gap_max=request.band_gap_max,
        energy_above_hull_max=request.energy_above_hull_max,
        density_min=request.density_min,
        density_max=request.density_max,
        bulk_modulus_min=request.bulk_modulus_min,
        mbj_bandgap_min=request.mbj_bandgap_min,
        mbj_bandgap_max=request.mbj_bandgap_max,
        is_stable=request.is_stable,
        is_metal=request.is_metal,
        is_magnetic=request.is_magnetic,
        is_gap_direct=request.is_gap_direct,
        nelements_min=request.nelements_min,
        nelements_max=request.nelements_max,
        crystal_system=_optional_text(request.crystal_system),
        ordering=_optional_text(request.ordering),
        has_structure=request.has_structure,
        exclude_deprecated=request.exclude_deprecated,
        limit=max(1, min(request.limit, 500)),
        offset=max(0, request.offset),
    )
    materials = search_materials(filters)
    return SearchResult(
        materials=materials,
        count=len(materials),
        request=request,
        database_path=str(settings.duckdb_path),
    )


def get_database_overview() -> dict[str, Any]:
    settings = get_settings()
    if not settings.duckdb_path.exists():
        return {
            "materials": 0,
            "source_records": 0,
            "properties": 0,
            "with_structures": 0,
            "multi_source_materials": 0,
            "database_path": str(settings.duckdb_path),
            "database_exists": False,
        }
    stats = database_stats()
    return {
        **stats,
        "database_path": str(settings.duckdb_path),
        "database_exists": True,
    }


def fetch_material(material_id: str) -> dict[str, Any] | None:
    return get_material_by_id(material_id)


def parse_element_list(text: str) -> tuple[str, ...]:
    if not text or not text.strip():
        return ()
    parts: list[str] = []
    for chunk in text.replace(";", ",").replace(" ", ",").split(","):
        chunk = chunk.strip()
        if chunk:
            parts.append(chunk)
    return _clean_elements(parts)
