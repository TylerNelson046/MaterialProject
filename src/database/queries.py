"""Reusable queries against the unified materials DuckDB database."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import duckdb

from config.constants import CONFLICT_TRACKED_PROPERTIES
from config.settings import get_settings
from database.duckdb import MaterialsDatabase
from database.property_policy import conflict_summary


@dataclass(frozen=True)
class MaterialSearchFilters:
    """Filter criteria for known-material search."""

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
    limit: int = 100
    offset: int = 0


def _connect_readonly() -> duckdb.DuckDBPyConnection:
    settings = get_settings()
    if not settings.duckdb_path.exists():
        raise FileNotFoundError(
            f"Database not found at {settings.duckdb_path}. "
            "Run scripts/build_database.py first."
        )
    return duckdb.connect(str(settings.duckdb_path), read_only=True)


def search_materials(filters: MaterialSearchFilters) -> list[dict[str, Any]]:
    """Search the denormalized search_index with optional filters."""
    clauses: list[str] = ["1=1"]
    params: list[Any] = []

    if filters.exclude_deprecated:
        # Treat NULL as not-deprecated (non-MP sources).
        clauses.append("(deprecated IS NULL OR deprecated = FALSE)")

    if filters.formula_contains:
        clauses.append("lower(formula_pretty) LIKE ?")
        params.append(f"%{filters.formula_contains.lower()}%")

    if filters.chemsys:
        clauses.append("chemsys = ?")
        params.append(filters.chemsys)

    if filters.source:
        clauses.append(
            """
            material_id IN (
                SELECT material_id FROM source_records WHERE source = ?
            )
            """
        )
        params.append(filters.source)

    if filters.band_gap_min is not None:
        clauses.append("band_gap >= ?")
        params.append(filters.band_gap_min)
    if filters.band_gap_max is not None:
        clauses.append("band_gap <= ?")
        params.append(filters.band_gap_max)

    if filters.mbj_bandgap_min is not None:
        clauses.append("mbj_bandgap >= ?")
        params.append(filters.mbj_bandgap_min)
    if filters.mbj_bandgap_max is not None:
        clauses.append("mbj_bandgap <= ?")
        params.append(filters.mbj_bandgap_max)

    if filters.energy_above_hull_max is not None:
        clauses.append("energy_above_hull <= ?")
        params.append(filters.energy_above_hull_max)

    if filters.density_min is not None:
        clauses.append("density >= ?")
        params.append(filters.density_min)
    if filters.density_max is not None:
        clauses.append("density <= ?")
        params.append(filters.density_max)

    if filters.bulk_modulus_min is not None:
        clauses.append("bulk_modulus_vrh >= ?")
        params.append(filters.bulk_modulus_min)

    if filters.is_stable is not None:
        clauses.append("is_stable = ?")
        params.append(filters.is_stable)
    if filters.is_metal is not None:
        clauses.append("is_metal = ?")
        params.append(filters.is_metal)
    if filters.is_magnetic is not None:
        clauses.append("is_magnetic = ?")
        params.append(filters.is_magnetic)
    if filters.is_gap_direct is not None:
        clauses.append("is_gap_direct = ?")
        params.append(filters.is_gap_direct)

    if filters.nelements_min is not None:
        clauses.append("nelements >= ?")
        params.append(filters.nelements_min)
    if filters.nelements_max is not None:
        clauses.append("nelements <= ?")
        params.append(filters.nelements_max)

    if filters.crystal_system:
        clauses.append("lower(crystal_system) = ?")
        params.append(filters.crystal_system.lower())

    if filters.ordering:
        clauses.append("ordering = ?")
        params.append(filters.ordering)

    if filters.has_structure is True:
        clauses.append("structure_path IS NOT NULL")
    elif filters.has_structure is False:
        clauses.append("structure_path IS NULL")

    for element in filters.elements_include:
        clauses.append(
            """
            material_id IN (
                SELECT material_id FROM material_elements WHERE element = ?
            )
            """
        )
        params.append(element)

    for element in filters.elements_exclude:
        clauses.append(
            """
            material_id NOT IN (
                SELECT material_id FROM material_elements WHERE element = ?
            )
            """
        )
        params.append(element)

    sql = f"""
        SELECT *
        FROM search_index
        WHERE {' AND '.join(clauses)}
        ORDER BY energy_above_hull NULLS LAST, formula_pretty
        LIMIT ?
        OFFSET ?
    """
    params.extend([filters.limit, filters.offset])

    conn = _connect_readonly()
    try:
        result = conn.execute(sql, params).fetchdf()
    finally:
        conn.close()

    records = result.to_dict(orient="records")
    for row in records:
        elements = row.get("elements")
        if isinstance(elements, str):
            try:
                row["elements"] = json.loads(elements)
            except json.JSONDecodeError:
                pass
    return records


def get_material_properties(material_id: str) -> list[dict[str, Any]]:
    """All per-source property rows for a material (never collapsed/averaged)."""
    conn = _connect_readonly()
    try:
        frame = conn.execute(
            """
            SELECT
                p.source,
                p.source_id,
                p.name,
                p.value_double,
                p.value_bool,
                p.value_text,
                p.unit,
                p.method,
                p.origin
            FROM properties AS p
            WHERE p.material_id = ?
            ORDER BY
                CASE p.source
                    WHEN 'materials_project' THEN 0
                    WHEN 'jarvis' THEN 1
                    WHEN 'oqmd' THEN 2
                    WHEN 'aflow' THEN 3
                    WHEN 'cod' THEN 4
                    ELSE 9
                END,
                p.name
            """,
            [material_id],
        ).fetchdf()
    finally:
        conn.close()

    rows: list[dict[str, Any]] = []
    for raw in frame.to_dict(orient="records"):
        value: Any = raw.get("value_double")
        if value is None:
            value = raw.get("value_bool")
        if value is None:
            value = raw.get("value_text")
        rows.append(
            {
                "source": raw["source"],
                "source_id": raw["source_id"],
                "name": raw["name"],
                "value": value,
                "unit": raw["unit"],
                "method": raw.get("method"),
                "origin": raw.get("origin"),
            }
        )
    return rows


def get_property_conflicts(material_id: str) -> list[dict[str, Any]]:
    """Numeric cross-source disagreements for a material (spread > 0).

    Policy: values are never averaged — this returns the raw by-source map plus
    min/max/spread so callers can display evidence.
    """
    props = get_material_properties(material_id)
    by_name: dict[str, dict[str, float]] = {}
    units: dict[str, str] = {}
    for row in props:
        name = str(row["name"])
        if name not in CONFLICT_TRACKED_PROPERTIES:
            continue
        value = row.get("value")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        by_name.setdefault(name, {})[str(row["source"])] = float(value)
        units[name] = str(row.get("unit") or "")

    conflicts: list[dict[str, Any]] = []
    for name, values in sorted(by_name.items()):
        summary = conflict_summary(values)
        if summary is None or summary["spread"] <= 0:
            continue
        conflicts.append(
            {
                "name": name,
                "unit": units.get(name, ""),
                **summary,
            }
        )
    conflicts.sort(key=lambda row: (-float(row["spread"]), row["name"]))
    return conflicts


def get_material_by_id(material_id: str) -> dict[str, Any] | None:
    conn = _connect_readonly()
    try:
        result = conn.execute(
            "SELECT * FROM search_index WHERE material_id = ?",
            [material_id],
        ).fetchdf()
        if result.empty:
            return None
        row = result.to_dict(orient="records")[0]
        sources = conn.execute(
            """
            SELECT source, source_id, formula_pretty, space_group_symbol,
                   space_group_number, nsites, structure_path, density, volume
            FROM source_records
            WHERE material_id = ?
            ORDER BY CASE source
                WHEN 'materials_project' THEN 0
                WHEN 'jarvis' THEN 1
                WHEN 'oqmd' THEN 2
                WHEN 'aflow' THEN 3
                WHEN 'cod' THEN 4
                ELSE 9
            END
            """,
            [material_id],
        ).fetchdf()
        row["sources"] = sources.to_dict(orient="records")
        elements = row.get("elements")
        if isinstance(elements, str):
            try:
                row["elements"] = json.loads(elements)
            except json.JSONDecodeError:
                pass
    finally:
        conn.close()

    # Attach full multi-source property inventory + conflict summary.
    row["properties_by_source"] = get_material_properties(material_id)
    row["property_conflicts"] = get_property_conflicts(material_id)
    row["property_policy"] = (
        "search_index uses primary source values; nulls may be filled from the "
        "next-best linked source. Conflicting values are never averaged."
    )
    return row


def database_stats() -> dict[str, int]:
    db = MaterialsDatabase(read_only=True)
    with db.connect() as conn:
        materials = conn.execute("SELECT COUNT(*) FROM materials").fetchone()[0]
        sources = conn.execute("SELECT COUNT(*) FROM source_records").fetchone()[0]
        props = conn.execute("SELECT COUNT(*) FROM properties").fetchone()[0]
        with_structures = conn.execute(
            "SELECT COUNT(*) FROM search_index WHERE structure_path IS NOT NULL"
        ).fetchone()[0]
        deprecated = 0
        cols = {
            row[1]
            for row in conn.execute("PRAGMA table_info('search_index')").fetchall()
        }
        if "deprecated" in cols:
            deprecated = conn.execute(
                "SELECT COUNT(*) FROM search_index WHERE deprecated = TRUE"
            ).fetchone()[0]
        by_source = dict(
            conn.execute(
                "SELECT source, COUNT(*) FROM source_records GROUP BY source"
            ).fetchall()
        )
        multi = conn.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT material_id
                FROM source_records
                GROUP BY material_id
                HAVING COUNT(*) > 1
            )
            """
        ).fetchone()[0]
    return {
        "materials": int(materials),
        "source_records": int(sources),
        "properties": int(props),
        "with_structures": int(with_structures),
        "deprecated_materials": int(deprecated),
        "multi_source_materials": int(multi),
        "materials_project": int(by_source.get("materials_project", 0)),
        "jarvis": int(by_source.get("jarvis", 0)),
        "oqmd": int(by_source.get("oqmd", 0)),
        "aflow": int(by_source.get("aflow", 0)),
        "cod": int(by_source.get("cod", 0)),
    }
