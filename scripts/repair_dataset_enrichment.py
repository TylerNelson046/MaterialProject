"""Dataset enrichment repairs (safe to run while OQMD downloads raw JSONL).

1. Fill AFLOW ``space_group_symbol`` from ``space_group_number``
2. Coalesce null ``search_index`` property columns from linked sources
3. Backfill COD density (and property rows) from composition + cell volume
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config.settings import clear_settings_cache, get_settings  # noqa: E402
from database.duckdb import SEARCH_PROP_NAMES, MaterialsDatabase  # noqa: E402
from database.write_lock import duckdb_write_lock  # noqa: E402
from normalization.normalize_cod import parse_cod_formula  # noqa: E402

SOURCE_PRIORITY_SQL = """
CASE s.source
    WHEN 'materials_project' THEN 0
    WHEN 'jarvis' THEN 1
    WHEN 'oqmd' THEN 2
    WHEN 'aflow' THEN 3
    WHEN 'cod' THEN 4
    WHEN 'alexandria' THEN 5
    WHEN 'derived' THEN 6
    ELSE 99
END
"""

SOURCE_PRIORITY_SQL_SR = """
CASE source
    WHEN 'materials_project' THEN 0
    WHEN 'jarvis' THEN 1
    WHEN 'oqmd' THEN 2
    WHEN 'aflow' THEN 3
    WHEN 'cod' THEN 4
    WHEN 'alexandria' THEN 5
    WHEN 'derived' THEN 6
    ELSE 99
END
"""

# Properties to backfill on search_index from any linked source.
ENRICH_DOUBLE = (
    "volume",
    "density",
    "formation_energy_per_atom",
    "energy_above_hull",
    "energy_per_atom",
    "band_gap",
    "efermi",
    "cbm",
    "vbm",
    "total_magnetization",
    "total_magnetization_normalized_vol",
    "bulk_modulus_vrh",
    "shear_modulus_vrh",
    "homogeneous_poisson",
    "universal_anisotropy",
    "dielectric_total",
    "dielectric_ionic",
    "dielectric_electronic",
    "refractive_index",
    "weighted_surface_energy",
    "weighted_work_function",
    "mbj_bandgap",
    "avg_elec_mass",
    "avg_hole_mass",
    "slme",
    "tc_supercon",
    "exfoliation_energy",
    "neutron_absorption_cross_section",
    "melting_point_element_mix",
    "average_voltage",
    "capacity_grav",
    "capacity_vol",
    "energy_grav",
    "thermal_conductivity",
    "debye_temperature",
    "macroscopic_absorption",
    "radiation_hardness_score",
    "youngs_modulus_vrh",
    "heat_capacity_cv",
    "corrosion_resistance_score",
    "oxidation_tendency",
    "displacement_energy_mix",
    "radiation_damage_susceptibility",
    "energy_vol",
    "specific_stiffness",
    "specific_bulk_modulus",
    "neutron_mean_free_path",
    "macroscopic_absorption_per_mass",
    "melting_to_density_ratio",
    "thermal_to_density_ratio",
    "elastic_anisotropy_abs",
)
ENRICH_BOOL = (
    "is_gap_direct",
    "is_stable",
    "is_metal",
    "is_magnetic",
    "deprecated",
    "has_strong_neutron_absorber",
    "has_phonon",
    "is_near_hull",
    "is_semiconductor_window",
)
ENRICH_INT = ("num_magnetic_sites",)
ENRICH_TEXT = ("ordering", "working_ion")


def _sg_symbol_from_number(number: int) -> str | None:
    try:
        from pymatgen.symmetry.groups import SpaceGroup

        return SpaceGroup.from_int_number(int(number)).symbol
    except Exception:  # noqa: BLE001
        return None


def _sg_symbol_table() -> dict[int, str]:
    """Precompute HM symbols for space groups 1–230."""
    out: dict[int, str] = {}
    for n in range(1, 231):
        symbol = _sg_symbol_from_number(n)
        if symbol:
            out[n] = symbol
    return out


def repair_aflow_space_group_symbols(conn) -> int:
    table = _sg_symbol_table()
    rows = conn.execute(
        """
        SELECT source_id, space_group_number
        FROM source_records
        WHERE source = 'aflow'
          AND space_group_number IS NOT NULL
          AND (space_group_symbol IS NULL OR space_group_symbol = '')
        """
    ).fetchall()
    updates = []
    for source_id, number in rows:
        symbol = table.get(int(number))
        if symbol:
            updates.append({"source_id": source_id, "space_group_symbol": symbol})
    if not updates:
        print("AFLOW SG symbols: nothing to update")
        return 0
    frame = pd.DataFrame(updates)
    conn.register("aflow_sg", frame)
    conn.execute(
        """
        UPDATE source_records AS s
        SET space_group_symbol = a.space_group_symbol
        FROM aflow_sg AS a
        WHERE s.source = 'aflow' AND s.source_id = a.source_id
        """
    )
    conn.execute(
        """
        UPDATE search_index AS si
        SET space_group_symbol = a.space_group_symbol
        FROM aflow_sg AS a
        WHERE si.source = 'aflow' AND si.source_id = a.source_id
          AND (si.space_group_symbol IS NULL OR si.space_group_symbol = '')
        """
    )
    conn.execute(
        """
        UPDATE search_index AS si
        SET space_group_symbol = s.space_group_symbol
        FROM source_records AS s
        WHERE si.material_id = s.material_id
          AND s.source = 'aflow'
          AND s.space_group_symbol IS NOT NULL
          AND (si.space_group_symbol IS NULL OR si.space_group_symbol = '')
        """
    )
    conn.unregister("aflow_sg")
    print(f"AFLOW SG symbols filled: {len(updates):,}")
    return len(updates)


def repair_cod_density(conn, *, settings) -> int:
    raw_path = settings.raw_source_dir("cod") / "materials.jsonl"
    if not raw_path.exists():
        print("COD density: raw JSONL missing, skip")
        return 0

    density_rows: list[dict] = []
    with raw_path.open(encoding="utf-8") as handle:
        for line in handle:
            doc = json.loads(line)
            source_id = doc.get("source_id")
            volume = doc.get("vol")
            try:
                volume_f = float(volume) if volume is not None else None
            except (TypeError, ValueError):
                volume_f = None
            amounts = parse_cod_formula(
                doc.get("cellformula") or doc.get("calcformula") or doc.get("formula")
            )
            if not source_id or not volume_f or volume_f <= 0 or not amounts:
                continue
            try:
                from pymatgen.core import Composition

                mass_amu = float(Composition(amounts).weight)
                density = mass_amu / volume_f * 1.66053906660
            except Exception:  # noqa: BLE001
                continue
            density_rows.append(
                {
                    "source_id": source_id,
                    "density": density,
                    "volume": volume_f,
                }
            )

    if not density_rows:
        print("COD density: no rows computed")
        return 0

    frame = pd.DataFrame(density_rows)
    conn.register("cod_den", frame)
    # Update structure density on source_records
    conn.execute(
        """
        UPDATE source_records AS s
        SET density = c.density
        FROM cod_den AS c
        WHERE s.source = 'cod' AND s.source_id = c.source_id
        """
    )
    # Upsert density property rows
    conn.execute(
        """
        INSERT OR REPLACE INTO properties
            (source, source_id, material_id, name, value_double, value_bool, value_text,
             unit, origin, method, uncertainty)
        SELECT
            'cod',
            c.source_id,
            s.material_id,
            'density',
            c.density,
            NULL,
            NULL,
            'g/cm^3',
            'known',
            'cod_cell_mass_volume',
            NULL
        FROM cod_den AS c
        JOIN source_records AS s
          ON s.source = 'cod' AND s.source_id = c.source_id
        """
    )
    conn.execute(
        """
        UPDATE search_index AS si
        SET density = c.density
        FROM cod_den AS c
        WHERE si.source = 'cod' AND si.source_id = c.source_id
          AND si.density IS NULL
        """
    )
    conn.unregister("cod_den")
    print(f"COD density filled: {len(density_rows):,}")
    return len(density_rows)


def enrich_search_index_from_linked_sources(conn) -> None:
    """Fill null search_index columns from the best linked source property.

    Uses a bulk ``best_props`` temp table instead of per-row correlated updates.
    """
    print("Enriching search_index nulls from linked sources (bulk)...")

    conn.execute("DROP TABLE IF EXISTS best_props")
    conn.execute(
        f"""
        CREATE TEMP TABLE best_props AS
        SELECT
            s.material_id,
            p.name,
            arg_min(p.value_double, {SOURCE_PRIORITY_SQL}) AS value_double,
            arg_min(p.value_bool, {SOURCE_PRIORITY_SQL}) AS value_bool,
            arg_min(p.value_text, {SOURCE_PRIORITY_SQL}) AS value_text
        FROM properties AS p
        JOIN source_records AS s
          ON s.source = p.source AND s.source_id = p.source_id
        WHERE NOT (
            p.name = 'energy_above_hull'
            AND s.source IN ('aflow', 'cod')
        )
        GROUP BY s.material_id, p.name
        """
    )

    for name in ENRICH_DOUBLE:
        conn.execute(
            f"""
            UPDATE search_index AS si
            SET {name} = b.value_double
            FROM best_props AS b
            WHERE b.material_id = si.material_id
              AND b.name = ?
              AND si.{name} IS NULL
              AND b.value_double IS NOT NULL
            """,
            [name],
        )

    for name in ENRICH_BOOL:
        conn.execute(
            f"""
            UPDATE search_index AS si
            SET {name} = b.value_bool
            FROM best_props AS b
            WHERE b.material_id = si.material_id
              AND b.name = ?
              AND si.{name} IS NULL
              AND b.value_bool IS NOT NULL
            """,
            [name],
        )

    for name in ENRICH_INT:
        conn.execute(
            f"""
            UPDATE search_index AS si
            SET {name} = CAST(b.value_double AS INTEGER)
            FROM best_props AS b
            WHERE b.material_id = si.material_id
              AND b.name = ?
              AND si.{name} IS NULL
              AND b.value_double IS NOT NULL
            """,
            [name],
        )

    for name in ENRICH_TEXT:
        conn.execute(
            f"""
            UPDATE search_index AS si
            SET {name} = b.value_text
            FROM best_props AS b
            WHERE b.material_id = si.material_id
              AND b.name = ?
              AND si.{name} IS NULL
              AND b.value_text IS NOT NULL
            """,
            [name],
        )

    conn.execute("DROP TABLE IF EXISTS best_source_geom")
    conn.execute(
        f"""
        CREATE TEMP TABLE best_source_geom AS
        SELECT
            material_id,
            first(space_group_symbol ORDER BY {SOURCE_PRIORITY_SQL_SR})
                FILTER (WHERE space_group_symbol IS NOT NULL AND space_group_symbol != '')
                AS space_group_symbol,
            first(space_group_number ORDER BY {SOURCE_PRIORITY_SQL_SR})
                FILTER (WHERE space_group_number IS NOT NULL)
                AS space_group_number,
            first(density ORDER BY {SOURCE_PRIORITY_SQL_SR})
                FILTER (WHERE density IS NOT NULL)
                AS density
        FROM source_records
        GROUP BY material_id
        """
    )
    conn.execute(
        """
        UPDATE search_index AS si
        SET
            space_group_symbol = COALESCE(
                NULLIF(si.space_group_symbol, ''),
                g.space_group_symbol
            ),
            space_group_number = COALESCE(si.space_group_number, g.space_group_number),
            density = COALESCE(si.density, g.density)
        FROM best_source_geom AS g
        WHERE g.material_id = si.material_id
        """
    )
    conn.execute("DROP TABLE IF EXISTS best_props")
    conn.execute("DROP TABLE IF EXISTS best_source_geom")
    print("search_index enrichment done")


def _null_report(conn) -> dict[str, int]:
    cols = [
        "band_gap",
        "formation_energy_per_atom",
        "energy_above_hull",
        "density",
        "space_group_symbol",
        "space_group_number",
    ]
    out = {}
    for col in cols:
        out[col] = conn.execute(
            f"SELECT COUNT(*) FROM search_index WHERE {col} IS NULL OR CAST({col} AS VARCHAR)=''"
        ).fetchone()[0]
    out["aflow_missing_symbol"] = conn.execute(
        """
        SELECT COUNT(*) FROM source_records
        WHERE source='aflow' AND (space_group_symbol IS NULL OR space_group_symbol='')
        """
    ).fetchone()[0]
    out["cod_missing_density_prop"] = conn.execute(
        """
        SELECT COUNT(*) FROM source_records s
        WHERE s.source='cod'
          AND NOT EXISTS (
            SELECT 1 FROM properties p
            WHERE p.source='cod' AND p.source_id=s.source_id AND p.name='density'
          )
        """
    ).fetchone()[0]
    return out


def main() -> int:
    clear_settings_cache()
    settings = get_settings()
    settings.ensure_directories()

    with duckdb_write_lock(settings, label="dataset-enrichment"):
        db = MaterialsDatabase(settings.duckdb_path)
        with db.connect() as conn:
            print("Before:", _null_report(conn))
            repair_aflow_space_group_symbols(conn)
            repair_cod_density(conn, settings=settings)
            enrich_search_index_from_linked_sources(conn)
            after = _null_report(conn)
            print("After:", after)

    # Silence unused import warning for SEARCH_PROP_NAMES documentation intent
    _ = SEARCH_PROP_NAMES
    print("Dataset enrichment complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
