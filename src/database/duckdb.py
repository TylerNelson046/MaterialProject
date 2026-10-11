"""DuckDB persistence for the unified materials database."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

import duckdb
import pandas as pd

from config.settings import get_settings
from normalization.schema import MaterialRecord, SourceProperty, SourceRecord

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS materials (
    material_id VARCHAR PRIMARY KEY,
    formula_pretty VARCHAR NOT NULL,
    chemsys VARCHAR,
    nelements INTEGER,
    created_at VARCHAR,
    updated_at VARCHAR,
    notes VARCHAR
);

CREATE TABLE IF NOT EXISTS source_records (
    source VARCHAR NOT NULL,
    source_id VARCHAR NOT NULL,
    material_id VARCHAR NOT NULL,
    formula_pretty VARCHAR,
    reduced_formula VARCHAR,
    chemsys VARCHAR,
    elements JSON,
    nelements INTEGER,
    composition JSON,
    crystal_system VARCHAR,
    space_group_symbol VARCHAR,
    space_group_number INTEGER,
    point_group VARCHAR,
    nsites INTEGER,
    volume DOUBLE,
    density DOUBLE,
    structure_path VARCHAR,
    original_metadata JSON,
    retrieved_at VARCHAR,
    PRIMARY KEY (source, source_id)
);

CREATE TABLE IF NOT EXISTS properties (
    source VARCHAR NOT NULL,
    source_id VARCHAR NOT NULL,
    material_id VARCHAR NOT NULL,
    name VARCHAR NOT NULL,
    value_double DOUBLE,
    value_bool BOOLEAN,
    value_text VARCHAR,
    unit VARCHAR NOT NULL,
    origin VARCHAR NOT NULL,
    method VARCHAR,
    uncertainty DOUBLE,
    PRIMARY KEY (source, source_id, name)
);

CREATE TABLE IF NOT EXISTS material_elements (
    material_id VARCHAR NOT NULL,
    element VARCHAR NOT NULL,
    PRIMARY KEY (material_id, element)
);

CREATE TABLE IF NOT EXISTS search_index (
    material_id VARCHAR PRIMARY KEY,
    source VARCHAR NOT NULL,
    source_id VARCHAR NOT NULL,
    formula_pretty VARCHAR NOT NULL,
    chemsys VARCHAR,
    elements JSON,
    nelements INTEGER,
    nsites INTEGER,
    volume DOUBLE,
    density DOUBLE,
    formation_energy_per_atom DOUBLE,
    energy_above_hull DOUBLE,
    energy_per_atom DOUBLE,
    band_gap DOUBLE,
    is_gap_direct BOOLEAN,
    efermi DOUBLE,
    cbm DOUBLE,
    vbm DOUBLE,
    total_magnetization DOUBLE,
    total_magnetization_normalized_vol DOUBLE,
    num_magnetic_sites INTEGER,
    ordering VARCHAR,
    bulk_modulus_vrh DOUBLE,
    shear_modulus_vrh DOUBLE,
    homogeneous_poisson DOUBLE,
    universal_anisotropy DOUBLE,
    dielectric_total DOUBLE,
    dielectric_ionic DOUBLE,
    dielectric_electronic DOUBLE,
    refractive_index DOUBLE,
    weighted_surface_energy DOUBLE,
    weighted_work_function DOUBLE,
    is_stable BOOLEAN,
    is_metal BOOLEAN,
    is_magnetic BOOLEAN,
    theoretical BOOLEAN,
    crystal_system VARCHAR,
    space_group_symbol VARCHAR,
    space_group_number INTEGER,
    structure_path VARCHAR,
    n_sources INTEGER,
    deprecated BOOLEAN,
    mbj_bandgap DOUBLE,
    avg_elec_mass DOUBLE,
    avg_hole_mass DOUBLE,
    slme DOUBLE,
    tc_supercon DOUBLE,
    exfoliation_energy DOUBLE,
    neutron_absorption_cross_section DOUBLE,
    melting_point_element_mix DOUBLE,
    average_voltage DOUBLE,
    capacity_grav DOUBLE,
    capacity_vol DOUBLE,
    energy_grav DOUBLE,
    working_ion VARCHAR,
    thermal_conductivity DOUBLE,
    thermal_conductivity_method VARCHAR,
    thermal_conductivity_source VARCHAR,
    thermal_conductivity_proxy DOUBLE,
    debye_temperature DOUBLE,
    macroscopic_absorption DOUBLE,
    has_strong_neutron_absorber BOOLEAN,
    radiation_hardness_score DOUBLE,
    youngs_modulus_vrh DOUBLE,
    has_phonon BOOLEAN,
    heat_capacity_cv DOUBLE,
    corrosion_resistance_score DOUBLE,
    oxidation_tendency DOUBLE,
    displacement_energy_mix DOUBLE,
    radiation_damage_susceptibility DOUBLE,
    energy_vol DOUBLE,
    specific_stiffness DOUBLE,
    specific_bulk_modulus DOUBLE,
    neutron_mean_free_path DOUBLE,
    macroscopic_absorption_per_mass DOUBLE,
    melting_to_density_ratio DOUBLE,
    thermal_to_density_ratio DOUBLE,
    is_near_hull BOOLEAN,
    is_semiconductor_window BOOLEAN,
    elastic_anisotropy_abs DOUBLE,
    exclude_deprecated BOOLEAN,
    train_ok_thermo BOOLEAN,
    train_ok_gap BOOLEAN,
    train_ok_elastic BOOLEAN,
    train_ok_thermal_agl BOOLEAN,
    train_ok_battery BOOLEAN,
    label_conflict BOOLEAN,
    conflict_band_gap BOOLEAN,
    conflict_formation_energy BOOLEAN,
    conflict_energy_above_hull BOOLEAN,
    conflict_density BOOLEAN
);
"""

SEARCH_PROP_NAMES = (
    "volume",
    "density",
    "formation_energy_per_atom",
    "energy_above_hull",
    "energy_per_atom",
    "band_gap",
    "is_gap_direct",
    "efermi",
    "cbm",
    "vbm",
    "total_magnetization",
    "total_magnetization_normalized_vol",
    "num_magnetic_sites",
    "ordering",
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
    "is_stable",
    "is_metal",
    "is_magnetic",
    "theoretical",
    "deprecated",
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
    "working_ion",
    "thermal_conductivity",
    "debye_temperature",
    "macroscopic_absorption",
    "has_strong_neutron_absorber",
    "radiation_hardness_score",
    "youngs_modulus_vrh",
    "has_phonon",
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
    "is_near_hull",
    "is_semiconductor_window",
    "elastic_anisotropy_abs",
)

# Columns that may need ALTER TABLE on existing DuckDB files.
SEARCH_INDEX_EXTRA_COLUMNS: tuple[tuple[str, str], ...] = (
    ("n_sources", "INTEGER"),
    ("deprecated", "BOOLEAN"),
    ("mbj_bandgap", "DOUBLE"),
    ("avg_elec_mass", "DOUBLE"),
    ("avg_hole_mass", "DOUBLE"),
    ("slme", "DOUBLE"),
    ("tc_supercon", "DOUBLE"),
    ("exfoliation_energy", "DOUBLE"),
    ("neutron_absorption_cross_section", "DOUBLE"),
    ("melting_point_element_mix", "DOUBLE"),
    ("average_voltage", "DOUBLE"),
    ("capacity_grav", "DOUBLE"),
    ("capacity_vol", "DOUBLE"),
    ("energy_grav", "DOUBLE"),
    ("working_ion", "VARCHAR"),
    ("thermal_conductivity", "DOUBLE"),
    ("thermal_conductivity_method", "VARCHAR"),
    ("thermal_conductivity_source", "VARCHAR"),
    ("thermal_conductivity_proxy", "DOUBLE"),
    ("debye_temperature", "DOUBLE"),
    ("macroscopic_absorption", "DOUBLE"),
    ("has_strong_neutron_absorber", "BOOLEAN"),
    ("radiation_hardness_score", "DOUBLE"),
    ("youngs_modulus_vrh", "DOUBLE"),
    ("has_phonon", "BOOLEAN"),
    ("heat_capacity_cv", "DOUBLE"),
    ("corrosion_resistance_score", "DOUBLE"),
    ("oxidation_tendency", "DOUBLE"),
    ("displacement_energy_mix", "DOUBLE"),
    ("radiation_damage_susceptibility", "DOUBLE"),
    ("energy_vol", "DOUBLE"),
    ("specific_stiffness", "DOUBLE"),
    ("specific_bulk_modulus", "DOUBLE"),
    ("neutron_mean_free_path", "DOUBLE"),
    ("macroscopic_absorption_per_mass", "DOUBLE"),
    ("melting_to_density_ratio", "DOUBLE"),
    ("thermal_to_density_ratio", "DOUBLE"),
    ("is_near_hull", "BOOLEAN"),
    ("is_semiconductor_window", "BOOLEAN"),
    ("elastic_anisotropy_abs", "DOUBLE"),
    ("exclude_deprecated", "BOOLEAN"),
    ("train_ok_thermo", "BOOLEAN"),
    ("train_ok_gap", "BOOLEAN"),
    ("train_ok_elastic", "BOOLEAN"),
    ("train_ok_thermal_agl", "BOOLEAN"),
    ("train_ok_battery", "BOOLEAN"),
    ("label_conflict", "BOOLEAN"),
    ("conflict_band_gap", "BOOLEAN"),
    ("conflict_formation_energy", "BOOLEAN"),
    ("conflict_energy_above_hull", "BOOLEAN"),
    ("conflict_density", "BOOLEAN"),
)


def ensure_search_index_columns(conn: duckdb.DuckDBPyConnection) -> list[str]:
    """Add any missing search_index columns (idempotent migration)."""
    cols = {
        row[1]
        for row in conn.execute("PRAGMA table_info('search_index')").fetchall()
    }
    added: list[str] = []
    for name, sql_type in SEARCH_INDEX_EXTRA_COLUMNS:
        if name not in cols:
            conn.execute(f"ALTER TABLE search_index ADD COLUMN {name} {sql_type}")
            added.append(name)
    return added


def _property_value_columns(prop: SourceProperty) -> tuple[float | None, bool | None, str | None]:
    value = prop.value
    if isinstance(value, bool):
        return None, value, None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value), None, None
    if value is not None:
        return None, None, str(value)
    return None, None, None


def _prop_value(record: SourceRecord, name: str) -> Any:
    prop = record.get_property(name)
    return None if prop is None else prop.value


class MaterialsDatabase:
    """Create, write, and query the unified DuckDB materials store."""

    def __init__(self, path: Path | None = None, *, read_only: bool = False) -> None:
        settings = get_settings()
        self.path = path or settings.duckdb_path
        self.read_only = read_only
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connect(self) -> Iterator[duckdb.DuckDBPyConnection]:
        conn = duckdb.connect(str(self.path), read_only=self.read_only)
        try:
            yield conn
        finally:
            conn.close()

    def initialize(self) -> None:
        with self.connect() as conn:
            conn.execute(SCHEMA_SQL)
            ensure_search_index_columns(conn)

    def reset(self) -> None:
        with self.connect() as conn:
            for table in (
                "search_index",
                "material_elements",
                "properties",
                "source_records",
                "materials",
            ):
                conn.execute(f"DROP TABLE IF EXISTS {table}")
            conn.execute(SCHEMA_SQL)

    def insert_materials(
        self,
        materials: Iterable[MaterialRecord],
        *,
        batch_size: int = 2000,
    ) -> int:
        """Insert normalized materials in batches. Returns count written."""
        batch: list[MaterialRecord] = []
        total = 0
        for material in materials:
            batch.append(material)
            if len(batch) >= batch_size:
                total += self._insert_batch(batch)
                print(f"  ... inserted {total} materials")
                batch.clear()
        if batch:
            total += self._insert_batch(batch)
        return total

    def _insert_batch(self, materials: list[MaterialRecord]) -> int:
        material_rows: list[dict[str, Any]] = []
        source_rows: list[dict[str, Any]] = []
        property_rows: list[dict[str, Any]] = []
        element_rows: list[dict[str, Any]] = []
        search_rows: list[dict[str, Any]] = []

        for material in materials:
            primary = material.primary_source
            material_rows.append(
                {
                    "material_id": material.material_id,
                    "formula_pretty": material.formula_pretty,
                    "chemsys": primary.composition.chemsys,
                    "nelements": primary.composition.nelements,
                    "created_at": material.created_at,
                    "updated_at": material.updated_at,
                    "notes": material.notes,
                }
            )

            for source in material.sources:
                sym = source.symmetry
                struct = source.structure
                source_rows.append(
                    {
                        "source": source.source.value,
                        "source_id": source.source_id,
                        "material_id": material.material_id,
                        "formula_pretty": source.composition.formula_pretty,
                        "reduced_formula": source.composition.reduced_formula,
                        "chemsys": source.composition.chemsys,
                        "elements": json.dumps(list(source.composition.elements)),
                        "nelements": source.composition.nelements,
                        "composition": json.dumps(source.composition.composition),
                        "crystal_system": sym.crystal_system if sym else None,
                        "space_group_symbol": sym.space_group_symbol if sym else None,
                        "space_group_number": sym.space_group_number if sym else None,
                        "point_group": sym.point_group if sym else None,
                        "nsites": struct.nsites if struct else None,
                        "volume": struct.volume if struct else None,
                        "density": struct.density if struct else None,
                        "structure_path": struct.path if struct else None,
                        "original_metadata": json.dumps(source.original_metadata),
                        "retrieved_at": source.retrieved_at,
                    }
                )
                for prop in source.properties:
                    value_double, value_bool, value_text = _property_value_columns(prop)
                    property_rows.append(
                        {
                            "source": prop.source.value,
                            "source_id": prop.source_id,
                            "material_id": material.material_id,
                            "name": prop.name,
                            "value_double": value_double,
                            "value_bool": value_bool,
                            "value_text": value_text,
                            "unit": prop.unit,
                            "origin": prop.origin,
                            "method": prop.method,
                            "uncertainty": prop.uncertainty,
                        }
                    )
                for element in source.composition.elements:
                    element_rows.append(
                        {"material_id": material.material_id, "element": element}
                    )

            search_row: dict[str, Any] = {
                "material_id": material.material_id,
                "source": primary.source.value,
                "source_id": primary.source_id,
                "formula_pretty": primary.composition.formula_pretty,
                "chemsys": primary.composition.chemsys,
                "elements": json.dumps(list(primary.composition.elements)),
                "nelements": primary.composition.nelements,
                "nsites": primary.structure.nsites if primary.structure else None,
                "crystal_system": primary.symmetry.crystal_system if primary.symmetry else None,
                "space_group_symbol": (
                    primary.symmetry.space_group_symbol if primary.symmetry else None
                ),
                "space_group_number": (
                    primary.symmetry.space_group_number if primary.symmetry else None
                ),
                "structure_path": primary.structure.path if primary.structure else None,
            }
            for name in SEARCH_PROP_NAMES:
                search_row[name] = _prop_value(primary, name)
            search_row["n_sources"] = len(material.sources)
            search_rows.append(search_row)

        with self.connect() as conn:
            self._df_replace(conn, "materials", material_rows)
            self._df_replace(conn, "source_records", source_rows)
            self._df_replace(conn, "properties", property_rows)
            self._df_replace(conn, "material_elements", element_rows)
            self._df_replace(conn, "search_index", search_rows)

        return len(materials)

    @staticmethod
    def _df_replace(
        conn: duckdb.DuckDBPyConnection,
        table: str,
        rows: list[dict[str, Any]],
    ) -> None:
        if not rows:
            return
        frame = pd.DataFrame(rows)
        conn.register("tmp_batch", frame)
        cols = ", ".join(frame.columns)
        conn.execute(f"INSERT OR REPLACE INTO {table} ({cols}) SELECT {cols} FROM tmp_batch")
        conn.unregister("tmp_batch")

    def count_materials(self) -> int:
        with self.connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM materials").fetchone()
        return int(row[0]) if row else 0

    def next_material_index(self) -> int:
        """Return the next integer index for MAT-XXXXXX assignment."""
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT MAX(CAST(substr(material_id, 5) AS INTEGER))
                FROM materials
                WHERE material_id LIKE 'MAT-%'
                """
            ).fetchone()
        current = int(row[0]) if row and row[0] is not None else 0
        return current + 1

    def delete_source(self, source: str) -> int:
        """Remove all rows for a source.

        Multi-source aware: only deletes a material when no other sources remain.
        Returns the number of material rows removed.
        """
        with self.connect() as conn:
            affected = [
                r[0]
                for r in conn.execute(
                    "SELECT DISTINCT material_id FROM source_records WHERE source = ?",
                    [source],
                ).fetchall()
            ]
            if not affected:
                return 0

            conn.execute("DELETE FROM properties WHERE source = ?", [source])
            conn.execute("DELETE FROM source_records WHERE source = ?", [source])
            try:
                conn.execute("DELETE FROM entity_links WHERE source = ?", [source])
            except duckdb.Error:
                pass

            removed = 0
            for material_id in affected:
                remaining = conn.execute(
                    "SELECT COUNT(*) FROM source_records WHERE material_id = ?",
                    [material_id],
                ).fetchone()[0]
                if remaining == 0:
                    for table in (
                        "search_index",
                        "material_elements",
                        "materials",
                    ):
                        conn.execute(
                            f"DELETE FROM {table} WHERE material_id = ?",
                            [material_id],
                        )
                    removed += 1
                else:
                    # Keep material; leave search_index pointing at another source
                    # if the deleted source was primary — best-effort refresh.
                    primary = conn.execute(
                        """
                        SELECT source, source_id, formula_pretty, chemsys, elements,
                               nelements, nsites, crystal_system, space_group_symbol,
                               space_group_number, structure_path
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
                        LIMIT 1
                        """,
                        [material_id],
                    ).fetchone()
                    if primary is None:
                        continue
                    n_sources = int(remaining)
                    # Update materials + search_index identity columns.
                    conn.execute(
                        """
                        UPDATE materials
                        SET formula_pretty = ?, chemsys = ?, nelements = ?
                        WHERE material_id = ?
                        """,
                        [primary[2], primary[3], primary[5], material_id],
                    )
                    conn.execute(
                        """
                        UPDATE search_index
                        SET source = ?, source_id = ?, formula_pretty = ?,
                            chemsys = ?, elements = ?, nelements = ?,
                            nsites = ?, crystal_system = ?,
                            space_group_symbol = ?, space_group_number = ?,
                            structure_path = ?, n_sources = ?
                        WHERE material_id = ?
                        """,
                        [
                            primary[0],
                            primary[1],
                            primary[2],
                            primary[3],
                            primary[4],
                            primary[5],
                            primary[6],
                            primary[7],
                            primary[8],
                            primary[9],
                            primary[10],
                            n_sources,
                            material_id,
                        ],
                    )
        return removed

    def count_by_source(self, source: str) -> int:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM source_records WHERE source = ?",
                [source],
            ).fetchone()
        return int(row[0]) if row else 0
