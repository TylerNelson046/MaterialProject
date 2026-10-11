"""Compute cheap KPIs from existing search_index columns (no new downloads).

KPIs:
  energy_vol                  = energy_grav × density          (Wh/L)
  specific_stiffness          = youngs_modulus_vrh / density
  specific_bulk_modulus       = bulk_modulus_vrh / density
  neutron_mean_free_path      = 1 / macroscopic_absorption     (cm)
  macroscopic_absorption_per_mass = Σ_a / density
  melting_to_density_ratio    = T_m_mix / density
  thermal_to_density_ratio    = COALESCE(AGL κ, proxy κ) / density
  is_near_hull                = energy_above_hull ≤ 0.05
  is_semiconductor_window     = 0.5 ≤ band_gap ≤ 2.5
  elastic_anisotropy_abs      = abs(universal_anisotropy)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config.constants import PROPERTY_UNITS, DataSource, PropertyName  # noqa: E402
from config.settings import clear_settings_cache, get_settings  # noqa: E402
from database.duckdb import MaterialsDatabase, ensure_search_index_columns  # noqa: E402
from database.write_lock import duckdb_write_lock  # noqa: E402

SOURCE = DataSource.DERIVED.value
METHOD = "cheap-kpi-v1"
ORIGIN = "derived"
NEAR_HULL_EV = 0.05

DERIVED_NAMES = (
    PropertyName.ENERGY_VOL.value,
    PropertyName.SPECIFIC_STIFFNESS.value,
    PropertyName.SPECIFIC_BULK_MODULUS.value,
    PropertyName.NEUTRON_MEAN_FREE_PATH.value,
    PropertyName.MACROSCOPIC_ABSORPTION_PER_MASS.value,
    PropertyName.MELTING_TO_DENSITY_RATIO.value,
    PropertyName.THERMAL_TO_DENSITY_RATIO.value,
    PropertyName.IS_NEAR_HULL.value,
    PropertyName.IS_SEMICONDUCTOR_WINDOW.value,
    PropertyName.ELASTIC_ANISOTROPY_ABS.value,
)


def _double(
    *,
    material_id: str,
    source_id: str,
    name: PropertyName,
    value: float,
) -> dict[str, Any]:
    return {
        "source": SOURCE,
        "source_id": source_id,
        "material_id": material_id,
        "name": name.value,
        "value_double": float(value),
        "value_bool": None,
        "value_text": None,
        "unit": PROPERTY_UNITS[name],
        "origin": ORIGIN,
        "method": METHOD,
        "uncertainty": None,
    }


def _bool(
    *,
    material_id: str,
    source_id: str,
    name: PropertyName,
    value: bool,
) -> dict[str, Any]:
    return {
        "source": SOURCE,
        "source_id": source_id,
        "material_id": material_id,
        "name": name.value,
        "value_double": None,
        "value_bool": bool(value),
        "value_text": None,
        "unit": PROPERTY_UNITS[name],
        "origin": ORIGIN,
        "method": METHOD,
        "uncertainty": None,
    }


def _safe_div(num: float | None, den: float | None) -> float | None:
    if num is None or den is None:
        return None
    d = float(den)
    if d == 0.0 or d != d:
        return None
    return float(num) / d


def main() -> int:
    clear_settings_cache()
    settings = get_settings()

    with duckdb_write_lock(settings, label="cheap-kpis"):
        db = MaterialsDatabase(settings.duckdb_path)
        with db.connect() as conn:
            ensure_search_index_columns(conn)
            # Vectorized computation in DuckDB, then write properties + SI.
            conn.execute(
                """
                CREATE OR REPLACE TEMP TABLE kpi_calc AS
                SELECT
                    material_id,
                    CASE
                        WHEN energy_grav IS NOT NULL AND density IS NOT NULL
                             AND density > 0
                        THEN energy_grav * density
                        ELSE NULL
                    END AS energy_vol,
                    CASE
                        WHEN youngs_modulus_vrh IS NOT NULL AND density IS NOT NULL
                             AND density > 0
                        THEN youngs_modulus_vrh / density
                        ELSE NULL
                    END AS specific_stiffness,
                    CASE
                        WHEN bulk_modulus_vrh IS NOT NULL AND density IS NOT NULL
                             AND density > 0
                        THEN bulk_modulus_vrh / density
                        ELSE NULL
                    END AS specific_bulk_modulus,
                    CASE
                        WHEN macroscopic_absorption IS NOT NULL
                             AND macroscopic_absorption > 0
                        THEN 1.0 / macroscopic_absorption
                        ELSE NULL
                    END AS neutron_mean_free_path,
                    CASE
                        WHEN macroscopic_absorption IS NOT NULL AND density IS NOT NULL
                             AND density > 0
                        THEN macroscopic_absorption / density
                        ELSE NULL
                    END AS macroscopic_absorption_per_mass,
                    CASE
                        WHEN melting_point_element_mix IS NOT NULL
                             AND density IS NOT NULL AND density > 0
                        THEN melting_point_element_mix / density
                        ELSE NULL
                    END AS melting_to_density_ratio,
                    CASE
                        WHEN COALESCE(thermal_conductivity, thermal_conductivity_proxy)
                             IS NOT NULL
                             AND density IS NOT NULL AND density > 0
                        THEN COALESCE(thermal_conductivity, thermal_conductivity_proxy)
                             / density
                        ELSE NULL
                    END AS thermal_to_density_ratio,
                    CASE
                        WHEN energy_above_hull IS NOT NULL
                        THEN energy_above_hull <= ?
                        ELSE NULL
                    END AS is_near_hull,
                    CASE
                        WHEN band_gap IS NOT NULL
                        THEN band_gap >= 0.5 AND band_gap <= 2.5
                        ELSE NULL
                    END AS is_semiconductor_window,
                    CASE
                        WHEN universal_anisotropy IS NOT NULL
                        THEN ABS(universal_anisotropy)
                        ELSE NULL
                    END AS elastic_anisotropy_abs
                FROM search_index
                """,
                [NEAR_HULL_EV],
            )

            # Update search_index in one pass
            conn.execute(
                """
                UPDATE search_index AS si
                SET
                    energy_vol = k.energy_vol,
                    specific_stiffness = k.specific_stiffness,
                    specific_bulk_modulus = k.specific_bulk_modulus,
                    neutron_mean_free_path = k.neutron_mean_free_path,
                    macroscopic_absorption_per_mass = k.macroscopic_absorption_per_mass,
                    melting_to_density_ratio = k.melting_to_density_ratio,
                    thermal_to_density_ratio = k.thermal_to_density_ratio,
                    is_near_hull = k.is_near_hull,
                    is_semiconductor_window = k.is_semiconductor_window,
                    elastic_anisotropy_abs = k.elastic_anisotropy_abs
                FROM kpi_calc AS k
                WHERE si.material_id = k.material_id
                """
            )

            # Materialize property rows for provenance
            prop_frame = conn.execute(
                """
                SELECT * FROM kpi_calc
                WHERE energy_vol IS NOT NULL
                   OR specific_stiffness IS NOT NULL
                   OR specific_bulk_modulus IS NOT NULL
                   OR neutron_mean_free_path IS NOT NULL
                   OR macroscopic_absorption_per_mass IS NOT NULL
                   OR melting_to_density_ratio IS NOT NULL
                   OR thermal_to_density_ratio IS NOT NULL
                   OR is_near_hull IS NOT NULL
                   OR is_semiconductor_window IS NOT NULL
                   OR elastic_anisotropy_abs IS NOT NULL
                """
            ).fetchdf()

            prop_rows: list[dict[str, Any]] = []
            for row in prop_frame.itertuples(index=False):
                mid = str(row.material_id)
                sid = f"kpi-{mid}"
                pairs_d = (
                    (PropertyName.ENERGY_VOL, row.energy_vol),
                    (PropertyName.SPECIFIC_STIFFNESS, row.specific_stiffness),
                    (PropertyName.SPECIFIC_BULK_MODULUS, row.specific_bulk_modulus),
                    (PropertyName.NEUTRON_MEAN_FREE_PATH, row.neutron_mean_free_path),
                    (
                        PropertyName.MACROSCOPIC_ABSORPTION_PER_MASS,
                        row.macroscopic_absorption_per_mass,
                    ),
                    (PropertyName.MELTING_TO_DENSITY_RATIO, row.melting_to_density_ratio),
                    (PropertyName.THERMAL_TO_DENSITY_RATIO, row.thermal_to_density_ratio),
                    (PropertyName.ELASTIC_ANISOTROPY_ABS, row.elastic_anisotropy_abs),
                )
                for name, value in pairs_d:
                    if value is not None and not pd.isna(value):
                        prop_rows.append(
                            _double(material_id=mid, source_id=sid, name=name, value=float(value))
                        )
                if row.is_near_hull is not None and not pd.isna(row.is_near_hull):
                    prop_rows.append(
                        _bool(
                            material_id=mid,
                            source_id=sid,
                            name=PropertyName.IS_NEAR_HULL,
                            value=bool(row.is_near_hull),
                        )
                    )
                if row.is_semiconductor_window is not None and not pd.isna(
                    row.is_semiconductor_window
                ):
                    prop_rows.append(
                        _bool(
                            material_id=mid,
                            source_id=sid,
                            name=PropertyName.IS_SEMICONDUCTOR_WINDOW,
                            value=bool(row.is_semiconductor_window),
                        )
                    )

            conn.execute(
                f"""
                DELETE FROM properties
                WHERE source = ?
                  AND name IN ({",".join("?" for _ in DERIVED_NAMES)})
                """,
                [SOURCE, *DERIVED_NAMES],
            )
            if prop_rows:
                frame = pd.DataFrame(prop_rows)
                conn.register("kpi_props", frame)
                conn.execute("INSERT INTO properties SELECT * FROM kpi_props")
                conn.unregister("kpi_props")

            counts = conn.execute(
                """
                SELECT
                    COUNT(energy_vol),
                    COUNT(specific_stiffness),
                    COUNT(specific_bulk_modulus),
                    COUNT(neutron_mean_free_path),
                    COUNT(macroscopic_absorption_per_mass),
                    COUNT(melting_to_density_ratio),
                    COUNT(thermal_to_density_ratio),
                    COUNT(is_near_hull),
                    COUNT(is_semiconductor_window),
                    COUNT(elastic_anisotropy_abs),
                    SUM(CASE WHEN is_near_hull THEN 1 ELSE 0 END),
                    SUM(CASE WHEN is_semiconductor_window THEN 1 ELSE 0 END)
                FROM search_index
                """
            ).fetchone()
            labels = (
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
                "near_hull_true",
                "semiconductor_true",
            )
            print("Cheap KPIs:")
            for label, n in zip(labels, counts, strict=True):
                print(f"  {label}: {n:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
