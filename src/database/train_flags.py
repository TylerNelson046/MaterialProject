"""Train-ready boolean flags on ``search_index`` (Step 1.4).

One-row-per-material gates for common supervised heads. Refresh after
physical QA / provenance changes. Rules follow ``docs/exclusion_rules.md``
and ``docs/label_policy.md``.
"""

from __future__ import annotations

from typing import Any

import duckdb

from database.duckdb import ensure_search_index_columns
from database.exclusion_rules import (
    ELASTIC_DFT_METHODS,
    THERMAL_AGL_METHOD,
    sql_in_physical_range,
    sql_not_deprecated,
)

TRAIN_FLAG_COLUMNS: tuple[str, ...] = (
    "exclude_deprecated",
    "train_ok_thermo",
    "train_ok_gap",
    "train_ok_elastic",
    "train_ok_thermal_agl",
    "train_ok_battery",
)

TRAIN_FLAGS_VERSION = "1.4.0"


def refresh_train_flags(conn: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    """Compute and write all train flags on ``search_index``. Idempotent."""
    ensure_search_index_columns(conn)

    elastic_methods = ", ".join(f"'{m}'" for m in sorted(ELASTIC_DFT_METHODS))
    not_dep = sql_not_deprecated(table_alias="si")
    ef_ok = sql_in_physical_range("formation_energy_per_atom", table_alias="si")
    eh_ok = sql_in_physical_range("energy_above_hull", table_alias="si")
    gap_ok = sql_in_physical_range("band_gap", table_alias="si")
    b_ok = sql_in_physical_range("bulk_modulus_vrh", table_alias="si")
    g_ok = sql_in_physical_range("shear_modulus_vrh", table_alias="si")
    kappa_ok = sql_in_physical_range("thermal_conductivity", table_alias="si")
    debye_ok = sql_in_physical_range("debye_temperature", table_alias="si")
    cv_ok = sql_in_physical_range("heat_capacity_cv", table_alias="si")

    # Material IDs with DFT/AEL elastic provenance for B or G.
    conn.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE _elastic_dft AS
        SELECT DISTINCT material_id
        FROM properties
        WHERE name IN ('bulk_modulus_vrh', 'shear_modulus_vrh')
          AND value_double IS NOT NULL
          AND method IN ({elastic_methods})
        """
    )

    conn.execute(
        f"""
        UPDATE search_index AS si
        SET
            exclude_deprecated = (si.deprecated IS TRUE),
            train_ok_thermo = (
                {not_dep}
                AND {ef_ok}
                AND {eh_ok}
            ),
            train_ok_gap = (
                {not_dep}
                AND {gap_ok}
            ),
            train_ok_elastic = (
                {not_dep}
                AND {b_ok}
                AND {g_ok}
                AND EXISTS (
                    SELECT 1 FROM _elastic_dft AS e
                    WHERE e.material_id = si.material_id
                )
            ),
            train_ok_thermal_agl = (
                {not_dep}
                AND (
                    (
                        {kappa_ok}
                        AND si.thermal_conductivity_method = '{THERMAL_AGL_METHOD}'
                    )
                    OR {debye_ok}
                    OR {cv_ok}
                )
            ),
            train_ok_battery = (
                {not_dep}
                AND si.average_voltage IS NOT NULL
                AND si.capacity_grav IS NOT NULL
            )
        """
    )

    conn.execute("DROP TABLE IF EXISTS _elastic_dft")

    counts = {
        col: int(
            conn.execute(
                f"SELECT COUNT(*) FROM search_index WHERE {col} IS TRUE"
            ).fetchone()[0]
        )
        for col in TRAIN_FLAG_COLUMNS
    }
    n_total = int(conn.execute("SELECT COUNT(*) FROM search_index").fetchone()[0])
    bad_overlap = int(
        conn.execute(
            """
            SELECT COUNT(*) FROM search_index
            WHERE exclude_deprecated IS TRUE
              AND (
                train_ok_thermo IS TRUE
                OR train_ok_gap IS TRUE
                OR train_ok_elastic IS TRUE
                OR train_ok_thermal_agl IS TRUE
                OR train_ok_battery IS TRUE
              )
            """
        ).fetchone()[0]
    )

    return {
        "train_flags_version": TRAIN_FLAGS_VERSION,
        "n_materials": n_total,
        "counts_true": counts,
        "deprecated_and_train_ok_overlap": bad_overlap,
    }
