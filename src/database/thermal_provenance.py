"""Thermal conductivity provenance and proxy/DFT column split on ``search_index``.

Step 1.1: ``thermal_conductivity_method`` / ``_source`` tags.
Step 1.2: ``thermal_conductivity`` is AGL-backed only; element-mix lives in
``thermal_conductivity_proxy``.
"""

from __future__ import annotations

from typing import Any

import duckdb

from database.exclusion_rules import THERMAL_AGL_METHOD, THERMAL_PROXY_METHOD

THERMAL_PROP_NAME = "thermal_conductivity"


def ensure_thermal_provenance_columns(conn: duckdb.DuckDBPyConnection) -> list[str]:
    """Ensure SI provenance / proxy columns exist (delegates to search_index extras)."""
    from database.duckdb import ensure_search_index_columns

    return ensure_search_index_columns(conn)


def backfill_thermal_conductivity_provenance(
    conn: duckdb.DuckDBPyConnection,
) -> dict[str, Any]:
    """Populate ``thermal_conductivity_method`` / ``_source`` from ``properties``.

    Preference: ``aflow-agl`` over ``element-table-kappa`` (and any other method).
    Clears provenance when SI κ is NULL. Idempotent.

    After Step 1.2 split, SI ``thermal_conductivity`` should only hold AGL values;
    this backfill still tags whatever is present from ``properties`` winners.
    Prefer running ``separate_proxy_thermal_conductivity`` afterward.
    """
    ensure_thermal_provenance_columns(conn)

    cleared = conn.execute(
        """
        SELECT COUNT(*) FROM search_index
        WHERE thermal_conductivity IS NULL
          AND (
            thermal_conductivity_method IS NOT NULL
            OR thermal_conductivity_source IS NOT NULL
          )
        """
    ).fetchone()[0]
    conn.execute(
        """
        UPDATE search_index
        SET
            thermal_conductivity_method = NULL,
            thermal_conductivity_source = NULL
        WHERE thermal_conductivity IS NULL
          AND (
            thermal_conductivity_method IS NOT NULL
            OR thermal_conductivity_source IS NOT NULL
          )
        """
    )

    conn.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE _thermal_prov AS
        SELECT
            material_id,
            method,
            source,
            ROW_NUMBER() OVER (
                PARTITION BY material_id
                ORDER BY
                    CASE
                        WHEN method = '{THERMAL_AGL_METHOD}' THEN 0
                        WHEN method = '{THERMAL_PROXY_METHOD}' THEN 1
                        ELSE 2
                    END,
                    source
            ) AS rn
        FROM properties
        WHERE name = '{THERMAL_PROP_NAME}'
          AND value_double IS NOT NULL
          AND method IS NOT NULL
        """
    )

    before = conn.execute(
        """
        SELECT
            COUNT(*) FILTER (
                WHERE thermal_conductivity IS NOT NULL
                  AND thermal_conductivity_method IS NULL
            ) AS missing_method,
            COUNT(*) FILTER (
                WHERE thermal_conductivity_method = ?
            ) AS agl,
            COUNT(*) FILTER (
                WHERE thermal_conductivity_method = ?
            ) AS proxy
        FROM search_index
        """,
        [THERMAL_AGL_METHOD, THERMAL_PROXY_METHOD],
    ).fetchone()

    conn.execute(
        """
        UPDATE search_index AS si
        SET
            thermal_conductivity_method = p.method,
            thermal_conductivity_source = p.source
        FROM _thermal_prov AS p
        WHERE si.material_id = p.material_id
          AND p.rn = 1
          AND si.thermal_conductivity IS NOT NULL
        """
    )

    after = conn.execute(
        """
        SELECT
            COUNT(*) FILTER (
                WHERE thermal_conductivity IS NOT NULL
            ) AS kappa_rows,
            COUNT(*) FILTER (
                WHERE thermal_conductivity_method = ?
            ) AS agl,
            COUNT(*) FILTER (
                WHERE thermal_conductivity_method = ?
            ) AS proxy,
            COUNT(*) FILTER (
                WHERE thermal_conductivity IS NOT NULL
                  AND thermal_conductivity_method IS NULL
            ) AS unlabeled,
            COUNT(*) FILTER (
                WHERE thermal_conductivity IS NOT NULL
                  AND thermal_conductivity_method IS NOT NULL
                  AND thermal_conductivity_method NOT IN (?, ?)
            ) AS other_method
        FROM search_index
        """,
        [
            THERMAL_AGL_METHOD,
            THERMAL_PROXY_METHOD,
            THERMAL_AGL_METHOD,
            THERMAL_PROXY_METHOD,
        ],
    ).fetchone()

    conn.execute("DROP TABLE IF EXISTS _thermal_prov")

    return {
        "cleared_null_kappa_provenance": int(cleared),
        "before_missing_method": int(before[0]),
        "before_agl": int(before[1]),
        "before_proxy": int(before[2]),
        "kappa_rows": int(after[0]),
        "agl": int(after[1]),
        "proxy": int(after[2]),
        "unlabeled": int(after[3]),
        "other_method": int(after[4]),
    }


def separate_proxy_thermal_conductivity(
    conn: duckdb.DuckDBPyConnection,
) -> dict[str, Any]:
    """Split SI κ into AGL ``thermal_conductivity`` vs ``thermal_conductivity_proxy``.

    - Fill ``thermal_conductivity_proxy`` from ``properties`` method
      ``element-table-kappa``.
    - Set ``thermal_conductivity`` (and method/source) only from ``aflow-agl``.
    - Clear main κ / provenance on proxy-only rows.

    Idempotent. Properties rows are left unchanged (both methods remain under
    name ``thermal_conductivity``).
    """
    ensure_thermal_provenance_columns(conn)

    before = conn.execute(
        """
        SELECT
            COUNT(*) FILTER (WHERE thermal_conductivity IS NOT NULL) AS kappa,
            COUNT(*) FILTER (
                WHERE thermal_conductivity_method = ?
            ) AS method_agl,
            COUNT(*) FILTER (
                WHERE thermal_conductivity_method = ?
            ) AS method_proxy,
            COUNT(*) FILTER (
                WHERE thermal_conductivity_proxy IS NOT NULL
            ) AS proxy_col
        FROM search_index
        """,
        [THERMAL_AGL_METHOD, THERMAL_PROXY_METHOD],
    ).fetchone()

    # 1) Proxy column from element-mix properties.
    conn.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE _kappa_proxy AS
        SELECT material_id, MAX(value_double) AS kappa
        FROM properties
        WHERE name = '{THERMAL_PROP_NAME}'
          AND method = '{THERMAL_PROXY_METHOD}'
          AND value_double IS NOT NULL
        GROUP BY material_id
        """
    )
    conn.execute(
        """
        UPDATE search_index AS si
        SET thermal_conductivity_proxy = p.kappa
        FROM _kappa_proxy AS p
        WHERE si.material_id = p.material_id
        """
    )

    # 2) Main κ + provenance from AGL only.
    conn.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE _kappa_agl AS
        SELECT
            material_id,
            MAX(value_double) AS kappa,
            MAX(source) AS source
        FROM properties
        WHERE name = '{THERMAL_PROP_NAME}'
          AND method = '{THERMAL_AGL_METHOD}'
          AND value_double IS NOT NULL
        GROUP BY material_id
        """
    )
    conn.execute(
        """
        UPDATE search_index AS si
        SET
            thermal_conductivity = a.kappa,
            thermal_conductivity_method = ?,
            thermal_conductivity_source = a.source
        FROM _kappa_agl AS a
        WHERE si.material_id = a.material_id
        """,
        [THERMAL_AGL_METHOD],
    )

    # 3) Null main κ on rows without AGL (proxy must not sit in thermal_conductivity).
    cleared_main = conn.execute(
        """
        SELECT COUNT(*) FROM search_index AS si
        WHERE si.thermal_conductivity IS NOT NULL
          AND NOT EXISTS (
            SELECT 1 FROM _kappa_agl AS a WHERE a.material_id = si.material_id
          )
        """
    ).fetchone()[0]
    conn.execute(
        """
        UPDATE search_index AS si
        SET
            thermal_conductivity = NULL,
            thermal_conductivity_method = NULL,
            thermal_conductivity_source = NULL
        WHERE si.thermal_conductivity IS NOT NULL
          AND NOT EXISTS (
            SELECT 1 FROM _kappa_agl AS a WHERE a.material_id = si.material_id
          )
        """
    )

    # 4) Clear stale main provenance if main κ is null.
    conn.execute(
        """
        UPDATE search_index
        SET
            thermal_conductivity_method = NULL,
            thermal_conductivity_source = NULL
        WHERE thermal_conductivity IS NULL
          AND (
            thermal_conductivity_method IS NOT NULL
            OR thermal_conductivity_source IS NOT NULL
          )
        """
    )

    after = conn.execute(
        """
        SELECT
            COUNT(*) FILTER (WHERE thermal_conductivity IS NOT NULL) AS kappa_agl,
            COUNT(*) FILTER (
                WHERE thermal_conductivity_method = ?
            ) AS method_agl,
            COUNT(*) FILTER (
                WHERE thermal_conductivity_method = ?
            ) AS method_proxy,
            COUNT(*) FILTER (
                WHERE thermal_conductivity IS NOT NULL
                  AND (
                    thermal_conductivity_method IS DISTINCT FROM ?
                    OR thermal_conductivity_method IS NULL
                  )
            ) AS kappa_not_tagged_agl,
            COUNT(*) FILTER (
                WHERE thermal_conductivity_proxy IS NOT NULL
            ) AS proxy_col,
            COUNT(*) FILTER (
                WHERE thermal_conductivity IS NOT NULL
                  AND thermal_conductivity_proxy IS NOT NULL
            ) AS both
        FROM search_index
        """,
        [THERMAL_AGL_METHOD, THERMAL_PROXY_METHOD, THERMAL_AGL_METHOD],
    ).fetchone()

    conn.execute("DROP TABLE IF EXISTS _kappa_proxy")
    conn.execute("DROP TABLE IF EXISTS _kappa_agl")

    return {
        "before_kappa": int(before[0]),
        "before_method_agl": int(before[1]),
        "before_method_proxy": int(before[2]),
        "before_proxy_col": int(before[3]),
        "cleared_proxy_from_main": int(cleared_main),
        "kappa_agl": int(after[0]),
        "method_agl": int(after[1]),
        "method_proxy_on_main": int(after[2]),
        "kappa_not_tagged_agl": int(after[3]),
        "proxy_col": int(after[4]),
        "both_agl_and_proxy": int(after[5]),
    }
