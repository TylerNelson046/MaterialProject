"""Idempotent physical QA sanitize for training-safe labels (Step 1.3).

Nulls out-of-range ``search_index`` cells and deletes matching OOR
``properties`` rows. Never clips. Gates live in
``normalization.physical_ranges``.
"""

from __future__ import annotations

from typing import Any

import duckdb

from normalization.physical_ranges import (
    PROPERTY_ABS_GATES,
    PROPERTY_RANGE_GATES,
)


def count_oor_search_index(conn: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """Count OOR non-null cells on ``search_index`` per gated column."""
    out: dict[str, int] = {}
    cols = {
        row[1]
        for row in conn.execute("PRAGMA table_info('search_index')").fetchall()
    }
    for name, (lo, hi) in PROPERTY_RANGE_GATES.items():
        if name not in cols:
            continue
        out[name] = int(
            conn.execute(
                f"""
                SELECT COUNT(*) FROM search_index
                WHERE {name} IS NOT NULL
                  AND ({name} < ? OR {name} > ? OR {name} != {name})
                """,
                [lo, hi],
            ).fetchone()[0]
        )
    for name, abs_max in PROPERTY_ABS_GATES.items():
        if name not in cols:
            continue
        out[name] = int(
            conn.execute(
                f"""
                SELECT COUNT(*) FROM search_index
                WHERE {name} IS NOT NULL
                  AND (abs({name}) > ? OR {name} != {name})
                """,
                [abs_max],
            ).fetchone()[0]
        )
    return out


def count_oor_properties(conn: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """Count OOR ``properties.value_double`` rows per gated name."""
    out: dict[str, int] = {}
    for name, (lo, hi) in PROPERTY_RANGE_GATES.items():
        out[name] = int(
            conn.execute(
                """
                SELECT COUNT(*) FROM properties
                WHERE name = ?
                  AND value_double IS NOT NULL
                  AND (
                    value_double < ?
                    OR value_double > ?
                    OR value_double != value_double
                  )
                """,
                [name, lo, hi],
            ).fetchone()[0]
        )
    for name, abs_max in PROPERTY_ABS_GATES.items():
        out[name] = int(
            conn.execute(
                """
                SELECT COUNT(*) FROM properties
                WHERE name = ?
                  AND value_double IS NOT NULL
                  AND (
                    abs(value_double) > ?
                    OR value_double != value_double
                  )
                """,
                [name, abs_max],
            ).fetchone()[0]
        )
    return out


def sanitize_physical_outliers(conn: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    """Delete OOR property rows and null OOR search_index cells. Idempotent."""
    before_si = count_oor_search_index(conn)
    before_props = count_oor_properties(conn)

    si_nulled: dict[str, int] = {}
    props_deleted: dict[str, int] = {}

    cols = {
        row[1]
        for row in conn.execute("PRAGMA table_info('search_index')").fetchall()
    }

    for name, (lo, hi) in PROPERTY_RANGE_GATES.items():
        # Properties table (skip SI-only names that are not property keys, but
        # youngs / thermal names do exist in properties when enriched).
        before_p = conn.execute(
            """
            SELECT COUNT(*) FROM properties
            WHERE name = ? AND value_double IS NOT NULL
            """,
            [name],
        ).fetchone()[0]
        conn.execute(
            """
            DELETE FROM properties
            WHERE name = ?
              AND value_double IS NOT NULL
              AND (
                value_double < ?
                OR value_double > ?
                OR value_double != value_double
              )
            """,
            [name, lo, hi],
        )
        after_p = conn.execute(
            """
            SELECT COUNT(*) FROM properties
            WHERE name = ? AND value_double IS NOT NULL
            """,
            [name],
        ).fetchone()[0]
        props_deleted[name] = int(before_p) - int(after_p)

        if name not in cols:
            continue
        before_s = conn.execute(
            f"""
            SELECT COUNT(*) FROM search_index
            WHERE {name} IS NOT NULL
              AND ({name} < ? OR {name} > ? OR {name} != {name})
            """,
            [lo, hi],
        ).fetchone()[0]
        conn.execute(
            f"""
            UPDATE search_index
            SET {name} = NULL
            WHERE {name} IS NOT NULL
              AND ({name} < ? OR {name} > ? OR {name} != {name})
            """,
            [lo, hi],
        )
        si_nulled[name] = int(before_s)

    for name, abs_max in PROPERTY_ABS_GATES.items():
        before_p = conn.execute(
            """
            SELECT COUNT(*) FROM properties
            WHERE name = ? AND value_double IS NOT NULL
            """,
            [name],
        ).fetchone()[0]
        conn.execute(
            """
            DELETE FROM properties
            WHERE name = ?
              AND value_double IS NOT NULL
              AND (
                abs(value_double) > ?
                OR value_double != value_double
              )
            """,
            [name, abs_max],
        )
        after_p = conn.execute(
            """
            SELECT COUNT(*) FROM properties
            WHERE name = ? AND value_double IS NOT NULL
            """,
            [name],
        ).fetchone()[0]
        props_deleted[name] = int(before_p) - int(after_p)

        if name not in cols:
            continue
        before_s = conn.execute(
            f"""
            SELECT COUNT(*) FROM search_index
            WHERE {name} IS NOT NULL
              AND (abs({name}) > ? OR {name} != {name})
            """,
            [abs_max],
        ).fetchone()[0]
        conn.execute(
            f"""
            UPDATE search_index
            SET {name} = NULL
            WHERE {name} IS NOT NULL
              AND (abs({name}) > ? OR {name} != {name})
            """,
            [abs_max],
        )
        si_nulled[name] = int(before_s)

    # Clear AGL method/source when main κ was nulled.
    if "thermal_conductivity_method" in cols:
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

    after_si = count_oor_search_index(conn)
    after_props = count_oor_properties(conn)

    return {
        "before_search_index_oor": before_si,
        "before_properties_oor": before_props,
        "search_index_nulled": si_nulled,
        "properties_deleted": props_deleted,
        "after_search_index_oor": after_si,
        "after_properties_oor": after_props,
        "leftover_search_index": sum(after_si.values()),
        "leftover_properties": sum(after_props.values()),
    }
