"""Tests for thermal conductivity SI provenance and proxy/DFT split."""

from __future__ import annotations

import duckdb

from database.exclusion_rules import THERMAL_AGL_METHOD, THERMAL_PROXY_METHOD
from database.thermal_provenance import (
    backfill_thermal_conductivity_provenance,
    separate_proxy_thermal_conductivity,
)


def _setup(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE TABLE properties (
            source VARCHAR,
            source_id VARCHAR,
            material_id VARCHAR,
            name VARCHAR,
            value_double DOUBLE,
            value_bool BOOLEAN,
            value_text VARCHAR,
            unit VARCHAR,
            origin VARCHAR,
            method VARCHAR,
            uncertainty DOUBLE
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE search_index (
            material_id VARCHAR PRIMARY KEY,
            thermal_conductivity DOUBLE,
            thermal_conductivity_method VARCHAR,
            thermal_conductivity_source VARCHAR,
            thermal_conductivity_proxy DOUBLE
        )
        """
    )
    conn.execute(
        """
        INSERT INTO properties VALUES
        ('derived', 'thermal-mat-agl', 'mat-agl', 'thermal_conductivity',
         10.0, NULL, NULL, 'W/m-K', 'derived', ?, NULL),
        ('aflow', 'aflow-1', 'mat-agl', 'thermal_conductivity',
         3.0, NULL, NULL, 'W/m-K', 'database', ?, NULL),
        ('derived', 'thermal-mat-proxy', 'mat-proxy', 'thermal_conductivity',
         12.0, NULL, NULL, 'W/m-K', 'derived', ?, NULL)
        """,
        [THERMAL_PROXY_METHOD, THERMAL_AGL_METHOD, THERMAL_PROXY_METHOD],
    )
    # Pre-split state: mixed values live in thermal_conductivity.
    conn.execute(
        """
        INSERT INTO search_index VALUES
        ('mat-agl', 3.0, NULL, NULL, NULL),
        ('mat-proxy', 12.0, NULL, NULL, NULL),
        ('mat-empty', NULL, 'stale', 'stale', NULL)
        """
    )


def test_backfill_prefers_agl_and_clears_null_kappa(monkeypatch) -> None:
    conn = duckdb.connect(":memory:")
    _setup(conn)
    monkeypatch.setattr(
        "database.thermal_provenance.ensure_thermal_provenance_columns",
        lambda _conn: [],
    )

    stats = backfill_thermal_conductivity_provenance(conn)
    assert stats["agl"] == 1
    assert stats["proxy"] == 1
    assert stats["unlabeled"] == 0
    assert stats["cleared_null_kappa_provenance"] == 1

    rows = {
        r[0]: (r[1], r[2], r[3])
        for r in conn.execute(
            """
            SELECT material_id, thermal_conductivity,
                   thermal_conductivity_method, thermal_conductivity_source
            FROM search_index
            """
        ).fetchall()
    }
    assert rows["mat-agl"] == (3.0, THERMAL_AGL_METHOD, "aflow")
    assert rows["mat-proxy"] == (12.0, THERMAL_PROXY_METHOD, "derived")
    assert rows["mat-empty"] == (None, None, None)


def test_separate_proxy_moves_mix_off_main_column(monkeypatch) -> None:
    conn = duckdb.connect(":memory:")
    _setup(conn)
    monkeypatch.setattr(
        "database.thermal_provenance.ensure_thermal_provenance_columns",
        lambda _conn: [],
    )

    stats = separate_proxy_thermal_conductivity(conn)
    assert stats["kappa_agl"] == 1
    assert stats["proxy_col"] == 2
    assert stats["method_proxy_on_main"] == 0
    assert stats["kappa_not_tagged_agl"] == 0
    assert stats["cleared_proxy_from_main"] == 1
    assert stats["both_agl_and_proxy"] == 1

    rows = {
        r[0]: (r[1], r[2], r[3], r[4])
        for r in conn.execute(
            """
            SELECT material_id, thermal_conductivity,
                   thermal_conductivity_method, thermal_conductivity_source,
                   thermal_conductivity_proxy
            FROM search_index
            """
        ).fetchall()
    }
    assert rows["mat-agl"] == (3.0, THERMAL_AGL_METHOD, "aflow", 10.0)
    assert rows["mat-proxy"] == (None, None, None, 12.0)
    assert rows["mat-empty"] == (None, None, None, None)
