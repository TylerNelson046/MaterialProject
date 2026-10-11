"""Tests for physical QA sanitization (Step 1.3)."""

from __future__ import annotations

import duckdb

from database.physical_qa import sanitize_physical_outliers


def test_sanitize_nulls_si_and_deletes_properties() -> None:
    conn = duckdb.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE search_index (
            material_id VARCHAR PRIMARY KEY,
            density DOUBLE,
            band_gap DOUBLE,
            youngs_modulus_vrh DOUBLE,
            universal_anisotropy DOUBLE,
            thermal_conductivity DOUBLE,
            thermal_conductivity_method VARCHAR,
            thermal_conductivity_source VARCHAR
        )
        """
    )
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
        INSERT INTO search_index VALUES
        ('ok', 5.0, 1.0, 100.0, 2.0, 3.0, 'aflow-agl', 'aflow'),
        ('bad-density', 50.0, 1.0, 100.0, 2.0, NULL, NULL, NULL),
        ('bad-youngs', 5.0, 1.0, 5000.0, 2.0, NULL, NULL, NULL),
        ('bad-aniso', 5.0, 1.0, 100.0, 1e6, NULL, NULL, NULL),
        ('bad-kappa', 5.0, 1.0, 100.0, 2.0, 9000.0, 'aflow-agl', 'aflow')
        """
    )
    conn.execute(
        """
        INSERT INTO properties VALUES
        ('mp', '1', 'ok', 'density', 5.0, NULL, NULL, 'g/cm^3', 'known', 'mp-gga-pbe', NULL),
        ('mp', '2', 'bad-density', 'density', 50.0, NULL, NULL, 'g/cm^3', 'known', 'mp-gga-pbe', NULL),
        ('aflow', '3', 'bad-kappa', 'thermal_conductivity', 9000.0, NULL, NULL, 'W/m-K',
         'database', 'aflow-agl', NULL)
        """
    )

    report = sanitize_physical_outliers(conn)
    assert report["leftover_search_index"] == 0
    assert report["leftover_properties"] == 0
    assert report["search_index_nulled"]["density"] == 1
    assert report["search_index_nulled"]["youngs_modulus_vrh"] == 1
    assert report["search_index_nulled"]["universal_anisotropy"] == 1
    assert report["search_index_nulled"]["thermal_conductivity"] == 1
    assert report["properties_deleted"]["density"] == 1
    assert report["properties_deleted"]["thermal_conductivity"] == 1

    rows = {
        r[0]: r[1:]
        for r in conn.execute(
            """
            SELECT material_id, density, youngs_modulus_vrh, universal_anisotropy,
                   thermal_conductivity, thermal_conductivity_method
            FROM search_index
            """
        ).fetchall()
    }
    assert rows["ok"][0] == 5.0
    assert rows["bad-density"][0] is None
    assert rows["bad-youngs"][1] is None
    assert rows["bad-aniso"][2] is None
    assert rows["bad-kappa"][3] is None
    assert rows["bad-kappa"][4] is None  # method cleared with κ

    # Idempotent second pass.
    report2 = sanitize_physical_outliers(conn)
    assert report2["leftover_search_index"] == 0
    assert sum(report2["search_index_nulled"].values()) == 0
    assert sum(report2["properties_deleted"].values()) == 0
