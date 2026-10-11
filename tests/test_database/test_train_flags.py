"""Tests for train-ready flags (Step 1.4)."""

from __future__ import annotations

import duckdb

from database.exclusion_rules import THERMAL_AGL_METHOD
from database.train_flags import refresh_train_flags


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
            deprecated BOOLEAN,
            formation_energy_per_atom DOUBLE,
            energy_above_hull DOUBLE,
            band_gap DOUBLE,
            bulk_modulus_vrh DOUBLE,
            shear_modulus_vrh DOUBLE,
            thermal_conductivity DOUBLE,
            thermal_conductivity_method VARCHAR,
            debye_temperature DOUBLE,
            heat_capacity_cv DOUBLE,
            average_voltage DOUBLE,
            capacity_grav DOUBLE,
            exclude_deprecated BOOLEAN,
            train_ok_thermo BOOLEAN,
            train_ok_gap BOOLEAN,
            train_ok_elastic BOOLEAN,
            train_ok_thermal_agl BOOLEAN,
            train_ok_battery BOOLEAN
        )
        """
    )
    # Good thermo+gap material
    conn.execute(
        """
        INSERT INTO search_index VALUES
        ('good', FALSE, -1.0, 0.01, 1.2, 50.0, 30.0, 2.0, ?, 300.0, 10.0, 3.5, 100.0,
         NULL, NULL, NULL, NULL, NULL, NULL),
        ('deprecated', TRUE, -1.0, 0.01, 1.2, 50.0, 30.0, 2.0, ?, 300.0, 10.0, 3.5, 100.0,
         NULL, NULL, NULL, NULL, NULL, NULL),
        ('no-labels', FALSE, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
         NULL, NULL, NULL, NULL, NULL, NULL)
        """,
        [THERMAL_AGL_METHOD, THERMAL_AGL_METHOD],
    )
    conn.execute(
        """
        INSERT INTO properties VALUES
        ('jarvis', 'j1', 'good', 'bulk_modulus_vrh', 50.0, NULL, NULL, 'GPa',
         'known', 'jarvis-optb88vdw', NULL),
        ('jarvis', 'j1', 'good', 'shear_modulus_vrh', 30.0, NULL, NULL, 'GPa',
         'known', 'jarvis-optb88vdw', NULL),
        ('jarvis', 'j2', 'deprecated', 'bulk_modulus_vrh', 50.0, NULL, NULL, 'GPa',
         'known', 'jarvis-optb88vdw', NULL),
        ('jarvis', 'j2', 'deprecated', 'shear_modulus_vrh', 30.0, NULL, NULL, 'GPa',
         'known', 'jarvis-optb88vdw', NULL)
        """
    )


def test_refresh_train_flags(monkeypatch) -> None:
    conn = duckdb.connect(":memory:")
    _setup(conn)
    monkeypatch.setattr(
        "database.train_flags.ensure_search_index_columns",
        lambda _conn: [],
    )

    report = refresh_train_flags(conn)
    assert report["deprecated_and_train_ok_overlap"] == 0
    assert report["counts_true"]["exclude_deprecated"] == 1
    assert report["counts_true"]["train_ok_thermo"] == 1
    assert report["counts_true"]["train_ok_gap"] == 1
    assert report["counts_true"]["train_ok_elastic"] == 1
    assert report["counts_true"]["train_ok_thermal_agl"] == 1
    assert report["counts_true"]["train_ok_battery"] == 1

    rows = {
        r[0]: r[1:]
        for r in conn.execute(
            """
            SELECT material_id, exclude_deprecated, train_ok_thermo, train_ok_gap,
                   train_ok_elastic, train_ok_thermal_agl, train_ok_battery
            FROM search_index
            """
        ).fetchall()
    }
    assert rows["good"] == (False, True, True, True, True, True)
    assert rows["deprecated"] == (True, False, False, False, False, False)
    assert rows["no-labels"] == (False, False, False, False, False, False)
