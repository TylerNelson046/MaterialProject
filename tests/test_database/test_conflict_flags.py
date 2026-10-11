"""Tests for label conflict flag refresh (Step 3.3)."""

from __future__ import annotations

import duckdb

from database.conflict_flags import refresh_conflict_flags


def test_refresh_conflict_flags(monkeypatch) -> None:
    conn = duckdb.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE search_index (
            material_id VARCHAR PRIMARY KEY,
            label_conflict BOOLEAN,
            conflict_band_gap BOOLEAN,
            conflict_formation_energy BOOLEAN,
            conflict_energy_above_hull BOOLEAN,
            conflict_density BOOLEAN
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
        ('agree', NULL, NULL, NULL, NULL, NULL),
        ('gap-conflict', NULL, NULL, NULL, NULL, NULL),
        ('hull-aflow-ok', NULL, NULL, NULL, NULL, NULL)
        """
    )
    conn.execute(
        """
        INSERT INTO properties VALUES
        ('materials_project', '1', 'agree', 'band_gap', 1.0, NULL, NULL, 'eV', 'known', 'mp', NULL),
        ('jarvis', '1', 'agree', 'band_gap', 1.05, NULL, NULL, 'eV', 'known', 'j', NULL),
        ('materials_project', '2', 'gap-conflict', 'band_gap', 0.1, NULL, NULL, 'eV', 'known', 'mp', NULL),
        ('jarvis', '2', 'gap-conflict', 'band_gap', 1.5, NULL, NULL, 'eV', 'known', 'j', NULL),
        ('materials_project', '3', 'hull-aflow-ok', 'energy_above_hull', 0.01, NULL, NULL, 'eV/atom', 'known', 'mp', NULL),
        ('aflow', '3', 'hull-aflow-ok', 'energy_above_hull', 5.0, NULL, NULL, 'eV/atom', 'known', 'a', NULL)
        """
    )
    monkeypatch.setattr(
        "database.conflict_flags.ensure_search_index_columns",
        lambda _conn: [],
    )

    report = refresh_conflict_flags(conn)
    assert report["counts_true"]["conflict_band_gap"] == 1
    assert report["counts_true"]["label_conflict"] == 1
    # AFLOW hull must not create a conflict when only one hull-comparable source exists.
    assert report["counts_true"]["conflict_energy_above_hull"] == 0

    rows = {
        r[0]: r[1:]
        for r in conn.execute(
            """
            SELECT material_id, label_conflict, conflict_band_gap,
                   conflict_energy_above_hull
            FROM search_index
            """
        ).fetchall()
    }
    assert rows["agree"] == (False, False, False)
    assert rows["gap-conflict"] == (True, True, False)
    assert rows["hull-aflow-ok"] == (False, False, False)
