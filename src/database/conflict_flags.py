"""Cross-source label conflict flags on ``search_index`` (Step 3.3).

Flags materials where linked DFT sources disagree beyond tolerance on tracked
labels. Hull conflicts use hull-comparable sources only.
"""

from __future__ import annotations

from typing import Any

import duckdb

from config.constants import CONFLICT_TRACKED_PROPERTIES
from database.duckdb import ensure_search_index_columns
from database.property_policy import HULL_COMPARABLE_SOURCES, conflict_summary

CONFLICT_FLAGS_VERSION = "3.3.0"

# Absolute spread tolerances (flag when spread > tol among ≥2 sources).
CONFLICT_TOLERANCES: dict[str, float] = {
    "band_gap": 0.15,  # eV
    "mbj_bandgap": 0.15,
    "formation_energy_per_atom": 0.05,  # eV/atom
    "energy_above_hull": 0.05,  # eV/atom
    "density": 0.15,  # g/cm³ (matches ER density_abs_tol)
    "volume": 5.0,  # Å³ — loose geometry
    "bulk_modulus_vrh": 20.0,  # GPa
    "shear_modulus_vrh": 20.0,
    "total_magnetization": 0.5,  # μB
}

CONFLICT_FLAG_COLUMNS: tuple[str, ...] = (
    "label_conflict",
    "conflict_band_gap",
    "conflict_formation_energy",
    "conflict_energy_above_hull",
    "conflict_density",
)


def _flag_name(prop: str) -> str | None:
    return {
        "band_gap": "conflict_band_gap",
        "formation_energy_per_atom": "conflict_formation_energy",
        "energy_above_hull": "conflict_energy_above_hull",
        "density": "conflict_density",
    }.get(prop)


def refresh_conflict_flags(conn: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    """Compute conflict flags for all materials. Idempotent."""
    ensure_search_index_columns(conn)

    # Per-material, per-property values from properties (all linked sources).
    # Restrict to materials that appear in properties for tracked names.
    tracked = list(CONFLICT_TRACKED_PROPERTIES)
    placeholders = ", ".join("?" for _ in tracked)

    rows = conn.execute(
        f"""
        SELECT material_id, name, source, value_double
        FROM properties
        WHERE name IN ({placeholders})
          AND value_double IS NOT NULL
        """,
        tracked,
    ).fetchall()

    by_mat: dict[str, dict[str, dict[str, float]]] = {}
    for material_id, name, source, value in rows:
        by_mat.setdefault(str(material_id), {}).setdefault(str(name), {})[
            str(source)
        ] = float(value)

    material_ids = [
        r[0] for r in conn.execute("SELECT material_id FROM search_index").fetchall()
    ]

    updates: list[dict[str, Any]] = []
    for mid in material_ids:
        props = by_mat.get(str(mid), {})
        flags = {col: False for col in CONFLICT_FLAG_COLUMNS}
        any_conflict = False
        for prop, values in props.items():
            tol = CONFLICT_TOLERANCES.get(prop)
            if tol is None:
                continue
            filtered = dict(values)
            if prop == "energy_above_hull":
                filtered = {
                    src: val
                    for src, val in filtered.items()
                    if src in HULL_COMPARABLE_SOURCES
                }
            summary = conflict_summary(filtered)
            if summary is None:
                continue
            if float(summary["spread"]) > tol:
                any_conflict = True
                col = _flag_name(prop)
                if col is not None:
                    flags[col] = True
        flags["label_conflict"] = any_conflict
        flags["material_id"] = mid
        updates.append(flags)

    # Bulk update via temp table.
    import pandas as pd

    frame = pd.DataFrame(updates)
    conn.register("_conflict_flags", frame)
    conn.execute(
        """
        UPDATE search_index AS si
        SET
            label_conflict = f.label_conflict,
            conflict_band_gap = f.conflict_band_gap,
            conflict_formation_energy = f.conflict_formation_energy,
            conflict_energy_above_hull = f.conflict_energy_above_hull,
            conflict_density = f.conflict_density
        FROM _conflict_flags AS f
        WHERE si.material_id = f.material_id
        """
    )
    conn.unregister("_conflict_flags")

    counts = {
        col: int(
            conn.execute(
                f"SELECT COUNT(*) FROM search_index WHERE {col} IS TRUE"
            ).fetchone()[0]
        )
        for col in CONFLICT_FLAG_COLUMNS
    }
    return {
        "conflict_flags_version": CONFLICT_FLAGS_VERSION,
        "tolerances": CONFLICT_TOLERANCES,
        "counts_true": counts,
        "n_materials": len(material_ids),
    }
