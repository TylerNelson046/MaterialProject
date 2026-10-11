"""Repair search_index numeric labels to priority coalesce (Step 3.1 fix).

Rewrites SI values from ``properties`` using primary-then-SOURCE_PRIORITY
winners (never averages). Fixes AFLOW AEL overwrites that preferred AEL over
higher-priority primary values.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config.settings import clear_settings_cache, get_settings  # noqa: E402
from database.duckdb import MaterialsDatabase  # noqa: E402
from database.property_policy import coalesce_property  # noqa: E402
from database.write_lock import duckdb_write_lock  # noqa: E402

# Props where SI must match coalesce_property (elastic was AEL-overwritten).
REPAIR_PROPS = (
    "bulk_modulus_vrh",
    "shear_modulus_vrh",
    "homogeneous_poisson",
    "universal_anisotropy",
    "band_gap",
    "formation_energy_per_atom",
    "energy_above_hull",
    "density",
    "volume",
    "mbj_bandgap",
    "total_magnetization",
)


def main() -> int:
    clear_settings_cache()
    settings = get_settings()
    changed: dict[str, int] = {p: 0 for p in REPAIR_PROPS}

    with duckdb_write_lock(settings, label="repair-si-coalesce"):
        db = MaterialsDatabase(settings.duckdb_path)
        with db.connect() as conn:
            materials = conn.execute(
                """
                SELECT material_id, source
                FROM search_index
                WHERE n_sources > 1
                """
            ).fetchall()
            props = conn.execute(
                f"""
                SELECT material_id, name, source, value_double
                FROM properties
                WHERE name IN ({",".join("?" for _ in REPAIR_PROPS)})
                  AND value_double IS NOT NULL
                """,
                list(REPAIR_PROPS),
            ).fetchall()

            by_mat: dict[str, dict[str, dict[str, float]]] = defaultdict(
                lambda: defaultdict(dict)
            )
            for mid, name, source, value in props:
                by_mat[str(mid)][str(name)][str(source)] = float(value)

            updates: list[dict[str, Any]] = []
            for mid, primary in materials:
                mid = str(mid)
                row: dict[str, Any] = {"material_id": mid}
                prop_map = by_mat.get(mid, {})
                any_set = False
                for prop in REPAIR_PROPS:
                    values = prop_map.get(prop, {})
                    if not values:
                        row[prop] = None
                        continue
                    winner_val, _winner = coalesce_property(
                        values,
                        primary_source=str(primary),
                        property_name=prop,
                    )
                    row[prop] = winner_val
                    any_set = True
                if any_set:
                    updates.append(row)

            if not updates:
                print("No multi-source materials to repair")
                return 0

            # Compare to current SI and count changes.
            frame = pd.DataFrame(updates)
            conn.register("_coalesce_fix", frame)
            for prop in REPAIR_PROPS:
                n = conn.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM search_index AS si
                    JOIN _coalesce_fix AS c ON c.material_id = si.material_id
                    WHERE c.{prop} IS NOT NULL
                      AND (
                        si.{prop} IS NULL
                        OR abs(si.{prop} - c.{prop}) > 1e-9
                      )
                    """
                ).fetchone()[0]
                changed[prop] = int(n)
                conn.execute(
                    f"""
                    UPDATE search_index AS si
                    SET {prop} = c.{prop}
                    FROM _coalesce_fix AS c
                    WHERE si.material_id = c.material_id
                      AND c.{prop} IS NOT NULL
                      AND (
                        si.{prop} IS NULL
                        OR abs(si.{prop} - c.{prop}) > 1e-9
                      )
                    """
                )
            conn.unregister("_coalesce_fix")

    report = {
        "multi_source_materials": len(materials),
        "cells_changed": changed,
        "total_cells_changed": sum(changed.values()),
    }
    out = settings.processed_dir / "search_index_coalesce_repair.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
