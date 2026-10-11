"""Enrich DuckDB with nuclear screening properties (item: nuclear pack).

Derives per-material:
  - neutron_absorption_cross_section (barns) — atom-fraction weighted elemental σ_a
  - melting_point_element_mix (K) — atom-fraction weighted elemental T_m

These are screening estimates from element tables (not DFT). Written as
``derived`` properties linked to each material's primary search_index source.
"""

from __future__ import annotations

import json
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
from data_packs.element_nuclear import (  # noqa: E402
    MELTING_POINT_K,
    THERMAL_ABSORPTION_BARNS,
    atom_fraction_weighted,
)
from database.duckdb import MaterialsDatabase, ensure_search_index_columns  # noqa: E402
from database.write_lock import duckdb_write_lock  # noqa: E402

SOURCE = DataSource.DERIVED.value
METHOD = "element-table-nist"
ORIGIN = "derived"


def _parse_composition(raw: Any) -> dict[str, float] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None
    if not isinstance(raw, dict) or not raw:
        return None
    out: dict[str, float] = {}
    for key, val in raw.items():
        try:
            amount = float(val)
        except (TypeError, ValueError):
            return None
        if amount > 0:
            out[str(key)] = amount
    return out or None


def _prop_row(
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


def main() -> int:
    clear_settings_cache()
    settings = get_settings()

    with duckdb_write_lock(settings, label="nuclear-pack"):
        db = MaterialsDatabase(settings.duckdb_path)
        with db.connect() as conn:
            ensure_search_index_columns(conn)

            # Prefer primary search_index composition via joined source_records.
            rows = conn.execute(
                """
                SELECT
                    si.material_id,
                    si.source,
                    si.source_id,
                    s.composition
                FROM search_index AS si
                JOIN source_records AS s
                  ON s.source = si.source AND s.source_id = si.source_id
                """
            ).fetchall()

            prop_rows: list[dict[str, Any]] = []
            si_updates: list[dict[str, Any]] = []
            skipped = 0
            for material_id, primary_source, primary_id, composition_raw in rows:
                composition = _parse_composition(composition_raw)
                if not composition:
                    skipped += 1
                    continue
                sigma = atom_fraction_weighted(composition, THERMAL_ABSORPTION_BARNS)
                t_m = atom_fraction_weighted(composition, MELTING_POINT_K)
                if sigma is None and t_m is None:
                    skipped += 1
                    continue
                derived_id = f"nuclear-{material_id}"
                update: dict[str, Any] = {
                    "material_id": material_id,
                    "neutron_absorption_cross_section": None,
                    "melting_point_element_mix": None,
                }
                if sigma is not None:
                    prop_rows.append(
                        _prop_row(
                            material_id=material_id,
                            source_id=derived_id,
                            name=PropertyName.NEUTRON_ABSORPTION_CROSS_SECTION,
                            value=sigma,
                        )
                    )
                    update["neutron_absorption_cross_section"] = sigma
                if t_m is not None:
                    prop_rows.append(
                        _prop_row(
                            material_id=material_id,
                            source_id=derived_id,
                            name=PropertyName.MELTING_POINT_ELEMENT_MIX,
                            value=t_m,
                        )
                    )
                    update["melting_point_element_mix"] = t_m
                si_updates.append(update)

            # Replace prior nuclear-pack derived rows.
            conn.execute(
                """
                DELETE FROM properties
                WHERE source = ?
                  AND name IN (?, ?)
                """,
                [
                    SOURCE,
                    PropertyName.NEUTRON_ABSORPTION_CROSS_SECTION.value,
                    PropertyName.MELTING_POINT_ELEMENT_MIX.value,
                ],
            )

            if prop_rows:
                frame = pd.DataFrame(prop_rows)
                conn.register("nuclear_props", frame)
                conn.execute("INSERT INTO properties SELECT * FROM nuclear_props")
                conn.unregister("nuclear_props")

            if si_updates:
                frame = pd.DataFrame(si_updates)
                conn.register("nuclear_si", frame)
                conn.execute(
                    """
                    UPDATE search_index AS si
                    SET
                        neutron_absorption_cross_section = n.neutron_absorption_cross_section,
                        melting_point_element_mix = n.melting_point_element_mix
                    FROM nuclear_si AS n
                    WHERE si.material_id = n.material_id
                    """
                )
                conn.unregister("nuclear_si")

            n_sigma = conn.execute(
                """
                SELECT COUNT(*) FROM properties
                WHERE name = ? AND source = ?
                """,
                [PropertyName.NEUTRON_ABSORPTION_CROSS_SECTION.value, SOURCE],
            ).fetchone()[0]
            n_melt = conn.execute(
                """
                SELECT COUNT(*) FROM properties
                WHERE name = ? AND source = ?
                """,
                [PropertyName.MELTING_POINT_ELEMENT_MIX.value, SOURCE],
            ).fetchone()[0]
            print(
                f"Nuclear pack done: neutron σ_a={n_sigma:,}, "
                f"T_m mix={n_melt:,}, skipped={skipped:,}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
