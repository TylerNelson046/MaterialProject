"""Enrich DuckDB with element-mix thermal conductivity screening values.

Writes SI ``thermal_conductivity_proxy`` only (never the AGL-backed
``thermal_conductivity`` column). Rejects JARVIS ``nkappa``/``pkappa``.
AFLOW AGL κ is applied separately (see ``download_aflow_thermal.py``).
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
from data_packs.element_nuclear import atom_fraction_weighted  # noqa: E402
from data_packs.element_thermal import THERMAL_CONDUCTIVITY_WMK  # noqa: E402
from database.duckdb import MaterialsDatabase, ensure_search_index_columns  # noqa: E402
from database.write_lock import duckdb_write_lock  # noqa: E402

SOURCE = DataSource.DERIVED.value
METHOD = "element-table-kappa"
NAME = PropertyName.THERMAL_CONDUCTIVITY.value


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


def main() -> int:
    clear_settings_cache()
    settings = get_settings()

    with duckdb_write_lock(settings, label="thermal-pack-element"):
        db = MaterialsDatabase(settings.duckdb_path)
        with db.connect() as conn:
            ensure_search_index_columns(conn)
            rows = conn.execute(
                """
                SELECT si.material_id, s.composition
                FROM search_index AS si
                JOIN source_records AS s
                  ON s.source = si.source AND s.source_id = si.source_id
                """
            ).fetchall()

            prop_rows: list[dict[str, Any]] = []
            si_rows: list[dict[str, Any]] = []
            skipped = 0
            for material_id, composition_raw in rows:
                composition = _parse_composition(composition_raw)
                if not composition:
                    skipped += 1
                    continue
                kappa = atom_fraction_weighted(composition, THERMAL_CONDUCTIVITY_WMK)
                if kappa is None:
                    skipped += 1
                    continue
                derived_id = f"thermal-{material_id}"
                prop_rows.append(
                    {
                        "source": SOURCE,
                        "source_id": derived_id,
                        "material_id": material_id,
                        "name": NAME,
                        "value_double": float(kappa),
                        "value_bool": None,
                        "value_text": None,
                        "unit": PROPERTY_UNITS[PropertyName.THERMAL_CONDUCTIVITY],
                        "origin": "derived",
                        "method": METHOD,
                        "uncertainty": None,
                    }
                )
                si_rows.append(
                    {
                        "material_id": material_id,
                        "thermal_conductivity_proxy": kappa,
                    }
                )

            conn.execute(
                """
                DELETE FROM properties
                WHERE source = ? AND name = ? AND method = ?
                """,
                [SOURCE, NAME, METHOD],
            )
            # Never import JARVIS nkappa into this canonical name.
            conn.execute(
                """
                DELETE FROM properties
                WHERE source = 'jarvis' AND name = ? AND method LIKE '%kappa%'
                """,
                [NAME],
            )

            if prop_rows:
                frame = pd.DataFrame(prop_rows)
                conn.register("therm_props", frame)
                conn.execute("INSERT INTO properties SELECT * FROM therm_props")
                conn.unregister("therm_props")

            if si_rows:
                frame = pd.DataFrame(si_rows)
                conn.register("therm_si", frame)
                # Proxy column only — never write element-mix into thermal_conductivity.
                conn.execute(
                    """
                    UPDATE search_index AS si
                    SET thermal_conductivity_proxy = t.thermal_conductivity_proxy
                    FROM therm_si AS t
                    WHERE si.material_id = t.material_id
                    """
                )
                conn.unregister("therm_si")

            n = conn.execute(
                """
                SELECT COUNT(*) FROM properties
                WHERE name = ? AND method = ?
                """,
                [NAME, METHOD],
            ).fetchone()[0]
            print(f"Thermal pack (element mix) done: kappa={n:,}, skipped={skipped:,}")
            print("Note: JARVIS nkappa/pkappa rejected; AFLOW AGL pull when API is up.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
