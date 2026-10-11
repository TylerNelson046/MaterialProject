"""Enrich corrosion / oxidation screening proxies from element tables."""

from __future__ import annotations

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
from data_packs.element_corrosion import (  # noqa: E402
    CORROSION_RESISTANCE,
    OXIDATION_TENDENCY,
    atom_weighted,
)
from data_packs.radiation_screening import parse_composition  # noqa: E402
from database.duckdb import MaterialsDatabase, ensure_search_index_columns  # noqa: E402
from database.write_lock import duckdb_write_lock  # noqa: E402

SOURCE = DataSource.DERIVED.value
METHOD = "corrosion-proxy-v1"
ORIGIN = "derived"

DERIVED_NAMES = (
    PropertyName.CORROSION_RESISTANCE_SCORE.value,
    PropertyName.OXIDATION_TENDENCY.value,
)


def _double(
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

    with duckdb_write_lock(settings, label="corrosion-pack"):
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
            n_corr = n_ox = 0

            for material_id, composition_raw in rows:
                composition = parse_composition(composition_raw)
                if composition is None:
                    continue
                derived_id = f"corrosion-{material_id}"
                corr = atom_weighted(composition, CORROSION_RESISTANCE)
                ox = atom_weighted(composition, OXIDATION_TENDENCY)
                update: dict[str, Any] = {
                    "material_id": material_id,
                    "corrosion_resistance_score": None,
                    "oxidation_tendency": None,
                }
                if corr is not None:
                    update["corrosion_resistance_score"] = round(corr, 3)
                    prop_rows.append(
                        _double(
                            material_id=material_id,
                            source_id=derived_id,
                            name=PropertyName.CORROSION_RESISTANCE_SCORE,
                            value=corr,
                        )
                    )
                    n_corr += 1
                if ox is not None:
                    update["oxidation_tendency"] = round(ox, 3)
                    prop_rows.append(
                        _double(
                            material_id=material_id,
                            source_id=derived_id,
                            name=PropertyName.OXIDATION_TENDENCY,
                            value=ox,
                        )
                    )
                    n_ox += 1
                if update["corrosion_resistance_score"] is not None or update["oxidation_tendency"] is not None:
                    si_rows.append(update)

            conn.execute(
                f"""
                DELETE FROM properties
                WHERE source = ?
                  AND name IN ({",".join("?" for _ in DERIVED_NAMES)})
                """,
                [SOURCE, *DERIVED_NAMES],
            )
            if prop_rows:
                frame = pd.DataFrame(prop_rows)
                conn.register("corr_props", frame)
                conn.execute("INSERT INTO properties SELECT * FROM corr_props")
                conn.unregister("corr_props")
            if si_rows:
                frame = pd.DataFrame(si_rows)
                conn.register("corr_si", frame)
                conn.execute(
                    """
                    UPDATE search_index AS si
                    SET
                        corrosion_resistance_score = c.corrosion_resistance_score,
                        oxidation_tendency = c.oxidation_tendency
                    FROM corr_si AS c
                    WHERE si.material_id = c.material_id
                    """
                )
                conn.unregister("corr_si")
            print(f"Corrosion pack: resistance={n_corr:,}, oxidation={n_ox:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
