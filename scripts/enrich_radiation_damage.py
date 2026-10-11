"""Enrich radiation-damage susceptibility proxies (E_d mix + screening score)."""

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
from data_packs.element_radiation_damage import (  # noqa: E402
    displacement_energy_mix,
    radiation_damage_susceptibility,
)
from data_packs.radiation_screening import parse_composition  # noqa: E402
from database.duckdb import MaterialsDatabase, ensure_search_index_columns  # noqa: E402
from database.write_lock import duckdb_write_lock  # noqa: E402

SOURCE = DataSource.DERIVED.value
METHOD = "radiation-damage-proxy-v1"
ORIGIN = "derived"

DERIVED_NAMES = (
    PropertyName.DISPLACEMENT_ENERGY_MIX.value,
    PropertyName.RADIATION_DAMAGE_SUSCEPTIBILITY.value,
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

    with duckdb_write_lock(settings, label="radiation-damage"):
        db = MaterialsDatabase(settings.duckdb_path)
        with db.connect() as conn:
            ensure_search_index_columns(conn)
            rows = conn.execute(
                """
                SELECT
                    si.material_id,
                    s.composition,
                    si.density,
                    si.bulk_modulus_vrh
                FROM search_index AS si
                JOIN source_records AS s
                  ON s.source = si.source AND s.source_id = si.source_id
                """
            ).fetchall()

            prop_rows: list[dict[str, Any]] = []
            si_rows: list[dict[str, Any]] = []
            n_ed = n_sus = 0

            for material_id, composition_raw, density, bulk in rows:
                composition = parse_composition(composition_raw)
                if composition is None:
                    continue
                derived_id = f"rad-dmg-{material_id}"
                ed = displacement_energy_mix(composition)
                sus = radiation_damage_susceptibility(
                    ed_mix=ed,
                    density=float(density) if density is not None else None,
                    bulk_modulus=float(bulk) if bulk is not None else None,
                )
                update: dict[str, Any] = {
                    "material_id": material_id,
                    "displacement_energy_mix": None,
                    "radiation_damage_susceptibility": None,
                }
                if ed is not None:
                    update["displacement_energy_mix"] = round(float(ed), 3)
                    prop_rows.append(
                        _double(
                            material_id=material_id,
                            source_id=derived_id,
                            name=PropertyName.DISPLACEMENT_ENERGY_MIX,
                            value=ed,
                        )
                    )
                    n_ed += 1
                if sus is not None:
                    update["radiation_damage_susceptibility"] = sus
                    prop_rows.append(
                        _double(
                            material_id=material_id,
                            source_id=derived_id,
                            name=PropertyName.RADIATION_DAMAGE_SUSCEPTIBILITY,
                            value=sus,
                        )
                    )
                    n_sus += 1
                if update["displacement_energy_mix"] is not None or update[
                    "radiation_damage_susceptibility"
                ] is not None:
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
                conn.register("rd_props", frame)
                conn.execute("INSERT INTO properties SELECT * FROM rd_props")
                conn.unregister("rd_props")
            if si_rows:
                frame = pd.DataFrame(si_rows)
                conn.register("rd_si", frame)
                conn.execute(
                    """
                    UPDATE search_index AS si
                    SET
                        displacement_energy_mix = r.displacement_energy_mix,
                        radiation_damage_susceptibility = r.radiation_damage_susceptibility
                    FROM rd_si AS r
                    WHERE si.material_id = r.material_id
                    """
                )
                conn.unregister("rd_si")
            print(f"Radiation damage pack: E_d={n_ed:,}, susceptibility={n_sus:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
