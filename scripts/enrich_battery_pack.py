"""Attach MP insertion-electrode battery properties onto materials.

Reads ``data/raw/materials_project/insertion_electrodes.jsonl`` and writes
properties onto matching ``materials_project`` source_ids. When a material
appears in multiple electrodes, keeps the entry with highest ``energy_grav``
(fallback: ``capacity_grav``).
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
from database.duckdb import MaterialsDatabase, ensure_search_index_columns  # noqa: E402
from database.write_lock import duckdb_write_lock  # noqa: E402

SOURCE = DataSource.MATERIALS_PROJECT.value
METHOD = "mp-insertion-electrode"
ORIGIN = "database"
BATTERY_NAMES = (
    PropertyName.AVERAGE_VOLTAGE.value,
    PropertyName.CAPACITY_GRAV.value,
    PropertyName.CAPACITY_VOL.value,
    PropertyName.ENERGY_GRAV.value,
    PropertyName.WORKING_ION.value,
    PropertyName.BATTERY_FORMULA.value,
)


def _score(doc: dict[str, Any]) -> float:
    for key in ("energy_grav", "capacity_grav", "average_voltage"):
        val = doc.get(key)
        if val is None:
            continue
        try:
            return float(val)
        except (TypeError, ValueError):
            continue
    return float("-inf")


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return number


def _load_best_by_material(path: Path) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            ids = doc.get("material_ids") or []
            if not isinstance(ids, list):
                continue
            for mid in ids:
                mid_s = str(mid).strip()
                if not mid_s:
                    continue
                prev = best.get(mid_s)
                if prev is None or _score(doc) > _score(prev):
                    best[mid_s] = doc
    return best


def _double_row(
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
        "value_double": value,
        "value_bool": None,
        "value_text": None,
        "unit": PROPERTY_UNITS[name],
        "origin": ORIGIN,
        "method": METHOD,
        "uncertainty": None,
    }


def _text_row(
    *,
    material_id: str,
    source_id: str,
    name: PropertyName,
    value: str,
) -> dict[str, Any]:
    return {
        "source": SOURCE,
        "source_id": source_id,
        "material_id": material_id,
        "name": name.value,
        "value_double": None,
        "value_bool": None,
        "value_text": value,
        "unit": PROPERTY_UNITS[name],
        "origin": ORIGIN,
        "method": METHOD,
        "uncertainty": None,
    }


def main() -> int:
    clear_settings_cache()
    settings = get_settings()
    path = settings.raw_source_dir("materials_project") / "insertion_electrodes.jsonl"
    if not path.exists():
        raise SystemExit(f"Missing {path}; run scripts/download_mp_electrodes.py first.")

    best = _load_best_by_material(path)
    print(f"Unique MP material_ids with electrodes: {len(best):,}")

    with duckdb_write_lock(settings, label="battery-pack"):
        db = MaterialsDatabase(settings.duckdb_path)
        with db.connect() as conn:
            ensure_search_index_columns(conn)
            # Map mp source_id -> material_id
            mp_rows = conn.execute(
                """
                SELECT source_id, material_id
                FROM source_records
                WHERE source = 'materials_project'
                """
            ).fetchall()
            mp_to_mat = {str(sid): str(mid) for sid, mid in mp_rows}

            prop_rows: list[dict[str, Any]] = []
            si_rows: list[dict[str, Any]] = []
            linked = 0
            missing = 0
            for source_id, doc in best.items():
                material_id = mp_to_mat.get(source_id)
                if material_id is None:
                    missing += 1
                    continue
                linked += 1
                avg_v = _finite(doc.get("average_voltage"))
                cap_g = _finite(doc.get("capacity_grav"))
                cap_v = _finite(doc.get("capacity_vol"))
                eng_g = _finite(doc.get("energy_grav"))
                ion = doc.get("working_ion")
                formula = doc.get("battery_formula")
                if avg_v is not None:
                    prop_rows.append(
                        _double_row(
                            material_id=material_id,
                            source_id=source_id,
                            name=PropertyName.AVERAGE_VOLTAGE,
                            value=avg_v,
                        )
                    )
                if cap_g is not None:
                    prop_rows.append(
                        _double_row(
                            material_id=material_id,
                            source_id=source_id,
                            name=PropertyName.CAPACITY_GRAV,
                            value=cap_g,
                        )
                    )
                if cap_v is not None:
                    prop_rows.append(
                        _double_row(
                            material_id=material_id,
                            source_id=source_id,
                            name=PropertyName.CAPACITY_VOL,
                            value=cap_v,
                        )
                    )
                if eng_g is not None:
                    prop_rows.append(
                        _double_row(
                            material_id=material_id,
                            source_id=source_id,
                            name=PropertyName.ENERGY_GRAV,
                            value=eng_g,
                        )
                    )
                if ion:
                    prop_rows.append(
                        _text_row(
                            material_id=material_id,
                            source_id=source_id,
                            name=PropertyName.WORKING_ION,
                            value=str(ion),
                        )
                    )
                if formula:
                    prop_rows.append(
                        _text_row(
                            material_id=material_id,
                            source_id=source_id,
                            name=PropertyName.BATTERY_FORMULA,
                            value=str(formula),
                        )
                    )
                si_rows.append(
                    {
                        "material_id": material_id,
                        "average_voltage": avg_v,
                        "capacity_grav": cap_g,
                        "capacity_vol": cap_v,
                        "energy_grav": eng_g,
                        "working_ion": str(ion) if ion else None,
                    }
                )

            conn.execute(
                f"""
                DELETE FROM properties
                WHERE source = ?
                  AND name IN ({",".join("?" for _ in BATTERY_NAMES)})
                  AND method = ?
                """,
                [SOURCE, *BATTERY_NAMES, METHOD],
            )

            if prop_rows:
                frame = pd.DataFrame(prop_rows)
                conn.register("batt_props", frame)
                conn.execute(
                    """
                    INSERT INTO properties
                    SELECT * FROM batt_props
                    ON CONFLICT (source, source_id, name) DO UPDATE SET
                        material_id = excluded.material_id,
                        value_double = excluded.value_double,
                        value_bool = excluded.value_bool,
                        value_text = excluded.value_text,
                        unit = excluded.unit,
                        origin = excluded.origin,
                        method = excluded.method,
                        uncertainty = excluded.uncertainty
                    """
                )
                conn.unregister("batt_props")

            if si_rows:
                frame = pd.DataFrame(si_rows)
                conn.register("batt_si", frame)
                conn.execute(
                    """
                    UPDATE search_index AS si
                    SET
                        average_voltage = COALESCE(b.average_voltage, si.average_voltage),
                        capacity_grav = COALESCE(b.capacity_grav, si.capacity_grav),
                        capacity_vol = COALESCE(b.capacity_vol, si.capacity_vol),
                        energy_grav = COALESCE(b.energy_grav, si.energy_grav),
                        working_ion = COALESCE(b.working_ion, si.working_ion)
                    FROM batt_si AS b
                    WHERE si.material_id = b.material_id
                    """
                )
                conn.unregister("batt_si")

            counts = dict(
                conn.execute(
                    """
                    SELECT name, COUNT(*) FROM properties
                    WHERE method = ?
                    GROUP BY 1 ORDER BY 1
                    """,
                    [METHOD],
                ).fetchall()
            )
            print(
                f"Battery pack done: linked_materials={linked:,}, "
                f"unmatched_mp_ids={missing:,}, props={counts}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
