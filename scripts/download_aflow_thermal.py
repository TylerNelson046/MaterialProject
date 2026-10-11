"""Download AFLOW AGL thermal (κ, Debye, C_v) and attach to AFLOW materials.

Writes AGL κ into SI ``thermal_conductivity`` (not the proxy column) and stamps
``thermal_conductivity_method`` / ``_source``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config.constants import PROPERTY_UNITS, DataSource, PropertyName  # noqa: E402
from config.settings import clear_settings_cache, get_settings  # noqa: E402
from database.duckdb import MaterialsDatabase, ensure_search_index_columns  # noqa: E402
from database.write_lock import duckdb_write_lock  # noqa: E402
from normalization.normalize_aflow import _canonical_source_id  # noqa: E402

SOURCE = DataSource.AFLOW.value
METHOD = "aflow-agl"
PAGE_SIZE = 200


def _session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    session.headers.update({"Accept": "application/json"})
    return session


def _get(session: requests.Session, base_url: str, summons: str, *, retries: int = 10) -> list[dict]:
    url = f"{base_url.rstrip('/')}/?{summons}"
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = session.get(url, timeout=180, proxies={"http": None, "https": None})
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, list):
                return payload
            raise TypeError(f"Unexpected AFLUX payload: {type(payload)}")
        except Exception as exc:  # noqa: BLE001
            last = exc
            sleep_s = min(20 * attempt, 120)
            print(f"AFLOW AGL request failed ({attempt}/{retries}): {exc}", flush=True)
            time.sleep(sleep_s)
    raise RuntimeError(f"AFLOW AGL failed: {last}")


def download_jsonl(path: Path, *, base_url: str) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    session = _session()
    page = 1
    total = 0
    with path.open("w", encoding="utf-8") as handle:
        while True:
            # (*) requires field present so we skip empty ICSD rows.
            summons = (
                "auid,compound,"
                "agl_thermal_conductivity_300K(*),agl_debye(*),"
                "agl_heat_capacity_Cv_300K(*),"
                f"catalog(icsd),$paging({page},{PAGE_SIZE}),format(json)"
            )
            rows = _get(session, base_url, summons)
            if not rows:
                break
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
                total += 1
            print(f"AFLOW AGL page {page}: +{len(rows):,} (total {total:,})", flush=True)
            if len(rows) < PAGE_SIZE:
                break
            page += 1
    return total


def _finite(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return number


def enrich_from_jsonl(path: Path, settings) -> dict[str, int]:
    by_id: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            doc = json.loads(line)
            auid = str(doc.get("auid") or "").strip()
            if not auid:
                continue
            by_id[_canonical_source_id(auid)] = doc

    with duckdb_write_lock(settings, label="aflow-thermal"):
        db = MaterialsDatabase(settings.duckdb_path)
        with db.connect() as conn:
            ensure_search_index_columns(conn)
            aflow_rows = conn.execute(
                "SELECT source_id, material_id FROM source_records WHERE source='aflow'"
            ).fetchall()
            prop_rows: list[dict[str, Any]] = []
            si_rows: list[dict[str, Any]] = []
            linked = 0
            for source_id, material_id in aflow_rows:
                doc = by_id.get(str(source_id))
                if not doc:
                    continue
                kappa = _finite(doc.get("agl_thermal_conductivity_300K"))
                debye = _finite(doc.get("agl_debye"))
                cv = _finite(doc.get("agl_heat_capacity_Cv_300K"))
                if kappa is not None and not (0.001 <= kappa <= 5000):
                    kappa = None
                if debye is not None and not (1.0 <= debye <= 5000):
                    debye = None
                if cv is not None and not (0.0 <= cv <= 1.0e6):
                    cv = None
                if kappa is None and debye is None and cv is None:
                    continue
                linked += 1
                if kappa is not None:
                    prop_rows.append(
                        {
                            "source": SOURCE,
                            "source_id": source_id,
                            "material_id": material_id,
                            "name": PropertyName.THERMAL_CONDUCTIVITY.value,
                            "value_double": kappa,
                            "value_bool": None,
                            "value_text": None,
                            "unit": PROPERTY_UNITS[PropertyName.THERMAL_CONDUCTIVITY],
                            "origin": "database",
                            "method": METHOD,
                            "uncertainty": None,
                        }
                    )
                if debye is not None:
                    prop_rows.append(
                        {
                            "source": SOURCE,
                            "source_id": source_id,
                            "material_id": material_id,
                            "name": PropertyName.DEBYE_TEMPERATURE.value,
                            "value_double": debye,
                            "value_bool": None,
                            "value_text": None,
                            "unit": PROPERTY_UNITS[PropertyName.DEBYE_TEMPERATURE],
                            "origin": "database",
                            "method": METHOD,
                            "uncertainty": None,
                        }
                    )
                if cv is not None:
                    prop_rows.append(
                        {
                            "source": SOURCE,
                            "source_id": source_id,
                            "material_id": material_id,
                            "name": PropertyName.HEAT_CAPACITY_CV.value,
                            "value_double": cv,
                            "value_bool": None,
                            "value_text": None,
                            "unit": PROPERTY_UNITS[PropertyName.HEAT_CAPACITY_CV],
                            "origin": "database",
                            "method": METHOD,
                            "uncertainty": None,
                        }
                    )
                si_rows.append(
                    {
                        "material_id": material_id,
                        "thermal_conductivity": kappa,
                        "debye_temperature": debye,
                        "heat_capacity_cv": cv,
                    }
                )

            conn.execute(
                """
                DELETE FROM properties
                WHERE source = ? AND method = ?
                  AND name IN (?, ?, ?)
                """,
                [
                    SOURCE,
                    METHOD,
                    PropertyName.THERMAL_CONDUCTIVITY.value,
                    PropertyName.DEBYE_TEMPERATURE.value,
                    PropertyName.HEAT_CAPACITY_CV.value,
                ],
            )
            if prop_rows:
                frame = pd.DataFrame(prop_rows)
                conn.register("agl_props", frame)
                conn.execute(
                    """
                    INSERT INTO properties SELECT * FROM agl_props
                    ON CONFLICT (source, source_id, name) DO UPDATE SET
                        value_double = excluded.value_double,
                        method = excluded.method,
                        unit = excluded.unit,
                        material_id = excluded.material_id
                    """
                )
                conn.unregister("agl_props")
            if si_rows:
                frame = pd.DataFrame(si_rows)
                conn.register("agl_si", frame)
                conn.execute(
                    """
                    UPDATE search_index AS si
                    SET
                        debye_temperature = COALESCE(
                            a.debye_temperature, si.debye_temperature
                        ),
                        heat_capacity_cv = COALESCE(
                            a.heat_capacity_cv, si.heat_capacity_cv
                        )
                    FROM agl_si AS a
                    WHERE si.material_id = a.material_id
                    """
                )
                # Force AGL κ when available (overwrite element-mix).
                conn.execute(
                    """
                    UPDATE search_index AS si
                    SET
                        thermal_conductivity = a.thermal_conductivity,
                        thermal_conductivity_method = ?,
                        thermal_conductivity_source = ?
                    FROM agl_si AS a
                    WHERE si.material_id = a.material_id
                      AND a.thermal_conductivity IS NOT NULL
                    """,
                    [METHOD, SOURCE],
                )
                conn.unregister("agl_si")
            return {"linked": linked, "props": len(prop_rows), "downloaded": len(by_id)}


def main() -> int:
    clear_settings_cache()
    settings = get_settings()
    cfg = settings.database_config.get("aflow", {})
    base_url = str(cfg.get("base_url") or "https://aflow.org/API/aflux/")
    out = settings.raw_source_dir("aflow") / "agl_thermal.jsonl"
    n = download_jsonl(out, base_url=base_url)
    print(f"Downloaded {n:,} AFLOW AGL rows → {out}")
    stats = enrich_from_jsonl(out, settings)
    print(f"AFLOW AGL enrich: {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
