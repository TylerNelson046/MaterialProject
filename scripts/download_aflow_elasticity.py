"""Download/update AFLOW AEL elasticity for ICSD catalog (bulk + shear).

Use when aflow.org is reachable. Upserts bulk/shear onto existing AFLOW rows.
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
from normalization.physical_ranges import sanitize_elastic  # noqa: E402

SOURCE = DataSource.AFLOW.value
METHOD = "aflow-ael"
PAGE = 200


def _session() -> requests.Session:
    s = requests.Session()
    s.trust_env = False
    return s


def _get(session: requests.Session, base: str, summons: str, retries: int = 6) -> list[dict]:
    url = f"{base.rstrip('/')}/?{summons}"
    last = None
    for attempt in range(1, retries + 1):
        try:
            r = session.get(url, timeout=180, proxies={"http": None, "https": None})
            r.raise_for_status()
            payload = r.json()
            if isinstance(payload, list):
                return payload
            raise TypeError(type(payload))
        except Exception as exc:  # noqa: BLE001
            last = exc
            print(f"AFLOW AEL failed ({attempt}/{retries}): {exc}", flush=True)
            time.sleep(min(30 * attempt, 120))
    raise RuntimeError(last)


def main() -> int:
    clear_settings_cache()
    settings = get_settings()
    cfg = settings.database_config.get("aflow", {})
    base = str(cfg.get("base_url") or "https://aflow.org/API/aflux/")
    out = settings.raw_source_dir("aflow") / "ael_elastic.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    session = _session()
    page = 1
    total = 0
    with out.open("w", encoding="utf-8") as handle:
        while True:
            summons = (
                "auid,ael_bulk_modulus_vrh(*),ael_shear_modulus_vrh(*),"
                f"catalog(icsd),$paging({page},{PAGE}),format(json)"
            )
            rows = _get(session, base, summons)
            if not rows:
                break
            for row in rows:
                handle.write(json.dumps(row) + "\n")
                total += 1
            print(f"AEL page {page}: +{len(rows)} (total {total})", flush=True)
            if len(rows) < PAGE:
                break
            page += 1

    by_id: dict[str, dict] = {}
    with out.open(encoding="utf-8") as handle:
        for line in handle:
            doc = json.loads(line)
            auid = str(doc.get("auid") or "").strip()
            if auid:
                by_id[_canonical_source_id(auid)] = doc

    with duckdb_write_lock(settings, label="aflow-ael"):
        db = MaterialsDatabase(settings.duckdb_path)
        with db.connect() as conn:
            ensure_search_index_columns(conn)
            aflow = conn.execute(
                "SELECT source_id, material_id FROM source_records WHERE source='aflow'"
            ).fetchall()
            props: list[dict[str, Any]] = []
            si: list[dict[str, Any]] = []
            for sid, mid in aflow:
                doc = by_id.get(str(sid))
                if not doc:
                    continue
                b = sanitize_elastic(doc.get("ael_bulk_modulus_vrh"))
                g = sanitize_elastic(doc.get("ael_shear_modulus_vrh"))
                if b is None and g is None:
                    continue
                if b is not None:
                    props.append(
                        {
                            "source": SOURCE,
                            "source_id": sid,
                            "material_id": mid,
                            "name": PropertyName.BULK_MODULUS_VRH.value,
                            "value_double": b,
                            "value_bool": None,
                            "value_text": None,
                            "unit": PROPERTY_UNITS[PropertyName.BULK_MODULUS_VRH],
                            "origin": "database",
                            "method": METHOD,
                            "uncertainty": None,
                        }
                    )
                if g is not None:
                    props.append(
                        {
                            "source": SOURCE,
                            "source_id": sid,
                            "material_id": mid,
                            "name": PropertyName.SHEAR_MODULUS_VRH.value,
                            "value_double": g,
                            "value_bool": None,
                            "value_text": None,
                            "unit": PROPERTY_UNITS[PropertyName.SHEAR_MODULUS_VRH],
                            "origin": "database",
                            "method": METHOD,
                            "uncertainty": None,
                        }
                    )
                si.append({"material_id": mid, "bulk_modulus_vrh": b, "shear_modulus_vrh": g})

            if props:
                frame = pd.DataFrame(props)
                conn.register("ael_props", frame)
                conn.execute(
                    """
                    INSERT INTO properties SELECT * FROM ael_props
                    ON CONFLICT (source, source_id, name) DO UPDATE SET
                        value_double = excluded.value_double,
                        method = excluded.method,
                        material_id = excluded.material_id
                    """
                )
                conn.unregister("ael_props")
            if si:
                frame = pd.DataFrame(si)
                conn.register("ael_si", frame)
                conn.execute(
                    """
                    UPDATE search_index AS si
                    SET
                        bulk_modulus_vrh = COALESCE(
                            si.bulk_modulus_vrh, a.bulk_modulus_vrh
                        ),
                        shear_modulus_vrh = COALESCE(
                            si.shear_modulus_vrh, a.shear_modulus_vrh
                        )
                    FROM ael_si AS a
                    WHERE si.material_id = a.material_id
                    """
                )
                conn.unregister("ael_si")
            print(f"AFLOW AEL enrich linked rows: {len(si):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
