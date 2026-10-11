"""Attach MP has_phonon flag from local summary JSONL (phonon C_v not in API)."""

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
METHOD = "mp-has-props"


def _has_phonon(has_props: Any) -> bool | None:
    if has_props is None:
        return None
    if isinstance(has_props, dict):
        if "phonon" in has_props:
            return bool(has_props["phonon"])
        if "phonons" in has_props:
            return bool(has_props["phonons"])
        return None
    if isinstance(has_props, list):
        for item in has_props:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                if item[0] in {"phonon", "phonons"}:
                    return bool(item[1])
            elif item in {"phonon", "phonons"}:
                return True
    return None


def main() -> int:
    # Ensure property exists in enum usage via PropertyName if we added HAS_PHONON
    clear_settings_cache()
    settings = get_settings()
    path = settings.raw_source_dir("materials_project") / "materials.jsonl"
    if not path.exists():
        raise SystemExit(f"Missing {path}")

    flags: dict[str, bool] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            doc = json.loads(line)
            mid = str(doc.get("material_id") or "").strip()
            if not mid:
                continue
            flag = _has_phonon(doc.get("has_props"))
            if flag is not None:
                flags[mid] = flag

    print(f"MP phonon flags parsed: {len(flags):,} (true={sum(1 for v in flags.values() if v):,})")

    with duckdb_write_lock(settings, label="mp-phonon-flag"):
        db = MaterialsDatabase(settings.duckdb_path)
        with db.connect() as conn:
            ensure_search_index_columns(conn)
            # Ensure column exists even if not in SCHEMA (migration)
            cols = {r[1] for r in conn.execute("PRAGMA table_info('search_index')").fetchall()}
            if "has_phonon" not in cols:
                conn.execute("ALTER TABLE search_index ADD COLUMN has_phonon BOOLEAN")

            mp_rows = conn.execute(
                """
                SELECT source_id, material_id FROM source_records
                WHERE source = 'materials_project'
                """
            ).fetchall()
            prop_rows: list[dict[str, Any]] = []
            si_rows: list[dict[str, Any]] = []
            for source_id, material_id in mp_rows:
                if source_id not in flags:
                    continue
                value = flags[source_id]
                prop_rows.append(
                    {
                        "source": SOURCE,
                        "source_id": source_id,
                        "material_id": material_id,
                        "name": "has_phonon",
                        "value_double": None,
                        "value_bool": value,
                        "value_text": None,
                        "unit": "boolean",
                        "origin": "database",
                        "method": METHOD,
                        "uncertainty": None,
                    }
                )
                si_rows.append({"material_id": material_id, "has_phonon": value})

            conn.execute(
                """
                DELETE FROM properties
                WHERE source = ? AND name = 'has_phonon'
                """,
                [SOURCE],
            )
            if prop_rows:
                frame = pd.DataFrame(prop_rows)
                conn.register("ph_props", frame)
                conn.execute(
                    """
                    INSERT INTO properties SELECT * FROM ph_props
                    ON CONFLICT (source, source_id, name) DO UPDATE SET
                        value_bool = excluded.value_bool,
                        method = excluded.method
                    """
                )
                conn.unregister("ph_props")
            if si_rows:
                frame = pd.DataFrame(si_rows)
                conn.register("ph_si", frame)
                conn.execute(
                    """
                    UPDATE search_index AS si
                    SET has_phonon = p.has_phonon
                    FROM ph_si AS p
                    WHERE si.material_id = p.material_id
                    """
                )
                conn.unregister("ph_si")
            n = conn.execute(
                "SELECT COUNT(*) FROM properties WHERE name='has_phonon' AND value_bool"
            ).fetchone()[0]
            print(f"has_phonon=true materials: {n:,}")
            print(
                "Note: MP phonon docs lack searchable C_v/κ in current API; "
                "flag enables filtering materials with phonon computations."
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
