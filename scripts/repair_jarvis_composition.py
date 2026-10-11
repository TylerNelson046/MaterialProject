"""Backfill JARVIS composition/elements/nelements in DuckDB in place.

Preserves material_id links from entity resolution. Re-parses formulas from
raw JSONL (or existing formula_pretty) and updates source_records,
material_elements, materials, and search_index.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config.constants import DataSource  # noqa: E402
from config.settings import clear_settings_cache, get_settings  # noqa: E402
from database.duckdb import MaterialsDatabase  # noqa: E402
from database.write_lock import duckdb_write_lock  # noqa: E402
from normalization.normalize_jarvis import _composition_from_jarvis  # noqa: E402


def main() -> int:
    clear_settings_cache()
    settings = get_settings()
    raw_path = settings.raw_source_dir(DataSource.JARVIS) / "materials.jsonl"
    if not raw_path.exists():
        raise SystemExit(f"Missing {raw_path}")

    print(f"Reading formulas from {raw_path}...")
    rows: list[dict] = []
    with raw_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            jid = str(doc.get("jid") or "").strip()
            if not jid:
                continue
            comp = _composition_from_jarvis(doc)
            rows.append(
                {
                    "source_id": jid,
                    "formula_pretty": comp.formula_pretty,
                    "reduced_formula": comp.reduced_formula,
                    "chemsys": comp.chemsys,
                    "elements": json.dumps(list(comp.elements)),
                    "nelements": comp.nelements,
                    "composition": json.dumps(comp.composition),
                    "elements_list": list(comp.elements),
                }
            )
    print(f"Parsed {len(rows):,} JARVIS compositions")
    empty = sum(1 for r in rows if not r["elements_list"])
    if empty:
        print(f"WARNING: {empty} still have empty elements")
    else:
        print("All rows have non-empty elements")

    frame = pd.DataFrame(rows)

    with duckdb_write_lock(settings, label="jarvis-composition-repair"):
        db = MaterialsDatabase(settings.duckdb_path)
        with db.connect() as conn:
            before = conn.execute(
                """
                SELECT COUNT(*) FROM source_records
                WHERE source = 'jarvis'
                  AND (elements IS NULL OR elements = '[]' OR nelements IS NULL)
                """
            ).fetchone()[0]
            print(f"JARVIS rows needing repair before: {before:,}")

            conn.register("jarvis_fix", frame.drop(columns=["elements_list"]))
            conn.execute(
                """
                UPDATE source_records AS s
                SET
                    formula_pretty = j.formula_pretty,
                    reduced_formula = j.reduced_formula,
                    chemsys = j.chemsys,
                    elements = j.elements,
                    nelements = j.nelements,
                    composition = j.composition
                FROM jarvis_fix AS j
                WHERE s.source = 'jarvis' AND s.source_id = j.source_id
                """
            )
            conn.unregister("jarvis_fix")

            # Rebuild material_elements for materials that include a JARVIS source.
            print("Rebuilding material_elements for JARVIS-linked materials...")
            jarvis_mats = [
                r[0]
                for r in conn.execute(
                    "SELECT DISTINCT material_id FROM source_records WHERE source = 'jarvis'"
                ).fetchall()
            ]
            # Delete existing element rows for those materials, then re-insert
            # from ALL sources on the material (not just JARVIS).
            chunk = 1000
            for i in range(0, len(jarvis_mats), chunk):
                batch = jarvis_mats[i : i + chunk]
                placeholders = ",".join(["?"] * len(batch))
                conn.execute(
                    f"DELETE FROM material_elements WHERE material_id IN ({placeholders})",
                    batch,
                )

            # Collect elements from every source_record on those materials.
            element_rows: list[dict] = []
            src_rows = conn.execute(
                """
                SELECT material_id, elements
                FROM source_records
                WHERE material_id IN (
                    SELECT DISTINCT material_id FROM source_records WHERE source = 'jarvis'
                )
                """
            ).fetchall()
            seen: set[tuple[str, str]] = set()
            for material_id, elements_raw in src_rows:
                parsed: list[str] = []
                if isinstance(elements_raw, str):
                    try:
                        loaded = json.loads(elements_raw)
                        if isinstance(loaded, list):
                            parsed = [str(x) for x in loaded]
                    except json.JSONDecodeError:
                        parsed = []
                for el in parsed:
                    key = (material_id, el)
                    if key not in seen:
                        seen.add(key)
                        element_rows.append({"material_id": material_id, "element": el})
            if element_rows:
                MaterialsDatabase._df_replace(conn, "material_elements", element_rows)

            # Update materials + search_index identity fields when primary is JARVIS
            # or when the material row still has null nelements.
            print("Updating materials / search_index nelements + elements...")
            conn.execute(
                """
                UPDATE materials AS m
                SET
                    formula_pretty = s.formula_pretty,
                    chemsys = s.chemsys,
                    nelements = s.nelements
                FROM source_records AS s
                WHERE s.source = 'jarvis'
                  AND s.material_id = m.material_id
                  AND (
                    m.nelements IS NULL
                    OR EXISTS (
                        SELECT 1 FROM search_index si
                        WHERE si.material_id = m.material_id AND si.source = 'jarvis'
                    )
                  )
                """
            )
            # Prefer JARVIS source_record values only when search_index primary is jarvis.
            conn.execute(
                """
                UPDATE search_index AS si
                SET
                    formula_pretty = s.formula_pretty,
                    chemsys = s.chemsys,
                    elements = s.elements,
                    nelements = s.nelements
                FROM source_records AS s
                WHERE si.source = 'jarvis'
                  AND s.source = 'jarvis'
                  AND si.source_id = s.source_id
                  AND si.material_id = s.material_id
                """
            )
            # Also fill null nelements on multi-source materials whose primary isn't jarvis
            # but nelements was never set — use any non-null source on the material.
            conn.execute(
                """
                UPDATE search_index AS si
                SET
                    elements = s.elements,
                    nelements = s.nelements,
                    chemsys = COALESCE(si.chemsys, s.chemsys)
                FROM source_records AS s
                WHERE si.nelements IS NULL
                  AND s.material_id = si.material_id
                  AND s.nelements IS NOT NULL
                  AND s.source = (
                    SELECT s2.source FROM source_records s2
                    WHERE s2.material_id = si.material_id AND s2.nelements IS NOT NULL
                    ORDER BY CASE s2.source
                      WHEN 'materials_project' THEN 0
                      WHEN 'jarvis' THEN 1
                      WHEN 'oqmd' THEN 2
                      WHEN 'aflow' THEN 3
                      ELSE 9
                    END
                    LIMIT 1
                  )
                """
            )

            after = conn.execute(
                """
                SELECT COUNT(*) FROM source_records
                WHERE source = 'jarvis'
                  AND (elements IS NULL OR elements = '[]' OR nelements IS NULL)
                """
            ).fetchone()[0]
            si_null = conn.execute(
                "SELECT COUNT(*) FROM search_index WHERE nelements IS NULL"
            ).fetchone()[0]
            print(f"JARVIS rows still broken: {after:,}")
            print(f"search_index nelements null: {si_null:,}")

            sample = conn.execute(
                """
                SELECT source_id, formula_pretty, chemsys, elements, nelements
                FROM source_records WHERE source='jarvis' LIMIT 3
                """
            ).fetchall()
            print("sample:", sample)

    print("JARVIS composition repair complete.")
    return 0 if after == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
