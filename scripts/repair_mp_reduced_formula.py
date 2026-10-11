"""Backfill Materials Project reduced_formula strings in DuckDB.

Replaces stringified list/dict reprs with canonical reduced formulas
(e.g. ``LiEr2Ir``), preserving material_id / entity-resolution links.
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

from config.settings import clear_settings_cache, get_settings  # noqa: E402
from database.duckdb import MaterialsDatabase  # noqa: E402
from database.write_lock import duckdb_write_lock  # noqa: E402
from normalization.normalize_mp import _reduced_formula_string  # noqa: E402


def main() -> int:
    clear_settings_cache()
    settings = get_settings()
    db = MaterialsDatabase(settings.duckdb_path)

    print("Loading MP source_records for reduced_formula repair...")
    with db.connect() as conn:
        frame = conn.execute(
            """
            SELECT source_id, formula_pretty, composition, reduced_formula
            FROM source_records
            WHERE source = 'materials_project'
            """
        ).fetchdf()

    before_bad = int(
        frame["reduced_formula"].fillna("").astype(str).str.startswith("[").sum()
        + frame["reduced_formula"].fillna("").astype(str).str.startswith("{").sum()
    )
    print(f"MP rows: {len(frame):,}  bad reduced_formula: {before_bad:,}")

    updates: list[dict] = []
    failed = 0
    for row in frame.itertuples(index=False):
        composition = row.composition
        if isinstance(composition, str):
            try:
                composition = json.loads(composition)
            except json.JSONDecodeError:
                composition = None
        reduced = _reduced_formula_string(
            composition=composition,
            formula_pretty=row.formula_pretty,
        )
        if not reduced or str(reduced).startswith(("[", "{")):
            failed += 1
            continue
        updates.append(
            {
                "source_id": row.source_id,
                "reduced_formula": reduced,
            }
        )

    print(f"Prepared {len(updates):,} updates ({failed} failed)")
    if not updates:
        return 1

    upd = pd.DataFrame(updates)
    with duckdb_write_lock(settings, label="mp-reduced-formula-repair"):
        with db.connect() as conn:
            conn.register("mp_rf", upd)
            conn.execute(
                """
                UPDATE source_records AS s
                SET reduced_formula = m.reduced_formula
                FROM mp_rf AS m
                WHERE s.source = 'materials_project'
                  AND s.source_id = m.source_id
                """
            )
            conn.unregister("mp_rf")

            after_bad = conn.execute(
                """
                SELECT COUNT(*) FROM source_records
                WHERE source = 'materials_project'
                  AND (
                    reduced_formula LIKE '[%'
                    OR reduced_formula LIKE '{%'
                    OR reduced_formula IS NULL
                    OR reduced_formula = ''
                  )
                """
            ).fetchone()[0]
            sample = conn.execute(
                """
                SELECT source_id, formula_pretty, reduced_formula
                FROM source_records
                WHERE source = 'materials_project'
                LIMIT 5
                """
            ).fetchall()
            print(f"MP bad reduced_formula after: {after_bad:,}")
            print("sample:", sample)

    print("MP reduced_formula repair complete.")
    return 0 if after_bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
