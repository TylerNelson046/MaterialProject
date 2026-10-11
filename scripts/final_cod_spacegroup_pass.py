"""Final COD space-group recovery pass (Step 2.4).

Re-runs recoverable SG fill from raw COD + local CIFs, then exports the
accepted permanent-null set for the dataset card / plan.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "scripts"))

from config.settings import clear_settings_cache, get_settings  # noqa: E402
from database.duckdb import MaterialsDatabase, ensure_search_index_columns  # noqa: E402
from database.write_lock import duckdb_write_lock  # noqa: E402
from repair_dataset_quality_6_10 import repair_cod_space_groups  # noqa: E402


def main() -> int:
    clear_settings_cache()
    settings = get_settings()
    with duckdb_write_lock(settings, label="cod-sg-final"):
        db = MaterialsDatabase(settings.duckdb_path)
        with db.connect() as conn:
            ensure_search_index_columns(conn)
            before = int(
                conn.execute(
                    """
                    SELECT COUNT(*) FROM source_records
                    WHERE source = 'cod' AND space_group_number IS NULL
                    """
                ).fetchone()[0]
            )
            stats = repair_cod_space_groups(conn, settings=settings)
            after = int(
                conn.execute(
                    """
                    SELECT COUNT(*) FROM source_records
                    WHERE source = 'cod' AND space_group_number IS NULL
                    """
                ).fetchone()[0]
            )
            permanent = conn.execute(
                """
                SELECT source_id, formula_pretty, space_group_symbol, structure_path
                FROM source_records
                WHERE source = 'cod' AND space_group_number IS NULL
                ORDER BY source_id
                """
            ).fetchdf()

            # Sync SI from source_records for COD-primary rows still missing SG.
            conn.execute(
                """
                UPDATE search_index AS si
                SET
                    space_group_number = s.space_group_number,
                    space_group_symbol = COALESCE(si.space_group_symbol, s.space_group_symbol),
                    crystal_system = COALESCE(si.crystal_system, s.crystal_system)
                FROM source_records AS s
                WHERE si.source = 'cod'
                  AND s.source = 'cod'
                  AND s.source_id = si.source_id
                  AND si.space_group_number IS NULL
                  AND s.space_group_number IS NOT NULL
                """
            )
            si_null = int(
                conn.execute(
                    """
                    SELECT COUNT(*) FROM search_index
                    WHERE source = 'cod' AND space_group_number IS NULL
                    """
                ).fetchone()[0]
            )

    nulls_path = settings.processed_dir / "cod_permanent_sg_nulls.jsonl"
    with nulls_path.open("w", encoding="utf-8") as handle:
        for row in permanent.to_dict(orient="records"):
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    report = {
        "before_sr_sg_null": before,
        "repair_stats": stats,
        "after_sr_sg_null": after,
        "after_si_cod_primary_sg_null": si_null,
        "permanent_nulls_path": str(nulls_path),
        "accepted_permanent_nulls": after,
        "note": (
            "Residual COD space_group_number nulls are accepted as unrecoverable "
            "(CIF/analyzer/symbol failures)."
        ),
    }
    out = settings.processed_dir / "cod_sg_final_pass_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {out}")
    print(f"Wrote {nulls_path} ({after} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
