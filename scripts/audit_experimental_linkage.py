"""Experimental structure linkage audit (COD + AFLOW-ICSD).

ICSD bulk CIF download requires a paid license — not available here.
AFLOW's ICSD *catalog* summaries are already ingested (structures via AFLOW).
COD experimental CIFs are local. This script reports coverage and writes
``data/processed/experimental_linkage_audit.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config.settings import clear_settings_cache, get_settings  # noqa: E402
from database.duckdb import MaterialsDatabase  # noqa: E402


def main() -> int:
    clear_settings_cache()
    settings = get_settings()
    db = MaterialsDatabase(settings.duckdb_path)
    report: dict = {
        "icsd_license": {
            "status": "not_available",
            "note": (
                "Full ICSD CIF dump requires an institutional/commercial license. "
                "Not ingested. AFLOW ICSD-catalog entries provide computed props on "
                "ICSD-linked prototypes; COD provides open experimental structures."
            ),
        }
    }
    with db.connect() as conn:
        for source in ("cod", "aflow"):
            total = conn.execute(
                "SELECT COUNT(*) FROM source_records WHERE source = ?", [source]
            ).fetchone()[0]
            with_struct = conn.execute(
                """
                SELECT COUNT(*) FROM source_records
                WHERE source = ?
                  AND structure_path IS NOT NULL AND structure_path != ''
                """,
                [source],
            ).fetchone()[0]
            missing_sg = conn.execute(
                """
                SELECT COUNT(*) FROM source_records
                WHERE source = ? AND space_group_number IS NULL
                """,
                [source],
            ).fetchone()[0]
            report[source] = {
                "source_records": total,
                "with_structure_path": with_struct,
                "structure_coverage_pct": round(100.0 * with_struct / total, 2) if total else 0.0,
                "missing_space_group_number": missing_sg,
            }

        # How many search materials have at least one experimental-ish source
        exp = conn.execute(
            """
            SELECT COUNT(DISTINCT material_id) FROM source_records
            WHERE source IN ('cod', 'aflow')
            """
        ).fetchone()[0]
        report["materials_with_cod_or_aflow"] = exp
        report["materials_total"] = conn.execute(
            "SELECT COUNT(*) FROM materials"
        ).fetchone()[0]

    out = settings.project_root / "data" / "processed" / "experimental_linkage_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
