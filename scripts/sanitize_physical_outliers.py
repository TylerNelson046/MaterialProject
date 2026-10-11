"""Sanitize physical outliers in DuckDB (Step 1.3).

Idempotent. Uses ``physical_ranges`` gates — nulls SI cells, deletes OOR
property rows (never clips). Writes a JSON report under data/processed/.
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
from database.duckdb import MaterialsDatabase, ensure_search_index_columns  # noqa: E402
from database.physical_qa import sanitize_physical_outliers  # noqa: E402
from database.write_lock import duckdb_write_lock  # noqa: E402


def main() -> int:
    clear_settings_cache()
    settings = get_settings()
    with duckdb_write_lock(settings, label="physical-qa-sanitize"):
        db = MaterialsDatabase(settings.duckdb_path)
        with db.connect() as conn:
            ensure_search_index_columns(conn)
            report = sanitize_physical_outliers(conn)
    print(json.dumps(report, indent=2))
    out = settings.processed_dir / "physical_qa_sanitize_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    leftover = report["leftover_search_index"] + report["leftover_properties"]
    if leftover:
        print(f"WARNING: {leftover} leftover OOR values remain")
        return 1
    print("Physical QA clean: leftover OOR counts are zero")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
