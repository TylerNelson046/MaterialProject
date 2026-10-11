"""Backfill search_index thermal_conductivity method/source (Step 1.1).

Idempotent. Prefers AFLOW AGL over element-mix when both exist in properties.
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
from database.thermal_provenance import (  # noqa: E402
    backfill_thermal_conductivity_provenance,
)
from database.write_lock import duckdb_write_lock  # noqa: E402


def main() -> int:
    clear_settings_cache()
    settings = get_settings()
    with duckdb_write_lock(settings, label="thermal-provenance-backfill"):
        db = MaterialsDatabase(settings.duckdb_path)
        with db.connect() as conn:
            stats = backfill_thermal_conductivity_provenance(conn)
    print(json.dumps(stats, indent=2))
    out = settings.processed_dir / "thermal_conductivity_provenance_backfill.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
