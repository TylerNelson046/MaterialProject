"""Refresh search_index train-ready boolean flags (Step 1.4).

Idempotent. Writes counts to data/processed/train_flags_report.json.
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
from database.train_flags import refresh_train_flags  # noqa: E402
from database.write_lock import duckdb_write_lock  # noqa: E402


def main() -> int:
    clear_settings_cache()
    settings = get_settings()
    with duckdb_write_lock(settings, label="train-flags"):
        db = MaterialsDatabase(settings.duckdb_path)
        with db.connect() as conn:
            report = refresh_train_flags(conn)
    print(json.dumps(report, indent=2))
    out = settings.processed_dir / "train_flags_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    if report["deprecated_and_train_ok_overlap"]:
        print(
            "ERROR: deprecated materials marked train_ok_* "
            f"({report['deprecated_and_train_ok_overlap']})"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
