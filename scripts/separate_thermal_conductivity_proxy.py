"""Split SI thermal κ into AGL main column vs proxy column (Step 1.2).

Idempotent. After this runs:
  - ``thermal_conductivity`` = AFLOW AGL only (Niche / train-safe)
  - ``thermal_conductivity_proxy`` = element-mix screening values
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
from database.thermal_provenance import separate_proxy_thermal_conductivity  # noqa: E402
from database.write_lock import duckdb_write_lock  # noqa: E402


def main() -> int:
    clear_settings_cache()
    settings = get_settings()
    with duckdb_write_lock(settings, label="thermal-kappa-split"):
        db = MaterialsDatabase(settings.duckdb_path)
        with db.connect() as conn:
            stats = separate_proxy_thermal_conductivity(conn)
    print(json.dumps(stats, indent=2))
    out = settings.processed_dir / "thermal_conductivity_proxy_split.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
