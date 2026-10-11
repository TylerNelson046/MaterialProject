"""Alexandria formation-energy status for Step 2.5.

Current open slice (`alex_pbe_hull.jsonl`) has no formation energies.
This script verifies that fact and writes an explicit N/A report.
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


def _sample_keys(path: Path, *, n: int = 20) -> set[str]:
    keys: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if i >= n:
                break
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            if isinstance(doc, dict):
                keys.update(doc.keys())
    return keys


def main() -> int:
    clear_settings_cache()
    settings = get_settings()
    raw = settings.raw_source_dir("alexandria") / "alex_pbe_hull.jsonl"
    keys = sorted(_sample_keys(raw)) if raw.exists() else []
    formation_like = [
        k
        for k in keys
        if "formation" in k.lower() or k.lower() in {"e_form", "ef", "delta_e"}
    ]

    db = MaterialsDatabase(settings.duckdb_path, read_only=True)
    with db.connect() as conn:
        si_ef = int(
            conn.execute(
                """
                SELECT COUNT(*) FROM search_index
                WHERE source = 'alexandria' AND formation_energy_per_atom IS NOT NULL
                """
            ).fetchone()[0]
        )
        prop_ef = int(
            conn.execute(
                """
                SELECT COUNT(*) FROM properties
                WHERE source = 'alexandria' AND name = 'formation_energy_per_atom'
                """
            ).fetchone()[0]
        )
        alex_n = int(
            conn.execute(
                "SELECT COUNT(*) FROM search_index WHERE source = 'alexandria'"
            ).fetchone()[0]
        )

    status = "N/A — no open E_f in current slice"
    report = {
        "step": "2.5",
        "status": status,
        "raw_slice": str(raw) if raw.exists() else None,
        "raw_exists": raw.exists(),
        "sample_keys": keys,
        "formation_like_keys_in_sample": formation_like,
        "alexandria_primary_materials": alex_n,
        "search_index_formation_energy_fill": si_ef,
        "properties_formation_energy_rows": prop_ef,
        "decision": (
            "Do not invent E_f from total/corrected energies. "
            "Revisit only if a richer Alexandria dump with formation energies is obtained."
        ),
    }
    out = settings.processed_dir / "alexandria_formation_energy_status.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
