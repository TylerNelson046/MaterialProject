"""Export Alexandria structures from alex_pbe_hull.jsonl and backfill structure_path.

Alexandria on-hull slice has no E_f field — only total/corrected energies already
stored as energy_per_atom. Structure files enable viewing / structure ML.
"""

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

from config.settings import clear_settings_cache, get_settings  # noqa: E402
from database.duckdb import MaterialsDatabase  # noqa: E402
from database.write_lock import duckdb_write_lock  # noqa: E402
from normalization.normalize_alexandria import iter_raw_alexandria_documents  # noqa: E402


def _structure_from_atoms(atoms: dict[str, Any]):
    from pymatgen.core import Lattice, Structure

    return Structure(
        Lattice(atoms["lattice_mat"]),
        atoms["elements"],
        atoms["coords"],
        coords_are_cartesian=False,
    )


def main() -> int:
    clear_settings_cache()
    settings = get_settings()
    jsonl = settings.project_root / "data" / "raw" / "alexandria" / "alex_pbe_hull.jsonl"
    out_dir = settings.computational_structures_dir() / "alexandria"
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    path_rows: list[dict[str, str]] = []

    for doc in iter_raw_alexandria_documents(jsonl):
        source_id = str(doc.get("id") or "").strip()
        atoms = doc.get("atoms")
        if not source_id or not isinstance(atoms, dict):
            skipped += 1
            continue
        dest = out_dir / f"alexandria-{source_id}.json"
        rel = str(dest.relative_to(settings.project_root))
        if dest.exists() and dest.stat().st_size > 50:
            path_rows.append({"source_id": source_id, "structure_path": rel})
            skipped += 1
            continue
        try:
            structure = _structure_from_atoms(atoms)
            dest.write_text(structure.to_json(), encoding="utf-8")
            written += 1
            path_rows.append({"source_id": source_id, "structure_path": rel})
        except Exception as exc:  # noqa: BLE001
            print(f"skip {source_id}: {exc}", flush=True)
            skipped += 1
        if (written + skipped) % 10000 == 0:
            print(
                f"Alexandria structures progress: written={written:,} "
                f"kept/skipped={skipped:,}",
                flush=True,
            )

    print(f"Alexandria structures on disk: written={written:,}, reused/skipped={skipped:,}")

    if not path_rows:
        return 0

    with duckdb_write_lock(settings, label="alexandria-structures"):
        db = MaterialsDatabase(settings.duckdb_path)
        with db.connect() as conn:
            frame = pd.DataFrame(path_rows)
            conn.register("alex_paths", frame)
            conn.execute(
                """
                UPDATE source_records AS s
                SET structure_path = p.structure_path
                FROM alex_paths AS p
                WHERE s.source = 'alexandria' AND s.source_id = p.source_id
                """
            )
            # Fill search_index when empty (any linked alexandria row).
            conn.execute(
                """
                UPDATE search_index AS si
                SET structure_path = s.structure_path
                FROM source_records AS s
                WHERE s.source = 'alexandria'
                  AND s.material_id = si.material_id
                  AND s.structure_path IS NOT NULL
                  AND (si.structure_path IS NULL OR si.structure_path = '')
                """
            )
            # Prefer Alexandria path when search_index primary source is alexandria.
            conn.execute(
                """
                UPDATE search_index AS si
                SET structure_path = s.structure_path
                FROM source_records AS s
                WHERE si.source = 'alexandria'
                  AND s.source = 'alexandria'
                  AND s.source_id = si.source_id
                  AND s.structure_path IS NOT NULL
                """
            )
            conn.unregister("alex_paths")
            n = conn.execute(
                """
                SELECT COUNT(*) FROM source_records
                WHERE source='alexandria'
                  AND structure_path IS NOT NULL AND structure_path != ''
                """
            ).fetchone()[0]
            print(f"Alexandria source_records with structure_path: {n:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
