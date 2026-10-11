"""Backfill Alexandria space groups from stored atoms (enables entity merge)."""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
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

CRYSTAL_SYSTEMS = (
    (1, 2, "Triclinic"),
    (3, 15, "Monoclinic"),
    (16, 74, "Orthorhombic"),
    (75, 142, "Tetragonal"),
    (143, 167, "Trigonal"),
    (168, 194, "Hexagonal"),
    (195, 230, "Cubic"),
)
CHUNK = 2000


def _crystal_system(number: int | None) -> str | None:
    if number is None:
        return None
    for lo, hi, name in CRYSTAL_SYSTEMS:
        if lo <= int(number) <= hi:
            return name
    return None


def _sg_from_atoms(atoms: dict[str, Any]) -> tuple[int | None, str | None, str | None]:
    try:
        from pymatgen.core import Lattice, Structure
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        lattice = Lattice(atoms["lattice_mat"])
        structure = Structure(
            lattice,
            atoms["elements"],
            atoms["coords"],
            coords_are_cartesian=False,
        )
        for symprec in (0.1, 0.01, 0.2):
            try:
                sga = SpacegroupAnalyzer(structure, symprec=symprec)
                number = int(sga.get_space_group_number())
                symbol = str(sga.get_space_group_symbol())
                return number, symbol, _crystal_system(number)
            except Exception:  # noqa: BLE001
                continue
    except Exception:  # noqa: BLE001
        return None, None, None
    return None, None, None


def _worker(payload: tuple[str, dict[str, Any]]) -> dict[str, Any] | None:
    source_id, atoms = payload
    number, symbol, crystal = _sg_from_atoms(atoms)
    if number is None:
        return None
    return {
        "source_id": source_id,
        "space_group_number": number,
        "space_group_symbol": symbol,
        "crystal_system": crystal,
    }


def _iter_jobs(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            doc = json.loads(line)
            sid = str(doc.get("id") or "").strip()
            atoms = doc.get("atoms")
            if sid and isinstance(atoms, dict):
                yield sid, atoms


def main() -> int:
    clear_settings_cache()
    settings = get_settings()
    path = settings.project_root / "data" / "raw" / "alexandria" / "alex_pbe_hull.jsonl"
    if not path.exists():
        raise SystemExit(f"Missing {path}")

    workers = max(1, min(8, os.cpu_count() or 4))
    results: list[dict[str, Any]] = []
    batch: list[tuple[str, dict[str, Any]]] = []
    total = 0

    print(f"Alexandria SG backfill (workers={workers})...", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for job in _iter_jobs(path):
            batch.append(job)
            if len(batch) < CHUNK:
                continue
            for row in pool.map(_worker, batch, chunksize=20):
                total += 1
                if row is not None:
                    results.append(row)
            print(f"  processed {total:,} (filled {len(results):,})", flush=True)
            batch = []
        if batch:
            for row in pool.map(_worker, batch, chunksize=20):
                total += 1
                if row is not None:
                    results.append(row)
            print(f"  processed {total:,} (filled {len(results):,})", flush=True)

    print(f"Filled SG for {len(results):,} / {total:,}", flush=True)
    if not results:
        return 1

    with duckdb_write_lock(settings, label="alexandria-sg"):
        db = MaterialsDatabase(settings.duckdb_path)
        with db.connect() as conn:
            frame = pd.DataFrame(results)
            conn.register("alex_sg", frame)
            conn.execute(
                """
                UPDATE source_records AS s
                SET
                    space_group_number = a.space_group_number,
                    space_group_symbol = a.space_group_symbol,
                    crystal_system = a.crystal_system
                FROM alex_sg AS a
                WHERE s.source = 'alexandria' AND s.source_id = a.source_id
                """
            )
            conn.execute(
                """
                UPDATE search_index AS si
                SET
                    space_group_number = a.space_group_number,
                    space_group_symbol = a.space_group_symbol,
                    crystal_system = a.crystal_system
                FROM alex_sg AS a
                JOIN source_records AS s
                  ON s.source = 'alexandria' AND s.source_id = a.source_id
                WHERE si.material_id = s.material_id
                """
            )
            conn.unregister("alex_sg")
            missing = conn.execute(
                """
                SELECT COUNT(*) FROM source_records
                WHERE source='alexandria' AND space_group_number IS NULL
                """
            ).fetchone()[0]
            print(f"Alexandria still missing SG: {missing:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
