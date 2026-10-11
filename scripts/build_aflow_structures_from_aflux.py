"""Build AFLOW POSCAR files from AFLUX geometry JSON dumps.

Use when aflowlib.duke.edu / aflow.org is unreachable from this host but
geometry pages were fetched elsewhere (e.g. Cursor WebFetch).

Input: JSON array files (one AFLUX page each) or a JSONL of AFLUX rows.
Only writes structures that are still missing on disk, then updates DuckDB.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config.settings import clear_settings_cache, get_settings  # noqa: E402
from database.duckdb import MaterialsDatabase  # noqa: E402
from database.write_lock import duckdb_write_lock  # noqa: E402


def _source_id_from_auid(auid: str) -> str:
    text = str(auid).strip()
    if text.startswith("aflow:"):
        return "aflow-" + text.split(":", 1)[1]
    if text.startswith("aflow-"):
        return text
    return f"aflow-{text}"


def _poscar_from_row(row: dict[str, Any]) -> str:
    from pymatgen.core import Lattice, Structure

    geometry = row["geometry"]
    positions = row["positions_fractional"]
    species = row["species"]
    composition = row["composition"]
    symbols: list[str] = []
    for sp, count in zip(species, composition):
        symbols.extend([str(sp)] * int(count))
    if len(symbols) != len(positions):
        raise ValueError(
            f"{row.get('auid')}: species {len(symbols)} != positions {len(positions)}"
        )
    lattice = Lattice.from_parameters(
        float(geometry[0]),
        float(geometry[1]),
        float(geometry[2]),
        float(geometry[3]),
        float(geometry[4]),
        float(geometry[5]),
    )
    return Structure(lattice, symbols, positions).to(fmt="poscar")


def iter_rows(path: Path) -> Iterator[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return
    if path.suffix == ".jsonl":
        for line in text.splitlines():
            line = line.strip()
            if line:
                yield json.loads(line)
        return
    payload = json.loads(text)
    if isinstance(payload, list):
        for row in payload:
            if isinstance(row, dict):
                yield row
    elif isinstance(payload, dict):
        yield payload


def main(argv: list[str] | None = None) -> int:
    argv = list(argv or sys.argv[1:])
    if not argv:
        raise SystemExit(
            "Usage: build_aflow_structures_from_aflux.py <page.json|rows.jsonl>..."
        )

    clear_settings_cache()
    settings = get_settings()
    settings.computational_structures_dir().mkdir(parents=True, exist_ok=True)

    ok = skip = fail = 0
    path_rows: list[dict] = []
    for arg in argv:
        path = Path(arg)
        if not path.exists():
            print(f"missing input: {path}")
            continue
        for row in iter_rows(path):
            auid = row.get("auid")
            if not auid or not row.get("geometry") or not row.get("positions_fractional"):
                fail += 1
                continue
            source_id = _source_id_from_auid(str(auid))
            dest = settings.computational_structures_dir() / f"{source_id}.vasp"
            rel = str(dest.relative_to(settings.project_root))
            if dest.exists() and dest.stat().st_size > 80:
                skip += 1
                path_rows.append({"source_id": source_id, "structure_path": rel})
                continue
            try:
                text = _poscar_from_row(row)
                dest.write_text(text, encoding="utf-8")
                ok += 1
                path_rows.append({"source_id": source_id, "structure_path": rel})
            except Exception as exc:  # noqa: BLE001
                fail += 1
                print(f"fail {source_id}: {exc}")

    print(f"built ok={ok} skip={skip} fail={fail}")
    if not path_rows:
        return 1 if fail else 0

    frame = pd.DataFrame(path_rows).drop_duplicates("source_id")
    with duckdb_write_lock(settings, label="aflow-aflux-structures"):
        db = MaterialsDatabase(settings.duckdb_path)
        with db.connect() as conn:
            conn.register("aflow_paths", frame)
            conn.execute(
                """
                UPDATE source_records AS s
                SET structure_path = p.structure_path
                FROM aflow_paths AS p
                WHERE s.source='aflow' AND s.source_id=p.source_id
                """
            )
            conn.execute(
                """
                UPDATE search_index AS si
                SET structure_path = p.structure_path
                FROM aflow_paths AS p
                WHERE si.source='aflow' AND si.source_id=p.source_id
                """
            )
            conn.execute(
                """
                UPDATE search_index AS si
                SET structure_path = s.structure_path
                FROM source_records AS s
                WHERE si.material_id=s.material_id
                  AND si.structure_path IS NULL
                  AND s.structure_path IS NOT NULL
                """
            )
            conn.unregister("aflow_paths")
            missing = conn.execute(
                "SELECT COUNT(*) FROM source_records WHERE source='aflow' AND structure_path IS NULL"
            ).fetchone()[0]
            print(f"AFLOW missing structure_path after: {missing:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
