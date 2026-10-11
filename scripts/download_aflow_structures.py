"""Download / reconstruct AFLOW structures and backfill structure_path.

Primary path (preferred while aflowlib.duke.edu file hosting is down):
  AFLUX ``geometry`` + ``positions_fractional`` + ``species``/``composition``
  → POSCAR via pymatgen.

Fallback: classic ``CONTCAR.relax`` from the entry AURL host.
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config.settings import clear_settings_cache, get_settings  # noqa: E402
from database.duckdb import MaterialsDatabase  # noqa: E402
from database.write_lock import duckdb_write_lock  # noqa: E402

WORKERS = 8
TIMEOUT = 45
MAX_RETRIES = 3
AFLUX_BASE = "https://aflow.org/API/aflux/"


def _aurl_to_http(aurl: str) -> str:
    text = str(aurl).strip()
    if text.startswith("http://") or text.startswith("https://"):
        return text.rstrip("/")
    if ":" in text:
        host, path = text.split(":", 1)
        return f"http://{host}/{path}".rstrip("/")
    return f"http://aflowlib.duke.edu/{text}".rstrip("/")


def _structure_relpath(settings, source_id: str) -> Path:
    safe = source_id.replace("/", "_")
    return settings.computational_structures_dir() / f"{safe}.vasp"


def _auid_hash(source_id: str, auid: str | None) -> str:
    raw = str(auid or source_id)
    if raw.startswith("aflow:"):
        return raw.split(":", 1)[1]
    if raw.startswith("aflow-"):
        return raw.removeprefix("aflow-")
    return raw


def _poscar_from_aflux(row: dict[str, Any]) -> str:
    from pymatgen.core import Lattice, Structure

    geometry = row.get("geometry")
    positions = row.get("positions_fractional")
    species = row.get("species")
    composition = row.get("composition")
    if not geometry or not positions or not species or not composition:
        raise ValueError("AFLUX row missing geometry/positions/species/composition")
    if len(geometry) < 6:
        raise ValueError(f"Unexpected geometry: {geometry}")

    symbols: list[str] = []
    for sp, count in zip(species, composition):
        symbols.extend([str(sp)] * int(count))
    if len(symbols) != len(positions):
        raise ValueError(
            f"species count {len(symbols)} != positions {len(positions)}"
        )

    lattice = Lattice.from_parameters(
        float(geometry[0]),
        float(geometry[1]),
        float(geometry[2]),
        float(geometry[3]),
        float(geometry[4]),
        float(geometry[5]),
    )
    structure = Structure(lattice, symbols, positions)
    return structure.to(fmt="poscar")


def _fetch_via_aflux(auid_hash: str) -> str:
    """Pull geometry from AFLUX using wildcard auid matcher (colon form is flaky)."""
    summons = (
        f"auid(*{auid_hash}*),geometry,positions_fractional,"
        f"species,composition,natoms,format(json)"
    )
    url = AFLUX_BASE + "?" + summons
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, timeout=TIMEOUT)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list) or not payload:
                raise RuntimeError(f"AFLUX empty for auid(*{auid_hash}*)")
            return _poscar_from_aflux(payload[0])
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(min(2**attempt, 10))
    raise RuntimeError(f"AFLUX structure fetch failed for {auid_hash}: {last_error}")


def _fetch_contcar(aurl: str) -> str:
    base = _aurl_to_http(aurl)
    # Prefer aflow.org mirror of the same path when duke file host is down.
    bases = [base]
    if "aflowlib.duke.edu" in base:
        bases.append(base.replace("aflowlib.duke.edu", "aflow.org"))
        bases.append(base.replace("http://aflowlib.duke.edu", "https://aflow.org"))
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        for host_base in bases:
            for suffix in ("/CONTCAR.relax", "/CONTCAR.relax.vasp", "/CONTCAR.relax1"):
                url = host_base + suffix
                try:
                    response = requests.get(url, timeout=TIMEOUT)
                    if response.status_code != 200:
                        continue
                    text = response.text
                    if len(text) < 80 or "Not Found" in text[:120]:
                        continue
                    return text
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
        time.sleep(min(2**attempt, 10))
    raise RuntimeError(f"CONTCAR fetch failed for {aurl}: {last_error}")


def _fetch_structure(job: dict) -> str:
    errors: list[str] = []
    try:
        return _fetch_via_aflux(job["auid_hash"])
    except Exception as exc:  # noqa: BLE001
        errors.append(f"aflux:{exc}")
    try:
        return _fetch_contcar(job["aurl"])
    except Exception as exc:  # noqa: BLE001
        errors.append(f"contcar:{exc}")
    raise RuntimeError("; ".join(errors))


def main() -> int:
    clear_settings_cache()
    settings = get_settings()
    settings.ensure_directories()
    settings.computational_structures_dir().mkdir(parents=True, exist_ok=True)

    raw_path = settings.raw_source_dir("aflow") / "materials.jsonl"
    if not raw_path.exists():
        raise SystemExit(f"Missing {raw_path}")

    pending: list[dict] = []
    path_rows: list[dict] = []
    with raw_path.open(encoding="utf-8") as handle:
        for line in handle:
            doc = json.loads(line)
            source_id = doc.get("source_id")
            aurl = doc.get("aurl")
            if not source_id or not aurl:
                continue
            dest = _structure_relpath(settings, str(source_id))
            rel = str(dest.relative_to(settings.project_root))
            if dest.exists() and dest.stat().st_size > 80:
                path_rows.append({"source_id": str(source_id), "structure_path": rel})
                continue
            pending.append(
                {
                    "source_id": str(source_id),
                    "aurl": str(aurl),
                    "auid_hash": _auid_hash(str(source_id), doc.get("auid")),
                    "dest": dest,
                    "rel": rel,
                }
            )

    print(
        f"AFLOW structures: {len(path_rows) + len(pending):,} total, "
        f"{len(path_rows):,} on disk, {len(pending):,} to fetch "
        f"(AFLUX geometry first, CONTCAR fallback)",
        flush=True,
    )

    ok = fail = 0

    def _one(job: dict) -> tuple[str, dict | None]:
        try:
            text = _fetch_structure(job)
            job["dest"].parent.mkdir(parents=True, exist_ok=True)
            job["dest"].write_text(text, encoding="utf-8")
            return "ok", {"source_id": job["source_id"], "structure_path": job["rel"]}
        except Exception as exc:  # noqa: BLE001
            return "fail", {"source_id": job["source_id"], "error": str(exc)}

    if pending:
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = [pool.submit(_one, job) for job in pending]
            done = 0
            for fut in as_completed(futures):
                status, payload = fut.result()
                done += 1
                if status == "ok":
                    ok += 1
                    path_rows.append(payload)  # type: ignore[arg-type]
                else:
                    fail += 1
                if done % 200 == 0 or done == len(pending):
                    print(
                        f"  ... {done}/{len(pending)} "
                        f"(ok={ok} fail={fail})",
                        flush=True,
                    )

    print(f"Download finished: ok={ok} fail={fail} on_disk={len(path_rows):,}")
    if not path_rows:
        return 1

    frame = pd.DataFrame(path_rows)
    with duckdb_write_lock(settings, label="aflow-structures"):
        db = MaterialsDatabase(settings.duckdb_path)
        with db.connect() as conn:
            conn.register("aflow_paths", frame[["source_id", "structure_path"]])
            conn.execute(
                """
                UPDATE source_records AS s
                SET structure_path = p.structure_path
                FROM aflow_paths AS p
                WHERE s.source = 'aflow' AND s.source_id = p.source_id
                """
            )
            conn.execute(
                """
                UPDATE search_index AS si
                SET structure_path = p.structure_path
                FROM aflow_paths AS p
                WHERE si.source = 'aflow' AND si.source_id = p.source_id
                """
            )
            conn.execute(
                """
                UPDATE search_index AS si
                SET structure_path = s.structure_path
                FROM source_records AS s
                WHERE si.material_id = s.material_id
                  AND si.structure_path IS NULL
                  AND s.structure_path IS NOT NULL
                """
            )
            conn.unregister("aflow_paths")
            missing = conn.execute(
                """
                SELECT COUNT(*) FROM source_records
                WHERE source='aflow' AND structure_path IS NULL
                """
            ).fetchone()[0]
            with_struct = conn.execute(
                "SELECT COUNT(*) FROM search_index WHERE structure_path IS NOT NULL"
            ).fetchone()[0]
            print(f"AFLOW missing structure_path after: {missing:,}")
            print(f"search_index with structures: {with_struct:,}")

    print("AFLOW structure backfill complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
