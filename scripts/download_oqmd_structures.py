"""Download OQMD structures (unit_cell + sites) and backfill structure_path.

Re-paginates the on-hull OQMD slice **with sites included**, writes pymatgen
Structure JSON under ``data/structures/computational/oqmd-*.json``, then updates
DuckDB ``structure_path`` columns.

Resumable via ``data/raw/oqmd/structures_checkpoint.json`` (last successful
offset) and by skipping existing structure files. Survives OQMD 502/timeouts
through the client retry loop.
"""

from __future__ import annotations

import argparse
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
from databases.oqmd import OQMDClient  # noqa: E402

PAGE_SIZE = 50
CHECKPOINT_NAME = "structures_checkpoint.json"


def _parse_oqmd_site(site: str) -> tuple[str, list[float]]:
    text = str(site).strip()
    if "@" not in text:
        raise ValueError(f"Unexpected OQMD site format: {site!r}")
    element, rest = text.split("@", 1)
    coords = [float(x) for x in rest.split()]
    if len(coords) != 3:
        raise ValueError(f"Expected 3 fractional coords in site {site!r}")
    el = element.strip()
    if not el:
        raise ValueError(f"Missing element in site {site!r}")
    return el, coords


def structure_from_oqmd(doc: dict[str, Any]):
    from pymatgen.core import Lattice, Structure

    unit_cell = doc.get("unit_cell")
    sites = doc.get("sites")
    if not unit_cell or not sites:
        raise ValueError("OQMD row missing unit_cell or sites")

    species: list[str] = []
    frac_coords: list[list[float]] = []
    for site in sites:
        element, xyz = _parse_oqmd_site(site)
        species.append(element)
        frac_coords.append(xyz)

    return Structure(Lattice(unit_cell), species, frac_coords)


def _structure_path(settings, source_id: str) -> Path:
    safe = source_id.replace("/", "_")
    return settings.computational_structures_dir() / f"{safe}.json"


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"offset": 0, "wrote": 0, "skipped": 0, "failed": 0}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_checkpoint(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _backfill_structure_paths(settings, path_rows: list[dict[str, str]]) -> None:
    """Update DuckDB structure_path from on-disk OQMD files."""
    if not path_rows:
        return
    uniq = {row["source_id"]: row["structure_path"] for row in path_rows}
    frame = pd.DataFrame(
        [{"source_id": k, "structure_path": v} for k, v in uniq.items()]
    )
    with duckdb_write_lock(settings, label="oqmd-structures"):
        db = MaterialsDatabase(settings.duckdb_path)
        with db.connect() as conn:
            conn.register("oqmd_paths", frame)
            conn.execute(
                """
                UPDATE source_records AS s
                SET structure_path = p.structure_path
                FROM oqmd_paths AS p
                WHERE s.source = 'oqmd' AND s.source_id = p.source_id
                """
            )
            conn.execute(
                """
                UPDATE search_index AS si
                SET structure_path = p.structure_path
                FROM oqmd_paths AS p
                WHERE si.source = 'oqmd' AND si.source_id = p.source_id
                """
            )
            conn.execute(
                """
                UPDATE search_index AS si
                SET structure_path = s.structure_path
                FROM source_records AS s
                WHERE si.material_id = s.material_id
                  AND s.source = 'oqmd'
                  AND (si.structure_path IS NULL OR si.structure_path = '')
                  AND s.structure_path IS NOT NULL
                """
            )
            conn.unregister("oqmd_paths")
            with_struct = conn.execute(
                """
                SELECT COUNT(*) FROM source_records
                WHERE source='oqmd'
                  AND structure_path IS NOT NULL AND structure_path!=''
                """
            ).fetchone()[0]
            missing = conn.execute(
                """
                SELECT COUNT(*) FROM source_records
                WHERE source='oqmd'
                  AND (structure_path IS NULL OR structure_path='')
                """
            ).fetchone()[0]
            print(
                f"OQMD source_records with structure_path: {with_struct:,} "
                f"(still missing: {missing:,})",
                flush=True,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download OQMD structures + backfill DB.")
    parser.add_argument("--max-materials", type=int, default=None)
    parser.add_argument("--page-size", type=int, default=PAGE_SIZE)
    parser.add_argument(
        "--skip-db-update",
        action="store_true",
        help="Only write structure files; do not update DuckDB.",
    )
    parser.add_argument(
        "--reset-checkpoint",
        action="store_true",
        help="Ignore saved offset and start from 0 (still skips existing files).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    clear_settings_cache()
    settings = get_settings()
    settings.ensure_directories()
    settings.computational_structures_dir().mkdir(parents=True, exist_ok=True)

    raw_dir = settings.raw_source_dir("oqmd")
    checkpoint_path = raw_dir / CHECKPOINT_NAME
    checkpoint = (
        {"offset": 0, "wrote": 0, "skipped": 0, "failed": 0}
        if args.reset_checkpoint
        else _load_checkpoint(checkpoint_path)
    )

    client = OQMDClient(settings)
    # OQMD frequently returns 502s for long stretches; wait it out.
    available = None
    for wait_round in range(1, 10_000):
        try:
            available = client.count_available()
            break
        except Exception as exc:  # noqa: BLE001
            sleep_s = min(60 * wait_round, 600)
            print(
                f"OQMD unavailable ({exc}); retry availability in {sleep_s}s "
                f"(round {wait_round})",
                flush=True,
            )
            import time

            time.sleep(sleep_s)
    if available is None:
        raise RuntimeError("OQMD never became available")

    start_offset = int(checkpoint.get("offset") or 0)
    print(
        f"OQMD structures: available={available:,} "
        f"stability_max={client.stability_max} page_size={args.page_size} "
        f"resume_offset={start_offset:,}",
        flush=True,
    )

    path_rows: list[dict[str, str]] = []
    # Seed path_rows from files already on disk so DB backfill covers prior runs.
    for existing in settings.computational_structures_dir().glob("oqmd-*.json"):
        if existing.stat().st_size > 80:
            path_rows.append(
                {
                    "source_id": existing.stem,
                    "structure_path": str(existing.relative_to(settings.project_root)),
                }
            )

    seen: set[str] = {row["source_id"] for row in path_rows}
    wrote = skipped = failed = 0
    failed_samples: list[str] = []
    offset = start_offset
    processed_this_run = 0

    # Manual pagination so we can checkpoint the offset after each successful page.
    while True:
        if args.max_materials is not None and processed_this_run >= args.max_materials:
            break
        if offset >= available:
            break

        this_limit = args.page_size
        if args.max_materials is not None:
            this_limit = min(this_limit, args.max_materials - processed_this_run)

        # Outer retry around each page — client already retries internally,
        # but a full max-retry failure should not abort the whole job.
        payload = None
        for page_round in range(1, 1000):
            try:
                payload = client._get(
                    client._page_params(
                        limit=this_limit,
                        offset=offset,
                        include_sites=True,
                    )
                )
                break
            except Exception as exc:  # noqa: BLE001
                sleep_s = min(60 * page_round, 600)
                print(
                    f"Page offset={offset} failed ({exc}); "
                    f"retry in {sleep_s}s (round {page_round})",
                    flush=True,
                )
                import time

                time.sleep(sleep_s)
        if payload is None:
            raise RuntimeError(f"Could not fetch OQMD page at offset {offset}")

        rows = payload.get("data") or []
        if not rows:
            break

        for doc in rows:
            if not isinstance(doc, dict) or doc.get("entry_id") is None:
                continue
            source_id = f"oqmd-{doc['entry_id']}"
            processed_this_run += 1
            if source_id in seen:
                skipped += 1
                continue
            seen.add(source_id)

            dest = _structure_path(settings, source_id)
            rel = str(dest.relative_to(settings.project_root))
            if dest.exists() and dest.stat().st_size > 80:
                skipped += 1
                path_rows.append({"source_id": source_id, "structure_path": rel})
                continue

            try:
                structure = structure_from_oqmd(doc)
                dest.write_text(structure.to(fmt="json"), encoding="utf-8")
                wrote += 1
                path_rows.append({"source_id": source_id, "structure_path": rel})
            except Exception as exc:  # noqa: BLE001
                failed += 1
                if len(failed_samples) < 20:
                    failed_samples.append(f"{source_id}: {exc}")

            if args.max_materials is not None and processed_this_run >= args.max_materials:
                break

        offset += len(rows)
        checkpoint = {
            "offset": offset,
            "wrote": int(checkpoint.get("wrote") or 0) + wrote,
            "skipped": int(checkpoint.get("skipped") or 0) + skipped,
            "failed": int(checkpoint.get("failed") or 0) + failed,
            "available": available,
        }
        # Reset per-page counters into checkpoint accumulators carefully:
        # rewrite checkpoint with absolute totals from this process view.
        checkpoint = {
            "offset": offset,
            "wrote_this_run": wrote,
            "skipped_this_run": skipped,
            "failed_this_run": failed,
            "on_disk": len(path_rows),
            "available": available,
        }
        _save_checkpoint(checkpoint_path, checkpoint)

        if offset % 2000 < args.page_size or offset >= available:
            print(
                f"  ... offset={offset:,}/{available:,} "
                f"(wrote={wrote:,} skipped={skipped:,} fail={failed:,} "
                f"on_disk={len(path_rows):,})",
                flush=True,
            )

        # Periodically backfill DB so partial progress is searchable without
        # waiting for the full ~97k download to finish.
        if not args.skip_db_update and (
            offset % 2500 < args.page_size or offset >= available
        ):
            _backfill_structure_paths(settings, path_rows)

        next_link = (payload.get("links") or {}).get("next")
        if not next_link:
            break

    print(
        f"Download finished: wrote={wrote:,} skipped={skipped:,} fail={failed:,} "
        f"on_disk={len(path_rows):,} checkpoint_offset={offset:,}",
        flush=True,
    )
    if failed_samples:
        print("Sample failures:")
        for line in failed_samples:
            print(f"  {line}")

    manifest = {
        "source": "oqmd",
        "structures_wrote_this_run": wrote,
        "structures_skipped_this_run": skipped,
        "structures_failed_this_run": failed,
        "structures_on_disk": len(path_rows),
        "checkpoint_offset": offset,
        "available": available,
    }
    (raw_dir / "structures_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    if args.skip_db_update:
        return 0 if path_rows else 1
    if not path_rows:
        print("No structures on disk yet — skipping DB update (API may be down).")
        return 1

    _backfill_structure_paths(settings, path_rows)
    print("OQMD structure backfill complete.")
    return 0 if wrote > 0 or path_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
