"""Fully automated OQMD pipeline: small test → verify → full pull → load DB."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config.constants import DataSource  # noqa: E402
from config.settings import clear_settings_cache, get_settings  # noqa: E402
from database.duckdb import MaterialsDatabase  # noqa: E402
from database.entity_resolution import resolve_entities  # noqa: E402
from database.queries import MaterialSearchFilters, search_materials  # noqa: E402
from database.write_lock import duckdb_write_lock  # noqa: E402
from ingestion.ingest_oqmd import ingest_oqmd  # noqa: E402
from normalization.normalize_oqmd import (  # noqa: E402
    iter_load_normalized_oqmd,
    normalize_oqmd_document,
)
from pipeline.search import SearchRequest, run_search  # noqa: E402

LOG_PATH = ROOT / "data" / "raw" / "oqmd" / "pipeline.log"
TEST_COUNT = 100


def log(message: str) -> None:
    stamp = datetime.now(UTC).replace(microsecond=0).isoformat()
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def verify_oqmd_search(*, expect_min: int) -> None:
    hits = search_materials(
        MaterialSearchFilters(source=DataSource.OQMD.value, limit=5)
    )
    if len(hits) < 1:
        raise RuntimeError("OQMD search returned no rows")

    result = run_search(
        SearchRequest(
            source=DataSource.OQMD.value,
            energy_above_hull_max=0.1,
            limit=5,
        )
    )
    log(
        f"Verify search: source=oqmd sample={len(hits)} "
        f"near-hull={result.count} first={hits[0].get('source_id')}"
    )

    db = MaterialsDatabase(read_only=True)
    count = db.count_by_source(DataSource.OQMD.value)
    log(f"Verify DB OQMD count={count} (expect >= {expect_min})")
    if count < expect_min:
        raise RuntimeError(f"Expected >= {expect_min} OQMD rows, found {count}")


def unique_oqmd_source_ids() -> int:
    """Count unique source_ids in the raw OQMD JSONL (deduped entry_ids)."""
    settings = get_settings()
    path = settings.raw_source_dir(DataSource.OQMD) / "materials.jsonl"
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            sid = doc.get("source_id")
            if not sid and doc.get("entry_id") is not None:
                sid = f"oqmd-{doc['entry_id']}"
            if sid:
                seen.add(str(sid))
    return len(seen)


def load_oqmd_into_db(*, replace: bool) -> int:
    with duckdb_write_lock(label="oqmd"):
        db = MaterialsDatabase()
        db.initialize()
        if replace:
            removed = db.delete_source(DataSource.OQMD.value)
            log(f"Removed {removed} prior OQMD materials")
        start = db.next_material_index()
        log(f"Inserting OQMD from MAT index {start}")
        written = db.insert_materials(
            iter_load_normalized_oqmd(start_index=start),
            batch_size=2000,
        )
        log(f"Inserted {written} OQMD materials; total DB={db.count_materials()}")
        return written


def smoke_normalize_first_record() -> None:
    settings = get_settings()
    path = settings.raw_source_dir(DataSource.OQMD) / "materials.jsonl"
    first = path.read_text(encoding="utf-8").splitlines()[0]
    doc = json.loads(first)
    material = normalize_oqmd_document(doc, index=1, settings=settings)
    assert material.primary_source.source == DataSource.OQMD
    assert material.primary_source.source_id.startswith("oqmd-")
    log(
        f"Normalize smoke OK: {material.material_id} "
        f"{material.primary_source.source_id} {material.formula_pretty}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OQMD ingest + DB load pipeline.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip Phase 1 and resume Phase 2 ingest from existing JSONL.",
    )
    parser.add_argument(
        "--skip-entity-resolution",
        action="store_true",
        help="Do not re-run entity resolution after the full DB load.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    clear_settings_cache()
    settings = get_settings()
    settings.ensure_directories()
    if not args.resume:
        LOG_PATH.write_text("", encoding="utf-8")

    try:
        log("=== OQMD pipeline start ===" + (" (resume)" if args.resume else ""))

        if not args.resume:
            log(f"Phase 1: ingest test slice (max_materials={TEST_COUNT})")
            manifest = ingest_oqmd(settings=settings, max_materials=TEST_COUNT)
            log(f"Test ingest manifest count={manifest['count']}")
            if manifest["count"] < TEST_COUNT:
                raise RuntimeError("Test ingest returned fewer records than requested")

            smoke_normalize_first_record()
            written = load_oqmd_into_db(replace=True)
            if written < TEST_COUNT:
                raise RuntimeError("Test DB load wrote fewer records than expected")
            verify_oqmd_search(expect_min=TEST_COUNT)
            log("Phase 1 PASSED")

        log("Phase 2: ingest on-hull OQMD slice (stability<=0)")
        full_manifest = ingest_oqmd(
            settings=settings,
            max_materials=None,
            resume=args.resume,
        )
        log(
            f"Focused ingest manifest count={full_manifest['count']} "
            f"(available_reported={full_manifest.get('available_reported')}, "
            f"resumed_from={full_manifest.get('resumed_from_offset', 0)})"
        )
        if full_manifest["count"] < 1000:
            raise RuntimeError("Focused ingest count unexpectedly small")

        written_full = load_oqmd_into_db(replace=True)
        log(f"Full DB load written={written_full}")
        unique_ids = unique_oqmd_source_ids()
        log(f"Unique OQMD source_ids in raw={unique_ids}")
        # INSERT may count duplicate jsonl lines; verify against unique ids.
        expect = min(unique_ids, written_full)
        verify_oqmd_search(expect_min=expect)

        if not args.skip_entity_resolution:
            log("Re-running entity resolution after OQMD load...")
            stats = resolve_entities(settings=settings)
            log(f"Entity resolution stats={stats.to_dict()}")

        db = MaterialsDatabase(read_only=True)
        log(
            "=== OQMD pipeline COMPLETE === "
            f"mp={db.count_by_source(DataSource.MATERIALS_PROJECT.value)} "
            f"jarvis={db.count_by_source(DataSource.JARVIS.value)} "
            f"oqmd={db.count_by_source(DataSource.OQMD.value)} "
            f"aflow={db.count_by_source(DataSource.AFLOW.value)} "
            f"total={db.count_materials()}"
        )
        return 0
    except Exception as exc:
        log(f"FAILED: {exc}")
        log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
