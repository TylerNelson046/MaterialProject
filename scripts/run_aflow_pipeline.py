"""Fully automated AFLOW pipeline: small test → verify → full ICSD pull → load DB.

Raw download can run in parallel with OQMD. DuckDB loads use an exclusive flock
so concurrent pipelines never write the DB at the same time.
"""

from __future__ import annotations

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
from database.queries import MaterialSearchFilters, search_materials  # noqa: E402
from database.write_lock import duckdb_write_lock  # noqa: E402
from ingestion.ingest_aflow import ingest_aflow  # noqa: E402
from normalization.normalize_aflow import (  # noqa: E402
    iter_load_normalized_aflow,
    normalize_aflow_document,
)
from pipeline.search import SearchRequest, run_search  # noqa: E402

LOG_PATH = ROOT / "data" / "raw" / "aflow" / "pipeline.log"
TEST_COUNT = 100


def log(message: str) -> None:
    stamp = datetime.now(UTC).replace(microsecond=0).isoformat()
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def verify_aflow_search(*, expect_min: int) -> None:
    hits = search_materials(
        MaterialSearchFilters(source=DataSource.AFLOW.value, limit=5)
    )
    if len(hits) < 1:
        raise RuntimeError("AFLOW search returned no rows")

    result = run_search(
        SearchRequest(
            source=DataSource.AFLOW.value,
            limit=5,
        )
    )
    log(
        f"Verify search: source=aflow sample={len(hits)} "
        f"any={result.count} first={hits[0].get('source_id')}"
    )

    db = MaterialsDatabase(read_only=True)
    count = db.count_by_source(DataSource.AFLOW.value)
    log(f"Verify DB AFLOW count={count} (expect >= {expect_min})")
    if count < expect_min:
        raise RuntimeError(f"Expected >= {expect_min} AFLOW rows, found {count}")


def load_aflow_into_db(*, replace: bool) -> int:
    with duckdb_write_lock(label="aflow"):
        db = MaterialsDatabase()
        db.initialize()
        if replace:
            removed = db.delete_source(DataSource.AFLOW.value)
            log(f"Removed {removed} prior AFLOW materials")
        start = db.next_material_index()
        log(f"Inserting AFLOW from MAT index {start}")
        written = db.insert_materials(
            iter_load_normalized_aflow(start_index=start),
            batch_size=2000,
        )
        log(f"Inserted {written} AFLOW materials; total DB={db.count_materials()}")
        return written


def smoke_normalize_first_record() -> None:
    settings = get_settings()
    path = settings.raw_source_dir(DataSource.AFLOW) / "materials.jsonl"
    first = path.read_text(encoding="utf-8").splitlines()[0]
    doc = json.loads(first)
    material = normalize_aflow_document(doc, index=1, settings=settings)
    assert material.primary_source.source == DataSource.AFLOW
    assert material.primary_source.source_id.startswith("aflow-")
    log(
        f"Normalize smoke OK: {material.material_id} "
        f"{material.primary_source.source_id} {material.formula_pretty}"
    )


def main() -> int:
    clear_settings_cache()
    settings = get_settings()
    settings.ensure_directories()
    LOG_PATH.write_text("", encoding="utf-8")

    try:
        log("=== AFLOW pipeline start ===")

        log(f"Phase 1: ingest test slice (max_materials={TEST_COUNT})")
        manifest = ingest_aflow(settings=settings, max_materials=TEST_COUNT)
        log(f"Test ingest manifest count={manifest['count']}")
        if manifest["count"] < TEST_COUNT:
            raise RuntimeError("Test ingest returned fewer records than requested")

        smoke_normalize_first_record()
        written = load_aflow_into_db(replace=True)
        if written < TEST_COUNT:
            raise RuntimeError("Test DB load wrote fewer records than expected")
        verify_aflow_search(expect_min=TEST_COUNT)
        log("Phase 1 PASSED")

        log("Phase 2: ingest AFLOW ICSD catalog")
        full_manifest = ingest_aflow(settings=settings, max_materials=None)
        log(
            f"Focused ingest manifest count={full_manifest['count']} "
            f"(catalog={full_manifest.get('catalog')})"
        )
        if full_manifest["count"] < 1000:
            raise RuntimeError("Focused ingest count unexpectedly small")

        written_full = load_aflow_into_db(replace=True)
        log(f"Full DB load written={written_full}")
        verify_aflow_search(expect_min=written_full)

        with duckdb_write_lock(label="aflow-summary"):
            db = MaterialsDatabase(read_only=True)
            log(
                "=== AFLOW pipeline COMPLETE === "
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
