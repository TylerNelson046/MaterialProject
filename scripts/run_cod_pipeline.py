"""COD pipeline: smoke test → focused ingest (+ CIFs) → load DB → entity resolution."""

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
from database.entity_resolution import resolve_entities  # noqa: E402
from database.queries import MaterialSearchFilters, search_materials  # noqa: E402
from database.write_lock import duckdb_write_lock  # noqa: E402
from ingestion.ingest_cod import ingest_cod  # noqa: E402
from normalization.normalize_cod import (  # noqa: E402
    iter_load_normalized_cod,
    normalize_cod_document,
)
from pipeline.search import SearchRequest, run_search  # noqa: E402

LOG_PATH = ROOT / "data" / "raw" / "cod" / "pipeline.log"
TEST_COUNT = 50


def log(message: str) -> None:
    stamp = datetime.now(UTC).replace(microsecond=0).isoformat()
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def verify_cod_search(*, expect_min: int) -> None:
    hits = search_materials(
        MaterialSearchFilters(source=DataSource.COD.value, limit=5)
    )
    if len(hits) < 1:
        raise RuntimeError("COD search returned no rows")
    result = run_search(SearchRequest(source=DataSource.COD.value, limit=5))
    log(
        f"Verify search: source=cod sample={len(hits)} "
        f"any={result.count} first={hits[0].get('source_id')}"
    )
    db = MaterialsDatabase(read_only=True)
    count = db.count_by_source(DataSource.COD.value)
    log(f"Verify DB COD count={count} (expect >= {expect_min})")
    if count < expect_min:
        raise RuntimeError(f"Expected >= {expect_min} COD rows, found {count}")


def load_cod_into_db(*, replace: bool) -> int:
    with duckdb_write_lock(label="cod"):
        db = MaterialsDatabase()
        db.initialize()
        if replace:
            removed = db.delete_source(DataSource.COD.value)
            log(f"Removed {removed} prior COD materials")
        start = db.next_material_index()
        log(f"Inserting COD from MAT index {start}")
        written = db.insert_materials(
            iter_load_normalized_cod(start_index=start),
            batch_size=2000,
        )
        log(f"Inserted {written} COD materials; total DB={db.count_materials()}")
        return written


def smoke_normalize_first_record() -> None:
    settings = get_settings()
    path = settings.raw_source_dir(DataSource.COD) / "materials.jsonl"
    first = path.read_text(encoding="utf-8").splitlines()[0]
    doc = json.loads(first)
    material = normalize_cod_document(doc, index=1, settings=settings)
    assert material.primary_source.source == DataSource.COD
    assert material.primary_source.source_id.startswith("cod-")
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
        log("=== COD pipeline start ===")

        log(f"Phase 1: ingest test slice (max_materials={TEST_COUNT})")
        manifest = ingest_cod(
            settings=settings,
            max_materials=TEST_COUNT,
            download_structures=True,
            cif_workers=4,
        )
        log(f"Test ingest manifest count={manifest['count']} cif_ok={manifest.get('cif_ok')}")
        if manifest["count"] < TEST_COUNT:
            raise RuntimeError("Test ingest returned fewer records than requested")

        smoke_normalize_first_record()
        written = load_cod_into_db(replace=True)
        if written < TEST_COUNT:
            raise RuntimeError("Test DB load wrote fewer records than expected")
        verify_cod_search(expect_min=TEST_COUNT)
        log("Phase 1 PASSED")

        log("Phase 2: ingest focused COD catalog (no C, ≤4 elements)")
        full = ingest_cod(
            settings=settings,
            max_materials=None,
            download_structures=True,
            cif_workers=6,
        )
        log(
            f"Focused ingest count={full['count']} "
            f"cif_ok={full.get('cif_ok')} cif_fail={full.get('cif_fail')}"
        )
        if full["count"] < 1000:
            raise RuntimeError("Focused COD ingest unexpectedly small")

        written_full = load_cod_into_db(replace=True)
        log(f"Full DB load written={written_full}")
        verify_cod_search(expect_min=written_full)

        log("Re-running entity resolution after COD load...")
        stats = resolve_entities(settings=settings)
        log(f"Entity resolution stats={stats.to_dict()}")

        db = MaterialsDatabase(read_only=True)
        log(
            "=== COD pipeline COMPLETE === "
            f"mp={db.count_by_source(DataSource.MATERIALS_PROJECT.value)} "
            f"jarvis={db.count_by_source(DataSource.JARVIS.value)} "
            f"oqmd={db.count_by_source(DataSource.OQMD.value)} "
            f"aflow={db.count_by_source(DataSource.AFLOW.value)} "
            f"cod={db.count_by_source(DataSource.COD.value)} "
            f"total={db.count_materials()}"
        )
        return 0
    except Exception as exc:
        log(f"FAILED: {exc}")
        log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
