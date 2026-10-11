"""Fully automated JARVIS pipeline: small test → verify → full pull → load DB."""

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
from ingestion.ingest_jarvis import ingest_jarvis  # noqa: E402
from normalization.normalize_jarvis import (  # noqa: E402
    iter_load_normalized_jarvis,
    normalize_jarvis_document,
)
from pipeline.search import SearchRequest, run_search  # noqa: E402

LOG_PATH = ROOT / "data" / "raw" / "jarvis" / "pipeline.log"
TEST_COUNT = 100


def log(message: str) -> None:
    stamp = datetime.now(UTC).replace(microsecond=0).isoformat()
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def verify_jarvis_search(*, expect_min: int) -> None:
    overview_hits = search_materials(
        MaterialSearchFilters(source=DataSource.JARVIS.value, limit=5)
    )
    if len(overview_hits) < 1:
        raise RuntimeError("JARVIS search returned no rows")

    result = run_search(
        SearchRequest(
            source=DataSource.JARVIS.value,
            energy_above_hull_max=0.1,
            limit=5,
        )
    )
    log(
        f"Verify search: source=jarvis sample={len(overview_hits)} "
        f"near-hull={result.count} first={overview_hits[0].get('source_id')}"
    )
    if overview_hits[0].get("source") != DataSource.JARVIS.value:
        raise RuntimeError("Search hit source mismatch")

    db = MaterialsDatabase(read_only=True)
    count = db.count_by_source(DataSource.JARVIS.value)
    log(f"Verify DB JARVIS count={count} (expect >= {expect_min})")
    if count < expect_min:
        raise RuntimeError(f"Expected >= {expect_min} JARVIS rows, found {count}")


def load_jarvis_into_db(*, replace: bool) -> int:
    db = MaterialsDatabase()
    db.initialize()
    if replace:
        removed = db.delete_source(DataSource.JARVIS.value)
        log(f"Removed {removed} prior JARVIS materials")
    start = db.next_material_index()
    log(f"Inserting JARVIS from MAT index {start}")
    written = db.insert_materials(
        iter_load_normalized_jarvis(start_index=start),
        batch_size=2000,
    )
    log(f"Inserted {written} JARVIS materials; total DB={db.count_materials()}")
    return written


def smoke_normalize_first_record() -> None:
    settings = get_settings()
    path = settings.raw_source_dir(DataSource.JARVIS) / "materials.jsonl"
    first = path.read_text(encoding="utf-8").splitlines()[0]
    doc = json.loads(first)
    material = normalize_jarvis_document(doc, index=1, settings=settings)
    assert material.primary_source.source == DataSource.JARVIS
    assert material.primary_source.source_id.startswith("JVASP-")
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
        log("=== JARVIS pipeline start ===")

        # ---- Phase 1: small test ----
        log(f"Phase 1: ingest test slice (max_materials={TEST_COUNT})")
        manifest = ingest_jarvis(
            settings=settings,
            max_materials=TEST_COUNT,
            download_structures=True,
        )
        log(f"Test ingest manifest count={manifest['count']}")
        if manifest["count"] < TEST_COUNT:
            raise RuntimeError("Test ingest returned fewer records than requested")

        smoke_normalize_first_record()
        written = load_jarvis_into_db(replace=True)
        if written < TEST_COUNT:
            raise RuntimeError("Test DB load wrote fewer records than expected")
        verify_jarvis_search(expect_min=TEST_COUNT)
        log("Phase 1 PASSED")

        # ---- Phase 2: full catalog ----
        log("Phase 2: ingest full JARVIS dft_3d catalog")
        full_manifest = ingest_jarvis(
            settings=settings,
            max_materials=None,
            download_structures=True,
        )
        log(f"Full ingest manifest count={full_manifest['count']}")
        if full_manifest["count"] < 1000:
            raise RuntimeError("Full ingest count unexpectedly small")

        written_full = load_jarvis_into_db(replace=True)
        log(f"Full DB load written={written_full}")
        verify_jarvis_search(expect_min=written_full)

        db = MaterialsDatabase(read_only=True)
        log(
            "=== JARVIS pipeline COMPLETE === "
            f"mp={db.count_by_source(DataSource.MATERIALS_PROJECT.value)} "
            f"jarvis={db.count_by_source(DataSource.JARVIS.value)} "
            f"total={db.count_materials()}"
        )
        return 0
    except Exception as exc:
        log(f"FAILED: {exc}")
        log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
