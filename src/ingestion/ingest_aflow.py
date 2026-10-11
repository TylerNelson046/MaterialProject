"""Ingest AFLOW AFLUX summaries into data/raw/aflow/."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from config.constants import DataSource
from config.settings import Settings, get_settings
from databases.aflow import AFLOWClient


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _canonical_source_id(auid: str) -> str:
    """Normalize auid like 'aflow:abc' → 'aflow-abc'."""
    text = str(auid).strip()
    if text.startswith("aflow:"):
        return "aflow-" + text.removeprefix("aflow:")
    if text.startswith("aflow-"):
        return text
    return f"aflow-{text}"


def ingest_aflow(
    *,
    settings: Settings | None = None,
    max_materials: int | None = None,
    output_dir: Path | None = None,
    count_first: bool = False,
) -> dict[str, Any]:
    """Paginate AFLOW AFLUX and write raw JSONL summaries (ICSD by default)."""
    settings = settings or get_settings()
    settings.ensure_directories()

    cfg = settings.database_config.get("aflow", {})
    if max_materials is None:
        configured = cfg.get("max_materials")
        if configured in {None, "none", "null", False}:
            max_materials = None
        else:
            max_materials = int(configured)

    out_dir = output_dir or settings.raw_source_dir(DataSource.AFLOW)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = AFLOWClient(settings)
    available = None
    if count_first:
        print("Estimating AFLOW catalog size (AFLUX paging)...")
        available = client.count_available()
        print(f"AFLOW reports ~{available:,} records for current filters.")

    print(
        f"Fetching AFLOW summaries "
        f"(catalog={client.catalog}, max_materials={max_materials})..."
    )

    materials_path = out_dir / "materials.jsonl"
    sample_ids: list[str] = []
    count = 0

    with materials_path.open("w", encoding="utf-8") as handle:
        for record in client.iter_search(max_materials=max_materials):
            auid = record.get("auid")
            source_id = _canonical_source_id(str(auid)) if auid else None
            if source_id and len(sample_ids) < 5:
                sample_ids.append(source_id)

            record = {**record, "source_id": source_id}
            handle.write(json.dumps(record, default=str))
            handle.write("\n")
            count += 1
            if count % 10000 == 0:
                print(f"  ... {count} AFLOW records written")

    manifest = {
        "source": DataSource.AFLOW.value,
        "retrieved_at": _utc_now(),
        "count": count,
        "max_materials": max_materials,
        "available_reported": available,
        "catalog": client.catalog,
        "page_size": client.page_size,
        "materials_file": materials_path.name,
        "sample_ids": sample_ids,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Retrieved {count} AFLOW materials.")
    print(f"Wrote {materials_path}")
    print(f"Wrote {manifest_path}")
    return manifest


def main() -> None:
    ingest_aflow()


if __name__ == "__main__":
    main()
