"""Ingest OQMD formation-energy summaries into data/raw/oqmd/."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from config.constants import DataSource
from config.settings import Settings, get_settings
from databases.oqmd import OQMDClient


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def ingest_oqmd(
    *,
    settings: Settings | None = None,
    max_materials: int | None = None,
    output_dir: Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Paginate OQMD API and write raw JSONL summaries (no atomic sites).

    When ``resume=True``, append from the existing JSONL line count instead of
    rewriting from offset 0.
    """
    settings = settings or get_settings()
    settings.ensure_directories()

    cfg = settings.database_config.get("oqmd", {})
    if max_materials is None:
        configured = cfg.get("max_materials")
        if configured in {None, "none", "null", False}:
            max_materials = None
        else:
            max_materials = int(configured)

    out_dir = output_dir or settings.raw_source_dir(DataSource.OQMD)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = OQMDClient(settings)
    available = client.count_available()
    stab = client.stability_max
    print(
        f"OQMD reports {available:,} records for current filters "
        f"(stability_max={stab})."
    )

    materials_path = out_dir / "materials.jsonl"
    start_offset = _count_jsonl_lines(materials_path) if resume else 0
    mode = "a" if resume and start_offset else "w"
    if resume and start_offset:
        print(f"Resuming OQMD ingest from offset {start_offset:,}...")
    else:
        print(f"Fetching OQMD summaries (max_materials={max_materials})...")

    sample_ids: list[str] = []
    count = start_offset
    newly_written = 0
    seen_ids: set[str] = set()
    if resume and materials_path.exists():
        with materials_path.open(encoding="utf-8") as existing:
            for line in existing:
                line = line.strip()
                if not line:
                    continue
                try:
                    prior = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sid = prior.get("source_id")
                if sid:
                    seen_ids.add(str(sid))

    with materials_path.open(mode, encoding="utf-8") as handle:
        for record in client.iter_search(
            max_materials=max_materials,
            start_offset=start_offset,
        ):
            entry_id = record.get("entry_id")
            source_id = f"oqmd-{entry_id}" if entry_id is not None else None
            if source_id and source_id in seen_ids:
                # Same entry_id can appear under multiple calculation labels.
                continue
            if source_id:
                seen_ids.add(source_id)
                if len(sample_ids) < 5:
                    sample_ids.append(source_id)

            record = {**record, "source_id": source_id}
            handle.write(json.dumps(record, default=str))
            handle.write("\n")
            count += 1
            newly_written += 1
            if count % 10000 == 0:
                print(f"  ... {count} OQMD records written")

    manifest = {
        "source": DataSource.OQMD.value,
        "retrieved_at": _utc_now(),
        "count": len(seen_ids) if seen_ids else count,
        "newly_written": newly_written,
        "resumed_from_offset": start_offset if resume else 0,
        "max_materials": max_materials,
        "available_reported": available,
        "stability_max": client.stability_max,
        "page_size": client.page_size,
        "materials_file": materials_path.name,
        "fields_excluded": ["sites"],
        "unique_source_ids": len(seen_ids),
        "sample_ids": sample_ids,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Retrieved {count} OQMD materials (+{newly_written} this run).")
    print(f"Wrote {materials_path}")
    print(f"Wrote {manifest_path}")
    return manifest


def main() -> None:
    ingest_oqmd()


if __name__ == "__main__":
    main()
