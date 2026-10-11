"""Ingest Materials Project data from the local Arrow/Delta cache (fast path)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from deltalake import DeltaTable

from config.constants import DataSource
from config.settings import Settings, get_settings
from databases.materials_project import MaterialsProjectClient, ensure_mp_summary_dataset

# Heavy nested blobs kept out of the summary table unless explicitly needed.
DEFAULT_EXCLUDE_FROM_SUMMARY: tuple[str, ...] = (
    "structure",
    "dos",
    "bandstructure",
    "grain_boundaries",
    "xas",
    "origins",
    "builder_meta",
)


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _structure_path(settings: Settings, source_id: str) -> Path:
    safe = source_id.replace("/", "_")
    return settings.computational_structures_dir() / f"{safe}.json"


def _json_default(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def _write_jsonl_from_table(table: pa.Table, path: Path, *, chunk_size: int = 5000) -> int:
    """Stream an Arrow table to JSONL. Returns row count."""
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for batch in table.to_batches(max_chunksize=chunk_size):
            for row in batch.to_pylist():
                handle.write(json.dumps(row, default=_json_default))
                handle.write("\n")
                count += 1
            if count % 25000 == 0:
                print(f"  ... {count} summary rows written")
    return count


def _write_structures_from_table(
    table: pa.Table,
    settings: Settings,
    *,
    chunk_size: int = 2000,
    max_structures: int | None = None,
) -> int:
    """Write structure JSON files from an Arrow table with material_id + structure."""
    written = 0
    for batch in table.to_batches(max_chunksize=chunk_size):
        for row in batch.to_pylist():
            mid = str(row.get("material_id") or "")
            structure = row.get("structure")
            if not mid or not structure:
                continue
            path = _structure_path(settings, mid)
            path.write_text(json.dumps(structure, default=_json_default), encoding="utf-8")
            written += 1
            if written % 10000 == 0:
                print(f"  ... {written} structures written")
            if max_structures is not None and written >= max_structures:
                return written
    return written


def _select_summary_columns(table: pa.Table, fields: list[str]) -> pa.Table:
    available = set(table.column_names)
    keep = [f for f in fields if f in available and f not in DEFAULT_EXCLUDE_FROM_SUMMARY]
    # Always keep core identity fields if present.
    for required in ("material_id", "formula_pretty", "elements", "chemsys"):
        if required in available and required not in keep:
            keep.insert(0, required)
    return table.select(keep)


def ingest_materials_project(
    *,
    settings: Settings | None = None,
    max_materials: int | None = None,
    output_dir: Path | None = None,
    download_structures: bool | None = None,
    force_renew: bool = False,
) -> dict[str, Any]:
    """Export MP summaries from the local Delta dataset into project data/.

    This avoids slow Python iteration through the mp-api client wrappers.
    """
    settings = settings or get_settings()
    settings.ensure_directories()

    out_dir = output_dir or settings.raw_source_dir(DataSource.MATERIALS_PROJECT)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = MaterialsProjectClient(settings)
    limit = settings.mp_max_materials if max_materials is None else max_materials
    want_structures = (
        settings.mp_download_structures
        if download_structures is None
        else download_structures
    )

    print("Ensuring local Materials Project summary Delta dataset...")
    dataset_path = ensure_mp_summary_dataset(settings, force_renew=force_renew)
    print(f"Reading Delta table: {dataset_path}")

    delta = DeltaTable(str(dataset_path))
    full_table = delta.to_pyarrow_table()
    print(f"Delta rows available: {full_table.num_rows}")

    if limit is not None:
        full_table = full_table.slice(0, int(limit))
        print(f"Applying max_materials={limit} → {full_table.num_rows} rows")

    summary_table = _select_summary_columns(full_table, client.fields)
    print(f"Exporting {summary_table.num_columns} summary columns...")

    parquet_path = out_dir / "materials.parquet"
    jsonl_path = out_dir / "materials.jsonl"

    pq.write_table(summary_table, parquet_path)
    print(f"Wrote {parquet_path}")

    count = _write_jsonl_from_table(summary_table, jsonl_path)
    print(f"Wrote {jsonl_path} ({count} rows)")

    structures_written = 0
    if want_structures:
        if "material_id" in full_table.column_names and "structure" in full_table.column_names:
            print("Exporting crystal structures from Delta cache...")
            struct_table = full_table.select(["material_id", "structure"])
            structures_written = _write_structures_from_table(struct_table, settings)
            print(f"Wrote {structures_written} structure files.")
        else:
            print("Structure column not present in Delta table; skipping structures.")

    sample_ids: list[str] = []
    if "material_id" in summary_table.column_names and summary_table.num_rows:
        sample_ids = [
            str(x)
            for x in summary_table.column("material_id").to_pylist()[:5]
            if x is not None
        ]

    manifest = {
        "source": DataSource.MATERIALS_PROJECT.value,
        "retrieved_at": _utc_now(),
        "count": count,
        "max_materials": limit,
        "fields": summary_table.column_names,
        "materials_file": jsonl_path.name,
        "parquet_file": parquet_path.name,
        "dataset_path": str(dataset_path),
        "export_method": "delta_arrow",
        "structures_downloaded": want_structures,
        "structures_written": structures_written,
        "sample_ids": sample_ids,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}")
    return manifest


def main() -> None:
    ingest_materials_project()


if __name__ == "__main__":
    main()
