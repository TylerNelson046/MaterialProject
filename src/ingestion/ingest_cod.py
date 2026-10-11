"""Ingest COD metadata (+ optional CIFs) into data/raw/cod/."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from config.constants import DataSource
from config.settings import Settings, get_settings
from databases.cod import CODClient


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _canonical_source_id(file_id: str | int) -> str:
    return f"cod-{int(file_id)}"


def _cif_rel_path(settings: Settings, file_id: str | int) -> Path:
    return settings.experimental_structures_dir() / f"cod-{int(file_id)}.cif"


def ingest_cod(
    *,
    settings: Settings | None = None,
    max_materials: int | None = None,
    output_dir: Path | None = None,
    download_structures: bool | None = None,
    cif_workers: int = 6,
) -> dict[str, Any]:
    """Fetch focused COD metadata and optionally download CIF files."""
    settings = settings or get_settings()
    settings.ensure_directories()

    cfg = settings.database_config.get("cod", {})
    if max_materials is None:
        configured = cfg.get("max_materials")
        if configured in {None, "none", "null", False}:
            max_materials = None
        else:
            max_materials = int(configured)

    if download_structures is None:
        download_structures = bool(cfg.get("download_structures", True))

    out_dir = output_dir or settings.raw_source_dir(DataSource.COD)
    out_dir.mkdir(parents=True, exist_ok=True)
    settings.experimental_structures_dir().mkdir(parents=True, exist_ok=True)

    client = CODClient(settings)
    available = client.count_available()
    print(
        f"COD reports {available:,} records for current filters "
        f"(strict={client.strictmin}..{client.strictmax}, "
        f"exclude={client.exclude_elements})."
    )
    print(f"Fetching COD metadata (max_materials={max_materials})...")

    materials_path = out_dir / "materials.jsonl"
    sample_ids: list[str] = []
    records: list[dict[str, Any]] = []

    for row in client.iter_search(max_materials=max_materials):
        file_id = row.get("file")
        source_id = _canonical_source_id(file_id)
        if len(sample_ids) < 5:
            sample_ids.append(source_id)
        cif_path = _cif_rel_path(settings, file_id)
        record = {
            **row,
            "source_id": source_id,
            "cod_id": int(file_id),
            "cif_url": client.cif_url(file_id),
            "structure_path": str(cif_path.relative_to(settings.project_root)),
        }
        records.append(record)

    with materials_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, default=str))
            handle.write("\n")

    print(f"Wrote {len(records):,} COD metadata rows → {materials_path}")

    cif_ok = 0
    cif_skip = 0
    cif_fail = 0
    if download_structures and records:
        print(
            f"Downloading CIFs with {cif_workers} workers "
            f"({len(records):,} entries)..."
        )

        def _one(rec: dict[str, Any]) -> tuple[str, str]:
            cod_id = int(rec["cod_id"])
            dest = settings.project_root / rec["structure_path"]
            if dest.exists() and dest.stat().st_size > 50:
                return "skip", rec["source_id"]
            try:
                text = client.fetch_cif(cod_id)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(text, encoding="utf-8")
                time.sleep(0.05)  # polite pacing per worker
                return "ok", rec["source_id"]
            except Exception as exc:  # noqa: BLE001
                return "fail", f"{rec['source_id']}:{exc}"

        with ThreadPoolExecutor(max_workers=max(1, int(cif_workers))) as pool:
            futures = [pool.submit(_one, rec) for rec in records]
            done = 0
            for fut in as_completed(futures):
                status, _detail = fut.result()
                if status == "ok":
                    cif_ok += 1
                elif status == "skip":
                    cif_skip += 1
                else:
                    cif_fail += 1
                done += 1
                if done % 2000 == 0:
                    print(
                        f"  ... CIF progress {done}/{len(records)} "
                        f"(ok={cif_ok} skip={cif_skip} fail={cif_fail})"
                    )

        print(f"CIF download done: ok={cif_ok} skip={cif_skip} fail={cif_fail}")

    manifest = {
        "source": DataSource.COD.value,
        "retrieved_at": _utc_now(),
        "count": len(records),
        "max_materials": max_materials,
        "available_reported": available,
        "strictmin": client.strictmin,
        "strictmax": client.strictmax,
        "exclude_elements": client.exclude_elements,
        "download_structures": download_structures,
        "cif_ok": cif_ok,
        "cif_skip": cif_skip,
        "cif_fail": cif_fail,
        "materials_file": materials_path.name,
        "sample_ids": sample_ids,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}")
    return manifest


def main() -> None:
    ingest_cod()


if __name__ == "__main__":
    main()
