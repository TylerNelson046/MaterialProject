"""Ingest JARVIS-DFT 3D data into data/raw/jarvis/."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from config.constants import DataSource
from config.settings import Settings, get_settings
from databases.jarvis import JarvisClient


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _structure_path(settings: Settings, source_id: str) -> Path:
    safe = source_id.replace("/", "_")
    return settings.computational_structures_dir() / f"{safe}.json"


def ingest_jarvis(
    *,
    settings: Settings | None = None,
    max_materials: int | None = None,
    output_dir: Path | None = None,
    download_structures: bool = True,
) -> dict[str, Any]:
    """Download/export JARVIS-DFT records to raw JSONL (+ optional structures)."""
    settings = settings or get_settings()
    settings.ensure_directories()

    jarvis_cfg = settings.database_config.get("jarvis", {})
    if max_materials is None:
        configured = jarvis_cfg.get("max_materials")
        if configured in {None, "none", "null", False}:
            max_materials = None
        else:
            max_materials = int(configured)

    out_dir = output_dir or settings.raw_source_dir(DataSource.JARVIS)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = JarvisClient(settings)
    print(f"Fetching JARVIS dft_3d (max_materials={max_materials})...")

    materials_path = out_dir / "materials.jsonl"
    material_ids: list[str] = []
    structures_written = 0

    with materials_path.open("w", encoding="utf-8") as handle:
        for record in client.iter_search(max_materials=max_materials):
            mid = str(record.get("jid") or "").strip()
            if not mid:
                continue
            material_ids.append(mid)

            # Keep atoms in a side file; strip heavy blob from summary JSONL.
            atoms = record.get("atoms")
            summary = {k: v for k, v in record.items() if k != "atoms"}
            if download_structures and atoms:
                path = _structure_path(settings, mid)
                path.write_text(json.dumps(atoms, default=str), encoding="utf-8")
                structures_written += 1
                summary["structure_path"] = str(path.relative_to(settings.project_root))

            handle.write(json.dumps(summary, default=str))
            handle.write("\n")

            if len(material_ids) % 10000 == 0:
                print(f"  ... {len(material_ids)} JARVIS records written")

    manifest = {
        "source": DataSource.JARVIS.value,
        "dataset": client.dataset_name,
        "retrieved_at": _utc_now(),
        "count": len(material_ids),
        "max_materials": max_materials,
        "materials_file": materials_path.name,
        "structures_downloaded": download_structures,
        "structures_written": structures_written,
        "sample_ids": material_ids[:5],
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Retrieved {len(material_ids)} JARVIS materials.")
    print(f"Wrote {materials_path}")
    print(f"Wrote {manifest_path}")
    if download_structures:
        print(f"Wrote {structures_written} structure files.")
    return manifest


def main() -> None:
    ingest_jarvis()


if __name__ == "__main__":
    main()
