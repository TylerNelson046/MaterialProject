"""Dataset quality repairs 1–5 (idempotent).

1. Promote JARVIS specialty properties into canonical ``properties``
2. Flag MP ``deprecated`` on properties + search_index
3. Multi-source coalescing into search_index nulls (never average)
4. Backfill JARVIS volume from structure files
5. Conflict policy is enforced in queries (per-source rows + spread API);
   this script refreshes search_index consistently with that policy.
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

from config.constants import PROPERTY_UNITS, PropertyName  # noqa: E402
from config.settings import clear_settings_cache, get_settings  # noqa: E402
from database.duckdb import MaterialsDatabase, ensure_search_index_columns  # noqa: E402
from database.write_lock import duckdb_write_lock  # noqa: E402
from normalization.normalize_jarvis import (  # noqa: E402
    _num,
    _volume_from_jarvis_atoms,
    iter_raw_jarvis_documents,
)

# Import enrichment coalesce without running its CLI.
sys.path.insert(0, str(ROOT / "scripts"))
from repair_dataset_enrichment import enrich_search_index_from_linked_sources  # noqa: E402

JARVIS_SPECIALTY: tuple[tuple[str, PropertyName, str], ...] = (
    ("mbj_bandgap", PropertyName.MBJ_BANDGAP, "mbj_bandgap"),
    ("avg_elec_mass", PropertyName.AVG_ELEC_MASS, "avg_elec_mass"),
    ("avg_hole_mass", PropertyName.AVG_HOLE_MASS, "avg_hole_mass"),
    ("slme", PropertyName.SLME, "slme"),
    ("Tc_supercon", PropertyName.TC_SUPERCON, "tc_supercon"),
    ("exfoliation_energy", PropertyName.EXFOLIATION_ENERGY, "exfoliation_energy"),
)


def _upsert_double_props(
    conn,
    *,
    source: str,
    rows: list[dict[str, Any]],
    prop_name: str,
    unit: str,
    method: str,
) -> int:
    if not rows:
        return 0
    frame = pd.DataFrame(rows)
    tmp = f"tmp_{source}_{prop_name}"
    conn.register(tmp, frame)
    conn.execute(
        f"""
        INSERT OR REPLACE INTO properties
            (source, source_id, material_id, name, value_double, value_bool,
             value_text, unit, origin, method, uncertainty)
        SELECT
            '{source}',
            t.source_id,
            s.material_id,
            '{prop_name}',
            t.value,
            NULL,
            NULL,
            '{unit}',
            'known',
            '{method}',
            NULL
        FROM {tmp} AS t
        JOIN source_records AS s
          ON s.source = '{source}' AND s.source_id = t.source_id
        WHERE t.value IS NOT NULL
        """
    )
    n = int(conn.rowcount or 0)
    conn.unregister(tmp)
    return n


def repair_jarvis_specialty(conn, *, settings) -> dict[str, int]:
    path = settings.raw_source_dir("jarvis") / "materials.jsonl"
    print(f"JARVIS specialty: scanning {path}...")
    buckets: dict[str, list[dict[str, Any]]] = {canon.value: [] for _, canon, _ in JARVIS_SPECIALTY}
    scanned = 0
    for doc in iter_raw_jarvis_documents(path):
        scanned += 1
        source_id = str(doc.get("jid") or "").strip()
        if not source_id:
            continue
        for raw_key, canon, _ in JARVIS_SPECIALTY:
            value = _num(doc.get(raw_key))
            if value is None:
                continue
            buckets[canon.value].append({"source_id": source_id, "value": value})

    stats = {"scanned": scanned}
    for _, canon, _ in JARVIS_SPECIALTY:
        unit = PROPERTY_UNITS[canon]
        n = _upsert_double_props(
            conn,
            source="jarvis",
            rows=buckets[canon.value],
            prop_name=canon.value,
            unit=unit,
            method="jarvis_dft_3d",
        )
        stats[canon.value] = n
        # Push onto JARVIS-primary search rows.
        if buckets[canon.value]:
            frame = pd.DataFrame(buckets[canon.value])
            tmp = f"si_{canon.value}"
            conn.register(tmp, frame)
            conn.execute(
                f"""
                UPDATE search_index AS si
                SET {canon.value} = t.value
                FROM {tmp} AS t
                WHERE si.source = 'jarvis' AND si.source_id = t.source_id
                """
            )
            conn.unregister(tmp)
    print(f"JARVIS specialty: {stats}")
    return stats


def repair_jarvis_volume(conn, *, settings) -> dict[str, int]:
    struct_dir = settings.computational_structures_dir()
    print(f"JARVIS volume: scanning structures under {struct_dir}...")
    rows: list[dict[str, Any]] = []
    failed = 0
    for path in struct_dir.glob("JVASP-*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            failed += 1
            continue
        atoms = payload.get("atoms") if isinstance(payload, dict) else None
        if not isinstance(atoms, dict):
            atoms = payload if isinstance(payload, dict) else None
        volume = _volume_from_jarvis_atoms(atoms if isinstance(atoms, dict) else None)
        if volume is None:
            failed += 1
            continue
        rows.append({"source_id": path.stem, "value": volume})

    if not rows:
        print("JARVIS volume: nothing computed")
        return {"computed": 0, "failed": failed}

    frame = pd.DataFrame(rows)
    conn.register("jarvis_vol", frame)
    conn.execute(
        """
        UPDATE source_records AS s
        SET volume = v.value
        FROM jarvis_vol AS v
        WHERE s.source = 'jarvis' AND s.source_id = v.source_id
        """
    )
    n = _upsert_double_props(
        conn,
        source="jarvis",
        rows=rows,
        prop_name="volume",
        unit="Å^3",
        method="jarvis_structure_lattice",
    )
    conn.execute(
        """
        UPDATE search_index AS si
        SET volume = v.value
        FROM jarvis_vol AS v
        WHERE si.source = 'jarvis' AND si.source_id = v.source_id
        """
    )
    conn.unregister("jarvis_vol")
    stats = {"computed": len(rows), "upserted_props": n, "failed": failed}
    print(f"JARVIS volume: {stats}")
    return stats


def repair_mp_deprecated(conn, *, settings) -> dict[str, int]:
    """Flag deprecated MP materials from original_metadata / raw parquet."""
    print("MP deprecated: extracting flags...")
    rows = conn.execute(
        """
        SELECT source_id, original_metadata
        FROM source_records
        WHERE source = 'materials_project'
        """
    ).fetchall()
    updates: list[dict[str, Any]] = []
    for source_id, meta in rows:
        deprecated = None
        if isinstance(meta, dict):
            deprecated = meta.get("deprecated")
        elif isinstance(meta, str) and meta:
            try:
                parsed = json.loads(meta)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                deprecated = parsed.get("deprecated")
        if deprecated is None:
            continue
        updates.append(
            {
                "source_id": source_id,
                "deprecated": bool(deprecated),
            }
        )

    if not updates:
        # Fallback: scan parquet/jsonl if metadata missing.
        raw_dir = settings.raw_source_dir("materials_project")
        parquet = raw_dir / "materials.parquet"
        if parquet.exists():
            import pyarrow.parquet as pq

            table = pq.read_table(parquet, columns=["material_id", "deprecated"])
            for material_id, deprecated in zip(
                table.column("material_id").to_pylist(),
                table.column("deprecated").to_pylist(),
            ):
                if deprecated is None:
                    continue
                updates.append(
                    {
                        "source_id": str(material_id),
                        "deprecated": bool(deprecated),
                    }
                )

    if not updates:
        print("MP deprecated: no flags found")
        return {"flagged": 0}

    frame = pd.DataFrame(updates)
    conn.register("mp_depr", frame)
    conn.execute(
        """
        INSERT OR REPLACE INTO properties
            (source, source_id, material_id, name, value_double, value_bool,
             value_text, unit, origin, method, uncertainty)
        SELECT
            'materials_project',
            d.source_id,
            s.material_id,
            'deprecated',
            NULL,
            d.deprecated,
            NULL,
            'boolean',
            'known',
            'materials_project',
            NULL
        FROM mp_depr AS d
        JOIN source_records AS s
          ON s.source = 'materials_project' AND s.source_id = d.source_id
        """
    )
    # Any linked material with a deprecated MP sibling is flagged.
    conn.execute(
        """
        UPDATE search_index AS si
        SET deprecated = TRUE
        FROM source_records AS s
        JOIN mp_depr AS d
          ON d.source_id = s.source_id AND d.deprecated = TRUE
        WHERE s.source = 'materials_project'
          AND s.material_id = si.material_id
        """
    )
    conn.execute(
        """
        UPDATE search_index AS si
        SET deprecated = FALSE
        FROM mp_depr AS d
        WHERE si.source = 'materials_project'
          AND si.source_id = d.source_id
          AND d.deprecated = FALSE
          AND (si.deprecated IS NULL OR si.deprecated = FALSE)
        """
    )
    conn.unregister("mp_depr")
    n_true = sum(1 for row in updates if row["deprecated"])
    stats = {"flagged": len(updates), "deprecated_true": n_true}
    print(f"MP deprecated: {stats}")
    return stats


def _verify(conn) -> dict[str, Any]:
    out: dict[str, Any] = {}
    out["jarvis_mbj"] = conn.execute(
        "SELECT COUNT(*) FROM properties WHERE source='jarvis' AND name='mbj_bandgap'"
    ).fetchone()[0]
    out["jarvis_volume"] = conn.execute(
        "SELECT COUNT(*) FROM properties WHERE source='jarvis' AND name='volume'"
    ).fetchone()[0]
    out["mp_deprecated_prop"] = conn.execute(
        "SELECT COUNT(*) FROM properties WHERE source='materials_project' AND name='deprecated' AND value_bool=TRUE"
    ).fetchone()[0]
    out["si_deprecated"] = conn.execute(
        "SELECT COUNT(*) FROM search_index WHERE deprecated = TRUE"
    ).fetchone()[0]
    out["si_mbj"] = conn.execute(
        "SELECT COUNT(*) FROM search_index WHERE mbj_bandgap IS NOT NULL"
    ).fetchone()[0]
    out["si_volume_null"] = conn.execute(
        "SELECT COUNT(*) FROM search_index WHERE volume IS NULL"
    ).fetchone()[0]
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair dataset quality items 1–5.")
    parser.add_argument(
        "--skip-enrich",
        action="store_true",
        help="Skip multi-source null coalescing into search_index.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    clear_settings_cache()
    settings = get_settings()
    settings.ensure_directories()

    with duckdb_write_lock(settings, label="dataset-quality"):
        db = MaterialsDatabase(settings.duckdb_path)
        with db.connect() as conn:
            added = ensure_search_index_columns(conn)
            if added:
                print(f"Added search_index columns: {added}")
            print("Before:", _verify(conn))
            repair_jarvis_specialty(conn, settings=settings)
            repair_jarvis_volume(conn, settings=settings)
            repair_mp_deprecated(conn, settings=settings)
            if not args.skip_enrich:
                print("Coalescing null search_index fields from linked sources (no averaging)...")
                enrich_search_index_from_linked_sources(conn)
            print("After:", _verify(conn))

    print("Dataset quality repair (1–5) complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
