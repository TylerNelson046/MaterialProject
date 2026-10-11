"""Dataset quality repairs 6–10 (idempotent).

6. Physical QA gates — delete out-of-range property rows + null SI cells
7. AFLOW hull policy — mark metadata; never treat AFLOW as hull-comparable
8. COD space-group backfill from symbol / CIF
9. Refresh property ``method`` tags to functional-specific labels
10. Backfill OQMD structure_path from on-disk files (download is separate)
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
sys.path.insert(0, str(ROOT / "scripts"))

from config.settings import clear_settings_cache, get_settings  # noqa: E402
from database.duckdb import MaterialsDatabase, ensure_search_index_columns  # noqa: E402
from database.write_lock import duckdb_write_lock  # noqa: E402
from normalization.normalize_cod import _symmetry_from_cod, iter_raw_cod_documents  # noqa: E402
from repair_dataset_enrichment import enrich_search_index_from_linked_sources  # noqa: E402
from repair_p0_dataset import backfill_oqmd_structure_paths  # noqa: E402

METHOD_UPDATES: list[tuple[str, str | None, str, str]] = [
    # source, name (None=all for source), old_method, new_method
    ("materials_project", None, "materials_project", "mp-gga-pbe"),
    ("jarvis", "mbj_bandgap", "jarvis_dft_3d", "jarvis-mbj"),
    ("jarvis", "dielectric_total", "jarvis_dft_3d", "jarvis-dfpt"),
    ("jarvis", "dielectric_ionic", "jarvis_dft_3d", "jarvis-dfpt"),
    ("jarvis", "dielectric_electronic", "jarvis_dft_3d", "jarvis-dfpt"),
    ("jarvis", "tc_supercon", "jarvis_dft_3d", "jarvis-supercon"),
    ("jarvis", "volume", "jarvis_dft_3d", "jarvis-structure"),
    ("jarvis", "volume", "jarvis_structure_lattice", "jarvis-structure"),
    ("jarvis", None, "jarvis_dft_3d", "jarvis-optb88vdw"),
    ("oqmd", None, "oqmd", "oqmd-dft"),
    ("oqmd", "density", "oqmd_composition_volume", "oqmd-dft"),
    ("aflow", None, "aflow", "aflow-dft"),
    ("cod", None, "cod", "cod-experimental"),
    ("cod", "density", "cod_cell_mass_volume", "cod-experimental"),
]


def repair_physical_qa(conn) -> dict[str, int]:
    """Delegate to shared Step 1.3 sanitizer; return per-property delete counts."""
    from database.physical_qa import sanitize_physical_outliers

    print("Physical QA: removing out-of-range property rows...")
    report = sanitize_physical_outliers(conn)
    deleted = {
        name: int(n)
        for name, n in report["properties_deleted"].items()
        if int(n) > 0
    }
    print(f"Physical QA deleted: {deleted}")
    print(f"Physical QA SI nulled: {report['search_index_nulled']}")
    return deleted


def repair_aflow_hull_metadata(conn) -> int:
    """Stamp AFLOW source_records as not hull-comparable."""
    print("AFLOW hull policy: updating original_metadata flags...")
    rows = conn.execute(
        """
        SELECT source_id, original_metadata
        FROM source_records WHERE source = 'aflow'
        """
    ).fetchall()
    updates: list[dict[str, Any]] = []
    for source_id, meta in rows:
        if isinstance(meta, str) and meta:
            try:
                meta = json.loads(meta)
            except json.JSONDecodeError:
                meta = {}
        if not isinstance(meta, dict):
            meta = {}
        meta = dict(meta)
        meta["energy_above_hull_available"] = False
        meta["hull_comparable"] = False
        updates.append({"source_id": source_id, "original_metadata": json.dumps(meta)})
    if not updates:
        return 0
    frame = pd.DataFrame(updates)
    conn.register("aflow_meta", frame)
    conn.execute(
        """
        UPDATE source_records AS s
        SET original_metadata = m.original_metadata
        FROM aflow_meta AS m
        WHERE s.source = 'aflow' AND s.source_id = m.source_id
        """
    )
    # Ensure no AFLOW ehull rows exist (should already be none).
    conn.execute(
        """
        DELETE FROM properties
        WHERE source = 'aflow' AND name = 'energy_above_hull'
        """
    )
    conn.unregister("aflow_meta")
    print(f"AFLOW hull metadata updated: {len(updates):,}")
    return len(updates)


def repair_cod_space_groups(conn, *, settings) -> dict[str, int]:
    print("COD space groups: backfilling missing numbers...")
    path = settings.raw_source_dir("cod") / "materials.jsonl"
    rows: list[dict[str, Any]] = []
    filled_symbol = 0
    filled_cif = 0
    still_missing = 0
    for doc in iter_raw_cod_documents(path):
        source_id = str(doc.get("source_id") or "").strip()
        if not source_id:
            continue
        existing = _int_or_none(doc.get("sgNumber"))
        if existing is not None:
            continue
        sym = _symmetry_from_cod(doc, settings=settings, source_id=source_id)
        if sym.space_group_number is None:
            still_missing += 1
            continue
        # Heuristic: if raw sg was present, credit symbol path.
        if doc.get("sg"):
            filled_symbol += 1
        else:
            filled_cif += 1
        rows.append(
            {
                "source_id": source_id,
                "space_group_number": sym.space_group_number,
                "space_group_symbol": sym.space_group_symbol,
                "crystal_system": sym.crystal_system,
            }
        )
    if rows:
        frame = pd.DataFrame(rows)
        conn.register("cod_sg", frame)
        conn.execute(
            """
            UPDATE source_records AS s
            SET
                space_group_number = COALESCE(s.space_group_number, c.space_group_number),
                space_group_symbol = COALESCE(NULLIF(s.space_group_symbol, ''), c.space_group_symbol),
                crystal_system = COALESCE(NULLIF(s.crystal_system, ''), c.crystal_system)
            FROM cod_sg AS c
            WHERE s.source = 'cod' AND s.source_id = c.source_id
            """
        )
        conn.execute(
            """
            UPDATE search_index AS si
            SET
                space_group_number = COALESCE(si.space_group_number, c.space_group_number),
                space_group_symbol = COALESCE(NULLIF(si.space_group_symbol, ''), c.space_group_symbol),
                crystal_system = COALESCE(NULLIF(si.crystal_system, ''), c.crystal_system)
            FROM cod_sg AS c
            JOIN source_records AS s
              ON s.source = 'cod' AND s.source_id = c.source_id
            WHERE si.material_id = s.material_id
            """
        )
        conn.unregister("cod_sg")
    stats = {
        "filled": len(rows),
        "via_symbol_est": filled_symbol,
        "via_cif_est": filled_cif,
        "still_missing": still_missing,
    }
    print(f"COD space groups: {stats}")
    return stats


def _int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def repair_method_tags(conn) -> dict[str, int]:
    print("Method tags: updating property.method labels...")
    updated: dict[str, int] = {}
    for source, name, old_method, new_method in METHOD_UPDATES:
        if name is None:
            conn.execute(
                """
                UPDATE properties
                SET method = ?
                WHERE source = ? AND method = ?
                """,
                [new_method, source, old_method],
            )
        else:
            conn.execute(
                """
                UPDATE properties
                SET method = ?
                WHERE source = ? AND name = ? AND method = ?
                """,
                [new_method, source, name, old_method],
            )
        # DuckDB rowcount unreliable; count after.
        n = conn.execute(
            """
            SELECT COUNT(*) FROM properties
            WHERE source = ? AND method = ?
              AND (? IS NULL OR name = ?)
            """,
            [source, new_method, name, name],
        ).fetchone()[0]
        updated[f"{source}:{name or '*'}:{new_method}"] = int(n)
    print(f"Method tags present: { {k: v for k, v in updated.items() if v} }")
    return updated


def _verify(conn) -> dict[str, Any]:
    out: dict[str, Any] = {}
    out["aflow_ef_oor"] = conn.execute(
        """
        SELECT COUNT(*) FROM properties
        WHERE source='aflow' AND name='formation_energy_per_atom'
          AND (value_double < -10 OR value_double > 5)
        """
    ).fetchone()[0]
    out["any_density_oor"] = conn.execute(
        """
        SELECT COUNT(*) FROM properties
        WHERE name='density' AND (value_double < 0.01 OR value_double > 30)
        """
    ).fetchone()[0]
    out["cod_missing_sg"] = conn.execute(
        """
        SELECT COUNT(*) FROM source_records
        WHERE source='cod' AND space_group_number IS NULL
        """
    ).fetchone()[0]
    out["methods"] = dict(
        conn.execute(
            """
            SELECT method, COUNT(*) FROM properties
            GROUP BY 1 ORDER BY 2 DESC LIMIT 12
            """
        ).fetchall()
    )
    out["oqmd_struct"] = conn.execute(
        """
        SELECT COUNT(*) FROM source_records
        WHERE source='oqmd' AND structure_path IS NOT NULL AND structure_path != ''
        """
    ).fetchone()[0]
    out["aflow_ehull_props"] = conn.execute(
        """
        SELECT COUNT(*) FROM properties
        WHERE source='aflow' AND name='energy_above_hull'
        """
    ).fetchone()[0]
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair dataset quality items 6–10.")
    parser.add_argument(
        "--skip-enrich",
        action="store_true",
        help="Skip multi-source null coalescing after repairs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    clear_settings_cache()
    settings = get_settings()
    settings.ensure_directories()

    with duckdb_write_lock(settings, label="dataset-quality-6-10"):
        db = MaterialsDatabase(settings.duckdb_path)
        with db.connect() as conn:
            ensure_search_index_columns(conn)
            print("Before:", _verify(conn))
            repair_physical_qa(conn)
            repair_aflow_hull_metadata(conn)
            repair_cod_space_groups(conn, settings=settings)
            repair_method_tags(conn)
            backfill_oqmd_structure_paths(conn, settings=settings)
            if not args.skip_enrich:
                enrich_search_index_from_linked_sources(conn)
            print("After:", _verify(conn))

    print("Dataset quality repair (6–10) complete.")
    print(
        "Note: OQMD structure download is still resumable via "
        "`scripts/download_oqmd_structures.py` (API-limited)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
