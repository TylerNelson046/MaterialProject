"""Apply P0 dataset fixes without a full rebuild.

1. Sanitize Materials Project elastic / Poisson / anisotropy properties
2. Backfill OQMD space_group_number, crystal_system, and density
3. Wire ``structure_path`` for OQMD files already on disk
4. Optionally re-run entity resolution (recommended after SG/density fill)

Safe to re-run. Does not download new OQMD structures — use
``scripts/download_oqmd_structures.py`` for that.
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

from config.settings import clear_settings_cache, get_settings  # noqa: E402
from database.duckdb import MaterialsDatabase  # noqa: E402
from database.entity_resolution import resolve_entities  # noqa: E402
from database.write_lock import duckdb_write_lock  # noqa: E402
from normalization.normalize_mp import (  # noqa: E402
    _elastic_modulus_gpa,
    _poisson_ratio,
    _universal_anisotropy,
    iter_raw_mp_documents,
)
from normalization.normalize_oqmd import (  # noqa: E402
    _composition_from_oqmd,
    _density_from_composition_volume,
    _num,
    _symmetry_from_oqmd,
    iter_raw_oqmd_documents,
)
from database.fingerprints import space_group_number_from_symbol  # noqa: E402

ELASTIC_PROPS = (
    "bulk_modulus_vrh",
    "shear_modulus_vrh",
    "homogeneous_poisson",
    "universal_anisotropy",
)


def _raw_mp_path(settings) -> Path:
    raw_dir = settings.raw_source_dir("materials_project")
    parquet = raw_dir / "materials.parquet"
    jsonl = raw_dir / "materials.jsonl"
    if parquet.exists():
        return parquet
    if jsonl.exists():
        return jsonl
    raise FileNotFoundError(f"Raw MP data not found under {raw_dir}")


def repair_mp_elastic(conn, *, settings) -> dict[str, int]:
    """Re-extract MP elastic props with physical-range sanitization."""
    path = _raw_mp_path(settings)
    print(f"MP elastic: scanning {path}...")

    updates: list[dict[str, Any]] = []
    for doc in iter_raw_mp_documents(path):
        source_id = str(doc.get("material_id") or "").strip()
        if not source_id:
            continue
        bulk = _elastic_modulus_gpa(doc.get("bulk_modulus"))
        shear = _elastic_modulus_gpa(doc.get("shear_modulus"))
        poisson = _poisson_ratio(doc.get("homogeneous_poisson"))
        anisotropy = _universal_anisotropy(doc.get("universal_anisotropy"))
        # Only touch rows that had at least one elastic-related field in raw.
        if all(v is None for v in (bulk, shear, poisson, anisotropy)):
            raw_any = any(
                doc.get(k) is not None
                for k in (
                    "bulk_modulus",
                    "shear_modulus",
                    "homogeneous_poisson",
                    "universal_anisotropy",
                )
            )
            if not raw_any:
                continue
        updates.append(
            {
                "source_id": source_id,
                "bulk_modulus_vrh": bulk,
                "shear_modulus_vrh": shear,
                "homogeneous_poisson": poisson,
                "universal_anisotropy": anisotropy,
            }
        )

    if not updates:
        print("MP elastic: nothing to update")
        return {"scanned": 0, "deleted_bad": 0, "upserted": 0}

    frame = pd.DataFrame(updates)
    conn.register("mp_elastic", frame)

    # Drop existing elastic property rows for these materials, then re-insert
    # sanitized values (including dropping unrecoverable outliers as NULL/absent).
    conn.execute(
        """
        DELETE FROM properties
        WHERE source = 'materials_project'
          AND name IN (
            'bulk_modulus_vrh', 'shear_modulus_vrh',
            'homogeneous_poisson', 'universal_anisotropy'
          )
          AND source_id IN (SELECT source_id FROM mp_elastic)
        """
    )
    deleted_n = int(conn.rowcount or 0)

    units = {
        "bulk_modulus_vrh": "GPa",
        "shear_modulus_vrh": "GPa",
        "homogeneous_poisson": "dimensionless",
        "universal_anisotropy": "dimensionless",
    }
    upserted = 0
    for name, unit in units.items():
        conn.execute(
            f"""
            INSERT INTO properties
                (source, source_id, material_id, name, value_double, value_bool,
                 value_text, unit, origin, method, uncertainty)
            SELECT
                'materials_project',
                e.source_id,
                s.material_id,
                '{name}',
                e.{name},
                NULL,
                NULL,
                '{unit}',
                'known',
                'materials_project',
                NULL
            FROM mp_elastic AS e
            JOIN source_records AS s
              ON s.source = 'materials_project' AND s.source_id = e.source_id
            WHERE e.{name} IS NOT NULL
            """
        )
        upserted += int(conn.rowcount or 0)

    # Refresh search_index for MP-primary rows, then coalesce from linked sources.
    for name in ELASTIC_PROPS:
        conn.execute(
            f"""
            UPDATE search_index AS si
            SET {name} = e.{name}
            FROM mp_elastic AS e
            WHERE si.source = 'materials_project' AND si.source_id = e.source_id
            """
        )

    conn.unregister("mp_elastic")
    stats = {
        "scanned": len(updates),
        "deleted_bad": deleted_n,
        "upserted": upserted,
    }
    print(f"MP elastic: {stats}")
    return stats


def _build_oqmd_enrichment_frame(settings) -> tuple[pd.DataFrame, int]:
    """CPU-bound OQMD enrichment built outside the DuckDB write lock."""
    path = settings.raw_source_dir("oqmd") / "materials.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)

    # Warm SG cache once for unique symbols (pymatgen SpaceGroup is expensive).
    symbols: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            doc = json.loads(line)
            sg = doc.get("spacegroup")
            if sg:
                symbols.add(str(sg))
    for symbol in symbols:
        space_group_number_from_symbol(symbol)

    rows: list[dict[str, Any]] = []
    sg_fail = 0
    for doc in iter_raw_oqmd_documents(path):
        source_id = str(doc.get("source_id") or "").strip()
        if not source_id:
            entry_id = doc.get("entry_id")
            if entry_id is None:
                continue
            source_id = f"oqmd-{entry_id}"
        composition = _composition_from_oqmd(doc)
        sym = _symmetry_from_oqmd(doc)
        density = _density_from_composition_volume(
            composition.composition, _num(doc.get("volume"))
        )
        if sym.space_group_symbol and sym.space_group_number is None:
            sg_fail += 1
        rows.append(
            {
                "source_id": source_id,
                "space_group_symbol": sym.space_group_symbol,
                "space_group_number": sym.space_group_number,
                "crystal_system": sym.crystal_system,
                "density": density,
                "volume": _num(doc.get("volume")),
            }
        )
    return pd.DataFrame(rows), sg_fail


def repair_oqmd_symmetry_density(
    conn,
    *,
    settings,
    frame: pd.DataFrame | None = None,
    sg_fail: int = 0,
) -> dict[str, int]:
    """Fill OQMD SG number / crystal system / density from raw JSONL."""
    if frame is None:
        path = settings.raw_source_dir("oqmd") / "materials.jsonl"
        if not path.exists():
            print(f"OQMD repair: missing {path}")
            return {"rows": 0}
        print(f"OQMD SG/density: scanning {path}...")
        frame, sg_fail = _build_oqmd_enrichment_frame(settings)

    conn.register("oqmd_fix", frame)
    conn.execute(
        """
        UPDATE source_records AS s
        SET
            space_group_symbol = COALESCE(
                NULLIF(s.space_group_symbol, ''),
                f.space_group_symbol
            ),
            space_group_number = COALESCE(s.space_group_number, f.space_group_number),
            crystal_system = COALESCE(NULLIF(s.crystal_system, ''), f.crystal_system),
            density = COALESCE(s.density, f.density),
            volume = COALESCE(s.volume, f.volume)
        FROM oqmd_fix AS f
        WHERE s.source = 'oqmd' AND s.source_id = f.source_id
        """
    )

    conn.execute(
        """
        INSERT OR REPLACE INTO properties
            (source, source_id, material_id, name, value_double, value_bool,
             value_text, unit, origin, method, uncertainty)
        SELECT
            'oqmd',
            f.source_id,
            s.material_id,
            'density',
            f.density,
            NULL,
            NULL,
            'g/cm^3',
            'known',
            'oqmd_composition_volume',
            NULL
        FROM oqmd_fix AS f
        JOIN source_records AS s
          ON s.source = 'oqmd' AND s.source_id = f.source_id
        WHERE f.density IS NOT NULL
        """
    )

    conn.execute(
        """
        UPDATE search_index AS si
        SET
            space_group_symbol = COALESCE(
                NULLIF(si.space_group_symbol, ''),
                f.space_group_symbol
            ),
            space_group_number = COALESCE(si.space_group_number, f.space_group_number),
            crystal_system = COALESCE(NULLIF(si.crystal_system, ''), f.crystal_system),
            density = COALESCE(si.density, f.density),
            volume = COALESCE(si.volume, f.volume)
        FROM oqmd_fix AS f
        WHERE si.source = 'oqmd' AND si.source_id = f.source_id
        """
    )
    conn.execute(
        """
        UPDATE search_index AS si
        SET density = s.density
        FROM source_records AS s
        WHERE si.material_id = s.material_id
          AND s.source = 'oqmd'
          AND s.density IS NOT NULL
          AND si.density IS NULL
        """
    )
    conn.execute(
        """
        UPDATE search_index AS si
        SET
            space_group_number = COALESCE(si.space_group_number, s.space_group_number),
            space_group_symbol = COALESCE(
                NULLIF(si.space_group_symbol, ''),
                s.space_group_symbol
            ),
            crystal_system = COALESCE(NULLIF(si.crystal_system, ''), s.crystal_system)
        FROM source_records AS s
        WHERE si.material_id = s.material_id
          AND s.source = 'oqmd'
          AND s.space_group_number IS NOT NULL
          AND (
            si.space_group_number IS NULL
            OR si.space_group_symbol IS NULL
            OR si.space_group_symbol = ''
            OR si.crystal_system IS NULL
            OR si.crystal_system = ''
          )
        """
    )
    conn.unregister("oqmd_fix")

    with_sg = conn.execute(
        """
        SELECT COUNT(*) FROM source_records
        WHERE source='oqmd' AND space_group_number IS NOT NULL
        """
    ).fetchone()[0]
    with_dens = conn.execute(
        """
        SELECT COUNT(*) FROM source_records
        WHERE source='oqmd' AND density IS NOT NULL
        """
    ).fetchone()[0]
    stats = {
        "rows": len(frame),
        "sg_parse_failures": sg_fail,
        "oqmd_with_sg_number": int(with_sg),
        "oqmd_with_density": int(with_dens),
    }
    print(f"OQMD SG/density: {stats}")
    return stats


def backfill_oqmd_structure_paths(conn, *, settings) -> dict[str, int]:
    """Set structure_path for oqmd-*.json files already on disk."""
    struct_dir = settings.computational_structures_dir()
    path_rows: list[dict[str, str]] = []
    for path in struct_dir.glob("oqmd-*.json"):
        if path.stat().st_size <= 80:
            continue
        path_rows.append(
            {
                "source_id": path.stem,
                "structure_path": str(path.relative_to(settings.project_root)),
            }
        )
    if not path_rows:
        print("OQMD structures: no files on disk to backfill")
        return {"on_disk": 0, "linked": 0}

    frame = pd.DataFrame(path_rows)
    conn.register("oqmd_paths", frame)
    conn.execute(
        """
        UPDATE source_records AS s
        SET structure_path = p.structure_path
        FROM oqmd_paths AS p
        WHERE s.source = 'oqmd' AND s.source_id = p.source_id
        """
    )
    conn.execute(
        """
        UPDATE search_index AS si
        SET structure_path = p.structure_path
        FROM oqmd_paths AS p
        WHERE si.source = 'oqmd' AND si.source_id = p.source_id
        """
    )
    conn.execute(
        """
        UPDATE search_index AS si
        SET structure_path = s.structure_path
        FROM source_records AS s
        WHERE si.material_id = s.material_id
          AND s.source = 'oqmd'
          AND (si.structure_path IS NULL OR si.structure_path = '')
          AND s.structure_path IS NOT NULL
        """
    )
    conn.unregister("oqmd_paths")
    linked = conn.execute(
        """
        SELECT COUNT(*) FROM source_records
        WHERE source='oqmd'
          AND structure_path IS NOT NULL AND structure_path != ''
        """
    ).fetchone()[0]
    stats = {"on_disk": len(path_rows), "linked": int(linked)}
    print(f"OQMD structure paths: {stats}")
    return stats


def _verify(conn) -> dict[str, Any]:
    out: dict[str, Any] = {}
    out["mp_bulk_oor"] = conn.execute(
        """
        SELECT COUNT(*) FROM properties
        WHERE source='materials_project' AND name='bulk_modulus_vrh'
          AND (value_double < 0 OR value_double > 1000)
        """
    ).fetchone()[0]
    out["mp_shear_oor"] = conn.execute(
        """
        SELECT COUNT(*) FROM properties
        WHERE source='materials_project' AND name='shear_modulus_vrh'
          AND (value_double < 0 OR value_double > 1000)
        """
    ).fetchone()[0]
    out["mp_bulk_ok"] = conn.execute(
        """
        SELECT COUNT(*) FROM properties
        WHERE source='materials_project' AND name='bulk_modulus_vrh'
          AND value_double BETWEEN 0 AND 1000
        """
    ).fetchone()[0]
    out["oqmd_sg"] = conn.execute(
        "SELECT COUNT(*) FROM source_records WHERE source='oqmd' AND space_group_number IS NOT NULL"
    ).fetchone()[0]
    out["oqmd_density"] = conn.execute(
        "SELECT COUNT(*) FROM source_records WHERE source='oqmd' AND density IS NOT NULL"
    ).fetchone()[0]
    out["oqmd_struct"] = conn.execute(
        """
        SELECT COUNT(*) FROM source_records
        WHERE source='oqmd' AND structure_path IS NOT NULL AND structure_path != ''
        """
    ).fetchone()[0]
    out["si_density_null"] = conn.execute(
        "SELECT COUNT(*) FROM search_index WHERE density IS NULL"
    ).fetchone()[0]
    out["si_struct_null"] = conn.execute(
        "SELECT COUNT(*) FROM search_index WHERE structure_path IS NULL OR structure_path=''"
    ).fetchone()[0]
    out["si_sg_null"] = conn.execute(
        "SELECT COUNT(*) FROM search_index WHERE space_group_number IS NULL"
    ).fetchone()[0]
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair P0 dataset quality issues.")
    parser.add_argument(
        "--skip-entity-resolution",
        action="store_true",
        help="Do not re-run entity resolution after OQMD SG/density fill.",
    )
    parser.add_argument(
        "--only",
        choices=("elastic", "oqmd", "structures", "all"),
        default="all",
        help="Run a subset of repairs (default: all).",
    )
    return parser.parse_args()


def _refresh_oqmd_fields_on_search_index(conn) -> None:
    """Push already-repaired OQMD SG/density/structure onto search_index."""
    conn.execute(
        """
        UPDATE search_index AS si
        SET
            space_group_symbol = COALESCE(
                NULLIF(si.space_group_symbol, ''),
                s.space_group_symbol
            ),
            space_group_number = COALESCE(si.space_group_number, s.space_group_number),
            crystal_system = COALESCE(NULLIF(si.crystal_system, ''), s.crystal_system),
            density = COALESCE(si.density, s.density),
            volume = COALESCE(si.volume, s.volume),
            structure_path = COALESCE(NULLIF(si.structure_path, ''), s.structure_path)
        FROM source_records AS s
        WHERE si.material_id = s.material_id
          AND s.source = 'oqmd'
        """
    )


def main() -> int:
    args = parse_args()
    clear_settings_cache()
    settings = get_settings()
    settings.ensure_directories()

    oqmd_frame = None
    sg_fail = 0
    if args.only in {"oqmd", "all"}:
        print("Building OQMD SG/density enrichment frame (CPU, outside DB lock)...")
        oqmd_frame, sg_fail = _build_oqmd_enrichment_frame(settings)
        print(f"  prepared {len(oqmd_frame):,} rows (sg_parse_failures={sg_fail})")

    with duckdb_write_lock(settings, label="repair-p0"):
        db = MaterialsDatabase(settings.duckdb_path)
        with db.connect() as conn:
            print("Before:", _verify(conn))
            if args.only in {"elastic", "all"}:
                repair_mp_elastic(conn, settings=settings)
            if args.only in {"oqmd", "all"} and oqmd_frame is not None:
                repair_oqmd_symmetry_density(
                    conn, settings=settings, frame=oqmd_frame, sg_fail=sg_fail
                )
            if args.only in {"structures", "all"}:
                backfill_oqmd_structure_paths(conn, settings=settings)
            print("After repairs (pre-ER):", _verify(conn))

    if not args.skip_entity_resolution and args.only in {"oqmd", "all"}:
        print("Re-running entity resolution with improved OQMD fingerprints...")
        stats = resolve_entities(settings=settings)
        print("Entity resolution:", json.dumps(stats.to_dict(), indent=2, default=str))
        with duckdb_write_lock(settings, label="repair-p0-post-er"):
            db = MaterialsDatabase(settings.duckdb_path)
            with db.connect() as conn:
                backfill_oqmd_structure_paths(conn, settings=settings)
                _refresh_oqmd_fields_on_search_index(conn)
                print("Final:", _verify(conn))

    print("P0 repair complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
