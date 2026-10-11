"""Build or append normalized materials into DuckDB."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config.constants import DataSource  # noqa: E402
from config.settings import clear_settings_cache, get_settings  # noqa: E402
from database.duckdb import MaterialsDatabase  # noqa: E402
from database.queries import MaterialSearchFilters, search_materials  # noqa: E402
from normalization.normalize_aflow import iter_load_normalized_aflow  # noqa: E402
from normalization.normalize_cod import iter_load_normalized_cod  # noqa: E402
from normalization.normalize_jarvis import iter_load_normalized_jarvis  # noqa: E402
from normalization.normalize_mp import iter_load_normalized_mp  # noqa: E402
from normalization.normalize_oqmd import iter_load_normalized_oqmd  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the unified materials DuckDB.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop existing tables before loading.",
    )
    parser.add_argument(
        "--source",
        choices=["materials_project", "jarvis", "oqmd", "aflow", "cod", "all"],
        default="all",
        help="Which source(s) to load.",
    )
    parser.add_argument(
        "--replace-source",
        action="store_true",
        help="Delete existing rows for the selected source before insert.",
    )
    return parser.parse_args()


def load_mp(db: MaterialsDatabase) -> int:
    print("Loading Materials Project from index 1...")
    written = db.insert_materials(iter_load_normalized_mp(), batch_size=2000)
    print(f"Wrote {written} MP materials ({db.count_materials()} total in DB).")
    return written


def load_jarvis(db: MaterialsDatabase, *, replace: bool) -> int:
    if replace:
        removed = db.delete_source(DataSource.JARVIS.value)
        print(f"Removed {removed} existing JARVIS materials")
    start = db.next_material_index()
    print(f"Loading JARVIS starting at MAT index {start}...")
    written = db.insert_materials(
        iter_load_normalized_jarvis(start_index=start),
        batch_size=2000,
    )
    print(f"Wrote {written} JARVIS materials ({db.count_materials()} total in DB).")
    return written


def load_oqmd(db: MaterialsDatabase, *, replace: bool) -> int:
    if replace:
        removed = db.delete_source(DataSource.OQMD.value)
        print(f"Removed {removed} existing OQMD materials")
    start = db.next_material_index()
    print(f"Loading OQMD starting at MAT index {start}...")
    written = db.insert_materials(
        iter_load_normalized_oqmd(start_index=start),
        batch_size=2000,
    )
    print(f"Wrote {written} OQMD materials ({db.count_materials()} total in DB).")
    return written


def load_aflow(db: MaterialsDatabase, *, replace: bool) -> int:
    if replace:
        removed = db.delete_source(DataSource.AFLOW.value)
        print(f"Removed {removed} existing AFLOW materials")
    start = db.next_material_index()
    print(f"Loading AFLOW starting at MAT index {start}...")
    written = db.insert_materials(
        iter_load_normalized_aflow(start_index=start),
        batch_size=2000,
    )
    print(f"Wrote {written} AFLOW materials ({db.count_materials()} total in DB).")
    return written


def load_cod(db: MaterialsDatabase, *, replace: bool) -> int:
    if replace:
        removed = db.delete_source(DataSource.COD.value)
        print(f"Removed {removed} existing COD materials")
    start = db.next_material_index()
    print(f"Loading COD starting at MAT index {start}...")
    written = db.insert_materials(
        iter_load_normalized_cod(start_index=start),
        batch_size=2000,
    )
    print(f"Wrote {written} COD materials ({db.count_materials()} total in DB).")
    return written


def main() -> None:
    args = parse_args()
    clear_settings_cache()
    settings = get_settings()
    settings.ensure_directories()

    db = MaterialsDatabase(settings.duckdb_path)
    if args.reset or not settings.duckdb_path.exists():
        print(f"Resetting database at {settings.duckdb_path}")
        db.reset()
    else:
        db.initialize()

    if args.reset and args.source == "all":
        load_mp(db)
        load_jarvis(db, replace=False)
        load_oqmd(db, replace=False)
        load_aflow(db, replace=False)
        load_cod(db, replace=False)
    elif args.source == "materials_project":
        if not args.reset:
            raise SystemExit("Loading Materials Project without --reset would duplicate IDs.")
        load_mp(db)
    elif args.source == "jarvis":
        load_jarvis(db, replace=args.replace_source or True)
    elif args.source == "oqmd":
        load_oqmd(db, replace=args.replace_source or True)
    elif args.source == "aflow":
        load_aflow(db, replace=args.replace_source or True)
    elif args.source == "cod":
        load_cod(db, replace=args.replace_source or True)
    elif args.source == "all":
        load_jarvis(db, replace=False)
        load_oqmd(db, replace=False)
        load_aflow(db, replace=False)
        load_cod(db, replace=args.replace_source or True)

    sample = search_materials(
        MaterialSearchFilters(is_stable=True, band_gap_max=2.0, limit=3)
    )
    print(f"Smoke search (stable, band_gap<=2): {len(sample)} hit(s)")
    for row in sample:
        print(
            f"  {row['material_id']} {row['source']} {row['source_id']} "
            f"{row['formula_pretty']} Eg={row.get('band_gap')}"
        )


if __name__ == "__main__":
    main()
