"""Ingest Alexandria PBE on-hull catalog into DuckDB (scoped, ~117k).

Creates new materials (does not rebuild full entity resolution). Links are
1:1 alexandria source → new material_id. Idempotent: deletes prior alexandria
rows before re-insert.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config.constants import DataSource  # noqa: E402
from config.settings import clear_settings_cache, get_settings  # noqa: E402
from database.duckdb import MaterialsDatabase, ensure_search_index_columns  # noqa: E402
from database.write_lock import duckdb_write_lock  # noqa: E402
from normalization.normalize_alexandria import (  # noqa: E402
    iter_raw_alexandria_documents,
    normalize_alexandria_document,
)
from normalization.schema import make_material_id, utc_now_iso  # noqa: E402

SOURCE = DataSource.ALEXANDRIA.value


def _prop_frame_rows(material_id: str, src) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prop in src.properties:
        value = prop.value
        rows.append(
            {
                "source": SOURCE,
                "source_id": src.source_id,
                "material_id": material_id,
                "name": prop.name,
                "value_double": float(value)
                if isinstance(value, (int, float)) and not isinstance(value, bool)
                else None,
                "value_bool": value if isinstance(value, bool) else None,
                "value_text": value if isinstance(value, str) else None,
                "unit": prop.unit,
                "origin": prop.origin,
                "method": prop.method,
                "uncertainty": prop.uncertainty,
            }
        )
    return rows


def main() -> int:
    clear_settings_cache()
    settings = get_settings()
    path = settings.project_root / "data" / "raw" / "alexandria" / "alex_pbe_hull.jsonl"
    if not path.exists():
        raise SystemExit(f"Missing {path}; download alex_pbe_hull first.")

    now = utc_now_iso()
    materials: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    props: list[dict[str, Any]] = []
    elements: list[dict[str, Any]] = []
    search_rows: list[dict[str, Any]] = []

    with duckdb_write_lock(settings, label="alexandria-hull"):
        db = MaterialsDatabase(settings.duckdb_path)
        with db.connect() as conn:
            ensure_search_index_columns(conn)
            max_id = conn.execute("SELECT MAX(material_id) FROM materials").fetchone()[0]
            next_idx = 1
            if max_id and str(max_id).startswith("MAT-"):
                next_idx = int(str(max_id).split("-", 1)[1]) + 1

            # Remove prior alexandria ingest (idempotent).
            old_mids = [
                r[0]
                for r in conn.execute(
                    "SELECT DISTINCT material_id FROM source_records WHERE source = ?",
                    [SOURCE],
                ).fetchall()
            ]
            if old_mids:
                conn.execute("DELETE FROM properties WHERE source = ?", [SOURCE])
                conn.execute("DELETE FROM source_records WHERE source = ?", [SOURCE])
                # Only drop materials that have no remaining sources.
                conn.execute(
                    """
                    DELETE FROM search_index
                    WHERE material_id IN (
                        SELECT m.material_id FROM materials m
                        WHERE m.material_id NOT IN (SELECT DISTINCT material_id FROM source_records)
                    )
                    """
                )
                conn.execute(
                    """
                    DELETE FROM material_elements
                    WHERE material_id NOT IN (SELECT material_id FROM materials)
                       OR material_id NOT IN (SELECT DISTINCT material_id FROM source_records)
                    """
                )
                conn.execute(
                    """
                    DELETE FROM materials
                    WHERE material_id NOT IN (SELECT DISTINCT material_id FROM source_records)
                    """
                )
                max_id = conn.execute("SELECT MAX(material_id) FROM materials").fetchone()[0]
                next_idx = 1
                if max_id and str(max_id).startswith("MAT-"):
                    next_idx = int(str(max_id).split("-", 1)[1]) + 1

            skipped = 0
            for doc in iter_raw_alexandria_documents(path):
                src = normalize_alexandria_document(doc)
                if src is None:
                    skipped += 1
                    continue
                mid = make_material_id(next_idx)
                next_idx += 1
                struct = src.structure
                nsites = struct.nsites if struct else None
                volume = struct.volume if struct else None
                density = struct.density if struct else None
                materials.append(
                    {
                        "material_id": mid,
                        "formula_pretty": src.composition.formula_pretty,
                        "chemsys": src.composition.chemsys,
                        "nelements": src.composition.nelements,
                        "created_at": now,
                        "updated_at": now,
                        "notes": "alexandria alex_pbe_hull",
                    }
                )
                sources.append(
                    {
                        "source": SOURCE,
                        "source_id": src.source_id,
                        "material_id": mid,
                        "formula_pretty": src.composition.formula_pretty,
                        "reduced_formula": src.composition.reduced_formula,
                        "chemsys": src.composition.chemsys,
                        "elements": json.dumps(list(src.composition.elements)),
                        "nelements": src.composition.nelements,
                        "composition": json.dumps(src.composition.composition),
                        "crystal_system": None,
                        "space_group_symbol": None,
                        "space_group_number": None,
                        "point_group": None,
                        "nsites": nsites,
                        "volume": volume,
                        "density": density,
                        "structure_path": None,
                        "original_metadata": json.dumps(src.original_metadata),
                        "retrieved_at": src.retrieved_at,
                    }
                )
                props.extend(_prop_frame_rows(mid, src))
                for el in src.composition.elements:
                    elements.append({"material_id": mid, "element": el})

                prop_map = {p.name: p.value for p in src.properties}
                search_rows.append(
                    {
                        "material_id": mid,
                        "source": SOURCE,
                        "source_id": src.source_id,
                        "formula_pretty": src.composition.formula_pretty,
                        "chemsys": src.composition.chemsys,
                        "elements": json.dumps(list(src.composition.elements)),
                        "nelements": src.composition.nelements,
                        "nsites": nsites,
                        "volume": prop_map.get("volume"),
                        "density": prop_map.get("density"),
                        "formation_energy_per_atom": None,
                        "energy_above_hull": prop_map.get("energy_above_hull"),
                        "energy_per_atom": prop_map.get("energy_per_atom"),
                        "band_gap": None,
                        "is_gap_direct": None,
                        "efermi": None,
                        "cbm": None,
                        "vbm": None,
                        "total_magnetization": None,
                        "total_magnetization_normalized_vol": None,
                        "num_magnetic_sites": None,
                        "ordering": None,
                        "bulk_modulus_vrh": None,
                        "shear_modulus_vrh": None,
                        "homogeneous_poisson": None,
                        "universal_anisotropy": None,
                        "dielectric_total": None,
                        "dielectric_ionic": None,
                        "dielectric_electronic": None,
                        "refractive_index": None,
                        "weighted_surface_energy": None,
                        "weighted_work_function": None,
                        "is_stable": True,
                        "is_metal": None,
                        "is_magnetic": None,
                        "theoretical": True,
                        "crystal_system": None,
                        "space_group_symbol": None,
                        "space_group_number": None,
                        "structure_path": None,
                        "n_sources": 1,
                        "deprecated": False,
                    }
                )

            print(
                f"Prepared alexandria rows: materials={len(materials):,}, "
                f"props={len(props):,}, skipped={skipped:,}"
            )

            for name, rows in (
                ("materials", materials),
                ("source_records", sources),
                ("properties", props),
                ("material_elements", elements),
            ):
                if not rows:
                    continue
                frame = pd.DataFrame(rows)
                conn.register(f"alex_{name}", frame)
                conn.execute(f"INSERT INTO {name} SELECT * FROM alex_{name}")
                conn.unregister(f"alex_{name}")

            if search_rows:
                # Align to existing search_index columns.
                cols = [
                    r[1]
                    for r in conn.execute("PRAGMA table_info('search_index')").fetchall()
                ]
                frame = pd.DataFrame(search_rows)
                for col in cols:
                    if col not in frame.columns:
                        frame[col] = None
                frame = frame[cols]
                conn.register("alex_si", frame)
                conn.execute("INSERT INTO search_index SELECT * FROM alex_si")
                conn.unregister("alex_si")

            n_src = conn.execute(
                "SELECT COUNT(*) FROM source_records WHERE source = ?", [SOURCE]
            ).fetchone()[0]
            n_mat = conn.execute("SELECT COUNT(*) FROM materials").fetchone()[0]
            print(f"Alexandria ingest done: source_records={n_src:,}, materials_total={n_mat:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
