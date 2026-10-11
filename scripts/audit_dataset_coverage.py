"""Dataset coverage audit: property fill rates, rare-element bias, COD SG gaps.

Writes ``data/processed/dataset_coverage_audit.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config.settings import clear_settings_cache, get_settings  # noqa: E402
from database.duckdb import SEARCH_PROP_NAMES, MaterialsDatabase  # noqa: E402

# Elements often under-represented vs common set in materials DBs.
RARE_FOCUS = (
    "Sc", "Y", "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er",
    "Tm", "Yb", "Lu", "Hf", "Ta", "Re", "Os", "Ir", "Tc", "Pm", "Ac", "Th", "U",
    "Np", "Pu",
)


def main() -> int:
    clear_settings_cache()
    settings = get_settings()
    db = MaterialsDatabase(settings.duckdb_path)
    report: dict = {}
    with db.connect() as conn:
        n_mat = conn.execute("SELECT COUNT(*) FROM search_index").fetchone()[0]
        report["search_index_rows"] = n_mat
        report["materials"] = conn.execute("SELECT COUNT(*) FROM materials").fetchone()[0]
        report["sources"] = dict(
            conn.execute(
                "SELECT source, COUNT(*) FROM source_records GROUP BY 1 ORDER BY 1"
            ).fetchall()
        )

        fill: dict[str, dict] = {}
        for name in SEARCH_PROP_NAMES:
            try:
                nonnull = conn.execute(
                    f"SELECT COUNT(*) FROM search_index WHERE {name} IS NOT NULL"
                ).fetchone()[0]
            except Exception:  # noqa: BLE001 - column may be missing
                continue
            fill[name] = {
                "non_null": int(nonnull),
                "fill_pct": round(100.0 * nonnull / n_mat, 2) if n_mat else 0.0,
            }
        report["search_index_fill"] = fill

        # Property table counts for domain packs
        domain_props = [
            "neutron_absorption_cross_section",
            "melting_point_element_mix",
            "thermal_conductivity",
            "average_voltage",
            "capacity_grav",
            "working_ion",
            "debye_temperature",
        ]
        report["domain_property_rows"] = {}
        for name in domain_props:
            report["domain_property_rows"][name] = conn.execute(
                "SELECT COUNT(*) FROM properties WHERE name = ?", [name]
            ).fetchone()[0]

        # Element frequency
        el_counts = dict(
            conn.execute(
                """
                SELECT element, COUNT(*) AS c
                FROM material_elements
                GROUP BY 1
                ORDER BY c DESC
                """
            ).fetchall()
        )
        report["n_unique_elements"] = len(el_counts)
        report["top_elements"] = dict(list(el_counts.items())[:20])
        rare = {el: el_counts.get(el, 0) for el in RARE_FOCUS}
        report["rare_element_material_counts"] = rare
        report["rare_elements_absent"] = [el for el, c in rare.items() if c == 0]

        report["cod_missing_space_group"] = conn.execute(
            """
            SELECT COUNT(*) FROM source_records
            WHERE source = 'cod' AND space_group_number IS NULL
            """
        ).fetchone()[0]
        report["oqmd_structure_coverage"] = dict(
            zip(
                ["with_path", "total"],
                conn.execute(
                    """
                    SELECT
                      SUM(CASE WHEN structure_path IS NOT NULL AND structure_path != '' THEN 1 ELSE 0 END),
                      COUNT(*)
                    FROM source_records WHERE source = 'oqmd'
                    """
                ).fetchone(),
            )
        )

        # Structure coverage all sources
        report["structure_coverage"] = {}
        for source, total, with_s in conn.execute(
            """
            SELECT source, COUNT(*),
              SUM(CASE WHEN structure_path IS NOT NULL AND structure_path != '' THEN 1 ELSE 0 END)
            FROM source_records GROUP BY 1 ORDER BY 1
            """
        ).fetchall():
            report["structure_coverage"][source] = {
                "total": total,
                "with_path": with_s,
                "pct": round(100.0 * with_s / total, 2) if total else 0.0,
            }

    out = settings.project_root / "data" / "processed" / "dataset_coverage_audit.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"Coverage audit: materials={report['materials']:,}, "
        f"search_rows={report['search_index_rows']:,}, "
        f"cod_missing_sg={report['cod_missing_space_group']}, "
        f"rare_absent={report['rare_elements_absent']}"
    )
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
