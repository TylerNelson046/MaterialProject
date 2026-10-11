"""Audit multi-source property coalesce (Step 3.1).

Verifies:
1. SQL enrich priority matches ``property_policy.SOURCE_PRIORITY``
2. Sampled multi-source SI values match ``coalesce_property`` (never the mean)
3. Hull energies are never attributable only to AFLOW/COD
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "scripts"))

from config.constants import CONFLICT_TRACKED_PROPERTIES  # noqa: E402
from config.settings import clear_settings_cache, get_settings  # noqa: E402
from database.duckdb import MaterialsDatabase  # noqa: E402
from database.property_policy import (  # noqa: E402
    HULL_COMPARABLE_SOURCES,
    SOURCE_PRIORITY,
    coalesce_property,
    is_hull_comparable,
)
from repair_dataset_enrichment import SOURCE_PRIORITY_SQL  # noqa: E402

SAMPLE_LIMIT = 5000
PROP_SAMPLE = (
    "band_gap",
    "formation_energy_per_atom",
    "energy_above_hull",
    "density",
    "bulk_modulus_vrh",
)


def _sql_priority_map() -> dict[str, int]:
    """Parse CASE arms from SOURCE_PRIORITY_SQL into a source→rank map."""
    mapping: dict[str, int] = {}
    for line in SOURCE_PRIORITY_SQL.splitlines():
        line = line.strip()
        if "WHEN '" not in line or "THEN" not in line:
            continue
        # WHEN 'materials_project' THEN 0
        src = line.split("'")[1]
        rank = int(line.rsplit("THEN", 1)[1].strip())
        mapping[src] = rank
    return mapping


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def main() -> int:
    clear_settings_cache()
    settings = get_settings()
    db = MaterialsDatabase(settings.duckdb_path, read_only=True)

    sql_map = _sql_priority_map()
    priority_mismatches = []
    for src, rank in SOURCE_PRIORITY.items():
        if sql_map.get(src) != rank:
            priority_mismatches.append(
                {"source": src, "policy_rank": rank, "sql_rank": sql_map.get(src)}
            )
    for src, rank in sql_map.items():
        if src not in SOURCE_PRIORITY:
            priority_mismatches.append(
                {"source": src, "policy_rank": None, "sql_rank": rank}
            )

    with db.connect() as conn:
        multi = conn.execute(
            """
            SELECT material_id, source AS primary_source, n_sources
            FROM search_index
            WHERE n_sources > 1
            ORDER BY material_id
            LIMIT ?
            """,
            [SAMPLE_LIMIT],
        ).fetchall()

        mismatches: list[dict[str, Any]] = []
        averaged_hits = 0
        checked = 0
        primary_wins = 0
        coalesce_fallback = 0
        hull_violations = 0

        for material_id, primary_source, _n in multi:
            si = conn.execute(
                f"""
                SELECT {", ".join(PROP_SAMPLE)}
                FROM search_index WHERE material_id = ?
                """,
                [material_id],
            ).fetchone()
            if si is None:
                continue
            si_map = dict(zip(PROP_SAMPLE, si, strict=True))

            prop_rows = conn.execute(
                """
                SELECT name, source, value_double
                FROM properties
                WHERE material_id = ?
                  AND name IN (?, ?, ?, ?, ?)
                  AND value_double IS NOT NULL
                """,
                [material_id, *PROP_SAMPLE],
            ).fetchall()
            by_prop: dict[str, dict[str, float]] = defaultdict(dict)
            for name, source, value in prop_rows:
                by_prop[str(name)][str(source)] = float(value)

            for prop in PROP_SAMPLE:
                values = by_prop.get(prop, {})
                if not values:
                    continue
                expected, winner = coalesce_property(
                    values,
                    primary_source=str(primary_source),
                    property_name=prop,
                )
                actual = si_map.get(prop)
                checked += 1
                if expected is None:
                    continue
                if actual is None:
                    # Null SI with available linked values is a fill miss, not averaging.
                    mismatches.append(
                        {
                            "material_id": material_id,
                            "property": prop,
                            "issue": "si_null_but_coalesce_available",
                            "expected": expected,
                            "winner": winner,
                            "by_source": values,
                        }
                    )
                    continue
                if abs(float(actual) - float(expected)) > 1e-9:
                    nums = list(values.values())
                    if len(nums) >= 2 and abs(float(actual) - _mean(nums)) < 1e-9:
                        averaged_hits += 1
                        issue = "matches_mean_not_coalesce"
                    else:
                        issue = "si_ne_coalesce_winner"
                    mismatches.append(
                        {
                            "material_id": material_id,
                            "property": prop,
                            "issue": issue,
                            "si_value": float(actual),
                            "expected": float(expected),
                            "winner": winner,
                            "by_source": values,
                        }
                    )
                else:
                    if winner == primary_source:
                        primary_wins += 1
                    else:
                        coalesce_fallback += 1

                if prop == "energy_above_hull" and actual is not None:
                    # Winner must be hull-comparable when any comparable source exists.
                    comparable = {
                        s: v for s, v in values.items() if is_hull_comparable(s)
                    }
                    if comparable and winner is not None and not is_hull_comparable(
                        winner
                    ):
                        hull_violations += 1
                        mismatches.append(
                            {
                                "material_id": material_id,
                                "property": prop,
                                "issue": "hull_winner_not_comparable",
                                "winner": winner,
                                "by_source": values,
                            }
                        )

        # Global: SI ehull rows whose only property rows are AFLOW/COD (no comparable).
        ehull_bad = conn.execute(
            f"""
            SELECT COUNT(*) FROM search_index AS si
            WHERE si.energy_above_hull IS NOT NULL
              AND NOT EXISTS (
                SELECT 1 FROM properties AS p
                WHERE p.material_id = si.material_id
                  AND p.name = 'energy_above_hull'
                  AND p.value_double IS NOT NULL
                  AND p.source IN ({",".join(repr(s) for s in sorted(HULL_COMPARABLE_SOURCES))})
              )
              AND EXISTS (
                SELECT 1 FROM properties AS p
                WHERE p.material_id = si.material_id
                  AND p.name = 'energy_above_hull'
                  AND p.value_double IS NOT NULL
                  AND p.source IN ('aflow', 'cod')
              )
            """
        ).fetchone()[0]

        n_multi = conn.execute(
            "SELECT COUNT(*) FROM search_index WHERE n_sources > 1"
        ).fetchone()[0]

    report = {
        "step": "3.1",
        "source_priority_policy": dict(
            sorted(SOURCE_PRIORITY.items(), key=lambda item: item[1])
        ),
        "source_priority_sql": sql_map,
        "priority_mismatches": priority_mismatches,
        "hull_comparable_sources": sorted(HULL_COMPARABLE_SOURCES),
        "conflict_tracked_properties": list(CONFLICT_TRACKED_PROPERTIES),
        "sample": {
            "multi_source_materials_total": int(n_multi),
            "materials_sampled": len(multi),
            "property_checks": checked,
            "primary_wins": primary_wins,
            "coalesce_fallback_wins": coalesce_fallback,
            "averaged_hits": averaged_hits,
            "hull_winner_violations_in_sample": hull_violations,
            "mismatch_count": len(mismatches),
            "mismatches_sample": mismatches[:50],
        },
        "global_hull_only_aflow_cod_on_si": int(ehull_bad),
        "verdict": (
            "PASS"
            if not priority_mismatches
            and averaged_hits == 0
            and hull_violations == 0
            and int(ehull_bad) == 0
            and len(mismatches) == 0
            else "REVIEW"
        ),
        "notes": (
            "Coalesce is priority null-fill only; values are never averaged. "
            "Alexandria/derived ranks in SQL enrich were aligned to property_policy "
            "as part of this step."
        ),
    }

    out = settings.processed_dir / "coalesce_audit.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md = settings.processed_dir / "coalesce_audit.md"
    md.write_text(
        "\n".join(
            [
                "# Coalesce audit (Step 3.1)",
                "",
                f"**Verdict:** {report['verdict']}",
                "",
                "## Priority order",
                "",
                "| Source | Rank |",
                "|--------|-----:|",
                *[
                    f"| `{s}` | {r} |"
                    for s, r in report["source_priority_policy"].items()
                ],
                "",
                f"- Priority SQL mismatches: **{len(priority_mismatches)}**",
                f"- Multi-source materials: **{n_multi:,}** (sampled {len(multi):,})",
                f"- Checks: {checked:,}; primary wins {primary_wins:,}; "
                f"fallback {coalesce_fallback:,}",
                f"- Averaged-value hits: **{averaged_hits}**",
                f"- Hull-only AFLOW/COD on SI: **{ehull_bad}**",
                f"- Sample mismatches: **{len(mismatches)}**",
                "",
                "Full JSON: `data/processed/coalesce_audit.json`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({k: report[k] for k in ("verdict", "sample", "priority_mismatches", "global_hull_only_aflow_cod_on_si")}, indent=2))
    print(f"Wrote {out}")
    print(f"Wrote {md}")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
