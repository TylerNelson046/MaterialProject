"""Cross-database entity resolution for unified materials.

Links source records that share a composition + space-group + nsites
fingerprint onto a single internal ``material_id``, preserving every source
row and its properties.

Resolution is idempotent: it first restores one material per source record,
then applies bulk merges.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from config.constants import DataSource
from config.settings import Settings, get_settings
from database.duckdb import SEARCH_PROP_NAMES, MaterialsDatabase, ensure_search_index_columns
from database.fingerprints import density_or_vpa_compatible, structure_fingerprint
from database.property_policy import coalesce_property, source_rank
from database.write_lock import duckdb_write_lock
from normalization.schema import make_material_id

SOURCE_PRIORITY: dict[str, int] = {
    DataSource.MATERIALS_PROJECT.value: 0,
    DataSource.JARVIS.value: 1,
    DataSource.OQMD.value: 2,
    DataSource.AFLOW.value: 3,
    DataSource.COD.value: 4,
    DataSource.ALEXANDRIA.value: 5,
    DataSource.DERIVED.value: 6,
}


@dataclass(frozen=True)
class ResolutionStats:
    source_records: int
    fingerprinted: int
    incomplete: int
    merge_groups: int
    materials_before: int
    materials_after: int
    sources_relinked: int
    multi_source_materials: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_records": self.source_records,
            "fingerprinted": self.fingerprinted,
            "incomplete": self.incomplete,
            "merge_groups": self.merge_groups,
            "materials_before": self.materials_before,
            "materials_after": self.materials_after,
            "sources_relinked": self.sources_relinked,
            "multi_source_materials": self.multi_source_materials,
        }


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _source_rank(source: str) -> int:
    return source_rank(source)


def _rows_compatible(
    a: dict[str, Any],
    b: dict[str, Any],
    *,
    density_rel_tol: float,
    density_abs_tol: float,
    vpa_rel_tol: float,
    vpa_abs_tol: float,
) -> bool:
    return density_or_vpa_compatible(
        density_a=a.get("density"),
        density_b=b.get("density"),
        volume_a=a.get("volume"),
        volume_b=b.get("volume"),
        nsites_a=a.get("nsites"),
        nsites_b=b.get("nsites"),
        density_rel_tol=density_rel_tol,
        density_abs_tol=density_abs_tol,
        vpa_rel_tol=vpa_rel_tol,
        vpa_abs_tol=vpa_abs_tol,
    )


def _connected_components(n: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i, j in edges:
        union(i, j)
    buckets: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        buckets[find(i)].append(i)
    return list(buckets.values())


def ensure_resolution_schema(db: MaterialsDatabase) -> None:
    with db.connect() as conn:
        ensure_search_index_columns(conn)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS entity_links (
                fingerprint VARCHAR NOT NULL,
                material_id VARCHAR NOT NULL,
                source VARCHAR NOT NULL,
                source_id VARCHAR NOT NULL,
                resolved_at VARCHAR NOT NULL,
                PRIMARY KEY (source, source_id)
            )
            """
        )


def _load_source_frame(db: MaterialsDatabase) -> pd.DataFrame:
    with db.connect() as conn:
        return conn.execute(
            """
            SELECT
                source,
                source_id,
                material_id,
                formula_pretty,
                reduced_formula,
                composition,
                space_group_symbol,
                space_group_number,
                nsites,
                crystal_system,
                structure_path,
                chemsys,
                elements,
                nelements,
                volume,
                density,
                retrieved_at,
                original_metadata
            FROM source_records
            ORDER BY
                CASE source
                    WHEN 'materials_project' THEN 0
                    WHEN 'jarvis' THEN 1
                    WHEN 'oqmd' THEN 2
                    WHEN 'aflow' THEN 3
                    WHEN 'cod' THEN 4
                    WHEN 'alexandria' THEN 5
                    ELSE 9
                END,
                source_id
            """
        ).fetchdf()


def _prop_lookup(db: MaterialsDatabase) -> dict[tuple[str, str], dict[str, Any]]:
    with db.connect() as conn:
        frame = conn.execute(
            """
            SELECT source, source_id, name, value_double, value_bool, value_text
            FROM properties
            """
        ).fetchdf()
    out: dict[tuple[str, str], dict[str, Any]] = defaultdict(dict)
    for row in frame.itertuples(index=False):
        value: Any
        if row.value_double is not None and not pd.isna(row.value_double):
            value = float(row.value_double)
        elif row.value_bool is not None and not pd.isna(row.value_bool):
            value = bool(row.value_bool)
        elif row.value_text is not None and not (
            isinstance(row.value_text, float) and pd.isna(row.value_text)
        ):
            value = row.value_text
        else:
            continue
        out[(str(row.source), str(row.source_id))][str(row.name)] = value
    return out


def resolve_entities(
    *,
    settings: Settings | None = None,
    dry_run: bool = False,
    require_cross_source: bool = True,
) -> ResolutionStats:
    """Merge cross-source duplicates onto shared material IDs (bulk, idempotent)."""
    settings = settings or get_settings()
    cfg = settings.database_config.get("entity_resolution", {})
    strategy = str(cfg.get("strategy") or "composition_and_structure_fingerprint")
    if strategy != "composition_and_structure_fingerprint":
        raise ValueError(f"Unsupported entity resolution strategy: {strategy}")
    density_rel_tol = float(cfg.get("density_rel_tol", 0.05))
    density_abs_tol = float(cfg.get("density_abs_tol", 0.15))
    vpa_rel_tol = float(cfg.get("vpa_rel_tol", 0.05))
    vpa_abs_tol = float(cfg.get("vpa_abs_tol", 0.5))

    db = MaterialsDatabase(settings.duckdb_path)
    db.initialize()
    ensure_resolution_schema(db)

    materials_before = db.count_materials()
    frame = _load_source_frame(db)
    rows = frame.to_dict(orient="records")

    fingerprints: list[str | None] = []
    incomplete = 0
    for row in rows:
        fp = structure_fingerprint(
            composition=row.get("composition"),
            formula_pretty=row.get("formula_pretty"),
            reduced_formula=row.get("reduced_formula"),
            space_group_number=row.get("space_group_number"),
            space_group_symbol=row.get("space_group_symbol"),
            nsites=row.get("nsites"),
        )
        if fp is None:
            incomplete += 1
        fingerprints.append(fp)

    groups: dict[str, list[int]] = defaultdict(list)
    for idx, fp in enumerate(fingerprints):
        if fp is not None:
            groups[fp].append(idx)

    merge_groups: list[tuple[str, list[int]]] = []
    for fp, indices in groups.items():
        if len(indices) < 2:
            continue
        by_source: dict[str, list[int]] = defaultdict(list)
        for i in indices:
            by_source[str(rows[i]["source"])].append(i)
        if require_cross_source and len(by_source) < 2:
            continue
        # One representative per database — avoids collapsing dozens of
        # same-chemistry MP/JARVIS variants into a single entity.
        representatives: list[int] = []
        for _source, idxs in by_source.items():
            best = sorted(
                idxs,
                key=lambda i: (
                    0 if rows[i].get("structure_path") else 1,
                    0
                    if rows[i].get("density") is not None
                    and not (
                        isinstance(rows[i].get("density"), float)
                        and str(rows[i].get("density")) == "nan"
                    )
                    else 1,
                    str(rows[i]["source_id"]),
                ),
            )[0]
            representatives.append(best)
        if len(representatives) < 2:
            continue

        # Soft density / VPA gate: only merge reps that agree within tolerance.
        # Connected components allow e.g. MP↔JARVIS while leaving a mismatched
        # AFLOW density as its own material.
        edges: list[tuple[int, int]] = []
        for i in range(len(representatives)):
            for j in range(i + 1, len(representatives)):
                if _rows_compatible(
                    rows[representatives[i]],
                    rows[representatives[j]],
                    density_rel_tol=density_rel_tol,
                    density_abs_tol=density_abs_tol,
                    vpa_rel_tol=vpa_rel_tol,
                    vpa_abs_tol=vpa_abs_tol,
                ):
                    edges.append((i, j))
        for component in _connected_components(len(representatives), edges):
            if len(component) < 2:
                continue
            merge_groups.append((fp, [representatives[k] for k in component]))

    # Assign identity: start 1:1, then collapse merge groups onto a canonical id.
    assigned = [make_material_id(i + 1) for i in range(len(rows))]
    sources_relinked = 0
    for fp, indices in merge_groups:
        canonical_idx = sorted(
            indices,
            key=lambda i: (
                _source_rank(str(rows[i]["source"])),
                str(rows[i]["material_id"]),
                str(rows[i]["source_id"]),
            ),
        )[0]
        canonical_id = assigned[canonical_idx]
        for i in indices:
            if assigned[i] != canonical_id:
                sources_relinked += 1
            assigned[i] = canonical_id

    from collections import Counter

    counts = Counter(assigned)
    unique_material_ids = sorted(counts.keys(), key=lambda mid: int(mid.split("-", 1)[1]))
    materials_after = len(unique_material_ids)
    multi_source_materials = sum(1 for n in counts.values() if n > 1)

    if dry_run:
        return ResolutionStats(
            source_records=len(rows),
            fingerprinted=len(rows) - incomplete,
            incomplete=incomplete,
            merge_groups=len(merge_groups),
            materials_before=materials_before,
            materials_after=materials_after,
            sources_relinked=sources_relinked,
            multi_source_materials=multi_source_materials,
        )

    print(
        f"Entity resolution: {len(merge_groups):,} merge groups, "
        f"{materials_after:,} materials after "
        f"(from {materials_before:,})...",
        flush=True,
    )

    resolved_at = _utc_now()
    props = _prop_lookup(db)

    # Build replacement tables in memory, then swap in one write transaction.
    id_map_rows = [
        {
            "source": rows[i]["source"],
            "source_id": rows[i]["source_id"],
            "new_material_id": assigned[i],
            "fingerprint": fingerprints[i],
        }
        for i in range(len(rows))
    ]

    material_rows: list[dict[str, Any]] = []
    search_rows: list[dict[str, Any]] = []
    element_rows: list[dict[str, Any]] = []
    link_rows: list[dict[str, Any]] = []

    members_by_id: dict[str, list[int]] = defaultdict(list)
    for i, mid in enumerate(assigned):
        members_by_id[mid].append(i)

    for mid in unique_material_ids:
        indices = members_by_id[mid]
        member_rows = [rows[i] for i in indices]
        primary = sorted(
            member_rows,
            key=lambda r: (
                _source_rank(str(r["source"])),
                0 if r.get("structure_path") else 1,
                str(r["source_id"]),
            ),
        )[0]
        material_rows.append(
            {
                "material_id": mid,
                "formula_pretty": primary.get("formula_pretty") or "unknown",
                "chemsys": primary.get("chemsys"),
                "nelements": primary.get("nelements"),
                "created_at": resolved_at,
                "updated_at": resolved_at,
                "notes": (
                    f"entity_resolved:{fingerprints[indices[0]]}"
                    if len(indices) > 1 and fingerprints[indices[0]]
                    else None
                ),
            }
        )

        primary_props = props.get((str(primary["source"]), str(primary["source_id"])), {})
        # Multi-source resolution: never average. Start from primary; fill nulls
        # from next-best linked source (priority order).
        member_prop_maps = [
            props.get((str(rows[i]["source"]), str(rows[i]["source_id"])), {})
            for i in indices
        ]
        elements_raw = primary.get("elements")
        if not isinstance(elements_raw, str):
            elements_raw = json.dumps(elements_raw) if elements_raw is not None else "[]"

        search_row: dict[str, Any] = {
            "material_id": mid,
            "source": primary["source"],
            "source_id": primary["source_id"],
            "formula_pretty": primary.get("formula_pretty") or "unknown",
            "chemsys": primary.get("chemsys"),
            "elements": elements_raw,
            "nelements": primary.get("nelements"),
            "nsites": primary.get("nsites"),
            "crystal_system": primary.get("crystal_system"),
            "space_group_symbol": primary.get("space_group_symbol"),
            "space_group_number": primary.get("space_group_number"),
            "structure_path": primary.get("structure_path"),
            "n_sources": len(indices),
        }
        for name in SEARCH_PROP_NAMES:
            by_source: dict[str, Any] = {}
            for member_idx, prop_map in zip(indices, member_prop_maps):
                val = prop_map.get(name)
                if val is not None:
                    by_source[str(rows[member_idx]["source"])] = val
            # Prefer primary's own value when present.
            if primary_props.get(name) is not None:
                search_row[name] = primary_props.get(name)
            else:
                value, _src = coalesce_property(
                    by_source,
                    primary_source=str(primary["source"]),
                    property_name=name,
                )
                search_row[name] = value
        # Deprecated: true if ANY linked source marks deprecated (usually MP).
        deprecated_flags = [
            member_prop_maps[j].get("deprecated")
            for j in range(len(member_prop_maps))
        ]
        if any(flag is True for flag in deprecated_flags):
            search_row["deprecated"] = True
        elif any(flag is False for flag in deprecated_flags):
            search_row["deprecated"] = False
        else:
            search_row["deprecated"] = search_row.get("deprecated")
        search_rows.append(search_row)

        seen_els: set[str] = set()
        for member in member_rows:
            raw = member.get("elements")
            parsed: list[str] = []
            if isinstance(raw, str):
                try:
                    loaded = json.loads(raw)
                    if isinstance(loaded, list):
                        parsed = [str(x) for x in loaded]
                except json.JSONDecodeError:
                    parsed = []
            elif isinstance(raw, list):
                parsed = [str(x) for x in raw]
            seen_els.update(parsed)
        for el in sorted(seen_els):
            element_rows.append({"material_id": mid, "element": el})

        if len(indices) > 1:
            for i in indices:
                if fingerprints[i] is None:
                    continue
                link_rows.append(
                    {
                        "fingerprint": fingerprints[i],
                        "material_id": mid,
                        "source": rows[i]["source"],
                        "source_id": rows[i]["source_id"],
                        "resolved_at": resolved_at,
                    }
                )

    with duckdb_write_lock(settings, label="entity-resolution"):
        with db.connect() as conn:
            print("  writing identity map + rebuilding tables...", flush=True)
            conn.register("er_map", pd.DataFrame(id_map_rows))
            conn.execute(
                """
                UPDATE source_records AS s
                SET material_id = m.new_material_id
                FROM er_map AS m
                WHERE s.source = m.source AND s.source_id = m.source_id
                """
            )
            conn.execute(
                """
                UPDATE properties AS p
                SET material_id = m.new_material_id
                FROM er_map AS m
                WHERE p.source = m.source AND p.source_id = m.source_id
                """
            )
            conn.unregister("er_map")

            conn.execute("DELETE FROM materials")
            conn.execute("DELETE FROM search_index")
            conn.execute("DELETE FROM material_elements")
            conn.execute("DELETE FROM entity_links")

            MaterialsDatabase._df_replace(conn, "materials", material_rows)
            MaterialsDatabase._df_replace(conn, "search_index", search_rows)
            MaterialsDatabase._df_replace(conn, "material_elements", element_rows)
            MaterialsDatabase._df_replace(conn, "entity_links", link_rows)

            materials_after = conn.execute("SELECT COUNT(*) FROM materials").fetchone()[0]
            multi = conn.execute(
                """
                SELECT COUNT(*) FROM (
                    SELECT material_id
                    FROM source_records
                    GROUP BY material_id
                    HAVING COUNT(*) > 1
                )
                """
            ).fetchone()[0]

    manifest = {
        "resolved_at": resolved_at,
        "strategy": strategy,
        "fingerprint": "reduced_composition|space_group_number|nsites",
        "require_cross_source": require_cross_source,
        "stats": {
            "source_records": len(rows),
            "fingerprinted": len(rows) - incomplete,
            "incomplete": incomplete,
            "merge_groups": len(merge_groups),
            "materials_before": materials_before,
            "materials_after": int(materials_after),
            "sources_relinked": sources_relinked,
            "multi_source_materials": int(multi),
        },
    }
    out_dir = Path(settings.duckdb_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "entity_resolution_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print("Entity resolution write complete.", flush=True)

    return ResolutionStats(
        source_records=len(rows),
        fingerprinted=len(rows) - incomplete,
        incomplete=incomplete,
        merge_groups=len(merge_groups),
        materials_before=materials_before,
        materials_after=int(materials_after),
        sources_relinked=sources_relinked,
        multi_source_materials=int(multi),
    )
