"""Cross-source property resolution policy.

Rules (dataset quality):
1. Never silently average conflicting source values into ``search_index``.
2. ``search_index`` holds the **primary** source's value; nulls may be filled
   from the next-best linked source (priority order) — that is coalescing, not
   averaging.
3. Every source keeps its own ``properties`` row forever.
4. Detail views must expose per-source values and numeric spread for tracked
   properties (see ``CONFLICT_TRACKED_PROPERTIES``).
5. AFLOW (and COD) are **not hull-comparable** — never coalesce
   ``energy_above_hull`` from those sources (they do not provide it).
"""

from __future__ import annotations

from typing import Any

from config.constants import CONFLICT_TRACKED_PROPERTIES, DataSource

SOURCE_PRIORITY: dict[str, int] = {
    DataSource.MATERIALS_PROJECT.value: 0,
    DataSource.JARVIS.value: 1,
    DataSource.OQMD.value: 2,
    DataSource.AFLOW.value: 3,
    DataSource.COD.value: 4,
    DataSource.ALEXANDRIA.value: 5,
    DataSource.DERIVED.value: 6,
}

# Sources that publish a meaningful energy-above-hull on a comparable hull.
HULL_COMPARABLE_SOURCES: frozenset[str] = frozenset(
    {
        DataSource.MATERIALS_PROJECT.value,
        DataSource.JARVIS.value,
        DataSource.OQMD.value,
    }
)

# Preferred method tags written by normalizers (for docs / UI).
METHOD_TAGS: dict[str, str] = {
    "mp-gga-pbe": "Materials Project GGA/PBE",
    "jarvis-optb88vdw": "JARVIS OptB88vdW",
    "jarvis-mbj": "JARVIS mBJ meta-GGA",
    "jarvis-dfpt": "JARVIS DFPT",
    "jarvis-supercon": "JARVIS superconductivity",
    "jarvis-structure": "JARVIS structure-derived",
    "oqmd-dft": "OQMD DFT",
    "aflow-dft": "AFLOW DFT (no hull)",
    "cod-experimental": "COD experimental",
}


def source_rank(source: str) -> int:
    return SOURCE_PRIORITY.get(source, 99)


def is_hull_comparable(source: str) -> bool:
    return source in HULL_COMPARABLE_SOURCES


def coalesce_property(
    values_by_source: dict[str, Any],
    *,
    primary_source: str | None = None,
    property_name: str | None = None,
) -> tuple[Any, str | None]:
    """Pick one display value without averaging.

    Preference: primary source if it has a non-null value, else lowest
    ``source_rank`` among sources that have a value.
    """
    filtered = dict(values_by_source)
    if property_name == "energy_above_hull":
        filtered = {
            src: val
            for src, val in filtered.items()
            if is_hull_comparable(src) and val is not None
        }
    if primary_source and filtered.get(primary_source) is not None:
        return filtered[primary_source], primary_source
    ranked = sorted(
        ((source_rank(src), src, val) for src, val in filtered.items() if val is not None),
        key=lambda item: (item[0], item[1]),
    )
    if not ranked:
        return None, None
    _, src, val = ranked[0]
    return val, src


def conflict_summary(values_by_source: dict[str, float]) -> dict[str, Any] | None:
    """Build min/max/spread stats for one numeric property across sources."""
    cleaned = {
        src: float(val)
        for src, val in values_by_source.items()
        if val is not None and val == val  # drop NaN
    }
    if len(cleaned) < 2:
        return None
    nums = list(cleaned.values())
    lo, hi = min(nums), max(nums)
    return {
        "n_sources": len(cleaned),
        "min": lo,
        "max": hi,
        "spread": hi - lo,
        "by_source": cleaned,
        "tracked": True,
    }


def is_tracked_conflict_property(name: str) -> bool:
    return name in CONFLICT_TRACKED_PROPERTIES
