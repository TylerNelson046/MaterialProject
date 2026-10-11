"""Hard exclusion and comparability rules for training exports.

Authoritative prose: ``docs/exclusion_rules.md``.
Label tiers: ``docs/label_policy.md``.

These helpers encode exclusion policy so train flags and training exports share
one filter contract (no averaging, deprecated drop, hull comparability,
physical-range quarantine).
"""

from __future__ import annotations

from typing import Any

from database.property_policy import (
    HULL_COMPARABLE_SOURCES,
    SOURCE_PRIORITY,
    coalesce_property,
    is_hull_comparable,
)
from normalization.physical_ranges import PROPERTY_RANGE_GATES, finite_float, in_range

# Rule / policy version stamped into export metadata.
EXCLUSION_RULES_VERSION = "0.3.1"

# v1 supervised training: drop Ready-head rows with per-property label conflicts.
EXCLUDE_LABEL_CONFLICTS_V1 = True
CONFLICT_EXCLUDE_TARGETS: frozenset[str] = frozenset(
    {
        "energy_above_hull",
        "formation_energy_per_atom",
        "band_gap",
        "density",
    }
)

# Thermal κ method tags (mixed SI column until Steps 1.1–1.2).
THERMAL_AGL_METHOD = "aflow-agl"
THERMAL_PROXY_METHOD = "element-table-kappa"
THERMAL_AGL_METHODS: frozenset[str] = frozenset({THERMAL_AGL_METHOD})
THERMAL_PROXY_METHODS: frozenset[str] = frozenset({THERMAL_PROXY_METHOD})

# Elastic DFT / AEL methods acceptable as Niche ground truth.
ELASTIC_DFT_METHODS: frozenset[str] = frozenset(
    {
        "jarvis-optb88vdw",
        "mp-gga-pbe",
        "aflow-ael",
    }
)

# Battery electrode method.
BATTERY_METHOD = "mp-insertion-electrode"


def sql_not_deprecated(*, table_alias: str | None = None) -> str:
    """SQL predicate: keep rows that are not explicitly deprecated.

    ``deprecated IS NOT TRUE`` retains FALSE and NULL, drops TRUE (EX-DEP).
    """
    col = f"{table_alias}.deprecated" if table_alias else "deprecated"
    return f"({col} IS NOT TRUE)"


def sql_structure_present(*, table_alias: str | None = None) -> str:
    """SQL predicate for structure / graph model gates (EX-STRUCT)."""
    col = f"{table_alias}.structure_path" if table_alias else "structure_path"
    return f"({col} IS NOT NULL)"


def sql_base_exclusion(*, table_alias: str | None = None) -> str:
    """Minimum material-level gate for all supervised training exports."""
    return sql_not_deprecated(table_alias=table_alias)


def sql_in_physical_range(
    column: str,
    *,
    table_alias: str | None = None,
    property_name: str | None = None,
) -> str:
    """SQL predicate: column is non-null and inside the physical gate.

    Uses ``PROPERTY_RANGE_GATES`` keyed by ``property_name`` (defaults to
    ``column``). Raises ``KeyError`` if no gate is defined.
    """
    key = property_name or column
    lo, hi = PROPERTY_RANGE_GATES[key]
    col = f"{table_alias}.{column}" if table_alias else column
    return f"({col} IS NOT NULL AND {col} >= {lo} AND {col} <= {hi})"


def sql_label_train_gate(
    column: str,
    *,
    table_alias: str | None = None,
    property_name: str | None = None,
    require_physical_range: bool = True,
) -> str:
    """Base exclusion + label present (+ optional physical range)."""
    parts = [sql_base_exclusion(table_alias=table_alias)]
    col = f"{table_alias}.{column}" if table_alias else column
    key = property_name or column
    if require_physical_range and key in PROPERTY_RANGE_GATES:
        parts.append(
            sql_in_physical_range(
                column, table_alias=table_alias, property_name=key
            )
        )
    else:
        parts.append(f"({col} IS NOT NULL)")
    return " AND ".join(parts)


def is_excluded_deprecated(deprecated: bool | None) -> bool:
    """Return True if the material must be dropped (EX-DEP)."""
    return deprecated is True


def is_physically_ok(property_name: str, value: Any) -> bool:
    """True when value is finite and inside the quarantine gate (EX-OOR)."""
    if property_name not in PROPERTY_RANGE_GATES:
        number = finite_float(value)
        return number is not None
    lo, hi = PROPERTY_RANGE_GATES[property_name]
    return in_range(value, lo, hi) is not None


def quarantine_value(property_name: str, value: Any) -> float | None:
    """Null out-of-range values; never clip (EX-OOR)."""
    if property_name not in PROPERTY_RANGE_GATES:
        return finite_float(value)
    lo, hi = PROPERTY_RANGE_GATES[property_name]
    return in_range(value, lo, hi)


def is_agl_thermal_method(method: str | None) -> bool:
    """True when thermal κ / Debye / C_v may be used as Niche DFT-like labels."""
    return method in THERMAL_AGL_METHODS


def is_proxy_thermal_method(method: str | None) -> bool:
    """True when κ is element-mix proxy (EX-MIXED-KAPPA / Proxy tier)."""
    return method in THERMAL_PROXY_METHODS


def may_coalesce_hull_from(source: str) -> bool:
    """Whether ``source`` may contribute to energy_above_hull coalesce (CMP-HULL)."""
    return is_hull_comparable(source)


def resolve_without_averaging(
    values_by_source: dict[str, Any],
    *,
    primary_source: str | None = None,
    property_name: str | None = None,
) -> tuple[Any, str | None]:
    """Single-winner coalesce — never a mean (CMP-NO-AVG / CMP-COALESCE)."""
    return coalesce_property(
        values_by_source,
        primary_source=primary_source,
        property_name=property_name,
    )


def export_rule_metadata() -> dict[str, Any]:
    """Compact metadata block for Parquet/JSON training exports."""
    return {
        "exclusion_rules_version": EXCLUSION_RULES_VERSION,
        "exclude_deprecated": True,
        "quarantine_out_of_range": True,
        "never_average_sources": True,
        "exclude_label_conflicts_v1": EXCLUDE_LABEL_CONFLICTS_V1,
        "conflict_exclude_targets": sorted(CONFLICT_EXCLUDE_TARGETS),
        "hull_comparable_sources": sorted(HULL_COMPARABLE_SOURCES),
        "source_priority": dict(
            sorted(SOURCE_PRIORITY.items(), key=lambda item: item[1])
        ),
        "thermal_agl_method": THERMAL_AGL_METHOD,
        "thermal_proxy_method": THERMAL_PROXY_METHOD,
        "physical_range_gates": {
            name: {"lo": lo, "hi": hi} for name, (lo, hi) in PROPERTY_RANGE_GATES.items()
        },
    }
