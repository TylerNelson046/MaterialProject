"""Tests for training exclusion / comparability encoding (Step 0.3)."""

from __future__ import annotations

from database.exclusion_rules import (
    CONFLICT_EXCLUDE_TARGETS,
    EXCLUDE_LABEL_CONFLICTS_V1,
    EXCLUSION_RULES_VERSION,
    export_rule_metadata,
    is_agl_thermal_method,
    is_excluded_deprecated,
    is_physically_ok,
    is_proxy_thermal_method,
    may_coalesce_hull_from,
    quarantine_value,
    resolve_without_averaging,
    sql_base_exclusion,
    sql_in_physical_range,
    sql_label_train_gate,
    sql_not_deprecated,
    sql_structure_present,
)


def test_deprecated_exclusion_keeps_false_and_null() -> None:
    assert is_excluded_deprecated(True) is True
    assert is_excluded_deprecated(False) is False
    assert is_excluded_deprecated(None) is False
    assert "IS NOT TRUE" in sql_not_deprecated()
    assert sql_base_exclusion() == sql_not_deprecated()


def test_structure_and_aliased_sql() -> None:
    assert sql_structure_present() == "(structure_path IS NOT NULL)"
    assert "si.deprecated" in sql_not_deprecated(table_alias="si")
    assert "si.structure_path" in sql_structure_present(table_alias="si")


def test_physical_quarantine_nulls_not_clips() -> None:
    assert is_physically_ok("density", 5.0) is True
    assert is_physically_ok("density", 0.0) is False
    assert is_physically_ok("density", 50.0) is False
    assert quarantine_value("density", 50.0) is None
    assert quarantine_value("band_gap", 1.1) == 1.1
    sql = sql_in_physical_range("density")
    assert "density IS NOT NULL" in sql
    assert ">= 0.01" in sql
    assert "<= 30.0" in sql


def test_label_train_gate_includes_base_and_range() -> None:
    gate = sql_label_train_gate("band_gap", table_alias="si")
    assert "si.deprecated IS NOT TRUE" in gate
    assert "si.band_gap IS NOT NULL" in gate
    assert "si.band_gap >= 0.0" in gate


def test_hull_comparability_and_no_averaging() -> None:
    assert may_coalesce_hull_from("materials_project") is True
    assert may_coalesce_hull_from("aflow") is False
    assert may_coalesce_hull_from("cod") is False
    value, source = resolve_without_averaging(
        {"aflow": 0.0, "jarvis": 0.05, "oqmd": 0.2},
        primary_source="aflow",
        property_name="energy_above_hull",
    )
    assert value == 0.05
    assert source == "jarvis"


def test_thermal_method_tags() -> None:
    assert is_agl_thermal_method("aflow-agl") is True
    assert is_proxy_thermal_method("element-table-kappa") is True
    assert is_agl_thermal_method("element-table-kappa") is False


def test_export_metadata_contract() -> None:
    meta = export_rule_metadata()
    assert meta["exclusion_rules_version"] == EXCLUSION_RULES_VERSION
    assert meta["never_average_sources"] is True
    assert meta["exclude_deprecated"] is True
    assert meta["exclude_label_conflicts_v1"] is True
    assert EXCLUDE_LABEL_CONFLICTS_V1 is True
    assert "band_gap" in meta["conflict_exclude_targets"]
    assert set(meta["conflict_exclude_targets"]) == set(CONFLICT_EXCLUDE_TARGETS)
    assert "materials_project" in meta["hull_comparable_sources"]
    assert "aflow" not in meta["hull_comparable_sources"]
    assert "density" in meta["physical_range_gates"]
