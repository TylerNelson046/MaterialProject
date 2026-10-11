"""Tests for cross-source property policy (no averaging)."""

from __future__ import annotations

from database.property_policy import (
    coalesce_property,
    conflict_summary,
    is_hull_comparable,
)


def test_coalesce_prefers_primary_when_present() -> None:
    value, source = coalesce_property(
        {"jarvis": 1.0, "materials_project": 2.0},
        primary_source="materials_project",
    )
    assert value == 2.0
    assert source == "materials_project"


def test_coalesce_falls_back_by_priority_without_averaging() -> None:
    value, source = coalesce_property(
        {"jarvis": 1.2, "aflow": 3.4},
        primary_source="materials_project",
    )
    assert value == 1.2
    assert source == "jarvis"


def test_coalesce_ehull_skips_non_comparable_sources() -> None:
    value, source = coalesce_property(
        {"aflow": 0.0, "cod": 0.1, "jarvis": 0.05},
        primary_source="aflow",
        property_name="energy_above_hull",
    )
    assert value == 0.05
    assert source == "jarvis"
    assert not is_hull_comparable("aflow")
    assert is_hull_comparable("materials_project")


def test_conflict_summary_reports_spread() -> None:
    summary = conflict_summary({"materials_project": 0.1, "jarvis": 0.6})
    assert summary is not None
    assert summary["n_sources"] == 2
    assert abs(summary["spread"] - 0.5) < 1e-9
    assert summary["min"] == 0.1
    assert summary["max"] == 0.6
