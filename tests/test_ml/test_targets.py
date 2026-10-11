"""Tests for models.yaml target registry."""

from __future__ import annotations

from ml.targets import (
    get_target,
    known_targets,
    list_targets,
    load_models_config,
    niche_targets,
    ready_targets,
)


def test_load_models_config_has_export_and_targets() -> None:
    cfg = load_models_config()
    assert "export" in cfg
    assert "targets" in cfg
    assert cfg["training"]["default_tier"] == "Ready"


def test_ready_and_niche_partition() -> None:
    ready = ready_targets()
    niche = niche_targets()
    assert ready == (
        "energy_above_hull",
        "formation_energy_per_atom",
        "band_gap",
        "density",
    )
    assert "thermal_conductivity_agl" in niche
    assert set(ready).isdisjoint(niche)
    assert set(ready) | set(niche) == set(known_targets())


def test_get_target_band_gap() -> None:
    spec = get_target("band_gap")
    assert spec.tier == "Ready"
    assert spec.parquet == "band_gap.parquet"
    assert spec.label_col == "label"
    assert spec.is_ready


def test_list_targets_tier_filter() -> None:
    ready = list_targets(tier="Ready")
    assert all(s.is_ready for s in ready)
    assert len(ready) == 4
