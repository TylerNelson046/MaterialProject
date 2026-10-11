"""Tests for configs/mace.yaml loading (no torch required)."""

from __future__ import annotations

from ml.mace.config import device_preference, foundation_model_id, load_mace_config


def test_load_mace_config_contract() -> None:
    cfg = load_mace_config()
    assert cfg["contract_version"] == "4.0.0"
    assert cfg["targets"]["primary"] == "formation_energy_per_atom"
    assert cfg["splits"]["allow_resplit"] is False
    assert cfg["training"]["energy_only"] is True


def test_foundation_model_id_pinned() -> None:
    assert foundation_model_id() == "medium-mpa-0"


def test_device_preference_order() -> None:
    assert device_preference() == ("cuda", "mps", "cpu")
