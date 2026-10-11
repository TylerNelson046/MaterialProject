"""Dual-family leaderboard tests (Phase 4.6+ ranking-first)."""

from __future__ import annotations

from pathlib import Path

from ml.registry import register_run, write_leaderboard


def test_leaderboard_dual_family_section(tmp_path: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    register_run(
        models,
        {
            "target": "formation_energy_per_atom",
            "model_family": "lightgbm",
            "tier": "Ready",
            "run_id": "lgbm_1",
            "run_dir": "data/models/x/lgbm_1",
            "metrics": {
                "test": {
                    "mae": 0.16,
                    "rmse": 0.3,
                    "r2": 0.9,
                    "spearman_r": 0.95,
                }
            },
            "counts": {"train": 100, "test": 20},
        },
    )
    register_run(
        models,
        {
            "target": "formation_energy_per_atom",
            "model_family": "mace_finetuned",
            "tier": "Ready",
            "run_id": "mace_1",
            "run_dir": "data/models/x/mace_1",
            "metrics": {
                "test": {
                    "mae": 0.82,
                    "rmse": 1.1,
                    "r2": 0.5,
                    "spearman_r": 0.80,
                }
            },
            "counts": {"train": 64, "test": 32},
        },
    )
    text = (models / "LEADERBOARD.md").read_text(encoding="utf-8")
    assert "mace_finetuned" in text
    assert "Dual-family" in text
    assert "Spearman" in text
    assert "+0.66" in text  # MAE delta 0.82 - 0.16
    assert "-0.15" in text  # Spearman delta 0.80 - 0.95


def test_leaderboard_prefers_pretrained_frozen_first(tmp_path: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    for family, mae, spearman in (
        ("lightgbm", 0.2, 0.9),
        ("mace_pretrained", 0.27, 0.92),
        ("mace_finetuned", 0.9, 0.5),
    ):
        register_run(
            models,
            {
                "target": "formation_energy_per_atom",
                "model_family": family,
                "run_id": family,
                "run_dir": f"d/{family}",
                "metrics": {"test": {"mae": mae, "spearman_r": spearman}},
                "counts": {},
            },
        )
    text = write_leaderboard(models).read_text(encoding="utf-8")
    assert "mace_finetuned" in text
    # Frozen-first: dual section cites pretrained, not fine-tune.
    dual = text.split("## Dual-family")[1]
    assert "mace_pretrained" in dual
    assert "mace_finetuned" not in dual
    assert "0.92" in dual  # pretrained Spearman
    assert "0.27" in dual  # pretrained MAE
