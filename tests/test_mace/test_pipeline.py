"""Phase 4.9 pipeline preset tests (frozen-first)."""

from __future__ import annotations

from ml.mace.pipeline import format_pipeline_plan, get_run_preset, plan_pipeline


def test_smoke_preset_has_limits() -> None:
    run = get_run_preset("smoke")
    assert run["device"] == "cpu"
    assert run["eval_limit_per_split"] == 200
    assert "formation_energy_per_atom" in run["pack_targets"]
    assert "band_gap" not in run["pack_targets"]


def test_full_preset_unbounded() -> None:
    run = get_run_preset("full")
    assert run["device"] == "auto"
    assert run["eval_limit_per_split"] is None
    assert "train_epochs" not in run or run.get("train_epochs") is None


def test_plan_pipeline_excludes_finetune_by_default() -> None:
    steps = plan_pipeline("smoke")
    names = [s.name for s in steps]
    assert "eval_pretrained" in names
    assert "eform_cache" in names
    assert "hull" in names
    assert "train_finetune" not in names
    assert not any(n.startswith("embedding") for n in names)


def test_plan_pipeline_dry_commands() -> None:
    steps = plan_pipeline("smoke", stages=("pack", "density"))
    names = [s.name for s in steps]
    assert any(n.startswith("pack:") for n in names)
    assert "density" in names
    text = format_pipeline_plan(steps)
    assert "export_mace_pack.py" in text
    assert "eval_mace_density.py" in text
