"""MACE predict helpers (Phase 4.6) — no torch forward required."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ml.mace.predict import _resolve_structure_inputs, latest_mace_run, load_mace_run


def _write_run(
    root: Path,
    *,
    family: str,
    run_id: str,
    with_checkpoint: bool = True,
) -> Path:
    run_dir = root / "data" / "models" / "mace" / "formation_energy_per_atom" / run_id
    run_dir.mkdir(parents=True)
    if with_checkpoint:
        ckpt = run_dir / "checkpoint"
        ckpt.mkdir()
        (ckpt / "model.model").write_bytes(b"fake")
    (run_dir / "metrics.json").write_text(
        json.dumps({"registry_metrics": {"test": {"mae": 0.5}}}),
        encoding="utf-8",
    )
    record = {
        "target": "formation_energy_per_atom",
        "model_family": family,
        "run_id": run_id,
        "run_dir": str(run_dir.relative_to(root)),
        "foundation_model": "medium-mpa-0",
        "metrics": {"test": {"mae": 0.5}},
    }
    (run_dir / "run_record.json").write_text(json.dumps(record), encoding="utf-8")
    return run_dir


def test_latest_and_load_mace_run(tmp_path: Path) -> None:
    cfg = {
        "output": {"models_dir": "data/models"},
        "foundation": {"model_id": "medium-mpa-0"},
        "energy": {"reference_protocol": "elemental_e0_v1"},
    }

    _write_run(tmp_path, family="mace_pretrained", run_id="pre_1")
    ft = _write_run(tmp_path, family="mace_finetuned", run_id="ft_1")

    registry = {
        "version": 1,
        "runs": {
            "formation_energy_per_atom": [
                {
                    "target": "formation_energy_per_atom",
                    "model_family": "mace_pretrained",
                    "run_id": "pre_1",
                    "run_dir": "data/models/mace/formation_energy_per_atom/pre_1",
                },
                {
                    "target": "formation_energy_per_atom",
                    "model_family": "mace_finetuned",
                    "run_id": "ft_1",
                    "run_dir": "data/models/mace/formation_energy_per_atom/ft_1",
                },
            ]
        },
    }
    (tmp_path / "data" / "models").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "models" / "registry.json").write_text(
        json.dumps(registry), encoding="utf-8"
    )

    latest_ft = latest_mace_run(
        "formation_energy_per_atom",
        model_family="mace_finetuned",
        config=cfg,
        project_root=tmp_path,
    )
    assert latest_ft["run_id"] == "ft_1"

    # Frozen-first default prefers pretrained when family is omitted.
    latest_default = latest_mace_run(
        "formation_energy_per_atom",
        model_family=None,
        config=cfg,
        project_root=tmp_path,
    )
    assert latest_default["run_id"] == "pre_1"

    loaded = load_mace_run(
        "formation_energy_per_atom",
        model_family="mace_finetuned",
        config=cfg,
        project_root=tmp_path,
    )
    assert loaded.run_id == "ft_1"
    assert loaded.checkpoint == ft / "checkpoint" / "model.model"
    assert loaded.metrics["registry_metrics"]["test"]["mae"] == 0.5


def test_resolve_structure_inputs_paths_and_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ml.mace import predict as predict_mod

    pack = pd.DataFrame(
        [
            {
                "material_id": "MAT-1",
                "structure_path": "data/structures/a.json",
                "split": "test",
                "label": 0.1,
                "mace_train_ok": True,
            }
        ]
    )
    monkeypatch.setattr(predict_mod, "load_mace_pack_frame", lambda *a, **k: pack)

    rows = _resolve_structure_inputs(
        structure_paths=["data/structures/b.json"],
        material_ids=["MAT-1"],
        target="formation_energy_per_atom",
        config={},
    )
    assert len(rows) == 2
    assert rows[0]["structure_path"] == "data/structures/b.json"
    assert rows[1]["material_id"] == "MAT-1"
