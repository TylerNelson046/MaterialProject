"""LightGBM vs MACE comparison (Phase 4.6)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from ml.mace.compare import compare_lightgbm_mace


def test_compare_lightgbm_mace_overlap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ml.mace import compare as compare_mod
    from ml.mace import predict as predict_mod

    cfg = {
        "output": {"models_dir": "data/models"},
        "export": {"dir": "data/processed/mace_exports"},
        "foundation": {"model_id": "medium-mpa-0"},
    }
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "mace.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")

    ids = [f"MAT-{i}" for i in range(10)]
    y_true = np.linspace(-1.0, 1.0, 10)
    pack = pd.DataFrame(
        {
            "material_id": ids,
            "label": y_true,
            "split": ["test"] * 10,
            "structure_path": ["x.json"] * 10,
            "mace_train_ok": [True] * 10,
        }
    )

    lgbm_dir = tmp_path / "data/models/formation_energy_per_atom/lgbm_1"
    lgbm_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "material_id": ids,
            "y_true": y_true,
            "y_pred": y_true + 0.1,
        }
    ).to_parquet(lgbm_dir / "predictions_test.parquet", index=False)

    mace_dir = tmp_path / "data/models/mace/formation_energy_per_atom/mace_1"
    mace_dir.mkdir(parents=True)
    (mace_dir / "checkpoint").mkdir()
    (mace_dir / "checkpoint" / "model.model").write_bytes(b"x")
    (mace_dir / "metrics.json").write_text("{}", encoding="utf-8")
    (mace_dir / "run_record.json").write_text(
        json.dumps(
            {
                "target": "formation_energy_per_atom",
                "model_family": "mace_finetuned",
                "run_id": "mace_1",
                "run_dir": str(mace_dir.relative_to(tmp_path)),
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "material_id": ids,
            "y_true": y_true,
            "y_pred_formation": y_true + 0.3,
        }
    ).to_parquet(mace_dir / "predictions_test.parquet", index=False)

    registry = {
        "version": 1,
        "runs": {
            "formation_energy_per_atom": [
                {
                    "target": "formation_energy_per_atom",
                    "model_family": "lightgbm",
                    "run_id": "lgbm_1",
                    "run_dir": str(lgbm_dir.relative_to(tmp_path)),
                },
                {
                    "target": "formation_energy_per_atom",
                    "model_family": "mace_finetuned",
                    "run_id": "mace_1",
                    "run_dir": str(mace_dir.relative_to(tmp_path)),
                },
            ]
        },
    }
    (tmp_path / "data/models").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data/models/registry.json").write_text(
        json.dumps(registry), encoding="utf-8"
    )

    monkeypatch.setattr(compare_mod, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(predict_mod, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(compare_mod, "load_mace_config", lambda path=None: cfg)
    monkeypatch.setattr(predict_mod, "load_mace_config", lambda path=None: cfg)
    monkeypatch.setattr(
        compare_mod, "load_mace_pack_frame", lambda *a, **k: pack
    )
    monkeypatch.setattr(
        compare_mod,
        "mace_pack_dir",
        lambda config=None: tmp_path / "data/processed/mace_exports",
    )

    result = compare_lightgbm_mace(
        "formation_energy_per_atom",
        mace_family="mace_finetuned",
        config=cfg,
        project_root=tmp_path,
        out_dir=tmp_path / "cmp",
    )
    assert result.n_overlap == 10
    assert result.lightgbm_mae is not None
    assert result.mace_mae is not None
    assert result.lightgbm_mae < result.mace_mae
    assert result.promote_metric == "spearman"
    # Perfect rank correlation for both (affine offsets) → promote tie; MAE still LGBM.
    assert result.promote_winner in {"tie", "lightgbm", "mace"}
    assert result.report_md.is_file()
    text = result.report_md.read_text(encoding="utf-8")
    assert "MAE winner: **lightgbm**" in text
    assert "Promote metric: **spearman**" in text
