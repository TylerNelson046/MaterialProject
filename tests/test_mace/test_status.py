"""Phase 4.9 status collector tests."""

from __future__ import annotations

from pathlib import Path

from ml.mace.status import collect_mace_status


def test_collect_mace_status_no_devices(tmp_path: Path, monkeypatch) -> None:
    from ml.mace import status as status_mod

    monkeypatch.setattr(status_mod, "PROJECT_ROOT", tmp_path)
    # Minimal layout so collector does not require real packs.
    (tmp_path / "data" / "models").mkdir(parents=True)
    (tmp_path / "data" / "models" / "registry.json").write_text(
        '{"version": 1, "runs": {}}', encoding="utf-8"
    )
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "mace.yaml").write_text(
        """
foundation:
  model_id: medium-mpa-0
  cache_dir: data/models/mace/checkpoints
output:
  models_dir: data/models
export:
  mace_pack_dir: data/processed/mace_exports
runs:
  smoke: {}
""",
        encoding="utf-8",
    )
    from ml.mace import config as config_mod

    monkeypatch.setattr(config_mod, "PROJECT_ROOT", tmp_path)
    config_mod.load_mace_config.cache_clear()
    monkeypatch.setattr(
        status_mod,
        "load_mace_config",
        lambda path=None: {
            "foundation": {"model_id": "medium-mpa-0", "cache_dir": "data/models/mace/checkpoints"},
            "output": {"models_dir": "data/models"},
            "export": {"mace_pack_dir": "data/processed/mace_exports"},
            "runs": {"smoke": {}},
        },
    )
    monkeypatch.setattr(status_mod, "foundation_model_id", lambda config=None: "medium-mpa-0")
    monkeypatch.setattr(
        status_mod,
        "mace_pack_dir",
        lambda config=None: tmp_path / "data/processed/mace_exports",
    )
    monkeypatch.setattr(
        status_mod,
        "checkpoint_cache_dir",
        lambda config=None: tmp_path / "data/models/mace/checkpoints",
    )

    status = collect_mace_status(
        project_root=tmp_path,
        include_devices=False,
        config={
            "foundation": {"model_id": "medium-mpa-0"},
            "output": {"models_dir": "data/models"},
            "export": {"mace_pack_dir": "data/processed/mace_exports"},
            "runs": {"smoke": {}},
        },
    )
    assert status["foundation_model_id"] == "medium-mpa-0"
    assert status["devices"] is None
    assert "smoke" in status["runs_presets"]
