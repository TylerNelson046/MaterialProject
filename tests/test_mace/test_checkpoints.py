"""Checkpoint path helpers (no network unless forced)."""

from __future__ import annotations

from pathlib import Path

from ml.mace.checkpoints import (
    checkpoint_cache_dir,
    checkpoint_filename,
    checkpoint_url,
    ensure_foundation_checkpoint,
)


def test_checkpoint_url_for_default_model() -> None:
    url = checkpoint_url("medium-mpa-0")
    assert url.endswith("mace-mpa-0-medium.model")
    assert "mace_mpa_0" in url


def test_checkpoint_filename_from_url() -> None:
    assert (
        checkpoint_filename(
            "medium-mpa-0",
            "https://example.com/path/mace-mpa-0-medium.model",
        )
        == "mace-mpa-0-medium.model"
    )


def test_ensure_uses_existing_local_file(tmp_path: Path, monkeypatch) -> None:
    from ml.mace import checkpoints as ckpt_mod

    monkeypatch.setattr(ckpt_mod, "checkpoint_cache_dir", lambda config=None: tmp_path)
    local = tmp_path / "mace-mpa-0-medium.model"
    local.write_bytes(b"fake-weights")
    path = ensure_foundation_checkpoint("medium-mpa-0")
    assert path == local.resolve()


def test_cache_dir_default_under_data_models() -> None:
    cache = checkpoint_cache_dir()
    assert cache.parts[-4:] == ("data", "models", "mace", "checkpoints")
