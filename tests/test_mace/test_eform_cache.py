"""E_form cache path helpers."""

from __future__ import annotations

from pathlib import Path

from ml.mace.eform_cache import default_eform_cache_path
from ml.mace.reference import PROTOCOL_ELEMENTAL_BULK_V1


def test_default_eform_cache_path(tmp_path: Path) -> None:
    cfg = {
        "export": {"mace_pack_dir": str(tmp_path / "pack")},
        "foundation": {"model_id": "medium-mpa-0"},
    }
    path = default_eform_cache_path(
        "formation_energy_per_atom",
        protocol=PROTOCOL_ELEMENTAL_BULK_V1,
        config=cfg,
    )
    assert path.name == "eform_elemental_bulk_v1.parquet"
    assert "medium-mpa-0" in str(path)
    assert "formation_energy_per_atom" in str(path)
