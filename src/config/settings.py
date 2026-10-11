"""Application-wide settings: paths, API keys, and config loading."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_env() -> None:
    """Load .env from the project root without overriding existing env vars."""
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in config file: {path}")
    return data


@dataclass(frozen=True)
class Settings:
    """Resolved runtime settings for data collection and catalog builds."""

    project_root: Path = PROJECT_ROOT

    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    raw_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "raw")
    processed_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "processed")
    structures_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "structures")
    features_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "features")
    models_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "models")
    configs_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "configs")

    duckdb_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "processed" / "materials.duckdb"
    )

    mp_api_key: str | None = None
    mp_chunk_size: int = 1000
    mp_max_materials: int | None = None
    mp_download_structures: bool = True
    mp_structure_batch_size: int = 200
    duckdb_read_only: bool = False
    database_config: dict[str, Any] = field(default_factory=dict)

    def raw_source_dir(self, source: str) -> Path:
        return self.raw_dir / source

    def computational_structures_dir(self) -> Path:
        return self.structures_dir / "computational"

    def experimental_structures_dir(self) -> Path:
        return self.structures_dir / "experimental"

    def ensure_directories(self) -> None:
        for path in (
            self.raw_dir,
            self.processed_dir,
            self.processed_dir / "metadata",
            self.structures_dir / "experimental",
            self.structures_dir / "computational",
            self.structures_dir / "generated",
            self.features_dir,
            self.models_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def require_mp_api_key(self) -> str:
        if not self.mp_api_key:
            raise RuntimeError(
                "MP_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        return self.mp_api_key


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and cache settings from environment and YAML configs."""
    _load_env()

    database_config = _load_yaml(PROJECT_ROOT / "configs" / "database.yaml")
    mp_cfg = database_config.get("materials_project", {})
    duckdb_cfg = database_config.get("duckdb", {})

    max_materials = mp_cfg.get("max_materials", None)
    if max_materials in {"none", "null"} or max_materials is False:
        max_materials = None

    return Settings(
        mp_api_key=os.getenv("MP_API_KEY") or None,
        mp_chunk_size=int(mp_cfg.get("chunk_size", 1000)),
        mp_max_materials=int(max_materials) if max_materials is not None else None,
        mp_download_structures=bool(mp_cfg.get("download_structures", True)),
        mp_structure_batch_size=int(mp_cfg.get("structure_batch_size", 200)),
        duckdb_path=PROJECT_ROOT
        / duckdb_cfg.get("path", "data/processed/materials.duckdb"),
        duckdb_read_only=bool(duckdb_cfg.get("read_only", False)),
        database_config=database_config,
    )


def clear_settings_cache() -> None:
    """Clear cached settings (useful after config changes in-process)."""
    get_settings.cache_clear()
