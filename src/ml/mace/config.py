"""Load ``configs/mace.yaml``."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from config.settings import PROJECT_ROOT


@lru_cache(maxsize=4)
def load_mace_config(path: str | None = None) -> dict[str, Any]:
    """Load MACE config mapping (cached by path string)."""
    cfg_path = Path(path) if path else PROJECT_ROOT / "configs" / "mace.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"MACE config not found: {cfg_path}")
    with cfg_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {cfg_path}")
    return data


def foundation_model_id(config: dict[str, Any] | None = None) -> str:
    """Return the pinned foundation loader id (never rely on library default)."""
    cfg = config if config is not None else load_mace_config()
    foundation = cfg.get("foundation") or {}
    model_id = foundation.get("model_id")
    if not model_id:
        raise ValueError("configs/mace.yaml foundation.model_id is required")
    return str(model_id)


def device_preference(config: dict[str, Any] | None = None) -> tuple[str, ...]:
    """Ordered device preference from config (default cuda → mps → cpu)."""
    cfg = config if config is not None else load_mace_config()
    device_cfg = cfg.get("device") or {}
    pref = device_cfg.get("preference") or ["cuda", "mps", "cpu"]
    return tuple(str(item) for item in pref)
