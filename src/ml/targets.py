"""Target registry for the training export package.

Reads ``configs/models.yaml`` and exposes Ready / Niche heads used by
``ml.data``, ``ml.train``, and the CLI entry points.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from config.settings import PROJECT_ROOT, get_settings
from ml.transforms import normalize_transform


@dataclass(frozen=True)
class TargetSpec:
    """One supervised property head (parquet table + training options)."""

    name: str
    tier: str
    parquet: str
    label_col: str = "label"
    # Optional label map in model space (e.g. "log1p"); metrics stay in raw units.
    transform: str | None = None

    @property
    def is_ready(self) -> bool:
        return self.tier.lower() == "ready"

    @property
    def is_niche(self) -> bool:
        return self.tier.lower() == "niche"


def load_models_config(path: Path | None = None) -> dict[str, Any]:
    """Load ``configs/models.yaml`` (or an explicit override path)."""
    if path is None:
        cfg = get_settings().models_config
        if cfg:
            return cfg
        path = PROJECT_ROOT / "configs" / "models.yaml"
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in models config: {path}")
    return data


def list_targets(
    config: dict[str, Any] | None = None,
    *,
    tier: str | None = None,
) -> list[TargetSpec]:
    """Return target specs, optionally filtered by tier (``Ready`` / ``Niche``)."""
    cfg = config if config is not None else load_models_config()
    raw = cfg.get("targets") or {}
    if not isinstance(raw, dict):
        raise ValueError("models.yaml 'targets' must be a mapping")

    specs: list[TargetSpec] = []
    for name, meta in raw.items():
        if not isinstance(meta, dict):
            raise ValueError(f"Invalid target entry for {name!r}")
        spec = TargetSpec(
            name=str(name),
            tier=str(meta.get("tier", "")),
            parquet=str(meta.get("parquet", f"{name}.parquet")),
            label_col=str(meta.get("label_col", "label")),
            transform=normalize_transform(meta.get("transform")),
        )
        if tier is not None and spec.tier.lower() != tier.lower():
            continue
        specs.append(spec)
    return specs


def get_target(name: str, config: dict[str, Any] | None = None) -> TargetSpec:
    """Look up a single target by name."""
    key = name.strip()
    for spec in list_targets(config):
        if spec.name == key:
            return spec
    known = ", ".join(known_targets(config))
    raise KeyError(f"Unknown target {name!r}. Known: {known}")


def known_targets(config: dict[str, Any] | None = None) -> tuple[str, ...]:
    """All configured target names, in YAML order."""
    return tuple(spec.name for spec in list_targets(config))


def ready_targets(config: dict[str, Any] | None = None) -> tuple[str, ...]:
    """Ready-tier target names (primary Phase 3 scope)."""
    return tuple(spec.name for spec in list_targets(config, tier="Ready"))


def niche_targets(config: dict[str, Any] | None = None) -> tuple[str, ...]:
    """Niche-tier target names (smaller / method-specific sets)."""
    return tuple(spec.name for spec in list_targets(config, tier="Niche"))
