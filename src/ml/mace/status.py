"""Read-only MACE stack status for operators (Phase 4.9)."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from config.settings import PROJECT_ROOT
from ml.mace.checkpoints import checkpoint_cache_dir
from ml.mace.config import foundation_model_id, load_mace_config
from ml.mace.pack import mace_pack_dir
from ml.registry import load_registry


def _safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def collect_mace_status(
    *,
    config: dict[str, Any] | None = None,
    project_root: Path | None = None,
    include_devices: bool = True,
) -> dict[str, Any]:
    """Return a JSON-serializable status document for the local MACE stack."""
    cfg = config if config is not None else load_mace_config()
    root = project_root or PROJECT_ROOT
    pack_root = mace_pack_dir(cfg)
    models_dir = root / str((cfg.get("output") or {}).get("models_dir") or "data/models")
    cache_dir = checkpoint_cache_dir(cfg)

    index_path = pack_root / "targets_index.json"
    targets_index = None
    if index_path.is_file():
        targets_index = json.loads(index_path.read_text(encoding="utf-8"))

    pack_targets: list[dict[str, Any]] = []
    targets_dir = pack_root / "targets"
    if targets_dir.is_dir():
        for parquet in sorted(targets_dir.glob("*.parquet")):
            audit = pack_root / "structure_audit" / parquet.stem / "summary.json"
            pack_targets.append(
                {
                    "target": parquet.stem,
                    "parquet": _safe_rel(parquet, root),
                    "bytes": parquet.stat().st_size,
                    "audit_summary": _safe_rel(audit, root) if audit.is_file() else None,
                }
            )

    features_root = pack_root / "features"
    embedding_caches: list[dict[str, Any]] = []
    if features_root.is_dir():
        for meta in sorted(features_root.glob("*/*/*.json")):
            try:
                payload = json.loads(meta.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                payload = {}
            embedding_caches.append(
                {
                    "path": _safe_rel(meta.with_suffix(".parquet"), root),
                    "meta": _safe_rel(meta, root),
                    "target": payload.get("target"),
                    "n_rows": payload.get("n_rows"),
                    "protocol": payload.get("protocol"),
                }
            )

    registry = load_registry(models_dir)
    latest_mace: list[dict[str, Any]] = []
    for target, runs in (registry.get("runs") or {}).items():
        for family in (
            "mace_finetuned",
            "mace_pretrained",
            "mace_embedding_head",
        ):
            for record in reversed(runs or []):
                if str(record.get("model_family") or "") == family:
                    metrics = (record.get("metrics") or {}).get("test") or {}
                    latest_mace.append(
                        {
                            "target": target,
                            "model_family": family,
                            "run_id": record.get("run_id"),
                            "test_mae": metrics.get("mae"),
                            "promoted_vs_lightgbm": record.get("promoted_vs_lightgbm"),
                        }
                    )
                    break

    checkpoints = []
    if cache_dir.is_dir():
        for path in sorted(cache_dir.glob("*.model")):
            checkpoints.append(
                {
                    "path": _safe_rel(path, root),
                    "bytes": path.stat().st_size,
                }
            )

    devices: dict[str, Any] | None = None
    if include_devices:
        try:
            from ml.mace.device import describe_devices

            devices = describe_devices()
        except Exception as exc:  # noqa: BLE001
            devices = {"error": f"{type(exc).__name__}: {exc}"}

    return {
        "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "python": sys.version.split()[0],
        "project_root": str(root),
        "foundation_model_id": foundation_model_id(cfg),
        "devices": devices,
        "checkpoint_cache": {
            "dir": _safe_rel(cache_dir, root),
            "files": checkpoints,
        },
        "mace_pack": {
            "dir": _safe_rel(pack_root, root),
            "targets_index": targets_index,
            "targets": pack_targets,
        },
        "embedding_caches": embedding_caches,
        "latest_mace_runs": latest_mace,
        "leaderboard": _safe_rel(models_dir / "LEADERBOARD.md", root)
        if (models_dir / "LEADERBOARD.md").is_file()
        else None,
        "runs_presets": sorted((cfg.get("runs") or {}).keys()),
    }
