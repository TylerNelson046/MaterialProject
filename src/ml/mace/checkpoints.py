"""Resolve / download pinned MACE foundation checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from config.settings import PROJECT_ROOT
from ml.mace.config import foundation_model_id, load_mace_config

DEFAULT_CHECKPOINT_URLS: dict[str, str] = {
    "medium-mpa-0": (
        "https://github.com/ACEsuit/mace-mp/releases/download/mace_mpa_0/mace-mpa-0-medium.model"
    ),
}


def checkpoint_cache_dir(config: dict[str, Any] | None = None) -> Path:
    """Directory for locally cached foundation ``.model`` files."""
    cfg = config if config is not None else load_mace_config()
    foundation = cfg.get("foundation") or {}
    raw = foundation.get("cache_dir")
    if raw:
        path = Path(raw)
        return path if path.is_absolute() else PROJECT_ROOT / path
    return PROJECT_ROOT / "data" / "models" / "mace" / "checkpoints"


def checkpoint_url(model_id: str, config: dict[str, Any] | None = None) -> str:
    """Return download URL for a foundation model id."""
    cfg = config if config is not None else load_mace_config()
    foundation = cfg.get("foundation") or {}
    configured = foundation.get("checkpoint_url")
    if configured and foundation_model_id(cfg) == model_id:
        return str(configured)
    if model_id in DEFAULT_CHECKPOINT_URLS:
        return DEFAULT_CHECKPOINT_URLS[model_id]
    raise KeyError(
        f"No checkpoint URL for model_id={model_id!r}. "
        "Set foundation.checkpoint_url in configs/mace.yaml or pass a local path."
    )


def checkpoint_filename(model_id: str, url: str) -> str:
    """Pick a stable on-disk filename for the checkpoint."""
    name = Path(urlparse(url).path).name
    if name:
        return name
    return f"{model_id.replace('/', '_')}.model"


def ensure_foundation_checkpoint(
    model_id: str | None = None,
    *,
    config: dict[str, Any] | None = None,
    force_download: bool = False,
) -> Path:
    """Return a local path to the foundation weights, downloading if needed.

    Uses ``requests`` (more reliable on some macOS Python builds than
    ``urllib`` inside ``mace_mp``). Cache dir defaults to
    ``data/models/mace/checkpoints/`` (gitignored).
    """
    cfg = config if config is not None else load_mace_config()
    resolved_id = model_id or foundation_model_id(cfg)

    # Already a filesystem path?
    as_path = Path(resolved_id)
    if as_path.is_file():
        return as_path.resolve()

    url = checkpoint_url(resolved_id, cfg)
    cache_dir = checkpoint_cache_dir(cfg)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / checkpoint_filename(resolved_id, url)

    if dest.is_file() and dest.stat().st_size > 0 and not force_download:
        return dest.resolve()

    foundation = cfg.get("foundation") or {}
    retries = int(foundation.get("checkpoint_download_retries", 3))
    retries = max(1, retries)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            with requests.get(url, stream=True, timeout=120) as response:
                response.raise_for_status()
                with tmp.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            tmp.replace(dest)
            last_error = None
            break
        except Exception as exc:  # noqa: BLE001 — retry then surface
            last_error = exc
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            if attempt < retries:
                continue

    if last_error is not None:
        raise RuntimeError(
            f"Failed to download MACE checkpoint for {resolved_id!r} from {url} "
            f"after {retries} attempt(s). "
            f"Download manually into {cache_dir} and retry."
        ) from last_error

    if not dest.is_file() or dest.stat().st_size == 0:
        raise RuntimeError(f"Checkpoint missing or empty after download: {dest}")
    return dest.resolve()
