"""Load the pinned MACE foundation ASE calculator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ml.mace import require_mace_deps
from ml.mace.checkpoints import ensure_foundation_checkpoint
from ml.mace.config import foundation_model_id, load_mace_config
from ml.mace.device import resolve_device
from ml.mace.runtime import coerce_stable_dtype, configure_mace_runtime, warn_if_unstable_python


def load_foundation_calculator(
    *,
    model_id: str | None = None,
    device: str | None = None,
    config: dict[str, Any] | None = None,
    checkpoint: Path | str | None = None,
) -> tuple[Any, str, Path, str]:
    """Return ``(calculator, model_id, checkpoint_path, device)``.

    If ``checkpoint`` is a local ``.model`` path (fine-tuned or foundation),
    that file is loaded directly. Otherwise the pinned foundation id is
    downloaded/cached via ``ensure_foundation_checkpoint``.
    """
    configure_mace_runtime()
    warn_if_unstable_python()
    require_mace_deps()
    from mace.calculators import mace_mp

    cfg = config if config is not None else load_mace_config()
    resolved_model = model_id or foundation_model_id(cfg)
    resolved_device = resolve_device(device)
    foundation = cfg.get("foundation") or {}
    default_dtype = coerce_stable_dtype(str(foundation.get("default_dtype") or "float32"))
    dispersion = bool(foundation.get("dispersion", False))
    if checkpoint is not None:
        ckpt_path = Path(checkpoint)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"MACE checkpoint not found: {ckpt_path}")
    else:
        ckpt_path = ensure_foundation_checkpoint(resolved_model, config=cfg)
    calc = mace_mp(
        model=str(ckpt_path),
        device=resolved_device,
        default_dtype=default_dtype,
        dispersion=dispersion,
    )
    return calc, resolved_model, ckpt_path, resolved_device
