"""Resolve torch device for MACE (cuda / mps / cpu)."""

from __future__ import annotations

from ml.mace.config import device_preference, load_mace_config

ALLOWED_DEVICES = frozenset({"cuda", "mps", "cpu"})


def _torch():
    from ml.mace import require_mace_deps
    from ml.mace.runtime import configure_mace_runtime

    # Thread caps must be set before the first torch import in this process.
    configure_mace_runtime()
    require_mace_deps()
    import torch

    torch.set_num_threads(1)
    set_num_interop = getattr(torch, "set_num_interop_threads", None)
    if callable(set_num_interop):
        try:
            set_num_interop(1)
        except RuntimeError:
            # Already initialized in this process — ignore.
            pass
    return torch


def cuda_available() -> bool:
    """True when CUDA tensors can be allocated."""
    torch = _torch()
    return bool(torch.cuda.is_available())


def mps_available() -> bool:
    """True when Apple MPS backend is available."""
    torch = _torch()
    backend = getattr(torch.backends, "mps", None)
    if backend is None:
        return False
    return bool(backend.is_available())


def describe_devices() -> dict[str, bool | str]:
    """Return a small diagnostic map of device availability."""
    torch = _torch()
    info: dict[str, bool | str] = {
        "cuda": cuda_available(),
        "mps": mps_available(),
        "cpu": True,
        "torch_version": str(torch.__version__),
    }
    if info["cuda"]:
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
    return info


def resolve_device(
    explicit: str | None = None,
    *,
    preference: tuple[str, ...] | list[str] | None = None,
) -> str:
    """Pick a torch device string.

    Parameters
    ----------
    explicit:
        If set (``cuda`` / ``mps`` / ``cpu``), require that device or raise
        ``RuntimeError`` with a clear message.
    preference:
        Ordered fallback list. Defaults to ``configs/mace.yaml`` → ``device.preference``.
    """
    if explicit is not None:
        name = explicit.strip().lower()
        # Phase 4.9: ``auto`` / empty → fall through to config preference order.
        if name in {"", "auto", "default", "pref", "preference"}:
            explicit = None
        elif name not in ALLOWED_DEVICES:
            raise ValueError(
                f"Unknown device {explicit!r}; expected one of "
                f"{sorted(ALLOWED_DEVICES)} or 'auto'"
            )
    if explicit is not None:
        name = explicit.strip().lower()
        if name == "cuda" and not cuda_available():
            raise RuntimeError(
                "CUDA requested but torch.cuda.is_available() is False. "
                "Install a CUDA build of PyTorch or pass --device cpu."
            )
        if name == "mps" and not mps_available():
            raise RuntimeError(
                "MPS requested but torch.backends.mps.is_available() is False. "
                "Use --device cpu or cuda."
            )
        return name

    order = tuple(preference) if preference is not None else device_preference(load_mace_config())
    for name in order:
        key = str(name).strip().lower()
        if key not in ALLOWED_DEVICES:
            raise ValueError(
                f"Unknown device preference {name!r}; expected one of {sorted(ALLOWED_DEVICES)}"
            )
        if key == "cuda" and cuda_available():
            return "cuda"
        if key == "mps" and mps_available():
            return "mps"
        if key == "cpu":
            return "cpu"

    return "cpu"
