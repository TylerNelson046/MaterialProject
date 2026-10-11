"""MACE structure-model helpers (Phase 4+).

Optional install::

    pip install -e ".[mace]"

Lives under ``ml.mace`` (not top-level ``mace``) so it does not collide with
the ``mace-torch`` package namespace. Contracts: ``docs/mace_plan.md``.
"""

from __future__ import annotations

__all__ = [
    "mace_deps_available",
    "require_mace_deps",
]


def mace_deps_available() -> bool:
    """Return True when torch, ase, and mace-torch import successfully."""
    try:
        import ase  # noqa: F401
        import mace  # noqa: F401  — mace-torch
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def require_mace_deps() -> None:
    """Raise ``ImportError`` with install hint if the ``[mace]`` extra is missing."""
    if mace_deps_available():
        return
    raise ImportError(
        "MACE dependencies are not installed. Install the optional extra:\n"
        '  pip install -e ".[mace]"\n'
        "See docs/mace_training.md for the known-good stack."
    )
