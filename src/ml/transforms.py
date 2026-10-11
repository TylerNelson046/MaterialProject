"""Label transforms for skewed regression targets.

Training may fit in a transformed space (e.g. ``log1p``) while metrics and
predictions are reported in original physical units via the inverse map.
"""

from __future__ import annotations

import numpy as np

# Canonical names accepted in configs/models.yaml → targets.*.transform
SUPPORTED_TRANSFORMS = frozenset({None, "", "identity", "none", "null", "log1p"})


def normalize_transform(name: str | None) -> str | None:
    """Normalize a transform name; ``None`` means identity (no transform)."""
    if name is None:
        return None
    text = str(name).strip().lower()
    if text in {"", "identity", "none", "null"}:
        return None
    if text != "log1p":
        raise ValueError(f"Unsupported label transform {name!r}")
    return text


def transform_y(y: np.ndarray, transform: str | None) -> np.ndarray:
    """Map labels into model space."""
    kind = normalize_transform(transform)
    arr = np.asarray(y, dtype=np.float64)
    if kind is None:
        return arr
    if kind == "log1p":
        # Clip tiny negatives from numerical noise before log1p.
        return np.log1p(np.clip(arr, 0.0, None))
    raise ValueError(f"Unsupported label transform {transform!r}")


def inverse_transform_y(y: np.ndarray, transform: str | None) -> np.ndarray:
    """Map model outputs back to original label units."""
    kind = normalize_transform(transform)
    arr = np.asarray(y, dtype=np.float64)
    if kind is None:
        return arr
    if kind == "log1p":
        return np.expm1(arr)
    raise ValueError(f"Unsupported label transform {transform!r}")
