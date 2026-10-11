"""Process-level torch / MACE stability helpers.

Python 3.14 + current torch wheels can **segfault** during MACE forwards
(especially ``float64``), which macOS reports as “Python quit unexpectedly”.
Call ``configure_mace_runtime()`` before importing torch / running smokes.
"""

from __future__ import annotations

import os
import sys
import warnings


def configure_mace_runtime() -> None:
    """Set conservative thread env vars before heavy torch work."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("TORCH_NUM_THREADS", "1")


def warn_if_unstable_python() -> None:
    """Emit a one-line warning on Python versions known to crash with MACE."""
    if sys.version_info >= (3, 14):
        warnings.warn(
            "Python 3.14 + torch/MACE can segfault on macOS (Crash Reporter: "
            '"Python quit unexpectedly"). Prefer Python 3.12 or 3.13 for the '
            "[mace] extra. Smoke/eval defaults to float32; do not use float64 "
            "on 3.14.",
            UserWarning,
            stacklevel=2,
        )


def coerce_stable_dtype(dtype: str) -> str:
    """Return a dtype for MACE; prefer float32 on Python >= 3.14.

    On 3.14, ``float64`` requests are coerced to ``float32`` with a warning —
    float64 has aborted the process on this stack even when threads are capped.
    """
    name = str(dtype).strip().lower()
    if name not in {"float32", "float64"}:
        raise ValueError(f"Unsupported dtype {dtype!r}; expected float32 or float64")
    if name == "float64" and sys.version_info >= (3, 14):
        warnings.warn(
            "Coercing MACE dtype float64 → float32 on Python 3.14 to avoid "
            "native crashes. Recreate the venv with Python 3.12/3.13 for float64.",
            UserWarning,
            stacklevel=2,
        )
        return "float32"
    return name
