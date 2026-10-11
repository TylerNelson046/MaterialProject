#!/usr/bin/env python3
"""Phase 4.1 smoke: load pinned MACE foundation model and print one energy.

Usage::

    pip install -e ".[mace]"
    python scripts/smoke_mace.py --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Cap threads BEFORE importing torch/mace — required on Python 3.14 / macOS
# or the process can segfault (“Python quit unexpectedly”).
from ml.mace.runtime import configure_mace_runtime, warn_if_unstable_python  # noqa: E402

configure_mace_runtime()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device (default: cpu). Use cuda/mps when available.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override foundation model id (default: configs/mace.yaml).",
    )
    parser.add_argument(
        "--dtype",
        default=None,
        choices=["float32", "float64"],
        help="Override foundation.default_dtype from configs/mace.yaml.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print SmokeResult as JSON instead of a short summary.",
    )
    args = parser.parse_args(argv)

    from ml.mace import mace_deps_available, require_mace_deps
    from ml.mace.config import load_mace_config
    from ml.mace.device import describe_devices
    from ml.mace.smoke import run_smoke

    warn_if_unstable_python()
    if not mace_deps_available():
        require_mace_deps()

    config = load_mace_config()
    if args.dtype:
        foundation = dict(config.get("foundation") or {})
        foundation["default_dtype"] = args.dtype
        config = {**config, "foundation": foundation}

    result = run_smoke(device=args.device, model_id=args.model, config=config)
    # Probe devices after the forward so MPS/CUDA init cannot destabilize smoke.
    devices = describe_devices()
    payload = {
        "model_id": result.model_id,
        "device": result.device,
        "formula": result.formula,
        "n_atoms": result.n_atoms,
        "energy_eV": result.energy_eV,
        "energy_per_atom_eV": result.energy_per_atom_eV,
        "devices": devices,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            f"MACE smoke OK | model={result.model_id} device={result.device} "
            f"formula={result.formula} n_atoms={result.n_atoms} "
            f"E={result.energy_eV:.6f} eV "
            f"E/atom={result.energy_per_atom_eV:.6f} eV/atom"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
