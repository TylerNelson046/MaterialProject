#!/usr/bin/env python3
"""Phase 4.6: predict with a registered MACE run.

Examples::

    python scripts/predict_mace.py --structure data/structures/computational/JVASP-10.json
    python scripts/predict_mace.py --material-id MAT-243797 --family mace_pretrained
    python scripts/predict_mace.py --material-id MAT-1 --material-id MAT-2 --json
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

from ml.mace.runtime import configure_mace_runtime  # noqa: E402

configure_mace_runtime()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="formation_energy_per_atom")
    parser.add_argument(
        "--family",
        default="mace_pretrained",
        choices=["mace_pretrained", "mace_finetuned"],
        help="Registry model_family to load (frozen-first default: mace_pretrained).",
    )
    parser.add_argument(
        "--structure",
        action="append",
        dest="structures",
        default=None,
        help="Structure file path (repeatable).",
    )
    parser.add_argument(
        "--material-id",
        action="append",
        dest="material_ids",
        default=None,
        help="Pack material_id (repeatable; resolves structure_path).",
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    from ml.mace.predict import load_mace_run, predict_structures

    loaded = load_mace_run(
        args.target,
        model_family=args.family,
        run_dir=args.run_dir,
    )
    frame = predict_structures(
        structure_paths=args.structures,
        material_ids=args.material_ids,
        target=args.target,
        model_family=args.family,
        run_dir=args.run_dir,
        device=args.device,
    )
    if args.json:
        print(json.dumps(frame.to_dict(orient="records"), indent=2))
    else:
        test_mae = None
        reg = loaded.metrics.get("registry_metrics") or {}
        if isinstance(reg, dict):
            block = reg.get("test") or {}
            if isinstance(block, dict):
                test_mae = block.get("mae")
        if test_mae is None:
            # Fallback: nested splits.formation_energy.mae
            splits = loaded.metrics.get("splits") or {}
            if isinstance(splits, dict):
                test_block = splits.get("test") or {}
                if isinstance(test_block, dict):
                    fe = test_block.get("formation_energy") or {}
                    if isinstance(fe, dict):
                        test_mae = fe.get("mae")
        print(
            f"Using {loaded.model_family} run {loaded.run_id} "
            f"(test E_form MAE={test_mae})"
        )
        cols = [
            c
            for c in (
                "material_id",
                "formula",
                "formation_energy_per_atom",
                "energy_per_atom_eV",
                "energy_total_eV",
                "ok",
                "reason",
            )
            if c in frame.columns
        ]
        print(frame.loc[:, cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
