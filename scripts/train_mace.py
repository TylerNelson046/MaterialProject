#!/usr/bin/env python3
"""Phase 4.5: energy-only fine-tune of foundation MACE on the structure pack.

Example::

    # CPU smoke
    python scripts/train_mace.py --device cpu --max-train-rows 64 --epochs 3 \\
        --batch-size 1 --eval-limit-per-split 32

    # Larger fine-tune (GPU recommended)
    python scripts/train_mace.py --device cuda --max-train-rows 5000 --epochs 30
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
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model", default=None, help="Foundation model id override.")
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-val-rows", type=int, default=None)
    parser.add_argument("--max-test-rows", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="max_num_epochs")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--eval-limit-per-split",
        type=int,
        default=None,
        help="Cap post-train eval rows per split (default: full val/test).",
    )
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    from ml.mace.train import train_mace

    result = train_mace(
        args.target,
        device=args.device,
        model_id=args.model,
        max_train_rows=args.max_train_rows,
        max_val_rows=args.max_val_rows,
        max_test_rows=args.max_test_rows,
        max_num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        skip_eval=args.skip_eval,
        eval_limit_per_split=args.eval_limit_per_split,
        register=not args.no_register,
    )
    test = ((result.metrics.get("splits") or {}).get("test") or {}).get(
        "formation_energy"
    ) or {}
    payload = {
        "run_id": result.run_id,
        "run_dir": str(result.run_dir),
        "checkpoint": str(result.checkpoint),
        "train_log": str(result.train_log),
        "counts": result.counts,
        "test_formation_mae": test.get("mae"),
        "test_formation_rmse": test.get("rmse"),
        "test_formation_r2": test.get("r2"),
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            f"MACE fine-tune OK | run={result.run_id} "
            f"ckpt={result.checkpoint} "
            f"test_Eform_MAE={payload['test_formation_mae']}"
        )
        print(f"Log: {result.train_log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
