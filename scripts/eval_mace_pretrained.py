#!/usr/bin/env python3
"""Phase 4.4: zero-shot pretrained MACE eval on the structure pack.

Example::

    python scripts/eval_mace_pretrained.py --device cpu --limit-per-split 200
    python scripts/eval_mace_pretrained.py --device cpu --splits val test
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
        "--splits",
        nargs="+",
        default=["val", "test"],
        choices=["train", "val", "test"],
    )
    parser.add_argument(
        "--limit-per-split",
        type=int,
        default=None,
        help="Cap rows per split (smoke / quick metrics).",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model", default=None, help="Override foundation model id.")
    parser.add_argument(
        "--no-register",
        action="store_true",
        help="Skip writing registry.json / LEADERBOARD.md.",
    )
    parser.add_argument(
        "--no-predictions",
        action="store_true",
        help="Skip predictions_*.parquet.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    from ml.mace.eval_pretrained import eval_pretrained

    result = eval_pretrained(
        args.target,
        splits=tuple(args.splits),
        limit_per_split=args.limit_per_split,
        device=args.device,
        model_id=args.model,
        write_predictions=not args.no_predictions,
        register=not args.no_register,
    )
    test = ((result.metrics.get("splits") or {}).get("test") or {}).get(
        "formation_energy"
    ) or {}
    test_corr = ((result.metrics.get("splits") or {}).get("test") or {}).get(
        "energy_per_atom_vs_label"
    ) or {}
    payload = {
        "run_id": result.run_id,
        "run_dir": str(result.run_dir),
        "model_id": result.model_id,
        "device": result.device,
        "counts": result.counts,
        "n_failed_loads": result.n_failed_loads,
        "n_failed_infer": result.n_failed_infer,
        "test_formation_mae": test.get("mae"),
        "test_formation_rmse": test.get("rmse"),
        "test_formation_r2": test.get("r2"),
        "test_energy_per_atom_spearman": test_corr.get("spearman_r"),
        "test_energy_per_atom_pearson": test_corr.get("pearson_r"),
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            f"MACE pretrained eval OK | run={result.run_id} "
            f"counts={result.counts} "
            f"test_Eform_MAE={payload['test_formation_mae']} "
            f"test_E/atom_Spearman={payload['test_energy_per_atom_spearman']}"
        )
        print(f"Wrote {result.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
