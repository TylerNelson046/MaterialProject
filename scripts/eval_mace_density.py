#!/usr/bin/env python3
"""Phase 4.7: evaluate structure density protocols vs Ready density labels.

Examples::

    python scripts/eval_mace_density.py --limit-per-split 200 --cod-limit 100
    python scripts/eval_mace_density.py --protocol geometry_relax_v1 --device cpu \\
        --limit-per-split 20 --no-cod-holdout
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
    parser.add_argument(
        "--protocol",
        default=None,
        choices=["geometry_v1", "geometry_relax_v1"],
        help="Density protocol (default: configs/mace.yaml → geometry_v1).",
    )
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--limit-per-split", type=int, default=None)
    parser.add_argument(
        "--cod-limit",
        type=int,
        default=None,
        help="Max COD holdout rows (holdout_cod_geometry).",
    )
    parser.add_argument(
        "--no-cod-holdout",
        action="store_true",
        help="Skip experimental COD geometry holdout eval.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--relax-fmax", type=float, default=None)
    parser.add_argument("--relax-steps", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    from ml.mace.eval_density import eval_density

    result = eval_density(
        protocol=args.protocol,
        splits=tuple(args.splits),
        limit_per_split=args.limit_per_split,
        also_cod_holdout=not args.no_cod_holdout,
        cod_limit=args.cod_limit,
        device=args.device,
        relax_fmax=args.relax_fmax,
        relax_steps=args.relax_steps,
    )
    payload = {
        "run_id": result.run_id,
        "run_dir": str(result.run_dir),
        "protocol": result.protocol,
        "counts": result.counts,
        "registry_metrics": (result.metrics or {}).get("registry_metrics"),
        "n_failed_loads": result.n_failed_loads,
        "n_failed_infer": result.n_failed_infer,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            f"Density eval OK | protocol={result.protocol} run={result.run_id} "
            f"counts={result.counts}"
        )
        print(f"Wrote {result.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
