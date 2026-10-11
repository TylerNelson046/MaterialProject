#!/usr/bin/env python3
"""Phase 4.7: evaluate MACE phase-diagram E_hull protocol (not a naive regressor).

Examples::

    python scripts/eval_mace_hull.py --device cpu \\
        --limit-reference 128 --limit-per-split 64
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
        choices=["mace_pd_eform_v1"],
        help="Hull protocol (default: configs/mace.yaml → mace_pd_eform_v1).",
    )
    parser.add_argument("--reference-split", default="train")
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--limit-reference", type=int, default=None)
    parser.add_argument("--limit-per-split", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--no-eform-cache",
        action="store_true",
        help="Always re-infer reference E_form (ignore cache).",
    )
    parser.add_argument(
        "--eform-cache",
        type=Path,
        default=None,
        help="Path to eform_*.parquet (default: pack features cache).",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    from ml.mace.eval_hull import eval_hull

    result = eval_hull(
        protocol=args.protocol,
        reference_split=args.reference_split,
        eval_splits=tuple(args.splits),
        limit_reference=args.limit_reference,
        limit_per_split=args.limit_per_split,
        device=args.device,
        use_eform_cache=not args.no_eform_cache,
        eform_cache_path=args.eform_cache,
    )
    payload = {
        "run_id": result.run_id,
        "run_dir": str(result.run_dir),
        "protocol": result.protocol,
        "n_reference": result.n_reference,
        "counts": result.counts,
        "registry_metrics": (result.metrics or {}).get("registry_metrics"),
        "n_failed_loads": result.n_failed_loads,
        "n_failed_infer": result.n_failed_infer,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            f"Hull eval OK | protocol={result.protocol} n_ref={result.n_reference} "
            f"run={result.run_id} counts={result.counts}"
        )
        print(f"Wrote {result.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
