#!/usr/bin/env python3
"""Build / resume frozen MACE formation-energy cache for hull + scoring.

Examples::

    python scripts/cache_mace_eform.py --device cpu --limit-per-split 64
    python scripts/cache_mace_eform.py --splits train --overwrite
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
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument("--limit-per-split", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    from ml.mace.eform_cache import build_eform_cache

    result = build_eform_cache(
        args.target,
        splits=tuple(args.splits) if args.splits else None,
        limit_per_split=args.limit_per_split,
        device=args.device,
        cache_path=args.out,
        overwrite=args.overwrite,
    )
    payload = {
        "target": result.target,
        "cache_path": str(result.cache_path),
        "n_rows": result.n_rows,
        "n_failed_loads": result.n_failed_loads,
        "n_failed_infer": result.n_failed_infer,
        "protocol": result.protocol,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            f"E_form cache OK | protocol={result.protocol} n={result.n_rows} "
            f"fail_load={result.n_failed_loads} fail_infer={result.n_failed_infer}"
        )
        print(f"Wrote {result.cache_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
