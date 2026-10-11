#!/usr/bin/env python3
"""Phase 4.8: extract pooled MACE descriptors for a structure pack.

Example::

    python scripts/extract_mace_embeddings.py --target band_gap --device cpu \\
        --limit-per-split 200
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
    parser.add_argument("--target", default="band_gap")
    parser.add_argument(
        "--protocol",
        default="mace_desc_meanstd_v1",
        choices=["mace_desc_meanstd_v1"],
    )
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument("--limit-per-split", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    from ml.mace.embeddings import extract_embeddings

    result = extract_embeddings(
        args.target,
        protocol=args.protocol,
        splits=tuple(args.splits) if args.splits else None,
        limit_per_split=args.limit_per_split,
        device=args.device,
        cache_path=args.cache_path,
        overwrite=args.overwrite,
    )
    payload = {
        "target": result.target,
        "protocol": result.protocol,
        "cache_path": str(result.cache_path),
        "n_rows": result.n_rows,
        "n_features": result.n_features,
        "n_failed_loads": result.n_failed_loads,
        "n_failed_infer": result.n_failed_infer,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            f"Embeddings OK | target={result.target} rows={result.n_rows} "
            f"feat={result.n_features} → {result.cache_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
